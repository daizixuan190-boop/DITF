import argparse
import gc
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import PILToTensor
from tqdm import tqdm

from src.flux.feat_flux import Featurizer4Eval

warnings.filterwarnings("ignore")


def build_post_feature(ft_raw: torch.Tensor, ada: torch.Tensor, pre_norm: nn.LayerNorm, use_cd: bool) -> torch.Tensor:
    ft = ft_raw.clone()
    if use_cd:
        ft[:, 154, :, :] = 0.0
        ft[:, 1446, :, :] = 0.0
    bsz, channels, height, width = ft.shape
    ft = rearrange(ft, "b c h w -> b (h w) c")
    ft = pre_norm(ft)
    ft = rearrange(ft, "b (h w) c -> b c h w", h=height, w=width)
    shift = ada[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    scale = ada[0][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return (1 + scale) * ft + shift


def topk_candidates(cos_map: np.ndarray, topk: int) -> list[tuple[int, int, float]]:
    flat = cos_map.reshape(-1)
    k = min(int(topk), int(flat.size))
    if k <= 0:
        return []
    top_idx = np.argpartition(-flat, k - 1)[:k]
    top_idx = top_idx[np.argsort(-flat[top_idx])]
    height, width = cos_map.shape
    return [(int(idx % width), int(idx // width), float(flat[idx])) for idx in top_idx]


def reverse_cycle_rerank(
    src_ft: torch.Tensor,
    trg_ft: torch.Tensor,
    src_point: list[int],
    src_threshold: float,
    candidate_list: list[tuple[int, int, float]],
    reverse_weight: float,
) -> tuple[int, int]:
    if not candidate_list:
        return src_point[0], src_point[1]

    src_height, src_width = src_ft.shape[-2:]
    channels = src_ft.shape[1]
    src_matrix = src_ft.view(channels, -1).transpose(0, 1)
    src_matrix = F.normalize(src_matrix, dim=1)

    best_score = None
    best_xy = None
    src_x, src_y = int(src_point[0]), int(src_point[1])

    for cand_x, cand_y, forward_score in candidate_list:
        trg_vec = trg_ft[0, :, cand_y, cand_x].view(1, channels)
        trg_vec = F.normalize(trg_vec, dim=1).transpose(0, 1)
        reverse_scores = torch.mm(src_matrix, trg_vec).view(src_height, src_width)
        rev_flat = int(reverse_scores.argmax())
        rev_y, rev_x = np.unravel_index(rev_flat, (src_height, src_width))
        reverse_dist = ((rev_x - src_x) ** 2 + (rev_y - src_y) ** 2) ** 0.5
        reverse_dist = float(reverse_dist / max(src_threshold, 1e-6))
        combined = float(forward_score - reverse_weight * reverse_dist)
        if best_score is None or combined > best_score:
            best_score = combined
            best_xy = (cand_x, cand_y)

    return best_xy if best_xy is not None else (candidate_list[0][0], candidate_list[0][1])


def maybe_extract_features(args, all_cats, cat2img):
    if args.reuse_saved_features:
        return

    if args.dit_model != "flux":
        raise ValueError("Only flux is supported.")

    dit_model = Featurizer4Eval(cat_list=all_cats[:], ensemble_size=args.ensemble_size)
    with open("spair_detailed_captions.json", "r", encoding="utf-8") as f:
        captions = json.load(f)

    print("saving all test images' features...")
    os.makedirs(args.save_path, exist_ok=True)
    for cat in tqdm(all_cats):
        output_dict = {}
        ada_dict = {}
        for image_name in cat2img[cat]:
            img = Image.open(os.path.join(args.dataset_path, "JPEGImages", cat, image_name))
            image_arr = np.array(img)
            in_h, in_w = image_arr.shape[:2]
            scale = args.img_size[0] / max(in_h, in_w)
            out_h = int(round(in_h * scale / 16)) * 16
            out_w = int(round(in_w * scale / 16)) * 16
            img = img.resize((out_w, out_h))
            img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
            caption = captions[cat + image_name]
            feat, ada = dit_model.forward(
                args,
                img_tensor,
                caption=caption,
                category=cat,
                timestep=args.t,
                block_idx=args.k,
                ensemble_size=args.ensemble_size,
            )
            output_dict[image_name] = feat.cpu()
            ada_dict[image_name] = ada.cpu()
        torch.save(output_dict, os.path.join(args.save_path, f"{cat}.pth"))
        torch.save(ada_dict, os.path.join(args.save_path, f"{cat}_ada.pth"))

    del dit_model
    gc.collect()
    torch.cuda.empty_cache()


def collect_spair_lists(dataset_path: str) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    test_path = os.path.join(dataset_path, "PairAnnotation", "test")
    json_list = os.listdir(test_path)
    all_cats = os.listdir(os.path.join(dataset_path, "JPEGImages"))

    cat2json = {}
    cat2img = {}
    for cat in all_cats:
        cat_pairs = [name for name in json_list if cat in name]
        cat2json[cat] = cat_pairs
        cat_images: list[str] = []
        for json_name in cat_pairs:
            with open(os.path.join(test_path, json_name), "r", encoding="utf-8") as f:
                data = json.load(f)
            if data["src_imname"] not in cat_images:
                cat_images.append(data["src_imname"])
            if data["trg_imname"] not in cat_images:
                cat_images.append(data["trg_imname"])
        cat2img[cat] = cat_images
    return all_cats, cat2json, cat2img


def main(args):
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            print(f"{arg}: {value}")

    torch.cuda.set_device(0)

    all_cats, cat2json, cat2img = collect_spair_lists(args.dataset_path)
    maybe_extract_features(args, all_cats, cat2img)

    print("saving all test images' features...")
    os.makedirs(args.save_path, exist_ok=True)
    if args.reuse_saved_features:
        print("Reusing existing saved features; skip feature extraction.")

    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)

    total_pck = []
    all_correct = 0
    all_total = 0
    mean_image_sum = 0.0
    mean_point_sum = 0.0
    result = {"image": {}, "point": {}}
    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")

    print(f"Category numbers: {len(all_cats)}")
    for cat in all_cats:
        cat_list = cat2json[cat]
        if args.max_pairs_per_cat > 0:
            cat_list = cat_list[: args.max_pairs_per_cat]

        output_dict = torch.load(os.path.join(args.save_path, f"{cat}.pth"))
        ada_dict = torch.load(os.path.join(args.save_path, f"{cat}_ada.pth"))

        cat_pck = []
        cat_correct = 0
        cat_total = 0

        for json_name in tqdm(cat_list):
            with open(os.path.join(test_path, json_name), "r", encoding="utf-8") as f:
                data = json.load(f)

            src_img_size = data["src_imsize"][:2][::-1]
            trg_img_size = data["trg_imsize"][:2][::-1]

            src_ft_post = build_post_feature(
                output_dict[data["src_imname"]].cuda(),
                ada_dict[data["src_imname"]].cuda(),
                pre_norm,
                args.cd,
            )
            trg_ft_post = build_post_feature(
                output_dict[data["trg_imname"]].cuda(),
                ada_dict[data["trg_imname"]].cuda(),
                pre_norm,
                args.cd,
            )

            src_ft = nn.Upsample(size=src_img_size, mode="bilinear")(src_ft_post).to(torch.float16)
            trg_ft = nn.Upsample(size=trg_img_size, mode="bilinear")(trg_ft_post).to(torch.float16)

            src_h, src_w = src_ft.shape[-2:]
            trg_h, trg_w = trg_ft.shape[-2:]
            channels = src_ft.shape[1]

            trg_matrix = trg_ft.view(channels, -1).transpose(0, 1)
            trg_matrix = F.normalize(trg_matrix, dim=1)

            src_bndbox = data["src_bndbox"]
            trg_bndbox = data["trg_bndbox"]
            src_threshold = max(src_bndbox[3] - src_bndbox[1], src_bndbox[2] - src_bndbox[0])
            trg_threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            total = 0
            correct = 0
            for idx in range(len(data["src_kps"])):
                total += 1
                cat_total += 1
                all_total += 1

                src_point = data["src_kps"][idx]
                trg_point = data["trg_kps"][idx]
                src_x, src_y = int(src_point[0]), int(src_point[1])

                src_vec = src_ft[0, :, src_y, src_x].view(1, channels)
                src_vec = F.normalize(src_vec, dim=1).transpose(0, 1)
                cos_map = torch.mm(trg_matrix, src_vec).view(trg_h, trg_w).cpu().numpy()

                if args.identity_cycle_rerank:
                    candidates = topk_candidates(cos_map, args.icr_topk)
                    pred_x, pred_y = reverse_cycle_rerank(
                        src_ft,
                        trg_ft,
                        src_point,
                        src_threshold,
                        candidates,
                        args.icr_reverse_weight,
                    )
                else:
                    max_y, max_x = np.unravel_index(int(cos_map.argmax()), cos_map.shape)
                    pred_x, pred_y = int(max_x), int(max_y)

                dist = ((pred_x - trg_point[0]) ** 2 + (pred_y - trg_point[1]) ** 2) ** 0.5
                if (dist / trg_threshold) <= 0.1:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1

            cat_pck.append(correct / total)
            torch.cuda.empty_cache()

        total_pck.extend(cat_pck)
        mean_image_sum += np.mean(cat_pck) * 100
        mean_point_sum += cat_correct / cat_total * 100

        print(f"{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}")
        print(f"{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}")
        result["image"][cat] = round(np.mean(cat_pck) * 100, 2)
        result["point"][cat] = round(cat_correct / cat_total * 100, 2)

    print(f"All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}")
    print(f"All per point PCK@0.1: {all_correct / all_total * 100:.2f}")
    print(f"Mean per image PCK@0.1: {mean_image_sum / len(all_cats):.2f}")
    print(f"Mean per point PCK@0.1: {mean_point_sum / len(all_cats):.2f}")

    result["image"]["All"] = round(np.mean(total_pck) * 100, 2)
    result["point"]["All"] = round(all_correct / all_total * 100, 2)
    result["image"]["Mean"] = round(mean_image_sum / len(all_cats), 2)
    result["point"]["Mean"] = round(mean_point_sum / len(all_cats), 2)

    save_dir = os.path.join("layers_cat", args.dit_model)
    os.makedirs(save_dir, exist_ok=True)
    method_tag = "identity_cycle" if args.identity_cycle_rerank else "baseline"
    if args.identity_cycle_rerank:
        method_tag += f"_topk{args.icr_topk}_rw{args.icr_reverse_weight}"
    with open(os.path.join(save_dir, f"t{args.t}_b{args.k}_e{args.ensemble_size}_{method_tag}.json"), "w+", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPair-71k Evaluation Script with Identity-Cycle Rerank")
    parser.add_argument("--dataset_path", type=str, default="/dataset/SPair-71k", help="path to spair dataset")
    parser.add_argument("--dataset", type=str, default="SPair", help="dataset name")
    parser.add_argument("--save_path", type=str, default="/scratch/lt453/spair_ft/", help="path to save features")
    parser.add_argument("--dit_model", choices=["flux"], default="flux", help="which dit version to use")
    parser.add_argument("--img_size", nargs="+", type=int, default=[768, 768], help="input resize [w, h]")
    parser.add_argument("--t", default=260, type=int, help="diffusion timestep")
    parser.add_argument("--k", nargs="+", type=int, default=[28], help="which dit block to extract the ft map")
    parser.add_argument("--ensemble_size", default=8, type=int, help="ensemble size for getting an image ft map")
    parser.add_argument("--cd", action="store_true", default=False, help="whether adopt channel discard")
    parser.add_argument("--reuse_saved_features", action="store_true", default=False, help="skip feature extraction and reuse saved .pth features")
    parser.add_argument("--max_pairs_per_cat", default=0, type=int, help="optional category-wise cap for quick validation")
    parser.add_argument("--identity_cycle_rerank", action="store_true", default=False, help="rerank local forward candidates by reverse cycle consistency to suppress part confusion")
    parser.add_argument("--icr_topk", default=5, type=int, help="number of forward candidates considered for identity-cycle reranking")
    parser.add_argument("--icr_reverse_weight", default=0.1, type=float, help="penalty weight for reverse inconsistency distance")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(args)
