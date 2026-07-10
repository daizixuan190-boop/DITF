import argparse
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from src.flux.feat_flux import Featurizer4Eval
import os
import gc
import json
from PIL import Image
import torch.nn as nn
from einops import rearrange
import time
from torchvision.transforms import PILToTensor, ToPILImage

import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment


def topk_candidates_from_scores(score_vec, width, topk):
    k = min(int(topk), int(score_vec.numel()))
    if k <= 0:
        return []
    top_vals, top_idx = torch.topk(score_vec, k=k, dim=0)
    cand_y = torch.div(top_idx, width, rounding_mode='floor')
    cand_x = top_idx % width
    return [
        (int(cand_x[i].item()), int(cand_y[i].item()), float(top_vals[i].item()))
        for i in range(k)
    ]


def topk_candidates(cos_map, topk):
    flat = cos_map.reshape(-1)
    k = min(int(topk), int(flat.size))
    if k <= 0:
        return []
    top_idx = np.argpartition(-flat, k - 1)[:k]
    top_idx = top_idx[np.argsort(-flat[top_idx])]
    h, w = cos_map.shape
    return [(int(idx % w), int(idx // w), float(flat[idx])) for idx in top_idx]


def local_structure_score(src_ft, trg_ft, src_x, src_y, cand_x, cand_y, radius):
    _, channels, src_h, src_w = src_ft.shape
    _, _, trg_h, trg_w = trg_ft.shape
    sims = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sx = src_x + dx
            sy = src_y + dy
            tx = cand_x + dx
            ty = cand_y + dy
            if sx < 0 or sx >= src_w or sy < 0 or sy >= src_h:
                continue
            if tx < 0 or tx >= trg_w or ty < 0 or ty >= trg_h:
                continue
            src_vec = src_ft[0, :, sy, sx].view(1, channels)
            trg_vec = trg_ft[0, :, ty, tx].view(1, channels)
            sim = torch.sum(F.normalize(src_vec, dim=1) * F.normalize(trg_vec, dim=1)).item()
            sims.append(sim)
    if not sims:
        return 0.0
    return float(np.mean(sims))


def reverse_cycle_distance(src_ft, trg_ft, src_x, src_y, cand_x, cand_y, src_threshold):
    _, channels, src_h, src_w = src_ft.shape
    trg_vec = trg_ft[0, :, cand_y, cand_x].view(1, channels)
    trg_vec = F.normalize(trg_vec, dim=1).transpose(0, 1)
    src_matrix = src_ft.view(channels, -1).transpose(0, 1)
    src_matrix = F.normalize(src_matrix, dim=1)
    reverse_scores = torch.mm(src_matrix, trg_vec).view(src_h, src_w)
    rev_flat = int(reverse_scores.argmax())
    rev_y, rev_x = np.unravel_index(rev_flat, (src_h, src_w))
    reverse_dist = ((rev_x - src_x) ** 2 + (rev_y - src_y) ** 2) ** 0.5
    return float(reverse_dist / max(src_threshold, 1e-6))


def confusion_local_rerank(src_ft, trg_ft, src_point, candidate_list, src_threshold, args):
    if not candidate_list:
        return src_point[0], src_point[1]

    src_x, src_y = int(src_point[0]), int(src_point[1])
    best_xy = None
    best_score = None
    for cand_x, cand_y, forward_score in candidate_list:
        reverse_penalty = reverse_cycle_distance(src_ft, trg_ft, src_x, src_y, cand_x, cand_y, src_threshold)
        structure_bonus = local_structure_score(
            src_ft,
            trg_ft,
            src_x,
            src_y,
            cand_x,
            cand_y,
            args.clr_radius,
        )
        combined = (
            float(forward_score)
            + args.clr_structure_weight * structure_bonus
            - args.clr_reverse_weight * reverse_penalty
        )
        if best_score is None or combined > best_score:
            best_score = combined
            best_xy = (cand_x, cand_y)
    return best_xy if best_xy is not None else (candidate_list[0][0], candidate_list[0][1])


def build_pair_candidate_records(src_ft, trg_ft, src_points, src_threshold, topk, reverse_weight):
    channels = src_ft.shape[1]
    src_h, src_w = src_ft.shape[-2:]
    trg_h, trg_w = trg_ft.shape[-2:]

    src_matrix = src_ft.view(channels, -1).transpose(0, 1)
    src_matrix = F.normalize(src_matrix, dim=1)
    trg_matrix = trg_ft.view(channels, -1).transpose(0, 1)
    trg_matrix = F.normalize(trg_matrix, dim=1)

    src_xy = torch.tensor(
        [(int(point[0]), int(point[1])) for point in src_points],
        device=src_ft.device,
        dtype=torch.long,
    )
    src_vecs = src_ft[0, :, src_xy[:, 1], src_xy[:, 0]].transpose(0, 1).contiguous()
    src_vecs = F.normalize(src_vecs, dim=1)
    score_mat = torch.mm(trg_matrix, src_vecs.transpose(0, 1))
    k = min(int(topk), int(score_mat.shape[0]))
    top_vals, top_idx = torch.topk(score_mat, k=k, dim=0)
    cand_y = torch.div(top_idx, trg_w, rounding_mode='floor')
    cand_x = top_idx % trg_w

    cand_flat_idx = top_idx.reshape(-1)
    cand_vecs = trg_matrix[cand_flat_idx]
    reverse_scores = torch.mm(src_matrix, cand_vecs.transpose(0, 1))
    rev_idx = torch.argmax(reverse_scores, dim=0)
    rev_y = torch.div(rev_idx, src_w, rounding_mode='floor')
    rev_x = rev_idx % src_w

    src_x_repeat = src_xy[:, 0].float().repeat_interleave(k)
    src_y_repeat = src_xy[:, 1].float().repeat_interleave(k)
    reverse_dist = torch.sqrt(
        (rev_x.float() - src_x_repeat) ** 2 + (rev_y.float() - src_y_repeat) ** 2
    ) / max(float(src_threshold), 1e-6)
    reverse_dist = reverse_dist.view(k, -1)
    base_score_mat = top_vals.float() - float(reverse_weight) * reverse_dist

    records = []
    for point_idx, src_point in enumerate(src_points):
        src_x, src_y = int(src_point[0]), int(src_point[1])
        base_scores = base_score_mat[:, point_idx]
        if k > 1:
            margin = float((base_scores[0] - base_scores[1]).item())
        else:
            margin = float(base_scores[0].item())

        records.append(
            {
                "src_xy": (src_x, src_y),
                "cand_x": cand_x[:, point_idx],
                "cand_y": cand_y[:, point_idx],
                "base_scores": base_scores,
                "anchor_idx": int(torch.argmax(base_scores).item()),
                "margin": margin,
            }
        )

    return records


def confusion_pair_rerank(records, src_threshold, trg_threshold, args):
    if not records:
        return []

    device = records[0]["base_scores"].device
    src_xy = torch.tensor([record["src_xy"] for record in records], device=device, dtype=torch.float32)
    src_pair_dists = torch.cdist(src_xy, src_xy) / max(float(src_threshold), 1e-6)

    current_choice = [record["anchor_idx"] for record in records]
    anchor_strength = torch.tensor(
        [max(record["margin"], 0.0) for record in records],
        device=device,
        dtype=torch.float32,
    )
    if float(anchor_strength.max().item()) > 0.0:
        anchor_strength = anchor_strength / anchor_strength.max().clamp_min(1e-6)
    else:
        anchor_strength = torch.ones_like(anchor_strength)

    for _ in range(max(int(args.cpr_iters), 1)):
        changed = False
        anchor_xy = torch.stack(
            [
                torch.tensor(
                    [
                        float(records[i]["cand_x"][current_choice[i]].item()),
                        float(records[i]["cand_y"][current_choice[i]].item()),
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                for i in range(len(records))
            ],
            dim=0,
        )

        for i, record in enumerate(records):
            cand_xy = torch.stack(
                [record["cand_x"].float(), record["cand_y"].float()],
                dim=1,
            )
            target_dists = torch.cdist(cand_xy, anchor_xy) / max(float(trg_threshold), 1e-6)
            overlap = (1.0 - target_dists / float(args.cpr_radius)).clamp(min=0.0)

            sep_mask = (src_pair_dists[i] >= float(args.cpr_src_separation)).float()
            sep_mask[i] = 0.0
            penalty = overlap * sep_mask.unsqueeze(0) * anchor_strength.unsqueeze(0)
            final_scores = record["base_scores"] - float(args.cpr_weight) * penalty.sum(dim=1)

            best_idx = int(torch.argmax(final_scores).item())
            if best_idx != current_choice[i]:
                current_choice[i] = best_idx
                changed = True

        if not changed:
            break

    pred_xy = []
    for i, record in enumerate(records):
        pred_xy.append(
            (
                int(record["cand_x"][current_choice[i]].item()),
                int(record["cand_y"][current_choice[i]].item()),
            )
        )
    return pred_xy


def cluster_target_slots(records, trg_threshold, radius):
    slot_centers = []
    slots = []
    norm_radius = max(float(trg_threshold), 1e-6) * float(radius)

    for src_idx, record in enumerate(records):
        cand_x = record["cand_x"].detach().cpu().tolist()
        cand_y = record["cand_y"].detach().cpu().tolist()
        base_scores = record["base_scores"].detach().cpu().tolist()
        for cand_idx, (x, y, score) in enumerate(zip(cand_x, cand_y, base_scores)):
            assigned = None
            for slot_idx, center in enumerate(slot_centers):
                dist = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5
                if dist <= norm_radius:
                    assigned = slot_idx
                    break
            if assigned is None:
                assigned = len(slot_centers)
                slot_centers.append([float(x), float(y), 1.0])
                slots.append([])
            else:
                center = slot_centers[assigned]
                center[0] += float(x)
                center[1] += float(y)
                center[2] += 1.0
            slots[assigned].append(
                {
                    "src_idx": src_idx,
                    "cand_idx": cand_idx,
                    "x": int(x),
                    "y": int(y),
                    "score": float(score),
                }
            )

    slot_summary = []
    for slot_idx, center in enumerate(slot_centers):
        slot_summary.append(
            {
                "center_x": center[0] / center[2],
                "center_y": center[1] / center[2],
                "members": slots[slot_idx],
            }
        )
    return slot_summary


def assignment_confusion_rerank(records, trg_threshold, args):
    num_points = len(records)
    if num_points == 0:
        return []

    slots = cluster_target_slots(records, trg_threshold, args.acr_radius)
    num_slots = len(slots)
    total_cols = num_slots + num_points
    score_matrix = np.full((num_points, total_cols), -1e6, dtype=np.float32)
    pick_matrix = [[None for _ in range(total_cols)] for _ in range(num_points)]

    for slot_idx, slot in enumerate(slots):
        for member in slot["members"]:
            src_idx = member["src_idx"]
            if member["score"] > score_matrix[src_idx, slot_idx]:
                score_matrix[src_idx, slot_idx] = member["score"]
                pick_matrix[src_idx][slot_idx] = (member["x"], member["y"])

    for src_idx, record in enumerate(records):
        anchor_idx = int(torch.argmax(record["base_scores"]).item())
        fallback_score = float(record["base_scores"][anchor_idx].item()) - float(args.acr_fallback_penalty)
        col_idx = num_slots + src_idx
        score_matrix[src_idx, col_idx] = fallback_score
        pick_matrix[src_idx][col_idx] = (
            int(record["cand_x"][anchor_idx].item()),
            int(record["cand_y"][anchor_idx].item()),
        )

    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    chosen = [None] * num_points
    for row, col in zip(row_ind, col_ind):
        chosen[row] = pick_matrix[row][col]

    pred_xy = []
    for src_idx, record in enumerate(records):
        xy = chosen[src_idx]
        if xy is None:
            anchor_idx = int(torch.argmax(record["base_scores"]).item())
            xy = (
                int(record["cand_x"][anchor_idx].item()),
                int(record["cand_y"][anchor_idx].item()),
            )
        pred_xy.append(xy)
    return pred_xy

def quantile_risk_map(values, quantile, tail="high"):
    flat = values.flatten()
    threshold = torch.quantile(flat, quantile)

    if tail == "high":
        ref = flat.max()
        denom = (ref - threshold).clamp_min(1e-6)
        return ((values - threshold) / denom).clamp(min=0.0, max=1.0)

    if tail == "low":
        ref = flat.min()
        denom = (threshold - ref).clamp_min(1e-6)
        return ((threshold - values) / denom).clamp(min=0.0, max=1.0)

    raise ValueError(f"Unsupported tail: {tail}")


def maybe_apply_shift_calibration(ft_ln, shift, scale, args):
    content = (1 + scale) * ft_ln
    ft_post = content + shift
    if not args.shift_calibration and not args.joint_calibration:
        return ft_post

    shift_energy = torch.sum(shift.float() ** 2, dim=1, keepdim=True)
    content_energy = torch.sum(content.float() ** 2, dim=1, keepdim=True)
    post_energy = torch.sum(ft_post.float() ** 2, dim=1, keepdim=True).clamp_min(1e-6)
    shift_ratio = shift_energy / post_energy
    content_ratio = content_energy / post_energy

    if args.joint_calibration:
        high_shift = quantile_risk_map(
            shift_ratio,
            args.joint_shift_quantile,
            tail="high",
        )
        low_content = quantile_risk_map(
            content_ratio,
            args.joint_content_quantile,
            tail="low",
        )
        joint_risk = high_shift * low_content

        lambda_map = 1.0 - args.joint_shift_strength * joint_risk
        lambda_map = lambda_map.clamp(min=args.joint_min_shift_lambda, max=1.0)

        content_gain = 1.0 + args.joint_content_strength * joint_risk
        content_gain = content_gain.clamp(max=args.joint_max_content_gain)

        return content_gain.to(content.dtype) * content + lambda_map.to(content.dtype) * shift

    excess = quantile_risk_map(
        shift_ratio,
        args.shift_calibration_quantile,
        tail="high",
    )
    lambda_map = 1.0 - args.shift_calibration_strength * excess
    lambda_map = lambda_map.clamp(min=args.shift_calibration_min_lambda, max=1.0)

    return content + lambda_map.to(content.dtype) * shift

def main(args):
    for arg in vars(args):
        value = getattr(args,arg)
        if value is not None:
            print('%s: %s' % (str(arg),str(value)))

    torch.cuda.set_device(0)

    dataset_path = args.dataset_path
    test_path = 'PairAnnotation/test'
    json_list = os.listdir(os.path.join(dataset_path, test_path))
    all_cats = os.listdir(os.path.join(dataset_path, 'JPEGImages'))
    cat2json = {}

    for cat in all_cats:
        cat_list = []
        for i in json_list:
            if cat in i:
                cat_list.append(i)
        cat2json[cat] = cat_list

    # get test image path for all cats
    cat2img = {}
    for cat in all_cats:
        cat2img[cat] = []
        cat_list = cat2json[cat]
        for json_path in cat_list:
            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)
                temp_f.close()
            src_imname = data['src_imname']
            trg_imname = data['trg_imname']
            if src_imname not in cat2img[cat]:
                cat2img[cat].append(src_imname)
            if trg_imname not in cat2img[cat]:
                cat2img[cat].append(trg_imname)

    if not args.reuse_saved_features:
        if args.dit_model == 'flux':
            dit_model = Featurizer4Eval(cat_list=all_cats[:], ensemble_size=args.ensemble_size)
        else:
            raise Exception("model must be in [flux] ")

    print("saving all test images' features...")
    os.makedirs(args.save_path, exist_ok=True)
    if args.reuse_saved_features:
        print("Reusing existing saved features; skip feature extraction.")
    
    captions = {}
    if not args.reuse_saved_features:
        # detailed caption generated by pretrained MLLM. Bring about 0.3% gain for flux
        with open("spair_detailed_captions.json") as temp_f:
            captions = json.load(temp_f)
    
    for cat in tqdm(all_cats):
        if args.reuse_saved_features:
            feat_path = os.path.join(args.save_path, f'{cat}.pth')
            ada_path = os.path.join(args.save_path, f'{cat}_ada.pth')
            if os.path.exists(feat_path) and os.path.exists(ada_path):
                continue
            raise FileNotFoundError(f"Missing cached features for {cat}: {feat_path} or {ada_path}")

        output_dict = {}
        ada_dict = {}
        
        image_list = cat2img[cat]
        for image_path in image_list:
            
            img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, image_path))
            
            ###preprocess 
            
            image_arr = np.array(img)
            in_h, in_w = image_arr.shape[:2]
            scale = args.img_size[0] / max(in_h, in_w)
            H = int(round(in_h * scale / 16)) * 16  # 保证是16的倍数
            W = int(round(in_w * scale / 16)) * 16
            img_size = (W, H)
            img = img.resize(img_size)
            img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
            
            
            caption = captions[cat+image_path]
            
            output_dict[image_path], ada_dict[image_path] = dit_model.forward(args, 
                                                                        img_tensor,
                                                                        caption=caption,
                                                                        category=cat,
                                                                        timestep=args.t,
                                                                        block_idx=args.k,
                                                                        ensemble_size=args.ensemble_size)
            
            output_dict[image_path], ada_dict[image_path] = output_dict[image_path].cpu(), ada_dict[image_path].cpu()
        
        torch.save(output_dict, os.path.join(args.save_path, f'{cat}.pth'))
        torch.save(ada_dict, os.path.join(args.save_path, f'{cat}_ada.pth'))
    
    if not args.reuse_saved_features:
        del dit_model
        gc.collect()
        torch.cuda.empty_cache()

    total_pck = []
    all_correct = 0
    all_total = 0
    
    
    #### layernorm of our adaln for dit feature, 3072 is feature dimension of flux.
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)
    
        
    mean_image_sum=0
    mean_point_sum=0
    
    result={"image":{},"point":{}}
    
    print("Category numbers: %s"%len(all_cats))
    for cat in all_cats:
        cat_list = cat2json[cat]
        #### load data feature
        output_dict = torch.load(os.path.join(args.save_path, f'{cat}.pth'))
        ada_dict = torch.load(os.path.join(args.save_path, f'{cat}_ada.pth'))
        
        cat_pck = []
        cat_correct = 0
        cat_total = 0
        if args.max_pairs_per_cat > 0:
            cat_list = cat_list[:args.max_pairs_per_cat]
        
        for cat_idx, json_path in enumerate(tqdm(cat_list)):
            
            ##load image pair
            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)

            src_img_size = data['src_imsize'][:2][::-1]
            trg_img_size = data['trg_imsize'][:2][::-1]
            
            # B,C,H,W = 
            src_ft_raw = output_dict[data['src_imname']].cuda()
            B,C,H,W = src_ft_raw.shape
            src_ada = ada_dict[data['src_imname']].cuda()
            
            # Channel discard
            # We suppress Massive Activations (MAs) in DiT features by discarding their channels, 
            # preventing LayerNorm from propagating their adverse influence to the remaining dimensions. 
            # For a given DiT, the MA dimensions are fixed and easy to identify; we simply zero those channels.
            if args.cd:
                src_ft_raw[:,154,:,:]=0.0
                src_ft_raw[:,1446,:,:]=0.0
            
            src_ft = rearrange(src_ft_raw, "b c h w -> b (h w) c")
            src_ft = pre_norm(src_ft)
            src_ft = rearrange(src_ft, "b (h w) c -> b c h w", h=H, w=W)
            src_ft_raw = src_ft.clone()
            
            src_shift = src_ada[0][0]
            src_scale = src_ada[0][1]
            
            src_shift = src_shift.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            src_scale = src_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            src_ft = maybe_apply_shift_calibration(src_ft, src_shift, src_scale, args)
            
            
            trg_ft_raw = output_dict[data['trg_imname']].cuda()
            B,C,H,W = trg_ft_raw.shape
            trg_ada = ada_dict[data['trg_imname']].cuda()
            
            
            # Channel discard
            if args.cd:
                trg_ft_raw[:,154,:,:]=0.0
                trg_ft_raw[:,1446,:,:]=0.0
            
            trg_ft = rearrange(trg_ft_raw, "b c h w -> b (h w) c")
            trg_ft = pre_norm(trg_ft)
            trg_ft = rearrange(trg_ft, "b (h w) c -> b c h w", h=H, w=W)
            trg_ft_raw = trg_ft.clone()
            
            trg_shift = trg_ada[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            trg_scale = trg_ada[0][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            trg_ft = maybe_apply_shift_calibration(trg_ft, trg_shift, trg_scale, args)
            
            src_ft = src_ft.to(torch.float16)
            B, C, H, W = src_ft.shape
            trg_ft = trg_ft.to(torch.float16)
            
                
            src_ft = nn.Upsample(size=src_img_size, mode='bilinear')(src_ft)
            trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft)
            
            
            h = trg_ft.shape[-2]
            w = trg_ft.shape[-1]

            src_bndbox = data['src_bndbox']
            trg_bndbox = data['trg_bndbox']
            src_threshold = max(src_bndbox[3] - src_bndbox[1], src_bndbox[2] - src_bndbox[0])
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            total = 0
            correct = 0
            trg_list = []
            src_points = data['src_kps']
            trg_points = data['trg_kps']
            num_channel = src_ft.size(1)
            trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1)
            trg_vec = F.normalize(trg_vec, dim=1)

            pair_predictions = None
            if args.assignment_confusion_rerank or args.confusion_pair_rerank:
                pair_records = build_pair_candidate_records(
                    src_ft,
                    trg_ft,
                    src_points,
                    src_threshold,
                    args.acr_topk if args.assignment_confusion_rerank else args.cpr_topk,
                    args.acr_reverse_weight if args.assignment_confusion_rerank else args.cpr_reverse_weight,
                )
                if args.assignment_confusion_rerank:
                    pair_predictions = assignment_confusion_rerank(
                        pair_records,
                        threshold,
                        args,
                    )
                else:
                    pair_predictions = confusion_pair_rerank(
                        pair_records,
                        src_threshold,
                        threshold,
                        args,
                    )

            for idx in range(len(src_points)):
                total += 1
                cat_total += 1
                all_total += 1
                src_point = src_points[idx]
                trg_point = trg_points[idx]

                if pair_predictions is not None:
                    pred_x, pred_y = pair_predictions[idx]
                else:
                    src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel)
                    src_vec = F.normalize(src_vec, dim=1).transpose(0, 1)
                    score_vec = torch.mm(trg_vec, src_vec).squeeze(1)

                    if args.confusion_local_rerank:
                        candidates = topk_candidates_from_scores(score_vec, w, args.clr_topk)
                        pred_x, pred_y = confusion_local_rerank(
                            src_ft,
                            trg_ft,
                            src_point,
                            candidates,
                            src_threshold,
                            args,
                        )
                    else:
                        max_idx = int(torch.argmax(score_vec).item())
                        pred_y = max_idx // w
                        pred_x = max_idx % w

                trg_list.append([pred_x, pred_y])
                dist = ((pred_x - trg_point[0]) ** 2 + (pred_y - trg_point[1]) ** 2) ** 0.5
                if (dist / threshold) <= 0.1:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1

            cat_pck.append(correct / total)
            
        total_pck.extend(cat_pck)
        
        mean_image_sum = mean_image_sum + np.mean(cat_pck) * 100
        
        mean_point_sum = mean_point_sum + cat_correct / cat_total * 100
        
        
        print(f'{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}')
        print(f'{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}')
        
        result['image'][cat] = round(np.mean(cat_pck) * 100, 2)
        result['point'][cat] = round(cat_correct / cat_total * 100, 2)
        
    print(f'All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}')
    print(f'All per point PCK@0.1: {all_correct / all_total * 100:.2f}')
    
    print(f'Mean per image PCK@0.1: {mean_image_sum / len(all_cats):.2f}')
    print(f'Mean per point PCK@0.1: {mean_point_sum / len(all_cats):.2f}')
    
    result['image']["All"] = round(np.mean(total_pck) * 100, 2)
    result['point']["All"] = round(all_correct / all_total * 100, 2)
    
    result['image']["Mean"] = round(mean_image_sum / len(all_cats), 2)
    result['point']["Mean"] = round(mean_point_sum / len(all_cats), 2)
    
    
    save_dir = os.path.join("layers_cat", args.dit_model)
    os.makedirs(save_dir, exist_ok=True)
    method_tag = "baseline"
    if args.assignment_confusion_rerank:
        method_tag = (
            f"acr_topk{args.acr_topk}_r{args.acr_radius}_fb{args.acr_fallback_penalty}"
            f"_rw{args.acr_reverse_weight}"
        )
    elif args.confusion_pair_rerank:
        method_tag = (
            f"cpr_topk{args.cpr_topk}_r{args.cpr_radius}_w{args.cpr_weight}"
            f"_src{args.cpr_src_separation}_it{args.cpr_iters}_rw{args.cpr_reverse_weight}"
        )
    elif args.confusion_local_rerank:
        method_tag = f"clr_topk{args.clr_topk}_r{args.clr_radius}_sw{args.clr_structure_weight}_rw{args.clr_reverse_weight}"
    with open(os.path.join(save_dir, f't{args.t}_b{args.k}_e{args.ensemble_size}_{method_tag}.json'), 'w+') as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # print("test")
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Script')
    parser.add_argument('--dataset_path', type=str, default='/dataset/SPair-71k', help='path to spair dataset')
    parser.add_argument('--dataset', type=str, default='SPair', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='/scratch/lt453/spair_ft/', help='path to save features')
    parser.add_argument('--dit_model', choices=['flux'], default='flux', help="which dit version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t', default=260, type=int, help='t for diffusion') ###调参[1,1000]
    parser.add_argument('--k', nargs='+', type=int, default=[28], help='which dit block to extract the ft map') ###调参[0,57]
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    parser.add_argument("--cd", action="store_true", default=False, help='whether adopt channel discard.')
    parser.add_argument("--reuse_saved_features", action="store_true", default=False, help="skip feature extraction and reuse saved .pth features")
    parser.add_argument("--max_pairs_per_cat", default=0, type=int, help="optional category-wise cap for quick validation")
    parser.add_argument("--shift_calibration", action="store_true", default=False, help="apply training-free post-AdaLN shift calibration before matching")
    parser.add_argument("--shift_calibration_quantile", default=0.75, type=float, help="only suppress tokens above this shift-ratio quantile")
    parser.add_argument("--shift_calibration_strength", default=0.5, type=float, help="suppression strength for high-shift tokens")
    parser.add_argument("--shift_calibration_min_lambda", default=0.2, type=float, help="minimum retained shift scaling after calibration")
    parser.add_argument("--joint_calibration", action="store_true", default=False, help="apply joint high-shift low-content calibration before matching")
    parser.add_argument("--joint_shift_quantile", default=0.75, type=float, help="high-shift token threshold for joint calibration")
    parser.add_argument("--joint_content_quantile", default=0.25, type=float, help="low-content token threshold for joint calibration")
    parser.add_argument("--joint_shift_strength", default=0.5, type=float, help="shift suppression strength under joint calibration")
    parser.add_argument("--joint_min_shift_lambda", default=0.2, type=float, help="minimum retained shift scaling under joint calibration")
    parser.add_argument("--joint_content_strength", default=0.25, type=float, help="content amplification strength under joint calibration")
    parser.add_argument("--joint_max_content_gain", default=1.5, type=float, help="upper bound for content amplification under joint calibration")
    parser.add_argument("--confusion_local_rerank", action="store_true", default=False, help="rerank local forward candidates using reverse consistency plus local structure agreement")
    parser.add_argument("--clr_topk", default=5, type=int, help="number of forward candidates considered for confusion-aware reranking")
    parser.add_argument("--clr_radius", default=1, type=int, help="local structure radius in pixels")
    parser.add_argument("--clr_structure_weight", default=0.2, type=float, help="weight for local structure agreement bonus")
    parser.add_argument("--clr_reverse_weight", default=0.1, type=float, help="weight for reverse inconsistency penalty")
    parser.add_argument("--confusion_pair_rerank", action="store_true", default=False, help="rerank keypoints jointly to suppress collapse onto another target part")
    parser.add_argument("--cpr_topk", default=5, type=int, help="number of forward candidates retained per source keypoint for pair-level reranking")
    parser.add_argument("--cpr_radius", default=0.6, type=float, help="target-space suppression radius normalized by target threshold")
    parser.add_argument("--cpr_weight", default=0.2, type=float, help="strength of pair-level overlap suppression")
    parser.add_argument("--cpr_src_separation", default=0.6, type=float, help="minimum source-space separation for two keypoints to repel each other")
    parser.add_argument("--cpr_iters", default=2, type=int, help="number of pair-level reranking refinement iterations")
    parser.add_argument("--cpr_reverse_weight", default=0.1, type=float, help="reverse-cycle penalty weight inside pair-level candidate scoring")
    parser.add_argument("--assignment_confusion_rerank", action="store_true", default=False, help="solve a unique-slot assignment over local target basins to reduce other-part attraction")
    parser.add_argument("--acr_topk", default=5, type=int, help="number of local candidates retained per source keypoint for assignment reranking")
    parser.add_argument("--acr_radius", default=0.6, type=float, help="target-space clustering radius normalized by target threshold")
    parser.add_argument("--acr_fallback_penalty", default=0.05, type=float, help="penalty for falling back to the source-specific best candidate instead of a shared unique slot")
    parser.add_argument("--acr_reverse_weight", default=0.1, type=float, help="reverse-cycle penalty weight inside assignment candidate scoring")
    args = parser.parse_args()
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # print(args)
    main(args)
