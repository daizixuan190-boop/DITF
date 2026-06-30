import gc

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, calculate_similarity


def prepare_txt(bs, t5, clip, prompt, device="cuda"):
    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3, dtype=txt.dtype, device=txt.device)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return txt.to(device), txt_ids.to(device), vec.to(device)


def prepare(img):
    bs, c, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3, dtype=img.dtype, device=img.device)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device, dtype=img.dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device, dtype=img.dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    return img, img_ids


class Featurizer:
    def __init__(self, name="flux-dev", null_prompt="", device="cuda"):
        self.name = name
        self.device = device
        self.null_prompt = null_prompt

        t5 = load_t5(device, max_length=512)
        clip = load_clip(device)
        model = load_flow_model(name, device=device)
        ae = load_ae(name, device=device)

        self.t5 = t5
        self.clip = clip
        self.model = model
        self.ae = ae

    @torch.no_grad()
    def forward(self, img_tensor, prompt="", t=261, up_ft_index=1, ensemble_size=8):
        raise NotImplementedError


class Featurizer4Eval(Featurizer):
    def __init__(self, flux_id="flux-dev", null_prompt="", cat_list=None, ensemble_size=1, device="cuda"):
        self.name = flux_id
        self.device = device
        self.null_prompt = null_prompt
        self.model = None
        self.ae = None
        self.t5 = load_t5("cpu", max_length=512)
        self.clip = load_clip("cpu")

        if cat_list is None:
            cat_list = []
        cat_list = list(cat_list)
        if "image" not in cat_list:
            cat_list.append("image")

        prompts = {cat: f"a photo of a {cat}" for cat in cat_list}

        print("Init T5 prompt cache")
        cat2txt = {}
        with torch.no_grad():
            for cat, prompt in prompts.items():
                txt = self.t5([prompt])
                if txt.shape[0] == 1 and ensemble_size > 1:
                    txt = repeat(txt, "1 ... -> bs ...", bs=ensemble_size)
                txt_ids = torch.zeros(ensemble_size, txt.shape[1], 3, dtype=txt.dtype, device=txt.device)
                cat2txt[cat] = (txt.cpu(), txt_ids.cpu())

        print("Init CLIP prompt cache")
        cat2vec = {}
        with torch.no_grad():
            for cat, prompt in prompts.items():
                vec = self.clip([prompt])
                if vec.shape[0] == 1 and ensemble_size > 1:
                    vec = repeat(vec, "1 ... -> bs ...", bs=ensemble_size)
                cat2vec[cat] = vec.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        self.cat2prompt_embeds = {
            cat: (cat2txt[cat][0], cat2txt[cat][1], cat2vec[cat]) for cat in cat_list
        }

    def _lazy_init_models(self):
        if self.model is None:
            print("Init model")
            self.model = load_flow_model(self.name, device=self.device)
        if self.ae is None:
            print("Init AE on CPU")
            self.ae = load_ae(self.name, device="cpu")

    @torch.no_grad()
    def forward(
        self,
        args,
        img_tensor,
        caption="a photo of a image",
        category="image",
        timestep=261,
        block_idx=1,
        ensemble_size=1,
        guidance=3.5,
    ):
        self._lazy_init_models()

        ae_dtype = next(self.ae.parameters()).dtype
        img_tensor = img_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1, 1)
        img_tensor_ae = img_tensor.to("cpu", dtype=ae_dtype)

        use_cached_prompt = caption == "a photo of a image"
        if use_cached_prompt:
            key = category if category in self.cat2prompt_embeds else "image"
            prompt_embeds, text_ids, vec = self.cat2prompt_embeds[key]
            prompt_embeds = prompt_embeds.to(self.device, dtype=torch.bfloat16, non_blocking=True)
            text_ids = text_ids.to(self.device, dtype=torch.bfloat16, non_blocking=True)
            vec = vec.to(self.device, dtype=torch.bfloat16, non_blocking=True)
        else:
            prompt_embeds, text_ids, vec = prepare_txt(
                bs=ensemble_size,
                t5=self.t5,
                clip=self.clip,
                prompt=caption,
                device=self.device,
            )

        device = self.device
        t = timestep / 1000

        latents = self.ae.encode(img_tensor_ae)
        latents = latents.to(self.device, dtype=torch.bfloat16)

        noise = torch.randn_like(latents, device=device)
        latents_noisy = t * noise + (1.0 - t) * latents
        ensem, c, h, w = latents_noisy.shape

        img, img_ids = prepare(img=latents_noisy)

        t_vec = torch.full((img.shape[0],), t, dtype=img.dtype, device=img.device)
        guidance_vec = torch.full((img.shape[0],), guidance, dtype=img.dtype, device=img.device)

        model_output = self.model.forward_feat(
            img=img,
            img_ids=img_ids,
            txt=prompt_embeds,
            txt_ids=text_ids,
            y=vec,
            timesteps=t_vec,
            ft_indices=block_idx,
            cat=category,
            guidance=guidance_vec,
        )

        mod = model_output[1]
        dit_feat = model_output[0]
        dit_feat = rearrange(dit_feat, "b (h w) c -> b h w c", h=h // 2, w=w // 2)
        dit_feat = dit_feat.permute(0, 3, 1, 2)
        dit_feat = dit_feat.mean(0, keepdim=True)

        mod = [
            mod.shift.mean(0, keepdim=True),
            mod.scale.mean(0, keepdim=True),
            mod.gate.mean(0, keepdim=True),
        ]

        return dit_feat, torch.cat(mod, dim=1)
