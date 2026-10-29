"""Benchmark on modern flow models (krea2 / moss / anima), in one process.

Loads the diffusion model + its text encoder + VAE, renders a high-NFE
reference with euler (exact-step, flow-correct) and then a matrix of samplers.
Everything runs in-process because the int8 checkpoints are not bit-reproducible
across sessions.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

COMFY = r"G:\AI\comfy"
PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, COMFY)
sys.path.insert(0, PARENT)

import omni_sampler  # noqa: E402,F401
import comfy.model_management
import comfy.sample
import comfy.sd
import comfy.samplers
import comfy.utils
import folder_paths
from omni_sampler.tools.sd_bench import ssim

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
DM = os.path.join(COMFY, "models", "diffusion_models")
TE = os.path.join(COMFY, "models", "text_encoders")
VAE_DIR = os.path.join(COMFY, "models", "vae")

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"
SEED = 7

FAMILIES = {
    "krea2": dict(
        unet=os.path.join(DM, "krea2", "krea2TurboOfficialComfy_krea2TurboInt8.safetensors"),
        te=[os.path.join(TE, "qwen3vl_4b_fp8_scaled.safetensors")],
        clip_type="KREA2",
        vae=os.path.join(VAE_DIR, "wan_2.1_vae.safetensors"),
        cfg=1.0,
        res=1024,
        vae_channels=None,
    ),
    "moss": dict(
        unet=os.path.join(DM, "moss", "moss_image_base_int8_convrot.safetensors"),
        te=[os.path.join(TE, "qwen3.5_9b_int8_convrot.safetensors")],
        clip_type="MOSS_IMAGE",
        vae=os.path.join(VAE_DIR, "flux2-vae.safetensors"),
        cfg=3.5,
        res=1024,
        vae_channels=None,
    ),
    "anima": dict(
        unet=os.path.join(DM, "anima", "anima-base-v1.0.safetensors"),
        te=[os.path.join(TE, "qwen_3_06b_base.safetensors"), os.path.join(TE, "t5xxl_fp8_e4m3fn.safetensors")],
        clip_type="COSMOS",
        vae=os.path.join(VAE_DIR, "hdrVAEAnimaKrea2QWEN_bf16.safetensors"),
        cfg=3.5,
        res=1024,
        vae_channels=None,
    ),
}


def load(family):
    cfg = FAMILIES[family]
    t0 = time.time()
    model = comfy.sd.load_diffusion_model(cfg["unet"], model_options={})
    ms = model.get_model_object("model_sampling")
    lf = model.get_model_object("latent_format")
    ctype = getattr(comfy.sd.CLIPType, cfg["clip_type"])
    clip = comfy.sd.load_clip(cfg["te"], embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=ctype)
    vae_sd = comfy.utils.load_torch_file(cfg["vae"], safe_load=True)
    vae = comfy.sd.VAE(sd=vae_sd)
    print(f"[{family}] loaded in {time.time() - t0:.1f}s  sampling={type(ms).__name__} "
          f"flow={isinstance(ms, __import__('comfy.model_sampling', fromlist=['CONST']).CONST)} "
          f"latent_ch={lf.latent_channels} ratio={lf.spacial_downscale_ratio} "
          f"sigma_max={float(ms.sigma_max):.3f} vram={torch.cuda.memory_allocated() / 2**30:.1f}GB")
    return model, clip, vae, lf


VARIANTS = {
    "default": ("omni_hybrid", {}),
    "no_corr": ("omni_hybrid", {"corrector": "none"}),
    "o1": ("omni_hybrid", {"order": 1}),
    "nogate": ("omni_hybrid", {"min_step_ratio": 0.0}),
    "exact": ("omni_hybrid", {"x0_average": "exact"}),
    "nolowend": ("omni_hybrid", {"lower_order_to_end": 0}),
    "sde": ("omni_hybrid", {"eta": 0.3}),
    "cfgpp_s0": ("omni_hybrid", {"cfg_pp": True, "cfg_pp_strength": 0.0}),
    "cfgpp_s03": ("omni_hybrid", {"cfg_pp": True, "cfg_pp_strength": 0.3}),
    "euler": ("euler", {}),
    "dpmpp_2m": ("dpmpp_2m", {}),
    "dpmpp_2m_sde": ("dpmpp_2m_sde", {}),
    "res_multistep": ("res_multistep", {}),
    "heun": ("heun", {}),
    "nc_o1": ("omni_hybrid", {"corrector": "none", "order": 1}),
    "nc_nogate": ("omni_hybrid", {"corrector": "none", "min_step_ratio": 0.0}),
    "nc_nolowend": ("omni_hybrid", {"corrector": "none", "lower_order_to_end": 0}),
    "nc_exact": ("omni_hybrid", {"corrector": "none", "x0_average": "exact"}),
    "nc_sde": ("omni_hybrid", {"corrector": "none", "eta": 0.3}),
    "corr1": ("omni_hybrid", {"corrector": "first"}),
    "nc_cfgpp_s1": ("omni_hybrid", {"corrector": "none", "cfg_pp": True, "cfg_pp_strength": 1.0}),
}


def to_hwc(decoded):
    """(B,C,H,W) or video (B,C,T,H,W) or (B,H,W,C) -> single (H,W,3) tensor."""
    img = decoded
    if img.dim() == 5:            # video VAE (Wan21): take the first frame
        img = img[:, :, 0]
    img = img.detach().float().cpu()[0]
    img = ((img + 1.0) / 2.0).clamp(0, 1)
    if img.dim() == 4:            # (C,T,H,W) leftover
        img = img[:, 0]
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    return img.permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--steps", default="6,8,12")
    ap.add_argument("--ref-steps", type=int, default=100)
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--samplers", default="euler,dpmpp_2m,omni_hybrid,omni_hybrid_cfg_pp")
    ap.add_argument("--tag", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--ref-variant", default="")
    ap.add_argument("--res", type=int, default=0)
    args = ap.parse_args()

    model, clip, vae, lf = load(args.family)
    cfg_scale = FAMILIES[args.family]["cfg"]
    res = args.res or FAMILIES[args.family]["res"]
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    latent = torch.zeros([1, lf.latent_channels, res // lf.spacial_downscale_ratio,
                          res // lf.spacial_downscale_ratio])
    it = comfy.model_management.intermediate_device()

    def render(steps, name, opts=None):
        sampler = comfy.samplers.ksampler(name, opts or {})
        torch.manual_seed(SEED)
        noise = comfy.sample.prepare_noise(latent, SEED)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler=args.scheduler, denoise=1.0, model_options=model.model_options)
        t0 = time.time()
        with torch.no_grad():
            s = comfy.samplers.sample(
                model, noise, pos, neg, cfg_scale, model.load_device, sampler, ks.sigmas,
                model_options=model.model_options, latent_image=latent, disable_pbar=True, seed=SEED)
            img = vae.decode(s.to(model.load_device))
        dt = time.time() - t0
        return to_hwc(img).to(it), dt

    tag = f"{args.family}{('_' + args.tag) if args.tag else ''}"
    if args.ref_variant:
        rname, ropts = VARIANTS[args.ref_variant]
    else:
        rname, ropts = "euler", {}
    ref, dt = render(args.ref_steps, rname, ropts)
    Image.fromarray((ref[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
        os.path.join(OUT, f"{tag}_ref_{args.ref_steps}{args.ref_variant or rname}.png"))
    print(f"[{args.family}] reference {args.ref_steps} euler in {dt:.1f}s")

    print(f"{'steps':>5s} {'variant':22s} {'ssim':>7s} {'sec':>6s}")
    if args.variants:
        specs = [(k, *VARIANTS[k]) for k in args.variants.split(",")]
    else:
        specs = [(n, n, {}) for n in args.samplers.split(",")]
    for steps in [int(s) for s in args.steps.split(",")]:
        for label, name, opts in specs:
            try:
                img, dt = render(steps, name, opts)
                Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(OUT, f"{tag}_{label}_{steps}.png"))
                print(f"{steps:5d} {label:22s} {ssim(img.cpu(), ref.cpu()):7.4f} {dt:6.1f}")
            except Exception as exc:  # noqa: BLE001
                print(f"{steps:5d} {label:22s} FAILED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()








