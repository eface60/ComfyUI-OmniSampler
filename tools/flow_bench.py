"""A/B on a real flow-matching model (PixelDiT 1.3B, 1024px).

Reference = 100-step euler (exact-step formulation, so it is flow-correct).
Metrics: SSIM against that reference, wall time. Includes comfyui-ppm's
euler_gamma_cfgpp when that pack is installed.
"""

from __future__ import annotations

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
import comfy.sample
import comfy.sd
import comfy.samplers
import folder_paths
from omni_sampler.tools.sd_bench import ssim

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark"
SEED = 7
RES = 1024
SCHED = "simple"
UNET = os.path.join(COMFY, "models", "diffusion_models", "pixeldit_1300m_1024px_bf16.safetensors")
TE = os.path.join(COMFY, "models", "text_encoders", "gemma_2_2b_it_elm_bf16.safetensors")


def main():
    model = comfy.sd.load_diffusion_model(UNET, model_options={})
    clip = comfy.sd.load_clip([TE], embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.PIXELDIT)
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    latent = torch.zeros([1, 3, RES, RES])
    print("available gamma/cfgpp names:",
          [n for n in comfy.samplers.SAMPLER_NAMES if "gamma" in n or "cfg_pp" in n][:8])

    def render(steps, name, opts=None):
        sampler = comfy.samplers.ksampler(name, opts or {})
        torch.manual_seed(SEED)
        noise = comfy.sample.prepare_noise(latent, SEED)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler=SCHED, denoise=1.0, model_options=model.model_options)
        t0 = time.time()
        with torch.no_grad():
            s = comfy.samplers.sample(
                model, noise, pos, neg, 4.0, model.load_device, sampler, ks.sigmas,
                model_options=model.model_options, latent_image=latent, disable_pbar=True, seed=SEED)
        dt = time.time() - t0
        x = s.detach().cpu()[0].float()
        x = ((x + 1.0) / 2.0).clamp(0, 1) if x.min() < -0.05 else x.clamp(0, 1)
        return x.unsqueeze(0), dt

    ref, _ = render(100, "euler")
    Image.fromarray((ref[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
        os.path.join(OUT, "flow_ref_100euler.png"))

    variants = [
        ("euler", None, 0),
        ("dpmpp_2m", None, 0),
        ("omni_hybrid", None, 1),
        ("omni_hybrid_cfg_pp", None, 0),
    ]
    if "euler_gamma_cfgpp" in comfy.samplers.SAMPLER_NAMES:
        variants.append(("euler_gamma_cfgpp", None, 0))

    print(f"{'steps':>5s} {'sampler':22s} {'ssim':>7s} {'sec':>6s} {'nfe':>4s}")
    for steps in (12, 20):
        for name, opts, extra in variants:
            try:
                img, dt = render(steps, name, opts)
                Image.fromarray((img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
                    os.path.join(OUT, f"flow_{name}_{steps}.png"))
                print(f"{steps:5d} {name:22s} {ssim(img, ref):7.4f} {dt:6.1f} {steps + extra:4d}")
            except Exception as exc:  # noqa: BLE001
                print(f"{steps:5d} {name:22s} FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
