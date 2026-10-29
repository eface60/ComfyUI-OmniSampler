"""Visual check of CFG++ at cfg=1 on a real SDXL model.

Renders the same seed/prompt with:
  * omni_hybrid            (no CFG++)
  * omni_hybrid_cfg_pp     (strength 0.5, the default)
  * omni_hybrid_cfg_pp     (strength 1.0, reference CFG++)
  * comfy's euler_cfg_pp   (reference implementation)
at cfg=1, where "no guidance" is what a user expects.
"""

from __future__ import annotations

import json
import os
import sys

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

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, sharp focus, 85mm photo"
NEG = "blurry, low quality, watermark, text"
SEED = 42
RES = 1024
STEPS = 20


def main():
    ckpt = None
    for root, _, files in os.walk(os.path.join(COMFY, "models", "checkpoints")):
        for f in files:
            if f.endswith(".safetensors") and "SUPIR" not in f and os.path.getsize(os.path.join(root, f)) > 3 * 1024 ** 3:
                ckpt = ckpt or os.path.join(root, f)
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        ckpt, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings")
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    latent = torch.zeros([1, 4, RES // 8, RES // 8])

    def run(tag, cfg, sampler_name=None, sampler_obj=None, extra=None):
        torch.manual_seed(SEED)
        noise = comfy.sample.prepare_noise(latent, SEED)
        sampler = sampler_obj or comfy.samplers.sampler_object(sampler_name)
        if extra:
            sampler = comfy.samplers.KSAMPLER(sampler.sampler_function, {**sampler.extra_options, **extra})
        ks = comfy.samplers.KSampler(model, steps=STEPS, device=model.load_device, sampler=sampler_name or "euler",
                                     scheduler="karras", denoise=1.0, model_options=model.model_options)
        ks.sampler = sampler_name or ks.sampler
        sigmas = ks.sigmas
        samples = comfy.samplers.sample(
            model, noise, positive, negative, cfg, model.load_device, sampler, sigmas,
            model_options=model.model_options, latent_image=latent, disable_pbar=True, seed=SEED,
        )
        with torch.no_grad():
            img = vae.decode(samples.to(model.load_device))
        img = ((img + 1.0) / 2.0).clamp(0, 1).detach().cpu()[0]
        if img.shape[0] == 3:  # CHW -> HWC (this VAE decodes to NHWC)
            img = img.permute(1, 2, 0)
        arr = (img.numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(OUT, f"cfgpp_{tag}.png"))
        print(f"  {tag:28s} cfg={cfg}  mean={float(img.mean()):.3f} std={float(img.std()):.3f}")
        return img

    print(f"checkpoint: {ckpt}")
    print(f"prompt: {PROMPT}\ncfg = 1.0 (no guidance expected)\n")
    run("plain_cfg1", 1.0, sampler_name="omni_hybrid")
    run("cfgpp_s0.5_cfg1", 1.0, sampler_obj=comfy.samplers.sampler_object("omni_hybrid_cfg_pp"))
    run("cfgpp_s1.0_cfg1", 1.0,
        sampler_obj=comfy.samplers.sampler_object("omni_hybrid_cfg_pp"), extra={"cfg_pp_strength": 1.0})
    run("comfy_euler_cfgpp_cfg1", 1.0, sampler_name="euler_cfg_pp")
    print(f"\nimages in {OUT}")


if __name__ == "__main__":
    main()
