"""Equal-NFE A/B on SDXL: PECE vs corrector=none vs the current default.

Each seed gets its own 80-step dpmpp_2m reference. All variants share the
seed/prompt/cfg/schedule, so differences are the integrator only.
"""

from __future__ import annotations

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
from omni_sampler.tools.sd_bench import ssim

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, sharp focus, 85mm photo"
NEG = "blurry, low quality, watermark, text"
RES = 1024
SEEDS = (42, 1234)
STEPS = (12, 20)


def main():
    ck = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(COMFY, "models", "checkpoints"))
          for f in fs if f.endswith(".safetensors") and "SUPIR" not in f
          and os.path.getsize(os.path.join(r, f)) > 3 * 1024 ** 3][0]
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        ck, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    latent = torch.zeros([1, 4, RES // 8, RES // 8])

    def render(steps, opts, seed, sampler_name="omni_hybrid"):
        sampler = comfy.samplers.ksampler(sampler_name, opts)
        torch.manual_seed(seed)
        noise = comfy.sample.prepare_noise(latent, seed)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler="karras", denoise=1.0, model_options=model.model_options)
        with torch.no_grad():
            samples = comfy.samplers.sample(
                model, noise, pos, neg, 5.0, model.load_device, sampler, ks.sigmas,
                model_options=model.model_options, latent_image=latent, disable_pbar=True, seed=seed)
            img = vae.decode(samples.to(model.load_device))
        img = ((img + 1.0) / 2.0).clamp(0, 1).detach().cpu()[0]
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        return img.permute(2, 0, 1).unsqueeze(0)

    VARIANTS = [
        ("current(corr1)", {}, +1),
        ("corr=none", {"corrector": "none"}, 0),
        ("pece", {"pece": True}, 0),
    ]
    rows = {}
    for seed in SEEDS:
        ref = render(80, {}, seed, sampler_name="dpmpp_2m")
        for steps in STEPS:
            for name, opts, extra in VARIANTS:
                t = render(steps, opts, seed)
                rows.setdefault((steps, name), []).append(ssim(t, ref))
                if seed == SEEDS[0]:
                    Image.fromarray((t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
                        os.path.join(OUT, f"ab2_{name}_{steps}.png"))

    print(f"{'steps':>5s} {'variant':15s} {'ssim(mean)':>10s} {'per seed':>18s} {'nfe':>4s}")
    for steps in STEPS:
        for name, opts, extra in VARIANTS:
            v = rows[(steps, name)]
            print(f"{steps:5d} {name:15s} {sum(v)/len(v):10.4f} {str([round(x, 4) for x in v]):>18s} {steps + extra:4d}")


if __name__ == "__main__":
    main()
