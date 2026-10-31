"""Cold vs warm vs previews: find the real source of the '3x slower' report."""

from __future__ import annotations

import os
import sys
import time

import torch

COMFY = r"G:\AI\comfy"
WORK = r"G:\AI\agent working folders\generic"
sys.path.insert(0, COMFY)
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "omni_sampler"))

import omni_core  # noqa: E402
import omni_sampler  # noqa: E402,F401
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import latent_preview  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, load  # noqa: E402

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"


def main(family="krea2", res=1024, steps=10):
    model, clip, vae, lf = load(family)
    cfg = FAMILIES[family]["cfg"]
    hw = res // lf.spacial_downscale_ratio
    latent = (torch.zeros([1, lf.latent_channels, 1, hw, hw])
              if getattr(lf, "latent_dimensions", 2) == 3 else
              torch.zeros([1, lf.latent_channels, hw, hw]))
    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    def run(label, sampler, preview=False):
        noise = comfy.sample.prepare_noise(latent, 7)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler="simple", denoise=1.0, model_options=model.model_options)
        cb = latent_preview.prepare_callback(model, steps) if preview else None
        t0 = time.perf_counter()
        with torch.no_grad():
            comfy.samplers.sample(model, noise, pos, neg, cfg, model.load_device, sampler, ks.sigmas,
                                  model_options=model.model_options, latent_image=latent,
                                  callback=cb, disable_pbar=True, seed=7)
        print(f"  {label:34s} {time.perf_counter() - t0:6.2f}s")

    euler = comfy.samplers.sampler_object("euler")
    omni = comfy.samplers.KSAMPLER(lambda m, x, s, **k: omni_core.sample_omni(m, x, s, **k))

    print(f"[{family}] {res}px {steps} steps cfg={cfg}")
    run("euler cold", euler)
    run("euler warm", euler)
    run("omni cold", omni)
    run("omni warm", omni)
    run("omni warm, previews on", omni, preview=True)
    run("omni warm, previews on (again)", omni, preview=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "krea2")
