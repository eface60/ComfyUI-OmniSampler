"""Compare comfyui-ppm's euler_gamma / euler_gamma+CFG++ against omni on the
modern flow models.

ppm's euler_gamma: sigma_hat = sigma + s_sigma_diff*(sigma/sigma_max), re-noise
with sqrt(sigma_hat^2-sigma^2), evaluate at sigma_hat, Euler step from
sigma_hat. Its cfg_pp branch is eps-only (alpha_t == 1), which matters on flow
models.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, COMFY)
sys.path.insert(0, PARENT)

import omni_sampler  # noqa: E402,F401
import comfy.sample
import comfy.samplers
from omni_sampler.tools.modern_bench import FAMILIES, load, to_hwc  # noqa: E402
from omni_sampler.tools.sd_bench import ssim  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"
SEED = 7
PPM = os.path.join(COMFY, "custom_nodes", "comfyui-ppm", "sampling", "ppm_sampling.py")


def load_ppm():
    spec = importlib.util.spec_from_file_location("ppm_sampling_standalone", PPM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(family, steps_list):
    model, clip, vae, lf = load(family)
    ppm = load_ppm()
    cfg = FAMILIES[family]["cfg"]
    res = 768
    hw = res // lf.spacial_downscale_ratio
    ch = lf.latent_channels
    latent = torch.zeros([1, ch, 1, hw, hw]) if FAMILIES[family].get("temporal") else torch.zeros([1, ch, hw, hw])
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    def render(steps, sampler, label, mmodel=None):
        torch.manual_seed(SEED)
        noise = comfy.sample.prepare_noise(latent, SEED)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler="simple", denoise=1.0, model_options=model.model_options)
        with torch.no_grad():
            s = comfy.samplers.sample(
                model, noise, pos, neg, cfg, model.load_device, sampler, ks.sigmas,
                model_options=model.model_options, latent_image=latent, disable_pbar=True, seed=SEED)
            img = vae.decode(s.to(model.load_device))
        t = to_hwc(img)
        Image.fromarray((t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            os.path.join(OUT, f"cmp_{family}_{label}_{steps}.png"))
        return t

    ref = render(40, comfy.samplers.sampler_object("euler"), "ref")
    variants = [
        ("omni", comfy.samplers.ksampler("omni_hybrid", {})),
        ("omni_cfgpp1", comfy.samplers.ksampler("omni_hybrid", {"cfg_pp": True, "cfg_pp_strength": 1.0})),
        ("ppm_gamma", comfy.samplers.KSAMPLER(ppm.sample_euler_gamma)),
        ("ppm_gamma_cfgpp", comfy.samplers.KSAMPLER(functools.partial(ppm.sample_euler_gamma, cfg_pp=True))),
    ]
    print(f"[{family}] cfg={cfg}  steps  variant            ssim")
    for steps in steps_list:
        for label, sampler in variants:
            t = render(steps, sampler, label)
            print(f"  {steps:3d}  {label:18s} {ssim(t, ref):.4f}")


if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else "moss"
    steps = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["8", "12"])]
    main(fam, steps)
