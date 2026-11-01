"""Latent-space verification on the modern flow models.

The sampler produces latents; the VAE is a lossy, spatially mixing transform, so
comparing decoded pixels confounds decoder behaviour with integration accuracy.
Primary metric here is relative L2 between the final latent and a converged
reference latent. Decoded previews are written for human inspection only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from PIL import Image

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
WORK = os.environ.get(
    "OMNI_WORK", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, COMFY)
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "omni_sampler"))
sys.path.insert(0, os.path.join(WORK, "omni_experiments", "skip"))
sys.path.insert(0, os.path.join(WORK, "omni_experiments", "aflops"))

import omni_core  # noqa: E402
import omni_sampler  # noqa: E402,F401
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, load, to_hwc  # noqa: E402

OUT = os.path.join(WORK, "omni_sampler", "outputs")
PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"
SEED = 7


def rel_l2(a, b):
    a = a.detach().float()
    b = b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--steps", default="6,8,10")
    ap.add_argument("--ref-steps", type=int, default=60)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--variants", default="omni,euler,corr1")
    ap.add_argument("--save", default="")   # comma list of labels whose previews to save
    args = ap.parse_args()

    import omni_core_skip
    import flowspace

    model, clip, vae, lf = load(args.family)
    cfg_scale = FAMILIES[args.family]["cfg"]
    hw = args.res // lf.spacial_downscale_ratio
    latent = (torch.zeros([1, lf.latent_channels, 1, hw, hw])
              if getattr(lf, "latent_dimensions", 2) == 3 else
              torch.zeros([1, lf.latent_channels, hw, hw]))
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    class Wrapped:
        def __init__(self, m, extra):
            self._m, self._e = m, extra or {}

        def __call__(self, x, sigma):
            return self._m(x, sigma, **self._e)

        def __getattr__(self, name):
            return getattr(self._m, name)

    def flow_ab2(m, x, s, extra_args=None, callback=None, disable=None, **kw):
        return flowspace.sample_flow_space(Wrapped(m, extra_args), x, s, mode="aflops", c_max=0.0)

    VAR = {
        "omni": (lambda: comfy.samplers.KSAMPLER(
            lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, **kw)), 0, 1),
        "corr1": (lambda: comfy.samplers.KSAMPLER(
            lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, corrector="first", **kw)), 0, 1),
        "nocorr": (lambda: comfy.samplers.KSAMPLER(
            lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, corrector="none", **kw)), 0, 1),
        "cfgpp": (lambda: comfy.samplers.KSAMPLER(
            lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, cfg_pp=True, cfg_pp_strength=1.0, **kw)), 0, 1),
        "euler": (lambda: comfy.samplers.sampler_object("euler"), 0, 1),
        "dpmpp_2m": (lambda: comfy.samplers.sampler_object("dpmpp_2m"), 0, 1),
        # skip: 2x steps, ~1x evaluations
        "skip2": (lambda: comfy.samplers.KSAMPLER(
            lambda m, x, s, **kw: omni_core_skip.sample_omni(m, x, s, skip_every=2, skip_order=2, **kw)), 0, 2),
        "flowab2": (lambda: comfy.samplers.KSAMPLER(flow_ab2), 0, 1),
    }

    def render(steps, label):
        mk, _, mult = VAR[label]
        n = steps * mult
        torch.manual_seed(SEED)
        noise = comfy.sample.prepare_noise(latent, SEED)
        ks = comfy.samplers.KSampler(model, steps=n, device=model.load_device, sampler="euler",
                                     scheduler="simple", denoise=1.0, model_options=model.model_options)
        with torch.no_grad():
            s = comfy.samplers.sample(
                model, noise, pos, neg, cfg_scale, model.load_device, mk(), ks.sigmas,
                model_options=model.model_options, latent_image=latent, disable_pbar=True, seed=SEED)
        return s.detach().cpu(), n

    def preview(samples, name):
        with torch.no_grad():
            img = vae.decode(samples.to(model.load_device))
        t = to_hwc(img)
        arr = (t[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(arr).save(os.path.join(OUT, f"{name}.png"))
        assert arr.shape[0] == arr.shape[1] > 256, f"bad preview shape {arr.shape}"

    ref, ref_n = render(args.ref_steps, "nocorr")
    print(f"[{args.family}] ref = omni/nocorr @{ref_n} steps  latent {tuple(ref.shape)}")

    rows = []
    for n in [int(v) for v in args.steps.split(",")]:
        for label in args.variants.split(","):
            s, steps = render(n, label)
            e = rel_l2(s, ref)
            rows.append(dict(steps=steps, variant=label, rel_l2=e))
            print(f"  {steps:3d} {label:10s} rel_l2={e:.4e}")
            if label in args.save.split(","):
                preview(s, f"lat_{args.family}_{label}_{steps}")

    with open(os.path.join(OUT, f"latent_{args.family}.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
