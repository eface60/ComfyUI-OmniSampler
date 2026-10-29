"""GPU verification of the two subagent winners on a modern flow model.

  1) skip_every=2 (FSampler epsilon extrapolation): 2n steps, ~n evaluations,
     compared against the baseline at n steps (equal NFE).
  2) flow-space 2nd order with c=0 (the aflops experiment's real winner):
     same step count and NFE as the baseline.

Both are wrapped as KSAMPLERs so they go through the normal ComfyUI path.
Reference = omni (corrector none) at 60 steps, which is converged for these
flow models.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

COMFY = r"G:\AI\comfy"
WORK = r"G:\AI\agent working folders\generic"
PARENT = os.path.join(WORK)
EXP = os.path.join(WORK, "omni_experiments")
sys.path.insert(0, COMFY)
sys.path.insert(0, PARENT)
sys.path.insert(0, os.path.join(WORK, "omni_sampler"))
sys.path.insert(0, os.path.join(EXP, "skip"))
sys.path.insert(0, os.path.join(EXP, "aflops"))

import omni_sampler  # noqa: E402,F401
import omni_core  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, load, to_hwc  # noqa: E402
from omni_sampler.tools.sd_bench import ssim  # noqa: E402

OUT = os.path.join(WORK, "omni_sampler", "outputs")
PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"
SEED = 7


class WrappedModel:
    """Gives a (model, extra_args) pair the (x, sigma) call signature that
    flowspace.py and other standalone samplers expect."""

    def __init__(self, model, extra_args):
        self._m = model
        self._e = extra_args or {}

    def __call__(self, x, sigma):
        return self._m(x, sigma, **self._e)

    def __getattr__(self, name):
        return getattr(self._m, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--steps", default="10,12")
    ap.add_argument("--ref-steps", type=int, default=60)
    ap.add_argument("--res", type=int, default=768)
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

    calls = {"n": 0}

    def counted(model_obj):
        """Model wrapper that counts evaluations."""

        class C:
            def __init__(self, m):
                self._m = m

            def __call__(self, x, sigma, **kw):
                calls["n"] += 1
                return self._m(x, sigma, **kw)

            def __getattr__(self, name):
                return getattr(self._m, name)

        return C(model_obj)

    def render(steps, sampler_fn, label):
        calls["n"] = 0
        sampler = comfy.samplers.KSAMPLER(sampler_fn)
        torch.manual_seed(SEED)
        noise = comfy.sample.prepare_noise(latent, SEED)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler="simple", denoise=1.0, model_options=model.model_options)
        ks.model = counted(model)          # count real evaluations
        with torch.no_grad():
            s = comfy.samplers.sample(
                model, noise, pos, neg, cfg_scale, model.load_device, sampler, ks.sigmas,
                model_options=model.model_options, latent_image=latent, disable_pbar=True, seed=SEED)
            img = vae.decode(s.to(model.load_device))
        t = to_hwc(img)
        Image.fromarray((t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            os.path.join(OUT, f"win_{args.family}_{label}.png"))
        return t, calls["n"]

    def baseline(m, x, s, **kw):
        return omni_core.sample_omni(m, x, s, **kw)

    def skip2(m, x, s, **kw):
        return omni_core_skip.sample_omni(m, x, s, skip_every=2, skip_order=2, **kw)

    def flow_ab2(m, x, s, extra_args=None, callback=None, disable=None, **kw):
        wm = WrappedModel(m, extra_args)
        return flowspace.sample_flow_space(wm, x, s, mode="aflops", c_max=0.0,
                                           corrector=kw.get("corrector", "auto"))

    ref, ref_nfe = render(args.ref_steps, baseline, f"ref{args.ref_steps}")
    print(f"[{args.family}] ref {args.ref_steps} steps, {ref_nfe} evals")

    print(f"{'steps':>5s} {'variant':12s} {'ssim':>7s} {'evals':>6s}")
    for n in [int(v) for v in args.steps.split(",")]:
        for label, fn, st in (("omni", baseline, n),
                              ("skip2", skip2, 2 * n),
                              ("flow_ab2", flow_ab2, n)):
            t, nfe = render(st, fn, f"{label}_{st}")
            print(f"{st:5d} {label:12s} {ssim(t, ref):7.4f} {nfe:6d}")


if __name__ == "__main__":
    main()


