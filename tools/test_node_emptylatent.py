"""Regression: the A/B node must accept a plain empty latent like the stock
KSampler does.

The user's crash was `mat1 and mat2 shapes cannot be multiplied (8960x16 and
64x6144)` inside krea2's patch embedding, because the node passed an unadapted
4-channel 4-D empty latent straight to the model. nodes.py::common_ksampler
first calls comfy.sample.fix_empty_latent_channels; the node now does too.
"""

from __future__ import annotations

import os
import sys

import torch

COMFY = r"G:\AI\comfy"
WORK = r"G:\AI\agent working folders\generic"
sys.path.insert(0, COMFY)
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "omni_sampler"))

import omni_sampler  # noqa: E402,F401
import comfy.sample  # noqa: E402
import comfy.utils  # noqa: E402
from omni_sampler.omni_nodes import OmniKSampler  # noqa: E402
from omni_sampler.tools.modern_bench import load  # noqa: E402


def main(family="krea2", res=512, steps=6):
    model, clip, vae, lf = load(family)
    pos = clip.encode_from_tokens_scheduled(clip.tokenize("a red fox in a snowy forest"))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize("blurry, low quality"))

    # a stock EmptyLatentImage-style latent: 4 channels, 4-D, all zeros
    raw = torch.zeros([1, 4, res // 8, res // 8])
    fixed = comfy.sample.fix_empty_latent_channels(model, raw)
    print(f"[{family}] input latent {tuple(raw.shape)} -> after fix {tuple(fixed.shape)}"
          f" (latent_format channels={lf.latent_channels}, dims={getattr(lf, 'latent_dimensions', 2)})")

    node = OmniKSampler()
    base = dict(
        model=model, positive=pos, negative=neg, seed=7, steps=steps, cfg=1.0,
        scheduler="simple", denoise=1.0, order=2, integrator="exp", corrector="first",
        x0_average="ab2", min_step_ratio=0.5, lower_order_to_end=1, eta=0.0, s_noise=1.0,
        cfg_pp=False, cfg_pp_strength=0.5, s_churn=0.0, churn_mode="karras",
        noise_mode="brownian", skip_every=1, skip_order=2, final_extrapolate=0.0,
    )
    out, summary = node.sample(variant="A: shipped (v1.3.1)", latent_image={"samples": raw}, **base)
    s = out["samples"]
    print("  OK", tuple(s.shape), "finite:", bool(torch.isfinite(s).all()))
    print("  returned keys:", sorted(out.keys()))
    assert s.shape == tuple(fixed.shape), (s.shape, fixed.shape)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "krea2")
