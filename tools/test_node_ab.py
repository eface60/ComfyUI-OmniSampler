"""End-to-end test of the full A/B node on a real flow model (GPU).

Renders the same seed/prompt with three presets through OmniKSampler, reports
wall time and how far B/C land from A, and asserts the outputs are well formed.
"""

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

import omni_sampler  # noqa: E402,F401
import comfy.sample  # noqa: E402
from omni_sampler.omni_nodes import OmniKSampler  # noqa: E402
from omni_sampler.tools.modern_bench import load  # noqa: E402

PROMPT = "a red fox in a snowy forest"
NEG = "blurry, low quality"


def main(family="moss", res=512, steps=8):
    model, clip, vae, lf = load(family)
    hw = res // lf.spacial_downscale_ratio
    latent = {"samples": torch.zeros([1, lf.latent_channels, hw, hw])}
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    node = OmniKSampler()
    base = dict(
        model=model, positive=pos, negative=neg, latent_image=latent, seed=7, steps=steps,
        cfg=3.5, scheduler="simple", denoise=1.0, order=2, integrator="exp", corrector="first",
        x0_average="ab2", min_step_ratio=0.5, lower_order_to_end=1, eta=0.0, s_noise=1.0,
        cfg_pp=False, cfg_pp_strength=0.5, s_churn=0.0, churn_mode="karras",
        noise_mode="brownian", skip_every=1, skip_order=2, final_extrapolate=0.0,
    )
    outs = {}
    for v in ("A: shipped (v1.3.1)", "B: no corrector", "C: skip every 2nd eval",
              "D: ppm gamma churn"):
        t0 = time.time()
        out, summary = node.sample(variant=v, **base)
        dt = time.time() - t0
        s = out["samples"]
        assert s.shape == latent["samples"].shape, (s.shape, latent["samples"].shape)
        assert torch.isfinite(s).all()
        outs[v] = s.detach().float()
        print(f"  {v:26s} {dt:5.2f}s  {summary}")
    a = outs["A: shipped (v1.3.1)"]
    rel = lambda x: float((x - a).norm() / a.norm())
    for v in outs:
        if v.startswith("A"):
            continue
        print(f"  {v:26s} rel_l2 vs A = {rel(outs[v]):.3e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "moss")
