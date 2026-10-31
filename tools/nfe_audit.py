"""Audit the real number of model evaluations and wall time per sampler.

The node only *estimates* NFE from the step count; this counts the actual calls
to comfy.samplers.calc_cond_batch, which is on the path of every conditional /
unconditional model evaluation.
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

import omni_core  # noqa: E402
import omni_sampler  # noqa: E402,F401
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import latent_preview  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, load  # noqa: E402

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"

_calls = {"n": 0}
_orig = comfy.samplers.calc_cond_batch


def _counted(*a, **kw):
    _calls["n"] += 1
    return _orig(*a, **kw)


comfy.samplers.calc_cond_batch = _counted


def main(family="moss", res=768, steps=8):
    model, clip, vae, lf = load(family)
    cfg = FAMILIES[family]["cfg"]
    flow = omni_core.is_flow_model(model.get_model_object("model_sampling"))
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
        _calls["n"] = 0
        t0 = time.time()
        with torch.no_grad():
            comfy.samplers.sample(model, noise, pos, neg, cfg, model.load_device, sampler, ks.sigmas,
                                  model_options=model.model_options, latent_image=latent,
                                  callback=cb, disable_pbar=True, seed=7)
        dt = time.time() - t0
        print(f"  {label:34s} calls={_calls['n']:3d}  {dt:6.2f}s  ({dt / max(_calls['n'], 1):.2f}s/call)")

    euler = comfy.samplers.sampler_object("euler")
    omni = comfy.samplers.KSAMPLER(lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, **kw))
    omni_nocorr = comfy.samplers.KSAMPLER(
        lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, corrector="none", **kw))
    omni_eta = comfy.samplers.KSAMPLER(lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, eta=0.3, **kw))
    omni_cfgpp = comfy.samplers.KSAMPLER(
        lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, cfg_pp=True, cfg_pp_strength=1.0, **kw))
    omni_skip = comfy.samplers.KSAMPLER(
        lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, skip_every=2, **kw))

    print(f"[{family}] {res}px {steps} steps cfg={cfg} flow={flow}")
    run("euler (stock)", euler)
    run("omni default", omni)
    run("omni default + previews", omni, preview=True)
    run("omni corrector=none", omni_nocorr)
    run("omni eta=0.3", omni_eta)
    run("omni cfg_pp=1.0", omni_cfgpp)
    run("omni skip_every=2", omni_skip)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "moss")
