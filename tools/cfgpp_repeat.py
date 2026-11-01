"""Is CFG++ really slower at cfg > 1? Interleaved repeats, medians.

Single-shot timings on this machine varied by 70% between identical forwards, and
MOSS and Krea2 disagreed in sign, so the arms are run alternately several times
and compared by median. Forward counts are recorded every run to prove the two
arms do the same amount of network work.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
WORK = os.environ.get(
    "OMNI_WORK", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, COMFY)
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "omni_sampler"))

import omni_core  # noqa: E402
import omni_sampler  # noqa: E402,F401
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
from comfy.model_base import BaseModel  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, load  # noqa: E402

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"

_stats = {"fwd": 0, "batch": 0}
_orig_apply = BaseModel.apply_model


def _apply(self, x, timestep, **kw):
    _stats["fwd"] += 1
    _stats["batch"] += int(x.shape[0])
    return _orig_apply(self, x, timestep, **kw)


BaseModel.apply_model = _apply


def main(family="krea2", res=768, steps=8, cfg_scale=5.0, reps=3):
    model, clip, vae, lf = load(family)
    hw = res // lf.spacial_downscale_ratio
    latent = (torch.zeros([1, lf.latent_channels, 1, hw, hw])
              if getattr(lf, "latent_dimensions", 2) == 3 else
              torch.zeros([1, lf.latent_channels, hw, hw]))
    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    def make(**kw):
        return comfy.samplers.KSAMPLER(
            lambda m, x, s, **k: omni_core.sample_omni(m, x, s, corrector="none", **kw, **k))

    arms = {"plain": make(), "cfg_pp": make(cfg_pp=True, cfg_pp_strength=1.0)}

    def run(sampler):
        noise = comfy.sample.prepare_noise(latent, 7)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler="simple", denoise=1.0, model_options=model.model_options)
        _stats.update(fwd=0, batch=0)
        t0 = time.perf_counter()
        with torch.no_grad():
            comfy.samplers.sample(model, noise, pos, neg, cfg_scale, model.load_device, sampler,
                                  ks.sigmas, model_options=model.model_options, latent_image=latent,
                                  disable_pbar=True, seed=7)
        dt = time.perf_counter() - t0
        return dt, _stats["fwd"], _stats["batch"]

    print(f"[{family}] {res}px {steps} steps cfg={cfg_scale}, corrector=none, {reps} interleaved reps")
    run(arms["plain"])                      # warmup, discarded
    times = {"plain": [], "cfg_pp": []}
    counts = {}
    for _ in range(reps):
        for name in ("plain", "cfg_pp"):
            dt, fwd, batch = run(arms[name])
            times[name].append(dt)
            counts[name] = (fwd, batch)
    for name in ("plain", "cfg_pp"):
        med = statistics.median(times[name])
        print(f"  {name:7s} median={med:6.2f}s  runs={[round(t, 2) for t in times[name]]}  "
              f"forwards={counts[name][0]} batch_sum={counts[name][1]}")
    ratio = statistics.median(times["cfg_pp"]) / statistics.median(times["plain"])
    print(f"  cfg_pp / plain = {ratio:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "krea2")
