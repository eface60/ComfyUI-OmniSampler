"""Does CFG++ run the unconditional pass twice at cfg > 1?

Counts real network forwards (`BaseModel.apply_model`) and their batch width, so
a doubled unconditional shows up as extra forwards or a wider batch, not just as
wall time. Runs plain vs cfg_pp at cfg=1 and at cfg>1 on a real flow model.
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
from comfy.model_base import BaseModel  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, load  # noqa: E402

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"

_stats = {"fwd": 0, "batch": 0, "t": 0.0}
_orig_apply = BaseModel.apply_model


def _apply(self, x, timestep, **kw):
    t0 = time.perf_counter()
    out = _orig_apply(self, x, timestep, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _stats["t"] += time.perf_counter() - t0
    _stats["fwd"] += 1
    _stats["batch"] += int(x.shape[0])
    return out


BaseModel.apply_model = _apply


def main(family="moss", res=768, steps=8):
    model, clip, vae, lf = load(family)
    cfg = FAMILIES[family]["cfg"]
    hw = res // lf.spacial_downscale_ratio
    latent = (torch.zeros([1, lf.latent_channels, 1, hw, hw])
              if getattr(lf, "latent_dimensions", 2) == 3 else
              torch.zeros([1, lf.latent_channels, hw, hw]))
    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    def run(label, cfg_scale, sampler):
        noise = comfy.sample.prepare_noise(latent, 7)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler="simple", denoise=1.0, model_options=model.model_options)
        _stats.update(fwd=0, batch=0, t=0.0)
        t0 = time.perf_counter()
        with torch.no_grad():
            comfy.samplers.sample(model, noise, pos, neg, cfg_scale, model.load_device, sampler,
                                  ks.sigmas, model_options=model.model_options, latent_image=latent,
                                  disable_pbar=True, seed=7)
        total = time.perf_counter() - t0
        print(f"  {label:30s} total={total:6.2f}s model={_stats['t']:6.2f}s "
              f"forwards={_stats['fwd']:3d} batch_sum={_stats['batch']:3d}")

    mk = lambda **kw: comfy.samplers.KSAMPLER(lambda m, x, s, **k: omni_core.sample_omni(m, x, s, corrector="none", **kw, **k))
    plain = mk()
    cpp = mk(cfg_pp=True, cfg_pp_strength=1.0)

    print(f"[{family}] {res}px {steps} steps, model cfg={cfg}, corrector=none")
    run("plain warmup", cfg, plain)
    run(f"plain cfg={cfg}", cfg, plain)
    run(f"cfg_pp cfg={cfg}", cfg, cpp)
    run("plain cfg=5", 5.0, plain)
    run("cfg_pp cfg=5", 5.0, cpp)
    if cfg != 1.0:
        run("plain cfg=1", 1.0, plain)
        run("cfg_pp cfg=1", 1.0, cpp)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "moss")
