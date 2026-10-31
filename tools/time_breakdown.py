"""Where does the wall time go: model evaluations vs sampler overhead."""

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
from omni_sampler.tools.modern_bench import FAMILIES, load  # noqa: E402

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"

_t = {"model": 0.0, "calls": 0}
_orig = comfy.samplers.calc_cond_batch


def _timed(*a, **kw):
    t0 = time.perf_counter()
    out = _orig(*a, **kw)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    _t["model"] += time.perf_counter() - t0
    _t["calls"] += 1
    return out


comfy.samplers.calc_cond_batch = _timed


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

    def run(label, sampler):
        noise = comfy.sample.prepare_noise(latent, 7)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler="simple", denoise=1.0, model_options=model.model_options)
        _t["model"], _t["calls"] = 0.0, 0
        t0 = time.perf_counter()
        with torch.no_grad():
            comfy.samplers.sample(model, noise, pos, neg, cfg, model.load_device, sampler, ks.sigmas,
                                  model_options=model.model_options, latent_image=latent,
                                  disable_pbar=True, seed=7)
        total = time.perf_counter() - t0
        model_t = _t["model"]
        print(f"  {label:30s} total={total:6.2f}s  model={model_t:6.2f}s ({_t['calls']} calls)  "
              f"overhead={total - model_t:6.2f}s")

    euler = comfy.samplers.sampler_object("euler")
    mk = lambda **kw: comfy.samplers.KSAMPLER(lambda m, x, s, **k: omni_core.sample_omni(m, x, s, **kw, **k))

    print(f"[{family}] {res}px {steps} steps cfg={cfg}")
    run("euler (stock)", euler)
    run("omni default", mk())
    run("omni corrector=none", mk(corrector="none"))
    run("omni eta=0.3 brownian", mk(eta=0.3))
    run("omni eta=0.3 white", mk(eta=0.3, noise_mode="white"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "krea2")
