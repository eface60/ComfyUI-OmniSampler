"""Cost of the CFG++ branch: it needs the unconditional prediction.

At cfg == 1 ComfyUI's usual optimisation skips the unconditional pass entirely
(only `cond` is evaluated). CFG++ cannot work without `u`, so the sampler sets
`disable_cfg1_optimization=True`, which doubles the model batch. At cfg > 1 the
unconditional pass happens anyway, so CFG++ should be free there. This script
measures both.
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
from omni_sampler.tools.modern_bench import FAMILIES, load  # noqa: E402

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"

_calls = {"n": 0, "elems": 0}
_orig = comfy.samplers.calc_cond_batch


def _counted(model, conds, x, timestep, model_options):
    _calls["n"] += 1
    _calls["elems"] += x.shape[0]
    return _orig(model, conds, x, timestep, model_options)


comfy.samplers.calc_cond_batch = _counted


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

    def run(label, cfg_scale, sampler):
        noise = comfy.sample.prepare_noise(latent, 7)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler="simple", denoise=1.0, model_options=model.model_options)
        _calls["n"], _calls["elems"] = 0, 0
        t0 = time.perf_counter()
        with torch.no_grad():
            comfy.samplers.sample(model, noise, pos, neg, cfg_scale, model.load_device, sampler,
                                  ks.sigmas, model_options=model.model_options, latent_image=latent,
                                  disable_pbar=True, seed=7)
        print(f"  {label:34s} {time.perf_counter() - t0:6.2f}s  calls={_calls['n']:2d} "
              f"batch_elems={_calls['elems']:3d}")

    mk = lambda **kw: comfy.samplers.KSAMPLER(lambda m, x, s, **k: omni_core.sample_omni(m, x, s, **kw, **k))
    plain = mk(corrector="none")
    cpp = mk(corrector="none", cfg_pp=True, cfg_pp_strength=1.0)

    print(f"[{family}] {res}px {steps} steps, model cfg={cfg}")
    run("corrector=none (warmup)", cfg, plain)
    run("corrector=none", cfg, plain)
    run("cfg_pp=1.0", cfg, cpp)
    if cfg == 1.0:
        run("corrector=none at cfg=3.5", 3.5, plain)
        run("cfg_pp=1.0 at cfg=3.5", 3.5, cpp)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "krea2")
