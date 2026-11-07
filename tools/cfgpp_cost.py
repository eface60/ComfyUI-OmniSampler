"""Cost of needing the unconditional prediction: CFG++ and the gap features.

At cfg == 1 ComfyUI's usual optimisation skips the unconditional pass entirely
(only `cond` is evaluated). CFG++ cannot work without `u`, and neither can
guidance-gap shaping (`gap_lf` / `gap_cap` / `gap_parallel` /
`gap_parallel_local` / `gap_stats`), because the thing they reshape is
`cond - uncond`. Any of them therefore sets `disable_cfg1_optimization=True`,
which doubles the model batch. At cfg > 1 the unconditional pass happens anyway,
so they should be free there, and free on top of CFG++.

That is the answer to "why is gap_local slower on my distilled cfg-1 model": it
is the forced second pass, not the per-pixel maths. `batch` sums the batch of
every model forward, so the doubling is visible even when wall time is noisy.
Neither `len(conds)` nor `x.shape[0]` is a usable signal here: ComfyUI passes
both conds every step and concatenates them into one batch inside
`calc_cond_batch`.
"""

from __future__ import annotations

import os
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
from omni_sampler.tools.modern_bench import FAMILIES, load  # noqa: E402

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"

_calls = {"n": 0, "batch": 0}
_orig = comfy.samplers.calc_cond_batch


def _counted(model, conds, x, timestep, model_options):
    _calls["n"] += 1
    return _orig(model, conds, x, timestep, model_options)


comfy.samplers.calc_cond_batch = _counted


def install_model_counter(model):
    """Count the batch the diffusion model is actually asked to run.

    Patching ``apply_model`` on the loaded class (not on ``BaseModel``) also
    catches a family that overrides it.  The sum is the number of predictions:
    one per step at cfg = 1 normally, two when the unconditional pass is forced.
    """
    cls = type(getattr(model, "model", model))
    orig = cls.apply_model

    def counted(self, x, *a, **kw):
        _calls["batch"] += x.shape[0]
        return orig(self, x, *a, **kw)

    cls.apply_model = counted


def main(family="krea2", res=1024, steps=10):
    model, clip, vae, lf = load(family)
    cfg = FAMILIES[family]["cfg"]
    hw = res // lf.spacial_downscale_ratio
    latent = (torch.zeros([1, lf.latent_channels, 1, hw, hw])
              if getattr(lf, "latent_dimensions", 2) == 3 else
              torch.zeros([1, lf.latent_channels, hw, hw]))
    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    install_model_counter(model)
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    def run(label, cfg_scale, sampler):
        noise = comfy.sample.prepare_noise(latent, 7)
        ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                     scheduler="simple", denoise=1.0, model_options=model.model_options)
        _calls["n"], _calls["batch"] = 0, 0
        t0 = time.perf_counter()
        with torch.no_grad():
            comfy.samplers.sample(model, noise, pos, neg, cfg_scale, model.load_device, sampler,
                                  ks.sigmas, model_options=model.model_options, latent_image=latent,
                                  disable_pbar=True, seed=7)
        print(f"  {label:34s} {time.perf_counter() - t0:6.2f}s  calls={_calls['n']:2d} "
              f"batch={_calls['batch']:3d}")

    mk = lambda **kw: comfy.samplers.KSAMPLER(lambda m, x, s, **k: omni_core.sample_omni(m, x, s, **kw, **k))
    plain = mk(corrector="none")
    cpp = mk(corrector="none", cfg_pp=True, cfg_pp_strength=1.0)
    gap = mk(corrector="none", gap_parallel_local=0.6, gap_local_start=0.3)
    cpp_gap = mk(corrector="none", cfg_pp=True, cfg_pp_strength=1.0,
                 gap_parallel_local=0.6, gap_local_start=0.3)

    print(f"[{family}] {res}px {steps} steps, model cfg={cfg}")
    run("corrector=none (warmup)", cfg, plain)
    run("corrector=none", cfg, plain)
    run("gap_local 0.6", cfg, gap)
    run("cfg_pp=1.0", cfg, cpp)
    run("cfg_pp=1.0 + gap_local 0.6", cfg, cpp_gap)
    if cfg == 1.0:
        run("corrector=none at cfg=3.5", 3.5, plain)
        run("gap_local 0.6 at cfg=3.5", 3.5, gap)
        run("cfg_pp=1.0 at cfg=3.5", 3.5, cpp)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "krea2")
