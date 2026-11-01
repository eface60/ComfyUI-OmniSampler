"""Real-model A/B for omni's two spatial controls.

Runs the same prompt, seed and schedule with only the named control changed, and
reports (a) the control's own reachability counters from `omni_core.LAST_STATS`
and (b) the latent and decoded-image distance to the base arm.  A second base arm
is rendered as a reproducibility control: the model's own run-to-run drift is the
floor below which no difference can be attributed to the knob.

Usage:
    python tools/spatial_ab.py krea2 --steps 12 --res 768
    python tools/spatial_ab.py moss  --steps 12 --res 1024 --arms base,base2,pxo,cfgm

Outputs go to `outputs/spatial_<family>_<steps>_<arm>.png` and `.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PARENT = os.path.dirname(REPO)
sys.path.insert(0, COMFY)
sys.path.insert(0, PARENT)

import comfy.model_management  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.utils  # noqa: E402
import omni_sampler  # noqa: E402,F401
from omni_sampler import omni_core  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, PROMPT, NEG, load, to_hwc  # noqa: E402
from omni_sampler.tools.sd_bench import ssim  # noqa: E402

OUT = os.path.join(PARENT, "outputs")
SEED = 7

ARMS = {
    "base": {},
    "base2": {},
    "pxo": {"px_order": True},
    "pxo_loose": {"px_order": True, "px_order_tol": 0.2},
    "cfgm": {"cfg_map": True},
    "both": {"px_order": True, "cfg_map": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--arms", default="base,base2,pxo,cfgm")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    model, clip, vae, lf = load(args.family)
    cfg_scale = FAMILIES[args.family]["cfg"]
    res = args.res or FAMILIES[args.family]["res"]
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    hw = res // lf.spacial_downscale_ratio
    if getattr(lf, "latent_dimensions", 2) == 3:
        latent = torch.zeros([1, lf.latent_channels, 1, hw, hw])
    else:
        latent = torch.zeros([1, lf.latent_channels, hw, hw])
    it = comfy.model_management.intermediate_device()

    ks = comfy.samplers.KSampler(model, steps=args.steps, device=model.load_device,
                                 sampler="euler", scheduler=args.scheduler, denoise=1.0,
                                 model_options=model.model_options)
    tag = args.tag or f"{args.family}_{args.steps}s_{args.scheduler}"
    results = {}
    for arm in args.arms.split(","):
        opts = dict(omni_core.DEFAULTS)
        opts.update(ARMS[arm])
        torch.manual_seed(SEED)
        noise = comfy.sample.prepare_noise(latent, SEED)
        sampler = comfy.samplers.KSAMPLER(
            lambda m, x, sig, extra_args=None, callback=None, disable=None, **kw:
                omni_core.sample_omni(m, x, sig, extra_args=extra_args,
                                      callback=callback, disable=disable, **opts))
        t0 = time.time()
        with torch.no_grad():
            s = comfy.samplers.sample(
                model, noise, pos, neg, cfg_scale, model.load_device, sampler, ks.sigmas,
                model_options=model.model_options, latent_image=latent,
                disable_pbar=True, seed=SEED)
            img = vae.decode(s.to(model.load_device))
        dt = time.time() - t0
        out = to_hwc(img).to(it)
        Image.fromarray((out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
            os.path.join(OUT, f"spatial_{tag}_{arm}.png"))
        stats = dict(omni_core.LAST_STATS)
        results[arm] = {"latent": s.detach().cpu(), "image": out.detach().cpu(),
                        "stats": stats, "sec": round(dt, 2)}
        print(f"[{arm}] {dt:5.1f}s  {json.dumps({k: v for k, v in stats.items() if k.startswith(('px_', 'cfg_'))})}",
              flush=True)

    base = results["base"]
    print("\narm        latent_rel   image_rms   ssim_to_base   reach")
    for arm, r in results.items():
        if arm == "base":
            continue
        lr = float((r["latent"] - base["latent"]).norm() / base["latent"].norm().clamp_min(1e-12))
        ir = float((r["image"] - base["image"]).pow(2).mean().sqrt())
        ss = float(ssim(r["image"], base["image"]))
        print(f"{arm:10s} {lr:10.4e} {ir:11.5f} {ss:14.5f}   {r['stats']}")
        r["latent_rel"] = lr
        r["image_rms"] = ir
        r["ssim"] = ss
    if "base2" in results:
        print("\nNEGATIVE CONTROL base vs base2: latent_rel %.3e, image_rms %.5f" % (
            results["base2"]["latent_rel"], results["base2"]["image_rms"]))
        print("Any arm whose distance is not clearly above this floor is NOT RESOLVED by this run.")

    with open(os.path.join(OUT, f"spatial_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump({a: {k: (v if k != "latent" and k != "image" else None)
                       for k, v in r.items()} for a, r in results.items()}, fh, indent=1)
    torch.save({a: r["latent"] for a, r in results.items()},
               os.path.join(OUT, f"spatial_{tag}_latents.pt"))


if __name__ == "__main__":
    main()
