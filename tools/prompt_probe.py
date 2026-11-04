"""Can sampler settings be predicted from the prompt?  Two feasibility measurements.

**Exp 1 -- across-prompt spread of the guidance signal (a 3-step lookahead).**
For 8 prompts and 2 seeds, run the *first three steps* of the 12-step schedule
(the real trajectory, same noise, truncated sigma ladder) and record the gap
statistics.  If the across-prompt spread of `par_share` / `lf_share` / `rms_ratio`
is small compared with the across-seed spread, there is nothing for a prompt
predictor to read.

**Exp 2 -- prompt vs seed variance of the finished image.**  6 prompts x 3 seeds,
full 12-step run, fixed settings.  For each image metric, the between-prompt
variance and the within-prompt (seed) variance, and their ratio.  If the seed
dominates, no prompt-level predictor can beat a good default -- that is the
ceiling of the whole idea.

    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\prompt_probe.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)
sys.path.insert(0, COMFY)
sys.path.insert(0, PARENT)
sys.path.insert(0, REPO)

import omni_sampler  # noqa: E402,F401
from omni_sampler import omni_core  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, load, to_hwc  # noqa: E402
from omni_sampler.tools.gap_shape_bench import image_metrics  # noqa: E402

OUT = os.path.join(REPO, "outputs", "prompt_probe")

PROMPTS = [
    ("fox", "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"),
    ("apple", "a single ripe red apple on a white table, studio product photo, soft shadow"),
    ("market", "a crowded medieval market square with dozens of merchants, banners and stalls, wide shot"),
    ("portrait", "close-up portrait of an old fisherman, freckles, weathered skin, 85mm photo"),
    ("neon", "a neon-lit cyberpunk alley at night, rain, reflective puddles, cinematic"),
    ("harbor", "an oil painting of a harbor at dusk, thick impasto brush strokes"),
    ("chess", "two cats playing chess at a wooden table, painting"),
    ("text", "text reading OMNI painted on a concrete wall, photo"),
]
NEG = "blurry, low quality, watermark, text"
SETTINGS = {"gap_stats": True, "cfg_pp": True, "cfg_pp_strength": 1.0, "corrector": "none"}


def latent_for(lf, w, h):
    ratio = int(lf.spacial_downscale_ratio)
    hw, ww = h // ratio, w // ratio
    ch = int(lf.latent_channels)
    if getattr(lf, "latent_dimensions", 2) == 3:
        return torch.zeros([1, ch, 1, hw, ww])
    return torch.zeros([1, ch, hw, ww])


def run(model, clip, vae, lf, prompt, seed, steps, lookahead=None, decode=False):
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    latent = latent_for(lf, *RES)
    torch.manual_seed(seed)
    noise = comfy.sample.prepare_noise(latent, seed)
    sampler = comfy.samplers.ksampler("omni_hybrid", dict(SETTINGS))
    ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="euler",
                                 scheduler="simple", denoise=1.0, model_options=model.model_options)
    sigmas = ks.sigmas if lookahead is None else ks.sigmas[: lookahead + 1]
    t0 = time.time()
    with torch.no_grad():
        s = comfy.samplers.sample(
            model, noise, pos, neg, 1.0, model.load_device, sampler, sigmas,
            model_options=model.model_options, latent_image=latent,
            disable_pbar=True, seed=seed)
        st = dict(omni_core.LAST_STATS)
        img = None
        if decode:
            try:
                img = vae.decode(s.to(model.load_device))
            except Exception as exc:  # noqa: BLE001
                print(f"    (decode skipped: {type(exc).__name__})", flush=True)
    dt = time.time() - t0
    t = to_hwc(img).float() if img is not None else None
    return st, (image_metrics(t) if t is not None else {}), dt


def exp1(model, clip, vae, lf, seeds):
    print("\n=== exp 1: 3-step lookahead, %d prompts x %d seeds ===" % (len(PROMPTS), len(seeds)))
    rows = {}
    for name, prompt in PROMPTS:
        for seed in seeds:
            st, _, dt = run(model, clip, vae, lf, prompt, seed, 12, lookahead=3)
            ser = st.get("gap_series") or []
            if not ser:
                continue
            rows[(name, seed)] = {
                "par": [m.get("par_share") for _, m in ser],
                "lf": [m.get("lf_share") for _, m in ser],
                "rms": [m.get("rms_ratio") for _, m in ser],
                "p99": [m.get("p99_ratio") for _, m in ser],
                "rev": [m.get("rev", 0.0) for _, m in ser],
            }
            print(f"    {name:9s} seed={seed} {dt:5.1f}s  steps={len(ser)}", flush=True)

    def stat(key, step):
        per_prompt = {}
        for (name, seed), r in rows.items():
            per_prompt.setdefault(name, []).append(r[key][step])
        means = [statistics.mean(v) for v in per_prompt.values()]
        within = statistics.mean([statistics.pstdev(v) for v in per_prompt.values() if len(v) > 1] or [0.0])
        return means, within

    for key in ("par", "lf", "rms", "p99"):
        print(f"\n  {key}: mean across prompts at each lookahead step (seed noise in parentheses)")
        for step in range(min(len(r["par"]) for r in rows.values())):
            means, within = stat(key, step)
            spread = statistics.pstdev(means) if len(means) > 1 else 0.0
            print(f"    step {step}: prompts {statistics.mean(means):.3f} +/- {spread:.3f} "
                  f"(across prompts)   seed sd {within:.3f}   ratio {spread / within if within else float('inf'):.2f}")
    return rows


def exp2(model, clip, vae, lf, seeds, prompt_names):
    print("\n=== exp 2: prompt x seed variance of the finished image ===")
    rows = {}
    for name, prompt in PROMPTS:
        if name not in prompt_names:
            continue
        for seed in seeds:
            st, m, dt = run(model, clip, vae, lf, prompt, seed, 12, decode=True)
            rows[(name, seed)] = m
            print(f"    {name:9s} seed={seed} {dt:5.1f}s  "
                  f"sat={m.get('sat')} hf={m.get('hf')} lum={m.get('lum')}", flush=True)
    metrics = sorted({k for m in rows.values() for k in m})
    print(f"\n{'metric':6s} {'grand mean':>10s} {'prompt sd':>9s} {'seed sd':>8s} {'F':>6s}")
    out = {}
    for key in metrics:
        per_prompt = {}
        for (name, seed), m in rows.items():
            if key in m:
                per_prompt.setdefault(name, []).append(m[key])
        if len(per_prompt) < 2:
            continue
        means = [statistics.mean(v) for v in per_prompt.values()]
        within = statistics.mean([statistics.pstdev(v) for v in per_prompt.values() if len(v) > 1] or [0.0])
        between = statistics.pstdev(means)
        out[key] = {"grand": statistics.mean(means), "prompt_sd": between,
                    "seed_sd": within, "F": (between / within if within else None)}
        print(f"{key:6s} {out[key]['grand']:10.4f} {between:9.4f} {within:8.4f} "
              f"{out[key]['F'] if out[key]['F'] is not None else float('nan'):6.2f}")
    return rows, out


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="1024x1408")
    ap.add_argument("--seeds", default="7,11,23")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-exp1", action="store_true")
    ap.add_argument("--skip-exp2", action="store_true")
    args = ap.parse_args()
    RES = tuple(int(v) for v in args.res.lower().split("x"))
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)

    model, clip, vae, lf = load("krea2")
    print(f"[krea2] {RES[0]}x{RES[1]}  seeds={seeds}")
    if not args.skip_exp1:
        rows1 = exp1(model, clip, vae, lf, seeds[:2])
        json.dump({f"{k[0]}|{k[1]}": v for k, v in rows1.items()},
                  open(os.path.join(args.out, "exp1.json"), "w"), indent=1)
    if not args.skip_exp2:
        rows2, summary = exp2(model, clip, vae, lf, seeds, ["fox", "apple", "market", "portrait", "neon", "harbor"])
        json.dump({"rows": {f"{k[0]}|{k[1]}": v for k, v in rows2.items()}, "summary": summary},
                  open(os.path.join(args.out, "exp2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
