"""Real model: does guidance-gap shaping fix the high-resolution burn?

Renders the matrix from docs/EXPERIMENT_GAP_SHAPING.md on a real checkpoint at a
chosen resolution, saves the images and prints reference-free metrics plus the
per-step cond/uncond gap.

Why reference-free: a 6 MP high-step reference costs more than the whole matrix,
and the question here is not "which is closer to a reference" but "does this arm
burn, and does the gap measurement separate the resolutions".  The gap
statistics are the measurement the hypothesis rests on, so every arm that
installs the hook prints them.

    # 1.4 MP: the Krea2 turbo case that is known to be fine
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\gap_shape_bench.py --res 1024x1408

    # 6 MP: the case that burns.  Slow: cfg_pp at cfg 1 doubles the batch.
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\gap_shape_bench.py --res 2048x2944 --arms probe,cfgpp,hf,cap,hf_cap

Arms (docs/EXPERIMENT_GAP_SHAPING.md says what each one answers):
    plain    no guidance at all                   (baseline artifacts)
    probe    gap_stats only, image unchanged      (is the gap different at 6 MP?)
    cfgpp    CFG++ strength 1.0                   (the burn)
    s05      CFG++ strength 0.5                   (the artifacts)
    hf       CFG++ + gap_lf 0.05                  (tone band moved to u)
    cap      CFG++ + gap_cap 0.2                  (bounded overshoot)
    hf_cap   CFG++ + both                         (the omni_hybrid_gap preset)
    project  CFG++ + gap_parallel 0.0             (contrast component removed)
    local    CFG++ + gap_parallel_local 0.6       (contrast relieved on the hot tail)
    local_late  the same, from 30% of the steps   (keeps the early structure intact)
    cfg2     CFG++ at cfg 2                       (the cfgpp_check calibration arm)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)
sys.path.insert(0, COMFY)
sys.path.insert(0, PARENT)
sys.path.insert(0, REPO)

import omni_sampler  # noqa: E402,F401
# The package module, not the top-level one: the registered sampler closes over
# omni_sampler.omni_core, and a bare `import omni_core` would load a second
# module object whose LAST_STATS stays empty.
from omni_sampler import omni_core  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, NEG, PROMPT, load, to_hwc  # noqa: E402

OUT = os.path.join(REPO, "outputs", "gap_shaping")
SEED = 7

CFGPP = {"cfg_pp": True, "cfg_pp_strength": 1.0, "gap_stats": True}
ARMS = {
    "plain": ({}, 1.0),
    "probe": ({"gap_stats": True}, 1.0),
    "cfgpp": (dict(CFGPP), 1.0),
    "s05": ({"cfg_pp": True, "cfg_pp_strength": 0.5, "gap_stats": True}, 1.0),
    "hf": ({**CFGPP, "gap_lf": 0.05, "gap_lf_keep": 0.0}, 1.0),
    "cap": ({**CFGPP, "gap_cap": 0.2}, 1.0),
    "hf_cap": ({**CFGPP, "gap_lf": 0.05, "gap_lf_keep": 0.0, "gap_cap": 0.2}, 1.0),
    "project": ({**CFGPP, "gap_parallel": 0.0}, 1.0),
    "local": ({**CFGPP, "gap_parallel_local": 0.6}, 1.0),
    "local_late": ({**CFGPP, "gap_parallel_local": 0.6, "gap_local_start": 0.3}, 1.0),
    # The author's tuned default: 60% of the contrast everywhere, none on the
    # hot tail after 30% of the steps (the two knobs subtract, so 0.6 - 0.6 = 0).
    "author": ({**CFGPP, "gap_parallel": 0.6, "gap_parallel_local": 0.6,
                "gap_local_start": 0.3}, 1.0),
    "cfg2": (dict(CFGPP), 2.0),
}

_LAP = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)


def image_metrics(img):
    """Reference-free metrics of a decoded (1,3,H,W) image in [0,1].

    ``sat``  mean chroma (max-min over RGB) -- saturation, the burn symptom;
    ``clip`` fraction of channels at the rail -- blown highlights;
    ``lum``  mean luminance -- the exposure shift;
    ``hf``   mean |Laplacian| of luminance over the flattest half of the image,
             i.e. grain/banding in smooth areas -- the artifact proxy.
    """
    x = img[:, :3].float()                                  # (1,3,H,W)
    sat = float((x.amax(dim=1) - x.amin(dim=1)).mean())
    clip = float(((x <= 0.004) | (x >= 0.996)).float().mean())
    lum = x.mean(dim=1, keepdim=True)                       # (1,1,H,W)
    lap = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="replicate"), _LAP.to(lum.device))
    g = lum.new_zeros(lum.shape)
    g[:, :, :, 1:] += (lum[:, :, :, 1:] - lum[:, :, :, :-1]).abs()
    g[:, :, :, :-1] += (lum[:, :, :, 1:] - lum[:, :, :, :-1]).abs()
    g[:, :, 1:, :] += (lum[:, :, 1:, :] - lum[:, :, :-1, :]).abs()
    g[:, :, :-1, :] += (lum[:, :, 1:, :] - lum[:, :, :-1, :]).abs()
    flat = (g <= torch.quantile(g.reshape(-1), 0.5)).float()
    hf = float((lap.abs() * flat).sum() / flat.sum().clamp_min(1.0))
    return {"sat": round(sat, 5), "clip": round(clip, 5),
            "hf": round(hf, 6), "lum": round(float(lum.mean()), 5)}


def gap_fields(stats):
    keys = ("gap_rms_ratio_mean", "gap_lf_share_mean", "gap_par_share_mean",
            "gap_cap_mean", "gap_steps", "gap_seen", "gap_reason")
    return {k.replace("gap_", ""): stats.get(k) for k in keys}


def _fmt(v):
    return "   -  " if v is None else f"{v:5.3f}"


def _num(v, width, digits):
    """Fixed-width cell that renders a dash when a metric was not computed."""
    return f"{v:>{width}.{digits}f}" if v is not None else "-".rjust(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="krea2", choices=list(FAMILIES))
    ap.add_argument("--res", default="1024x1408", help="WxH, both multiples of 8")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-decode", action="store_true",
                    help="skip the VAE decode (latent-space statistics only); the 6 MP "
                         "decode is both slow and, on the Wan VAE, broken under tiling")
    ap.add_argument("--dry-run", action="store_true", help="build the plan, load nothing")
    args = ap.parse_args()

    w, h = (int(v) for v in args.res.lower().split("x"))
    if w % 8 or h % 8:
        raise SystemExit(f"--res {args.res}: both sides must be multiples of 8 (the VAE ratio)")
    names = [n.strip() for n in args.arms.split(",") if n.strip()]
    for n in names:
        if n not in ARMS:
            raise SystemExit(f"unknown arm {n!r}; choose from {sorted(ARMS)}")
    os.makedirs(args.out, exist_ok=True)

    print(f"[{args.family}] {w}x{h} = {w * h / 1e6:.2f} MP, {args.steps} steps, "
          f"scheduler={args.scheduler}, seed={args.seed}")
    for n in names:
        opts, cfg = ARMS[n]
        print(f"  {n:8s} cfg={cfg:g}  {opts if opts else '(no overrides)'}")
    if args.dry_run:
        return

    model, clip, vae, lf = load(args.family)
    ratio = int(lf.spacial_downscale_ratio)
    hw, ww = h // ratio, w // ratio
    ch = int(lf.latent_channels)
    if getattr(lf, "latent_dimensions", 2) == 3:
        latent = torch.zeros([1, ch, 1, hw, ww])
    else:
        latent = torch.zeros([1, ch, hw, ww])
    print(f"[{args.family}] latent {tuple(latent.shape)}  ratio={ratio}  "
          f"vram={torch.cuda.memory_allocated() / 2**30:.1f}GB")

    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    cfg_family = float(FAMILIES[args.family]["cfg"])

    def render(arm, opts, cfg):
        sampler = comfy.samplers.ksampler("omni_hybrid", opts)
        torch.manual_seed(args.seed)
        noise = comfy.sample.prepare_noise(latent, args.seed)
        ks = comfy.samplers.KSampler(model, steps=args.steps, device=model.load_device,
                                     sampler="euler", scheduler=args.scheduler,
                                     denoise=1.0, model_options=model.model_options)
        t0 = time.time()
        with torch.no_grad():
            s = comfy.samplers.sample(
                model, noise, pos, neg, cfg, model.load_device, sampler, ks.sigmas,
                model_options=model.model_options, latent_image=latent,
                disable_pbar=True, seed=args.seed)
            st = dict(omni_core.LAST_STATS)
            img = None
            if not args.no_decode:
                # A 6 MP full decode can exhaust a 24 GB card once the model is
                # resident, and this Wan VAE's tiled path is broken in this
                # ComfyUI build (process_output does an in-place op on an
                # inference tensor).  So: decode small sizes, skip big ones.
                try:
                    if w * h > 2_500_000:
                        raise RuntimeError("large decode skipped")
                    img = vae.decode(s.to(model.load_device))
                except Exception as exc:  # noqa: BLE001
                    print(f"    (decode skipped: {type(exc).__name__}: {exc})", flush=True)
        dt = time.time() - t0
        if img is None:
            return None, dt, st
        t = to_hwc(img).float()
        label = f"{args.family}_{arm}_{w}x{h}_{args.steps}"
        Image.fromarray((t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"{label}.png"))
        return t, dt, st

    header = (f"{'arm':8s} {'cfg':>4s} {'sec':>6s} {'sat':>7s} {'clip':>7s} "
              f"{'hf':>8s} {'lum':>6s} | gap rms    lf    par   cap")
    print(header)
    rows = []
    for name in names:
        opts, cfg = ARMS[name]
        cfg = float(cfg if name == "cfg2" else cfg_family)
        try:
            t, dt, st = render(name, opts, cfg)
            m = image_metrics(t) if t is not None else {}
            g = gap_fields(st)
            rows.append({"arm": name, "cfg": cfg, "sec": round(dt, 1), **m,
                         "gap": g, "gap_series": st.get("gap_series") or []})
            print(f"{name:8s} {cfg:4g} {dt:6.1f} "
                  f"{_num(m.get('sat'), 7, 4)} {_num(m.get('clip'), 7, 4)} "
                  f"{_num(m.get('hf'), 8, 5)} {_num(m.get('lum'), 6, 4)} | "
                  f"{_fmt(g.get('rms_ratio_mean'))} {_fmt(g.get('lf_share_mean'))} "
                  f"{_fmt(g.get('par_share_mean'))} {_fmt(g.get('cap_mean'))}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad arm must not kill the matrix
            print(f"{name:8s} FAILED {type(exc).__name__}: {exc}", flush=True)
            rows.append({"arm": name, "cfg": cfg, "failed": f"{type(exc).__name__}: {exc}"})

    meta = {"family": args.family, "res": args.res, "steps": args.steps,
            "scheduler": args.scheduler, "seed": args.seed, "rows": rows}
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    with open(os.path.join(args.out, "results.txt"), "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            if "failed" in r:
                fh.write(f"{r['arm']:8s} FAILED {r['failed']}\n")
                continue
            g = r["gap"]
            fh.write(f"{r['arm']:8s} {r['cfg']:4g} {r['sec']:6.1f} "
                     f"{_num(r.get('sat'), 7, 4)} {_num(r.get('clip'), 7, 4)} "
                     f"{_num(r.get('hf'), 8, 5)} {_num(r.get('lum'), 6, 4)} | "
                     f"{_fmt(g.get('rms_ratio_mean'))} {_fmt(g.get('lf_share_mean'))} "
                     f"{_fmt(g.get('par_share_mean'))} {_fmt(g.get('cap_mean'))}\n")
    print(f"\nwrote {len(rows)} rows to {os.path.join(args.out, 'results.txt')}")


if __name__ == "__main__":
    main()
