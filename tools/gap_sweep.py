"""Cross-model guidance-gap sweep: is there a model-independent law?

Runs the `gap_stats` probe (output unchanged) over a resolution ladder on every
installed family and records the per-step cond/uncond gap: magnitude relative to
the prediction, per-channel tone (DC), the tail a cap would bite, and the
low-frequency energy profile at five bands.  The question is whether those
*normalised* numbers are the same function of trajectory position and relative
resolution for every model -- i.e. whether an automatic setting can be derived
once, or whether it has to be measured per run.

Nothing here changes an image: `gap_stats` only records.

    # the three families on the core ladder
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\gap_sweep.py

    # one family, quick
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\gap_sweep.py --families krea2 --res 512x512,1024x1024

Results are appended per config to outputs/gap_sweep/records.jsonl, so a long
run that is interrupted keeps everything it has already measured.
"""

from __future__ import annotations

import argparse
import json
import os
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
# Import the package's module, not the top-level one: the registered sampler
# function closes over omni_sampler.omni_core, and a bare `import omni_core`
# loads a SECOND module object with its own LAST_STATS (silently empty).
from omni_sampler import omni_core  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, PROMPT, NEG, load  # noqa: E402

OUT = os.path.join(REPO, "outputs", "gap_sweep")

# A ladder in area, not in side length: the hypothesis is about uncertainty
# growing with the number of tokens, so the law should be a power of area.
DEFAULT_RES = "512x512,768x768,1024x1024,1408x1408,2048x2048"


def latent_for(lf, w, h):
    ratio = int(lf.spacial_downscale_ratio)
    hw, ww = h // ratio, w // ratio
    ch = int(lf.latent_channels)
    if getattr(lf, "latent_dimensions", 2) == 3:
        return torch.zeros([1, ch, 1, hw, ww])
    return torch.zeros([1, ch, hw, ww])


def summarise(series, cfg):
    """Per-step means, with the magnitudes divided back to the raw gap.

    The post-CFG delta is `cfg * (cond - uncond)`, so every magnitude is divided
    by cfg to compare a cfg-1 model with a cfg-3.5 one.  The ratios (lf_share,
    par_share) are scale invariant and need no correction.
    """
    if not series:
        return {}
    def mean(key):
        vals = [m[key] for _, m in series if key in m]
        return sum(vals) / len(vals) if vals else None
    out = {
        "steps": len(series),
        "rms_ratio": mean("rms_ratio"),
        "dc_ratio": mean("dc_ratio"),
        "p99_ratio": mean("p99_ratio"),
        "lf_share": mean("lf_share"),
        "par_share": mean("par_share"),
        "u_rms": mean("u_rms"),
        "d_rms": mean("d_rms"),
        "rms_first": series[0][1].get("rms_ratio"),
        "rms_last": series[-1][1].get("rms_ratio"),
    }
    for b in (1, 2, 5, 10, 20):
        v = mean(f"lf_share_{b:02d}")
        if v is not None:
            out[f"lf_{b:02d}"] = v
    # raw (pre-cfg) magnitudes
    for k in ("rms_ratio", "dc_ratio", "p99_ratio", "rms_first", "rms_last"):
        if out.get(k) is not None:
            out[k + "_raw"] = out[k] / cfg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="krea2,moss,anima")
    ap.add_argument("--res", default=DEFAULT_RES)
    ap.add_argument("--extra-res", default="", help="additional WxH, e.g. 1024x1408,2048x2944")
    ap.add_argument("--steps", default="12", help="comma list of step counts")
    ap.add_argument("--seeds", default="7", help="comma list of seeds")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--unet", default="", help="family=path to override the checkpoint")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records_path = os.path.join(args.out, "records.jsonl")
    overrides = {}
    for item in args.unet.split(","):
        if "=" in item:
            fam, path = item.split("=", 1)
            overrides[fam.strip()] = path.strip()

    res_list = [r.strip() for r in (args.res.split(",") + args.extra_res.split(",")) if r.strip()]
    step_list = [int(s) for s in args.steps.split(",")]
    seed_list = [int(s) for s in args.seeds.split(",")]
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    print(f"families={families} res={res_list} steps={step_list} seeds={seed_list} "
          f"scheduler={args.scheduler}", flush=True)
    for fam in families:
        if fam not in FAMILIES:
            raise SystemExit(f"unknown family {fam!r}")
        if fam in overrides:
            FAMILIES[fam]["unet"] = overrides[fam]
        print(f"\n=== {fam}: {os.path.basename(FAMILIES[fam]['unet'])}", flush=True)
        model, clip, vae, lf = load(fam)
        ms = model.get_model_object("model_sampling")
        flow = isinstance(ms, comfy.model_sampling.CONST)
        cfg = float(FAMILIES[fam]["cfg"])
        pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
        print(f"    cfg={cfg} flow={flow} sampling={type(ms).__name__}", flush=True)

        for res in res_list:
            w, h = (int(v) for v in res.lower().split("x"))
            if w % 8 or h % 8:
                print(f"    skip {res}: not a multiple of 8")
                continue
            latent = latent_for(lf, w, h)
            for steps in step_list:
                for seed in seed_list:
                    label = f"{fam} {res} {steps}st seed={seed}"
                    try:
                        sampler = comfy.samplers.ksampler("omni_hybrid", {"gap_stats": True})
                        torch.manual_seed(seed)
                        noise = comfy.sample.prepare_noise(latent, seed)
                        ks = comfy.samplers.KSampler(
                            model, steps=steps, device=model.load_device, sampler="euler",
                            scheduler=args.scheduler, denoise=1.0,
                            model_options=model.model_options)
                        t0 = time.time()
                        with torch.no_grad():
                            comfy.samplers.sample(
                                model, noise, pos, neg, cfg, model.load_device, sampler,
                                ks.sigmas, model_options=model.model_options,
                                latent_image=latent, disable_pbar=True, seed=seed)
                        dt = time.time() - t0
                        st = dict(omni_core.LAST_STATS)
                        rec = {
                            "family": fam, "unet": os.path.basename(FAMILIES[fam]["unet"]),
                            "flow": bool(flow), "sampling": type(ms).__name__,
                            "res": res, "w": w, "h": h, "area": w * h / 1e6,
                            "steps": steps, "seed": seed, "scheduler": args.scheduler,
                            "cfg": cfg, "sec": round(dt, 1),
                            "latent": list(latent.shape),
                            "summary": summarise(st.get("gap_series") or [], cfg),
                            "series": st.get("gap_series") or [],
                        }
                        with open(records_path, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        s = rec["summary"]
                        print(f"    {label:32s} {dt:6.1f}s  "
                              f"rms={s.get('rms_ratio_raw', 0):.4f} "
                              f"dc={s.get('dc_ratio_raw', 0):.4f} "
                              f"p99={s.get('p99_ratio_raw', 0):.3f} "
                              f"lf05={s.get('lf_05', 0):.3f} "
                              f"par={s.get('par_share', 0):.3f}", flush=True)
                    except Exception as exc:  # noqa: BLE001 - one config must not kill the sweep
                        print(f"    {label:32s} FAILED {type(exc).__name__}: {exc}", flush=True)
                        with open(records_path, "a") as fh:
                            fh.write(json.dumps({"family": fam, "res": res, "steps": steps,
                                                 "seed": seed, "failed": f"{type(exc).__name__}: {exc}"}) + "\n")

        del model, clip, vae
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
