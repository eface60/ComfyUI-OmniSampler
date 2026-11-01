"""Mode-free accuracy metric: per-interval local ODE error.

With these models, comparing to a high-NFE reference measures which basin the
trajectory fell into, not integration accuracy (see docs/CORRECTIONS.md).  Inside
a single interval [sigma_i, sigma_{i+1}] the basin is already decided, so the
sampler's step can be checked against a fine sub-solve from the same x_i.

Implementation avoids re-plumbing the guider: it re-runs the *same* sampler on a
refined sub-schedule starting from the captured x_i and compares the denoised
(x0) predictions at the shared endpoint node.

Two details make or break the number, and both were wrong in the first version:

* The fine run's endpoint is its record index ``substeps`` (the callback fires
  at every node, including the final one), not ``substeps - 1``.  Using the
  penultimate node compares two different sigmas and can inflate the interval-0
  error by an order of magnitude.
* The number of substeps must resolve the interval.  A flow schedule's first
  lambda gap is ~12.7; 16 uniform substeps cannot integrate that at first order,
  so the reference is itself wrong.  The substep count is now
  ``max(--substeps, ceil(h / --max-lambda-step))`` (docs/CORRECTIONS.md).

Variants are named configurations applied to BOTH the coarse run and the
refined reference (the reference uses order 1 and no corrector, but keeps the
guidance mode).  Stochastic or probe-dependent options (eta, churn, skip,
cfg_map) are refused: the reference is a deterministic ODE sub-solve.

Before measuring, the script repeats the first coarse run and its refined
reference and requires both to be bit-identical.  If they are not, it writes no
JSON and exits non-zero: the rows would measure run-to-run drift, not
integration error.  (MOSS failed at 10 steps with 7% coarse drift; Krea2 failed
with 28% coarse and 12% reference drift, so neither model is quotable on this
stack.)  The output name carries the variant list, so two runs cannot silently
overwrite each other.

    $env:COMFY_DIR = "C:\\path\\to\\ComfyUI"
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\local_metric.py moss ^
        --steps 10 --res 768 --intervals 5 --variants first,none
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

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

OUT = os.path.join(WORK, "omni_sampler", "outputs")
PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"
SEED = 7

# The named variants.  Only deterministic, step-shape options belong here: the
# fine reference is a deterministic ODE sub-solve of the same path.
VARIANTS = {
    "base": {},
    "none": {"corrector": "none"},
    "first": {"corrector": "first"},
    "last": {"corrector": "last"},
    "first_last": {"corrector": "first_last"},
    "auto": {"corrector": "auto"},
    "exact": {"x0_average": "exact"},
    "cfg_pp": {"cfg_pp": True},
    "cfg_pp_px": {"cfg_pp": True, "px_order": True},
    "cfg_pp_corr": {"cfg_pp": True, "corrector": "first", "corrector_cfgpp": True},
}


def _variant(name):
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; choose from {', '.join(VARIANTS)}")
    opts = VARIANTS[name]
    if float(opts.get("eta", 0.0)) > 0.0 or float(opts.get("s_churn", 0.0)) > 0.0:
        raise SystemExit(
            f"variant {name!r} is stochastic; the local reference is a "
            "deterministic ODE sub-solve"
        )
    if int(opts.get("skip_every", 1)) > 1:
        raise SystemExit(
            f"variant {name!r} skips model evaluations; the refined reference "
            "would skip them too"
        )
    if bool(opts.get("cfg_map", False)):
        raise SystemExit(
            f"variant {name!r} uses cfg_map, whose map is measured over the "
            "remaining trajectory and is not valid on a refined sub-schedule"
        )
    return opts


def sampler_for(opts, order):
    """Build the KSAMPLER for one arm; the refined reference forces order 1."""
    run_opts = dict(opts)
    run_opts["order"] = order
    if order == 1:
        run_opts["corrector"] = "none"
    return comfy.samplers.KSAMPLER(
        lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, **run_opts, **kw))


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def build_schedule(l0, l1, n_sub, s_i, flow):
    """Uniform-in-lambda refined schedule with the interval's true start sigma."""
    sub = [omni_core.sigma_of_lambda(l0 + (l1 - l0) * k / n_sub, flow)
           for k in range(n_sub + 1)]
    sub[0] = s_i
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--intervals", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=16,
                    help="minimum substeps per interval (the resolved count can be higher)")
    ap.add_argument("--max-lambda-step", type=float, default=0.2,
                    help="maximum lambda spacing in the refined reference")
    ap.add_argument("--max-substeps", type=int, default=256)
    ap.add_argument("--variants", default="first,none",
                    help="comma-separated names from " + ", ".join(VARIANTS))
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--ref-check", dest="ref_check", action="store_true", default=True,
                    help="double the first interval's reference to report its own error")
    ap.add_argument("--no-ref-check", dest="ref_check", action="store_false")
    ap.add_argument("--repeat-check", dest="repeat_check", action="store_true", default=True,
                    help="run the first arm and its refined reference twice and abort "
                         "unless both are bit-identical")
    ap.add_argument("--no-repeat-check", dest="repeat_check", action="store_false")
    ap.add_argument("--ref-tol", type=float, default=1e-2,
                    help="warn when the reference's own error exceeds this")
    args = ap.parse_args()

    model, clip, vae, lf = load(args.family)
    flow = omni_core.is_flow_model(model.get_model_object("model_sampling"))
    cfg_scale = FAMILIES[args.family]["cfg"]
    hw = args.res // lf.spacial_downscale_ratio
    latent = (torch.zeros([1, lf.latent_channels, 1, hw, hw])
              if getattr(lf, "latent_dimensions", 2) == 3 else
              torch.zeros([1, lf.latent_channels, hw, hw]))
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    def run(sigmas, x_start, sampler, record):
        s0 = float(sigmas[0])
        noise = (x_start / s0).detach()

        def cb(step, x0, x, total):
            record.append((x.detach().clone(), x0.detach().clone()))

        torch.manual_seed(SEED)
        with torch.no_grad():
            comfy.samplers.sample(
                model, noise, pos, neg, cfg_scale, model.load_device, sampler,
                torch.as_tensor(sigmas, device=model.load_device), model_options=model.model_options,
                latent_image=latent, callback=cb, disable_pbar=True, seed=SEED)

    base_sig = comfy.samplers.KSampler(
        model, steps=args.steps, device=model.load_device, sampler="euler",
        scheduler=args.scheduler, denoise=1.0, model_options=model.model_options).sigmas

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    if not names:
        raise SystemExit("no variants given")
    start = 0 if flow else 1   # eps max_denoise distorts a substep run at sigma_max

    if args.repeat_check:
        # A number is only admissible if the run that produces it reproduces.
        # This is the invariant that failed on MOSS: identical runs moved the
        # decoded trajectory by 7% (zero start) and up to 67% (random start), so
        # every local-error row for that model was noise.  Check the coarse run
        # and the first refined reference before measuring anything.
        first_opts = _variant(names[0])
        coarse, coarse2 = [], []
        run([float(s) for s in base_sig], latent,
            sampler_for(first_opts, int(first_opts.get("order", 2))), coarse)
        run([float(s) for s in base_sig], latent,
            sampler_for(first_opts, int(first_opts.get("order", 2))), coarse2)
        s_i = float(base_sig[start])
        s_next = float(base_sig[start + 1])
        l0 = omni_core.half_log_snr(omni_core.clamp_sigma(s_i, flow), flow)
        l1 = omni_core.half_log_snr(omni_core.clamp_sigma(s_next, flow), flow)
        n_sub = min(int(args.max_substeps),
                    max(int(args.substeps), int(math.ceil(abs(l1 - l0)
                                                         / max(args.max_lambda_step, 1e-6)))))
        sub = build_schedule(l0, l1, n_sub, s_i, flow)
        fine, fine2 = [], []
        run(sub + [0.0], coarse[start][0], sampler_for(first_opts, order=1), fine)
        run(sub + [0.0], coarse[start][0], sampler_for(first_opts, order=1), fine2)
        coarse_rel = _rel(coarse[-1][1], coarse2[-1][1])
        fine_rel = _rel(fine[n_sub][1], fine2[n_sub][1])
        print(f"repeat check ({names[0]}): coarse rel={coarse_rel:.3e} "
              f"fine rel={fine_rel:.3e}")
        if not (torch.equal(coarse[-1][1], coarse2[-1][1])
                and torch.equal(fine[n_sub][1], fine2[n_sub][1])):
            print("[FAIL] the sampler is not reproducible in this process; the "
                  "local-error rows would measure run-to-run drift, not integration "
                  "error. No JSON was written.")
            return 1

    rows = []
    os.makedirs(OUT, exist_ok=True)
    for variant in names:
        opts = _variant(variant)
        rec = []
        run([float(s) for s in base_sig], latent, sampler_for(opts, int(opts.get("order", 2))), rec)
        if len(rec) < start + 2:
            raise SystemExit(
                f"{variant}: steps={args.steps} gives {len(rec)} recorded nodes; "
                f"at least {start + 2} are needed for one interval"
            )
        errs = []
        intervals = []
        # rec[i + 1] is the shared endpoint of interval i, so the last usable
        # interval is len(rec) - 2; the final callback is the sigma=0 denoise.
        for i in range(start, min(args.intervals, len(rec) - 2) + 1):
            s_i = float(base_sig[i])
            s_next = float(base_sig[i + 1])
            x_i = rec[i][0]
            l0 = omni_core.half_log_snr(omni_core.clamp_sigma(s_i, flow), flow)
            l1 = omni_core.half_log_snr(omni_core.clamp_sigma(s_next, flow), flow)
            h = abs(l1 - l0)
            n_sub = max(int(args.substeps), int(math.ceil(h / max(args.max_lambda_step, 1e-6))))
            if n_sub > args.max_substeps:
                print(f"  [warn] interval {i}: {n_sub} substeps capped to {args.max_substeps}")
                n_sub = int(args.max_substeps)

            sub = build_schedule(l0, l1, n_sub, s_i, flow)
            fine_rec = []
            run(sub + [0.0], x_i, sampler_for(opts, order=1), fine_rec)
            if len(fine_rec) != n_sub + 1:
                raise SystemExit(
                    f"interval {i}: expected {n_sub + 1} nodes, got {len(fine_rec)}; "
                    "the endpoint index assumption no longer holds"
                )
            den_fine = fine_rec[n_sub][1]
            den_coarse = rec[i + 1][1]
            err = _rel(den_coarse, den_fine)

            ref_unc = None
            if args.ref_check and i == start:
                n_ref = min(2 * n_sub, args.max_substeps)
                if n_ref <= n_sub:
                    print(f"  [warn] interval {i}: substep cap leaves no room to "
                          "check the reference; ref_unc not measured")
                else:
                    ref_rec = []
                    run(build_schedule(l0, l1, n_ref, s_i, flow) + [0.0], x_i,
                        sampler_for(opts, order=1), ref_rec)
                    if len(ref_rec) == n_ref + 1:
                        ref_unc = _rel(ref_rec[n_ref][1], den_fine)

            errs.append(err)
            intervals.append(dict(interval=i, h=h, substeps=n_sub, err=err, ref_unc=ref_unc))
            ref_txt = f" ref_unc={ref_unc:.1e}" if ref_unc is not None else ""
            print(f"  {variant:10s} i={i} h={h:7.3f} n={n_sub:3d} err={err:.3e}{ref_txt}")

        mean = sum(errs) / len(errs) if errs else float("nan")
        first_unc = intervals[0]["ref_unc"] if intervals else None
        if first_unc is not None and first_unc > args.ref_tol:
            print(f"  [warn] {variant}: first-interval reference is not converged "
                  f"(ref_unc={first_unc:.2e} > {args.ref_tol:.2e}); interval {start} is unresolved")
        rows.append(dict(variant=variant, mean_local=mean,
                         max_local=max(errs) if errs else None,
                         ref_unc_first=first_unc, per_interval=intervals))
        print(f"  {variant:10s} mean={mean:.4e} max={max(errs) if errs else float('nan'):.4e}")

    path = os.path.join(OUT, f"localerr_{args.family}_{'-'.join(names)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(family=args.family, steps=args.steps, res=args.res,
                       scheduler=args.scheduler, max_lambda_step=args.max_lambda_step,
                       variants=[r["variant"] for r in rows], rows=rows), f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
