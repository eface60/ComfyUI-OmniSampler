"""Benchmark the omni sampler against every built-in ComfyUI sampler.

Test problem: analytic Gaussian mixture -> the exact PF-ODE is integrated
(DOP853, rtol 1e-10) and used as ground truth.  Metrics:

    err  = ||x_out - x0_ref|| / ||x0_ref||
    mode = fraction of samples that land on the same mixture mode as the
           reference (only meaningful for stochastic samplers, which may
           legitimately explore other modes)

Usage (from the repository root, with ComfyUI's venv python; COMFY_DIR
overrides the default path):

    $env:COMFY_DIR = "C:\\path\\to\\ComfyUI"
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\toy_bench.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfy.k_diffusion.sampling as kds

import omni_core
from tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init, reference_endpoint, uniform_sigmas

torch.set_default_dtype(torch.float32)

N_SAMPLES = 160
STEPS = (6, 12, 20)
SCHEDULES = ("karras", "uniform")


class Counter:
    def __init__(self, model):
        self.model = model
        self.n = 0
        self.prior = model.prior
        self.flow = model.flow
        self.inner_model = model.inner_model
        self.model_sampling = model.model_sampling
        self.sigma_max = model.sigma_max
        self.sigma_min = model.sigma_min

    def __call__(self, x, sigma, **kw):
        self.n += 1
        return self.model(x, sigma, **kw)


def build_case(flow, n, schedule, seed=0):
    prior = MixturePrior(dim=2, seed=0)
    model = ToyModel(prior, flow=flow)
    if schedule == "karras":
        sig = karras_sigmas(n, model.sigma_min, model.sigma_max)
    else:
        sig = uniform_sigmas(n, model.sigma_min, model.sigma_max)
    sig_t = torch.tensor([float(s) for s in sig])
    x_init, _ = make_init(prior, N_SAMPLES, float(sig[0]), flow, torch.Generator().manual_seed(seed))
    ref = reference_endpoint(x_init, float(sig[0]), float(sig[-2]), prior, flow)
    ref_x0 = prior.posterior_mean(ref, torch.full((N_SAMPLES,), float(sig[-2])), flow=flow)
    return model, sig_t, x_init, ref_x0


def score(out, ref_x0, prior):
    e = float((out - ref_x0).norm() / ref_x0.norm().clamp_min(1e-30))
    same = float((prior.nearest_mode(out) == prior.nearest_mode(ref_x0)).float().mean())
    return e, same


def run_comfy(name, model, sig, x_init, extra=None):
    fn = getattr(kds, f"sample_{name}")
    counter = Counter(model)
    out = fn(counter, x_init.clone(), sig.clone(), extra_args={}, callback=None, disable=True, **(extra or {}))
    return out, counter.n


def run_omni(model, sig, x_init, **kw):
    counter = Counter(model)
    out = omni_core.sample_omni(
        counter, x_init.clone(), sig.clone(), extra_args={"seed": 12345}, disable=True, **kw
    )
    return out, counter.n


COMFY = [
    ("euler", {}),
    ("heun", {}),
    ("dpmpp_2m", {}),
    ("dpmpp_3m_sde", {"eta": 1.0}),
    ("dpmpp_2m_sde", {"eta": 1.0}),
    ("dpmpp_sde", {"eta": 1.0}),
    ("lms", {}),
    ("ipndm", {}),
    ("deis", {}),
    ("res_multistep", {}),
    ("exp_heun_2_x0", {}),
    ("er_sde", {}),
    ("seeds_2", {"eta": 1.0}),
    ("heunpp2", {}),
    ("sa_solver", {}),
    ("gradient_estimation", {}),
    ("dpmpp_2s_ancestral", {"eta": 1.0}),
    ("euler_ancestral", {"eta": 1.0}),
]

OMNI = [
    ("omni (defaults)", {}),
    ("omni o1", dict(order=1)),
    ("omni no-gate", dict(min_step_ratio=0.0)),
    ("omni exact-kernel", dict(x0_average="exact")),
    ("omni no-corrector", dict(corrector="none")),
    ("omni sde eta.5", dict(eta=0.5)),
    ("omni sde eta1", dict(eta=1.0)),
]


def compare():
    summary = {}
    for flow in (False, True):
        for schedule in SCHEDULES:
            tag = f"{'FLOW' if flow else 'EPS '} / {schedule}"
            print(f"\n=== {tag} ===  (N={N_SAMPLES} samples)")
            cases = {n: build_case(flow, n, schedule) for n in STEPS}
            header = f"{'sampler':24s} {'nfe':>4s} " + " ".join(
                f"{'err@'+str(n):>9s} {'md@'+str(n):>5s}" for n in STEPS
            )
            print(header)
            print("-" * len(header))
            rows = []
            for name, kw in COMFY:
                errs, modes, nfes, ok = [], [], [], True
                for n in STEPS:
                    model, sig, x_init, ref = cases[n]
                    try:
                        out, nfe = run_comfy(name, model, sig, x_init, kw)
                        e, m = score(out, ref, model.prior)
                        errs.append(e)
                        modes.append(m)
                        nfes.append(nfe)
                    except Exception as exc:  # noqa: BLE001
                        ok = False
                        print(f"    [skip {name}: {type(exc).__name__}: {exc}]")
                        break
                if ok:
                    rows.append((name, max(nfes), errs, modes))
            for label, kw in OMNI:
                errs, modes, nfes = [], [], []
                for n in STEPS:
                    model, sig, x_init, ref = cases[n]
                    out, nfe = run_omni(model, sig, x_init, **kw)
                    e, m = score(out, ref, model.prior)
                    errs.append(e)
                    modes.append(m)
                    nfes.append(nfe)
                rows.append((label, max(nfes), errs, modes))
            for name, nfe, errs, modes in rows:
                cells = []
                for e, m in zip(errs, modes):
                    cells.append(f"{e:9.2e} {m:5.2f}")
                print(f"{name:24s} {nfe:4d} " + " ".join(cells))
            # ranking by mean err and by mean mode accuracy
            rank_err = {n: sum(e) / len(e) for n, _, e, _ in rows}
            rank_mode = {n: sum(m) / len(m) for n, _, _, m in rows}
            for n, v in rank_err.items():
                summary.setdefault(n, []).append(v)
            best_e = sorted(rank_err.items(), key=lambda kv: kv[1])[:5]
            best_m = sorted(rank_mode.items(), key=lambda kv: -kv[1])[:5]
            print("  best mean err : " + ", ".join(f"{k}={v:.2e}" for k, v in best_e))
            print("  best mean mode: " + ", ".join(f"{k}={v:.2f}" for k, v in best_m))
    print("\n=== mean relative error across all 4 cases (lower is better) ===")
    for name, vals in sorted(summary.items(), key=lambda kv: sum(kv[1])):
        print(f"  {name:24s} {sum(vals)/len(vals):.3e}   " + " ".join(f"{v:.1e}" for v in vals))


if __name__ == "__main__":
    compare()

