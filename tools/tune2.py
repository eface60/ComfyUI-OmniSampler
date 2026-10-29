"""Settle the remaining defaults: lower_order_to_end, final_extrapolate, corrector.

Also measures accuracy against the *true* sigma->0 endpoint, which is the fair
way to judge the endpoint extrapolation (the main benchmark compares at the
last sigma node instead).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omni_core
from tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init, reference_endpoint, uniform_sigmas

torch.set_default_dtype(torch.float32)
N = 128
CASES = [(False, "karras"), (False, "uniform"), (True, "karras"), (True, "uniform")]
STEPS = (6, 12)


class Counter:
    def __init__(self, model):
        self.model = model
        self.inner_model = model.inner_model
        self.model_sampling = model.model_sampling
        self.prior = model.prior
        self.flow = model.flow
        self.sigma_max = model.sigma_max
        self.sigma_min = model.sigma_min
        self.n = 0

    def __call__(self, x, sigma, **kw):
        self.n += 1
        return self.model(x, sigma, **kw)


def build(flow, schedule, n):
    prior = MixturePrior(dim=2, seed=0)
    model = ToyModel(prior, flow=flow)
    sig = karras_sigmas(n, model.sigma_min, model.sigma_max) if schedule == "karras" else \
        uniform_sigmas(n, model.sigma_min, model.sigma_max)
    sig_t = torch.tensor([float(s) for s in sig])
    x_init, _ = make_init(prior, N, float(sig[0]), flow, torch.Generator().manual_seed(0))
    ref_last = reference_endpoint(x_init, float(sig[0]), float(sig[-2]), prior, flow)
    ref_x0_last = prior.posterior_mean(ref_last, torch.full((N,), float(sig[-2])), flow=flow)
    ref_true = reference_endpoint(x_init, float(sig[0]), 1e-4, prior, flow)
    return model, sig_t, x_init, ref_x0_last, ref_true


VARIANTS = [
    ("base", dict(order=2, x0_average="ab2", min_step_ratio=0.5)),
    ("lowend1", dict(order=2, x0_average="ab2", min_step_ratio=0.5, lower_order_to_end=1)),
    ("lowend2", dict(order=2, x0_average="ab2", min_step_ratio=0.5, lower_order_to_end=2)),
    ("fe0.5", dict(order=2, x0_average="ab2", min_step_ratio=0.5, final_extrapolate=0.5)),
    ("fe1", dict(order=2, x0_average="ab2", min_step_ratio=0.5, final_extrapolate=1.0)),
    ("corr first", dict(order=2, x0_average="ab2", min_step_ratio=0.5, corrector="first")),
    ("corr last", dict(order=2, x0_average="ab2", min_step_ratio=0.5, corrector="last")),
    ("eta.5", dict(order=2, x0_average="ab2", min_step_ratio=0.5, eta=0.5)),
]


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main():
    rows = []  # (variant, n, err_last, err_true, nfe)
    for flow, schedule in CASES:
        for n in STEPS:
            model, sig_t, x_init, ref_last, ref_true = build(flow, schedule, n)
            for label, kw in VARIANTS:
                c = Counter(model)
                out = omni_core.sample_omni(
                    c, x_init.clone(), sig_t, extra_args={"seed": 7}, disable=True, **kw
                )
                rows.append((label, n, rel(out, ref_last), rel(out, ref_true), c.n))

    print(f"{'variant':12s} {'case':6s} " + " ".join(f"{'e_last@'+str(n):>10s}" for n in STEPS)
          + "   " + " ".join(f"{'e_true@'+str(n):>10s}" for n in STEPS) + "   nfe")
    for label, _ in VARIANTS:
        cells_last, cells_true, nfes = [], [], []
        for n in STEPS:
            v = [r for r in rows if r[0] == label and r[1] == n]
            cells_last.append(sum(x[2] for x in v) / len(v))
            cells_true.append(sum(x[3] for x in v) / len(v))
            nfes.append(max(x[4] for x in v))
        print(f"{label:12s} {'mean':6s} " + " ".join(f"{v:10.2e}" for v in cells_last)
              + "   " + " ".join(f"{v:10.2e}" for v in cells_true) + f"   {max(nfes)}")


if __name__ == "__main__":
    main()
