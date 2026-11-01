"""Compare integrator modes (exponential vs Adams-Bashforth in sigma) and the
AB implementation against comfy's ipndm, on the analytic problem."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("COMFY_DIR", r"G:\AI\comfy"))

import comfy.k_diffusion.sampling as kds
import omni_core
from tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init, reference_endpoint, uniform_sigmas

torch.set_default_dtype(torch.float32)
N = 128
STEPS = (6, 12, 20)


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


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


MODES = [
    ("exp o2 (default)", dict(integrator="exp", order=2)),
    ("ab o1", dict(integrator="ab", order=1)),
    ("ab o2", dict(integrator="ab", order=2)),
    ("ab o3", dict(integrator="ab", order=3)),
    ("ab o4", dict(integrator="ab", order=4)),
]


def main():
    print(f"{'case':16s} {'n':>3s} " + " ".join(f"{m[0]:>17s}" for m in MODES) + f" {'ipndm':>17s} {'dpmpp_2m':>17s}")
    for flow, sched in ((False, "karras"), (False, "uniform"), (True, "karras"), (True, "uniform")):
        for n in STEPS:
            prior = MixturePrior(dim=2, seed=0)
            model = ToyModel(prior, flow=flow)
            sig = karras_sigmas(n, model.sigma_min, model.sigma_max) if sched == "karras" else \
                uniform_sigmas(n, model.sigma_min, model.sigma_max)
            sig_t = torch.tensor([float(s) for s in sig])
            x_init, _ = make_init(prior, N, float(sig[0]), flow, torch.Generator().manual_seed(0))
            ref = reference_endpoint(x_init, float(sig[0]), float(sig[-2]), prior, flow)
            ref_x0 = prior.posterior_mean(ref, torch.full((N,), float(sig[-2])), flow=flow)
            cells = []
            for _, kw in MODES:
                c = Counter(model)
                out = omni_core.sample_omni(
                    c, x_init.clone(), sig_t, extra_args={"seed": 3}, disable=True,
                    corrector="none", lower_order_to_end=1, **kw
                )
                cells.append(rel(out, ref_x0))
            c = Counter(model)
            ip = kds.sample_ipndm(c, x_init.clone(), sig_t, extra_args={}, disable=True)
            d2 = Counter(model)
            dp = kds.sample_dpmpp_2m(d2, x_init.clone(), sig_t, extra_args={}, disable=True)
            print(f"{('flow' if flow else 'eps')+'/'+sched:16s} {n:3d} " + " ".join(f"{v:17.2e}" for v in cells)
                  + f" {rel(ip, ref_x0):17.2e} {rel(dp, ref_x0):17.2e}")


if __name__ == "__main__":
    main()
