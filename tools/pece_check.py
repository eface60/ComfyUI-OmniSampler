"""A/B: PECE (deferred evaluation, corrected every step, 1 NFE/node) vs the
current corrector-on-step-0 arrangement.

Variants are compared at equal and at equal-quality NFE on the analytic problem;
`sd_cfgpp`-style rendering is done separately.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omni_core
from tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init, reference_endpoint, uniform_sigmas

torch.set_default_dtype(torch.float32)
N = 160

VARIANTS = [
    ("corrector=first (current)", dict(pece=False, corrector="first")),
    ("corrector=none", dict(pece=False, corrector="none")),
    ("PECE", dict(pece=True)),
    ("PECE order=1", dict(pece=True, order=1)),
    ("PECE eta=0.3", dict(pece=True, eta=0.3)),
]


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


def main():
    for flow, sched in ((False, "karras"), (True, "uniform")):
        print(f"\n=== {'flow' if flow else 'eps '} / {sched} ===")
        print(f"{'variant':28s} " + " ".join(f"{'n='+str(n):>22s}" for n in (6, 12, 20)))
        for label, kw in VARIANTS:
            cells = []
            for n in (6, 12, 20):
                prior = MixturePrior(dim=2, seed=0)
                model = ToyModel(prior, flow=flow)
                sig = (karras_sigmas if sched == "karras" else uniform_sigmas)(n, model.sigma_min, model.sigma_max)
                st = torch.tensor([float(s) for s in sig])
                x, _ = make_init(prior, N, float(sig[0]), flow, torch.Generator().manual_seed(0))
                ref = reference_endpoint(x, float(sig[0]), float(sig[-2]), prior, flow)
                rx = prior.posterior_mean(ref, torch.full((N,), float(sig[-2])), flow=flow)
                c = Counter(model)
                out = omni_core.sample_omni(c, x.clone(), st, extra_args={"seed": 3}, disable=True, **kw)
                err = float((out - rx).norm() / rx.norm())
                cells.append(f"{err:.3e} ({c.n} nfe)")
            print(f"{label:28s} " + " ".join(f"{v:>22s}" for v in cells))


if __name__ == "__main__":
    main()
