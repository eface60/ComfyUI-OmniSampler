"""Per-step diagnostic: where do omni and comfy's samplers diverge?"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("COMFY_DIR", r"G:\AI\comfy"))

import comfy.k_diffusion.sampling as kds
import omni_core
from omni_core import alpha_of, clamp_sigma, half_log_snr
from tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init, reference_endpoint, uniform_sigmas

torch.set_default_dtype(torch.float32)


class Rec:
    def __init__(self, model, sigmas, prior, flow):
        self.model = model
        self.sigmas = [float(s) for s in sigmas]
        self.prior = prior
        self.flow = flow
        self.snaps = []
        self.inner_model = model.inner_model
        self.model_sampling = model.model_sampling
        self.prior_ = model.prior

    def __call__(self, x, sigma, **kw):
        self.snaps.append(x.detach().clone())
        return self.model(x, sigma, **kw)


def ref_traj(x_init, sigmas, prior, flow):
    out = []
    for s in sigmas:
        if s <= 0:
            out.append(None)
            continue
        out.append(reference_endpoint(x_init, float(sigmas[0]), float(s), prior, flow))
    return out


def main(flow, schedule, n):
    prior = MixturePrior(dim=2, seed=0)
    model = ToyModel(prior, flow=flow)
    if schedule == "karras":
        sig = karras_sigmas(n, model.sigma_min, model.sigma_max)
    else:
        sig = uniform_sigmas(n, model.sigma_min, model.sigma_max)
    sig_t = torch.tensor([float(s) for s in sig])
    x_init, _ = make_init(prior, 64, float(sig[0]), flow, torch.Generator().manual_seed(0))
    traj = ref_traj(x_init, sig, prior, flow)

    def rel(x, i):
        if traj[i] is None:
            return float("nan")
        return float((x - traj[i]).norm() / traj[i].norm())

    print(f"\n{'FLOW' if flow else 'EPS'} / {schedule} / n={n}")
    print("sigmas:", " ".join(f"{float(s):.3g}" for s in sig))
    print("lam gaps:", " ".join(
        f"{half_log_snr(clamp_sigma(float(sig[i+1]), flow), flow) - half_log_snr(clamp_sigma(float(sig[i]), flow), flow):.2f}"
        for i in range(len(sig) - 1)
    ))
    runs = {}
    for label, fn in (
        ("omni o1", lambda m: omni_core.sample_omni(m, x_init.clone(), sig_t, extra_args={}, disable=True, order=1)),
        ("omni o2", lambda m: omni_core.sample_omni(m, x_init.clone(), sig_t, extra_args={}, disable=True, order=2)),
        ("omni o2 r0", lambda m: omni_core.sample_omni(m, x_init.clone(), sig_t, extra_args={}, disable=True, order=2, min_step_ratio=0.0)),
        ("dpmpp_2m", lambda m: kds.sample_dpmpp_2m(m, x_init.clone(), sig_t, extra_args={}, disable=True)),
        ("heun", lambda m: kds.sample_heun(m, x_init.clone(), sig_t, extra_args={}, disable=True)),
    ):
        rec = Rec(model, sig, prior, flow)
        fn(rec)
        # snaps[k] is the state at the k-th model evaluation == sigmas[k]
        errs = [rel(rec.snaps[i], i) for i in range(len(rec.snaps))]
        runs[label] = errs
        print(f"  {label:10s} " + " ".join(f"{e:.2e}" for e in errs))
    return runs


if __name__ == "__main__":
    for flow in (True, False):
        for sched in ("karras", "uniform"):
            main(flow, sched, 6)
