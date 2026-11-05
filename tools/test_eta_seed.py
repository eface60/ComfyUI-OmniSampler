"""Regression: eta (SDE) must accept negative seeds, and eta=0 must not build a
noise sampler.

torchsde's BrownianTree passes the seed to numpy's SeedSequence, which raises
"ValueError: expected non-negative integer" for negative values. ComfyUI seed
nodes can supply them even though the widget declares min=0, so the sampler
normalises instead of crashing.

The "eta=0 never constructs a noise sampler" claim used to be printed rather
than asserted, and a model-call counter could not prove it: the sampler object
is built in a different code path. This version spies on the two noise-sampler
constructors in ``omni_core`` and asserts the call counts, on both model
families, so a regression fails the script instead of printing a comment.

    $env:COMFY_DIR = "C:\\path\\to\\ComfyUI"
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\test_eta_seed.py
"""

from __future__ import annotations

import os
import sys

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
WORK = os.environ.get(
    "OMNI_WORK", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, COMFY)
sys.path.insert(0, WORK)

from omni_sampler import omni_core as core  # noqa: E402
from omni_sampler.tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


class Counter:
    """Delegating wrapper that counts model evaluations."""

    def __init__(self, model):
        self._m = model
        self.inner_model = model.inner_model
        self.model_sampling = model.model_sampling
        self.prior = model.prior
        self.flow = model.flow
        self.sigma_max = model.sigma_max
        self.sigma_min = model.sigma_min
        self.n = 0

    def __call__(self, x, sigma, **kw):
        self.n += 1
        return self._m(x, sigma, **kw)


# Spy on the two constructors sample_omni uses.  They are module globals, so
# replacing the attribute changes what the running function resolves.
_BUILT = {"brownian": 0, "white": 0}
_BT = core.BrownianTreeNoiseSampler
_DEF = core.default_noise_sampler


def _spy_brownian(*args, **kwargs):
    _BUILT["brownian"] += 1
    return _BT(*args, **kwargs)


def _spy_white(*args, **kwargs):
    _BUILT["white"] += 1
    return _DEF(*args, **kwargs)


def run(seed, eta, flow=False, noise_mode="brownian"):
    for key in _BUILT:
        _BUILT[key] = 0
    prior = MixturePrior(dim=2, seed=0)
    model = ToyModel(prior, flow=flow)
    sig = torch.tensor([float(s) for s in karras_sigmas(8, model.sigma_min, model.sigma_max)])
    x, _ = make_init(prior, 32, float(sig[0]), flow, torch.Generator().manual_seed(0))
    c = Counter(model)
    out = core.sample_omni(c, x.clone(), sig, extra_args={"seed": seed}, disable=True,
                           eta=eta, noise_mode=noise_mode)
    return out, c.n


def main():
    torch.set_num_threads(2)
    core.BrownianTreeNoiseSampler = _spy_brownian
    core.default_noise_sampler = _spy_white
    try:
        for flow in (False, True):
            tag = "flow" if flow else "eps "
            out, n = run(seed=-1, eta=0.0, flow=flow)
            check(f"{tag}: eta=0 builds no noise sampler",
                  _BUILT["brownian"] == 0 and _BUILT["white"] == 0, str(_BUILT))
            out, n = run(seed=-1, eta=0.3, flow=flow)
            check(f"{tag}: eta>0 builds the brownian sampler",
                  _BUILT["brownian"] == 1 and _BUILT["white"] == 0, str(_BUILT))
            check(f"{tag}: eta=0.3 output is finite", bool(torch.isfinite(out).all()),
                  f"nfe={n}")
            out, n = run(seed=-1, eta=0.3, flow=flow, noise_mode="white")
            check(f"{tag}: noise_mode=white builds the white sampler",
                  _BUILT["white"] == 1 and _BUILT["brownian"] == 0, str(_BUILT))

        for seed in (-1, -123456789, 7):
            out, n = run(seed=seed, eta=0.3)
            check(f"negative/positive seed {seed} runs", bool(torch.isfinite(out).all()),
                  f"nfe={n}")

        a, _ = run(seed=-1, eta=0.3)
        b, _ = run(seed=-1, eta=0.3)
        c, _ = run(seed=7, eta=0.3)
        check("same seed reproduces bit-exactly", torch.equal(a, b))
        check("different seed differs", not torch.equal(a, c))
    finally:
        core.BrownianTreeNoiseSampler = _BT
        core.default_noise_sampler = _DEF

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
