"""Regression: eta (SDE) must accept negative seeds.

torchsde's BrownianTree passes the seed to numpy's SeedSequence, which raises
"ValueError: expected non-negative integer" for negative values. ComfyUI seed
nodes can supply them even though the widget declares min=0, so the sampler
normalises instead of crashing.

Also checks that eta=0 never constructs a noise sampler (the bug must not be
reachable when the feature is off).
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, r"G:\AI\comfy")
sys.path.insert(0, r"G:\AI\agent working folders\generic")

from omni_sampler.omni_core import sample_omni  # noqa: E402
from omni_sampler.tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init  # noqa: E402


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


def run(seed, eta):
    prior = MixturePrior(dim=2, seed=0)
    model = ToyModel(prior, flow=False)
    sig = torch.tensor([float(s) for s in karras_sigmas(8, model.sigma_min, model.sigma_max)])
    x, _ = make_init(prior, 32, float(sig[0]), False, torch.Generator().manual_seed(0))
    c = Counter(model)
    out = sample_omni(c, x.clone(), sig, extra_args={"seed": seed}, disable=True, eta=eta)
    return out, c.n


def main():
    torch.set_num_threads(2)
    _, n_off = run(seed=-1, eta=0.0)
    print(f"eta=0.0  nfe={n_off} (noise sampler not used)")
    for seed in (-1, -123456789, 7):
        out, n = run(seed=seed, eta=0.3)
        assert torch.isfinite(out).all(), f"non-finite output for seed {seed}"
        print(f"eta=0.3 seed={seed:>12d}  nfe={n}  finite=True")
    a, _ = run(seed=-1, eta=0.3)
    b, _ = run(seed=-1, eta=0.3)
    c, _ = run(seed=7, eta=0.3)
    assert torch.equal(a, b), "same seed must reproduce"
    assert not torch.equal(a, c), "different seeds must differ"
    print("reproducible: same seed identical, different seed differs")


if __name__ == "__main__":
    main()
