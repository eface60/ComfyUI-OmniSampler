"""churn x eta: after churning, is the state still a sample of p(sigma_hat)?

This is a one-step, exactly-solvable test.  With a *constant* x0 the denoiser is
exact, so the model's distribution at sigma is known in closed form:

    p_sigma = N(alpha(sigma) * x0, sigma^2),      alpha = 1 (eps), 1 - sigma (flow)

and the sampler's SDE step (eta = 1) is the exact OU transition, so if the state
entering the step is distributed as p_sigma, the state leaving it must be
distributed as p_sigma_t:  mean = alpha_t * x0, variance = sigma_t^2.  That is
what makes this test decisive rather than a benchmark: any deviation is a bug in
the churn step, not model error or mode choice.

The churn re-noise is only correct if it maps p_sigma_s onto p_sigma_hat.  For an
eps model that is plain additive noise (alpha = 1):

    x_hat = x + sqrt(sigma_hat^2 - sigma_s^2) * z

For a flow model (alpha = 1 - sigma) the exact forward step is

    x_hat = r * x + sqrt(sigma_hat^2 - (r*sigma_s)^2) * z,   r = alpha_hat/alpha_s

which the additive form gets wrong in its mean: the variance comes out right
(x_hat still has sigma_hat^2) but the x0 component is not rescaled, so the step
after it inherits a biased mean.  This script measures both quantities and fails
if either is off.

    $env:COMFY_DIR = "C:\\path\\to\\ComfyUI"
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\test_churn_eta.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(REPO))

from omni_sampler import omni_core  # noqa: E402
from omni_sampler.tools.toy import MixturePrior, ToyModel  # noqa: E402

FAILS = []
N = 100_000
X0 = torch.tensor([[2.0, -1.0]])


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


class ConstX0(ToyModel):
    """Denoiser that always returns the same x0: the distribution is exact."""

    def __init__(self, flow):
        super().__init__(MixturePrior(dim=2, seed=0), flow=flow)
        self.x0 = X0.clone()

    def __call__(self, x, sigma, **kw):
        return self.x0.expand_as(x)


def run(flow, sigmas, churn, noise_mode="white"):
    model = ConstX0(flow)
    sig = torch.tensor([float(s) for s in sigmas])
    g = torch.Generator().manual_seed(1234)
    eps = torch.randn(N, 2, generator=g)
    x0 = X0.expand(N, 2)
    alpha_s = (1.0 - float(sig[0])) if flow else 1.0
    x = alpha_s * x0 + float(sig[0]) * eps
    out = omni_core.sample_omni(
        model, x, sig, extra_args={"seed": 0}, disable=True,
        order=1, eta=1.0, eta_window=(0.0, 1.0), eta_softness=0.0,
        s_churn=churn, churn_mode="karras", noise_mode=noise_mode,
        corrector="none", lower_order_to_end=0, final_extrapolate=0.0,
    )
    alpha_t = (1.0 - float(sig[-1])) if flow else 1.0
    resid = out - alpha_t * x0
    var_ratio = float(resid.var(dim=0).mean() / float(sig[-1]) ** 2)
    mean_dev = float(resid.mean(dim=0).abs().max()) / float(sig[-1])
    return var_ratio, mean_dev, dict(omni_core.LAST_STATS)


def case(label, flow, sigmas, churn):
    var_ratio, mean_dev, st = run(flow, sigmas, churn)
    ok = abs(var_ratio - 1.0) < 0.02 and mean_dev < 0.01
    detail = (f"var_ratio={var_ratio:.4f} mean/sigma={mean_dev:.4f} "
              f"churn_steps={st.get('churn_steps')} calls={st.get('calls')}")
    check(f"{label} ({'flow' if flow else 'eps'}) marginal is exact", ok, detail)
    return ok


def test_eps_bit_identity():
    print("\n== eps churn is bit-identical to the old additive formula ==")
    x = torch.randn(4096, 2, generator=torch.Generator().manual_seed(5))
    s_s, s_hat, noise = 1.0, 1.4142135623730951, 1.0
    d = noise * math.sqrt(s_hat ** 2 - s_s ** 2)
    torch.manual_seed(0)
    old = x + torch.randn_like(x) * d
    torch.manual_seed(0)
    new, fired = omni_core._churn_forward(x, s_s, s_hat, False, noise)
    check("eps karras churn is bit-identical", bool(fired) and torch.equal(old, new))
    torch.manual_seed(0)
    old_g = x - torch.randn_like(x) * d
    torch.manual_seed(0)
    new_g, fired_g = omni_core._churn_forward(x, s_s, s_hat, False, noise, sign=-1.0)
    check("eps gamma churn is bit-identical", bool(fired_g) and torch.equal(old_g, new_g))
    same, fired_no = omni_core._churn_forward(x, s_s, s_s, False, noise)
    check("a degenerate move draws nothing", not fired_no and torch.equal(same, x))


def main():
    torch.set_num_threads(2)
    print(f"constant x0={X0.tolist()[0]}, N={N}, one or two steps, white noise\n")
    # Every arm must land on p(sigma_t): the additive churn form used to fail the
    # flow arms with a 7.2%-of-sigma mean bias and an exact variance.
    case("eta=1, churn=0", False, (2.0, 0.5), 0.0)
    case("eta=1, churn=1", False, (2.0, 0.5), 1.0)
    case("eta=1, churn=4", False, (2.0, 0.5), 4.0)
    case("eta=1, churn=0", True, (0.9, 0.6, 0.4), 0.0)
    case("eta=1, churn=1", True, (0.9, 0.6, 0.4), 1.0)
    case("eta=1, churn=4", True, (0.9, 0.6, 0.4), 4.0)
    test_eps_bit_identity()
    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
