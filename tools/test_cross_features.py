"""Cross-feature wiring checks: the pairs no single-feature test covers.

1. **order clamp on exp.**  The exp path keeps one previous x0, so order 3-4
   cannot consume more history.  They must be clamped to 2, and the clamp must
   be recorded in LAST_STATS.  ab is the path that really uses 1-4.
2. **ab x churn.**  ab is deterministic-only; churn > 0 must fall back to exp,
   bit-identically for the same RNG state, and say so in LAST_STATS.
3. **final_extrapolate x churn.**  The endpoint fit must be given the sigmas the
   model actually saw (`sigma_hat`), not the un-churned schedule entries.
4. **churn x the flow sigma domain.**  A flow model is only defined on sigma < 1,
   so karras churn at step 0 must not evaluate above the schedule's first sigma;
   at `sigma == sigma_max` it is a bit-exact no-op.  eps models keep k-diffusion's
   uncapped behaviour.

Run with ComfyUI's venv python (COMFY_DIR overrides the default path):

    $env:COMFY_DIR = "C:\\path\\to\\ComfyUI"
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\test_cross_features.py
"""

from __future__ import annotations

import os
import sys

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFY)
sys.path.insert(0, REPO)

import omni_core  # noqa: E402
from tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def setup(flow=False, n=12, seed=11):
    prior = MixturePrior(dim=2, seed=0)
    model = ToyModel(prior, flow=flow)
    sig = torch.tensor([float(s) for s in karras_sigmas(n, model.sigma_min, model.sigma_max)])
    x, _ = make_init(prior, 64, float(sig[0]), flow, torch.Generator().manual_seed(seed))
    return model, sig, x


def run(model, sig, x, **kw):
    return omni_core.sample_omni(
        model, x.clone(), sig.clone(), extra_args={"seed": 0}, disable=True, **kw
    )


class SigmaSpy(ToyModel):
    """ToyModel that records every sigma the sampler evaluates it at."""

    def __init__(self, prior, flow=False):
        super().__init__(prior, flow=flow)
        self.seen = []

    def __call__(self, x, sigma, **kwargs):
        self.seen.append(float(torch.as_tensor(sigma).reshape(-1)[0]))
        return super().__call__(x, sigma, **kwargs)


def test_order_clamp():
    print("\n=== 1. order 3-4 is clamped to 2 on exp ===")
    model, sig, x = setup()
    o2 = run(model, sig, x, order=2, corrector="none", lower_order_to_end=0)
    o4 = run(model, sig, x, order=4, corrector="none", lower_order_to_end=0)
    st = dict(omni_core.LAST_STATS)
    check("order=4 is bit-identical to order=2", torch.equal(o2, o4))
    check("the request is recorded", st.get("order_requested") == 4,
          f"order_requested={st.get('order_requested')}")
    check("the clamp is recorded", st.get("order_effective") == 2,
          f"order_effective={st.get('order_effective')}")

    model, sig, x = setup(flow=True)
    f2 = run(model, sig, x, order=2, corrector="none", lower_order_to_end=0)
    f4 = run(model, sig, x, order=4, corrector="none", lower_order_to_end=0)
    check("flow: order=4 is bit-identical to order=2", torch.equal(f2, f4))


def test_ab_uses_order():
    print("\n=== 2. ab really uses order 1-4 (the clamp is exp-only) ===")
    model, sig, x = setup()
    a1 = run(model, sig, x, integrator="ab", order=1, corrector="none", lower_order_to_end=0)
    a2 = run(model, sig, x, integrator="ab", order=2, corrector="none", lower_order_to_end=0)
    st = dict(omni_core.LAST_STATS)
    check("ab order=2 differs from order=1", rel(a2, a1) > 1e-3, f"rel={rel(a2, a1):.2e}")
    check("ab keeps the requested order", st.get("order_effective") == 2
          and st.get("integrator") == "ab",
          f"order_effective={st.get('order_effective')} integrator={st.get('integrator')}")


def test_ab_churn_falls_back():
    print("\n=== 3. ab + churn falls back to exp, bit-identically ===")
    model, sig, x = setup()
    torch.manual_seed(1234)
    ab = run(model, sig, x, integrator="ab", s_churn=3.0, s_noise=0.7,
             corrector="none", lower_order_to_end=0)
    st = dict(omni_core.LAST_STATS)
    torch.manual_seed(1234)
    exp = run(model, sig, x, integrator="exp", s_churn=3.0, s_noise=0.7,
              corrector="none", lower_order_to_end=0)
    check("ab+churn equals exp+churn for the same RNG state", torch.equal(ab, exp))
    check("LAST_STATS reports the effective integrator", st.get("integrator") == "exp",
          f"integrator={st.get('integrator')}")

    # Negative control: without churn, ab really is the ab path.
    torch.manual_seed(1234)
    ab0 = run(model, sig, x, integrator="ab", corrector="none", lower_order_to_end=0)
    st0 = dict(omni_core.LAST_STATS)
    check("no-churn ab still reports ab", st0.get("integrator") == "ab",
          f"integrator={st0.get('integrator')}")
    check("no-churn ab differs from exp", not torch.equal(ab0, run(
        model, sig, x, integrator="exp", corrector="none", lower_order_to_end=0)))


def test_final_extrapolate_sigmas():
    print("\n=== 4. final_extrapolate receives the sigmas the model saw ===")
    model, sig, x = setup()
    n = len(sig) - 1
    raw = [float(s) for s in sig]
    gamma = min(100.0 / n, 2 ** 0.5 - 1.0)
    seen = {}
    original = omni_core._final_extrapolate

    def spy(x0, values, sigmas, weight):
        seen["sigmas"] = list(sigmas)
        return original(x0, values, sigmas, weight)

    omni_core._final_extrapolate = spy
    try:
        run(model, sig, x, s_churn=100.0, s_noise=0.0, final_extrapolate=1.0,
            corrector="none")
    finally:
        omni_core._final_extrapolate = original

    expected = [raw[n - 3] * (1.0 + gamma), raw[n - 2] * (1.0 + gamma), raw[n - 1] * (1.0 + gamma)]
    got = seen.get("sigmas")
    check("the endpoint fit ran", got is not None, str(got))
    if got is not None:
        check("the abscissae are the churned evaluation sigmas",
              all(abs(a - b) < 1e-9 for a, b in zip(got, expected)),
              f"got={[round(v, 6) for v in got]} expected={[round(v, 6) for v in expected]}")

    # Negative control: with churn off the abscissae are the raw schedule
    # entries, so the fix cannot have changed the default path.
    seen.clear()
    omni_core._final_extrapolate = spy
    try:
        run(model, sig, x, s_churn=0.0, final_extrapolate=1.0, corrector="none")
    finally:
        omni_core._final_extrapolate = original
    got0 = seen.get("sigmas")
    check("no churn: the abscissae are the raw schedule entries",
          got0 is not None
          and all(abs(a - b) < 1e-12 for a, b in zip(got0, raw[n - 3:n])),
          f"got={got0}")


def test_churn_flow_domain():
    print("\n=== 5. karras churn stays inside the flow model's sigma domain ===")
    prior = MixturePrior(dim=2, seed=0)
    model = SigmaSpy(prior, flow=True)
    sig = torch.tensor([float(s) for s in karras_sigmas(12, model.sigma_min, model.sigma_max)])
    x, _ = make_init(prior, 64, float(sig[0]), True, torch.Generator().manual_seed(11))
    n = len(sig) - 1
    gamma = min(8.0 / n, 2 ** 0.5 - 1.0)
    expected_capped = sum(1 for s in sig[:-1] if float(s) * (1 + gamma) > float(sig[0]))

    torch.manual_seed(7)
    run(model, sig, x, s_churn=8.0, churn_mode="karras", corrector="none")
    st = dict(omni_core.LAST_STATS)
    check("no evaluation sigma exceeds sigma_max",
          max(model.seen) <= float(sig[0]) + 1e-9,
          f"max={max(model.seen):.6f} sigma_max={float(sig[0]):.6f}")
    check("only the over-domain steps are capped", st.get("churn_capped") == expected_capped,
          f"churn_capped={st.get('churn_capped')} expected={expected_capped}")
    check("every step below the cap still churns", st.get("churn_steps") == n - 1,
          f"churn_steps={st.get('churn_steps')} of {n}")

    # A one-step flow run evaluates once, at sigma == sigma_max, so churn there has
    # to be a bit-exact no-op: same sigma, no draw, same output.
    prior1 = MixturePrior(dim=2, seed=0)
    m1 = SigmaSpy(prior1, flow=True)
    sig1 = torch.tensor([float(s) for s in karras_sigmas(1, m1.sigma_min, m1.sigma_max)])
    x1, _ = make_init(prior1, 64, float(sig1[0]), True, torch.Generator().manual_seed(11))
    torch.manual_seed(7)
    a = run(m1, sig1, x1, s_churn=8.0, churn_mode="karras", corrector="none")
    st1 = dict(omni_core.LAST_STATS)
    torch.manual_seed(7)
    b = run(m1, sig1, x1, s_churn=0.0, corrector="none")
    check("churn at sigma_max is a bit-exact no-op", torch.equal(a, b))
    check("it takes no churn draw", st1.get("churn_steps") == 0,
          f"churn_steps={st1.get('churn_steps')}")
    check("and reports the cap", st1.get("churn_capped") == 1,
          f"churn_capped={st1.get('churn_capped')}")

    # Negative control: eps has no upper sigma domain, so karras churn must keep
    # k-diffusion's behaviour and evaluate above the schedule's first sigma.
    prior_e = MixturePrior(dim=2, seed=0)
    me = SigmaSpy(prior_e, flow=False)
    sige = torch.tensor([float(s) for s in karras_sigmas(12, me.sigma_min, me.sigma_max)])
    xe, _ = make_init(prior_e, 64, float(sige[0]), False, torch.Generator().manual_seed(11))
    torch.manual_seed(7)
    run(me, sige, xe, s_churn=8.0, churn_mode="karras", corrector="none")
    ste = dict(omni_core.LAST_STATS)
    check("eps: every step injects churn", ste.get("churn_steps") == len(sige) - 1,
          f"churn_steps={ste.get('churn_steps')}")
    check("eps: nothing is capped", ste.get("churn_capped") == 0,
          f"churn_capped={ste.get('churn_capped')}")
    check("eps: the model is evaluated above sigma_max",
          max(me.seen) > float(sige[0]) + 1e-6,
          f"max={max(me.seen):.4f} sigma_max={float(sige[0]):.4f}")


def main():
    test_order_clamp()
    test_ab_uses_order()
    test_ab_churn_falls_back()
    test_final_extrapolate_sigmas()
    test_churn_flow_domain()
    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
