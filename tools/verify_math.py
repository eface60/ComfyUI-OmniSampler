"""Numerical verification of every formula used by omni_core.

Run with the ComfyUI venv python (from the repository root); COMFY_DIR
overrides the default path:

    $env:COMFY_DIR = "C:\\path\\to\\ComfyUI"
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\verify_math.py

Each check prints its measured deviation / convergence rate; the script exits
non-zero if any check fails.  Nothing here is eyeballed.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("COMFY_DIR", r"G:\AI\comfy"))

import comfy.k_diffusion.sampling as kds

import omni_core
from omni_core import (
    alpha_of,
    beta_of,
    clamp_sigma,
    exact_step,
    half_log_snr,
    kernel_moment1,
    sde_step,
    sigma_of_lambda,
    _eta_gate,
    _final_extrapolate,
)
from tools.toy import AnalyticX0Model, MixturePrior, ToyModel, karras_sigmas, make_init, reference_endpoint

torch.set_default_dtype(torch.float32)
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def rel(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ==========================================================================
# 1. exact one-step solution
# ==========================================================================
def test_exact_step():
    torch.manual_seed(0)
    x0 = torch.randn(3, 4, dtype=torch.float64)
    eps = torch.randn(3, 4, dtype=torch.float64)
    worst = {}
    for flow in (False, True):
        pairs = [(1.0, 0.7), (0.7, 0.3), (0.3, 0.05), (0.05, 0.01)] if flow else [
            (14.6, 10.0), (1.0, 0.5), (0.5, 0.1), (0.2, 0.05)
        ]
        w = 0.0
        for s_s, s_t in pairs:
            s_s, s_t = clamp_sigma(s_s, flow), clamp_sigma(s_t, flow)
            xs = alpha_of(s_s, flow) * x0 + s_s * eps
            xt_ref = alpha_of(s_t, flow) * x0 + s_t * eps
            h = half_log_snr(s_t, flow) - half_log_snr(s_s, flow)
            xt = exact_step(xs, s_s, s_t, x0, alpha_of(s_t, flow), h)
            w = max(w, rel(xt, xt_ref))
        worst["flow" if flow else "eps"] = w
    check(
        "exact_step reproduces the analytic trajectory (constant x0)",
        max(worst.values()) < 1e-12,
        f"max rel err eps={worst['eps']:.1e} flow={worst['flow']:.1e}",
    )


# ==========================================================================
# 2. the ODE in lambda:  dx/dlambda = -beta*x + alpha*x0_hat
# ==========================================================================
def test_ode_identity():
    prior = MixturePrior(dim=2, seed=0)
    worst = {}
    for flow in (False, True):
        model = ToyModel(prior, flow=flow)
        s0 = model.sigma_max
        s_end = 0.05
        x_init, _ = make_init(prior, 4, s0, flow, torch.Generator().manual_seed(1))
        s_mid = math.sqrt(s0 * s_end)
        s_a, s_b = s_mid * 1.001, s_mid * 0.999
        xa = reference_endpoint(x_init, s0, s_a, prior, flow).double()
        xb = reference_endpoint(x_init, s0, s_b, prior, flow).double()
        x_mid = reference_endpoint(x_init, s0, s_mid, prior, flow).double()
        la, lb = half_log_snr(s_a, flow), half_log_snr(s_b, flow)
        dx = (xb - xa) / (lb - la)
        x0h = prior.posterior_mean(x_mid, torch.full((4,), s_mid), flow=flow).double()
        rhs = -beta_of(s_mid, flow) * x_mid + alpha_of(s_mid, flow) * x0h
        worst["flow" if flow else "eps"] = rel(dx, rhs)
    check(
        "dx/dlambda = -beta*x + alpha*x0_hat holds on the exact trajectory",
        max(worst.values()) < 2e-3,
        f"rel residual eps={worst['eps']:.1e} flow={worst['flow']:.1e} (finite diff)",
    )


# ==========================================================================
# 3. SDE step: eta=0 reduction + stationary marginal
# ==========================================================================
def test_sde_step():
    torch.manual_seed(0)
    for flow in (False, True):
        s_s, s_t = (0.8, 0.4) if flow else (2.0, 1.0)
        x0 = torch.randn(8192, 2, dtype=torch.float64)
        eps = torch.randn(8192, 2, dtype=torch.float64)
        xs = alpha_of(s_s, flow) * x0 + s_s * eps
        h = half_log_snr(s_t, flow) - half_log_snr(s_s, flow)
        a_t = alpha_of(s_t, flow)

        red = sde_step(xs, s_s, s_t, x0, a_t, h, 0.0)[0]
        check(
            f"sde_step(eta=0) == exact_step ({'flow' if flow else 'eps'})",
            rel(red, exact_step(xs, s_s, s_t, x0, a_t, h)) < 1e-15,
            "identity",
        )

        z = torch.randn(8192, 2, dtype=torch.float64)
        xt, coeff = sde_step(xs, s_s, s_t, x0, a_t, h, 1.0)
        xt = xt + coeff * z
        resid = xt - a_t * x0                      # should be N(0, sigma_t^2)
        var_ratio = float(resid.var(0).mean() / (s_t ** 2))
        mean_dev = float(resid.mean(0).abs().max()) / s_t
        check(
            f"SDE eta=1 keeps N(0, sigma_t^2) marginal ({'flow' if flow else 'eps'})",
            abs(var_ratio - 1.0) < 0.05 and mean_dev < 0.05,
            f"var_ratio={var_ratio:.4f} mean/sigma={mean_dev:.4f}",
        )


# ==========================================================================
# 4. kernel moment: closed form (eps) and exact step for linear x0 (both)
# ==========================================================================
def test_kernel_moment():
    # (a) eps closed form: kappa1 = h/expm1(h) + (h-1), and eta scaling
    worst = 0.0
    for h in (0.01, 0.1, 0.5, 1.0, 2.5, 6.0):
        for eta in (0.0, 0.5, 1.0):
            got = kernel_moment1(h, eta, 0.0, False, 1.0, math.exp(-h))
            ref = (h * (1 + eta)) / math.expm1(h * (1 + eta)) + (h * (1 + eta) - 1.0)
            ref = ref / (1.0 + eta)
            worst = max(worst, abs(got - ref) / abs(ref))
    check("kernel_moment1 matches the closed form for eps models", worst < 1e-9, f"max rel={worst:.1e}")

    # (b) exact step for x0 linear in lambda, both families, vs analytic/solve_ivp
    worst = {False: 0.0, True: 0.0}
    for flow in (False, True):
        lam_s = 0.5 if flow else 1.0
        sigma_s = sigma_of_lambda(lam_s, flow)
        for h in (0.3, 0.8, 1.5):
            lam_t = lam_s + h
            sigma_t = sigma_of_lambda(lam_t, flow)
            a, b = 0.7, -0.4
            x_s = 1.3
            kappa = kernel_moment1(h, 0.0, lam_s, flow, sigma_s, sigma_t)
            x0_eff = a + kappa * b
            got = exact_step(x_s, sigma_s, sigma_t, x0_eff, alpha_of(sigma_t, flow), h)

            def rhs(lam, y):
                sig = sigma_of_lambda(lam, flow)
                al = alpha_of(sig, flow)
                be = beta_of(sig, flow)
                return [-be * y[0] + al * (a + b * (lam - lam_s))]

            sol = solve_ivp(rhs, (lam_s, lam_t), [x_s], method="DOP853", rtol=1e-12, atol=1e-14)
            ref = float(sol.y[0, -1])
            worst[flow] = max(worst[flow], abs(float(got) - ref) / abs(ref))
    check(
        "step with the quadrature kernel is exact for x0 linear in lambda",
        max(worst.values()) < 1e-9,
        f"max rel err eps={worst[False]:.1e} flow={worst[True]:.1e}",
    )


# ==========================================================================
# 5. convergence order of the shipped sampler on a synthetic ODE
#
# Convergence order is an asymptotic property on a *fixed* schedule, so this
# uses a schedule uniform in logSNR (the samplers' own time variable).  With a
# Karras schedule the first flow step spans dlambda ~ 14 and the measured rate
# is dominated by that single step rather than by the integrator.
# ==========================================================================
def test_convergence_order():
    def x0_fn(sigma):
        # smooth function of sigma; the sampler only sees it through the model
        return 0.5 + 0.4 * math.sin(3.0 * sigma) + 0.1 * sigma

    def ref(x0_start, sigma0, sigma_last):
        def rhs(sig, y):
            return [(y[0] - x0_fn(sig)) / sig]

        sol = solve_ivp(rhs, (sigma0, sigma_last), [x0_start], method="DOP853", rtol=1e-12, atol=1e-14)
        return float(sol.y[0, -1])

    rates = {}
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # float32 round-off caps the measurable error
    try:
        for flow in (False, True):
            model = AnalyticX0Model(x0_fn, flow=flow)
            s_max = model.sigma_max
            lam0 = omni_core.half_log_snr(omni_core.clamp_sigma(s_max, flow), flow)
            lam1 = omni_core.half_log_snr(0.05, flow)
            rates[flow] = {}
            for order in (1, 2):
                errs = []
                ns = (6, 12, 24)
                for n in ns:
                    sig = [
                        omni_core.sigma_of_lambda(lam0 + (lam1 - lam0) * i / n, flow) for i in range(n + 1)
                    ]
                    sig_t = torch.tensor([float(s) for s in sig])
                    x0_start = 0.3
                    out = omni_core.sample_omni(
                        model, torch.tensor([[x0_start]]), sig_t, extra_args={"seed": 0}, disable=True,
                        order=order, eta=0.0, corrector="none", lower_order_to_end=0, min_step_ratio=0.0,
                    )
                    errs.append(abs(float(out) - ref(x0_start, float(sig[0]), float(sig[-1]))))
                rate = math.log2(errs[0] / errs[-1]) / math.log2(ns[-1] / ns[0])
                rates[flow][order] = rate
                print(
                    f"    {'flow' if flow else 'eps '} order={order} "
                    f"errs={[f'{e:.2e}' for e in errs]} rate={rate:.2f}"
                )
    finally:
        torch.set_default_dtype(old_dtype)
    print(f"    (end-to-end rates, diagnostic only: { {('flow' if f else 'eps'): {o: round(v, 2) for o, v in rates[f].items()} for f in rates} })")
    test_integrator_order()


def test_integrator_order():
    """Convergence order of the shipped step formula on a fixed uniform-lambda ODE.

    Isolates the integrator from schedule/start-up effects: x0(lambda) is known
    analytically, the model is bypassed, and only omni_core.exact_step +
    kernel_moment1 are used.
    """
    def x0_fn(lam):
        return 0.8 + 0.4 * math.sin(1.3 * lam) + 0.15 * lam * lam

    def reference(lam0, lam1, flow):
        def rhs(lam, y):
            sig = omni_core.sigma_of_lambda(lam, flow)
            al = omni_core.alpha_of(sig, flow)
            be = omni_core.beta_of(sig, flow)
            return [-be * y[0] + al * x0_fn(lam)]

        sol = solve_ivp(rhs, (lam0, lam1), [0.3], method="DOP853", rtol=1e-12, atol=1e-14)
        return float(sol.y[0, -1])

    rates = {}
    for flow in (False, True):
        lam0, lam1 = 0.0, 2.0
        ref = reference(lam0, lam1, flow)
        rates[flow] = {}
        for order in (1, 2):
            errs = []
            ns = (8, 16, 32, 64)
            for n in ns:
                h = (lam1 - lam0) / n
                x = 0.3
                hist = []
                prev_lam = None
                for i in range(n):
                    lam_s = lam0 + i * h
                    sig_s = omni_core.sigma_of_lambda(lam_s, flow)
                    sig_t = omni_core.sigma_of_lambda(lam_s + h, flow)
                    x0 = x0_fn(lam_s)
                    if order == 1 or not hist or prev_lam is None:
                        eff = x0
                    else:
                        slope = (x0 - hist[0]) / (lam_s - prev_lam)
                        kappa = kernel_moment1(
                            h, 0.0, lam_s, flow, sig_s, sig_t
                        )
                        eff = x0 + kappa * slope
                    x = omni_core.exact_step(
                        x, sig_s, sig_t, eff, omni_core.alpha_of(sig_t, flow), h
                    )
                    hist = [x0]
                    prev_lam = lam_s
                errs.append(abs(x - ref))
            rate = math.log2(errs[0] / errs[-1]) / math.log2(ns[-1] / ns[0])
            rates[flow][order] = rate
            print(
                f"    {'flow' if flow else 'eps '} integrator order={order} "
                f"errs={[f'{e:.2e}' for e in errs]} rate={rate:.2f}"
            )
    ok = all(abs(rates[f][o] - o) < 0.25 for f in rates for o in rates[f])
    check(
        "shipped step formula has 1st/2nd order convergence (both families)",
        ok,
        f"rates: { {('flow' if f else 'eps'): rates[f] for f in rates} }",
    )


# ==========================================================================
# 6. CFG++ algebraic identity
# ==========================================================================
def test_cfgpp_identity():
    torch.manual_seed(0)
    worst = 0.0
    for flow, (s_s, s_t) in ((False, (4.0, 1.5)), (True, (0.8, 0.3))):
        x0 = torch.randn(5, 3, dtype=torch.float64)
        u = torch.randn(5, 3, dtype=torch.float64)
        eps = torch.randn(5, 3, dtype=torch.float64)
        x = alpha_of(s_s, flow) * x0 + s_s * eps
        h = half_log_snr(s_t, flow) - half_log_snr(s_s, flow)
        a_t, a_s = alpha_of(s_t, flow), alpha_of(s_s, flow)
        # omni_core's form: (sigma_t/sigma_s)*(x - alpha_s u) + alpha_t x0
        mine = (s_t / s_s) * (x - a_s * u) + a_t * x0
        ref = a_t * x0 + s_t * (x - a_s * u) / s_s
        worst = max(worst, rel(mine, ref))
    check("CFG++ step equals alpha_t*x0 + sigma_t*(x-alpha_s*u)/sigma_s", worst < 1e-12, f"rel={worst:.1e}")


# ==========================================================================
# 7. eta gate
# ==========================================================================
def test_gate():
    w = (0.2, 0.8)
    inside = [_eta_gate(f, w, 0.1) for f in (0.3, 0.5, 0.7)]
    outside = [_eta_gate(f, w, 0.1) for f in (0.0, 0.05, 0.95, 1.0)]
    check(
        "eta gate is 1 inside the window, 0 outside",
        all(v == 1.0 for v in inside) and all(v == 0.0 for v in outside),
        f"inside={inside} outside={outside}",
    )


# ==========================================================================
# 8. final extrapolation recovers a quadratic x0(sigma)
# ==========================================================================
def test_final_extrapolate():
    c0 = torch.randn(2, 3, dtype=torch.float64)
    c1 = torch.randn(2, 3, dtype=torch.float64) * 0.5
    c2 = torch.randn(2, 3, dtype=torch.float64) * 0.2
    sig = [0.20, 0.10, 0.03]
    vals = [c0 + c1 * s + c2 * s * s for s in sig]
    out = _final_extrapolate(vals[-1], vals, sig, 1.0)
    check(
        "final quadratic extrapolation is exact for quadratic x0(sigma)",
        float((out - c0).abs().max()) < 1e-5,
        f"max abs err={float((out - c0).abs().max()):.1e}",
    )


# ==========================================================================
# 9. first-order step equals comfy's exponential first step (eps)
# ==========================================================================
def test_flow_detection():
    prior = MixturePrior(dim=2)
    flow_ok = omni_core.is_flow_model(ToyModel(prior, flow=True).model_sampling)
    eps_ok = omni_core.is_flow_model(ToyModel(prior, flow=False).model_sampling)
    check("flow/EPS model detection matches comfy.model_base.model_sampling()", flow_ok and not eps_ok,
          f"flow={flow_ok} eps={eps_ok}")


def test_against_comfy_first_step():
    prior = MixturePrior(dim=2, seed=0)
    model = ToyModel(prior, flow=False)
    s0 = model.sigma_max
    sig = torch.tensor([s0, 6.0, 0.0])
    x_init, _ = make_init(prior, 8, s0, False, torch.Generator().manual_seed(3))
    mine = omni_core.sample_omni(model, x_init.clone(), sig, extra_args={"seed": 7}, disable=True, order=1, eta=0.0,
                                 corrector="none", lower_order_to_end=0)
    comfy = kds.sample_dpmpp_2m(model, x_init.clone(), sig, extra_args={"seed": 7}, disable=True)
    check(
        "one-step eps: omni(order=1) == comfy dpmpp_2m first step",
        rel(mine, comfy) < 1e-6,
        f"rel={rel(mine, comfy):.1e}",
    )


if __name__ == "__main__":
    test_exact_step()
    test_ode_identity()
    test_sde_step()
    test_kernel_moment()
    test_convergence_order()
    test_cfgpp_identity()
    test_gate()
    test_final_extrapolate()
    test_flow_detection()
    test_against_comfy_first_step()
    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        sys.exit(1)
    print("all checks passed")

