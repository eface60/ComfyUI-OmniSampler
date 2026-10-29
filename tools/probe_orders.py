"""Probe the exact-kernel multistep coefficients and their convergence order.

The one-step solution of  dx/dlambda = -beta x + alpha x0  is

    x(h) = (sigma_t/sigma_s) x_s + alpha_t * INT_0^h e^{-(1+eta)u} * beta_eff * ... x0

and, because the normalised kernel weight is e^u/(e^h-1) for BOTH eps and flow
models (checked below), the optimal average of x0 is

    x0_eff = x0 + kappa1(h_k) c1 + kappa2(h_k) c2
    kappa_j(h) = INT_0^h u^j e^u du / (e^h - 1)

with x0(lambda_s+u) = x0 + c1 u + c2 u^2 fitted through the history points.
This script (a) checks kappa against direct quadrature, (b) checks that the
kernel shape is family independent, (c) measures the convergence order of the
resulting step schemes for both families.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omni_core import alpha_of, beta_of, half_log_snr


def kappa1(h):
    if h < 1e-5:
        # INT u e^u / (e^h-1) = h/2 + h^2/12 - h^4/720 ...
        return h / 2 + h * h / 12
    E = math.expm1(h)
    return h / E + (h - 1.0)


def kappa2(h):
    if h < 1e-5:
        # h^2/3 - h^3/12 + h^5/...
        return h * h / 3 - h ** 3 / 12
    E = math.expm1(h)
    return (h * h - 2 * h) / E + (h * h - 2 * h + 2)


def kappa_quad(h, j, n=200000):
    """Reference value of INT_0^h u^j e^u du / (e^h-1) by Simpson."""
    u = np.linspace(0, h, n)
    f = (u ** j) * np.exp(u)
    val = np.trapezoid(f, u)
    return val / math.expm1(h)


def test_kappa():
    print("kappa check (h, kappa1, ref, kappa2, ref)")
    worst = 0.0
    for h in (1e-3, 0.01, 0.1, 0.7, 1.5, 3.0, 6.0):
        k1, r1 = kappa1(h), kappa_quad(h, 1)
        k2, r2 = kappa2(h), kappa_quad(h, 2)
        worst = max(worst, abs(k1 - r1) / abs(r1), abs(k2 - r2) / abs(r2))
        print(f"  h={h:<6} k1={k1:.10f} ref={r1:.10f}   k2={k2:.10f} ref={r2:.10f}")
    print(f"  worst rel err = {worst:.2e}\n")
    return worst < 1e-6


def kernel_shape(flow, h, n=4000):
    """Numerically recover the normalised kernel INT e^{-(1+eta)u}... for the ODE.

    For the homogeneous+source ODE, the weight on x0(u) is
        w(u) = exp(-INT_0^u beta) * alpha(u)
    normalised by INT_0^h w.  Compare its shape with e^u/(e^h-1).
    """
    u = np.linspace(0, h, n)
    # integrate beta along the schedule: lambda(u) = lambda_s + u
    def sigma_of(lam):
        # invert lambda( sigma ): eps -> sigma = exp(-lam); flow -> sigma = 1/(1+e^lam)
        return np.exp(-lam) if not flow else 1.0 / (1.0 + np.exp(lam))

    lam_s = 0.0 if not flow else 0.5
    lam = lam_s + u
    sig = sigma_of(lam)
    beta = np.where(flow, 1.0 - sig, 1.0) if flow else np.ones_like(sig)
    # cumulative integral of beta over lambda
    B = np.concatenate([[0.0], np.cumsum(0.5 * (beta[1:] + beta[:-1]) * np.diff(lam))])
    w = np.exp(-B) * (1.0 - sig if flow else 1.0)
    w = w / w.sum()
    ref = np.exp(u) / math.expm1(h)
    ref = ref / ref.sum()
    return float(np.abs(w - ref).max())


def oem_reference(lam0, lam1, x0_fn, flow, npts=4000):
    """High-accuracy reference for dx/dlambda = -beta x + alpha x0(lambda)."""
    def sigma_of(lam):
        return np.exp(-lam) if not flow else 1.0 / (1.0 + np.exp(lam))

    def rhs(lam, y):
        s = sigma_of(lam)
        alpha = 1.0 if not flow else 1.0 - s
        beta = 1.0 if not flow else alpha
        return -beta * y + alpha * x0_fn(lam)

    sol = solve_ivp(rhs, (lam0, lam1), [0.3], method="DOP853", rtol=1e-12, atol=1e-14)
    return float(sol.y[0, -1])


def run_scheme(n, flow, order, lam0=0.0, lam1=2.0, scheme="exact", warm=False):
    """Uniform-lambda stepping with the exact exponential step.

    ``warm`` seeds the multistep history with exact values, which isolates the
    method's asymptotic order from the (first-order) startup step.
    """
    def sigma_of(lam):
        return np.exp(-lam) if not flow else 1.0 / (1.0 + np.exp(lam))

    # x0(lambda): smooth, nonlinear
    def x0_fn(lam):
        return 0.8 + 0.4 * math.sin(1.3 * lam) + 0.15 * lam * lam

    h = (lam1 - lam0) / n
    lams = [lam0 + i * h for i in range(n + 1)]
    x = 0.3
    hist_x0 = [x0_fn(lam0 - h), x0_fn(lam0 - 2 * h)] if warm else []
    for i in range(n):
        s_s, s_t = sigma_of(lams[i]), sigma_of(lams[i + 1])
        a_t = 1.0 if not flow else 1.0 - s_t
        x0 = x0_fn(lams[i])
        if order == 1 or not hist_x0:
            eff = x0
        elif order == 2:
            if scheme == "exact":
                eff = x0 + kappa1(h) * (x0 - hist_x0[0]) / h
            else:  # dpmpp_2m coefficient
                eff = x0 + 0.5 * h * (x0 - hist_x0[0]) / h
        else:
            if len(hist_x0) < 2:
                eff = x0 + 0.5 * h * (x0 - hist_x0[0]) / h
            else:
                u1, u2 = -h, -2 * h
                d1 = (hist_x0[0] - x0) / u1
                d2 = (hist_x0[1] - x0) / u2
                c2 = (d1 - d2) / (u1 - u2)
                c1 = d1 - c2 * u1
                eff = x0 + kappa1(h) * c1 + kappa2(h) * c2
        # exact exponential step
        x = (s_t / s_s) * x + a_t * (-math.expm1(-h)) * eff
        hist_x0 = [x0] + hist_x0[:1]
    return x


def test_orders():
    print("convergence order (uniform lambda, 2.0 span)")
    ok = True
    for flow in (False, True):
        ref = oem_reference(0.0, 2.0, lambda l: 0.8 + 0.4 * math.sin(1.3 * l) + 0.15 * l * l, flow)
        for warm in (False, True):
            for order in (1, 2, 3):
                errs = []
                for n in (10, 20, 40, 80):
                    v = run_scheme(n, flow, order, warm=warm)
                    errs.append(abs(v - ref))
                rate = math.log2(errs[0] / errs[-1]) / math.log2(8.0)
                good = abs(rate - order) < 0.25
                ok = ok and (good or warm)
                print(f"  {'flow' if flow else 'eps '} order={order} warm={int(warm)} errs={[f'{e:.2e}' for e in errs]} rate={rate:.2f} {'OK' if good else 'BAD'}")
    print()
    return ok


if __name__ == "__main__":
    print("kernel shape deviation (eps vs flow, should be ~0):")
    for h in (0.5, 1.0, 2.0):
        print(f"  h={h}: eps={kernel_shape(False, h):.2e} flow={kernel_shape(True, h):.2e}")
    print()
    test_kappa()
    test_orders()
