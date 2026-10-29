"""Omni-Hybrid sampler: one integration engine, per-regime components.

Geometry that holds for BOTH model families ComfyUI supports
(epsilon/v-prediction and flow matching):

    x(s) = alpha(s) * x0 + sigma(s) * eps                    (model input space)
    alpha(s) = s * exp(lambda(s)),  lambda(s) = log(alpha/s)
    d x / d lambda = -beta(lambda) * x + alpha(lambda) * x0
    beta = -dlog(sigma)/dlambda        (= 1 for eps/VP, = alpha for flow)

The exact one-step solution for a locally constant x0 is

    x(t) = (sigma_t/sigma_s) * x_s + alpha_t * (1 - exp(-h)) * x0,   h = lambda_t - lambda_s,

which is *the* correct update for flow-matching models too (alpha_t(1-e^-h) =
1 - sigma_t/sigma_s for both families).  What differs between families is how
x0 is averaged over the step: the normalised kernel is

    w(u) = (sigma_t / sigma(u))^(1+eta) * alpha(u),   u = lambda - lambda_s,

which is e^u for eps models but not for flow models for large steps.  The
moments of w are integrated by Gauss-Legendre quadrature per step (16 nodes;
cost is negligible next to a model call) instead of using the eps-only
coefficient h/2 that DPM-Solver++(2M) hard-codes.  See tools/verify_math.py
for the numerical checks against closed forms and against an exact ODE solve.

Components and origin
---------------------
* exponential integrator in half-logSNR time   -> dpmpp_2m_sde / exp_heun / er_sde / seeds_2
* exact flow-matching step (alpha != 1)        -> fixes dpmpp_2m's eps-only h
* variable-step x0 averaging                   -> dpmpp_2m / res_multistep
* eta-gated SDE with the exponential update    -> dpmpp_2m_sde drift + sa_solver tau gate
* Brownian-tree correlated noise               -> dpmpp_sde / dpmpp_2m_sde
* CFG++ (unconditional prediction as noise)    -> euler_cfg_pp / res_multistep_cfg_pp
* trapezoidal corrector on selected steps      -> heun / sa_solver (PECE)
* churn at high sigma                          -> euler / heun (Karras)
* order reduction near sigma=0                 -> sa_solver
* final x0 extrapolation to sigma=0            -> cfgpp_ud10_ab
* sigma=1 / sigma=0 guards                     -> dpmpp_sde (offset_first_sigma_for_snr)

A third-order variant was implemented and measured but not shipped: on a fixed
N-step schedule the first-order start-up step dominates (2.09 measured vs 3.0
with an exact warm start), so it adds risk without benefit.  See README.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import comfy.model_patcher
import comfy.model_sampling
import comfy.utils
from comfy.k_diffusion.sampling import (
    BrownianTreeNoiseSampler,
    default_noise_sampler,
)

_ETA_EPS = 1e-6
_SIGMA_FLOOR = 1e-6

# 16-node Gauss-Legendre rule on [0, 1] for the kernel moments.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_X = (0.5 * (_GL_X + 1.0)).tolist()
_GL_W = (0.5 * _GL_W).tolist()


# --------------------------------------------------------------------------
# geometry helpers (pure python scalars; verified in tools/verify_math.py)
# --------------------------------------------------------------------------
def is_flow_model(model_sampling) -> bool:
    """True for flow-matching models (alpha = 1 - sigma) rather than eps models."""
    return isinstance(model_sampling, comfy.model_sampling.CONST)


def _find_model_sampling(model):
    """Resolve the model_sampling object through the guider/inpaint wrapper chain."""
    inner = getattr(model, "inner_model", None)
    patcher = getattr(inner, "model_patcher", None)
    if patcher is not None:
        try:
            return patcher.get_model_object("model_sampling")
        except Exception:  # noqa: BLE001 - fall through to attribute lookup
            pass
    for obj in (inner, getattr(inner, "inner_model", None), model):
        ms = getattr(obj, "model_sampling", None)
        if ms is not None:
            return ms
    return None


def alpha_of(sigma: float, flow: bool) -> float:
    return (1.0 - sigma) if flow else 1.0


def sigma_of_lambda(lam: float, flow: bool) -> float:
    """Inverse of half_log_snr."""
    if flow:
        return 1.0 / (1.0 + math.exp(lam))
    return math.exp(-lam)


def half_log_snr(sigma: float, flow: bool) -> float:
    """lambda = log(alpha / sigma). Caller must keep sigma inside (0, 1) for flow."""
    if flow:
        return math.log((1.0 - sigma) / sigma)
    return -math.log(sigma)


def beta_of(sigma: float, flow: bool) -> float:
    """-dlog(sigma)/dlambda: decay rate of the homogeneous part of the ODE."""
    return alpha_of(sigma, flow) if flow else 1.0


def kernel_moment1(h: float, eta: float, lam_s: float, flow: bool, sigma_s: float, sigma_t: float) -> float:
    """kappa1 = weighted mean of u over [0, h] under the normalised kernel.

    w(u) = (sigma_t / sigma(u))^(1+eta) * alpha(u).  For eps models this equals
    the closed form  h/expm1(h) + h - 1  (when eta=0); for flow models there is
    no elementary closed form, hence the quadrature.
    """
    if h <= 0.0:
        return 0.0
    weights = []
    us = []
    for a, wq in zip(_GL_X, _GL_W):
        u = a * h
        us.append(u)
        sig = sigma_of_lambda(lam_s + u, flow)
        alpha = alpha_of(sig, flow)
        w = (sigma_t / sig) ** (1.0 + eta) * alpha
        weights.append(w * wq)
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return 0.5 * h
    return sum(u * w for u, w in zip(us, weights)) / total


def exact_step(x, sigma_s, sigma_t, x0, alpha_t, h):
    """Exact solution of dx/dlambda = -beta x + alpha x0 for constant x0."""
    return (sigma_t / sigma_s) * x + alpha_t * (-math.expm1(-h)) * x0


def sde_step(x, sigma_s, sigma_t, x0, alpha_t, h, eta):
    """Exponential-integrator step with an OU-style noise term.

    eta=0 reproduces ``exact_step``.  For eta>0 the drift uses
    h_eta = h*(1+eta) and the state is decayed by exp(-eta*h); the injected
    noise then has exactly the variance that keeps the marginal stationary
    (eta=1 is ancestral sampling).

    Returns (x_new, noise_coefficient); the coefficient includes sigma_t.
    """
    h_eta = h * (1.0 + eta)
    x_new = (sigma_t / sigma_s) * math.exp(-eta * h) * x
    x_new = x_new + alpha_t * (-math.expm1(-h_eta)) * x0
    if eta <= 0.0:
        return x_new, 0.0
    noise_coeff = sigma_t * math.sqrt(max(0.0, -math.expm1(-2.0 * eta * h)))
    return x_new, noise_coeff


def lagrange_integral_weights(nodes, a: float, b: float):
    """Weights w_i with INT_a^b p(s) ds = sum_i w_i p(nodes_i) for any polynomial
    p of degree < len(nodes).  Used by the variable-step Adams-Bashforth mode
    (the sigma-space integrator that ipndm/deis implement with fixed tables)."""
    m = len(nodes)
    if m == 0:
        return []
    if m == 1:
        return [b - a]
    if min(abs(nodes[i] - nodes[j]) for i in range(m) for j in range(i + 1, m)) < 1e-12:
        return None  # degenerate nodes; caller falls back to order 1
    weights = [0.0] * m
    for q, wq in zip(_GL_X, _GL_W):
        s = a + (b - a) * q
        basis = []
        for i in range(m):
            li = 1.0
            for j in range(m):
                if j != i:
                    li *= (s - nodes[j]) / (nodes[i] - nodes[j])
            basis.append(li)
        for i in range(m):
            weights[i] += wq * (b - a) * basis[i]
    return weights


# --------------------------------------------------------------------------
# per-step controls
# --------------------------------------------------------------------------
def clamp_sigma(sigma: float, flow: bool) -> float:
    if flow:
        return min(max(sigma, _SIGMA_FLOOR), 1.0 - _SIGMA_FLOOR)
    return max(sigma, 1e-30)


def _eta_gate(frac: float, window, softness: float) -> float:
    """1 inside ``window``, smoothly 0 outside; ``frac`` is lambda progress."""
    lo, hi = window
    if softness <= 0.0:
        return 1.0 if lo <= frac <= hi else 0.0
    if frac < lo:
        t = (frac - (lo - softness)) / softness
    elif frac > hi:
        t = ((hi + softness) - frac) / softness
    else:
        return 1.0
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _resolve_order(order, i, n_steps, lower_order_to_end):
    o = min(int(order), i + 1)  # one previous evaluation per extra order
    # The last step (sigma_t == 0) only returns the denoiser output, so "the
    # last N steps" must mean the last N *real* integration steps.
    first_lowered = n_steps - 1 - int(lower_order_to_end)
    if lower_order_to_end > 0 and i >= first_lowered:
        o = 1
    return o


def _bounded_delta(delta, reference, max_rel=0.5):
    """Scale ``delta`` down if its RMS exceeds ``max_rel`` times that of x0."""
    d_rms = delta.pow(2).mean().sqrt()
    r_rms = reference.pow(2).mean().sqrt()
    if not torch.isfinite(d_rms) or float(d_rms) == 0.0:
        return delta
    limit = max_rel * float(r_rms)
    if limit <= 0.0:
        return delta
    scale = min(1.0, limit / float(d_rms))
    return delta * scale


def _final_extrapolate(x0, values, sigmas, weight):
    """Blend x0 with an extrapolation of x0 to sigma=0 (quadratic if possible)."""
    if weight <= 0.0:
        return x0
    pts = [
        (float(s), v)
        for s, v in zip(sigmas, values)
        if s is not None and s > 0.0 and v is not None
    ]
    if len(pts) < 2:
        return x0
    if len(pts) >= 3:
        (s2, v2), (s1, v1), (s0, v0) = pts[-3], pts[-2], pts[-1]
        d01, d02, d12 = s0 - s1, s0 - s2, s1 - s2
        if min(abs(d01), abs(d02), abs(d12)) < 1e-9:
            return x0
        w0 = s1 * s2 / (d01 * d02)
        w1 = s0 * s2 / ((s1 - s0) * d12)
        w2 = s0 * s1 / ((s2 - s0) * (s2 - s1))
        pred = w0 * v0 + w1 * v1 + w2 * v2
    else:
        (s1, v1), (s0, v0) = pts[-2], pts[-1]
        if abs(s0 - s1) < 1e-9:
            return x0
        pred = v0 - s0 * (v0 - v1) / (s0 - s1)
    return x0 + float(weight) * _bounded_delta(pred - x0, x0)


@torch.no_grad()
def sample_omni(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    order=2,
    integrator="exp",
    eta=0.0,
    s_noise=1.0,
    eta_window=(0.10, 0.90),
    eta_softness=0.10,
    lower_order_to_end=1,
    s_churn=0.0,
    churn_mode="karras",
    s_tmin=0.0,
    s_tmax=float("inf"),
    noise_mode="brownian",
    cfg_pp=False,
    cfg_pp_strength=0.5,
    corrector="first",
    final_extrapolate=0.0,
    x0_average="ab2",
    min_step_ratio=0.5,
    max_step_lambda=None,
):
    """Unified exponential-integrator sampler (see module docstring).

    ``x0_average`` selects the coefficient used for the linear-in-lambda x0
    average: "exact" uses the quadrature kernel moment, "ab2" uses DPM-Solver++
    (2M)'s h/2 (the eps small-step limit).  Measured in tools/toy_bench.py.

    ``min_step_ratio`` and ``max_step_lambda`` guard the multistep extrapolation:
    if the current lambda step is more than 1/min_step_ratio times the previous
    one (or longer than max_step_lambda), the linear history fit is unreliable
    and the step falls back to first order.  Both are measured, not guessed.
    """
    if extra_args is None:
        extra_args = {}
    sigmas = sigmas.to(x.device)
    n_steps = len(sigmas) - 1
    if n_steps <= 0:
        return x

    model_sampling = _find_model_sampling(model)
    flow = is_flow_model(model_sampling)
    s_noise = float(s_noise) * float(getattr(model_sampling, "noise_scale", 1.0))

    s_in = x.new_ones([x.shape[0]])

    raw = [float(s) for s in sigmas]
    sig_geo = [clamp_sigma(s, flow) for s in raw]
    lams = [half_log_snr(s, flow) for s in sig_geo]
    alphas = [alpha_of(s, flow) for s in sig_geo]
    lam_span = lams[n_steps - 1] - lams[0]
    if abs(lam_span) < 1e-9:
        lam_span = 1.0

    # ---- CFG++ plumbing -------------------------------------------------
    uncond_denoised = None
    if cfg_pp:
        def post_cfg_function(args):
            nonlocal uncond_denoised
            uncond_denoised = args.get("uncond_denoised")
            return args["denoised"]

        model_options = dict(extra_args.get("model_options", {}))
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
            model_options, post_cfg_function, disable_cfg1_optimization=True
        )

    # ---- corrector schedule --------------------------------------------
    # Measured: the step-0 trapezoidal corrector helps epsilon models but hurts
    # flow-matching ones (Krea2 turbo -0.05 SSIM at 10 steps, MOSS -0.025 at 10
    # and -0.016 at 16, vs +0.02 on SDXL), so "auto" resolves by family.
    corr = str(corrector).lower()
    if corr == "auto":
        corr = "none" if flow else "first"
    corr_steps = set()
    if corr in ("first", "first_last"):
        corr_steps.add(0)
    if corr in ("last", "first_last") and n_steps >= 3:
        corr_steps.add(n_steps - 2)

    # ---- noise ---------------------------------------------------------
    noise_sampler = None
    if eta > _ETA_EPS and s_noise > 0.0:
        positive = [s for s in raw if s > 0.0]
        seed = extra_args.get("seed", None)
        if noise_mode == "brownian" and positive and min(positive) > 0.0:
            noise_sampler = BrownianTreeNoiseSampler(x, min(positive), max(positive), seed=seed, cpu=True)
        else:
            noise_sampler = default_noise_sampler(x, seed=seed)

    hist_x0 = []
    hist_u = []
    prev_lam = None
    prev_denoised = None
    prev_denoised2 = None
    d_hist = []
    d_nodes = []
    # Adams-Bashforth in sigma shares the ODE dx/dsigma = (x - x0)/sigma, which
    # holds for both families, but it has no exponential linearisation and no
    # OU noise term, so it is only used for the deterministic non-CFG++ case.
    use_ab = (str(integrator).lower() == "ab") and eta <= _ETA_EPS and not cfg_pp

    for i in comfy.utils.model_trange(n_steps, disable=disable):
        sigma_s = raw[i]
        sigma_t = raw[i + 1]
        lam_s = lams[i]

        # ---- churn ----
        # "karras": relative gamma (Karras eq. 8), capped at sqrt(2)-1
        # "gamma":  comfyui-ppm's euler_gamma law - an absolute offset
        #           proportional to sigma/sigma_max, skipped when it would
        #           exceed sigma_max (which is what ppm does at the first step)
        sigma_eval = sigma_s
        if s_churn > 0.0 and s_tmin <= sigma_s <= s_tmax:
            if str(churn_mode) == "gamma":
                sigma_hat = sigma_s + s_churn * (sigma_s / max(raw[0], 1e-9))
                if sigma_t > 0.0 and sigma_hat <= raw[0]:
                    sigma_eval = sigma_hat
                    # ppm subtracts the re-noise draw; matching its sign keeps
                    # seeds interchangeable with euler_gamma
                    x = x - torch.randn_like(x) * (
                        s_noise * math.sqrt(max(0.0, sigma_eval ** 2 - sigma_s ** 2))
                    )
            else:
                gamma = min(s_churn / max(n_steps, 1), 2 ** 0.5 - 1.0)
                sigma_eval = sigma_s * (gamma + 1.0)
                if gamma > 0.0:
                    x = x + torch.randn_like(x) * (
                        s_noise * math.sqrt(max(0.0, sigma_eval ** 2 - sigma_s ** 2))
                    )

        # churn moved the state to sigma_eval, so the step must start there
        # (comfy/k-diffusion use dt = sigma_next - sigma_hat for the same reason)
        sig_from = clamp_sigma(sigma_eval, flow) if sigma_eval != sigma_s else sig_geo[i]
        lam_from = half_log_snr(sig_from, flow)
        alpha_from = alpha_of(sig_from, flow)

        # ---- model evaluation ----
        denoised = model(x, sigma_eval * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma_s,
                    "sigma_hat": sigma_eval,
                    "denoised": denoised,
                }
            )

        # ---- final step: output x0 (optionally extrapolated to sigma=0) ----
        if sigma_t <= 0.0:
            x = _final_extrapolate(
                denoised,
                [prev_denoised2, prev_denoised, denoised],
                [raw[i - 2] if i >= 2 else None, raw[i - 1] if i >= 1 else None, raw[i]],
                final_extrapolate,
            )
            break

        h = lams[i + 1] - lam_from
        if h <= 0.0 or not math.isfinite(h):
            x = denoised
            prev_denoised2, prev_denoised = prev_denoised, denoised
            prev_lam = lam_s
            continue

        order_used = _resolve_order(order, i, n_steps, lower_order_to_end)

        # ---- Adams-Bashforth in sigma (ipndm/deis style, variable step) ----
        if use_ab:
            d_cur = (x - denoised) / sig_geo[i]
            d_hist = [d_cur] + d_hist
            d_nodes = [sig_geo[i]] + d_nodes
            m = min(order_used, len(d_hist), 4)
            if min_step_ratio and m > 1 and len(d_nodes) >= 2:
                prev_step = d_nodes[1] - d_nodes[0]  # positive: sigma decreases
                cur_step = sig_geo[i] - sig_geo[i + 1]
                if cur_step > 0 and (prev_step / cur_step) < min_step_ratio:
                    m = 1
            weights = lagrange_integral_weights(d_nodes[:m], sig_geo[i], sig_geo[i + 1])
            if weights is None:
                weights, m = [sig_geo[i + 1] - sig_geo[i]], 1
            for w, dv in zip(weights, d_hist[:m]):
                x = x + w * dv
            hist_x0 = [denoised] + hist_x0[:1]
            prev_denoised2, prev_denoised = prev_denoised, denoised
            prev_lam = lam_s
            if len(d_hist) > 4:
                d_hist, d_nodes = d_hist[:4], d_nodes[:4]
            continue

        # ---- x0 averaged over the step (exact kernel moments) ----
        eta_i = 0.0
        if eta > _ETA_EPS:
            frac = (lam_s - lams[0]) / lam_span
            eta_i = eta * _eta_gate(frac, eta_window, eta_softness)

        x0_eff = denoised
        kappa = 0.0
        h_prev = 0.0
        extrapolate = False
        if order_used >= 2 and hist_x0 and prev_lam is not None:
            h_prev = lam_from - prev_lam
            # stability gate: a much longer current step than the previous one
            # makes the linear history fit unreliable (amplification ~ h/h_prev)
            unstable = h_prev <= 0.0
            if min_step_ratio and min_step_ratio > 0.0:
                unstable = unstable or (h_prev / h) < min_step_ratio
            if max_step_lambda is not None:
                unstable = unstable or h > float(max_step_lambda)
            if unstable:
                order_used = 1
            else:
                extrapolate = True
                if x0_average == "exact":
                    kappa = kernel_moment1(h, eta_i, lam_from, flow, sig_from, sig_geo[i + 1])
                else:
                    kappa = 0.5 * h * (1.0 + eta_i)
                x0_eff = denoised + kappa * (denoised - hist_x0[0]) / h_prev

        # ---- guidance: CFG++ uses the unconditional prediction as noise ----
        # x_t = (sigma_t/sigma_s) * (x - alpha_s*u) + alpha_t * x0
        # The multistep average must be applied to u as well: otherwise the
        # difference (kappa*dx0 vs kappa*du) is injected as extra guidance, and
        # CFG++ stops being a no-op when cond == uncond.
        #
        # cfg_pp_strength interpolates the noise estimate between the guided
        # prediction (0 = plain sampling, exact) and the unconditional one
        # (1 = faithful CFG++, identical to comfy's euler_cfg_pp).
        u_this_step = uncond_denoised
        if cfg_pp and u_this_step is not None:
            s_u = float(cfg_pp_strength)
            s_u = 0.0 if s_u < 0.0 else (1.0 if s_u > 1.0 else s_u)
            if s_u < 1.0:
                u_this_step = denoised + s_u * (u_this_step - denoised)
            u_eff = u_this_step
            if extrapolate and hist_u:
                u_eff = u_this_step + kappa * (u_this_step - hist_u[0]) / h_prev
            x_base = x - alpha_from * u_eff
            coef_x0 = 1.0
        else:
            x_base = x
            coef_x0 = -math.expm1(-h * (1.0 + eta_i))

        decay = (sig_geo[i + 1] / sig_from) * math.exp(-eta_i * h)
        noise_coeff = (
            sig_geo[i + 1] * math.sqrt(max(0.0, -math.expm1(-2.0 * eta_i * h)))
            if eta_i > 0.0
            else 0.0
        )

        def advance(x0_term, with_noise):
            out = decay * x_base + alphas[i + 1] * coef_x0 * x0_term
            if with_noise and noise_coeff > 0.0 and s_noise > 0.0:
                out = out + noise_coeff * s_noise * noise_sampler(sigmas[i], sigmas[i + 1])
            return out

        # ---- optional trapezoidal corrector (one extra NFE) ----
        # Not used with CFG++: the corrector refines the guided x0 but cannot
        # re-evaluate the unconditional prediction at the predicted state, so it
        # would shift the effective guidance.
        if i in corr_steps and not cfg_pp:
            x_pred = advance(x0_eff, with_noise=False)
            x0_check = model(x_pred, sig_geo[i + 1] * s_in, **extra_args)
            x0_eff = 0.5 * (x0_eff + x0_check)
            # the extra evaluation must not leak its (later) uncond into history
            uncond_denoised = u_this_step

        x = advance(x0_eff, with_noise=True)

        # ---- history ----
        hist_x0 = [denoised] + hist_x0[:1]
        if u_this_step is not None:
            hist_u = [u_this_step] + hist_u[:1]
        prev_denoised2, prev_denoised = prev_denoised, denoised
        prev_lam = lam_from

    return x


DEFAULTS = dict(
    order=2,
    integrator="exp",
    eta=0.0,
    s_noise=1.0,
    eta_window=(0.10, 0.90),
    eta_softness=0.10,
    lower_order_to_end=1,
    s_churn=0.0,
    churn_mode="karras",
    s_tmin=0.0,
    s_tmax=float("inf"),
    noise_mode="brownian",
    cfg_pp=False,
    cfg_pp_strength=0.5,
    corrector="first",
    final_extrapolate=0.0,
    x0_average="ab2",
    min_step_ratio=0.5,
    max_step_lambda=None,
)

# Registered preset variants (each also selectable through the Sampler node).
VARIANTS = {
    "omni_hybrid": {},                       # measured defaults
    "omni_hybrid_sde": {"eta": 0.5},         # mid-trajectory stochasticity
    "omni_hybrid_cfg_pp": {"cfg_pp": True},  # CFG++ guidance
}








