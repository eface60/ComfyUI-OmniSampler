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
import torch.nn.functional as F

import comfy.model_patcher
import comfy.model_sampling
import comfy.utils
from comfy.k_diffusion.sampling import (
    BrownianTreeNoiseSampler,
    default_noise_sampler,
)

_ETA_EPS = 1e-6
_SIGMA_FLOOR = 1e-6

# Per-run diagnostics for the two optional spatial controls.  The node reads this
# dict after the run and writes it into the config string, so an ON arm can be
# told apart from an OFF arm without opening the image.  It is replaced (not
# cleared) at the start of each run, so a caller holding the previous dict still
# describes the run that produced it.
LAST_STATS = {}

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
# NFE skipping: epsilon extrapolation (comfyui-FSampler's mechanism)
# --------------------------------------------------------------------------
def lagrange_extrapolate(nodes, values, target):
    """Extrapolate the polynomial through (nodes_i, values_i) to ``target``.

    FSampler hard-codes the uniform coefficients (2e_n - e_{n-1}, ...); using the
    actual node positions keeps the same order on a non-uniform grid.
    """
    m = len(nodes)
    if m == 0:
        return None
    out = None
    for i in range(m):
        li = 1.0
        for j in range(m):
            if j != i:
                d = nodes[i] - nodes[j]
                if abs(d) < 1e-12:
                    return None
                li *= (target - nodes[j]) / d
        term = values[i] * li
        out = term if out is None else out + term
    return out


def extrapolate_eps(hist, target_lam, target_sigma, order, space):
    """epsilon at the current node from the last ``order`` real evaluations.

    order 1 = hold, 2 = linear (FSampler's ``2e_n - e_{n-1}``), 3 = quadratic.
    ``hist`` is newest-first [(lam, sigma, eps), ...].
    """
    if order < 1 or len(hist) < order:
        return None
    if order == 1:
        return hist[0][2]
    pts = hist[:order]
    if str(space).lower() == "sigma":
        nodes = [p[1] for p in pts]
        target = target_sigma
    else:
        nodes = [p[0] for p in pts]
        target = target_lam
    return lagrange_extrapolate(nodes, [p[2] for p in pts], target)


def valid_eps_hat(eps_hat, prev_eps, min_abs=1e-6, min_rel=1e-6, max_rel=50.0):
    """FSampler's guard rails: finite, and RMS size within [min_rel, max_rel]."""
    if eps_hat is None:
        return False
    if not bool(torch.isfinite(eps_hat).all()):
        return False
    hat = float(eps_hat.pow(2).mean().sqrt())
    if not math.isfinite(hat) or hat < float(min_abs):
        return False
    if prev_eps is not None:
        prev = float(prev_eps.pow(2).mean().sqrt())
        if prev > 0.0:
            rel = hat / prev
            if rel < float(min_rel) or rel > float(max_rel):
                return False
    return True


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


# --------------------------------------------------------------------------
# Optional spatial controls (PD-AOS ports, OFF by default)
# --------------------------------------------------------------------------
def _find_cfg_guider(model):
    """Return the CFGGuider in the wrapper chain, or None.

    ``KSAMPLER.sample`` receives the guider directly; ``comfy.samplers.sample``
    wraps it in ``KSamplerX0Inpaint``, whose ``inner_model`` is the guider.  The
    map hook reads ``cond_denoised``/``uncond_denoised``/``cond_scale`` from
    ComfyUI's own ``cfg_function``, so the object only has to be identified (to
    fail cleanly when the caller has no CFG); its scale is not changed.
    """
    cur = model
    for _ in range(4):
        if cur is None:
            return None
        if hasattr(cur, "set_cfg") and hasattr(cur, "cfg"):
            return cur
        cur = getattr(cur, "inner_model", None)
    return None


def _blur_mask(mask):
    """3x3 replicate-padded box blur for a [B,1,H,W] mask (no-op otherwise)."""
    if mask.dim() != 4:
        return mask
    x = F.pad(mask.float(), (1, 1, 1, 1), mode="replicate")
    return F.avg_pool2d(x, 3, stride=1).to(mask.dtype)


def _px_order_mask(hist, lam_mid, tol, smooth=True):
    """Per-pixel allowance in [0,1] for the order-2 x0 extrapolation.

    ``hist`` is newest-first ``[(lambda, x0), ...]`` with at least three real
    model evaluations.  The allowance is the Lagrange remainder of the LINEAR
    extrapolation through the two newest points, evaluated at ``lam_mid``:

        err(x) = |f''(x)| / 2 * prod_j |x - x_j|,  j in {lambda_s, lambda_s-1}

    ``f2`` below estimates ``f''`` as twice the second divided difference, and
    ``prod`` is the node product, so the computed ``err`` is **twice** the
    bound above.  The factor is deliberate and conservative: the divided
    difference is a point estimate of a curvature that can grow inside the
    step, and over-refusing a pixel costs only one order.  `px_order_tol` is
    therefore in units of that doubled bound.  The mask is
    ``clamp(1 - err/(tol*|x0|), 0, 1)``: a pixel at the bound gets half the
    extrapolation, and at twice the bound it runs as order 1.

    This is the Runge validity test PD-AOS uses (`order_by_bound`), reduced to
    the one order omni's exponential integrator has.  It needs no probe and no
    extra model call: the three x0 values are already in the sampler's history.
    """
    if len(hist) < 3:
        return None
    l0, x0 = hist[0]
    l1, x1 = hist[1]
    l2, x2 = hist[2]
    h1 = l0 - l1
    h2 = l1 - l2
    hs = l0 - l2
    if min(abs(h1), abs(h2), abs(hs)) < 1e-8:
        return None
    # second divided difference, times 2 -> f'' estimate
    f2 = 2.0 * ((x0 - x1) / h1 - (x1 - x2) / h2) / hs
    prod = abs((lam_mid - l0) * (lam_mid - l1))
    err = f2.abs() * prod
    dims = tuple(range(1, x0.dim()))
    ref = x0.float().pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    ratio = err.float().mean(dim=1, keepdim=True) / (float(tol) * ref)
    mask = (1.0 - ratio).clamp(0.0, 1.0)
    if smooth:
        mask = _blur_mask(mask)
    return mask.to(x0.dtype)


def _cfg_map_from_gap(cond, uncond, floor, pool=8):
    """Per-region guidance ratio from the cond/uncond gap, or None.

    The map is the magnitude of the conditional-minus-unconditional prediction,
    normalised to its own range and pooled to ``pool``x``pool``.  ``floor`` is
    the fraction of the guidance delta left in the lowest-gap region (0.85 = the
    PD-AOS shipped value), so a flat background keeps 85% of the late guidance
    while prompt-specific regions keep all of it.

    Only the SHAPE is taken from PD-AOS; the quantity is measured on the run's
    own mid-trajectory prediction instead of a separate probe, so there is no
    probe cache and no extra NFE.
    """
    if cond is None or uncond is None:
        return None
    gd = (cond.float() - uncond.float())
    if gd.dim() == 5:
        gd = gd.mean(dim=2)          # collapse frames; broadcast back later
    if gd.dim() != 4:
        return None
    pm = gd.pow(2).mean(dim=1).sqrt()          # [B,H,W], channel RMS
    if pm.numel() == 0:
        return None
    p90 = float(torch.quantile(pm.reshape(-1), 0.9))
    if p90 <= 1e-8:
        return None
    pm = (pm / p90).clamp(0.0, 1.0)
    if min(pm.shape[-2:]) > int(pool):
        pm = F.adaptive_avg_pool2d(pm.unsqueeze(1), (int(pool), int(pool)))[:, 0]
    lo = float(pm.min())
    hi = float(pm.max())
    if hi - lo <= 1e-6:
        return None
    p = (pm - lo) / (hi - lo)
    m = float(floor) + (1.0 - float(floor)) * p
    return m.unsqueeze(1)


def _apply_cfg_map(den, uncond, scale, cmap):
    """Scale the guidance delta by a per-region ratio; None/invalid = unchanged.

    `cmap` is RELATIVE (floor..1), not an absolute CFG value: the delta this
    sampler's guider produced is multiplied by it.  `scale` is only a validity
    check -- at cfg <= 1 ComfyUI hands the hook a zero unconditional prediction,
    and applying a ratio to that is not a guidance change.
    """
    if den is None or uncond is None or cmap is None:
        return den
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return den
    if scale <= 1.0 + 1e-6:
        return den
    ratio = cmap.to(device=den.device, dtype=den.dtype)
    if tuple(ratio.shape[-2:]) != tuple(den.shape[-2:]):
        ratio = F.interpolate(ratio, size=tuple(den.shape[-2:]),
                              mode="bilinear", align_corners=False)
    if den.dim() == 5 and ratio.dim() == 4:
        ratio = ratio.unsqueeze(2)               # broadcast over frames
    return uncond + (den - uncond) * ratio


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
    # NFE skipping (experimental, default off): evaluate only every k-th step
    # and reconstruct epsilon from the previous real evaluations
    skip_every=1,
    skip_order=2,
    skip_space="lam",
    skip_min_alpha=0.02,
    skip_protect_first=1,
    skip_protect_last=1,
    skip_min_abs=1e-6,
    skip_min_rel=1e-6,
    skip_max_rel=50.0,
    # ---- optional spatial controls (PD-AOS ports; both OFF by default) ----
    # px_order: let each pixel decide whether the order-2 x0 extrapolation is
    # valid for it (Lagrange remainder).  cfg_map: scale the late guidance delta
    # per region by 0.85..1.0, measured from the run's own cond/uncond gap.
    px_order=False,
    px_order_tol=2.0,
    px_order_smooth=True,
    cfg_map=False,
    cfg_map_floor=0.85,
    cfg_map_start=0.5,
    cfg_map_pool=8,
):
    """Unified exponential-integrator sampler (see module docstring).

    ``x0_average`` selects the coefficient used for the linear-in-lambda x0
    average: "exact" uses the quadrature kernel moment, "ab2" uses DPM-Solver++
    (2M)'s h/2 (the eps small-step limit).  Measured in tools/toy_bench.py.

    ``min_step_ratio`` and ``max_step_lambda`` guard the multistep extrapolation:
    if the current lambda step is more than 1/min_step_ratio times the previous
    one (or longer than max_step_lambda), the linear history fit is unreliable
    and the step falls back to first order.  Both are measured, not guessed.

    ``px_order`` and ``cfg_map`` default OFF, and OFF is bit-identical to the
    sampler without them.  When ON:

    * ``px_order`` gives the order-2 extrapolation only to pixels whose
      Lagrange remainder is under ``px_order_tol`` (relative to local |x0|).
      Pixels over the bound run as order 1.  It costs no extra model call.
    * ``cfg_map`` builds an ``cfg_map_pool`` x ``cfg_map_pool`` map from the
      cond/uncond gap at the first step after ``cfg_map_start`` of the
      trajectory, then scales the guidance delta in later steps so the
      lowest-gap region keeps ``cfg_map_floor`` of it.  It needs cfg > 1 (there
      is no unconditional prediction otherwise) and is disabled with CFG++.
    """
    global LAST_STATS
    LAST_STATS = {}
    if extra_args is None:
        extra_args = {}
    sigmas = sigmas.to(x.device)
    n_steps = len(sigmas) - 1
    if n_steps <= 0:
        return x
    n_calls = 0      # real model evaluations, for the node's diagnostics
    n_skipped = 0
    px_order = bool(px_order)
    px_order_tol = max(1e-4, float(px_order_tol))
    cfg_map = bool(cfg_map)
    cfg_map_reason = None
    if cfg_map and cfg_pp:
        cfg_map = False
        cfg_map_reason = "cfg_pp"
    if cfg_map and float(cfg_map_floor) >= 1.0 - 1e-9:
        cfg_map = False
        cfg_map_reason = "floor_is_1"
    cfg_map_floor = min(max(float(cfg_map_floor), 0.0), 1.0)
    cfg_map_start = min(max(float(cfg_map_start), 0.0), 0.99)

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

    # ---- guidance plumbing: CFG++ capture and/or the per-region CFG map ----
    # One post-CFG hook serves both.  CFG++ needs the unconditional prediction
    # for its noise estimate; the map needs both predictions at one mid-run
    # step and then scales the guidance delta on later steps.  A graph's own
    # post-CFG hooks are appended after ours, so this neither replaces nor
    # reorders them (the pd_aos stage-2 build learned that replacing the graph's
    # hook makes the feature silently inert).
    uncond_denoised = None
    cfg_state = {"map": None, "cond": None, "uncond": None, "calls": 0}
    # strength 0 makes CFG++ exactly plain sampling (u is blended away), so the
    # hook -- and with it disable_cfg1_optimization, which forces the
    # unconditional pass at cfg=1 -- must not be installed at all.
    cfgpp_active = bool(cfg_pp) and float(cfg_pp_strength) > 0.0
    if cfgpp_active or cfg_map:
        def post_cfg_function(args):
            nonlocal uncond_denoised
            u = args.get("uncond_denoised")
            if u is not None:
                uncond_denoised = u
            if not cfg_map:
                return args["denoised"]
            # At cond_scale == 1 ComfyUI's cfg1 optimisation skips the
            # unconditional pass and hands the hook a ZERO tensor, not None.
            # Building a map from `cond - 0` would scale the whole prediction;
            # refuse anything at or below 1.0 instead.
            try:
                scale = float(args.get("cond_scale", 1.0))
            except (TypeError, ValueError):
                scale = 1.0
            if scale <= 1.0 + 1e-6:
                return args["denoised"]
            cfg_state["calls"] += 1
            c = args.get("cond_denoised")
            if u is not None:
                cfg_state["uncond"] = u.detach()
            if c is not None:
                cfg_state["cond"] = c.detach()
            return _apply_cfg_map(args.get("denoised"), u, scale, cfg_state["map"])

        model_options = dict(extra_args.get("model_options", {}))
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
            model_options, post_cfg_function,
            disable_cfg1_optimization=bool(cfgpp_active),
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
        if isinstance(seed, (int, float)):
            seed = int(seed)
            # torchsde's BrownianTree feeds this to numpy's SeedSequence, which
            # rejects negative integers. Linked seed nodes can supply them even
            # though the widget has min=0, so normalise instead of crashing
            # ("ValueError: expected non-negative integer").
            if seed < 0:
                seed = seed % (2 ** 64)
        if noise_mode == "brownian" and positive and min(positive) > 0.0:
            try:
                noise_sampler = BrownianTreeNoiseSampler(x, min(positive), max(positive), seed=seed, cpu=True)
            except Exception as exc:  # degenerate interval, bad seed, ...
                print(f"[omni_hybrid] brownian tree unavailable ({exc}); falling back to white noise")
                noise_sampler = default_noise_sampler(x, seed=seed)
        else:
            noise_sampler = default_noise_sampler(x, seed=seed)

    hist_x0 = []
    hist_u = []
    eps_hist = []      # newest-first [(lam, sigma, eps)] for skip_every
    px_hist = []       # newest-first [(lam, x0)] from REAL model evaluations only
    px_steps = 0
    px_mask_sum = 0.0
    px_mask_min = None
    cfg_map_steps = 0
    cfg_map_seen = 0
    cfg_hook_calls = 0
    cfg_guider = _find_cfg_guider(model) if cfg_map else None
    if cfg_map and cfg_guider is None:
        cfg_map = False
        cfg_map_reason = "no_guider"
    prev_lam = None
    prev_denoised = None
    prev_denoised2 = None
    d_hist = []
    d_nodes = []
    # Adams-Bashforth in sigma shares the ODE dx/dsigma = (x - x0)/sigma, which
    # holds for both families, but it has no exponential linearisation and no
    # OU noise term, so it is only used for the deterministic non-CFG++ case.
    use_ab = (str(integrator).lower() == "ab") and eta <= _ETA_EPS and not cfgpp_active

    for i in comfy.utils.model_trange(n_steps, disable=disable):
        sigma_s = raw[i]
        sigma_t = raw[i + 1]
        lam_s = lams[i]
        if cfg_map and cfg_state["map"] is not None:
            cfg_map_steps += 1

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

        # ---- model evaluation (or epsilon extrapolation on a skipped step) ----
        skipped = False
        if (
            int(skip_every) > 1
            and i > 0
            and sigma_t > 0.0
            and (i % int(skip_every)) != 0
            and alpha_from > float(skip_min_alpha)
            and i >= int(skip_protect_first)
            and i <= n_steps - 1 - int(skip_protect_last)
        ):
            eps_hat = extrapolate_eps(eps_hist, lam_from, sig_from, int(skip_order), skip_space)
            prev_eps = eps_hist[0][2] if eps_hist else None
            if valid_eps_hat(eps_hat, prev_eps, skip_min_abs, skip_min_rel, skip_max_rel):
                denoised = (x - eps_hat * sig_from) / alpha_from
                skipped = True
                n_skipped += 1
            else:
                denoised = model(x, sigma_eval * s_in, **extra_args)
                n_calls += 1
                eps_hist.insert(0, (lam_from, sig_from, (x - denoised * alpha_from) / sig_from))
                del eps_hist[4:]
        else:
            denoised = model(x, sigma_eval * s_in, **extra_args)
            n_calls += 1
            eps_hist.insert(0, (lam_from, sig_from, (x - denoised * alpha_from) / sig_from))
            del eps_hist[4:]
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

        # ---- history for the two spatial controls --------------------------
        # Skipped steps reconstruct x0 from the epsilon series; they are not a
        # model measurement, so they must not enter the Runge stencil.
        if px_order and not skipped:
            px_hist.insert(0, (lam_from, denoised.detach()))
            del px_hist[3:]
        if cfg_map and cfg_state["map"] is None:
            frac = ((lam_s - lams[0]) / lam_span
                    if abs(lam_span) > 1e-12 else 1.0)
            fresh = cfg_state["calls"] > cfg_hook_calls
            if fresh:
                cfg_hook_calls = cfg_state["calls"]
            if fresh and frac >= cfg_map_start:
                cmap = _cfg_map_from_gap(cfg_state["cond"], cfg_state["uncond"],
                                         cfg_map_floor, cfg_map_pool)
                if cmap is not None:
                    cfg_state["map"] = cmap
                    cfg_map_seen = 1

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
                if px_order and len(px_hist) >= 3:
                    mask = _px_order_mask(px_hist, lam_from + kappa,
                                          px_order_tol, px_order_smooth)
                    if mask is not None:
                        x0_eff = denoised + mask * (x0_eff - denoised)
                        px_steps += 1
                        px_mask_sum += float(mask.mean())
                        _mn = float(mask.min())
                        px_mask_min = _mn if px_mask_min is None else min(px_mask_min, _mn)

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
        if cfgpp_active and u_this_step is not None:
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
        if i in corr_steps and not cfgpp_active:
            x_pred = advance(x0_eff, with_noise=False)
            x0_check = model(x_pred, sig_geo[i + 1] * s_in, **extra_args)
            n_calls += 1
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

    if cfg_map and not cfg_map_seen and cfg_map_reason is None:
        cfg_map_reason = "no_gap" if cfg_state["calls"] else "no_uncond"
    LAST_STATS.update({
        "calls": int(n_calls),
        "steps": int(n_steps),
        "skipped": int(n_skipped),
        "corrector_steps": int(len(corr_steps) if not cfgpp_active else 0),
        "px_order": bool(px_order),
        "px_order_tol": round(float(px_order_tol), 6),
        "px_order_steps": int(px_steps),
        "px_order_mean": (round(px_mask_sum / px_steps, 4) if px_steps else None),
        "px_order_min": (round(px_mask_min, 4) if px_mask_min is not None else None),
        "cfg_map": bool(cfg_map),
        "cfg_map_built": bool(cfg_map_seen),
        "cfg_map_steps": int(cfg_map_steps),
        "cfg_map_reason": cfg_map_reason,
        "cfg_map_floor": round(float(cfg_map_floor), 4),
        "cfg_map_start": round(float(cfg_map_start), 4),
        "cfg_map_region": int(cfg_map_pool),
    })
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
    skip_every=1,
    skip_order=2,
    skip_space="lam",
    skip_min_alpha=0.02,
    skip_protect_first=1,
    skip_protect_last=1,
    skip_min_abs=1e-6,
    skip_min_rel=1e-6,
    skip_max_rel=50.0,
    # Optional spatial controls, ported from PD-AOS and OFF by default.  The
    # defaults are SAFE FALLBACKS, not measured optima: `px_order_tol` 2.0 is the
    # value at which the gate was neutral on Krea2 at 8 steps (SSIM to a 40-step
    # reference 0.8617 vs 0.8620 without it; 0.05 scored 0.8477, a real loss), and
    # `cfg_map_floor` 0.85 is PD-AOS's own shipped floor.  Neither has an
    # image-preference measurement behind it yet; see PD_AOS_AUDIT.md.
    px_order=False,
    px_order_tol=2.0,
    px_order_smooth=True,
    cfg_map=False,
    cfg_map_floor=0.85,
    cfg_map_start=0.5,
    cfg_map_pool=8,
)

# A/B presets exposed by the full sampler node. Each is a delta on DEFAULTS.
AB_PRESETS = {
    "A: shipped (v1.3.1)": {},
    "B: no corrector": {"corrector": "none"},
    "C: skip every 2nd eval": {"skip_every": 2, "skip_order": 2},
    "C2: skip every 2nd, order 3": {"skip_every": 2, "skip_order": 3},
    "D: ppm gamma churn": {"churn_mode": "gamma", "s_churn": 2.0},
    "E: CFG++ mild (0.5)": {"cfg_pp": True, "cfg_pp_strength": 0.5},
    "F: CFG++ full (1.0)": {"cfg_pp": True, "cfg_pp_strength": 1.0},
    "G: order 1": {"order": 1},
    "H: exact kernel": {"x0_average": "exact"},
    "I: SDE eta 0.3": {"eta": 0.3},
    "J: no gate": {"min_step_ratio": 0.0},
    "K: per-pixel order": {"px_order": True},
    "L: region CFG map": {"cfg_map": True},
}

# Registered preset variants (each also selectable through the Sampler node).
VARIANTS = {
    "omni_hybrid": {},                       # measured defaults
    "omni_hybrid_sde": {"eta": 0.5},         # mid-trajectory stochasticity
    "omni_hybrid_cfg_pp": {"cfg_pp": True},  # CFG++ guidance
}









