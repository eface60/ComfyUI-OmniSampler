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
* Brownian-tree noise (same draw as white)    -> dpmpp_sde / dpmpp_2m_sde
* CFG++ (unconditional prediction as noise)    -> euler_cfg_pp / res_multistep_cfg_pp
* trapezoidal corrector on selected steps      -> heun / sa_solver (PECE)
* churn at high sigma                          -> euler / heun (Karras)
* order reduction near sigma=0                 -> sa_solver
* final x0 extrapolation to sigma=0            -> cfgpp_ud10_ab
* sigma=1 / sigma=0 guards                     -> dpmpp_sde (offset_first_sigma_for_snr)

A third-order variant was implemented and measured but not shipped: on a fixed
N-step schedule the first-order start-up step dominates (2.09 measured vs 3.0
with an exact warm start), so it adds risk without benefit.  The `order` widget
therefore clamps 3-4 to 2 on the exp path and records the value in force in
LAST_STATS; `ab` still uses the full history.  See tools/probe_orders.py.
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


def _churn_forward(x, sigma_s, sigma_hat, flow, s_noise, sign=1.0):
    """Re-noise ``x`` from p(sigma_s) to p(sigma_hat): the exact forward step.

    In (alpha, sigma) coordinates the forward process is

        x_hat = r * x + sqrt(sigma_hat^2 - (r*sigma_s)^2) * z,
        r = alpha_hat / alpha_s,

    which reduces to k-diffusion's additive ``x + sqrt(sigma_hat^2 - sigma_s^2) z``
    when ``alpha = 1`` (epsilon models), so the eps path is bit-identical.  For
    flow models ``alpha = 1 - sigma`` and the additive form leaves the x0
    component unscaled: the variance still comes out right (both forms give
    sigma_hat^2) but the mean does not, and the step after the churn inherits
    that bias.  Measured on a constant-x0 toy at eta = 1: mean deviation 7.2% of
    sigma_t with the additive form, 0.2% with this one
    (``tools/test_churn_eta.py``).

    Returns ``(x, fired)``; ``fired`` is False when the move is degenerate
    (sigma_hat <= sigma_s, or alpha_s <= 0, which a flow model reaches only at
    sigma = 1 where the cap already suppresses churn).
    """
    if sigma_hat <= sigma_s:
        return x, False
    if flow:
        a_s = 1.0 - sigma_s
        a_h = 1.0 - sigma_hat
        if a_s <= 1e-12:
            return x, False
        ratio = a_h / a_s
        var = sigma_hat * sigma_hat - (ratio * sigma_s) ** 2
    else:
        ratio = 1.0
        var = sigma_hat * sigma_hat - sigma_s * sigma_s
    diff = s_noise * math.sqrt(max(0.0, var))
    return ratio * x + sign * torch.randn_like(x) * diff, True


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
    # The exp path keeps only the one previous x0 (`hist_x0`), so callers pass
    # order <= 2 there; `ab` keeps up to four derivatives.
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
    Under CFG++ the caller applies the returned mask to the unconditional
    extrapolation as well; a mask on x0 alone would break the no-op invariant
    (docs/TECHNICAL.md, "CFG++ and what cfg means").
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


# --------------------------------------------------------------------------
# Guidance-gap shaping (experimental, OFF by default)
# --------------------------------------------------------------------------
# The guidance delta -- the conditional minus the unconditional prediction, in
# x0 space -- is the only thing CFG adds to a step.  It is the same object for
# every model family (an eps model and a flow model both hand the post-CFG hook
# a denoised prediction) and for every guidance path (plain cfg > 1, cfg = 1
# with the unconditional pass forced, and CFG++), so shaping it applies to all
# of them at once instead of patching one path.  See
# docs/EXPERIMENT_GAP_SHAPING.md for the measurements that motivate it.
#
# Every knob is neutral at its default (0, 0, 0, 1), and a fully neutral set
# installs no hook at all, so "off" is bit-identical to the sampler without it.
_GAP_EPS = 1e-8
# Width of the local gate in units of the reference push: a pixel pushing 35%
# harder than the reference gets the full local relief.
_GAP_LOCAL_SOFT = 0.35
# Where the gate starts: the top tenth of the image's contrast push.  This is a
# tolerance (how much of the frame may be relieved), not a strength.
_GAP_LOCAL_TAIL = 0.90


def _gap_4d(t):
    """(flat 4-D view, restore) for a 4-D or 5-D latent, else (None, None)."""
    if t.dim() == 4:
        return t, (lambda y: y)
    if t.dim() == 5:
        b, c, tt, h, w = t.shape
        return t.reshape(b * tt, c, h, w), (lambda y: y.reshape(b, c, tt, h, w))
    return None, None


def _gap_lowpass(x, frac):
    """Low-pass of a 4-D float tensor; band width ``frac`` of the short side.

    The band is expressed as a fraction of the latent's short side so the same
    value means the same thing at 1024 and at 2048.  ``band -> 0`` is the global
    per-channel mean, so a very small ``gap_lf`` removes only the tone
    (exposure, colour) and leaves every structure in the delta.
    """
    h, w = x.shape[-2:]
    band = float(frac) * min(h, w)
    if band <= 1.0:
        return x.mean(dim=(-2, -1), keepdim=True).expand_as(x)
    f = max(2, int(round(band)))
    small = F.interpolate(x, size=(max(1, h // f), max(1, w // f)), mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def _gap_pixel_rms(x):
    """Per-pixel channel RMS of a 4-D tensor -> [B,1,H,W]."""
    return x.pow(2).mean(dim=1, keepdim=True).sqrt()


def _gap_project(delta, cond, keep):
    """Keep ``keep`` of the delta's component along cond (per pixel).

    The component parallel to the conditional prediction is the one that scales
    it up or down, i.e. the contrast/exposure part of the guidance; the
    orthogonal component carries the prompt-specific structure.  ``keep = 1``
    leaves the delta untouched, ``0`` removes the parallel part (APG-style).
    ``keep`` may be a scalar or a per-pixel [B,1,H,W] map (see
    ``_gap_local_gate``).
    """
    denom = cond.pow(2).sum(dim=1, keepdim=True).clamp_min(_GAP_EPS)
    par = (delta * cond).sum(dim=1, keepdim=True) / denom * cond
    return par * keep + (delta - par)


def _gap_local_gain(frac, start):
    """Relief ramp: 0 before ``start`` of the trajectory, 1 after it.

    The local gate is recomputed every step from that step's own push
    distribution.  Early in a run the top decile is not highlights, it is
    composition-sized regions -- relieving contrast there fights the structure
    the model is still laying down, which is exactly the "colour blobs that get
    resolved later" failure.  Late in a run the same top decile is the clipping
    tail, which is what the relief is for.  ``start`` is a trajectory fraction
    (logSNR progress), so it is schedule- and resolution-independent.
    """
    start = float(start)
    if start <= 0.0:
        return 1.0
    t = (float(frac) - start) / max(1.0 - start, 1e-6)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gap_local_gate(delta, cond, softness=_GAP_LOCAL_SOFT, tail=_GAP_LOCAL_TAIL):
    """Per-pixel gate that fires where the contrast push is locally excessive.

    Borrowed from PD-AOS's local fields, whose one transferable idea is the
    *normalisation*: no absolute threshold, a per-pixel quantity compared against
    a reference measured from the run itself, and a smooth gate instead of a
    mask.  Here the quantity is the signed relative push of the parallel
    component, ``(delta . cond) / ||cond||^2`` (positive = the guidance is
    pushing this pixel brighter / more contrast), and the reference is the
    ``tail`` quantile of that push over the image at this step.

    A *tail* reference rather than a mean or a block mean: a block cannot see a
    region that is uniformly pushed (its mean rises with it), and a whole-image
    excess is the job of the global ``gap_parallel``.  What this catches is the
    pixels pushing hardest relative to the rest of the frame -- the highlights
    that clip -- while every pixel under the tail is left untouched.

    Returns ``(gate, ref)`` with gate in [0, 1] and ref the quantile used.
    """
    denom = cond.pow(2).sum(dim=1, keepdim=True).clamp_min(_GAP_EPS)
    push = (delta * cond).sum(dim=1, keepdim=True) / denom
    q = torch.quantile(push.reshape(push.shape[0], -1).float(), float(tail), dim=1)
    q = q.view(-1, 1, 1, 1).to(push.dtype)
    scale = q.abs().clamp_min(1e-6)
    t = ((push - q) / (float(max(softness, 1e-3)) * scale)).clamp(0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    return _blur_mask(gate), float(q.mean())


def _gap_cap(delta, uncond, cap):
    """Per-pixel cap ``|delta| <= cap * |uncond|``, direction preserved."""
    limit = (float(cap) * _gap_pixel_rms(uncond)).clamp_min(_GAP_EPS)
    scale = (limit / _gap_pixel_rms(delta).clamp_min(_GAP_EPS)).clamp(max=1.0)
    return delta * scale, scale


def shape_gap(delta, uncond, cond=None, lf=0.0, lf_keep=0.0, cap=0.0, parallel=1.0,
              local=0.0):
    """Shape a guidance delta: project -> split -> cap, per pixel.

    ``lf`` is the low-pass band width as a fraction of the latent's short side
    (0 = off) and ``lf_keep`` the fraction of that band that is kept, so
    ``lf=0.05, lf_keep=0`` moves everything below 5% of the short side from the
    conditional prediction to the unconditional one: tone from u, structure from
    c.  ``cap`` bounds the per-pixel delta to that multiple of the local
    unconditional magnitude.  ``parallel`` keeps that fraction of the delta's
    component along c (1 = unchanged).  ``local`` additionally removes up to that
    fraction of the parallel component, but only where the guidance is locally
    excessive (``_gap_local_gate``): full contrast where it is normal, relief
    where it would burn.  Returns ``(delta, info)`` with the mean applied cap
    scale and local gate in ``info`` when either fired.
    """
    d4, restore = _gap_4d(delta)
    if d4 is None:
        return delta, {}
    u4, _ = _gap_4d(uncond) if uncond is not None else (None, None)
    c4, _ = _gap_4d(cond) if cond is not None else (None, None)
    dtype = d4.dtype
    d = d4.float()
    info = {}
    if (parallel < 1.0 or local > 0.0) and c4 is not None:
        if local > 0.0:
            gate, ref = _gap_local_gate(d, c4.float())
            keep = (float(parallel) - float(local) * gate).clamp(0.0, 1.0)
            info["local_mean"] = float(gate.mean())
            info["local_frac"] = float((gate > 0.5).float().mean())
            info["local_ref"] = ref
        else:
            keep = float(parallel)
        d = _gap_project(d, c4.float(), keep)
    if lf > 0.0 and lf_keep < 1.0:
        d = d - (1.0 - float(lf_keep)) * _gap_lowpass(d, lf)
    if cap > 0.0 and u4 is not None:
        d, scale = _gap_cap(d, u4.float(), cap)
        info["cap_scale"] = float(scale.mean())
    return restore(d.to(dtype)), info


# Low-pass band ladder used by gap_metrics: fractions of the latent's short
# side.  The profile (how much of the delta's energy sits below each band) is
# the measurement that decides whether a fixed band is meaningful across models.
_GAP_LF_BANDS = (0.01, 0.02, 0.05, 0.10, 0.20)


def gap_metrics(delta, uncond, cond=None, band=0.05):
    """Scale-free summary of a guidance delta, for the resolution diagnostic.

    ``rms_ratio`` = ``||delta|| / ||uncond||`` (magnitude; note the post-CFG
    delta is ``cfg * (cond - uncond)``, so divide by cfg to compare across cfg
    values), ``dc_ratio`` = the delta's per-channel spatial mean over the mean
    ``|uncond|`` (the tone/exposure part), ``p99_ratio`` = the 99th percentile of
    the per-pixel ``|delta| / |uncond|`` (the tail a cap would bite), ``lf_share``
    = energy fraction below ``band``, ``par_share`` = energy fraction along the
    conditional prediction, and ``lf_share_XX`` the same for every band in
    ``_GAP_LF_BANDS``.  All are means over the batch, so they are comparable
    across resolutions and model families.
    """
    d4, _ = _gap_4d(delta)
    if d4 is None:
        return {}
    u4, _ = _gap_4d(uncond) if uncond is not None else (None, None)
    c4, _ = _gap_4d(cond) if cond is not None else (None, None)
    d = d4.float()
    dims = (1, 2, 3)
    energy = d.pow(2).mean(dim=dims).clamp_min(_GAP_EPS)      # [B], for RMS
    # Energy fractions use SUMS: the projected/`lp` tensors do not have the same
    # element count as `d` (a projection keeps 1 channel), so a ratio of means
    # would be off by the channel count.
    d_energy = d.pow(2).sum(dim=dims).clamp_min(_GAP_EPS)     # [B]
    out = {}
    if u4 is not None:
        u = u4.float()
        u_e = u.pow(2).mean(dim=dims).clamp_min(_GAP_EPS)
        out["rms_ratio"] = float((energy.sqrt() / u_e.sqrt()).mean())
        # Absolute latent-space magnitudes: a shrinking *ratio* does not imply a
        # shrinking impulse if the prediction itself grows.
        out["u_rms"] = float(u_e.sqrt().mean())
        out["d_rms"] = float(energy.sqrt().mean())
        d_dc = d.mean(dim=(-2, -1)).abs().mean(dim=1)                  # [B]
        u_abs = u.abs().mean(dim=(-2, -1)).mean(dim=1).clamp_min(_GAP_EPS)
        out["dc_ratio"] = float((d_dc / u_abs).mean())
        ratio = _gap_pixel_rms(d) / _gap_pixel_rms(u).clamp_min(_GAP_EPS)
        if ratio.numel() > 0:
            q = torch.quantile(ratio.reshape(ratio.shape[0], -1), 0.99, dim=1)
            out["p99_ratio"] = float(q.mean())
    for b in _GAP_LF_BANDS:
        lp = _gap_lowpass(d, b)
        out[f"lf_share_{int(round(b * 100)):02d}"] = float(
            (lp.pow(2).sum(dim=dims) / d_energy).mean())
    if band not in _GAP_LF_BANDS:
        lp = _gap_lowpass(d, band)
        out["lf_share_auto"] = float((lp.pow(2).sum(dim=dims) / d_energy).mean())
    key = f"lf_share_{int(round(band * 100)):02d}"
    out["lf_share"] = out.get(key, out.get("lf_share_auto", 0.0))
    if c4 is not None:
        c = c4.float()
        denom = c.pow(2).sum(dim=1, keepdim=True).clamp_min(_GAP_EPS)
        par = (d * c).sum(dim=1, keepdim=True) / denom.sqrt()
        out["par_share"] = float((par.pow(2).sum(dim=dims) / d_energy).mean())
        # The signed relative push the local gate works with, so its tail can be
        # read directly: the median (the frame's normal push) and p90 (where the
        # gate starts).  Their ratio is the tail excess the relief works against.
        push = (d * c).sum(dim=1, keepdim=True) / denom
        flat = push.reshape(push.shape[0], -1).float()
        out["push_med"] = float(torch.quantile(flat, 0.5, dim=1).mean())
        out["push_p90"] = float(torch.quantile(flat, 0.9, dim=1).mean())
    return out


def gap_summary(series):
    """Mean/first/last/max of every metric over a ``[(sigma, metrics), ...]``."""
    keys = sorted({k for _, m in series for k, v in m.items()
                   if isinstance(v, (int, float))})
    out = {}
    for key in keys:
        vals = [m[key] for _, m in series if key in m]
        if not vals:
            continue
        out[key + "_mean"] = float(sum(vals) / len(vals))
        out[key + "_first"] = float(vals[0])
        out[key + "_last"] = float(vals[-1])
        out[key + "_max"] = float(max(vals))
    return out


def _gap_oor(den, uncond, k=3.0):
    """Fraction of the guided prediction beyond ``k`` unconditional sigmas.

    A cheap, self-normalising burn sensor: the unconditional prediction is the
    prior, so its per-channel RMS at this step is the natural unit in which "this
    x0 is outside the range the model itself produces" can be measured.  No
    quantile (too slow at 6 MP): the threshold is ``k * u_rms`` per channel,
    recomputed every step, so the number is comparable across models and
    resolutions.
    """
    d4, _ = _gap_4d(den)
    u4, _ = _gap_4d(uncond)
    if d4 is None or u4 is None:
        return None
    uu = u4.float()
    rms = uu.pow(2).mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(_GAP_EPS)
    return float((d4.float().abs() > k * rms).float().mean())


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
    churn_window=(0.0, 1.0),
    s_tmin=0.0,
    s_tmax=float("inf"),
    noise_mode="white",
    cfg_pp=False,
    cfg_pp_strength=0.5,
    corrector="first",
    corrector_cfgpp=False,
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
    # ---- experimental: guidance-gap shaping (all model families, all paths) ----
    gap_lf=0.0,
    gap_lf_keep=0.0,
    gap_cap=0.0,
    gap_parallel=1.0,
    gap_parallel_local=0.0,
    gap_local_start=0.0,
    gap_stats=False,
):
    """Unified exponential-integrator sampler (see module docstring).

    ``x0_average`` selects the coefficient used for the linear-in-lambda x0
    average: "exact" uses the quadrature kernel moment, "ab2" uses DPM-Solver++
    (2M)'s h/2 (the eps small-step limit).  Measured in tools/toy_bench.py.

    ``min_step_ratio`` and ``max_step_lambda`` guard the multistep extrapolation:
    if the current lambda step is more than 1/min_step_ratio times the previous
    one (or longer than max_step_lambda), the linear history fit is unreliable
    and the step falls back to first order.  Both are measured, not guessed.

    ``order`` above 2 is clamped to 2 on the ``exp`` path: the third-order fit
    was measured and removed (docs/ANALYSIS.md), and only ``ab`` keeps history
    beyond the two newest nodes.

    ``corrector_cfgpp`` allows the trapezoidal corrector under CFG++: the extra
    evaluation also produces the unconditional prediction at the next node, and
    both x0 and u are averaged with it, so the ``cond == uncond`` no-op
    invariant survives.  OFF (the default) keeps the v1.0 rule: the corrector is
    skipped under CFG++, because correcting only x0 would shift the guidance.

    ``px_order`` and ``cfg_map`` default OFF, and OFF is bit-identical to the
    sampler without them.  When ON:

    * ``px_order`` gives the order-2 extrapolation only to pixels whose
      Lagrange remainder is under ``px_order_tol`` (relative to local |x0|).
      Pixels over the bound run as order 1.  It costs no extra model call.
      Under CFG++ the same mask is applied to the unconditional extrapolation:
      the two histories must receive the same average or CFG++ stops being a
      no-op when cond == uncond.
    * ``cfg_map`` builds an ``cfg_map_pool`` x ``cfg_map_pool`` map from the
      cond/uncond gap at the first step after ``cfg_map_start`` of the
      trajectory, then scales the guidance delta in later steps so the
      lowest-gap region keeps ``cfg_map_floor`` of it.  It needs cfg > 1 (there
      is no unconditional prediction otherwise) and is disabled with CFG++.

    ``gap_lf``, ``gap_lf_keep``, ``gap_cap``, ``gap_parallel`` and
    ``gap_parallel_local`` shape the guidance delta (the conditional minus the
    unconditional prediction) in x0 space.  That delta is the same object for
    both model families and for every guidance path -- plain sampling at
    cfg > 1, cfg = 1 with the unconditional pass forced, and CFG++ -- so one
    shaping layer covers all of them.  ``gap_parallel_local`` is the local
    variant of ``gap_parallel``: it removes the contrast component only where
    the guidance is locally excessive, with the "excessive" reference measured
    per step from the run itself (no absolute threshold).  The defaults
    (0, 0, 0, 1, 0) are neutral and install no hook, so "off" is bit-identical
    to the sampler without the feature; see docs/EXPERIMENT_GAP_SHAPING.md for
    why the knobs exist and what the cross-model sweep measured.

    ``gap_stats`` records the cond/uncond gap per step (``rms_ratio``,
    ``lf_share``, ``par_share``) in ``LAST_STATS`` without changing the output.
    At cfg = 1 it forces the unconditional pass (2x batch) because there is no
    gap to measure otherwise.
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
    n_churn = 0      # steps that really injected churn noise
    n_churn_capped = 0  # steps where the flow-domain cap reduced sigma_hat
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

    # Churn window, as step fractions (0 = first step, 1 = last).  A fraction is
    # portable across model families and schedulers; a sigma offset is not (the
    # flow domain is 0..1, an eps schedule runs to 14.6), which is the same
    # absolute-threshold trap the gap sweep documented.
    try:
        cw_lo, cw_hi = (float(v) for v in churn_window)
    except (TypeError, ValueError):
        cw_lo, cw_hi = 0.0, 1.0
    cw_lo = min(max(cw_lo, 0.0), 1.0)
    cw_hi = min(max(cw_hi, 0.0), 1.0)
    if cw_hi < cw_lo:
        cw_lo, cw_hi = cw_hi, cw_lo
    n_churn_windowed = 0

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

    # ---- guidance plumbing: CFG++ capture, the CFG map, gap shaping ---------
    # One post-CFG hook serves all three.  CFG++ needs the unconditional
    # prediction for its noise estimate; the map needs both predictions at one
    # mid-run step and then scales the guidance delta on later steps; the gap
    # shaping rewrites that same delta.  A graph's own post-CFG hooks are
    # appended after ours, so this neither replaces nor reorders them (the
    # pd_aos stage-2 build learned that replacing the graph's hook makes the
    # feature silently inert).
    gap_lf = min(max(float(gap_lf), 0.0), 0.5)
    gap_lf_keep = min(max(float(gap_lf_keep), 0.0), 1.0)
    gap_cap = max(float(gap_cap), 0.0)
    gap_parallel = min(max(float(gap_parallel), 0.0), 1.0)
    gap_parallel_local = min(max(float(gap_parallel_local), 0.0), 1.0)
    gap_local_start = min(max(float(gap_local_start), 0.0), 0.95)
    gap_stats = bool(gap_stats)
    gap_shape_on = ((gap_lf > 0.0 and gap_lf_keep < 1.0)
                    or gap_cap > 0.0 or gap_parallel < 1.0
                    or gap_parallel_local > 0.0)
    gap_active = gap_shape_on or gap_stats
    # Step bookkeeping for the late-only local relief: the hook needs to know
    # which step it is on, and the loop is the only place that knows.
    gap_step = {"i": 0, "n": int(n_steps)}
    gap_series = []
    prev_den_for_rev = None
    gap_seen = 0
    gap_skipped = 0
    gap_local_sum = 0.0
    gap_local_frac_sum = 0.0
    gap_local_steps = 0
    gap_local_ref = None
    gap_cap_sum = 0.0
    gap_cap_steps = 0
    gap_reason = None
    uncond_denoised = None
    cfg_state = {"map": None, "cond": None, "uncond": None, "calls": 0}
    # The corrector's extra evaluation also runs the model (and therefore the
    # post-CFG hook) at the NEXT node's sigma.  It is a model call but not a
    # trajectory step: counting it would put a duplicate sigma in gap_series,
    # inflate every gap mean, and make `rev` dip (the two entries share a
    # sigma).  The hook reads this flag; see docs/EXPERIMENT_GAP_SHAPING.md.
    call_ctx = {"corrector": False}
    # strength 0 makes CFG++ exactly plain sampling (u is blended away), so the
    # hook -- and with it disable_cfg1_optimization, which forces the
    # unconditional pass at cfg=1 -- must not be installed at all.
    cfgpp_active = bool(cfg_pp) and float(cfg_pp_strength) > 0.0
    if cfgpp_active or cfg_map or gap_active:
        def post_cfg_function(args):
            nonlocal uncond_denoised, gap_seen, gap_skipped, gap_cap_sum, gap_cap_steps, prev_den_for_rev
            nonlocal gap_local_sum, gap_local_frac_sum, gap_local_steps, gap_local_ref
            u = args.get("uncond_denoised")
            den = args["denoised"]
            is_corr = call_ctx["corrector"]
            if u is not None:
                uncond_denoised = u
            if not (cfg_map or gap_active):
                return den
            # At cond_scale == 1 ComfyUI's cfg1 optimisation skips the
            # unconditional pass and hands the hook a ZERO tensor, not None.
            # A map or a delta built from it would be meaningless (a zero
            # "uncond" turns the whole prediction into a delta), so refuse.
            if u is None or not bool(torch.any(u != 0)):
                if not is_corr:
                    gap_skipped += 1
                return den
            if not is_corr:
                gap_seen += 1
            try:
                scale = float(args.get("cond_scale", 1.0))
            except (TypeError, ValueError):
                scale = 1.0
            c = args.get("cond_denoised")
            if cfg_map and scale > 1.0 + 1e-6:
                if not is_corr:
                    cfg_state["calls"] += 1
                    if c is not None:
                        cfg_state["cond"] = c.detach()
                    cfg_state["uncond"] = u.detach()
                den = _apply_cfg_map(den, u, scale, cfg_state["map"])
            if not gap_active:
                return den
            delta = den - u
            m = None
            sigma_now = 0.0
            if gap_stats and not is_corr:
                try:
                    sigma_now = float(torch.as_tensor(args.get("sigma", 0.0)).reshape(-1)[0])
                except (TypeError, ValueError):
                    sigma_now = 0.0
                band = gap_lf if gap_lf > 0.0 else 0.05
                m = gap_metrics(delta, u, c, band)
            if gap_shape_on:
                local_eff = gap_parallel_local
                if local_eff > 0.0 and gap_local_start > 0.0:
                    # Step fraction, not logSNR progress: the extreme sigmas eat
                    # most of the lambda span, so "30% of logSNR" is step 2 of 12
                    # -- not what "early in the generation" means.
                    frac_now = gap_step["i"] / max(gap_step["n"] - 1, 1)
                    local_eff *= _gap_local_gain(frac_now, gap_local_start)
                shaped, info = shape_gap(delta, u, c, gap_lf, gap_lf_keep,
                                         gap_cap, gap_parallel, local_eff)
                if "cap_scale" in info and not is_corr:
                    gap_cap_sum += info["cap_scale"]
                    gap_cap_steps += 1
                    if m is not None:
                        m["cap_scale"] = info["cap_scale"]
                if "local_mean" in info and not is_corr:
                    gap_local_sum += info["local_mean"]
                    gap_local_frac_sum += info["local_frac"]
                    gap_local_steps += 1
                    gap_local_ref = info["local_ref"]
                    if m is not None:
                        m["local_mean"] = info["local_mean"]
                        m["local_frac"] = info["local_frac"]
                den = u + shaped
            if m is not None and not is_corr:
                # Sensors, measured on what the step will actually use.  `rev`
                # is the model's own revision rate: high while it is still
                # deciding structure, low once the frame has settled.  `oor` is
                # the burn sensor: how much of the guided prediction is outside
                # the prior's own range.  Both are relative, so they are
                # comparable across models, resolutions and schedules.
                if prev_den_for_rev is not None:
                    m["rev"] = float((den - prev_den_for_rev).norm()
                                     / den.norm().clamp_min(_GAP_EPS))
                prev_den_for_rev = den.detach()
                oor = _gap_oor(den, u)
                if oor is not None:
                    m["oor"] = oor
                gap_series.append((sigma_now, m))
            return den

        model_options = dict(extra_args.get("model_options", {}))
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
            model_options, post_cfg_function,
            # gap shaping/stats needs the same forced unconditional pass at
            # cfg = 1 that CFG++ needs: without it there is no delta to shape or
            # to measure, and ComfyUI's shortcut hands the hook zeroes.
            disable_cfg1_optimization=bool(cfgpp_active or gap_active),
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
            # Two consumers with different domains: torchsde's BrownianTree feeds
            # numpy's SeedSequence (rejects negatives), while default_noise_sampler
            # feeds torch's generator (int64, and it raises "Overflow when
            # unpacking long long" above 2**63-1). Linked seed nodes can supply
            # negative values even though the widget has min=0, so normalise to a
            # non-negative int that BOTH accept.
            if seed < 0:
                seed = seed % (2 ** 63)
        if noise_mode == "brownian" and positive and min(positive) > 0.0:
            # torchsde answers a query on the device its tree was built on, and
            # comfy builds it on the CPU.  That costs 75 ms per step at 1024 px
            # and 300 ms at 2048 px: the dyadic bridge runs on the CPU and the
            # whole field crosses PCIe every step.  Built on the latent's device
            # it is 3-5 ms/step, paid for with ~50 MB of VRAM per MP of latent
            # (the tree caches recent intervals).  tools/noise_cost.py measures
            # both, with no checkpoint loaded.
            try:
                noise_sampler = BrownianTreeNoiseSampler(
                    x, min(positive), max(positive), seed=seed,
                    cpu=(x.device.type != "cuda"),
                )
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
    prev_sig_eval = None
    prev_sig_eval2 = None
    n_corr = 0
    u_strength = 1.0
    d_hist = []
    d_nodes = []
    # Adams-Bashforth in sigma shares the ODE dx/dsigma = (x - x0)/sigma, which
    # holds for both families, but it has no exponential linearisation and no
    # OU noise term, so it is only used for the deterministic non-CFG++ case.
    # Churn also forces exp: it re-noises the state and moves the model
    # evaluation to sigma_hat, where a sigma-space derivative must start; the
    # sigma-space loop below integrates from the un-churned sigma.
    use_ab = (
        (str(integrator).lower() == "ab")
        and eta <= _ETA_EPS
        and not cfgpp_active
        and float(s_churn) <= 0.0
    )
    # NFE skipping needs a model call to exist.  CFG++'s noise estimate is the
    # unconditional prediction captured by the hook on the CURRENT step, so on a
    # skipped step it would be a stale u from an earlier sigma -- a wrong step,
    # not a cheaper one.  CFG++ therefore evaluates every step, the same way it
    # forces the exp integrator; the resolved behaviour is recorded in
    # LAST_STATS["skip_reason"] and the A/B node reports it.
    skip_reason = None
    if int(skip_every) > 1 and cfgpp_active:
        skip_reason = "cfg_pp"
    # The exp path keeps one previous x0 (hist_x0), so it has no third-order
    # history to consume.  Clamp it here, before _resolve_order, and record both
    # the request and the value in force.
    order_eff = int(order) if use_ab else min(int(order), 2)

    for i in comfy.utils.model_trange(n_steps, disable=disable):
        sigma_s = raw[i]
        sigma_t = raw[i + 1]
        lam_s = lams[i]
        gap_step["i"] = i
        if cfg_map and cfg_state["map"] is not None:
            cfg_map_steps += 1

        # ---- churn ----
        # "karras": relative gamma (Karras eq. 8), capped at sqrt(2)-1 and, on a
        #           flow model, at the schedule's first sigma (see below)
        # "gamma":  comfyui-ppm's euler_gamma law - an absolute offset
        #           proportional to sigma/sigma_max, skipped when it would
        #           exceed sigma_max (which is what ppm does at the first step)
        sigma_eval = sigma_s
        frac_i = i / max(n_steps - 1, 1)
        churn_here = cw_lo <= frac_i <= cw_hi
        if s_churn > 0.0 and not churn_here:
            n_churn_windowed += 1
        if s_churn > 0.0 and churn_here and s_tmin <= sigma_s <= s_tmax:
            if str(churn_mode) == "gamma":
                sigma_hat = sigma_s + s_churn * (sigma_s / max(raw[0], 1e-9))
                if sigma_t > 0.0 and sigma_hat <= raw[0]:
                    sigma_eval = sigma_hat
                    # ppm subtracts the re-noise draw; matching its sign keeps
                    # seeds interchangeable with euler_gamma
                    x, fired = _churn_forward(x, sigma_s, sigma_eval, flow, s_noise,
                                              sign=-1.0)
                    if fired:
                        n_churn += 1
            else:
                gamma = min(s_churn / max(n_steps, 1), 2 ** 0.5 - 1.0)
                sigma_eval = sigma_s * (gamma + 1.0)
                if flow and sigma_eval > raw[0]:
                    # A flow model is parameterised on sigma < 1 (alpha = 1 - sigma)
                    # and clamp_sigma() pins the geometry at 1 - _SIGMA_FLOOR.  At
                    # step 0 of a full denoise sigma_s is already sigma_max = 1, so
                    # sigma_hat = 1.414 used to be noised at 1.414 while the step
                    # divided by 1.0: the state came out at 1.30 sigma instead of
                    # 0.92, i.e. the injected noise was never removed (+41% at
                    # gamma = sqrt(2)-1).  Cap the evaluation at the
                    # schedule's first sigma, which is what the gamma law does by
                    # skipping the step; when sigma_s == raw[0] churn is then a
                    # bit-exact no-op, and the noise drawn below stays consistent
                    # with the sigma the step integrates from.
                    sigma_eval = raw[0]
                    n_churn_capped += 1
                if gamma > 0.0 and sigma_eval > sigma_s:
                    x, fired = _churn_forward(x, sigma_s, sigma_eval, flow, s_noise)
                    if fired:
                        n_churn += 1

        # churn moved the state to sigma_eval, so the step must start there
        # (comfy/k-diffusion use dt = sigma_next - sigma_hat for the same reason)
        sig_from = clamp_sigma(sigma_eval, flow) if sigma_eval != sigma_s else sig_geo[i]
        lam_from = half_log_snr(sig_from, flow)
        alpha_from = alpha_of(sig_from, flow)

        # ---- model evaluation (or epsilon extrapolation on a skipped step) ----
        skipped = False
        if (
            skip_reason is None
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
        # The abscissae are the sigmas the model actually saw, not the schedule
        # entries: with churn the two differ by sigma_hat/sigma = 1+gamma, and
        # pairing a churned x0 with the un-churned sigma biases the fit.
        if sigma_t <= 0.0:
            x = _final_extrapolate(
                denoised,
                [prev_denoised2, prev_denoised, denoised],
                [prev_sig_eval2, prev_sig_eval, sigma_eval],
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
            prev_sig_eval2, prev_sig_eval = prev_sig_eval, sigma_eval
            prev_lam = lam_s
            continue

        order_used = _resolve_order(order_eff, i, n_steps, lower_order_to_end)

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
            prev_sig_eval2, prev_sig_eval = prev_sig_eval, sigma_eval
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
        px_mask = None
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
                        px_mask = mask
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
            u_strength = s_u
            if s_u < 1.0:
                u_this_step = denoised + s_u * (u_this_step - denoised)
            u_eff = u_this_step
            if extrapolate and hist_u:
                # Same per-pixel gate as x0: applying the mask to only one of
                # the two histories turns the difference into extra guidance,
                # so cond == uncond would stop being a no-op.
                w = px_mask if px_mask is not None else 1.0
                u_eff = u_this_step + w * kappa * (u_this_step - hist_u[0]) / h_prev
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
        # Plain path: correct the guided x0 only.
        # CFG++ path: OFF by default, because correcting x0 without correcting
        # u would shift the guidance.  With corrector_cfgpp the extra call's
        # unconditional prediction (captured by the post-CFG hook) is averaged
        # into u with the same trapezoid, which keeps cond == uncond a no-op.
        # If the hook did not deliver u, the step is left uncorrected rather
        # than corrected asymmetrically.
        do_corr = (i in corr_steps) and (
            not cfgpp_active or (corrector_cfgpp and u_this_step is not None)
        )
        if do_corr:
            x_pred = advance(x0_eff, with_noise=False)
            u_before = uncond_denoised
            # Mark the extra evaluation so the post-CFG hook does not record it
            # as a trajectory step (duplicate sigma, inflated gap means, a `rev`
            # dip).  Shaping and the CFG map are still applied to its output, so
            # the corrected prediction lives in the same space as x0_eff.
            call_ctx["corrector"] = True
            try:
                x0_check = model(x_pred, sig_geo[i + 1] * s_in, **extra_args)
            finally:
                call_ctx["corrector"] = False
            n_calls += 1
            if cfgpp_active:
                u_check_raw = uncond_denoised
                uncond_denoised = u_before
                if u_check_raw is not None:
                    u_check = (u_check_raw if u_strength >= 1.0
                               else x0_check + u_strength * (u_check_raw - x0_check))
                    x0_eff = 0.5 * (x0_eff + x0_check)
                    u_eff = 0.5 * (u_eff + u_check)
                    x_base = x - alpha_from * u_eff
                    n_corr += 1
            else:
                x0_eff = 0.5 * (x0_eff + x0_check)
                # the extra evaluation must not leak its (later) uncond into history
                uncond_denoised = u_this_step
                n_corr += 1

        x = advance(x0_eff, with_noise=True)

        # ---- history ----
        hist_x0 = [denoised] + hist_x0[:1]
        if u_this_step is not None:
            hist_u = [u_this_step] + hist_u[:1]
        prev_denoised2, prev_denoised = prev_denoised, denoised
        prev_sig_eval2, prev_sig_eval = prev_sig_eval, sigma_eval
        prev_lam = lam_from

    if cfg_map and not cfg_map_seen and cfg_map_reason is None:
        cfg_map_reason = "no_gap" if cfg_state["calls"] else "no_uncond"
    LAST_STATS.update({
        "calls": int(n_calls),
        "steps": int(n_steps),
        "skipped": int(n_skipped),
        "skip_every": int(skip_every),
        "skip_reason": skip_reason,
        "churn_steps": int(n_churn),
        "churn_capped": int(n_churn_capped),
        "churn_window": [round(cw_lo, 4), round(cw_hi, 4)],
        "churn_windowed": int(n_churn_windowed),
        "corrector_steps": int(n_corr),
        "corrector_cfgpp": bool(corrector_cfgpp),
        "order_requested": int(order),
        "order_effective": int(order_eff),
        "integrator": "ab" if use_ab else "exp",
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
    if gap_active:
        if gap_seen == 0 and gap_reason is None:
            gap_reason = "no_uncond" if gap_skipped else "no_cfg_step"
        LAST_STATS.update({
            "gap_shape": bool(gap_shape_on),
            "gap_stats": bool(gap_stats),
            "gap_lf": round(float(gap_lf), 4),
            "gap_lf_keep": round(float(gap_lf_keep), 4),
            "gap_cap": round(float(gap_cap), 4),
            "gap_parallel": round(float(gap_parallel), 4),
            "gap_parallel_local": round(float(gap_parallel_local), 4),
            "gap_local_start": round(float(gap_local_start), 4),
            "gap_local_steps": int(gap_local_steps),
            "gap_local_mean": (round(gap_local_sum / gap_local_steps, 5)
                               if gap_local_steps else None),
            "gap_local_frac": (round(gap_local_frac_sum / gap_local_steps, 5)
                               if gap_local_steps else None),
            "gap_local_ref": (round(float(gap_local_ref), 6) if gap_local_ref is not None else None),
            "gap_steps": int(len(gap_series)),
            "gap_seen": int(gap_seen),
            "gap_skipped": int(gap_skipped),
            "gap_reason": gap_reason,
            "gap_cap_steps": int(gap_cap_steps),
            "gap_cap_mean": (round(gap_cap_sum / gap_cap_steps, 5) if gap_cap_steps else None),
            # Per-step (sigma, metrics): the resolution diagnostic.  Kept as a
            # plain list of primitives so a tool can dump it as JSON.
            "gap_series": [[round(float(s), 5),
                            {k: round(float(v), 6) for k, v in sorted(m.items())}]
                           for s, m in gap_series],
        })
        LAST_STATS.update({f"gap_{k}": v for k, v in gap_summary(gap_series).items()})
    return x


# Defaults for sample_omni and for every node's widgets.  noise_mode is "white":
# the SDE asks for one N(0, 1) field per step and queries each step interval once,
# so a Brownian path's disjoint increments are independent anyway.  Both modes
# measure identical (std 1.000, cross-step correlation 0.00, tools/noise_cost.py
# --check) and the tree costs 75-300 ms per step more; its one remaining property,
# a seed that pins W(sigma) across step counts, is opt-in.
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
    churn_window=(0.0, 1.0),
    s_tmin=0.0,
    s_tmax=float("inf"),
    noise_mode="white",
    cfg_pp=False,
    cfg_pp_strength=0.5,
    corrector="first",
    corrector_cfgpp=False,
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
    # image-preference measurement behind it yet; see docs/PD_AOS_AUDIT.md.
    px_order=False,
    px_order_tol=2.0,
    px_order_smooth=True,
    cfg_map=False,
    cfg_map_floor=0.85,
    cfg_map_start=0.5,
    cfg_map_pool=8,
    # Experimental guidance-gap shaping (docs/EXPERIMENT_GAP_SHAPING.md).
    # Neutral at the defaults: gap_lf 0 / gap_lf_keep 0 / gap_cap 0 / gap_parallel
    # 1 install no hook, so the sampler is bit-identical to the release.  The
    # values the experiment proposes are NOT measured optima and are deliberately
    # not defaults; pick them on the A/B node or with the `omni_hybrid_gap` name.
    gap_lf=0.0,
    gap_lf_keep=0.0,
    gap_cap=0.0,
    gap_parallel=1.0,
    gap_parallel_local=0.0,
    gap_local_start=0.0,
    gap_stats=False,
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
    # The old eye-tuned author combination, kept as a lab row after its
    # registered sampler name was retired (the gap relief superseded it):
    # CFG++ 0.5 + gamma churn at the slider maximum + eta 1 + per-pixel order.
    "M: cfgpp+churn+eta+px (lab)": {
        "cfg_pp": True,
        "s_churn": 100.0,
        "churn_mode": "gamma",
        "eta": 1.0,
        "px_order": True,
    },
    # Guidance-gap shaping (experimental; docs/EXPERIMENT_GAP_SHAPING.md).  The
    # numbers are the experiment's starting points, not measurements: the rows
    # exist so the four arms can be compared at equal NFE on one node.
    "N: gap HF (5%)": {"gap_lf": 0.05, "gap_lf_keep": 0.0},
    "O: gap cap 0.2": {"gap_cap": 0.2},
    "P: gap HF + cap": {"gap_lf": 0.05, "gap_lf_keep": 0.0, "gap_cap": 0.2},
    # Local APG: CFG++ 1.0 with the contrast component relieved only where the
    # guidance is locally excessive.  The arm for the 6 MP burn.
    "R: gap local APG (0.6)": {
        "cfg_pp": True,
        "cfg_pp_strength": 1.0,
        "gap_parallel_local": 0.6,
    },
    # The late-only version: relief starts at 30% of the trajectory, where the
    # hot decile is the clipping tail rather than the composition.
    "S: gap local APG late (0.6 from 30%)": {
        "cfg_pp": True,
        "cfg_pp_strength": 1.0,
        "gap_parallel_local": 0.6,
        "gap_local_start": 0.3,
    },
    # The author's tuned default: 60% of the contrast everywhere, none on the
    # hot tail after 30% of the steps (the two knobs subtract, so 0.6 - 0.6 = 0).
    "T: author (0.6/0.6 from 30%)": {
        "cfg_pp": True,
        "cfg_pp_strength": 1.0,
        "gap_parallel": 0.6,
        "gap_parallel_local": 0.6,
        "gap_local_start": 0.3,
    },
    # The tail flattened only to the frame's median push (measured parity is
    # keep ~0.1-0.25, so 0.6 - 0.45 = 0.15 sits in the middle of it).
    "U: author, parity tail (0.6/0.45 from 30%)": {
        "cfg_pp": True,
        "cfg_pp_strength": 1.0,
        "gap_parallel": 0.6,
        "gap_parallel_local": 0.45,
        "gap_local_start": 0.3,
    },
    # The same relief without CFG++: for a normal cfg > 1 run (MOSS), to see
    # whether the contrast relief helps or costs anything where nothing burns.
    "V: relief only (0.6/0.6 from 30%, no CFG++)": {
        "gap_parallel": 0.6,
        "gap_parallel_local": 0.6,
        "gap_local_start": 0.3,
    },
    # Stats only: the output is unchanged, but the node prints the per-step
    # cond/uncond gap (at cfg = 1 this forces the unconditional pass, 2x batch).
    "Q: gap probe (no shaping)": {"gap_stats": True},
}

# Named settings (the preset dropdown on the simple node, and the sampler names
# registered in the stock KSampler / KSamplerSelect lists).  The eye-tuned
# author-preference combination was retired in favour of the measured gap
# setting below; its lab row M still exists for A/B comparison.
VARIANTS = {
    "omni_hybrid": {},                       # measured defaults
    "omni_hybrid_sde": {"eta": 0.5},         # mid-trajectory stochasticity
    "omni_hybrid_cfg_pp": {"cfg_pp": True},  # CFG++ guidance
    # Experimental (docs/EXPERIMENT_GAP_SHAPING.md): CFG++ with the contrast
    # component at 60%, the hot tail flattened after 30% of the steps.  This is
    # the tuned setting the author judged harmless at 1.4 MP and on MOSS, and
    # effective at Krea2 6 MP.  It is the T row.
    "omni_hybrid_gap": {
        "cfg_pp": True,
        "cfg_pp_strength": 1.0,
        "gap_parallel": 0.6,
        "gap_parallel_local": 0.6,
        "gap_local_start": 0.3,
    },
}









