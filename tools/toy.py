"""Analytic test problem used to validate sampler math.

A Gaussian-mixture data distribution gives a closed-form posterior mean
(Tweedie's denoiser), so any diffusion sampler can be run against an exact
reference ODE solution.  Both model families are supported:

    eps  (VE-like):  x_sigma = x0 + sigma * eps
    flow (CONST)  :  x_sigma = (1 - sigma) * x0 + sigma * eps

Both satisfy  d x / d sigma = (x - E[x0 | x_sigma]) / sigma  (checked in
tools/verify_math.py), so one reference integrator covers both.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, os.environ.get("COMFY_DIR", r"G:\AI\comfy"))

import comfy.model_sampling


class MixturePrior:
    def __init__(self, dim=2, n_modes=6, radius=2.5, scale=0.40, seed=0):
        g = torch.Generator().manual_seed(seed)
        ang = torch.linspace(0, 2 * math.pi, n_modes + 1)[:-1]
        mu = torch.zeros(n_modes, dim)
        mu[:, 0] = radius * torch.cos(ang)
        mu[:, 1] = radius * torch.sin(ang)
        if dim > 2:
            mu[:, 2:] = torch.randn(n_modes, dim - 2, generator=g) * 0.8
        self.mu = mu
        self.scale = torch.full((n_modes,), float(scale))
        self.dim = dim
        self.n_modes = n_modes
        self.log_weights = torch.zeros(n_modes) - math.log(n_modes)

    def sample(self, n, generator=None):
        k = torch.randint(0, self.n_modes, (n,), generator=generator)
        eps = torch.randn(n, self.dim, generator=generator)
        return self.mu[k] + self.scale[k, None] * eps

    def _log_joint(self, x, sigma, flow):
        """log p(x | k) + log w_k, plus the denominator, for each mode."""
        b = x.shape[0]
        s2 = (sigma ** 2)[:, None]                                # (B,1)
        mode_var = self.scale[None, :] ** 2                       # (1,K)
        mu = self.mu[None, :, :]                                  # (1,K,D)
        if flow:
            a = (1.0 - sigma)[:, None]                            # (B,1)
            mean = a[:, :, None] * mu                             # (B,K,D)
            var = a[:, :, None] ** 2 * mode_var[:, :, None] + s2[:, :, None]
        else:
            mean = mu
            var = mode_var[:, :, None] + s2[:, :, None]
        diff = x[:, None, :] - mean
        loglik = -0.5 * (diff ** 2).sum(-1) / var[..., 0] - 0.5 * torch.log(var[..., 0])
        return self.log_weights[None, :] + loglik, var

    def posterior_mean(self, x, sigma, flow=False):
        x = x if x.dim() > 1 else x[None, :]
        sigma = torch.as_tensor(sigma, dtype=x.dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(x.shape[0])
        logw, var = self._log_joint(x, sigma, flow)
        w = torch.softmax(logw, dim=1)
        if flow:
            a = (1.0 - sigma)[:, None]
        else:
            a = torch.ones_like(sigma)[:, None]
        mode_var = self.scale[None, :] ** 2
        s2 = (sigma ** 2)[:, None]
        num = a[:, :, None] * x[:, None, :] * mode_var[:, :, None] + self.mu[None, :, :] * s2[:, :, None]
        mean_k = num / var
        return (w[:, :, None] * mean_k).sum(1)

    def mode_weights(self, x, sigma, flow=False):
        x = x if x.dim() > 1 else x[None, :]
        sigma = torch.as_tensor(sigma, dtype=x.dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(x.shape[0])
        logw, _ = self._log_joint(x, sigma, flow)
        return torch.softmax(logw, dim=1)

    def nearest_mode(self, x):
        d = ((x[:, None, :] - self.mu[None, :, :]) ** 2).sum(-1)
        return d.argmin(dim=1)


class _PatchableFlow(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    """Mirrors comfy.model_base.model_sampling(): flow schedule + CONST denoiser."""


class _PatchableEps(comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.EPS):
    """Mirrors comfy.model_base.model_sampling(): discrete beta schedule + EPS."""


class _Patcher:
    def __init__(self, ms):
        self._ms = ms
        self.model = self

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self._ms


class _InnerModel:
    def __init__(self, ms):
        self.model_sampling = ms


class _Guider:
    def __init__(self, ms):
        self.inner_model = _InnerModel(ms)
        self.model_patcher = _Patcher(ms)
        self.model_sampling = ms


class ToyModel:
    """Callable with the same shape/attribute contract as comfy's sampler model."""

    def __init__(self, prior, flow=False):
        self.prior = prior
        self.flow = flow
        if flow:
            ms = _PatchableFlow(None)
        else:
            ms = _PatchableEps(None)
        self.model_sampling = ms
        self.inner_model = _Guider(ms)
        self.sigma_max = float(ms.sigma_max)
        self.sigma_min = float(ms.sigma_min)
        self.cfg = 1.0

    def __call__(self, x, sigma, **kwargs):
        s = torch.as_tensor(sigma, dtype=x.dtype, device=x.device).reshape(-1)
        if s.numel() == 1:
            s = s.expand(x.shape[0])
        return self.prior.posterior_mean(x, s.cpu(), flow=self.flow).to(x.device)


class AnalyticX0Model:
    """Denoiser that depends only on sigma: isolates the ODE integrator.

    Used by verify_math to measure the convergence order of the sampler's
    integration scheme independently of any learned model.
    """

    def __init__(self, x0_fn, flow=False):
        self.x0_fn = x0_fn
        self.flow = flow
        if flow:
            ms = _PatchableFlow(None)
        else:
            ms = _PatchableEps(None)
        self.model_sampling = ms
        self.inner_model = _Guider(ms)
        self.sigma_max = float(ms.sigma_max)
        self.sigma_min = float(ms.sigma_min)
        self.cfg = 1.0

    def __call__(self, x, sigma, **kwargs):
        s = torch.as_tensor(sigma, dtype=x.dtype, device=x.device).reshape(-1)
        if s.numel() == 1:
            s = s.expand(x.shape[0])
        vals = [self.x0_fn(float(v)) for v in s]
        v = torch.tensor(vals, dtype=x.dtype, device=x.device)
        return v.reshape(-1, *([1] * (x.dim() - 1))).expand_as(x).contiguous()


# --------------------------------------------------------------------------
def karras_sigmas(n, sigma_min, sigma_max, rho=7.0):
    ramp = np.linspace(0, 1, n)
    sig = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return np.append(sig, 0.0)


def simple_sigmas(n, sigma_min, sigma_max):
    return np.append(np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), n)), 0.0)


def uniform_sigmas(n, sigma_min, sigma_max):
    """Uniform in sigma -- the common 'simple' schedule for flow models."""
    return np.append(np.linspace(sigma_max, sigma_min, n), 0.0)


def reference_endpoint(x_init, sigma0, sigma_end, prior, flow):
    """Integrate d x / d sigma = (x - E[x0|x]) / sigma; return x at sigma_end.

    float64 + DOP853 at rtol 1e-10: discretization error is orders of magnitude
    below every sampler error measured by the benchmarks.
    """
    x_np = x_init.detach().double().cpu().numpy()
    b, d = x_np.shape
    s_min = max(float(sigma_end), 1e-5)

    def rhs(t, flat):
        x = torch.from_numpy(flat.reshape(b, d))
        sig = torch.full((b,), abs(float(t)), dtype=torch.float64)
        x0h = prior.posterior_mean(x, sig, flow=flow).double()
        return ((x - x0h) / abs(float(t))).numpy().reshape(-1)

    sol = solve_ivp(
        rhs,
        (float(sigma0), s_min),
        x_np.reshape(-1),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return torch.from_numpy(sol.y[:, -1].reshape(b, d)).float()


def make_init(prior, n, sigma0, flow, generator=None):
    x0 = prior.sample(n, generator=generator)
    eps = torch.randn(n, prior.dim, generator=generator)
    x = (1.0 - sigma0) * x0 + sigma0 * eps if flow else x0 + sigma0 * eps
    return x, x0



