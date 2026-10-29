"""CFG++ verification: invariants + effective guidance scale.

1. omni's CFG++ step must equal comfy's euler_cfg_pp (same formula).
2. when cond == uncond, CFG++ must be a no-op (this caught a real bug: the
   multistep x0 average was applied to the conditional prediction only, which
   injected ~2x extra guidance).
3. cfg_pp_strength=0 must equal plain sampling; 1 = faithful CFG++.
4. effective cfg: which standard-CFG scale does a given CFG++ cfg look like?
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfy.samplers
import comfy.k_diffusion.sampling as kds

import omni_core
from tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init

torch.set_default_dtype(torch.float32)
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


class CfgToy:
    """cond prior = mixture, uncond prior = broad Gaussian (the 'empty prompt')."""

    def __init__(self, flow=False, cfg=1.0):
        self.flow = flow
        self.cfg = float(cfg)
        self.prior = MixturePrior(dim=2, seed=0)
        self.uncond = MixturePrior(dim=2, n_modes=1, radius=0.0, scale=2.0, seed=1)
        base = ToyModel(self.prior, flow=flow)
        self.model_sampling = base.model_sampling
        self.inner_model = base.inner_model
        self.sigma_max = base.sigma_max
        self.sigma_min = base.sigma_min

    def __call__(self, x, sigma, model_options=None, seed=None, **kw):
        s = torch.as_tensor(sigma, dtype=x.dtype).reshape(-1)
        if s.numel() == 1:
            s = s.expand(x.shape[0])
        cond = self.prior.posterior_mean(x, s.cpu(), flow=self.flow)
        uncond = self.uncond.posterior_mean(x, s.cpu(), flow=self.flow)
        return comfy.samplers.cfg_function(None, cond, uncond, self.cfg, x, sigma, model_options or {})


class SameCfg(CfgToy):
    """cond == uncond, so CFG++ has to be a no-op."""

    def __init__(self, flow=False):
        super().__init__(flow=flow, cfg=1.0)
        self.uncond = self.prior


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def setup(flow=False, n=12, seed=11):
    prior = MixturePrior(dim=2, seed=0)
    probe = ToyModel(prior, flow=flow)
    sig = torch.tensor([float(s) for s in karras_sigmas(n, probe.sigma_min, probe.sigma_max)])
    x, _ = make_init(prior, 64, float(sig[0]), flow, torch.Generator().manual_seed(seed))
    return sig, x


def sample(model, sig, x, **kw):
    return omni_core.sample_omni(
        model, x.clone(), sig.clone(), extra_args={"model_options": {}, "seed": 0}, disable=True, **kw
    )


def main():
    sig, x = setup()

    print("\n=== 1. matches comfy euler_cfg_pp (order 1, strength 1) ===")
    for w in (1.0, 2.0, 4.0, 7.0):
        m = CfgToy(cfg=w)
        a = sample(m, sig, x, cfg_pp=True, cfg_pp_strength=1.0, order=1, eta=0.0,
                   corrector="none", lower_order_to_end=0)
        b = kds.sample_euler_cfg_pp(m, x.clone(), sig.clone(), extra_args={"model_options": {}, "seed": 0}, disable=True)
        d = rel(a, b)
        check(f"cfg={w}", d < 1e-3, f"rel={d:.1e}")

    print("\n=== 2. no-op invariant: cond == uncond (corrector off for both) ===")
    for order in (1, 2):
        m = SameCfg()
        a = sample(m, sig, x, cfg_pp=False, order=order, corrector="none", eta=0.0, lower_order_to_end=1)
        b = sample(m, sig, x, cfg_pp=True, cfg_pp_strength=1.0, order=order, corrector="none", eta=0.0,
                   lower_order_to_end=1)
        d = rel(b, a)
        check(f"order={order}", d < 1e-5, f"rel={d:.1e}")

    m = SameCfg()
    c_off = sample(m, sig, x, cfg_pp=True, corrector="none")
    c_on = sample(m, sig, x, cfg_pp=True, corrector="first")
    check("CFG++ ignores the corrector (documented policy)", rel(c_on, c_off) < 1e-12,
          f"rel={rel(c_on, c_off):.1e}")

    print("\n=== 3. cfg_pp_strength (cond != uncond) ===")
    m = SameCfg()
    plain_same = sample(m, sig, x, cfg_pp=False, corrector="none")
    s0 = sample(m, sig, x, cfg_pp=True, cfg_pp_strength=0.0, corrector="none")
    check("strength=0 equals plain sampling", rel(s0, plain_same) < 1e-6, f"rel={rel(s0, plain_same):.1e}")
    m = CfgToy(cfg=1.0)
    plain = sample(m, sig, x, cfg_pp=False, corrector="none")
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = sample(m, sig, x, cfg_pp=True, cfg_pp_strength=s, corrector="none")
        print(f"    strength={s:4.2f}  distance from plain cfg=1: {rel(out, plain):.3e}")

    print("\n=== 4. effective cfg (default path, order 2) ===")
    cond_only = sample(CfgToy(cfg=1.0), sig, x, cfg_pp=False)
    std = {w: sample(CfgToy(cfg=w), sig, x, cfg_pp=False)
           for w in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)}
    print("    standard: " + "  ".join(f"{w:g}:{rel(v, cond_only):.3f}" for w, v in std.items()))
    for strength in (0.0, 0.5, 1.0):
        line = []
        for w in (1.0, 2.0, 4.0):
            q = sample(CfgToy(cfg=w), sig, x, cfg_pp=True, cfg_pp_strength=strength)
            match = min(std, key=lambda k: rel(q, std[k]))
            line.append(f"cfg={w:g}->approx {match:.1f} (d={rel(q, cond_only):.3f})")
        print(f"    strength={strength:.1f}: " + "   ".join(line))

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
