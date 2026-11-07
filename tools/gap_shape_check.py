"""Guidance-gap shaping: invariants and the tone/structure test.

The feature rewrites the guidance delta -- the conditional minus the
unconditional prediction -- before that delta enters the step, for every model
family and every guidance path.  These checks pin what it must *not* do, then
demonstrate the one thing it is for: a low-frequency "tone" delta is removed
while a high-frequency "structure" delta survives.

1. **Neutral is bit-identical.**  With the shaping knobs at their defaults the
   hook must not change the output at all, and `gap_stats` (which installs the
   hook to measure) must not change it either.
2. **`cond == uncond` stays a no-op.**  Every knob on, the CFG++ path must still
   reproduce plain sampling exactly.
3. **The algebra.**  The cap never grows the delta and respects its bound and
   direction; the split removes the band and keeps the rest; the projection
   removes the parallel component; the neutral set is the identity; 5-D (video)
   latents keep their shape.
4. **End to end.**  A tone delta is removed, a structure delta is kept, and the
   per-pixel cap holds -- through ComfyUI's real post-CFG hook.
5. **The diagnostic.**  `gap_stats` records one entry per step and separates a
   tone delta from a structure delta.

Run with ComfyUI's venv python (COMFY_DIR overrides the default path):

    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\gap_shape_check.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFY)
sys.path.insert(0, REPO)

import comfy.samplers  # noqa: E402

import omni_core  # noqa: E402
from tools.toy import MixturePrior, ToyModel, karras_sigmas  # noqa: E402

torch.set_default_dtype(torch.float32)
FAILS = []
N = 8


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def cosine(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def patterns(shape=(2, 4, 32, 32)):
    """A broad positive blob (tone/exposure) and a 2-px checkerboard (structure)."""
    h, w = shape[-2], shape[-1]
    yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, h),
                            torch.linspace(-1.0, 1.0, w), indexing="ij")
    tone = 2.0 * torch.exp(-(xx ** 2 + yy ** 2) / (2 * 1.2 ** 2))
    tex = torch.sign(torch.sin(2 * math.pi * 8 * xx) * torch.sin(2 * math.pi * 8 * yy))
    return (tone[None, None].expand(shape).contiguous(),
            tex[None, None].expand(shape).contiguous())


class GapToy:
    """Constant denoiser on a 4-D latent: cond = base + delta, uncond = base.

    A denoiser that always predicts the same image makes the sampler return that
    prediction on the final (sigma = 0) step, so the shaping can be measured
    without integrating anything.  ``shortcut`` emulates ComfyUI's cfg1
    optimisation: at cond_scale == 1 and without `disable_cfg1_optimization` the
    unconditional branch is absent, and ComfyUI hands the hook zeroes.
    """

    def __init__(self, shape=(2, 4, 32, 32), flow=True, cfg=1.0, delta=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.flow = flow
        self.cfg = float(cfg)
        self.base = torch.randn(shape, generator=g) * 0.5
        self.delta = torch.zeros(shape) if delta is None else delta
        probe = ToyModel(MixturePrior(dim=2), flow=flow)
        self.model_sampling = probe.model_sampling
        self.sigma_max = probe.sigma_max
        self.sigma_min = probe.sigma_min

    def __call__(self, x, sigma, model_options=None, seed=None, **kw):
        model_options = model_options or {}
        cond = self.base + self.delta
        uncond = self.base
        if (math.isclose(self.cfg, 1.0)
                and not model_options.get("disable_cfg1_optimization", False)):
            uncond = torch.zeros_like(cond)
        return comfy.samplers.cfg_function(
            None, cond, uncond, self.cfg, x, sigma, model_options)


def setup(model, n=N, seed=3):
    sig = torch.tensor([float(s) for s in karras_sigmas(n, model.sigma_min, model.sigma_max)])
    x = torch.randn(model.base.shape, generator=torch.Generator().manual_seed(seed)) * float(sig[0])
    return sig, x


def run(model, sig, x, **kw):
    torch.manual_seed(0)
    return omni_core.sample_omni(
        model, x.clone(), sig.clone(), extra_args={"model_options": {}, "seed": 0},
        disable=True, **kw)


def last_stats():
    return dict(omni_core.LAST_STATS)


def test_neutral_identity():
    print("\n=== 1. neutral is bit-identical (and stats do not change the output) ===")
    tone, tex = patterns()
    delta = tone + 0.6 * tex
    m = GapToy(cfg=2.0, delta=delta)
    sig, x = setup(m)
    ref = run(m, sig, x, corrector="none")
    same = run(GapToy(cfg=2.0, delta=delta), sig, x, corrector="none",
               gap_lf=0.0, gap_lf_keep=0.0, gap_cap=0.0, gap_parallel=1.0)
    check("explicit neutral knobs match the default run", torch.equal(ref, same))

    stats_only = run(GapToy(cfg=2.0, delta=delta), sig, x, corrector="none", gap_stats=True)
    check("gap_stats alone does not change the output", torch.equal(ref, stats_only))
    st = last_stats()
    check("gap_stats recorded the run", st.get("gap_seen", 0) == N,
          f"gap_seen={st.get('gap_seen')} steps={st.get('steps')}")
    check("shaping really was off for the neutral run", not st.get("gap_shape", True))

    # Negative control: shaping at cfg = 2 must change the output, otherwise the
    # checks above would pass for a feature that never runs.
    shaped = run(GapToy(cfg=2.0, delta=delta), sig, x, corrector="none", gap_cap=0.1)
    check("shaping at cfg=2 changes the output", not torch.equal(ref, shaped),
          f"rel={rel(ref, shaped):.2e}")
    check("the cap engaged at cfg=2", last_stats().get("gap_cap_mean", 1.0) < 0.99,
          f"mean scale={last_stats().get('gap_cap_mean')}")


def test_noop_cond_eq_uncond():
    print("\n=== 2. cond == uncond stays a no-op with every knob on ===")
    m = GapToy(cfg=1.0, delta=torch.zeros(2, 4, 32, 32))
    sig, x = setup(m)
    plain = run(GapToy(cfg=1.0, delta=torch.zeros(2, 4, 32, 32)), sig, x, corrector="none")
    every = run(GapToy(cfg=1.0, delta=torch.zeros(2, 4, 32, 32)), sig, x, corrector="none",
                cfg_pp=True, cfg_pp_strength=1.0, gap_lf=0.05, gap_lf_keep=0.0,
                gap_cap=0.2, gap_parallel=0.0, gap_parallel_local=0.6)
    check("CFG++ + shaping == plain sampling", rel(every, plain) < 1e-6,
          f"rel={rel(every, plain):.1e}")
    check("the hook really ran", last_stats().get("gap_seen", 0) == N,
          f"gap_seen={last_stats().get('gap_seen')}")


def test_algebra():
    print("\n=== 3. the transforms' algebra ===")
    g = torch.Generator().manual_seed(1)
    delta = torch.randn(2, 4, 32, 32, generator=g)
    uncond = torch.randn(2, 4, 32, 32, generator=g) * 0.5
    cond = uncond + delta

    ident, _ = omni_core.shape_gap(delta, uncond, cond)
    check("neutral set is the identity", torch.equal(ident, delta))

    capped, info = omni_core.shape_gap(delta, uncond, cond, cap=0.3)
    lim = 0.3 * omni_core._gap_pixel_rms(uncond)
    got = omni_core._gap_pixel_rms(capped)
    check("cap respects its per-pixel bound",
          bool((got <= lim + 1e-5).all()), f"max ratio={(got / lim).max():.4f}")
    check("cap engages (the input was over the bound)", "cap_scale" in info,
          f"mean scale={info.get('cap_scale', 1.0):.3f}")
    # The cap is one scalar per pixel, so the capped vector must be the input
    # scaled by exactly that factor: the direction survives, only the length
    # changes.
    scale = got / omni_core._gap_pixel_rms(delta).clamp_min(1e-8)
    dot = (capped * delta).sum(dim=1, keepdim=True)
    want = (delta * delta).sum(dim=1, keepdim=True) * scale
    check("cap preserves the direction per pixel",
          float((dot - want).abs().max()) < 1e-4,
          f"max={float((dot - want).abs().max()):.2e}")
    small = torch.randn(2, 4, 32, 32, generator=g) * 1e-4
    capped2, info2 = omni_core.shape_gap(small, uncond, cond, cap=0.3)
    check("cap is a no-op under the bound", torch.equal(capped2, small))

    tone, tex = patterns()
    split, _ = omni_core.shape_gap(tone + 0.5 * tex, uncond, cond, lf=0.2, lf_keep=0.0)
    check("split removes the tone band",
          cosine(split, tone) < 0.2, f"cos(tone)={cosine(split, tone):.3f}")
    check("split keeps the structure",
          cosine(split, tex) > 0.97, f"cos(tex)={cosine(split, tex):.3f}")
    dc, _ = omni_core.shape_gap(torch.ones(2, 4, 32, 32), uncond, cond, lf=0.005)
    check("a tiny band removes only the DC", float(dc.abs().max()) < 1e-5,
          f"max={float(dc.abs().max()):.2e}")

    proj, _ = omni_core.shape_gap(delta, uncond, cond, parallel=0.0)
    par = (proj * cond).sum(dim=1, keepdim=True)
    check("projection removes the parallel component",
          float(par.abs().max()) < 1e-4, f"max={float(par.abs().max()):.2e}")

    five = torch.randn(1, 4, 2, 16, 16, generator=g)
    u5 = torch.randn(1, 4, 2, 16, 16, generator=g)
    out5, _ = omni_core.shape_gap(five, u5, u5 + five, lf=0.1, cap=0.5)
    check("a 5-D (video) latent keeps its shape", out5.shape == five.shape,
          str(tuple(out5.shape)))


def test_tone_vs_structure():
    print("\n=== 4. end to end: tone removed, structure kept, cap bounded ===")
    tone, tex = patterns()
    tex = tex * 0.6
    delta = tone + tex
    m = GapToy(cfg=1.0, delta=delta)
    sig, x = setup(m)

    base = run(GapToy(cfg=1.0, delta=delta), sig, x, corrector="none",
               cfg_pp=True, cfg_pp_strength=1.0)
    d0 = base - m.base
    check("the model's delta reaches the output", rel(d0, delta) < 1e-5,
          f"rel={rel(d0, delta):.2e}")

    shaped = run(GapToy(cfg=1.0, delta=delta), sig, x, corrector="none",
                 cfg_pp=True, cfg_pp_strength=1.0, gap_lf=0.2, gap_lf_keep=0.0)
    d1 = shaped - m.base
    struct = tex
    check("only the structure is left (tone energy removed)",
          abs(float(d1.pow(2).mean()) - float(struct.pow(2).mean()))
          < 0.05 * float(struct.pow(2).mean()),
          f"E={float(d1.pow(2).mean()):.3f} vs structure-only {float(struct.pow(2).mean()):.3f}")
    check("the mean shift (burn proxy) is gone",
          abs(float(d1.mean())) < 0.05 * abs(float(d0.mean())),
          f"mean {float(d1.mean()):+.4f} vs {float(d0.mean()):+.4f}")
    check("structure survives", cosine(d1, tex) > 0.95, f"cos(tex)={cosine(d1, tex):.3f}")

    capped = run(GapToy(cfg=1.0, delta=delta), sig, x, corrector="none",
                 cfg_pp=True, cfg_pp_strength=1.0, gap_cap=0.2)
    ratio = (omni_core._gap_pixel_rms(capped - m.base)
             / omni_core._gap_pixel_rms(m.base))
    check("per-pixel cap holds end to end", float(ratio.max()) <= 0.2 + 1e-3,
          f"max ratio={float(ratio.max()):.4f}")
    check("cap actually engaged", not torch.equal(capped, base),
          f"max ratio={float(ratio.max()):.4f}")


def test_stats_discriminate():
    print("\n=== 5. gap_stats: one entry per step, tone vs structure ===")
    tone, tex = patterns()
    m = GapToy(cfg=1.0, delta=tone)
    sig, x = setup(m)
    run(m, sig, x, corrector="none", gap_stats=True)
    st_tone = last_stats()
    series = st_tone.get("gap_series") or []
    check("one stats entry per sampling step", len(series) == N, f"{len(series)} of {N}")
    check("the series carries sigma and metrics",
          bool(series) and len(series[0]) == 2 and "lf_share" in series[0][1],
          str(series[0] if series else None))

    mt = GapToy(cfg=1.0, delta=tex)
    sig, x = setup(mt)
    run(mt, sig, x, corrector="none", gap_stats=True)
    st_tex = last_stats()
    lf_tone = st_tone.get("gap_lf_share_mean", 0.0)
    lf_tex = st_tex.get("gap_lf_share_mean", 1.0)
    check("lf_share separates a tone delta from a structure delta",
          lf_tone > 3.0 * lf_tex, f"tone={lf_tone:.3f} tex={lf_tex:.3f}")
    check("par_share is recorded", "gap_par_share_mean" in st_tone,
          f"par={st_tone.get('gap_par_share_mean')}")
    check("the revision sensor is recorded", "gap_rev_mean" in st_tone,
          f"rev={st_tone.get('gap_rev_mean')}")
    check("a constant denoiser has zero revision after the first step",
          st_tone.get("gap_rev_mean") == 0.0, f"rev={st_tone.get('gap_rev_mean')}")
    check("the out-of-range sensor sees the oversized prediction",
          (st_tone.get("gap_oor_mean") or 0.0) > 0.01,
          f"oor={st_tone.get('gap_oor_mean')}")


def test_local_gate():
    print("\n=== 6. local APG gate (borrowed from PD-AOS local fields) ===")
    g = torch.Generator().manual_seed(5)
    uni = torch.full((2, 4, 32, 32), 0.5)
    cond = torch.ones_like(uni)
    gate, ref = omni_core._gap_local_gate(uni, cond)
    check("a uniform push leaves the gate at 0", float(gate.max()) < 1e-3,
          f"max={float(gate.max()):.2e}")

    hot = uni.clone()
    hot[:, :, 8:16, 8:16] *= 4.0
    gate, ref = omni_core._gap_local_gate(hot, cond)
    inside = float(gate[:, :, 10:14, 10:14].mean())
    outside = float(gate[:, :, :4, :4].mean())
    check("the hotspot fires the gate", inside > 0.9, f"inside={inside:.3f}")
    check("the rest stays shut", outside < 1e-3, f"outside={outside:.3f}")
    gate2, _ = omni_core._gap_local_gate(hot * 2.0, cond * 2.0)
    check("the gate is scale-free", float((gate2 - gate).abs().max()) < 1e-4)

    # End to end: with parallel=1 and local=0.8, only the hotspot's contrast
    # component may move; the rest of the frame keeps its push.
    base = torch.randn(2, 4, 32, 32, generator=g) * 0.2
    uncond = torch.randn(2, 4, 32, 32, generator=g) * 0.5
    delta = base - base.mean(dim=(1, 2, 3), keepdim=True)
    hot = delta.clone()
    hot[:, :, 8:16, 8:16] += 3.0
    cond = uncond + hot
    out, info = omni_core.shape_gap(hot, uncond, cond, local=0.8)
    par_before = (hot * cond).sum(dim=1)
    par_after = (out * cond).sum(dim=1)
    r_in = float(par_after[:, 8:16, 8:16].abs().mean()
                 / par_before[:, 8:16, 8:16].abs().mean().clamp_min(1e-8))
    r_out = float(par_after[:, :4, :4].abs().mean()
                  / par_before[:, :4, :4].abs().mean().clamp_min(1e-8))
    check("contrast relieved inside the hotspot", r_in < 0.6, f"ratio={r_in:.3f}")
    check("contrast kept away from it", r_out > 0.95, f"ratio={r_out:.3f}")
    check("info reports the gate", "local_mean" in info and "local_ref" in info,
          str({k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()}))

    same, _ = omni_core.shape_gap(hot, uncond, cond, parallel=0.5, local=0.0)
    baseline, _ = omni_core.shape_gap(hot, uncond, cond, parallel=0.5)
    check("local=0 is bit-identical to the global projection", torch.equal(same, baseline))

    print("\n=== 7. the late-only ramp ===")
    check("relief is 0 before the start", omni_core._gap_local_gain(0.1, 0.4) == 0.0)
    check("relief is 1 at the end of the trajectory",
          abs(omni_core._gap_local_gain(1.0, 0.4) - 1.0) < 1e-9)
    check("start 0 keeps the old behaviour",
          omni_core._gap_local_gain(0.05, 0.0) == 1.0)

    tone, tex = patterns()
    delta = tone + 0.6 * tex
    sig, x = setup(GapToy(cfg=1.0, delta=delta))
    run(GapToy(cfg=1.0, delta=delta), sig, x, corrector="none",
        cfg_pp=True, cfg_pp_strength=1.0, gap_parallel_local=0.6)
    all_steps = last_stats().get("gap_local_steps", 0)
    run(GapToy(cfg=1.0, delta=delta), sig, x, corrector="none",
        cfg_pp=True, cfg_pp_strength=1.0, gap_parallel_local=0.6, gap_local_start=0.5)
    late_steps = last_stats().get("gap_local_steps", 0)
    check("without a start the relief runs every step", all_steps == N,
          f"{all_steps} of {N}")
    check("with a start it skips the early steps", 0 < late_steps < all_steps,
          f"{late_steps} of {all_steps}")


def test_corrector_is_not_a_step():
    print("\n=== 8. the corrector's extra evaluation is a call, not a step ===")
    # Regression: the post-CFG hook used to fire on the corrector's evaluation
    # too, which put a duplicate sigma in gap_series, inflated every gap mean by
    # ~10% and made `rev` dip at the same sigma.  The corrector is still a model
    # call (calls = steps + corrector_steps) but it must not be a series entry.
    tone, tex = patterns()
    delta = tone + 0.6 * tex
    sig, x = setup(GapToy(cfg=2.0, delta=delta))
    ref = run(GapToy(cfg=2.0, delta=delta), sig, x, corrector="first", gap_stats=True)
    st = last_stats()
    check("the corrector ran exactly once", st.get("corrector_steps") == 1,
          f"corrector_steps={st.get('corrector_steps')} calls={st.get('calls')}")
    check("one call per step plus the corrector", st.get("calls") == N + 1,
          f"calls={st.get('calls')} steps={st.get('steps')}")
    check("one series entry per step", st.get("gap_steps") == N,
          f"gap_steps={st.get('gap_steps')} steps={st.get('steps')}")
    check("gap_seen counts steps, not calls", st.get("gap_seen") == N,
          f"gap_seen={st.get('gap_seen')}")
    sigmas = [round(s, 6) for s, _ in st.get("gap_series", [])]
    check("no duplicate sigma in the series", len(sigmas) == len(set(sigmas)),
          f"sigmas={sigmas}")
    # The fix must not change what the sampler produces.
    plain = run(GapToy(cfg=2.0, delta=delta), sig, x, corrector="first")
    check("gap_stats + corrector is still bit-identical", torch.equal(ref, plain))


def test_eps_family():
    print("\n=== 9. eps family: the neutral/ON contract is family-independent ===")
    # shape_gap is a post-CFG transform and never sees alpha, so these checks are
    # expected to pass on eps exactly as on flow -- which is the point: the docs
    # claim "every model family", and until now only flow was exercised.
    tone, tex = patterns()
    delta = tone + 0.6 * tex
    m = GapToy(cfg=2.0, delta=delta, flow=False)
    sig, x = setup(m)
    base = run(GapToy(cfg=2.0, delta=delta, flow=False), sig, x, corrector="none")
    neutral = run(GapToy(cfg=2.0, delta=delta, flow=False), sig, x, corrector="none",
                  gap_lf=0.0, gap_lf_keep=0.0, gap_cap=0.0, gap_parallel=1.0,
                  gap_parallel_local=0.0)
    check("eps: neutral knobs are bit-identical", torch.equal(base, neutral))
    on = run(GapToy(cfg=2.0, delta=delta, flow=False), sig, x, corrector="none",
             gap_lf=0.2)
    check("eps: ON changes the step", not torch.equal(base, on))


def main():
    test_neutral_identity()
    test_noop_cond_eq_uncond()
    test_algebra()
    test_tone_vs_structure()
    test_stats_discriminate()
    test_local_gate()
    test_corrector_is_not_a_step()
    test_eps_family()
    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
