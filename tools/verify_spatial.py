"""Connectivity + effect proof for omni's two optional spatial controls.

What this checks
----------------
`px_order` (per-pixel order-2 validity, a PD-AOS port) and `cfg_map` (late
per-region guidance, a corrected PD-AOS port) are new knobs.  A knob is not
finished until the tests show all three of:

  * OFF is bit-identical to the previous release (here: `git show HEAD:omni_core.py`
    loaded side by side and run on the same toy problem);
  * ON reaches its site (a run counter), differs from OFF, and the same value
    reproduces;
  * a negative control that should be a no-op is a no-op (px_order with a huge
    tolerance; cfg_map with floor 1; cfg_map at cfg 1).

Every checker prints its numbers and the script exits non-zero if any fails.
The RED cases are exercised on the same data: a linear trajectory must be
admitted at order 2 (mask 1), a quadratic one must be refused, a constant gap
must produce no map, and a scale-1 hook must return the input unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(REPO))   # the `omni_sampler` package itself
sys.path.insert(0, HERE)

import toy  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(("  [PASS] " if ok else "  [FAIL] ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baseline_module():
    """The v1.4.2 omni_core.py, from git (read-only)."""
    r = subprocess.run(["git", "-C", REPO, "show", "v1.4.2:omni_core.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    fd, path = tempfile.mkstemp(suffix="_omni_head.py")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(r.stdout)
    try:
        return load_module(path, "omni_head")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# fake models
# ---------------------------------------------------------------------------
class FieldModel(toy.ToyModel):
    """Toy geometry, flat x0 = 0.5 x, 4-D latent; optional CFG decomposition."""

    def __init__(self, flow=True, gap_split=0.5):
        prior = toy.MixturePrior(dim=4)
        super().__init__(prior, flow=flow)
        self.cfg = 1.0
        self.gap_split = float(gap_split)

    def set_cfg(self, v):
        self.cfg = float(v)

    def __call__(self, x, sigma, **kw):
        cond = 0.5 * x
        if self.cfg <= 1.0 + 1e-9:
            return cond
        # A structured unconditional prediction: the left half is almost the
        # conditional one (small gap), the right half is far from it (large gap).
        gate = (x[..., : x.shape[-1] // 2] * 0.0 + 0.9)
        pad = torch.full_like(x[..., x.shape[-1] // 2:], 0.5)
        scale = torch.cat([gate, pad], dim=-1)
        uncond = cond * scale
        den = uncond + (cond - uncond) * self.cfg
        for fn in (kw.get("model_options") or {}).get("sampler_post_cfg_function", []):
            den = fn({"denoised": den, "uncond_denoised": uncond,
                      "cond_denoised": cond, "cond_scale": self.cfg})
        return den


class SpatialQuadraticModel(toy.ToyModel):
    """x0 = a + b*sigma + c*sigma^2, with c varying per pixel (a checkerboard)."""

    def __init__(self, a=0.2, b=0.1, c=0.0, c2=80.0, h=8, w=8):
        prior = toy.MixturePrior(dim=4)
        super().__init__(prior, flow=True)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        self.pattern = ((yy + xx) % 2).float().view(1, 1, h, w)
        self.a, self.b, self.c, self.c2 = a, b, c, c2

    def __call__(self, x, sigma, **kw):
        s = torch.as_tensor(sigma, dtype=x.dtype, device=x.device).reshape(-1)[0]
        c = self.c + self.c2 * self.pattern
        return self.a + self.b * s + c * s * s


def _sigmas(n=6):
    ms = toy._PatchableFlow(None)
    return torch.tensor(toy.karras_sigmas(n, 0.05, float(ms.sigma_max)), dtype=torch.float32)


# ---------------------------------------------------------------------------
def main():
    core = load_module(os.path.join(REPO, "omni_core.py"), "omni_core")
    print("omni_core:", os.path.join(REPO, "omni_core.py"))

    # ---- 1. _px_order_mask, analytic -------------------------------------
    lam = [2.0, 1.0, 0.0]
    base = torch.full((1, 4, 8, 8), 3.0)
    hist = [(lam[0], base), (lam[1], base), (lam[2], base)]
    m = core._px_order_mask(hist, 2.5, 0.05)
    check("constant trajectory is admitted (mask 1)", m is not None and float(m.min()) > 0.999,
          "min=%.4f" % float(m.min()) if m is not None else "None")

    # linear in lambda: f'' = 0
    hist = [(2.0, 3.0 + 2 * 2.0 + 0 * base), (1.0, 3.0 + 2 * 1.0 + 0 * base),
            (0.0, 3.0 + 0 * base)]
    m = core._px_order_mask(hist, 2.5, 0.05)
    check("linear trajectory is admitted (mask 1)", m is not None and float(m.min()) > 0.999,
          "min=%.4f" % float(m.min()) if m is not None else "None")

    # quadratic with large curvature: f'' = 2*10 = 20 at every pixel
    quad = [(l, 10.0 * l * l + 0 * base) for l in lam]
    m = core._px_order_mask(quad, 2.5, 0.05)
    check("quadratic trajectory is refused at tol 0.05", m is not None and float(m.max()) < 0.5,
          "max=%.4f" % float(m.max()) if m is not None else "None")
    m_loose = core._px_order_mask(quad, 2.5, 1.0)
    check("same trajectory admitted at tol 1.0 (the tolerance is live)",
          m_loose is not None and float(m_loose.max()) > float(m.max()),
          "max %.4f -> %.4f" % (float(m.max()), float(m_loose.max())))

    # spatial variation: only the curved pixels are refused
    pattern = ((torch.arange(8).view(1, 1, 1, 8) + torch.arange(8).view(1, 1, 8, 1)) % 2).float()
    xs = [0.2 + 0.1 * l + (80.0 * pattern) * l * l for l in lam]
    hist = [(lam[i], xs[i]) for i in range(3)]
    # smooth=False: the shipped path blurs the mask, which deliberately mixes a
    # checkerboard's neighbours; the separation test is about the criterion.
    m = core._px_order_mask(hist, 2.5, 0.05, smooth=False)
    check("spatial mask separates curved from flat pixels",
          m is not None and float(m[pattern.expand_as(m) > 0.5].mean()) < 0.2
          and float(m[pattern.expand_as(m) < 0.5].mean()) > 0.8,
          "curved %.3f flat %.3f" % (
              float(m[pattern.expand_as(m) > 0.5].mean()),
              float(m[pattern.expand_as(m) < 0.5].mean())))
    m = core._px_order_mask(hist, 2.5, 0.05)
    check("smoothed mask is bounded to [0,1] and keeps the spatial shape",
          m is not None and float(m.min()) >= 0.0 and float(m.max()) <= 1.0
          and m.shape[-2:] == base.shape[-2:],
          str(tuple(m.shape)) if m is not None else "None")

    # ---- 2. cfg map helpers ---------------------------------------------
    cond = torch.zeros(1, 4, 16, 16)
    cond[:, :, :, :8] = 1.0
    cond[:, :, :, 8:] = 4.0
    uncond = cond * 0.5
    cm = core._cfg_map_from_gap(cond, uncond, 0.85, 8)
    check("cfg map built with the expected shape", cm is not None and tuple(cm.shape) == (1, 1, 8, 8),
          str(tuple(cm.shape)) if cm is not None else "None")
    check("cfg map floor is reached and the top is 1",
          cm is not None and abs(float(cm.min()) - 0.85) < 1e-5 and abs(float(cm.max()) - 1.0) < 1e-5,
          "min=%.4f max=%.4f" % (float(cm.min()), float(cm.max())) if cm is not None else "None")
    check("cfg map is monotone in the gap",
          cm is not None and float(cm[:, :, :, :4].mean()) < float(cm[:, :, :, 4:].mean()),
          "left %.4f right %.4f" % (float(cm[:, :, :, :4].mean()), float(cm[:, :, :, 4:].mean())))
    check("constant gap produces no map (RED control)",
          core._cfg_map_from_gap(cond, cond, 0.85, 8) is None)

    den = uncond + (cond - uncond) * 2.0
    out = core._apply_cfg_map(den, uncond, 2.0, cm)
    import torch.nn.functional as F
    cm_up = F.interpolate(cm, size=tuple(den.shape[-2:]), mode="bilinear", align_corners=False)
    expect = uncond + (den - uncond) * cm_up
    check("map scales the guidance delta, not the prediction",
          torch.allclose(out, expect, atol=1e-6))
    one = torch.ones_like(cm)
    check("floor 1 is a bit-exact no-op (negative control)",
          torch.equal(core._apply_cfg_map(den, uncond, 2.0, one), den))
    check("cfg 1 is a no-op (ComfyUI hands the hook a zero uncond)",
          torch.equal(core._apply_cfg_map(den, uncond, 1.0, cm), den))

    cond5 = torch.randn(1, 4, 3, 16, 16, generator=torch.Generator().manual_seed(3))
    uncond5 = cond5 * 0.5
    cm5 = core._cfg_map_from_gap(cond5, uncond5, 0.85, 8)
    den5 = uncond5 + (cond5 - uncond5) * 2.0
    out5 = core._apply_cfg_map(den5, uncond5, 2.0, cm5)
    check("5-D latent: map collapses frames and broadcasts back",
          cm5 is not None and tuple(cm5.shape) == (1, 1, 8, 8)
          and tuple(out5.shape) == tuple(den5.shape) and bool(torch.isfinite(out5).all()),
          "map %s out %s" % (tuple(cm5.shape) if cm5 is not None else None,
                             tuple(out5.shape)))

    # ---- 3. sampler OFF is bit-identical to the previous release ---------
    head = baseline_module()
    sig = _sigmas()
    x0 = torch.randn(1, 4, generator=torch.Generator().manual_seed(11))
    if head is not None:
        a = head.sample_omni(toy.ToyModel(toy.MixturePrior(dim=4)),
                             x0.clone(), sig.clone(), order=2)
        b = core.sample_omni(toy.ToyModel(toy.MixturePrior(dim=4)), x0.clone(), sig.clone(), order=2)
        check("OFF is bit-identical to v1.4.2", torch.equal(a, b),
              "maxdiff %.3e" % float((a - b).abs().max()))
    else:
        print("  [SKIP] baseline module unavailable (git show failed)")

    # ---- 4. px_order reaches its site, and the no-op control ------------
    model = toy.ToyModel(toy.MixturePrior(dim=4))
    off = core.sample_omni(model, x0.clone(), sig.clone(), order=2)
    huge = core.sample_omni(model, x0.clone(), sig.clone(), order=2, px_order=True, px_order_tol=1e9)
    check("px_order with an infinite tolerance is bit-identical (negative control)",
          torch.equal(off, huge))
    stats = dict(core.LAST_STATS)
    check("px_order stats stamp the run", stats.get("px_order") is True and stats.get("px_order_steps", 0) >= 0,
          str(stats.get("px_order_steps")))
    on = core.sample_omni(model, x0.clone(), sig.clone(), order=2, px_order=True, px_order_tol=0.02)
    check("px_order reaches its site and changes the step", not torch.equal(off, on),
          "rel %.3e" % float((on - off).norm() / off.norm()))
    on2 = core.sample_omni(model, x0.clone(), sig.clone(), order=2, px_order=True, px_order_tol=0.02)
    check("same value reproduces bit-exactly", torch.equal(on, on2))
    stats = dict(core.LAST_STATS)
    check("px_order step counter is non-zero",
          stats.get("px_order_steps", 0) > 0 and stats.get("px_order_mean") is not None,
          "steps=%s mean=%s" % (stats.get("px_order_steps"), stats.get("px_order_mean")))

    # ---- 5. cfg_map reaches its site on a 4-D CFG model ----------------
    fm = FieldModel(flow=True)
    fm.set_cfg(2.0)
    x4 = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(5))
    sig4 = _sigmas(8)
    off4 = core.sample_omni(fm, x4.clone(), sig4.clone(), order=2)
    on4 = core.sample_omni(fm, x4.clone(), sig4.clone(), order=2, cfg_map=True, cfg_map_start=0.3)
    st = dict(core.LAST_STATS)
    check("cfg_map builds and applies", st.get("cfg_map_built") is True and st.get("cfg_map_steps", 0) > 0,
          "built=%s steps=%s reason=%s" % (st.get("cfg_map_built"), st.get("cfg_map_steps"),
                                           st.get("cfg_map_reason")))
    check("cfg_map changes the guidance path", not torch.equal(off4, on4),
          "rel %.3e" % float((on4 - off4).norm() / off4.norm()))
    on4b = core.sample_omni(fm, x4.clone(), sig4.clone(), order=2, cfg_map=True, cfg_map_start=0.3)
    check("cfg_map reproduces bit-exactly", torch.equal(on4, on4b))
    one4 = core.sample_omni(fm, x4.clone(), sig4.clone(), order=2, cfg_map=True,
                            cfg_map_start=0.3, cfg_map_floor=1.0)
    check("cfg_map floor 1 is bit-identical to OFF (negative control)", torch.equal(off4, one4))
    fm.set_cfg(1.0)
    off1 = core.sample_omni(fm, x4.clone(), sig4.clone(), order=2)
    one = core.sample_omni(fm, x4.clone(), sig4.clone(), order=2, cfg_map=True, cfg_map_start=0.3)
    check("cfg_map at cfg 1 does not build and is bit-identical",
          torch.equal(one, off1) and core.LAST_STATS.get("cfg_map_built") is False,
          "built=%s reason=%s" % (core.LAST_STATS.get("cfg_map_built"),
                                  core.LAST_STATS.get("cfg_map_reason")))

    # ---- 6. the node's widget bounds accept the engine defaults ----------
    # (a default outside its own widget range is silently clamped by ComfyUI,
    # which would make the default shown in the UI a lie.)
    try:
        from omni_sampler import omni_nodes
        from omni_sampler.omni_core import DEFAULTS
        bad = []
        for cls in (omni_nodes.OmniSamplerSelect, omni_nodes.OmniKSampler):
            for name, spec in cls.INPUT_TYPES()["required"].items():
                if not isinstance(spec, (list, tuple)) or len(spec) < 2:
                    continue
                kind, opts = spec[0], spec[1]
                if not isinstance(opts, dict) or "default" not in opts:
                    continue
                if kind in ("FLOAT", "INT"):
                    d = opts["default"]
                    lo = opts.get("min", float("-inf"))
                    hi = opts.get("max", float("inf"))
                    if not (lo <= d <= hi):
                        bad.append("%s.%s default %r outside [%r, %r]" % (cls.__name__, name, d, lo, hi))
        check("every node widget default is inside its own range", not bad, "; ".join(bad))
        sel = set(omni_nodes.OmniSamplerSelect.INPUT_TYPES()["required"])
        need = {"px_order", "px_order_tol", "cfg_map", "cfg_map_floor", "cfg_map_start"}
        check("the select node emits every new knob", need <= sel, str(sorted(need - sel)))
        check("engine defaults cover every new knob",
              all(k in DEFAULTS for k in need), str(sorted(need - set(DEFAULTS))))
    except Exception as exc:  # noqa: BLE001
        check("node widget census ran", False, "%s: %s" % (type(exc).__name__, exc))

    print()
    if FAILED:
        print("FAILED: %d check(s): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
