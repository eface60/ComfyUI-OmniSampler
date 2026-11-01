"""End-to-end test of the ComfyUI integration (registration + nodes + KSAMPLER).

Verifies that:
  * importing the package registers its sampler names (including the
    author-preference preset) in the stock KSampler / KSamplerAdvanced /
    KSamplerSelect dropdowns,
  * `comfy.samplers.ksampler()` / `sampler_object()` resolve them,
  * the author-preference preset forwards exactly its five overrides and no
    others into `sample_omni`, and its px_order/CFG++ switches are in force,
  * the OmniSamplerSelect node builds a working SAMPLER,
  * KSAMPLER.sample() (the real ComfyUI entry point: inpaint wrapper, noise
    scaling, inverse scaling, callbacks) runs the sampler on both model families
    and lands close to the analytic reference.
"""

from __future__ import annotations

import os
import sys
from unittest import mock

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # ...\omni_sampler
PARENT = os.path.dirname(PKG)                                           # ...\generic
sys.path.insert(0, PKG)
sys.path.insert(0, PARENT)
sys.path.insert(0, COMFY)   # ComfyUI's own `nodes` module must win the import

import comfy.samplers
import omni_sampler
from omni_sampler import omni_core
from omni_sampler.omni_core import VARIANTS
from tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init, reference_endpoint

torch.set_default_dtype(torch.float32)
FAILS = []

# The author-preference preset: exactly the knobs the author's A/B node sets
# off their defaults, and nothing else.
AUTHOR_PREF = "omni_hybrid_cfgpp_author_pref"
AUTHOR_PREF_OVERRIDES = {
    "cfg_pp": True,
    "s_churn": 100.0,
    "churn_mode": "gamma",
    "eta": 1.0,
    "px_order": True,
}


class TestGuider:
    """Minimal guider stand-in: one conditional pass, cfg=1, no CFG hook."""

    def __init__(self, toy):
        self.inner_model = toy
        self.cfg = 1.0
        self.model_patcher = toy.inner_model.model_patcher
        self.toy = toy

    def __call__(self, x, sigma, model_options=None, seed=None):
        return self.toy(x, sigma)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def test_registration():
    import nodes  # core node registry

    for name in VARIANTS:
        check(f"registered in comfy.samplers.SAMPLER_NAMES: {name}", name in comfy.samplers.SAMPLER_NAMES)
        check(f"registered in KSampler.SAMPLERS: {name}", name in comfy.samplers.KSampler.SAMPLERS)
        s = comfy.samplers.ksampler(name)
        check(f"ksampler() resolves {name}", isinstance(s, comfy.samplers.KSAMPLER))
        s2 = comfy.samplers.sampler_object(name)
        check(f"sampler_object() resolves {name}", isinstance(s2, comfy.samplers.KSAMPLER))

    it = nodes.KSampler.INPUT_TYPES()["required"]["sampler_name"][0]
    for name in VARIANTS:
        check(f"stock KSampler dropdown contains {name}", name in it)

    it2 = nodes.KSamplerAdvanced.INPUT_TYPES()["required"]["sampler_name"][0]
    for name in VARIANTS:
        check(f"KSamplerAdvanced dropdown contains {name}", name in it2)

    try:
        from comfy_extras.nodes_custom_sampler import KSamplerSelect

        schema = KSamplerSelect.define_schema()
        opts = schema.inputs[0].options
        for name in VARIANTS:
            check(f"KSamplerSelect dropdown contains {name}", name in opts)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not introspect KSamplerSelect schema: {exc}")


def test_node_object():
    cls = omni_sampler.NODE_CLASS_MAPPINGS["OmniSamplerSelect"]
    node = cls()
    req = cls.INPUT_TYPES()["required"]
    kwargs = {}
    for k, v in req.items():
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], list):
            kwargs[k] = v[0][0]
        elif isinstance(v, tuple) and len(v) > 1 and isinstance(v[1], dict):
            kwargs[k] = v[1].get("default")
        else:
            kwargs[k] = v[0]
    out = node.build(**{**kwargs, "eta": 0.0})
    sampler = out[0]
    check("OmniSamplerSelect returns a KSAMPLER", isinstance(sampler, comfy.samplers.KSAMPLER))


def test_author_pref():
    """The preset's five overrides are its whole delta, and they reach the sampler.

    Static census: VARIANTS names exactly the five knobs the A/B node shows off
    their defaults.  Wiring: the function KSamplerSelect resolves is called with
    a spy in place of sample_omni, so the kwargs are checked without sampling;
    an explicit kwarg must still win (negative control for the merge order).
    The toy run then proves the two observable switches are in force: the
    px_order site ran, and CFG++ suppressed the step-0 corrector that
    DEFAULTS["corrector"] would otherwise add.
    """
    check("author pref preset is listed", AUTHOR_PREF in VARIANTS)
    check(
        "author pref overrides exactly its five knobs",
        VARIANTS.get(AUTHOR_PREF) == AUTHOR_PREF_OVERRIDES,
        f"overrides={VARIANTS.get(AUTHOR_PREF)}",
    )

    captured = {}

    def spy(model, x, sigmas, extra_args=None, callback=None, disable=None, **opts):
        captured.update(opts)
        return x

    sampler = comfy.samplers.sampler_object(AUTHOR_PREF)
    with mock.patch.object(omni_sampler.omni_nodes, "sample_omni", spy):
        sampler.sampler_function(None, None, None)
    expected = {**omni_core.DEFAULTS, **AUTHOR_PREF_OVERRIDES}
    differing = {k: (captured.get(k), expected.get(k))
                 for k in set(expected) | set(captured)
                 if captured.get(k) != expected.get(k)}
    check(
        "author pref emits DEFAULTS plus exactly its five overrides",
        captured == expected,
        f"got, expected={differing}",
    )
    with mock.patch.object(omni_sampler.omni_nodes, "sample_omni", spy):
        sampler.sampler_function(None, None, None, eta=0.25)
    check("an explicit kwarg overrides the preset value", captured.get("eta") == 0.25,
          f"eta={captured.get('eta')!r}")

    prior = MixturePrior(dim=2, seed=0)
    model = ToyModel(prior, flow=True)
    n = 8
    sig = karras_sigmas(n, model.sigma_min, model.sigma_max)
    sig_t = torch.tensor([float(s) for s in sig])
    x_init, _ = make_init(prior, 32, float(sig[0]), True, torch.Generator().manual_seed(5))
    # Own generator: the preset's churn/eta consume the global RNG, and a test
    # must not change its numbers when another test runs before it.
    noise = torch.randn_like(x_init, generator=torch.Generator().manual_seed(7))
    out = sampler.sample(
        TestGuider(model),
        sig_t,
        {"model_options": {}, "seed": 1},
        None,
        noise,
        latent_image=torch.zeros_like(x_init),
        disable_pbar=True,
    )
    check("author pref runs through KSAMPLER", bool(torch.isfinite(out).all()),
          f"shape={tuple(out.shape)}")
    check(
        "author pref reaches the px_order site",
        omni_core.LAST_STATS.get("px_order") is True
        and omni_core.LAST_STATS.get("px_order_steps", 0) > 0,
        f"px_order_steps={omni_core.LAST_STATS.get('px_order_steps')}",
    )
    check(
        "author pref's CFG++ skips the default step-0 corrector",
        omni_core.LAST_STATS.get("corrector_steps") == 0,
        f"corrector_steps={omni_core.LAST_STATS.get('corrector_steps')}",
    )


def test_end_to_end():
    from comfy.samplers import KSAMPLER

    prior = MixturePrior(dim=2, seed=0)
    for flow in (False, True):
        model = ToyModel(prior, flow=flow)
        n = 10
        sig = karras_sigmas(n, model.sigma_min, model.sigma_max)
        sig_t = torch.tensor([float(s) for s in sig])
        x_init, _ = make_init(prior, 64, float(sig[0]), flow, torch.Generator().manual_seed(5))
        noise = torch.randn_like(x_init, generator=torch.Generator().manual_seed(7))
        latent = torch.zeros_like(x_init)
        # replicate exactly what KSAMPLER.sample() feeds the sampler
        x_start = model.model_sampling.noise_scaling(
            torch.tensor(float(sig[0])), noise, latent, True
        )
        ref = reference_endpoint(x_start, float(sig[0]), float(sig[-2]), prior, flow)
        ref_x0 = prior.posterior_mean(ref, torch.full((64,), float(sig[-2])), flow=flow)

        guider = TestGuider(model)
        steps = []
        sampler = KSAMPLER(omni_sample)
        out = sampler.sample(
            guider,
            sig_t,
            {"model_options": {}, "seed": 1},
            lambda *a: steps.append(a),
            noise,
            latent_image=latent,
            disable_pbar=True,
        )
        rel = float((out - ref_x0).norm() / ref_x0.norm())
        check(
            f"KSAMPLER.sample end-to-end ({'flow' if flow else 'eps'}) is accurate",
            rel < 0.12,
            f"rel err={rel:.3e}",
        )
        check(
            f"callback fired once per step ({'flow' if flow else 'eps'})",
            len(steps) == n,
            f"{len(steps)} calls",
        )


def omni_sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
    return omni_core.sample_omni(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        # pin the integrator configuration: this test checks the KSAMPLER plumbing
        # (inpaint wrapper, noise scaling, callbacks), not the default tuning
        order=2, corrector="first", **kw
    )


if __name__ == "__main__":
    test_registration()
    test_node_object()
    test_author_pref()
    test_end_to_end()
    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        sys.exit(1)
    print("all checks passed")




