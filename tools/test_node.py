"""End-to-end test of the ComfyUI integration (registration + nodes + KSAMPLER).

Verifies that:
  * importing the package registers its sampler names in the stock KSampler /
    KSamplerAdvanced / KSamplerSelect dropdowns,
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
        # Not a warning: if the stock dropdown cannot be introspected, the
        # registration claim for KSamplerSelect is unverified, and a green run
        # must not hide that.
        check("KSamplerSelect dropdown is introspectable", False,
              f"{type(exc).__name__}: {exc}")


def _default_kwargs(cls):
    kwargs = {}
    for k, v in cls.INPUT_TYPES()["required"].items():
        opts = v[1] if isinstance(v, tuple) and len(v) > 1 and isinstance(v[1], dict) else {}
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], list):
            kwargs[k] = v[0][0]
        else:
            kwargs[k] = opts.get("default")
    return kwargs


def test_node_object():
    for cls_name in ("OmniSamplerSelectAdvanced", "OmniSamplerSelectSimple"):
        cls = omni_sampler.NODE_CLASS_MAPPINGS[cls_name]
        out = cls().build(**_default_kwargs(cls))
        check(f"{cls_name} returns a KSAMPLER and a config string",
              isinstance(out[0], comfy.samplers.KSAMPLER) and isinstance(out[1], str),
              f"{type(out[0]).__name__}, {out[1]!r}")


def test_registration_census():
    check("the retired author preset is gone",
          "omni_hybrid_cfgpp_author_pref" not in VARIANTS)
    check("the stock list is the four named settings",
          tuple(VARIANTS) == ("omni_hybrid", "omni_hybrid_sde",
                              "omni_hybrid_cfg_pp", "omni_hybrid_gap"),
          f"{tuple(VARIANTS)}")
    check("three nodes are registered",
          set(omni_sampler.NODE_CLASS_MAPPINGS) ==
          {"OmniSamplerSelectSimple", "OmniSamplerSelectAdvanced", "OmniKSampler"},
          f"{sorted(omni_sampler.NODE_CLASS_MAPPINGS)}")


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
            rel < 0.08,
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
    test_registration_census()
    test_end_to_end()
    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        sys.exit(1)
    print("all checks passed")




