"""Parity + tooltip test for the two Omni-Hybrid nodes.

What this checks
----------------
* **Feature parity.**  `OmniKSampler` and `OmniSamplerSelect` expose the same
  sampler knobs.  The only allowed difference is the KSampler's own I/O
  (model / conditioning / latent / seed / steps / cfg / scheduler / denoise),
  which a SAMPLER object cannot own.
* **One definition.**  Both `INPUT_TYPES` are generated from `_KNOB_SPECS`, and
  both order lists contain exactly those keys, so adding a knob to one node
  cannot silently skip the other.
* **Tooltips.**  Every widget has a non-empty tooltip, and every choice of every
  combo is named in its tooltip (a tooltip is the only user-facing description).
* **Defaults.**  Both nodes default every sampler knob to `omni_core.DEFAULTS`
  (and the `variant` switch to the empty preset A).
* **Positional safety.**  The v1.5.1 widget order stays a prefix of the new
  order; ComfyUI stores widget values positionally, so inserting a widget would
  re-map every saved workflow.
* **Wiring.**  Building the select node emits every knob (including the new
  `churn_mode`, `skip_every`, `skip_order`) into the SAMPLER partial, and the
  KSampler path emits `eta_window` (parsed to a tuple) into `sample_omni`.
  A non-default build is compared with the default build as a negative control.

Run it with ComfyUI's venv python (COMFY_DIR overrides the default path):

    $env:COMFY_DIR = "C:\\path\\to\\ComfyUI"
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\test_node_parity.py
"""

from __future__ import annotations

import os
import sys
from unittest import mock

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
WORK = os.environ.get(
    "OMNI_WORK", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, COMFY)
sys.path.insert(0, WORK)

import comfy.samplers  # noqa: E402
import omni_sampler  # noqa: E402,F401
from omni_sampler import omni_core  # noqa: E402
from omni_sampler.omni_nodes import (  # noqa: E402
    _KNOB_SPECS,
    _KSAMPLER_ORDER,
    _SELECT_ORDER,
    OmniKSampler,
    OmniSamplerSelect,
    _parse_eta_window,
    _resolve_sampler_options,
)

FAILS = []

# The v1.5.1 order, read from `git show b3a9081:omni_nodes.py`.  Saved workflows
# store widget values positionally, so this must stay a prefix of the new order.
V151_SELECT_ORDER = [
    "order", "integrator", "eta", "s_noise", "corrector", "lower_order_to_end",
    "cfg_pp", "cfg_pp_strength", "x0_average", "min_step_ratio", "eta_window",
    "s_churn", "noise_mode", "final_extrapolate", "px_order", "px_order_tol",
    "cfg_map", "cfg_map_floor", "cfg_map_start",
]
V151_KSAMPLER_IO = [
    "model", "positive", "negative", "latent_image", "seed", "steps", "cfg",
    "scheduler", "denoise",
]
V151_KSAMPLER_ORDER = V151_KSAMPLER_IO + [
    "variant", "order", "integrator", "corrector", "x0_average", "min_step_ratio",
    "lower_order_to_end", "eta", "s_noise", "cfg_pp", "cfg_pp_strength", "s_churn",
    "churn_mode", "noise_mode", "skip_every", "skip_order", "final_extrapolate",
    "px_order", "px_order_tol", "cfg_map", "cfg_map_floor", "cfg_map_start",
]
KSAMPLER_IO = set(V151_KSAMPLER_IO)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _defaults():
    select = OmniSamplerSelect.INPUT_TYPES()["required"]
    ksam = OmniKSampler.INPUT_TYPES()["required"]
    return select, ksam


def test_parity():
    select, ksam = _defaults()
    select_keys = list(select)
    ksam_knobs = [k for k in ksam if k not in KSAMPLER_IO]

    check("select widget order has no duplicates", len(select_keys) == len(set(select_keys)))
    check("ksampler widget order has no duplicates", len(list(ksam)) == len(set(ksam)))
    check("same sampler knob set", set(select_keys) == set(ksam_knobs),
          f"select-only={sorted(set(select_keys) - set(ksam_knobs))}, "
          f"ksampler-only={sorted(set(ksam_knobs) - set(select_keys))}")
    check("both nodes cover _KNOB_SPECS",
          set(select_keys) == set(_KNOB_SPECS) == set(ksam_knobs))
    check("select order list matches its widgets", select_keys == _SELECT_ORDER)
    check("ksampler order list matches its knobs", ksam_knobs == _KSAMPLER_ORDER)
    check("only KSampler I/O keys added to the KSampler",
          set(ksam) - set(select_keys) == KSAMPLER_IO)
    check("variant options match AB_PRESETS",
          list(select["variant"][0]) == list(omni_core.AB_PRESETS) + ["custom (use knobs)"],
          f"options={list(select['variant'][0])}")


def test_positional_prefix():
    select, ksam = _defaults()
    check("select widget order keeps the v1.5.1 prefix",
          list(select)[:len(V151_SELECT_ORDER)] == V151_SELECT_ORDER,
          f"prefix={list(select)[:len(V151_SELECT_ORDER)]}")
    check("ksampler widget order keeps the v1.5.1 prefix",
          list(ksam)[:len(V151_KSAMPLER_ORDER)] == V151_KSAMPLER_ORDER,
          f"prefix={list(ksam)[:len(V151_KSAMPLER_ORDER)]}")


def test_defaults():
    select, ksam = _defaults()
    for name in _SELECT_ORDER:
        for node, req in (("select", select), ("ksampler", ksam)):
            entry = req[name]
            default = entry[1].get("default")
            if name == "eta_window":
                ok = _parse_eta_window(default) == omni_core.DEFAULTS[name]
            elif name == "variant":
                ok = default in omni_core.AB_PRESETS or default == "custom (use knobs)"
            else:
                ok = default == omni_core.DEFAULTS[name]
            label = ("default is a valid preset: variant" if name == "variant"
                     else f"default matches DEFAULTS: {name}")
            check(f"{node} {label}", ok, f"widget={default!r}")
        check(f"both nodes default {name} identically",
              select[name][1].get("default") == ksam[name][1].get("default"))


def test_tooltips():
    select, ksam = _defaults()
    for node, req in (("select", select), ("ksampler", ksam)):
        for name, entry in req.items():
            tip = entry[1].get("tooltip", "") if len(entry) > 1 else ""
            check(f"{node} tooltip exists: {name}", bool(tip.strip()))
            kind = entry[0]
            if isinstance(kind, list):
                missing = [str(opt) for opt in kind if str(opt) not in tip]
                check(f"{node} tooltip explains every option: {name}", not missing,
                      f"missing={missing}")


def test_resolver():
    # Parsing rules: valid pair, either separator; invalid input falls back.
    check("eta window parses comma", _parse_eta_window("0.2,0.8") == (0.2, 0.8))
    check("eta window parses semicolon", _parse_eta_window("0.2;0.8") == (0.2, 0.8))
    check("eta window accepts a tuple", _parse_eta_window((0.25, 0.75)) == (0.25, 0.75))
    check("eta window rejects reversed pair",
          _parse_eta_window("0.9,0.1") == omni_core.DEFAULTS["eta_window"])
    check("eta window rejects empty input",
          _parse_eta_window("") == omni_core.DEFAULTS["eta_window"])

    preset = _resolve_sampler_options({"variant": "I: SDE eta 0.3", "eta": 0.0})
    check("preset overrides the knob it names", preset.get("eta") == 0.3, f"eta={preset.get('eta')}")
    check("preset result carries the parsed eta window",
          preset.get("eta_window") == omni_core.DEFAULTS["eta_window"])
    custom = _resolve_sampler_options({"variant": "custom (use knobs)", "eta": 0.0})
    check("custom uses the knob value", custom.get("eta") == 0.0)
    partial_preset = _resolve_sampler_options(
        {"variant": "B: no corrector", "corrector": "first", "order": 1})
    check("preset leaves unnamed knobs alone", partial_preset.get("order") == 1)
    check("preset sets its named knob", partial_preset.get("corrector") == "none")


def _sentinel(name, entry):
    """A value different from the widget default, valid for its range."""
    kind, opts = entry
    default = opts.get("default")
    if name == "variant":
        return "custom (use knobs)"
    if name == "eta_window":
        return "0.20,0.80"
    if isinstance(kind, list):
        choices = [choice for choice in kind if choice != default]
        return choices[-1]
    if kind == "BOOLEAN":
        return not default
    if kind == "INT":
        return default + 1 if default + 1 <= opts["max"] else default - 1
    if kind == "FLOAT":
        step = opts.get("step", 0.01)
        value = default + step
        if value > opts["max"]:
            value = default - step
        return round(value, 6)
    raise AssertionError(f"no sentinel rule for {name} ({kind})")


def _independent_expected(inputs):
    """Expected sample_omni kwargs without calling the resolver under test."""
    expected = {k: v for k, v in inputs.items() if k != "variant"}
    expected["eta_window"] = (0.2, 0.8)
    return expected


def test_select_wiring():
    select, _ = _defaults()
    inputs = {name: _sentinel(name, entry) for name, entry in select.items()}
    sampler = OmniSamplerSelect().build(**inputs)[0]
    emitted = dict(sampler.sampler_function.keywords)
    expected = _independent_expected(inputs)
    check("select emits every knob into the SAMPLER",
          emitted == expected,
          f"missing={sorted(set(expected) - set(emitted))}, "
          f"extra={sorted(set(emitted) - set(expected))}")
    check("select passes eta_window as a tuple", emitted.get("eta_window") == (0.2, 0.8),
          f"eta_window={emitted.get('eta_window')!r}")
    for name in ("churn_mode", "skip_every", "skip_order"):
        check(f"select emits {name}", emitted.get(name) == inputs[name],
              f"{name}={emitted.get(name)!r}")
    again = dict(OmniSamplerSelect().build(**inputs)[0].sampler_function.keywords)
    check("select: the same values reproduce the same record", again == emitted)

    default_inputs = {name: entry[1].get("default") for name, entry in select.items()}
    default_sampler = OmniSamplerSelect().build(**default_inputs)[0]
    default_emitted = dict(default_sampler.sampler_function.keywords)
    check("negative control: the sentinel build differs from the default build",
          default_emitted != emitted)
    check("default build keeps eta_window at the shipped default",
          default_emitted.get("eta_window") == omni_core.DEFAULTS["eta_window"])

    preset_sampler = OmniSamplerSelect().build(
        **{**inputs, "variant": "I: SDE eta 0.3", "eta": 0.0})[0]
    check("select applies the preset row",
          dict(preset_sampler.sampler_function.keywords).get("eta") == 0.3)
    custom_sampler = OmniSamplerSelect().build(
        **{**inputs, "variant": "custom (use knobs)", "eta": 0.0})[0]
    check("select custom row keeps the knob value",
          dict(custom_sampler.sampler_function.keywords).get("eta") == 0.0)


class _FakeModel:
    load_device = "cpu"
    model_options = {}

    def get_model_object(self, name):
        assert name == "model_sampling"
        return object()          # not comfy.model_sampling.CONST -> eps family


class _FakeKSampler:
    def __init__(self, model, steps=12, **kwargs):
        self.sigmas = torch.linspace(1.0, 0.0, int(steps) + 1)


def _run_ksampler(inputs, variant):
    """Call OmniKSampler.sample with the model/sample pipeline mocked out.

    The mock captures the SAMPLER partial that the node hands to
    ``comfy.samplers.sample``; that is the boundary where a widget value either
    reaches ``sample_omni`` or is silently dropped.
    """
    captured = {}

    def fake_sample(model, noise, positive, negative, cfg, device, sampler, sigmas, **kwargs):
        captured["sampler"] = sampler
        captured["calls"] = captured.get("calls", 0) + 1
        return torch.zeros_like(noise)

    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    node = OmniKSampler()
    with mock.patch.object(comfy.samplers, "KSampler", _FakeKSampler), \
            mock.patch.object(comfy.samplers, "sample", fake_sample), \
            mock.patch.object(comfy.sample, "fix_empty_latent_channels",
                              lambda model, samples, *a: samples), \
            mock.patch.object(comfy.sample, "prepare_noise",
                              lambda samples, seed, batch_inds: torch.zeros_like(samples)), \
            mock.patch("latent_preview.prepare_callback", lambda model, steps: None):
        out, summary = node.sample(
            model=_FakeModel(), positive=None, negative=None, latent_image=latent,
            seed=0, steps=8, cfg=3.5, scheduler="simple", denoise=1.0,
            variant=variant, **inputs,
        )
    return captured["sampler"], out, summary, captured.get("calls", 0)


def test_ksampler_wiring():
    _, ksam = _defaults()
    inputs = {name: _sentinel(name, entry) for name, entry in ksam.items()
              if name not in KSAMPLER_IO and name != "variant"}
    sampler, out, summary, calls = _run_ksampler(inputs, "custom (use knobs)")
    emitted = dict(sampler.sampler_function.keywords)
    expected = _independent_expected(inputs)
    check("ksampler reached comfy.samplers.sample", calls == 1, f"{calls} call(s)")
    check("ksampler emits every knob into the SAMPLER",
          emitted == expected,
          f"missing={sorted(set(expected) - set(emitted))}, "
          f"extra={sorted(set(emitted) - set(expected))}")
    check("ksampler passes eta_window as a tuple",
          emitted.get("eta_window") == (0.2, 0.8),
          f"eta_window={emitted.get('eta_window')!r}")
    check("ksampler reports the changed eta window in its config",
          "eta_window=(0.2, 0.8)" in summary, summary)
    check("ksampler returns the latent", out["samples"].shape == (1, 4, 8, 8))
    again, _, _, _ = _run_ksampler(inputs, "custom (use knobs)")
    check("ksampler: the same values reproduce the same record",
          dict(again.sampler_function.keywords) == emitted)

    default_sampler, _, _, _ = _run_ksampler({}, "custom (use knobs)")
    default_emitted = dict(default_sampler.sampler_function.keywords)
    check("negative control: ksampler sentinel build differs from the default",
          default_emitted != emitted)
    check("ksampler default keeps eta_window at the shipped default",
          default_emitted.get("eta_window") == omni_core.DEFAULTS["eta_window"])

    preset_sampler, _, _, _ = _run_ksampler({"eta": 0.0}, "I: SDE eta 0.3")
    check("ksampler applies the preset row",
          dict(preset_sampler.sampler_function.keywords).get("eta") == 0.3)


def main():
    test_parity()
    test_positional_prefix()
    test_defaults()
    test_tooltips()
    test_resolver()
    test_select_wiring()
    test_ksampler_wiring()
    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
