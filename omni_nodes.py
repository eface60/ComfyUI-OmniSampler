"""ComfyUI node + sampler-name registration for the Omni-Hybrid sampler."""

from __future__ import annotations

import functools

import comfy.samplers
import comfy.k_diffusion.sampling as kds

from .omni_core import DEFAULTS, VARIANTS, sample_omni

CATEGORY = "sampling/samplers"


class OmniSamplerSelect:
    """Build an Omni-Hybrid SAMPLER object for SamplerCustom / SamplerCustomAdvanced."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "order": (
                    "INT",
                    {
                        "default": DEFAULTS["order"],
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "History order. For the exponential integrator 2 is the measured "
                                   "best (3+ are limited by the first-order start-up step). For the "
                                   "Adams-Bashforth integrator 1-4 are available.",
                    },
                ),
                "integrator": (
                    ["exp", "ab"],
                    {
                        "default": DEFAULTS["integrator"],
                        "tooltip": "exp = exponential integrator in logSNR (correct for both eps and "
                                   "flow models; supports eta / cfg_pp / corrector). ab = Adams-"
                                   "Bashforth in sigma (ipndm/deis style; deterministic only).",
                    },
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": DEFAULTS["eta"],
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Stochasticity. 0 = deterministic ODE. The noise is injected "
                                   "only in the middle of the trajectory (see eta_window).",
                    },
                ),
                "s_noise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False},
                ),
                "corrector": (
                    ["auto", "first", "none", "last", "first_last"],
                    {
                        "tooltip": "Trapezoidal corrector on selected steps (one extra model call each). "
                                   "auto = first for eps models, none for flow models (measured).",
                    },
                ),
                "lower_order_to_end": (
                    "INT",
                    {
                        "default": DEFAULTS["lower_order_to_end"],
                        "min": 0,
                        "max": 5,
                        "step": 1,
                        "tooltip": "Drop to first order for the last N real steps (guards the "
                                   "large-step tail of Karras-like schedules).",
                    },
                ),
                "cfg_pp": (
                    "BOOLEAN",
                    {
                        "default": DEFAULTS["cfg_pp"],
                        "tooltip": "CFG++ guidance: use the unconditional prediction as the noise "
                                   "estimate. Costs an extra model call per step when cfg == 1.",
                    },
                ),
                "cfg_pp_strength": (
                    "FLOAT",
                    {
                        "default": DEFAULTS["cfg_pp_strength"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": "How much of the unconditional prediction is used as the noise "
                                   "estimate. 0 = plain sampling (cfg keeps its usual meaning), "
                                   "1 = reference CFG++ (measured: behaves like ~2-2.5x the cfg "
                                   "value). 0.5 is the measured compromise.",
                    },
                ),
                "x0_average": (
                    ["ab2", "exact"],
                    {
                        "tooltip": "ab2 = DPM-Solver++(2M) coefficient (measured best); exact = "
                                   "quadrature kernel moment (exact for x0 linear in logSNR).",
                    },
                ),
                "min_step_ratio": (
                    "FLOAT",
                    {
                        "default": DEFAULTS["min_step_ratio"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": "Disable extrapolation when the current logSNR step is more "
                                   "than 1/ratio times the previous one (stability guard, 0 = off).",
                    },
                ),
                "eta_window": (
                    "STRING",
                    {
                        "default": "0.10,0.90",
                        "tooltip": "Trajectory window (fraction of logSNR travel) where noise is injected.",
                    },
                ),
                "s_churn": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False},
                ),
                "noise_mode": (["brownian", "white"],),
                "final_extrapolate": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": False},
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Omni-Hybrid sampler: exponential integrator in logSNR time (correct for both "
        "eps and flow models), variable-step x0 extrapolation with a stability guard, "
        "gated SDE noise, CFG++ and churn."
    )

    def build(self, **inputs):
        """All inputs come in by name from ComfyUI; unknown ones are ignored."""
        eta_window = inputs.get("eta_window", DEFAULTS["eta_window"])
        try:
            parts = [float(x) for x in str(eta_window).replace(";", ",").split(",") if x.strip()]
        except ValueError:
            parts = []
        window = (parts[0], parts[1]) if len(parts) == 2 and parts[0] < parts[1] else DEFAULTS["eta_window"]
        opts = dict(
            order=int(inputs.get("order", DEFAULTS["order"])),
            integrator=str(inputs.get("integrator", DEFAULTS["integrator"])),
            eta=float(inputs.get("eta", DEFAULTS["eta"])),
            s_noise=float(inputs.get("s_noise", DEFAULTS["s_noise"])),
            corrector=str(inputs.get("corrector", DEFAULTS["corrector"])),
            lower_order_to_end=int(inputs.get("lower_order_to_end", DEFAULTS["lower_order_to_end"])),
            cfg_pp=bool(inputs.get("cfg_pp", DEFAULTS["cfg_pp"])),
            cfg_pp_strength=float(inputs.get("cfg_pp_strength", DEFAULTS["cfg_pp_strength"])),
            x0_average=str(inputs.get("x0_average", DEFAULTS["x0_average"])),
            min_step_ratio=float(inputs.get("min_step_ratio", DEFAULTS["min_step_ratio"])),
            eta_window=window,
            s_churn=float(inputs.get("s_churn", DEFAULTS["s_churn"])),
            noise_mode=str(inputs.get("noise_mode", DEFAULTS["noise_mode"])),
            final_extrapolate=float(inputs.get("final_extrapolate", DEFAULTS["final_extrapolate"])),
        )
        return (comfy.samplers.KSAMPLER(functools.partial(sample_omni, **opts)),)


def _make_registered(name: str, overrides: dict):
    preset = dict(DEFAULTS)
    preset.update(overrides)

    def sampler_fn(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        opts = dict(preset)
        opts.update(kwargs)
        return sample_omni(
            model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, **opts
        )

    sampler_fn.__name__ = f"sample_{name}"
    sampler_fn.__doc__ = f"Omni-Hybrid preset '{name}' with overrides {overrides}"
    return sampler_fn


def register_samplers():
    """Expose the presets in the stock KSampler / KSamplerSelect dropdowns."""
    for name, overrides in VARIANTS.items():
        fn = _make_registered(name, overrides)
        setattr(kds, f"sample_{name}", fn)
        for lst in (
            comfy.samplers.KSAMPLER_NAMES,
            comfy.samplers.SAMPLER_NAMES,
            comfy.samplers.KSampler.SAMPLERS,
        ):
            if name not in lst:
                lst.append(name)


NODE_CLASS_MAPPINGS = {"OmniSamplerSelect": OmniSamplerSelect}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSamplerSelect": "Omni-Hybrid Sampler (select)"}

