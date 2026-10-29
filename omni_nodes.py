"""ComfyUI node + sampler-name registration for the Omni-Hybrid sampler."""

from __future__ import annotations

import functools

import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.k_diffusion.sampling as kds

from .omni_core import AB_PRESETS, DEFAULTS, VARIANTS, sample_omni

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
                    ["first", "auto", "none", "last", "first_last"],
                    {
                        "tooltip": "Trapezoidal corrector on selected steps (one extra model call each). "
                                   "auto = first for eps, none for flow (heuristic; the flow evidence "
                                   "was invalidated by mode bifurcation, see CORRECTIONS.md).",
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


class OmniKSampler:
    """Full KSampler with every Omni knob, plus an A/B preset switch.

    `variant` picks a preset row from omni_core.AB_PRESETS; "custom" uses the
    knobs as set. The effective configuration is returned as a STRING so the
    A/B runs can be told apart in the metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        presets = list(AB_PRESETS.keys()) + ["custom (use knobs)"]
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variant": (
                    presets,
                    {"tooltip": "A/B switch: pick a preset row, or 'custom' to use the knobs below. "
                                "Presets override only the settings they mention."},
                ),
                "order": ("INT", {"default": DEFAULTS["order"], "min": 1, "max": 4}),
                "integrator": (["exp", "ab"], {"default": DEFAULTS["integrator"]}),
                "corrector": (["first", "auto", "none", "last", "first_last"],
                              {"default": DEFAULTS["corrector"]}),
                "x0_average": (["ab2", "exact"], {"default": DEFAULTS["x0_average"]}),
                "min_step_ratio": ("FLOAT", {"default": DEFAULTS["min_step_ratio"], "min": 0.0,
                                             "max": 1.0, "step": 0.05, "round": False}),
                "lower_order_to_end": ("INT", {"default": DEFAULTS["lower_order_to_end"], "min": 0, "max": 5}),
                "eta": ("FLOAT", {"default": DEFAULTS["eta"], "min": 0.0, "max": 4.0,
                                  "step": 0.01, "round": False}),
                "s_noise": ("FLOAT", {"default": DEFAULTS["s_noise"], "min": 0.0, "max": 100.0,
                                      "step": 0.01, "round": False}),
                "cfg_pp": ("BOOLEAN", {"default": DEFAULTS["cfg_pp"]}),
                "cfg_pp_strength": ("FLOAT", {"default": DEFAULTS["cfg_pp_strength"], "min": 0.0,
                                              "max": 1.0, "step": 0.05, "round": False}),
                "s_churn": ("FLOAT", {"default": DEFAULTS["s_churn"], "min": 0.0, "max": 100.0,
                                      "step": 0.01, "round": False}),
                "churn_mode": (["karras", "gamma"], {"default": DEFAULTS["churn_mode"]}),
                "noise_mode": (["brownian", "white"], {"default": DEFAULTS["noise_mode"]}),
                "skip_every": ("INT", {"default": DEFAULTS["skip_every"], "min": 1, "max": 8,
                                       "tooltip": "1 = evaluate every step. 2 = evaluate every 2nd step and "
                                                  "extrapolate epsilon on the others (experimental)."}),
                "skip_order": ("INT", {"default": DEFAULTS["skip_order"], "min": 1, "max": 3}),
                "final_extrapolate": ("FLOAT", {"default": DEFAULTS["final_extrapolate"], "min": 0.0,
                                                "max": 1.0, "step": 0.05, "round": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "config")
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, scheduler, denoise,
               variant, **knobs):
        opts = {k: v for k, v in knobs.items() if k in DEFAULTS}
        if variant in AB_PRESETS:
            opts.update(AB_PRESETS[variant])
        ms = model.get_model_object("model_sampling")
        flow = isinstance(ms, comfy.model_sampling.CONST)
        corr = str(opts.get("corrector", DEFAULTS["corrector"]))
        if corr == "auto":
            corr = "none" if flow else "first"

        noise = comfy.sample.prepare_noise(latent_image["samples"], seed)
        ks = comfy.samplers.KSampler(
            model, steps=steps, device=model.load_device, sampler="euler", scheduler=scheduler,
            denoise=denoise, model_options=model.model_options,
        )
        sampler = comfy.samplers.KSAMPLER(functools.partial(sample_omni, **opts))
        samples = comfy.samplers.sample(
            model, noise, positive, negative, cfg, model.load_device, sampler, ks.sigmas,
            model_options=model.model_options, latent_image=latent_image["samples"],
            disable_pbar=False, seed=seed,
        )

        n = max(len(ks.sigmas) - 1, 1)
        sk = int(opts.get("skip_every", 1))
        # exactly mirror sample_omni's skip conditions (protections included)
        evals = sum(
            1 for i in range(n)
            if not (sk > 1 and i > 0 and i % sk != 0
                    and i >= int(DEFAULTS["skip_protect_first"]) and i <= n - 1 - int(DEFAULTS["skip_protect_last"]))
        )
        extra = 1 if (corr in ("first", "first_last") and not opts.get("cfg_pp", False)) else 0
        delta = {k: v for k, v in opts.items() if DEFAULTS.get(k) != v}
        summary = (
            f"variant={variant} flow={flow} steps={n} cfg={cfg} scheduler={scheduler} "
            f"evals~{evals + extra} | " + ", ".join(f"{k}={v}" for k, v in sorted(delta.items()))
        )
        print(f"[OmniKSampler] {summary}")
        return ({"samples": samples}, summary)


NODE_CLASS_MAPPINGS = {"OmniSamplerSelect": OmniSamplerSelect, "OmniKSampler": OmniKSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSamplerSelect": "Omni-Hybrid Sampler (select)",
    "OmniKSampler": "Omni-Hybrid KSampler (A/B)",
}


