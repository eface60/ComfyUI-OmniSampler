"""ComfyUI node + sampler-name registration for the Omni-Hybrid sampler."""

from __future__ import annotations

import functools

import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.k_diffusion.sampling as kds

from . import omni_core
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
                # ---- spatial controls (appended last: positional safety) ----
                "px_order": (
                    "BOOLEAN",
                    {
                        "default": DEFAULTS["px_order"],
                        "tooltip": "Per-pixel order: give the order-2 x0 extrapolation only to "
                                   "pixels whose measured curvature keeps its error under "
                                   "px_order_tol. Curved pixels fall back to order 1. No extra "
                                   "model calls. OFF is bit-identical to the plain sampler.",
                    },
                ),
                "px_order_tol": (
                    "FLOAT",
                    {
                        "default": DEFAULTS["px_order_tol"],
                        "min": 0.001,
                        "max": 20.0,
                        "step": 0.001,
                        "round": False,
                        "tooltip": "Only used when px_order is on. The allowed extrapolation error "
                                   "relative to the local x0 size. Lower refuses more pixels and "
                                   "looks more like order 1; higher approaches the plain global "
                                   "order-2 sampler. 2.0 is a safe fallback: at 8 steps on Krea2 it "
                                   "scored the same as the plain sampler, while 0.05 cost 0.014 SSIM.",
                    },
                ),
                "cfg_map": (
                    "BOOLEAN",
                    {
                        "default": DEFAULTS["cfg_map"],
                        "tooltip": "Late per-region CFG: measure the cond/uncond gap at "
                                   "cfg_map_start of the trajectory, then give the lowest-gap "
                                   "regions cfg_map_floor of the guidance delta on later steps. "
                                   "Needs cfg > 1 and is disabled with CFG++. OFF is bit-identical.",
                    },
                ),
                "cfg_map_floor": (
                    "FLOAT",
                    {
                        "default": DEFAULTS["cfg_map_floor"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Only used when cfg_map is on. The fraction of the late "
                                   "guidance delta kept in the lowest-gap region. 1.0 disables "
                                   "the map; 0.85 is the PD-AOS shipped floor.",
                    },
                ),
                "cfg_map_start": (
                    "FLOAT",
                    {
                        "default": DEFAULTS["cfg_map_start"],
                        "min": 0.0,
                        "max": 0.95,
                        "step": 0.05,
                        "round": False,
                        "tooltip": "Only used when cfg_map is on. Trajectory fraction (0 = start, "
                                   "1 = end) at which the map is measured and from which it is "
                                   "applied. 0.5 measures at mid-run.",
                    },
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
            px_order=bool(inputs.get("px_order", DEFAULTS["px_order"])),
            px_order_tol=float(inputs.get("px_order_tol", DEFAULTS["px_order_tol"])),
            cfg_map=bool(inputs.get("cfg_map", DEFAULTS["cfg_map"])),
            cfg_map_floor=float(inputs.get("cfg_map_floor", DEFAULTS["cfg_map_floor"])),
            cfg_map_start=float(inputs.get("cfg_map_start", DEFAULTS["cfg_map_start"])),
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
                "px_order": ("BOOLEAN", {"default": DEFAULTS["px_order"],
                                         "tooltip": "Per-pixel order gate (see Omni-Hybrid Sampler (select))."}),
                "px_order_tol": ("FLOAT", {"default": DEFAULTS["px_order_tol"], "min": 0.001,
                                           "max": 20.0, "step": 0.001, "round": False}),
                "cfg_map": ("BOOLEAN", {"default": DEFAULTS["cfg_map"],
                                        "tooltip": "Late per-region CFG map (needs cfg > 1)."}),
                "cfg_map_floor": ("FLOAT", {"default": DEFAULTS["cfg_map_floor"], "min": 0.0,
                                             "max": 1.0, "step": 0.01, "round": False}),
                "cfg_map_start": ("FLOAT", {"default": DEFAULTS["cfg_map_start"], "min": 0.0,
                                             "max": 0.95, "step": 0.05, "round": False}),
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

        # Mirror nodes.py::common_ksampler: adapt an empty latent to this model's
        # latent format (channel count / 5-D axis for video formats / downscale
        # ratios). Without this a Krea2 model receives an unadapted latent and
        # dies in its patch-embedding ("mat1 and mat2 shapes cannot be multiplied").
        import latent_preview

        latent = dict(latent_image)
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)
        callback = latent_preview.prepare_callback(model, steps)

        ks = comfy.samplers.KSampler(
            model, steps=steps, device=model.load_device, sampler="euler", scheduler=scheduler,
            denoise=denoise, model_options=model.model_options,
        )
        sampler = comfy.samplers.KSAMPLER(functools.partial(sample_omni, **opts))
        samples = comfy.samplers.sample(
            model, noise, positive, negative, cfg, model.load_device, sampler, ks.sigmas,
            model_options=model.model_options, latent_image=latent_samples,
            denoise_mask=noise_mask, callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
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
        spatial = ""
        if opts.get("px_order") or opts.get("cfg_map"):
            spatial = " | " + " ".join(
                f"{k}={omni_core.LAST_STATS[k]}" for k in sorted(omni_core.LAST_STATS)
                if k.startswith(("px_order", "cfg_map")))
        summary = (
            f"variant={variant} flow={flow} steps={n} cfg={cfg} scheduler={scheduler} "
            f"evals={omni_core.LAST_STATS.get('calls', evals + extra)}"
            f"/{evals + extra} | " + ", ".join(f"{k}={v}" for k, v in sorted(delta.items()))
            + spatial
        )
        print(f"[OmniKSampler] {summary}")
        out = dict(latent)
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return (out, summary)


NODE_CLASS_MAPPINGS = {"OmniSamplerSelect": OmniSamplerSelect, "OmniKSampler": OmniKSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSamplerSelect": "Omni-Hybrid Sampler (select)",
    "OmniKSampler": "Omni-Hybrid KSampler (A/B)",
}



