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

_VARIANT_PRESETS = list(AB_PRESETS.keys()) + ["custom (use knobs)"]


def _parse_eta_window(value):
    """Return a valid (low, high) eta window, or the shipped default.

    The widget is a STRING because ComfyUI has no two-float tuple widget; the
    sampler wants a tuple.  Both nodes go through this function so a malformed
    string cannot reach ``sample_omni`` (which unpacks exactly two values).
    """
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            lo, hi = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return DEFAULTS["eta_window"]
        return (lo, hi) if lo < hi else DEFAULTS["eta_window"]
    try:
        parts = [float(x) for x in str(value).replace(";", ",").split(",") if x.strip()]
    except ValueError:
        parts = []
    if len(parts) == 2 and parts[0] < parts[1]:
        return (parts[0], parts[1])
    return DEFAULTS["eta_window"]


def _resolve_sampler_options(inputs):
    """Turn node widget values into ``sample_omni`` keyword arguments.

    One resolver for both nodes: the preset row is applied first, then the eta
    window is parsed, so the KSampler and the SAMPLER node cannot drift apart.
    Keys not in ``DEFAULTS`` (the KSampler's model/latent I/O, the ``variant``
    widget itself) are dropped; ``sample_omni``'s own defaults cover the rest.
    """
    opts = {k: v for k, v in inputs.items() if k in DEFAULTS}
    variant = inputs.get("variant")
    if variant in AB_PRESETS:
        opts.update(AB_PRESETS[variant])
    opts["eta_window"] = _parse_eta_window(opts.get("eta_window", DEFAULTS["eta_window"]))
    return opts


# ---------------------------------------------------------------------------
# Sampler widgets.  One definition per knob, used by BOTH nodes: the widget
# sets are kept equal by tools/test_node_parity.py.  A tooltip names every
# choice of every combo, because a tooltip is the only place a user can learn
# what a value does without reading omni_core.py.
# ---------------------------------------------------------------------------
_KNOB_SPECS = {
    "variant": (
        _VARIANT_PRESETS,
        {
            "default": _VARIANT_PRESETS[0],
            "tooltip": (
                "Preset row for A/B runs. A: shipped (v1.3.1) = no overrides; "
                "B: no corrector = corrector none; "
                "C: skip every 2nd eval = skip_every 2, order 2; "
                "C2: skip every 2nd, order 3 = skip_every 2, order 3; "
                "D: ppm gamma churn = churn_mode gamma, s_churn 2.0; "
                "E: CFG++ mild (0.5) = cfg_pp on, strength 0.5; "
                "F: CFG++ full (1.0) = cfg_pp on, strength 1.0; "
                "G: order 1; H: exact kernel = x0_average exact; "
                "I: SDE eta 0.3; J: no gate = min_step_ratio 0; "
                "K: per-pixel order = px_order on; "
                "L: region CFG map = cfg_map on; "
                "M: author pref (cfgpp+churn+eta+px) = cfg_pp on, gamma churn 100, eta 1.0, "
                "px_order on (the registered author-preference name). "
                "A preset overrides only the knobs it names; custom (use knobs) "
                "uses every knob as set."
            ),
        },
    ),
    "order": (
        "INT",
        {
            "default": DEFAULTS["order"],
            "min": 1,
            "max": 4,
            "step": 1,
            "tooltip": (
                "History order of the multistep rule: how many previous model "
                "evaluations it uses. exp: 1 = first order, 2 = the measured "
                "best; 3-4 are clamped to 2, because the third-order fit was "
                "measured and removed and the exp path keeps one previous x0. "
                "ab: 1-4 are available. 1 = first order, no history."
            ),
        },
    ),
    "integrator": (
        ["exp", "ab"],
        {
            "default": DEFAULTS["integrator"],
            "tooltip": (
                "exp = exponential integrator in logSNR time; correct for both "
                "epsilon and flow models, and the only path that supports eta, "
                "cfg_pp, corrector, x0_average and px_order. ab = Adams-Bashforth "
                "in sigma time (ipndm/deis style); deterministic only, with no "
                "exponential step and no OU noise, so eta, cfg_pp, corrector, "
                "x0_average and px_order do not apply. With eta > 0, cfg_pp on, "
                "or churn > 0, ab is ignored and the exp path runs."
            ),
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
            "tooltip": (
                "SDE strength (stochasticity). 0 = deterministic ODE. Above 0, "
                "noise is injected only inside eta_window; 1 is ancestral-like; "
                "larger values add more noise. The ab integrator has no SDE term, "
                "so pairing ab with eta > 0 falls back to exp."
            ),
        },
    ),
    "s_noise": (
        "FLOAT",
        {
            "default": DEFAULTS["s_noise"],
            "min": 0.0,
            "max": 100.0,
            "step": 0.01,
            "round": False,
            "tooltip": (
                "Multiplier on noise: SDE noise when eta > 0, and churn noise "
                "when s_churn > 0. 1.0 = nominal; 0 = no noise. It does not "
                "change the deterministic drift."
            ),
        },
    ),
    "corrector": (
        ["first", "auto", "none", "last", "first_last"],
        {
            "default": DEFAULTS["corrector"],
            "tooltip": (
                "Trapezoidal corrector: evaluates one extra model call at the "
                "predicted state and averages the two x0 predictions. first = "
                "step 0 only (measured best for epsilon models); last = the last "
                "real step; first_last = both; none = off; auto = first for "
                "epsilon models and none for flow models (heuristic). Skipped by "
                "the ab integrator and (unless corrector_cfgpp is on) with CFG++."
            ),
        },
    ),
    "corrector_cfgpp": (
        "BOOLEAN",
        {
            "default": DEFAULTS["corrector_cfgpp"],
            "tooltip": (
                "Only used when cfg_pp is on. Also run the trapezoidal corrector "
                "under CFG++: the extra model call re-evaluates both the guided "
                "x0 and the unconditional prediction, and both are averaged, so "
                "CFG++ stays a no-op when cond == uncond. OFF (default) is the "
                "v1.0 rule: the corrector is skipped under CFG++, because "
                "correcting x0 alone would shift the guidance. Costs +1 model "
                "call; at cfg = 1 that call is a 2x batch."
            ),
        },
    ),
    "lower_order_to_end": (
        "INT",
        {
            "default": DEFAULTS["lower_order_to_end"],
            "min": 0,
            "max": 5,
            "step": 1,
            "tooltip": (
                "Force first order for the last N real integration steps, to "
                "guard the large-step tail of Karras-like schedules. 0 = off; 1 "
                "is the default. The final denoise step (sigma = 0) is not counted."
            ),
        },
    ),
    "cfg_pp": (
        "BOOLEAN",
        {
            "default": DEFAULTS["cfg_pp"],
            "tooltip": (
                "CFG++ guidance: use the unconditional prediction as the noise "
                "estimate in the step. It needs an unconditional pass, so at "
                "cfg = 1 it disables ComfyUI's cfg1 shortcut and doubles the "
                "model batch (measured 17.5 s -> 35.5 s at 1024px, 10 steps; "
                "free at cfg > 1). It is disabled with cfg_map, and it adds "
                "guidance even at cfg = 1, so do not use it on distilled cfg-1 "
                "models. The ab integrator ignores it (exp runs)."
            ),
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
            "tooltip": (
                "Only used when cfg_pp is on. Blends the noise estimate between "
                "the guided prediction (0 = plain sampling, no CFG++ effect) and "
                "the unconditional prediction (1 = reference CFG++, measured to "
                "act like ~2-2.5x the cfg value). 0.5 is the shipped compromise "
                "(cfg 1 acts like ~1.5)."
            ),
        },
    ),
    "x0_average": (
        ["ab2", "exact"],
        {
            "default": DEFAULTS["x0_average"],
            "tooltip": (
                "Coefficient used to average x0 over a step when order >= 2: ab2 "
                "= DPM-Solver++(2M)'s h/2 coefficient (measured best); exact = the "
                "quadrature moment of this integrator's own kernel (exact when x0 "
                "is linear in logSNR, for both model families). Ignored at order 1 "
                "and by the ab integrator."
            ),
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
            "tooltip": (
                "Stability guard on the multistep extrapolation. The history fit "
                "is dropped to first order when the current step is more than "
                "1/min_step_ratio times the previous one. 0 = off; 0.5 = drop when "
                "a step is more than 2x the previous step (measured best). Applies "
                "in logSNR for exp and in sigma for ab."
            ),
        },
    ),
    "eta_window": (
        "STRING",
        {
            "default": "0.10,0.90",
            "tooltip": (
                "Trajectory window for SDE noise, as two fractions of the logSNR "
                "journey from start (0) to end (1): low,high (comma or semicolon "
                "separated). 0.10,0.90 skips the first and last 10%; the window "
                "edges are smoothed over 0.10 of the journey. Only used when "
                "eta > 0. Invalid input falls back to 0.10,0.90."
            ),
        },
    ),
    "s_churn": (
        "FLOAT",
        {
            "default": DEFAULTS["s_churn"],
            "min": 0.0,
            "max": 100.0,
            "step": 0.01,
            "round": False,
            "tooltip": (
                "Re-noise the state before the model call (churn). 0 = off. A "
                "stochasticity dial, not an accuracy knob; it costs no extra model "
                "call. The size follows churn_mode. On a flow model karras churn is "
                "capped at the schedule's first sigma, so it cannot evaluate above "
                "sigma = 1 (gamma already skips those steps). With integrator ab, "
                "churn forces the exp path (ab is deterministic)."
            ),
        },
    ),
    "churn_mode": (
        ["karras", "gamma"],
        {
            "default": DEFAULTS["churn_mode"],
            "tooltip": (
                "Law used when s_churn > 0: karras = sigma_hat = sigma * (1 + "
                "gamma), gamma capped at sqrt(2)-1 (same as euler/heun), and on a "
                "flow model sigma_hat capped at the schedule's first sigma; gamma = "
                "comfyui-ppm's euler_gamma law, sigma_hat = sigma + s_churn * "
                "sigma / sigma_max, skipped when it would exceed the schedule's "
                "first sigma. Ignored when s_churn = 0."
            ),
        },
    ),
    "noise_mode": (
        ["brownian", "white"],
        {
            "default": DEFAULTS["noise_mode"],
            "tooltip": (
                "Noise used by the SDE when eta > 0: brownian = Brownian-tree "
                "time-correlated noise (dpmpp_sde style); white = independent "
                "Gaussian noise each step (cheaper). Falls back to white if the "
                "Brownian tree cannot be built, e.g. when the schedule has no "
                "positive sigma span. Ignored when eta = 0."
            ),
        },
    ),
    "skip_every": (
        "INT",
        {
            "default": DEFAULTS["skip_every"],
            "min": 1,
            "max": 8,
            "step": 1,
            "tooltip": (
                "NFE skipping (experimental): 1 = evaluate every step; N > 1 = "
                "evaluate every Nth step and reconstruct the denoised prediction "
                "on the others by extrapolating epsilon from previous real "
                "evaluations. The first step and the final denoise step always "
                "evaluate. Fewer model calls, but the trajectory changes and "
                "quality is unproven."
            ),
        },
    ),
    "skip_order": (
        "INT",
        {
            "default": DEFAULTS["skip_order"],
            "min": 1,
            "max": 3,
            "step": 1,
            "tooltip": (
                "Only used when skip_every > 1. Order of the epsilon "
                "extrapolation on skipped steps: 1 = hold the last value, 2 = "
                "linear, 3 = quadratic (through the last 1-3 real evaluations). "
                "2 is the default; higher is sharper but can overshoot."
            ),
        },
    ),
    "final_extrapolate": (
        "FLOAT",
        {
            "default": DEFAULTS["final_extrapolate"],
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "round": False,
            "tooltip": (
                "On the final denoise step, blend x0 with an extrapolation of x0 "
                "to sigma = 0 (quadratic when 3 points are available). 0 = off; "
                "1 = full extrapolation, bounded to 50% of the x0 RMS. Available, "
                "but no measured gain; the default is 0."
            ),
        },
    ),
    "px_order": (
        "BOOLEAN",
        {
            "default": DEFAULTS["px_order"],
            "tooltip": (
                "Per-pixel order gate for the order-2 x0 extrapolation: only "
                "pixels whose Lagrange remainder is under px_order_tol keep order "
                "2; the rest run as order 1. No extra model call. OFF is "
                "bit-identical to the plain sampler. Needs order >= 2 and the exp "
                "integrator (ignored by ab). Under CFG++ the same gate is applied "
                "to the unconditional extrapolation, so the cond == uncond no-op "
                "survives. Experimental: neutral at the default tolerance, worse "
                "when tight."
            ),
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
            "tooltip": (
                "Only used when px_order is on. Allowed extrapolation error as a "
                "multiple of the local x0 RMS. Lower refuses more pixels and looks "
                "more like order 1; higher approaches the plain order-2 sampler. "
                "2.0 is the shipped safe fallback (measured neutral on Krea2 at 8 "
                "steps); 0.05 cost 0.014 SSIM."
            ),
        },
    ),
    "cfg_map": (
        "BOOLEAN",
        {
            "default": DEFAULTS["cfg_map"],
            "tooltip": (
                "Late per-region CFG: at cfg_map_start, measure the conditional/"
                "unconditional gap, pool it to 8x8 regions, then keep "
                "cfg_map_floor of the guidance delta in the lowest-gap region (up "
                "to the full delta in the highest). Needs cfg > 1 and is disabled "
                "with CFG++, cfg_map_floor = 1 or no CFG guider. OFF is "
                "bit-identical. Experimental."
            ),
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
            "tooltip": (
                "Only used when cfg_map is on. The fraction of the late guidance "
                "delta kept in the lowest-gap region; other regions keep "
                "proportionally more, up to 1.0. 1.0 disables the map. 0.85 is the "
                "shipped value and moved a MOSS @ 8-step image by 0.014 RMS (SSIM "
                "0.9824) - experimental, with no image-preference measurement yet."
            ),
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
            "tooltip": (
                "Only used when cfg_map is on. Trajectory fraction (0 = start, "
                "1 = end) at which the cond/uncond gap is measured; the map "
                "applies on the steps after that. 0.5 measures mid-run. In a short "
                "run a late start can mean the map never applies."
            ),
        },
    ),
}


def _knob(name):
    """Return a fresh INPUT_TYPES entry for one sampler knob."""
    kind, opts = _KNOB_SPECS[name]
    if isinstance(kind, list):
        return (list(kind), dict(opts))
    return (kind, dict(opts))


# The KSampler's own I/O.  These are the widgets a SAMPLER object cannot own,
# so they are the only keys allowed to differ between the two nodes.
_KSAMPLER_IO = {
    "model": ("MODEL", {"tooltip": "Diffusion model to sample."}),
    "positive": ("CONDITIONING", {"tooltip": "Conditioning that guides toward the prompt."}),
    "negative": (
        "CONDITIONING",
        {"tooltip": "Conditioning that guides away from the negative prompt. Ignored at cfg = 1."},
    ),
    "latent_image": (
        "LATENT",
        {"tooltip": "Latent to sample from. An empty latent is adapted to the model's "
                    "channel count and dimensionality automatically."},
    ),
    "seed": (
        "INT",
        {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
         "tooltip": "Random seed. The same seed and settings reproduce the same noise; "
                    "a different sampler can still change the result."},
    ),
    "steps": (
        "INT",
        {"default": 12, "min": 1, "max": 1000,
         "tooltip": "Number of sampling steps; the sampler receives steps + 1 sigmas."},
    ),
    "cfg": (
        "FLOAT",
        {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False,
         "tooltip": "Classifier-free guidance scale. 1 = no guidance; higher pushes "
                    "harder toward positive and away from negative."},
    ),
    "scheduler": (
        comfy.samplers.KSampler.SCHEDULERS,
        {"tooltip": "Sigma schedule: where the steps are placed along the trajectory. "
                    "simple = evenly spaced entries from the model's own sigma table "
                    "(ComfyUI's usual choice for flow models); sgm_uniform = uniform "
                    "in timesteps, SGM-style; karras = Karras et al. spacing, denser "
                    "near sigma 0; exponential = equal ratio in sigma; ddim_uniform = "
                    "uniform on the model's native sigma grid; beta = "
                    "beta-distribution spacing; normal = uniform in timesteps; "
                    "linear_quadratic = linear at low noise then quadratic; "
                    "kl_optimal = KL-optimal spacing in sigma."},
    ),
    "denoise": (
        "FLOAT",
        {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
         "tooltip": "Fraction of the trajectory to run: 1.0 = full generation from "
                    "noise; lower values refine an existing latent."},
    ),
}

# v1.5.1 widget order (read from git show b3a9081:omni_nodes.py).  New widgets
# are APPENDED: ComfyUI stores widget values positionally, so a kept set must
# stay a prefix of the old order or every saved workflow shifts its values.
_V151_SELECT_ORDER = [
    "order", "integrator", "eta", "s_noise", "corrector", "lower_order_to_end",
    "cfg_pp", "cfg_pp_strength", "x0_average", "min_step_ratio", "eta_window",
    "s_churn", "noise_mode", "final_extrapolate", "px_order", "px_order_tol",
    "cfg_map", "cfg_map_floor", "cfg_map_start",
]
_SELECT_ORDER = _V151_SELECT_ORDER + [
    "churn_mode", "skip_every", "skip_order", "variant", "corrector_cfgpp",
]

_V151_KSAMPLER_ORDER = [
    "variant", "order", "integrator", "corrector", "x0_average", "min_step_ratio",
    "lower_order_to_end", "eta", "s_noise", "cfg_pp", "cfg_pp_strength", "s_churn",
    "churn_mode", "noise_mode", "skip_every", "skip_order", "final_extrapolate",
    "px_order", "px_order_tol", "cfg_map", "cfg_map_floor", "cfg_map_start",
]
_KSAMPLER_ORDER = _V151_KSAMPLER_ORDER + ["eta_window", "corrector_cfgpp"]


class OmniSamplerSelect:
    """Build an Omni-Hybrid SAMPLER object for SamplerCustom / SamplerCustomAdvanced.

    Exposes the same sampler knobs as OmniKSampler, including the A/B preset
    switch; only the sampling I/O (model, conditioning, latent, seed, steps,
    cfg, scheduler, denoise) stays on the full KSampler node, because a SAMPLER
    object cannot own them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {name: _knob(name) for name in _SELECT_ORDER}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Omni-Hybrid sampler: exponential integrator in logSNR time (correct for both "
        "eps and flow models), variable-step x0 extrapolation with a stability guard, "
        "gated SDE noise, CFG++, churn, and the same optional controls as the full "
        "Omni-Hybrid KSampler node."
    )

    def build(self, **inputs):
        """Build the KSAMPLER. All inputs arrive by name from ComfyUI."""
        opts = _resolve_sampler_options(inputs)
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
        required = dict(_KSAMPLER_IO)
        required.update({name: _knob(name) for name in _KSAMPLER_ORDER})
        return {"required": required}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "config")
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Full Omni-Hybrid sampler: model, conditioning and latent inputs plus every "
        "Omni knob and the A/B preset switch. Returns the latent and a config string."
    )

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, scheduler, denoise,
               variant, **knobs):
        opts = _resolve_sampler_options({**knobs, "variant": variant})
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
        # The denominator is an estimate; the numerator is LAST_STATS["calls"],
        # measured by the sampler.  Count what the resolved corrector actually
        # adds: "last" is one call, "first_last" is two, and CFG++ suppresses
        # them unless corrector_cfgpp is on.  cfg_pp_strength == 0 is plain
        # sampling, so it must not suppress the corrector.
        corr_steps = 0
        if corr in ("first", "first_last"):
            corr_steps += 1
        if corr in ("last", "first_last") and n >= 3:
            corr_steps += 1
        cfgpp_on = (bool(opts.get("cfg_pp", False))
                    and float(opts.get("cfg_pp_strength", DEFAULTS["cfg_pp_strength"])) > 0.0)
        if cfgpp_on and not opts.get("corrector_cfgpp", False):
            corr_steps = 0
        extra = corr_steps
        delta = {k: v for k, v in opts.items() if DEFAULTS.get(k) != v}
        spatial = ""
        if opts.get("px_order") or opts.get("cfg_map"):
            spatial = " | " + " ".join(
                f"{k}={omni_core.LAST_STATS[k]}" for k in sorted(omni_core.LAST_STATS)
                if k.startswith(("px_order", "cfg_map")))
        effective = ""
        if (omni_core.LAST_STATS.get("order_requested") is not None
                and omni_core.LAST_STATS.get("order_effective")
                != omni_core.LAST_STATS.get("order_requested")):
            effective = (f" | order {omni_core.LAST_STATS['order_requested']}"
                         f"->{omni_core.LAST_STATS['order_effective']}")
        if (str(opts.get("integrator", "exp")).lower() == "ab"
                and omni_core.LAST_STATS.get("integrator") == "exp"):
            effective += " | ab->exp"
        # Churn is the one knob whose effect depends on the schedule and the model
        # family (the gamma law can fire on 0 of n steps), so report what actually
        # ran rather than only the requested value.
        churn = ""
        if float(opts.get("s_churn", DEFAULTS["s_churn"])) > 0.0:
            churn = f" | churn {omni_core.LAST_STATS.get('churn_steps', 0)}/{n} steps"
            if omni_core.LAST_STATS.get("churn_capped"):
                churn += f" ({omni_core.LAST_STATS['churn_capped']} capped)"
        summary = (
            f"variant={variant} flow={flow} steps={n} cfg={cfg} scheduler={scheduler} "
            f"evals={omni_core.LAST_STATS.get('calls', evals + extra)}"
            f"/{evals + extra} | " + ", ".join(f"{k}={v}" for k, v in sorted(delta.items()))
            + spatial
            + effective
            + churn
            + (" | cfg++ batch x2 (cfg=1 uncond pass forced)" if opts.get("cfg_pp") and cfg <= 1.0
               else " | cfg++ free (uncond already evaluated)" if opts.get("cfg_pp") else "")
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
