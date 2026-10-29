# Changelog

## v1.0 — 2026-10-29

First tagged version. Installed as `custom_nodes/ComfyUI-OmniSampler`
(a junction to this repository, so the repo *is* what ComfyUI loads).

### Features

* One integrator that is correct for both model families: exponential step in
  half-logSNR time (`lambda = log(alpha/sigma)`), so flow-matching models use
  `h = log(alpha_t*sigma_s/(alpha_s*sigma_t))` instead of the eps-only
  `log(sigma_s/sigma_t)` that `dpmpp_2m`/`lms`/`ipndm`/`deis`/`res_multistep`
  hard-code.
* Variable-step `x0` averaging (`ab2`, DPM++(2M) coefficient) with an optional
  exact-kernel quadrature variant (`exact`), plus a step-ratio stability guard.
* `eta`-gated SDE noise (OU-consistent exponential drift) injected only in the
  middle of the trajectory; Brownian-tree or white noise.
* CFG++ guidance, with `cfg_pp_strength` (0 = plain sampling, 0.5 = default,
  1 = reference CFG++).
* Step-0 trapezoidal corrector (+1 NFE), order reduction on the last real step,
  Karras churn, optional endpoint extrapolation, Adams-Bashforth integrator.
* Registered sampler names: `omni_hybrid`, `omni_hybrid_sde`,
  `omni_hybrid_cfg_pp`; node `OmniSamplerSelect` exposes every knob.
* `tools/`: numerical verification (`verify_math.py`, `cfgpp_check.py`,
  `test_node.py`) and benchmarks (`toy_bench.py`, `sd_bench.py`, probes).

### Fixes in this version (found by the verification suite)

* **CFG++ guidance was too strong.** The multistep `x0` average was applied to
  the conditional prediction but not to the unconditional one, so
  `alpha_t*kappa*(dx0 - du)` was injected straight into the guidance direction.
  Symptom: `cfg=1` behaved like `cfg~4-5` on the default (order 2) path, and the
  case where CFG++ must be a no-op (`cond == uncond`) deviated by 16%.
  Now both predictions get the same average (`hist_u`), the corrector is skipped
  in CFG++ mode, and the uncond value is restored after the corrector's extra
  evaluation. `cond == uncond` reduces to plain sampling to 2.6e-7 when the
  multi-step path is enabled; `cfg=1` now behaves like `cfg~2.5` at strength 1.
* CFG++ now matches ComfyUI's `euler_cfg_pp` to 2.6e-7 / 5.9e-7 / 5.0e-7 at
  cfg 1 / 2 / 4 (order 1).
* `lower_order_to_end` used to affect only the final denoise step; it now targets
  the last N real integration steps.
* 3rd-order multistep was implemented, measured (2.09 vs 2.87 with exact
  history), and removed: the first-order start-up step dominates at practical
  step counts.
