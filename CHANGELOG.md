# Changelog

Newest first. Every entry says what changed and, where it was measured, the
number behind it; the scripts that produce those numbers are in `tools/`.

## Unreleased

### Fixed

* **Karras churn evaluated a flow model above its `sigma` domain.** A flow model
  is parameterised on `sigma < 1` (`alpha = 1 - sigma`), and step 0 of a full
  denoise already sits at `sigma_max = 1`, so `sigma_hat = 1.414` was noised for
  1.414 while `clamp_sigma` pinned the step's geometry at `1 - 1e-6`: the step
  divided by 1.0 instead of 1.414 and the state came out at 1.30 sigma instead of
  0.92, i.e. the injected noise was never removed (+41% at the `sqrt(2)-1` cap),
  in the step that matters most. The evaluation sigma is now capped at the
  schedule's first sigma, the guard the gamma law already had; when
  `sigma_s == raw[0]` churn is a bit-exact no-op, and steps below the cap churn
  normally. MOSS 768², cfg 3.5, `s_churn=8` (the worst case: gamma at its
  `sqrt(2)-1` cap), SSIM to a 40-step euler reference: 0.226 -> 0.511 at 8 steps
  and 0.288 -> 0.385 at 12. The control that skips step 0 entirely scored
  0.399 / 0.415, so the corruption is gone and what is left is ordinary churn
  noise. eps models are unchanged: no upper domain, so an SDXL schedule still
  evaluates at `sigma_hat = 20.7` on step 0, like k-diffusion.
  `LAST_STATS` gained `churn_steps` (steps that injected noise, both laws) and
  `churn_capped` (steps the flow cap reduced), and the A/B node prints
  `churn k/n steps` in its config string, so a run can say whether its churn
  fired at all.
* **`px_order` × CFG++ broke the CFG++ no-op invariant.** The per-pixel gate
  masked the x0 extrapolation but not the unconditional one, so every pixel the
  gate refused leaked an unmasked `kappa*(du)` term into the guidance direction
  — the v1.0 defect at per-pixel granularity. The same mask is now applied to
  both histories. On the analytic toy (`cond == uncond`, 12-step Karras, eps):
  7.63e-2 relative deviation at `px_order_tol=0.05` and 6.06e-3 at the shipped
  2.0 before the fix; 2.3e-7 and 1.9e-7 after, against 2.6e-7 with `px_order`
  off. `tools/cfgpp_check.py` now re-checks the invariant with `px_order` on.
  `omni_hybrid_cfgpp_author_pref` enables both features.
* **`order` 3–4 was inert on the `exp` path.** `hist_x0` keeps one previous x0,
  so `order=3/4` was bit-identical to `order=2` while the tooltip described the
  higher modes. `order` is now clamped to 2 on `exp`; `LAST_STATS` records
  `order_requested`/`order_effective` and the A/B node prints the clamp. `ab`
  still uses 1–4.
* **`ab` + churn divided by the wrong sigma.** Churn moves the model evaluation
  to `sigma_hat`, but the sigma-space derivative divided by the un-churned
  `sigma` (up to `1+gamma` = 1.414x high, gamma capped at `sqrt(2)-1`). `ab` is
  deterministic-only, so `s_churn > 0` now falls back to `exp` like `eta` and
  `cfg_pp`; `LAST_STATS["integrator"]` reports the path actually used.
* **`final_extrapolate` paired churned x0 values with un-churned sigmas.** The
  endpoint fit now receives the sigmas the model actually saw, recorded per
  step; with churn off those are the raw schedule entries, so the default path
  is unchanged.

### Added

* `corrector_cfgpp` (default false) on both nodes: run the trapezoidal corrector
  under CFG++ by averaging the extra call's unconditional prediction with the
  same trapezoid as x0. The post-CFG hook already captures that `u`; OFF keeps
  the v1.0 rule (the corrector is skipped under CFG++). `cond == uncond` stays a
  no-op to 2.2e-7 (eps) / 2.5e-7 (flow); +1 NFE (2x batch at cfg = 1).
* `M: author pref (cfgpp+churn+eta+px)` A/B row: the registered
  `omni_hybrid_cfgpp_author_pref` setting as one row, so the combination can be
  compared with its parts at equal NFE.
* `tools/test_cross_features.py`: the order clamp, the `ab` + churn fallback, the
  endpoint-fit sigmas and the flow churn cap, each with a negative control (the
  churn case records every evaluation sigma, asserts a one-step flow run is a
  bit-exact no-op, and checks that eps still evaluates above `sigma_max`).

### Changed

* `tools/local_metric.py`: the refined reference was read one node before the
  shared endpoint (`substeps - 1` instead of `substeps`), and it used a fixed 16
  substeps. The endpoint index is fixed, substeps now scale as
  `max(--substeps, ceil(h / 0.2))`, and named deterministic variants
  (`first,none,cfg_pp,cfg_pp_px,cfg_pp_corr,...`) replace the corrector-only
  switch. It also repeats the first arm and its refined reference and refuses to
  write a JSON unless both are bit-identical; the output name carries the
  variant list. `docs/CORRECTIONS.md` records the superseded table, the
  reproducibility failure, and the retraction.

### Measured

* **The local-error metric cannot resolve the corrector on MOSS, and the run
  that looked like a win is retracted.** With the endpoint/substep fix the MOSS
  table gave `first` 1.391e-1 vs `none` 1.533e-1 at interval 0 and a 0.8% mean
  difference on intervals 1-5. A repeat check then falsified the instrument for
  this model: two identical coarse runs from the same latent moved the final
  x0 by 7.3% (zero start) and 50-67% (unit-variance start) at 512², and the
  768² 51-substep reference repeated at 11.7%. Krea2 fails the same gate
  (coarse 2.83e-1, reference 1.18e-1 at 10 steps, 512², cfg 1), so both modern
  flow models are non-reproducible in this environment. The 1.4e-2 arm
  difference is far below that drift, so no corrector conclusion is admissible
  from these rows; the old "neutral within 1%" table is retracted as well.
  `tools/local_metric.py` now repeats the first arm and its refined reference
  and refuses to write a JSON unless both are bit-identical.
* **Corrector under CFG++ (same instrument, same model).** `cfg_pp` vs
  `cfg_pp_corr`: six-interval mean 2.864e-1 vs 2.868e-1; interval 0 2.384e-1 vs
  2.239e-1; intervals 2-3 worse by ~1.7e-2. The same repeat failure applies, so
  this is recorded as NOT RESOLVED — neither a win nor a loss.

### Repository

* Documentation split by audience: `README.md` is now a short landing page
  (what it is, install, which sampler name to pick), and everything technical —
  the integrator, every knob, the benchmarks, the model-family findings, the
  spatial controls and how to run the harness — moved to
  `docs/TECHNICAL.md`. Pointers across the docs and code comments follow it.
* Documentation reorganized: `ANALYSIS.md`, `AB_TESTING.md`, `CORRECTIONS.md`,
  `EXPERIMENTS.md` and `PD_AOS_AUDIT.md` now live in `docs/`; `docs/README.md`
  indexes the evidence and `tools/README.md` indexes the harness.
* Added `LICENSE` (MIT) and `CONTRIBUTING.md`.
* Fixed double-encoded UTF-8 in the Markdown files: the README's directory tree,
  the em dashes in several headings and the section symbols in
  `docs/PD_AOS_AUDIT.md` had been written as mojibake. Byte-order marks and
  CRLF line endings removed from all text files.
* The harness reads `COMFY_DIR`, `OMNI_WORK`, `PD_AOS_DIR` and `AFLOPS_OUT`
  instead of hard-coded absolute paths. The defaults are unchanged, so existing
  runs on the author's machine behave identically.

The default build is behaviorally unchanged: `corrector_cfgpp` is off, `order`
defaults to 2, churn defaults to 0 and `final_extrapolate` to 0. The only
default-path difference is that `order=3/4` no longer pretends to run.

## v1.5.5 — 2026-11-01

### Added

* `omni_hybrid_cfgpp_author_pref`, a fourth registered sampler name in the
  stock KSampler / KSamplerAdvanced / KSamplerSelect dropdowns. It is the
  author's own everyday A/B-node setting: CFG++ on, gamma churn at the slider
  maximum (`s_churn=100`), `eta=1`, and `px_order` on; every other knob is the
  shipped default. Provenance: chosen by eye across the author's own runs, not
  a measured optimum — no benchmark covers this combination, and the churn,
  eta and `px_order` measurements in `docs/TECHNICAL.md` were taken at other
  values.
* `tools/test_node.py` now proves the preset's delta is exactly those five
  knobs, that `sample_omni` receives `DEFAULTS` plus that delta, and that
  `px_order` and CFG++'s corrector suppression are in force on a toy run. The
  toy noise comes from its own generator, so a stochastic test cannot shift the
  existing end-to-end numbers.

### Notes

* The three existing sampler names are unchanged; this adds a dropdown entry
  and touches no default.

## v1.5.4 — 2026-10-31

### Code audit: CFG++ does not run the unconditional twice at cfg > 1

Claim to test: "at cfg 5 both arms do cond+uncond, but cfg_pp is still slower, so
the unconditional is probably run twice". Reading the path:

* `omni_core.py:595` installs the post-CFG hook only for `cfg_pp`/`cfg_map`; it
  copies `uncond_denoised` and returns `denoised` untouched. No model call.
* `omni_core.py:628` sets `disable_cfg1_optimization`, which
  `comfy/samplers.py::sampling_function` consults only when
  `math.isclose(cond_scale, 1.0)`; at cfg 5 the unconditional is passed either
  way, so the flag is a no-op there.
* The step body (`omni_core.py:874`) only blends `u`; the sampler's one other
  model call (the corrector) is skipped when CFG++ is active
  (`omni_core.py:905`), so CFG++ at cfg > 1 makes **one fewer** evaluation.
* model_options are cloned per run (`inner_sample` ->
  `create_model_options_clone`), so the flag cannot leak into another pass.

Forward counts (`apply_model`, not estimates): MOSS cfg 3.5 -> 16/16,
cfg 5 -> 16/16; Krea2 cfg 5 -> 16/16. Only cfg = 1 changes: 8 -> 16.

Interleaved repeats, Krea2 @768, 8 steps, cfg 5, 3 reps each:
plain median 10.10 s (runs 14.37/10.10/7.71), cfg_pp median 8.72 s
(runs 14.51/7.55/8.72), ratio 0.86. The per-run spread on identical work is
7.7-14.5 s, which is what produced the earlier 38% single-shot difference.

### Fixed

* `cfg_pp_strength = 0` made CFG++ exactly plain sampling but still installed the
  hook, and that forced the unconditional pass at cfg = 1 (2x cost for nothing).
  All CFG++ decisions now go through `cfgpp_active = cfg_pp and strength > 0`;
  with strength 0 no hook is installed. Verified bit-identical to plain
  (max diff 0.0) with the same evaluation count (9 for 8 steps).

## v1.5.2 — 2026-10-31

### Measurement: CFG++ costs 2x at cfg = 1, nothing at cfg > 1

Krea2 @1024, 10 steps, `corrector=none` in every arm:

| arm | time |
|---|---|
| cfg 1, plain | 17.54 s |
| cfg 1, `cfg_pp=1.0` | 35.45 s |
| cfg 3.5, plain | 35.03 s |
| cfg 3.5, `cfg_pp=1.0` | 35.19 s |

CFG++ needs the unconditional prediction `u`. At cfg = 1 ComfyUI normally skips
the unconditional pass entirely, and `disable_cfg1_optimization=True` turns that
back on, so the model batch doubles. At cfg > 1 the unconditional pass happens
anyway, so CFG++ is free. The number of `calc_cond_batch` calls is 10 in all
four arms: the cost is in the batch, not in extra calls, which is why the call
counter alone could not show it.

On distilled cfg-1 models it is also the wrong choice independently of cost
(measured earlier: Krea2 10 steps 0.9432 with CFG++ vs 0.9710 without).

### Changed

* The A/B node's `config` string now says which case applies:
  `cfg++ batch x2 (cfg=1 uncond pass forced)` or
  `cfg++ free (uncond already evaluated)`.
* Corrected the `cfg_pp` tooltip: it previously said "one extra model call per
  step", which is false — the call count is unchanged and the batch doubles.
* `tools/cfgpp_cost.py` reproduces the table above.

### Fixed

* **The two nodes now expose the same sampler knobs.** `OmniSamplerSelect`
  gained `churn_mode`, `skip_every`, `skip_order` and the A/B `variant` switch;
  `OmniKSampler` gained `eta_window`. The widgets are appended to each node's
  existing order, so positional values in saved workflows keep their slots. The
  only remaining difference is the KSampler's own I/O (model / conditioning /
  latent / seed / steps / cfg / scheduler / denoise), which a SAMPLER object
  cannot own.
* **`eta_window` was unusable in the KSampler.** The node had no widget for it,
  and the sampler expects a `(low, high)` tuple while the widget is a string.
  Both nodes now parse it through `_parse_eta_window`; invalid input falls back
  to `0.10,0.90` instead of being forwarded to `sample_omni`, which expects
  exactly two values.
* **Every sampler widget has a tooltip, and every combo names each of its
  choices** with what it does and when it is ignored (e.g. `ab` ignores
  `cfg_pp`/`corrector`/`px_order`; `px_order` needs order >= 2; `skip_order`
  only applies when `skip_every > 1`). The KSampler's model/conditioning/seed/
  steps/cfg/scheduler/denoise inputs have tooltips too.
* Both nodes' `INPUT_TYPES` are generated from one `_KNOB_SPECS` definition, so
  a knob's range, default or tooltip cannot drift between them. Defaults still
  come from `omni_core.DEFAULTS`; `tools/test_node_parity.py` asserts both
  nodes' defaults agree and remain a valid preset/range.

No sampler behaviour changed for a default build: the new widgets carry the
same values `sample_omni` already defaulted to, and `variant` defaults to the
empty preset A.

### Added

* `tools/test_node_parity.py`: knob-set parity, v1.5.1 widget-order prefix,
  default agreement, tooltip presence + per-choice coverage, and a wiring proof
  that both nodes emit every knob into the `sample_omni` partial (`eta_window`
  as a tuple, including a preset override and a default-build negative control).

## v1.5.1 — 2026-10-31

### Measurement: "it feels slow" was warmup, not NFE

Reported: euler 13 s vs omni 39 s on the same graph. Measured on Krea2 @1024,
10 steps, this machine:

| run | total |
|---|---|
| euler cold | 24.33 s |
| euler warm | 10.14 s |
| omni cold | 10.36 s |
| omni warm | 10.23 s |
| omni warm + previews | 10.00 s |

* Real model evaluations (counted at `calc_cond_batch`, not estimated):
  euler 10, omni 11 (10 steps + the step-0 corrector),
  omni `corrector=none` 10, omni `skip_every=2` 7, `eta=0.3` 11.
* Sampler overhead excluding model calls: **0.03 s** for the whole 10-step run.
* Per-call model time is the same for both (1.04 s omni vs 1.21 s euler warm).
* Previews: no measurable cost at this size.
* `eta=0.3` with the Brownian tree: +0.73 s of CPU overhead per 10 steps at
  1024 px (tree bridging per step); `noise_mode=white` is +0.04 s.

So omni costs one extra evaluation (the corrector) and otherwise runs at
euler's speed; the 3x was the cold first run after a model load.

### Changed

* The A/B node now reports `evals=measured/estimated` from `LAST_STATS`
  (`calls`, `steps`, `skipped`, `corrector_steps` are recorded by
  `sample_omni`) instead of only an estimate. It also exposed that the skip
  estimate was optimistic: variant C measured 7 calls where the estimate said
  6, because the guard rails rejected one extrapolation and fell back to a real
  evaluation.
* New tools: `tools/nfe_audit.py` (call counts + time per variant),
  `tools/time_breakdown.py` (model vs overhead), `tools/time_coldwarm.py`
  (cold/warm/previews).

## v1.5.0 — 2026-10-30

### Added

* **Two optional spatial controls, ported from PD-AOS and OFF by default**
  (`px_order`, `cfg_map`; appended to both nodes' widget lists so saved
  positional values keep their slots):
  * `px_order` / `px_order_tol`: per-pixel Lagrange-remainder gate on the
    order-2 x0 extrapolation; pixels over the bound run as order 1. No extra
    NFE. Default tolerance 2.0, the value measured neutral on Krea2 @8 steps.
  * `cfg_map` / `cfg_map_floor` / `cfg_map_start`: late per-region guidance
    map measured from the run's own cond/uncond gap. Needs cfg > 1 and is
    refused with CFG++.
  * `LAST_STATS` reports the reachability counters; `OmniKSampler` writes them
    into its config string, and A/B presets K/L drive each control.
* `tools/verify_spatial.py`: OFF is bit-identical to HEAD (side-by-side module
  load), ON reaches its site, a no-op value is bit-identical, and the analytic
  mask/map algebra is checked.
* `tools/spatial_ab.py`: same-seed A/B on a real model with a repeated-base
  reproducibility control.
* `docs/PD_AOS_AUDIT.md`: what the three PD-AOS mechanisms actually did (corpus
  census of 2,210 runs + direct A/B), what was ported, and what was not.

### Findings

* PD-AOS's per-pixel CFG map is an 8x8 region map from a 16 px prompt probe,
  floors the late guidance delta at 0.85, and is **inert on short `simple`
  schedules**: its ramp starts at `1 - s/sigma_max = 0.7`, so a 12-step MOSS
  schedule ending at sigma 0.314 never starts it (`stage1 == stage2` bit-exact).
* PD-AOS's per-pixel order is real but settles near order 1 (median `ord_mean`
  1.144; second-order on 4.6% of pixels), and porting the gate at a tight
  tolerance costs accuracy (Krea2 @8: 0.8477 vs 0.8636 SSIM). It ships off.
* PD-AOS's local fields move the drift term by a median 0.25% and could not be
  separated from run-to-run drift; not ported.

## v1.4.2 — 2026-10-30

### Fixed

* **`eta > 0` crashed with negative seeds** —
  `ValueError: expected non-negative integer` from numpy's `SeedSequence`,
  reached via `BrownianTreeNoiseSampler` -> `torchsde.BrownianTree`. ComfyUI
  seed nodes can emit negative integers even though the widget declares
  `min=0`, and stock comfy samplers (`dpmpp_2m_sde`) fail identically on the
  same input. `sample_omni` now normalises a negative seed with `seed % 2**64`
  before building the tree, and falls back to white noise if the tree still
  cannot be built (degenerate interval etc.) instead of aborting the run.
* Reproduced and covered by `tools/test_eta_seed.py`: seeds -1 and -123456789
  run with `eta=0.3`, `eta=0` never constructs a noise sampler, same seed
  reproduces bit-exactly and a different seed does not.

### Verified

* MOSS with eta=0.3: seeds 7, -1, -123456789 and 2**64-1 all run and produce
  different latents (norms 302.0 / 288.5 / 264.3 / 297.5).

## v1.4.1 — 2026-10-29

### Fixed

* **`OmniKSampler` crashed on Krea2** (`mat1 and mat2 shapes cannot be
  multiplied (8960x16 and 64x6144)` in `krea2/model.py::self.first`). The node
  passed the latent straight to the sampler; `nodes.py::common_ksampler` first
  calls `comfy.sample.fix_empty_latent_channels`, which adapts an empty latent
  to the model (4 -> 16 channels AND a 5-D axis for `Wan21`, 4 -> 128 channels
  for `Flux2`). Without it the model received an unadapted latent.
* The node now mirrors `common_ksampler` in full: latent adaptation,
  `batch_index`, `noise_mask`, `latent_preview` callback (previews work again),
  `PROGRESS_BAR_ENABLED`, and it returns the latent dict copy with the
  downscale-ratio keys removed.

Verified on Krea2 ((1,4,64,64) -> (1,16,1,64,64), 7 evals) and MOSS
((1,4,64,64) -> (1,128,64,64)); both finite, correct shape.
`tools/test_node_emptylatent.py` is the regression test.

## v1.4 — 2026-10-29

### Added

* **`Omni-Hybrid KSampler (A/B)`** (`OmniKSampler`): a full sampler node with
  every Omni knob plus an 11-row `variant` preset switch, returning the latent
  and a `config` string (also printed) with an exact `evals~N` count. See
  `docs/AB_TESTING.md` for the preset table and how to compare at equal NFE.
* `skip_every` / `skip_order` merged into `omni_core` (epsilon Lagrange
  extrapolation with FSampler's guard rails and protection of the first/last
  steps). Default `skip_every=1`; verified the default path is bit-identical to
  v1.3.1 (eps/karras n=12: 2.4624e-2, 13 evals).
* `tools/test_node_ab.py`: renders four presets through the node on a real flow
  model and reports wall time and latent distance to variant A.

### Notes

* Presets marked experimental (`C`, `C2`) are unproven; `B` and `C` are the two
  open questions left by the experiment round (see `docs/AB_TESTING.md`).

## v1.3 — 2026-10-29

### Fixed (measurement, not sampler)

* `tools/modern_bench.py` produced 768x61 (Krea2) and 768x1 (Anima) images: the
  Wan VAE returns `(B,T,H,W,C)` and `to_hwc` indexed the height axis, and Krea2
  was fed a 4-D latent although `Wan21` needs 5-D. Fixed; previews assert a
  square shape now.
* Discovered that these models have **no converged high-NFE reference**: the
  same seed at 40 vs 80 steps lands in a different composition (SDXL), and
  40->60 steps differ by 7-27% relative L2. Reference-based accuracy numbers on
  real models are therefore invalid. See `docs/CORRECTIONS.md`.

### Changed

* `corrector` default reverted to `first` (v1.0 behaviour). `auto` remains as a
  documented heuristic; its flow-model evidence did not survive the metric
  correction.

### Added

* `tools/latent_bench.py` (latent-space comparison, decoded previews optional)
  and `docs/CORRECTIONS.md`.

## v1.2 — 2026-10-29

### Fixed

* **Churn geometry.** When `s_churn > 0` the state is re-noised to
  `sigma_hat = sigma*(1+gamma)` and the model is evaluated there, but the step
  still integrated from the *original* sigma, so the injected noise was not
  removed. Steps now start from `sigma_hat` (`sig_from`/`lam_from`), matching
  k-diffusion's `dt = sigma_next - sigma_hat`. Affects churn users only
  (default `s_churn=0` is byte-identical).

### Added

* `churn_mode="gamma"`: comfyui-ppm's `euler_gamma` law
  (`sigma_hat = sigma + diff*sigma/sigma_max`, skipped above `sigma_max`).
  Verified against ppm's `sample_euler_gamma` to 2.4e-7 with the same seed.

### Findings

* ppm's gamma churn is family-dependent by construction: it fires on 18/20
  steps on an SDXL-class schedule (sigma_max 14.6) but 0/8 on a flow schedule
  (sigma_max 1.0) at its default `s_sigma_diff=2.0`.
* ppm's `cfg_pp` branch in that sampler is eps-only (`alpha_t == 1`), so on flow
  models it over-guides: MOSS cfg 3.5 @12 steps 0.580 vs 0.815 for plain omni.

## v1.1 — 2026-10-29

Anima (Cosmos-derived, Qwen 0.6B TE only) measured too: the step-0 corrector
*helps* there by ~0.004, against -0.016..-0.025 on Krea2/MOSS, so corrector=`auto`
stays the expectation-weighted default and the caveat is documented in
`docs/TECHNICAL.md`.

Measured on modern flow-match models (Krea2 turbo int8 @ cfg 1, MOSS image
int8 @ cfg 3.5, PixelDiT bf16) in addition to the analytic harness and SDXL.

### Changed

* `corrector` defaults to `auto`: `first` for epsilon models, `none` for
  flow-matching models. The step-0 trapezoidal corrector helps eps models and
  very low step counts, but costs 0.016-0.025 SSIM on flow models from 8 steps
  up (Krea2 turbo @10: 0.8829 -> 0.9001, MOSS @16: 0.8875 -> 0.9037). The eps
  path is unchanged (SDXL 12/20: 0.7434 / 0.7849).

### Notes

* `omni_hybrid_cfg_pp` is not appropriate for distilled cfg=1 models: CFG++
  adds guidance the model was trained without (Krea2 turbo @10 steps: 0.9432
  vs 0.9710 for plain omni).
* Added `tools/modern_bench.py` (Krea2 / MOSS / Anima harness) and
  `tools/flow_bench.py` (PixelDiT).

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
