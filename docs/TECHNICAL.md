# Technical reference

Everything behind the [README](../README.md): what the integrator does
differently, how each knob behaves, and the measurements — including the ones
that failed — that chose the defaults. Every number here comes from a script in
[`tools/`](../tools), and none of it is needed to sample an image.

## Contents

- [What makes it different](#what-makes-it-different)
- [Parameters](#parameters)
- [CFG++ and what "cfg" means](#cfg-and-what-cfg-means)
- [Measured results](#measured-results)
- [Notes by model family](#notes-by-model-family)
- [Optional spatial controls](#optional-spatial-controls-v15-off-by-default)
- [Guidance-gap shaping (experiment branch)](#guidance-gap-shaping-experiment-branch-off-by-default)
- [Limitations](#limitations)
- [Verify it yourself](#verify-it-yourself)
- [Where the evidence lives](#where-the-evidence-lives)

## What makes it different

1. **Correct for flow models, not just eps models.** The integrator works in
   half-logSNR time `lambda = log(alpha/sigma)`, so the step is
   `x_t = sigma_t/sigma_s * x + alpha_t(1-e^-h) * x0` with
   `h = log(alpha_t*sigma_s/(alpha_s*sigma_t))`. ComfyUI's `dpmpp_2m`, `lms`,
   `ipndm`, `deis`, `res_multistep`, `heun` all use the eps-only
   `h = log(sigma_s/sigma_t)`. Measured on a flow model on a uniform-sigma
   schedule at 6 steps: **0.044 vs 0.525** relative ODE error (11x).
2. **Step-ratio stability guard.** `dpmpp_2m`'s `1/(2r)` extrapolation explodes
   when the next step is much longer than the previous one. Omni drops to first
   order for that step (`min_step_ratio`), which cut an eps/uniform failure from
   0.44 to 0.10.
3. **Noise only where it helps.** `eta > 0` injects OU-consistent noise through
   the exponential drift, gated to the middle of the trajectory
   (`eta_window`), instead of on every step like `dpmpp_2m_sde`. The marginal
   stays exactly `N(0, sigma_t^2)` (verified).
4. **CFG++ done from the identity.** `x_t = alpha_t*x0 + sigma_t*(x-alpha_s*u)/sigma_s`
   (verified to 2.6e-17), with the multistep average applied to *both* x0 and
   the uncond prediction (see the CFG++ section below), and `cfg_pp_strength`
   to dial the effect.
5. **One extra model call on step 0** (`corrector="first"`). The first step has
   the largest `h`; correcting it was worth 20% error reduction for +1 NFE
   overall, and 2x on flow-at-sigma=1.
6. **Order reduction on the last real step** (`lower_order_to_end=1`), because
   Karras-like schedules end with a huge `h` where extrapolation hurts.

## Parameters

The advanced sampler node and the A/B KSampler expose these same sampler
widgets, generated from one definition; the simple node exposes only the named
settings below. Most defaults are measured best. `px_order` and `cfg_map` ship
OFF with safe-fallback values, because their measurement is a caution, not a
win; see the [spatial section](#optional-spatial-controls-v15-off-by-default).

| input | default | meaning |
|---|---|---|
| `order` | 2 | history order. `exp`: 1–2; 3–4 are clamped to 2 (the third-order fit was measured and removed; the exp path keeps one previous x0). `ab`: 1–4 |
| `integrator` | `exp` | `exp` = exponential in logSNR (all features); `ab` = Adams-Bashforth in sigma, deterministic only (eta, CFG++ or churn force `exp`) |
| `eta` | 0.0 | SDE strength (0 = ODE). `omni_hybrid_sde` uses 0.5 |
| `s_noise` | 1.0 | noise multiplier (SDE noise and churn noise) |
| `eta_window` | `0.10,0.90` | trajectory fractions where SDE noise is injected |
| `s_churn` | 0.0 | churn: re-noise the state before the model call (forces `exp` with `ab`) |
| `churn_mode` | `karras` | `karras` = relative law (euler/heun); `gamma` = comfyui-ppm's euler_gamma law |
| `noise_mode` | `brownian` | Brownian-tree correlated noise or white noise (only when `eta > 0`) |
| `corrector` | `first` | trapezoidal corrector on none/first/last/first_last steps (+1 NFE each); `auto` = first for eps, none for flow (heuristic); skipped by `ab` and (by default) with CFG++ |
| `corrector_cfgpp` | false | run the corrector under CFG++ too, averaging `u` as well as `x0` so `cond == uncond` stays a no-op; +1 NFE (2x batch at cfg = 1) |
| `lower_order_to_end` | 1 | first order for the last N real steps |
| `cfg_pp` | false | CFG++ (costs +1 model call per step when cfg == 1) |
| `cfg_pp_strength` | 0.5 | 0 = plain sampling, 1 = reference CFG++; see below |
| `x0_average` | `ab2` | `ab2` = DPM++(2M) coefficient; `exact` = quadrature kernel moment |
| `min_step_ratio` | 0.5 | disable extrapolation when the step grows by >2x (0 = off) |
| `skip_every` | 1 | 1 = evaluate every step; N = evaluate every Nth step (experimental) |
| `skip_order` | 2 | epsilon-extrapolation order on skipped steps (1–3) |
| `final_extrapolate` | 0.0 | blend of an `x0 -> sigma=0` extrapolation (available, no measured gain) |
| `px_order` | false | per-pixel order-2 gate (under CFG++ the same gate is applied to the `u` extrapolation); see the spatial section |
| `px_order_tol` | 2.0 | allowed Lagrange-remainder error (multiple of local x0 RMS) |
| `cfg_map` | false | late 8×8 per-region CFG map; needs cfg > 1 and no CFG++ |
| `cfg_map_floor` | 0.85 | fraction of the late guidance delta kept in the lowest-gap region |
| `cfg_map_start` | 0.5 | trajectory fraction where the cond/uncond gap is measured |
| `gap_lf` | 0.0 | experimental: remove the guidance delta's low-frequency band, as a fraction of the latent's short side (0 = off; see the gap-shaping section) |
| `gap_lf_keep` | 0.0 | fraction of that band kept (damps the removal instead of removing it) |
| `gap_cap` | 0.0 | per-pixel cap on the delta, as a multiple of the local unconditional magnitude (0 = off) |
| `gap_parallel` | 1.0 | fraction of the delta's component along the conditional prediction kept (1 = unchanged, 0 = projected away) |
| `gap_parallel_local` | 0.0 | experimental: remove up to this fraction of that contrast component, but only in the top tenth of the frame's per-pixel contrast push (PD-AOS-style self-normalised gate, no absolute threshold) |
| `gap_local_start` | 0.0 | step fraction (0 = first step, 1 = last) after which the local relief ramps in; early steps are left alone |
| `gap_stats` | false | record the per-step cond/uncond gap in `LAST_STATS` without changing the output; at cfg = 1 this forces the unconditional pass (2x batch) |

The same A/B preset rows are available on the A/B KSampler through `variant`
(see [AB_TESTING.md](AB_TESTING.md)). A preset overrides only the settings it
names, so the remaining knobs keep their values; choose `custom (use knobs)` to
use every knob as set. The simple sampler node uses the *named settings*
(`omni_hybrid`, `omni_hybrid_sde`, `omni_hybrid_cfg_pp`, `omni_hybrid_gap`)
instead of the lab rows.

Cost: **1 NFE per step + 1** (the step-0 corrector; `corrector_cfgpp` adds one
more under CFG++). `eta=0` and `cfg_pp=false` add nothing else.

## CFG++ and what "cfg" means

CFG++ moves guidance into the noise estimate:

```
x_t = alpha_t * x0_guided + sigma_t * (x - alpha_s * u) / sigma_s
```

so the conditional model only provides the signal and the unconditional one
provides the noise. That identity is exact in the code, but it has two
consequences worth knowing.

**1. CFG++ at cfg=1 is not the same as plain sampling at cfg=1.** Substituting
`x0_guided = u + w(x0_c - u)` gives, per step,

```
x_t = (sigma_t/sigma_s)x + (1-sigma_t/sigma_s)[x0_c + (w-1)delta] + alpha_t*e^-h * delta
```

i.e. CFG++ adds an extra `alpha_t*e^-h*delta` guidance term. Measured on the
analytic CFG toy (`tools/cfgpp_check.py`, matching by distance to the
conditioned-only result):

| CFG++ strength | cfg=1 looks like | cfg=2 looks like | cfg=4 looks like |
|---|---|---|---|
| 0.0 (off) | cfg 1.0 | cfg 2.0 | cfg 4.0 |
| 0.5 (default) | cfg 1.5 | cfg 3.0 | cfg 6.0 |
| 1.0 (reference CFG++) | cfg 2.5 | cfg 4.0 | diverges |

So with the reference behaviour, `cfg=1` really does look like `cfg≈2.5`, and
`cfg=4` blows up. The preset ships with `cfg_pp_strength=0.5`, which keeps cfg
numbers roughly meaningful; set it to `1.0` if you want to reproduce ComfyUI's
`euler_cfg_pp` exactly, or to `0.0` to use CFG++ plumbing as a no-op.

**2. A real bug, fixed — and one rule made optional.** The multistep x0 average
(`x0_eff`) was applied to the conditional prediction only. In CFG++ mode that
difference is injected straight into the guidance direction, which made the
*default* path (order 2) behave like ~4-5x the cfg value even after the term
above — a 16% deviation in the case where CFG++ must be a no-op (`cond ==
uncond`). Now both x0 and `u` get the same average, and the case `cond == uncond`
reduces to plain sampling to 3e-7.

The corrector is skipped in CFG++ mode by default: correcting x0 without
correcting `u` would shift the guidance. `corrector_cfgpp` opts into it instead:
the corrector's extra model call already re-evaluates the unconditional
prediction (the post-CFG hook captures it), so both histories are averaged with
the same trapezoid and `cond == uncond` stays a no-op (2.2e-7 eps, 2.5e-7 flow).
It costs one extra call; at cfg = 1 that call is a 2x batch.

The same symmetry rule applies to `px_order`: its per-pixel mask must gate the
`u` extrapolation as well as the x0 one, which it now does. A mask on x0 alone
re-created the v1.0 defect at per-pixel granularity — measured on the analytic
toy at `px_order_tol=0.05`, 7.6e-2 relative deviation before the fix and 2.3e-7
after.

Verification: `tools/cfgpp_check.py` checks the match with `euler_cfg_pp`
(1e-7 at cfg 1/2/4), the no-op invariant with `px_order` and `corrector_cfgpp`
on, `strength=0 == plain`, `strength=1` reproduces the reference, and prints the
effective-cfg table above.

## Measured results

**How to read these numbers.** The analytic harness has an exact reference, so
its relative errors are real integration errors. The real-model tables are
reference-based (SSIM against a high-step run), and those models turn out to
have no converged reference — the guided trajectory bifurcates, so a difference
in that metric is mostly *which mode the run landed in*. The Krea2 and Anima
numbers from the v1.1 round were additionally computed on malformed decodes and
are void. [CORRECTIONS.md](CORRECTIONS.md) records both findings and what they
leave standing; [docs/README.md](README.md) has the short version.

### Analytic ODE benchmark (`tools/toy_bench.py`)

Gaussian-mixture data, both model families, two schedules, 160 samples, exact
reference by DOP853. Metric: relative error of the final denoise; lower is
better. `nfe` is for 20 steps.

| sampler | nfe | eps/karras 6·12·20 | flow/uniform 6·12·20 |
|---|---|---|---|
| **omni (defaults)** | **21** | **9.3e-2 · 2.5e-2 · 7.2e-3** | **4.4e-2 · 9.3e-3 · 1.9e-3** |
| dpmpp_2m | 20 | 1.5e-1 · 2.5e-2 · 7.4e-3 | 5.3e-1 · 8.0e-2 · 3.2e-2 |
| euler | 20 | 1.4e-1 · 6.5e-2 · 3.8e-2 | 1.2e-1 · 5.5e-2 · 3.2e-2 |
| heun | 39 | 1.3e-1 · 1.8e-2 · 5.2e-3 | 1.3e-2 · 3.6e-3 · 1.4e-3 |
| ipndm | 20 | 2.7e-2 · 5.9e-3 · 1.5e-3 | 7.0e-2 · 7.8e-3 · 1.7e-3 |
| deis | 20 | 5.1e-2 · 1.2e-2 · 4.1e-3 | 4.6e-2 · 7.4e-3 · 2.1e-3 |
| res_multistep | 20 | 2.4e-1 · 3.5e-2 · 9.2e-3 | 9.1e-1 · 1.4e-1 · 5.2e-2 |
| dpmpp_2m_sde | 20 | 1.5e+0 · 1.3e+0 · 1.3e+0 | 1.8e+0 · 1.4e+0 · 1.4e+0 |

Mean over all four cases (including a hostile eps/uniform schedule):
`omni 1.05e-1`, heun 8.3e-2 (2x NFE), exp_heun_2_x0 8.4e-2 (2x NFE), ipndm 9.2e-2,
deis 1.0e-1, euler 1.7e-1, dpmpp_2m 2.0e-1, res_multistep 3.5e-1.

Stochastic samplers score badly on a *deterministic-path* metric by
construction; that is why the default is `eta=0` and why `omni_hybrid_sde` is a
separate name.

### Real images (`tools/sd_bench.py`)

SDXL (hassakuXL v31), 1024², cfg 5, karras, seed 42. Reference = dpmpp_2m at 80
steps. Metric: SSIM to the reference (higher = closer to the converged image).

| sampler | 12 steps | time | 20 steps | time |
|---|---|---|---|---|
| **omni_hybrid** | **0.744** | 6.5 s | 0.785 | 6.8 s |
| dpmpp_2m | 0.702 | 5.9 s | 0.789 | 9.4 s |
| res_multistep | 0.700 | 6.2 s | 0.790 | 9.6 s |
| euler | 0.684 | 6.0 s | 0.769 | 9.6 s |
| ipndm | 0.652 | 6.3 s | 0.756 | 9.6 s |
| heun (2x NFE) | 0.812 | 11.6 s | 0.913 | 15.3 s |
| dpmpp_2m_sde | 0.464 | 6.3 s | 0.465 | 6.9 s |

At equal NFE Omni leads at 12 steps and ties the multistep pack at 20; `heun`
wins at 2x the NFE, which is the expected accuracy/cost trade-off. Per the note
above, read this table as agreement with that reference, not as absolute
accuracy.

## Notes by model family

These are the findings that changed defaults after the first release. Where a
number is marked void above, it is kept here only because it explains why a
default moved.

### Modern flow-match models (v1.1)

Measured on the models that matter now: Krea2 turbo (int8, distilled, cfg 1),
MOSS image (int8, cfg 3.5) and PixelDiT (bf16). 768², `simple` scheduler,
same seed, reference = a converged run of the same sampler at 40-100 steps.
All runs are in-process (int8 checkpoints are not bit-reproducible across
sessions, so cross-session comparison would be meaningless).

| model | steps | euler | omni (v1.1) | omni (v1.0 defaults) |
|---|---|---|---|---|
| Krea2 turbo † | 6 | 0.8932 | 0.8884 | 0.9085 |
| Krea2 turbo † | 8 | 0.9062 | 0.9059 | 0.8927 |
| Krea2 turbo † | 10 | 0.8992 | **0.9001** | 0.8829 |
| MOSS | 10 | 0.8160 | 0.8149 | 0.7901 |
| MOSS | 16 | 0.8934 | **0.9037** | 0.8875 |

† void: the Krea2 rows were computed on malformed decodes (768×61 images), see
[CORRECTIONS.md](CORRECTIONS.md). The MOSS rows stand.

Two findings changed the defaults:

1. **The step-0 corrector is an eps-model tuning.** It helps epsilon models
   (SDXL +0.02 at 12/20 steps) and at very low step counts (Krea2@6: +0.02), but
   it *hurts* flow models from 8 steps up (Krea2@10: -0.017, MOSS@10: -0.025,
   MOSS@16: -0.016). `corrector="auto"` now resolves by family; the eps path is
   byte-for-byte the old behaviour (SDXL 12/20 steps: 0.7434 / 0.7849 vs
   0.7441 / 0.7849 before).
2. **CFG++ is wrong at cfg=1 for distilled models.** At cfg 1 on Krea2 turbo the
   `omni_hybrid_cfg_pp` preset drops to 0.8919 / 0.9432 versus 0.8994 / 0.9710
   for plain `omni_hybrid` — a turbo model is trained for exactly cfg 1, so the
   extra guidance CFG++ adds has nothing to correct. Use the cfg_pp preset for
   normal-cfg models only, and lower `cfg_pp_strength` (0.5 default) or `cfg`
   itself if it looks over-guided. (The Krea2 side of this row is void with the
   rest of the table; the analytic effective-cfg measurement in the CFG++
   section is what the recommendation rests on.)

On these models omni is roughly at euler at 6-10 steps and clearly ahead at
16+ (MOSS@16: +0.010), while `dpmpp_2m` trails both (0.783/0.845 on MOSS at
10/16 steps) — its eps-only step size is the reason.

### The corrector is model-dependent (honest caveat)

`auto` resolves by family, which is the expectation-weighted choice, not a law:

| model | 8-10 steps, corr on vs off | verdict |
|---|---|---|
| Krea2 turbo | 0.8927 vs 0.9059 (8) / 0.8829 vs 0.9001 (10) | off wins clearly |
| MOSS | 0.7901 vs 0.8149 (10) / 0.8875 vs 0.9037 (16) | off wins clearly |
| Anima (Qwen 0.6B TE) | 0.9893 vs 0.9858 (8) / 0.9947 vs 0.9893 (12) | on wins slightly |
| SDXL (eps) | 0.7441 vs 0.6985 (12) | on wins clearly |

So `auto` = `none` for flow loses ~0.004 on Anima but gains 0.016-0.025 on
Krea2/MOSS. If you sample Anima at 8-12 steps and want that last 0.004,
`corrector="first"` is the better setting for it.

### Churn modes (v1.2)

`s_churn` re-noises the state before the model call (a stochasticity dial, not an
accuracy knob). `churn_mode` picks the law:

* `karras` (default): relative `sigma_hat = sigma*(1+gamma)`, `gamma` capped at
  `sqrt(2)-1`, like `sample_euler`/`sample_heun`.
* `gamma`: comfyui-ppm's `euler_gamma` law, `sigma_hat = sigma + s_sigma_diff*sigma/sigma_max`,
  skipped when `sigma_hat > sigma_max`. Verified to reproduce ppm's
  `sample_euler_gamma` to **2.4e-7** (same seed, eps model).

A caveat worth knowing if you come from `euler_gamma`: its guard makes it
family-dependent. With `sigma_max = 1` (every flow schedule) a diff of 2.0 fires
on **0/8** steps of a `simple` 8-step schedule, while on an SDXL-class schedule
(`sigma_max = 14.6`) it fires on **18/20**. So on modern flow models "euler gamma"
is effectively plain Euler, and what remains is its CFG++ branch — which is
**eps-only** (`alpha_t == 1`) and therefore acts as extra guidance on flow models.
Measured on MOSS at cfg 3.5, 12 steps: plain euler_gamma 0.826, euler_gamma+CFG++
0.580, omni 0.815. `omni_hybrid` with `cfg_pp` off is the sane choice there.

The same review found and fixed a real bug: when churn moved the state to
`sigma_hat`, the step still integrated from the original sigma, so the injected
noise was never removed. Steps now start at `sigma_hat`, matching k-diffusion.

On a flow model the `karras` law is additionally capped at the schedule's first
sigma. A flow model is only defined below `sigma = 1` (`alpha = 1 - sigma`), and
`clamp_sigma` pins the geometry at `1 - 1e-6`, so step 0 of a full denoise used
to be noised for `sigma_hat = 1.414` but integrated by dividing by 1.0: the state
came out at 1.30 sigma instead of 0.92, so the injected noise was not removed
(+41% at the `sqrt(2)-1` cap), in the step that matters most. That step is now a
bit-exact no-op and the steps below the cap churn normally (MOSS 768², cfg 3.5,
`s_churn=8` at its `sqrt(2)-1` cap: SSIM 0.288 -> 0.385 at 12 steps and
0.226 -> 0.511 at 8, and the corrupted image becomes a valid sample). The cap
also means a partial denoise (`raw[0] < 1`) cannot churn above its starting
sigma. eps models keep k-diffusion's uncapped behaviour (`sigma_hat` up to 20.7
on an SDXL schedule). `LAST_STATS` reports `churn_steps` (steps that injected
noise, either law) and `churn_capped` (steps the flow cap reduced), which is the
quickest way to confirm a churn setting actually fires on a given schedule; the
A/B node prints `churn k/n steps` in its config string for the same reason.

### The retired author-preference name (v1.5.5, removed on the experiment branch)

omni_hybrid_cfgpp_author_pref was the author's everyday by-eye setting: CFG++
0.5, gamma churn at the slider maximum, eta=1, and the px_order gate. It was
never a measured optimum, and the gap relief superseded it, so the registered
name is gone: the stock KSampler / KSamplerAdvanced / KSamplerSelect lists now
carry the three measured settings plus the experimental omni_hybrid_gap.
The combination itself is still reachable as A/B row M: cfgpp+churn+eta+px
(lab) on the A/B node.

The gap relief's own setting (omni_hybrid_gap) is documented in
[EXPERIMENT_GAP_SHAPING.md](EXPERIMENT_GAP_SHAPING.md); it is not part of a
release, and it is the only extra name the branch registers.

## Optional spatial controls (v1.5, OFF by default)

Two of PD-AOS's three spatial mechanisms were ported, and both are **off**;
OFF is bit-identical to v1.4.2 (proved by `tools/verify_spatial.py`, which runs
the previous release's module side by side).

| option | what it does | measured |
|---|---|---|
| `px_order` / `px_order_tol` | gives the order-2 x0 extrapolation only to pixels whose Lagrange remainder is under the tolerance; the rest run as order 1. No extra NFE. Under CFG++ the same mask also gates the unconditional extrapolation, so the `cond == uncond` no-op survives. | On Krea2 @ 8 steps (512², SSIM to a 40-step reference, one process): plain 0.8620; tol 0.05 **0.8477**, 0.2 0.8550, 0.5 0.8587, 1.0 0.8592, 2.0 0.8617, 5.0 0.8608. Default 2.0 is the neutral value; a tight gate loses accuracy. |
| `cfg_map` / `cfg_map_floor` / `cfg_map_start` | measures the cond/uncond gap at `cfg_map_start` of the trajectory, pools it to 8×8, and scales the late guidance delta so the lowest-gap region keeps `cfg_map_floor` of it. Needs cfg > 1; disabled with CFG++. | On MOSS @ 8 steps (512², cfg 3.5): base vs a repeated base is bit-identical; the map builds at step 3, applies on 6 steps, and moves the decoded image 0.01424 RMS (SSIM 0.9824). |
| local fields | **not ported.** | PD-AOS's own typical amplitude was 0.25% of the drift term (p90 0.45%), it is probe-dependent, and its own A/B could not separate it from run-to-run drift. |

The audit that produced these numbers, including the two defects in the PD-AOS
originals (the guidance ramp never starts on short `simple` schedules, and the
"per-pixel" map is 8×8 regions measured from a 16 px probe), is
[PD_AOS_AUDIT.md](PD_AOS_AUDIT.md). `tools/spatial_ab.py` renders
`base`, `base2` (control), `pxo`, `cfgm` and `both` on one model/seed so the
operator can judge the appearance; the numbers above are not a substitute for
that.

## Guidance-gap shaping (experiment branch, OFF by default)

**Not a release feature, and the hypothesis behind it was tested and not
supported.** The branch `experiments/gap-shaping` started from one observation:
Krea2 turbo at 2048×2944 (12 steps, cfg 1) is artifact-free but burns with
`cfg_pp_strength` 1.0, artifacts with 0.5, and is fine at 1024×1408. The guess
was that the guidance gap grows at out-of-distribution sizes. A sweep of three
families (Krea2 turbo, MOSS base, Anima base 1.0, 0.26-6.0 MP, cfg 1/3.5/4)
measured the opposite: the gap *shrinks* with area (area^-0.19 / -0.34 / -0.07),
its absolute latent RMS drops from 0.319 to 0.239 (Krea2, 1 MP -> 6 MP), and
after the `alpha_t` weighting the 6 MP run receives 31% *less* effective
guidance than the 1.4 MP one. What is universal is the spectrum (the delta's
low-frequency share at 2/5/10/20% bands has a coefficient of variation of
0.10-0.12 across the three) and the front-loaded shape; what is not is the
magnitude (4.7x spread at 1 MP), so no absolute automatic threshold exists. The
tables and the reproduction commands are in
[EXPERIMENT_GAP_SHAPING.md](EXPERIMENT_GAP_SHAPING.md).

The quantity being shaped is the guidance delta, `denoised - uncond`, in x0
space. At `cfg == 1` the strength knob has a literal meaning: it is the fraction
of the cond/uncond gap that becomes guidance (the guided target *is* the
conditional prediction, so there is no cfg-1 no-op mode -- `tools/cfgpp_check.py`
measures cfg 1 at strength 1 as plain cfg ≈ 2.5 at 12 steps). On a distilled
cfg-1 model that extra guidance is a burn; at strength 0.5 half of the gap goes
into the noise estimate instead, which is where the artifacts come from.

The knobs rewrite the delta before it enters the step, in this order:
projection (`gap_parallel`, then the local gate `gap_parallel_local`) →
low-pass split (`gap_lf` / `gap_lf_keep`) → cap (`gap_cap`). Because every model
family hands the post-CFG hook a denoised
prediction, the same code covers eps and flow models, plain `cfg > 1`, the
forced-unconditional path at `cfg = 1`, and CFG++ (which then uses the shaped
delta as its signal and the raw uncond as its noise). It adds no model call;
the cost is the forced unconditional pass at `cfg = 1` that CFG++ already pays.

`gap_parallel_local` is the local version of the projection, added after the
author's 6 MP test: `gap_parallel 0` (remove the whole contrast component) fixes
the burn and washes the frame grey, which says the contrast lives in that
component and that a binary removal is too much. The local version removes the
same component only where the step's own per-pixel push
(`delta . cond / ||cond||^2`, positive = pushing brighter) sits in the top
decile of that step's distribution, with a smooth gate; every pixel under the
tail is untouched. That is the one transferable idea from PD-AOS's local fields
-- a per-pixel quantity normalised by a reference measured from the run itself,
so no absolute threshold is needed, which is exactly what the cross-model sweep
says is impossible. PD-AOS's own implementation was audited and left out: its
per-pixel modulation measured 0.25-0.45% of one step term and its own A/B could
not resolve it ([PD_AOS_AUDIT.md](PD_AOS_AUDIT.md)). The gate here is a
different signal and is **not measured on a real model yet**.

`gap_local_start` exists because the first 6 MP test of the local gate worked
but left early "colour blobs that resolved themselves later": a per-step gate
recomputes its decile every step, and early in a run that decile is not the
highlight tail -- it is the composition, so relieving contrast there fights the
structure the model is still laying down. The ramp keeps every step before
`gap_local_start` (a fraction of the *steps*, not of logSNR: the extreme sigmas
eat most of the lambda span, so 30% of logSNR is step 2 of 12) bit-identical to
the unshaped path, and fades the relief in after it. The author's Krea2 6 MP run
with the gate on every step had "the ending burn nearly completely gone" and the
early glitch; the ramp is the fix for the glitch.

Defaults are neutral and install no hook, so OFF is bit-identical. The knobs are
also state-free: they depend only on the current prediction, so the
`cond == uncond` no-op and `strength=0 == plain` invariants survive
(`tools/gap_shape_check.py`). `gap_stats` measures the gap without changing the
output: per step it records `rms_ratio` (`||delta||/||uncond||`), `lf_share`
(energy fraction below 5% of the short side) and `par_share` (energy fraction
along the conditional prediction), plus the mean applied cap scale when the cap
fires. That is the measurement the hypothesis is judged by: a 1024×1408 run and
a 2048×2944 run are compared on those numbers.

`tools/gap_shape_check.py` demonstrates the intended mechanism on a synthetic
pair: a smooth exposure-like delta plus a checkerboard "structure" delta,
through ComfyUI's real post-CFG hook. With `gap_lf` 0.2 the tone energy is
removed to 0.2% and the structure survives with cosine 0.99; with `gap_cap` 0.2
the per-pixel bound holds exactly. `tools/gap_shape_bench.py` runs the real-model
matrix.

## Limitations

* The toy benchmark measures ODE convergence and mode fidelity, not aesthetics.
  The SDXL numbers are the reality check.
* `cfg_pp` costs an extra forward pass per step when `cfg == 1`; that is inherent
  to CFG++ (`disable_cfg1_optimization`).
* Flow support is verified analytically and on a synthetic flow model, not yet on
  a real large flow checkpoint (the installed ones need multi-GB text encoders
  that were out of scope for this test run).
* For exotic `model_sampling` classes (`COSMOS_RFLOW`, `IMG_TO_IMG*`) Omni falls
  back to `alpha=1` like every other built-in sampler.
* `integrator="ab"` is deterministic-only; with `eta>0`, `cfg_pp` or
  `s_churn>0` the `exp` path is used. `order` 3–4 on the `exp` path are clamped
  to 2 (the third-order fit was measured and removed).
* `corrector_cfgpp` is wired and its no-op invariant is verified, but no
  real-model accuracy comparison has been run: the corrector itself is still an
  open question (see [CORRECTIONS.md](CORRECTIONS.md)).
* The gap-shaping knobs are experimental and neutral by default. They are
  invariants-verified and demonstrated on a synthetic tone/structure pair, and
  the cross-model sweep in [EXPERIMENT_GAP_SHAPING.md](EXPERIMENT_GAP_SHAPING.md)
  did run: it found the guidance gap *shrinking* with resolution, so the shaping
  is not a resolution fix, and no value of `gap_lf` / `gap_cap` / `gap_parallel`
  is a measured optimum.
* The only mode-free real-model instrument is `tools/local_metric.py`; its
  first-interval reference is only resolved when the substep count scales with
  the interval's `lambda` gap (the default does), and it refuses to write
  numbers unless the first arm and its refined reference repeat bit-exactly.
  Both modern flow models failed that gate (MOSS 7-67% drift; Krea2 28% coarse,
  12% reference), so their local-error rows are not quotable; see
  [CORRECTIONS.md](CORRECTIONS.md).
* The real-model evidence is reference-based and the references are not
  converged; see [CORRECTIONS.md](CORRECTIONS.md).

## Verify it yourself

The harness needs ComfyUI's own venv (it imports `torch` and `comfy`). Set
`COMFY_DIR` to your ComfyUI checkout — [`tools/README.md`](../tools/README.md)
explains the environment variables — then run from the repository root:

```powershell
$env:COMFY_DIR = "C:\path\to\ComfyUI"
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\verify_math.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\cfgpp_check.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\test_cross_features.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\test_node.py
```

[`tools/verify_math.py`](../tools/verify_math.py) checks, numerically and not by
assertion:

* `exact_step` reproduces the analytic trajectory for both families (1.5e-16),
* `dx/dlambda = -beta x + alpha x0` holds on the exact trajectory (5e-5),
* `sde_step(eta=0) == exact_step` and `eta=1` keeps `N(0, sigma_t^2)`
  (var ratio 0.997/1.015),
* the quadrature kernel equals the closed form for eps (2e-14) and makes the
  step exact for `x0` linear in lambda in both families (3e-13),
* the shipped sampler converges at first and second order (1.05 / 1.79-1.89),
* the CFG++ step equals `alpha_t x0 + sigma_t (x - alpha_s u)/sigma_s`
  (2.6e-17),
* the endpoint extrapolation is exact for a quadratic `x0(sigma)` (4e-16),
* flow/EPS detection matches `comfy.model_base.model_sampling()`,
* one eps step equals `dpmpp_2m`'s first step (1.6e-7).

[`tools/test_cross_features.py`](../tools/test_cross_features.py) checks the
pairs no single-feature test covers: `order` is clamped to 2 on `exp` (and `ab`
really uses 1-4), `ab` + churn is bit-identical to `exp` + churn,
`final_extrapolate` receives the churned evaluation sigmas, and karras churn
never evaluates a flow model above `sigma_max` (a one-step flow run is a
bit-exact no-op; eps still evaluates at 20.7).

[`tools/test_node.py`](../tools/test_node.py) checks that the names appear in the
stock KSampler, KSamplerAdvanced and KSamplerSelect dropdowns, that
`ksampler()` / `sampler_object()` resolve them, and that `KSAMPLER.sample()`
(with the inpaint wrapper, noise scaling and callbacks) lands within 4-6% of the
analytic reference on both families.

The rest of the harness is listed in [`tools/README.md`](../tools/README.md): the
benchmarks, the A/B renderers, and the regression tests for the bugs above.

## Where the evidence lives

| document | what it answers |
|---|---|
| [ANALYSIS.md](ANALYSIS.md) | what each of the 45 built-in samplers does, and what was reused or rejected |
| [AB_TESTING.md](AB_TESTING.md) | the A/B preset rows and how to compare two runs at equal NFE |
| [EXPERIMENTS.md](EXPERIMENTS.md) | leads from other custom-node packs that were tested and did not ship |
| [CORRECTIONS.md](CORRECTIONS.md) | the two measurement bugs, the local-metric instrument correction, and what a valid metric has to look like |
| [PD_AOS_AUDIT.md](PD_AOS_AUDIT.md) | audit of PD-AOS's spatial mechanisms: what they do and what was ported |
| [../CHANGELOG.md](../CHANGELOG.md) | what changed in each version and why |
| [../tools/README.md](../tools/README.md) | every verification and benchmark script, and what it proves |
