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

Both nodes expose these same sampler widgets, generated from one definition.
Most defaults are measured best. `px_order` and `cfg_map` ship OFF with
safe-fallback values, because their measurement is a caution, not a win; see the
[spatial section](#optional-spatial-controls-v15-off-by-default).

| input | default | meaning |
|---|---|---|
| `order` | 2 | history order. 2 for `exp` (3+ is limited by start-up), 1–4 for `ab` |
| `integrator` | `exp` | `exp` = exponential in logSNR (all features); `ab` = Adams-Bashforth in sigma, deterministic only |
| `eta` | 0.0 | SDE strength (0 = ODE). `omni_hybrid_sde` uses 0.5 |
| `s_noise` | 1.0 | noise multiplier (SDE noise and churn noise) |
| `eta_window` | `0.10,0.90` | trajectory fractions where SDE noise is injected |
| `s_churn` | 0.0 | churn: re-noise the state before the model call |
| `churn_mode` | `karras` | `karras` = relative law (euler/heun); `gamma` = comfyui-ppm's euler_gamma law |
| `noise_mode` | `brownian` | Brownian-tree correlated noise or white noise (only when `eta > 0`) |
| `corrector` | `first` | trapezoidal corrector on none/first/last/first_last steps (+1 NFE each); `auto` = first for eps, none for flow (heuristic) |
| `lower_order_to_end` | 1 | first order for the last N real steps |
| `cfg_pp` | false | CFG++ (costs +1 model call per step when cfg == 1) |
| `cfg_pp_strength` | 0.5 | 0 = plain sampling, 1 = reference CFG++; see below |
| `x0_average` | `ab2` | `ab2` = DPM++(2M) coefficient; `exact` = quadrature kernel moment |
| `min_step_ratio` | 0.5 | disable extrapolation when the step grows by >2x (0 = off) |
| `skip_every` | 1 | 1 = evaluate every step; N = evaluate every Nth step (experimental) |
| `skip_order` | 2 | epsilon-extrapolation order on skipped steps (1–3) |
| `final_extrapolate` | 0.0 | blend of an `x0 -> sigma=0` extrapolation (available, no measured gain) |
| `px_order` | false | per-pixel order-2 gate; see the spatial section |
| `px_order_tol` | 2.0 | allowed Lagrange-remainder error (multiple of local x0 RMS) |
| `cfg_map` | false | late 8×8 per-region CFG map; needs cfg > 1 and no CFG++ |
| `cfg_map_floor` | 0.85 | fraction of the late guidance delta kept in the lowest-gap region |
| `cfg_map_start` | 0.5 | trajectory fraction where the cond/uncond gap is measured |

The same A/B preset rows are available on both nodes through `variant`
(see [AB_TESTING.md](AB_TESTING.md)). A preset overrides only the settings it
names, so the remaining knobs keep their values; choose `custom (use knobs)` to
use every knob as set.

Cost: **1 NFE per step + 1** (the step-0 corrector). `eta=0` and `cfg_pp=false`
add nothing else.

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

**2. A real bug, fixed.** The multistep x0 average (`x0_eff`) was applied to the
conditional prediction only. In CFG++ mode that difference is injected straight
into the guidance direction, which made the *default* path (order 2) behave like
~4-5x the cfg value even after the term above — a 16% deviation in the case
where CFG++ must be a no-op (`cond == uncond`). Now both x0 and `u` get the same
average, the case `cond == uncond` reduces to plain sampling to 3e-7, and the
corrector is skipped in CFG++ mode (it cannot re-evaluate the uncond prediction,
so it would shift the guidance).

Verification: `tools/cfgpp_check.py` checks the match with `euler_cfg_pp`
(1e-7 at cfg 1/2/4), the no-op invariant, `strength=0 == plain`, `strength=1`
reproduces the reference, and prints the effective-cfg table above.

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

### Author preference preset (v1.5.5)

`omni_hybrid_cfgpp_author_pref` is the author's own everyday setting: CFG++ on
(`cfg_pp_strength` 0.5, the default), gamma churn at the slider maximum
(`s_churn=100`), `eta=1`, and the `px_order` gate on. Every other knob is the
shipped default from the table above.

This one is a preference, not a measured optimum: chosen by eye across the
author's own runs, with no benchmark covering the combination — the churn, eta
and `px_order` measurements here were taken at other values. The documented
laws still apply: the gamma mode skips any step where its offset would exceed
the schedule's first sigma, and CFG++ at `cfg=1` forces the unconditional pass
(a 2x model batch).

## Optional spatial controls (v1.5, OFF by default)

Two of PD-AOS's three spatial mechanisms were ported, and both are **off**;
OFF is bit-identical to v1.4.2 (proved by `tools/verify_spatial.py`, which runs
the previous release's module side by side).

| option | what it does | measured |
|---|---|---|
| `px_order` / `px_order_tol` | gives the order-2 x0 extrapolation only to pixels whose Lagrange remainder is under the tolerance; the rest run as order 1. No extra NFE. | On Krea2 @ 8 steps (512², SSIM to a 40-step reference, one process): plain 0.8620; tol 0.05 **0.8477**, 0.2 0.8550, 0.5 0.8587, 1.0 0.8592, 2.0 0.8617, 5.0 0.8608. Default 2.0 is the neutral value; a tight gate loses accuracy. |
| `cfg_map` / `cfg_map_floor` / `cfg_map_start` | measures the cond/uncond gap at `cfg_map_start` of the trajectory, pools it to 8×8, and scales the late guidance delta so the lowest-gap region keeps `cfg_map_floor` of it. Needs cfg > 1; disabled with CFG++. | On MOSS @ 8 steps (512², cfg 3.5): base vs a repeated base is bit-identical; the map builds at step 3, applies on 6 steps, and moves the decoded image 0.01424 RMS (SSIM 0.9824). |
| local fields | **not ported.** | PD-AOS's own typical amplitude was 0.25% of the drift term (p90 0.45%), it is probe-dependent, and its own A/B could not separate it from run-to-run drift. |

The audit that produced these numbers, including the two defects in the PD-AOS
originals (the guidance ramp never starts on short `simple` schedules, and the
"per-pixel" map is 8×8 regions measured from a 16 px probe), is
[PD_AOS_AUDIT.md](PD_AOS_AUDIT.md). `tools/spatial_ab.py` renders
`base`, `base2` (control), `pxo`, `cfgm` and `both` on one model/seed so the
operator can judge the appearance; the numbers above are not a substitute for
that.

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
* `integrator="ab"` is deterministic-only; with `eta>0` or `cfg_pp` the `exp`
  path is used.
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
| [CORRECTIONS.md](CORRECTIONS.md) | the two measurement bugs, and what a valid metric has to look like |
| [PD_AOS_AUDIT.md](PD_AOS_AUDIT.md) | audit of PD-AOS's spatial mechanisms: what they do and what was ported |
| [../CHANGELOG.md](../CHANGELOG.md) | what changed in each version and why |
| [../tools/README.md](../tools/README.md) | every verification and benchmark script, and what it proves |
