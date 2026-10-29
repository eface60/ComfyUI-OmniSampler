# ComfyUI-OmniSampler

A single diffusion sampler that recombines the measured best parts of ComfyUI's
45 built-in samplers, and fixes the parts that are only correct for
epsilon-prediction models. It installs as a normal custom node, registers three
names in the stock **KSampler / KSamplerAdvanced / KSamplerSelect** dropdowns,
and exposes a node with every knob.

```
custom_nodes/ComfyUI-OmniSampler/
â”œâ”€â”€ __init__.py        # registers the sampler names on import
â”œâ”€â”€ omni_core.py       # the integrator (only file needed at runtime)
â”œâ”€â”€ omni_nodes.py      # Sampler node + name registration
â”œâ”€â”€ ANALYSIS.md        # what each built-in sampler does and what was reused
â”œâ”€â”€ README.md          # this file
â””â”€â”€ tools/             # verification + benchmarks (not loaded by ComfyUI)
    â”œâ”€â”€ verify_math.py     # identities, orders, SDE marginals, CFG++ algebra
    â”œâ”€â”€ probe_orders.py    # why 3rd order was not shipped
    â”œâ”€â”€ toy_bench.py       # every built-in sampler vs omni on an exact ODE
    â”œâ”€â”€ tune_integrator.py # exp vs Adams-Bashforth, and vs ipndm
    â”œâ”€â”€ tune2.py           # corrector / order-to-end / endpoint extrapolation
    â”œâ”€â”€ cfgpp_check.py     # CFG++ invariants + effective-guidance table
    â”œâ”€â”€ sd_cfgpp.py        # real-image check of CFG++ at cfg=1
    â”œâ”€â”€ diag_steps.py      # per-step divergence diagnostic
    â”œâ”€â”€ test_node.py       # registration + KSAMPLER end-to-end test
    â””â”€â”€ sd_bench.py        # real SDXL image comparison
```

## Install

Copy the `omni_sampler` folder to `G:\AI\comfy\custom_nodes\ComfyUI-OmniSampler`
and restart ComfyUI. Nothing in `comfy/` is modified on disk; the sampler names
are appended to the in-memory registries at import.

Then in **KSampler** pick `omni_hybrid` (or `omni_hybrid_sde`,
`omni_hybrid_cfg_pp`). In `SamplerCustomAdvanced`, either pick one of those
names or use **Omni-Hybrid Sampler (select)** for full control.

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

## Parameters (defaults are the measured best)

| input | default | meaning |
|---|---|---|
| `order` | 2 | history order. 2 for `exp` (3+ is limited by start-up), 1â€“4 for `ab` |
| `integrator` | `exp` | `exp` = exponential in logSNR (all features); `ab` = Adams-Bashforth in sigma, deterministic only |
| `eta` | 0.0 | SDE strength (0 = ODE). `omni_hybrid_sde` uses 0.5 |
| `s_noise` | 1.0 | noise multiplier (also scales churn) |
| `corrector` | `auto` | trapezoidal corrector on `none/first/last/first_last` steps (+1 NFE each); `auto` = `first` for eps, `none` for flow (measured, see below) |
| `lower_order_to_end` | 1 | first order for the last N real steps |
| `cfg_pp` | false | CFG++ (costs +1 model call per step when cfg == 1) |
| `cfg_pp_strength` | 0.5 | 0 = plain sampling, 1 = reference CFG++; see below |
| `x0_average` | `ab2` | `ab2` = DPM++(2M) coefficient; `exact` = quadrature kernel moment |
| `min_step_ratio` | 0.5 | disable extrapolation when the step grows by >2x (0 = off) |
| `eta_window` | `0.10,0.90` | trajectory fraction where noise is injected |
| `s_churn` | 0.0 | Karras churn at high sigma |
| `noise_mode` | `brownian` | temporally correlated (Brownian tree) or white noise |
| `final_extrapolate` | 0.0 | blend of an `x0 -> sigma=0` extrapolation (available, no measured gain) |

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

So with the reference behaviour, `cfg=1` really does look like `cfgâ‰ˆ2.5`, and
`cfg=4` blows up. The preset ships with `cfg_pp_strength=0.5`, which keeps cfg
numbers roughly meaningful; set it to `1.0` if you want to reproduce ComfyUI's
`euler_cfg_pp` exactly, or to `0.0` to use CFG++ plumbing as a no-op.

**2. A real bug, fixed.** The multistep x0 average (`x0_eff`) was applied to the
conditional prediction only. In CFG++ mode that difference is injected straight
into the guidance direction, which made the *default* path (order 2) behave like
~4-5x the cfg value even after the term above â€” a 16% deviation in the case
where CFG++ must be a no-op (`cond == uncond`). Now both x0 and `u` get the same
average, the case `cond == uncond` reduces to plain sampling to 3e-7, and the
corrector is skipped in CFG++ mode (it cannot re-evaluate the uncond prediction,
so it would shift the guidance).

Verification: `tools/cfgpp_check.py` checks the match with `euler_cfg_pp`
(1e-7 at cfg 1/2/4), the no-op invariant, `strength=0 == plain`, `strength=1`
reproduces the reference, and prints the effective-cfg table above.

## Measured results

### Analytic ODE benchmark (`tools/toy_bench.py`)

Gaussian-mixture data, both model families, two schedules, 160 samples, exact
reference by DOP853. Metric: relative error of the final denoise; lower is
better. `nfe` is for 20 steps.

| sampler | nfe | eps/karras 6Â·12Â·20 | flow/uniform 6Â·12Â·20 |
|---|---|---|---|
| **omni (defaults)** | **21** | **9.3e-2 Â· 2.5e-2 Â· 7.2e-3** | **4.4e-2 Â· 9.3e-3 Â· 1.9e-3** |
| dpmpp_2m | 20 | 1.5e-1 Â· 2.5e-2 Â· 7.4e-3 | 5.3e-1 Â· 8.0e-2 Â· 3.2e-2 |
| euler | 20 | 1.4e-1 Â· 6.5e-2 Â· 3.8e-2 | 1.2e-1 Â· 5.5e-2 Â· 3.2e-2 |
| heun | 39 | 1.3e-1 Â· 1.8e-2 Â· 5.2e-3 | 1.3e-2 Â· 3.6e-3 Â· 1.4e-3 |
| ipndm | 20 | 2.7e-2 Â· 5.9e-3 Â· 1.5e-3 | 7.0e-2 Â· 7.8e-3 Â· 1.7e-3 |
| deis | 20 | 5.1e-2 Â· 1.2e-2 Â· 4.1e-3 | 4.6e-2 Â· 7.4e-3 Â· 2.1e-3 |
| res_multistep | 20 | 2.4e-1 Â· 3.5e-2 Â· 9.2e-3 | 9.1e-1 Â· 1.4e-1 Â· 5.2e-2 |
| dpmpp_2m_sde | 20 | 1.5e+0 Â· 1.3e+0 Â· 1.3e+0 | 1.8e+0 Â· 1.4e+0 Â· 1.4e+0 |

Mean over all four cases (including a hostile eps/uniform schedule):
`omni 1.05e-1`, heun 8.3e-2 (2x NFE), exp_heun_2_x0 8.4e-2 (2x NFE), ipndm 9.2e-2,
deis 1.0e-1, euler 1.7e-1, dpmpp_2m 2.0e-1, res_multistep 3.5e-1.

Stochastic samplers score badly on a *deterministic-path* metric by
construction; that is why the default is `eta=0` and why `omni_hybrid_sde` is a
separate name.

### Real images (`tools/sd_bench.py`)

SDXL (hassakuXL v31), 1024Â², cfg 5, karras, seed 42. Reference = dpmpp_2m at 80
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
wins at 2x the NFE, which is the expected accuracy/cost trade-off.

## Verify it yourself

From the ComfyUI directory, with the venv python:

```powershell
$env:PYTHONPATH="G:\AI\comfy;G:\AI\agent working folders\generic\omni_sampler"
& "G:\AI\comfy\venv\Scripts\python.exe" "G:\AI\agent working folders\generic\omni_sampler\tools\verify_math.py"
& "G:\AI\comfy\venv\Scripts\python.exe" "G:\AI\agent working folders\generic\omni_sampler\tools\cfgpp_check.py"
& "G:\AI\comfy\venv\Scripts\python.exe" "G:\AI\agent working folders\generic\omni_sampler\tools\test_node.py"
```

`verify_math.py` checks, numerically and not by assertion:

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

`test_node.py` checks that the names appear in the stock KSampler,
KSamplerAdvanced and KSamplerSelect dropdowns, that `ksampler()` /
`sampler_object()` resolve them, and that `KSAMPLER.sample()` (with the inpaint
wrapper, noise scaling and callbacks) lands within 4-6% of the analytic
reference on both families.

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

## Repository

This folder is the project repository (git, tag `v1.0`). It is self-contained:
`omni_core.py`, `omni_nodes.py` and `__init__.py` are all ComfyUI needs; the
`tools/` scripts are the verification and benchmark harness and are never
imported by ComfyUI.

```
git log --oneline --decorate     # history
git show v1.0                    # what this tagged version contains
git checkout v1.0 -- .           # revert the working tree to it
git diff v1.0 -- omni_core.py    # see what changed since
```

`outputs/` (benchmark renders) and `__pycache__/` are ignored.
`CHANGELOG.md records what changed in each version and why, EXPERIMENTS.md\nrecords the leads that were tested and rejected (with numbers).

## Modern flow-match models (v1.1)

Measured on the models that matter now: Krea2 turbo (int8, distilled, cfg 1),
MOSS image (int8, cfg 3.5) and PixelDiT (bf16). 768², `simple` scheduler,
same seed, reference = a converged run of the same sampler at 40-100 steps.
All runs are in-process (int8 checkpoints are not bit-reproducible across
sessions, so cross-session comparison would be meaningless).

| model | steps | euler | omni (v1.1) | omni (v1.0 defaults) |
|---|---|---|---|---|
| Krea2 turbo | 6 | 0.8932 | 0.8884 | 0.9085 |
| Krea2 turbo | 8 | 0.9062 | 0.9059 | 0.8927 |
| Krea2 turbo | 10 | 0.8992 | **0.9001** | 0.8829 |
| MOSS | 10 | 0.8160 | 0.8149 | 0.7901 |
| MOSS | 16 | 0.8934 | **0.9037** | 0.8875 |

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
   itself if it looks over-guided.

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

## Churn modes (v1.2)

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

