# What every built-in ComfyUI sampler does — and what Omni-Hybrid takes from it

Scope: the 45 samplers in `comfy/samplers.py::SAMPLER_NAMES` / `KSAMPLER_NAMES`
(this install: ComfyUI 0.42.5, `G:\AI\comfy`), i.e. the implementations in
`comfy/k_diffusion/sampling.py` plus `uni_pc`/`ddim` in `comfy/samplers.py`.
Line numbers refer to this install.

This is an engineering review, not a survey of papers: for every sampler it
states the update rule as implemented, what it is good and bad at, and whether
Omni-Hybrid reuses it — and where a component was measured and rejected, the
number is given.

---

## 0. The one identity everything reduces to

ComfyUI feeds samplers a model that returns a denoised estimate `x0`. Both
model families it supports share the same geometry:

| family | `alpha(s)` | model input | `beta = -dlog(sigma)/dlambda` |
|---|---|---|---|
| eps / VP / EDM | `1` | `x = x0 + s*eps` | `1` |
| flow (CONST) | `1 - s` | `x = (1-s)x0 + s*eps` | `1 - s` |

with `lambda = log(alpha/sigma)` ("half-logSNR", `sigma_to_half_log_snr`,
sampling.py:152). In those coordinates the PF-ODE is

```
dx/dlambda = -beta(lambda) * x + alpha(lambda) * x0
```

and for locally constant `x0` the exact one-step propagator is

```
x_t = (sigma_t/sigma_s) * x_s + alpha_t * (1 - exp(-h)) * x0 ,   h = lambda_t - lambda_s > 0
```

The first-order term is family independent: `alpha_t(1-e^-h) = 1 - sigma_t/sigma_s`
for eps *and* for flow. That is why `euler`, `dpmpp_2m`, `lms`, `ipndm`... "work"
on Flux, and why their *multistep* corrections quietly do not: they compute
`h = log(sigma_s/sigma_t)` (the eps h), while flow needs
`h = log(sigma_s*alpha_t/(sigma_t*alpha_s))`. `tools/verify_math.py` checks the
identity and the resulting ODE numerically.

---

## 1. First-order / exponential-Euler family

**`euler`** (sampling.py:190) — `to_d` + `x += d*dt`. Algebraically this *is*
the exact exponential step for constant `x0` (it equals `sigma_t/sigma_s*x +
(1-sigma_t/sigma_s)x0`), so it is "DDIM with a Karras step". 1 NFE/step, no
history, unconditionally stable, and therefore the fallback every other sampler
should degrade to. Also the only family member with Karras churn
(`s_churn/s_tmin/s_tmax`).
*Taken:* the exact step (as the base of every Omni step) and the churn.

**`euler_ancestral`** / **`euler_ancestral_RF`** (216/240) — Euler plus
`get_ancestral_step` noise: `sigma_up = eta*sqrt(sigma_t^2*(sigma_s^2-sigma_t^2)/sigma_s^2)`.
The RF variant exists because the VE formula is wrong for flow; it recomputes
`sigma_up` with the VP `alpha` ratio.
*Taken:* the idea that noise must be injected on the correct `(alpha, sigma)`
basis; **not** the "noise every step" schedule — see §7.

**`ddim`** — literally `ksampler("euler", inpaint_options={"random": True})`
(samplers.py:1394): identical to euler except the inpaint noise is re-drawn
deterministically.

**`ddpm`** (1079) — the original discrete-time DDPM ancestral sampler with the
VP `alpha_cumprod` algebra; 1 NFE/step, but only correct for VP/eps models and
it ignores the `alpha != 1` case entirely. Superseded by euler_ancestral for
practical use.

**`lcm`** (1083) — `x = denoised` then `noise_scaling(sigma_next, noise, x)`,
i.e. one-step-consistent: relies on the model output already being a clean
`x0`. Has `s_noise`/`s_noise_end` ramps and noise clipping. Useless for
multi-step quality (it discards the ODE trajectory), essential for LCM/Turbo.
Not reused.

## 2. Second order with extra NFE (single-step)

**`heun`** (269) — Euler predictor, then a second evaluation at `sigma_t`, then
average the two derivatives (`d_prime = (d+d_2)/2`). 2 NFE/step, and the last
step falls back to Euler. Second order, but it *also* re-evaluates at the
predicted point, which is what a corrector does.
*Taken:* the trapezoidal corrector, but only on selected steps (see §8), because
at 1-step-per-sigma it doubles the cost for a gain that the multistep path
already provides.

**`dpm_2`** (304) — DPM-Solver-2: midpoint in `log sigma`
(`sigma_mid = exp(lerp(log sigma_s, log sigma_t, 0.5))`), then a full step with
the midpoint derivative. 2 NFE/step, second order, *single step* (no history),
so it cannot drift. `DISCARD_PENULTIMATE_SIGMA_SAMPLERS` includes it because it
needs the extra node (samplers.py:1402).

**`heunpp2`** (1117) — 3-stage "Heun++" with `sigma`-weighted averages
(`w2 = sigma_{i+1}/(3*sigma_0)`, ...). Up to 3 NFE/step and 57 NFE at 20 steps
in the benchmark; accuracy was *worse* than plain heun on every analytic case
(0.028 vs 0.005 eps/karras@20). Not reused.

**`dpm_fast`** / **`dpm_adaptive`** (689/701) — the DPMSolver class: fixed or
PID-controlled adaptive step sizes with 1/2/3-step DPM-Solver formulas. They
*ignore* the provided sigma list (`dpm_fast` re-spaces its own grid; `dpm_adaptive`
loops until tolerance). Interesting because adaptive stepping is the right answer
to "large h at low NFE", but it makes NFE unpredictable, breaks per-step
previews, and cannot be combined with a user-chosen scheduler.
*Rejected:* Omni keeps the user's sigma grid (so previews, `denoise<1`,
two-pass sampling and `SplitSigmas` keep working) and instead guards the
extrapolation (§8).

**`dpmpp_2s_ancestral`** (716) — DPM-Solver++(2S): one midpoint evaluation plus
the `alpha_t(1-e^-h)` exponential step, with ancestral noise. 2 NFE/step,
second order, no history. Its `_cfg_pp` variant (1385) uses the unconditional
prediction as the noise estimate. *Not reused* (history-free second order is
strictly dominated by 2M at 1 NFE/step), but it is the reference for the
`eta` drift form.

## 3. Multistep, 1 NFE/step (the workhorses)

**`dpmpp_2m`** (864) — DPM-Solver++(2M): keeps `old_denoised`, forms
`denoised_d = (1+1/(2r))*denoised - (1/(2r))*old_denoised` with `r = h_prev/h`,
then the exponential step. 1 NFE/step, second order, no extra state. This is the
default "good" sampler in most ComfyUI workflows, and it is what Omni's
`x0_average="ab2"` reproduces exactly (verified to 1e-7 on a one-step eps case
and to 4 digits on the benchmark). Its two weaknesses: (a) `h` is eps-only, so
the extrapolation is wrong for flow models; (b) when the next step is much
longer than the previous one (`r` small) the `1/(2r)` factor explodes — measured
relative error 0.44 on an eps/uniform schedule where euler gets 0.13.

**`dpmpp_2m_cfg_pp`** (1429) — same with the CFG++ split (`x = denoised +
denoised_mix + e^-h x`). Shows how to thread `uncond_denoised` through
`post_cfg_function` with `disable_cfg1_optimization=True`.

**`lms`** (488) — Adams-Bashforth on `d` in `sigma` with coefficients from
`scipy.integrate.quad` (`linear_multistep_coeff`, 406), order up to 4. 1
NFE/step. Measured 0.082 (eps/karras@6) and 0.0048@20 — good, but noisy
coefficients are computed per step with `scipy.quad`, and it has no stability
guard for abrupt step-size changes.

**`ipndm` / `ipndm_v`** (1175/1218) — despite the name, *explicit* Adams-
Bashforth with hard-coded 1–4th order coefficients and the variable-step
generalisation (`ipndm_v`). 1 NFE/step. **The most accurate pure-ODE sampler in
the analytic benchmark** (2.7e-2 eps/karras@6, 1.5e-3@20; best mean over all
cases). It is essentially free, and it beats dpmpp_2m at low step counts.
*Taken:* as `integrator="ab"`, Omni implements the same family with Lagrange
weights computed from the actual sigma nodes. On a uniform-sigma schedule it
reproduces ipndm's numbers **exactly** (3.31e-1 / 1.60e-1 / 1.44e-1 on
eps/uniform at 6/12/20); on Karras schedules it is slightly *less* accurate than
ipndm (8.5e-2 vs 2.9e-2 at 6 steps), because the fixed AB4 coefficients ipndm
uses are numerically better conditioned here than the exact variable-step ones
(`sum|w|/|sum w|` = 6.7 vs 1.6). Kept as an option, not the default, because it
cannot carry `eta`/`cfg_pp`/`corrector`, and because on flow models the
exponential time variable is the better-conditioned choice.

**`deis`** (1285) — same AB family with precomputed coefficient tables
(`deis.get_deis_coeff_list`, tab/semi-implicit modes), orders 2–4. 1 NFE/step.
Close to ipndm in accuracy (2.2e-2 mean).

**`res_multistep`** (1462) — ComfyUI's own "RES" family: exponential integrator
using `phi1/phi2(-h)` and `c2 = (t_prev - t_old)/h`, i.e. the **exact** integral
of a locally linear `x0` (and `phi2`-weighted for larger `h`). 1 NFE/step.
Verified numerically: on eps it matches Omni's `x0_average="exact"` mode to 4
digits (both are the exact linear-x0 kernel). Its `_cfg_pp` and `_ancestral`
variants add the CFG++ shift and `eta` noise.
*Taken:* the φ-form insight and the CFG++ shift; the exact-kernel weighting is
available as `x0_average="exact"` but **not** the default — measured worse than
the simpler `h/2` on eps (0.236 vs 0.149 at 6 steps) because over-weighting the
slope amplifies curvature error. Data over theory.

**`gradient_estimation`** (1544) — Euler plus `d_bar = (gamma-1)(d - d_prev)`
with `gamma=2`. 1 NFE/step, effectively a cheap 2nd-order correction that needs
no `h` bookkeeping. Surprising result: it beat dpmpp_2m on both flow cases
(0.127 vs 0.128 eps/karras — a tie; 0.102 vs 0.525 flow/uniform; 0.127 vs 0.122
flow/karras). Simple and robust, but no second-order guarantee under variable
steps. Not taken (Omni's guard + AB2 dominates it on the same cases).

## 4. Stochastic / SDE samplers

**`dpmpp_sde`** (806) — DPM-Solver++ SDE with Brownian-tree noise
(`BrownianTreeNoiseSampler`, 127), `r=1/2` midpoint, and a proper half-logSNR
formulation for both families (`sigma_to_half_log_snr` + `offset_first_sigma_for_snr`
to avoid `logit(1)`). 2 NFE/step.
*Taken:* the Brownian-tree noise source and the sigma/logSNR guards.

**`dpmpp_2m_sde` / `_gpu` / `_heun`** (890/1033/945) — the multistep SDE:
`x = (sigma_t/sigma_s) e^{-eta h} x + alpha_t(1-e^{-(1+eta)h}) x0 + sigma_t
sqrt(1-e^{-2 eta h}) z`, plus a φ2 correction. 1 NFE/step. This is the single
most popular "quality" sampler in ComfyUI and the drift formula is exactly what
Omni uses for `eta>0` (verified: the constant-`x0` limit keeps the marginal
`N(0, sigma_t^2)`). Its weakness: `eta` applies to *every* step, including the
last ones where the added noise only destroys detail.

**`dpmpp_3m_sde`** (950) — 3rd-order multistep SDE; the extra `phi_3` term is
what motivates Omni's order study (§9). Measured *worst* of the SDE family on
the analytic cases (huge variance at 6 steps, 9.4 relative error on eps/uniform).

**`er_sde`** (1593) — ER-SDE-Solver-3 (arXiv 2309.06169), with the non-polynomial
noise scaler `w(lambda) = lambda*(exp(lambda^0.3)+10)` integrated by a 200-point
rectangle rule. Genuinely clever (it integrates a noise-scaling function that has
no elementary antiderivative), but 200 Python-level points per step and it is
stochastic, so its ODE-path error is dominated by injected noise (0.77–1.4 here).

**`seeds_2` / `seeds_3`** (1660/1735) — SEEDS exponential integrators in data
prediction with `phi_1/phi_2`, optional SDE noise (`eta`) and multi-stage
substeps. 2 resp. 3 NFE/step. `exp_heun_2_x0` (1723) is the **deterministic**
`r=1.0` special case = exponential Heun in x0; it is one of the best ODE
samplers measured (0.051 eps/karras@6, 0.0132@12) for 2 NFE/step.
*Taken:* the φ1/φ2 `expm1` formulation (used in the SDE drift), and the
deterministic `exp_heun_2_x0` as the reference for what 2 NFE/step can buy.

**`sa_solver` / `sa_solver_pece`** (1802/1907) — Stochastic Adams with a
**tau-gated noise interval** (`get_tau_interval_func(start_sigma, end_sigma)`),
a PECE corrector and *order reduction near the end* (`lower_order_to_end`).
1 NFE/step (+1 per PECE step).
*Taken:* both ideas — the mid-trajectory noise window (`eta_window`) and the
order reduction for the last real step (`lower_order_to_end=1`), which measurably
helped (eps/karras@6 with the exact kernel: 0.236 -> 0.129).

## 5. CFG++ guidance variants

`euler_cfg_pp` (1379), `euler_ancestral_cfg_pp` (1334), `dpmpp_2m_cfg_pp`
(1429), `dpmpp_2s_ancestral_cfg_pp` (1385), `res_multistep_cfg_pp` (1531),
`gradient_estimation_cfg_pp` (1588), `cfgpp_ud10_ab` (483).

CFG++ replaces the guided noise estimate with the **unconditional** one:

```
x_t = alpha_t * x0_cond + sigma_t * (x_s - alpha_s * u)/sigma_s
```

so the guided model only supplies the signal, never the noise. This needs the
uncond prediction even at `cfg=1` (`disable_cfg1_optimization=True`), which costs
an extra forward pass per step in that case. Omni implements exactly this
identity (verified to 2.6e-17) and applies the same `eta`/multistep treatment to
`x0_cond`.

`cfgpp_ud10_ab` additionally does an AB2 history blend and, on the last step,
**extrapolates `x0` to `sigma=0`** with a Lagrange fit through 3 denoise
predictions (`zero_weight`, `zero_order`). Omni exposes this as
`final_extrapolate`; it is off by default because it did not measurably help in
the analytic test (0.274 vs 0.270 at 6 steps with weight 1.0) — the endpoint bias
it targets is tiny at typical last sigmas.

**A bug this review found (and fixed in Omni).** In CFG++ the step is
`x_t = (sigma_t/sigma_s)(x - alpha_s*u) + alpha_t*x0`. Every multistep sampler
in ComfyUI that implements CFG++ (`dpmpp_2m_cfg_pp`, `res_multistep_cfg_pp`)
averages the *guided* prediction and the *unconditional* prediction
consistently — `res_multistep_cfg_pp` does it by shifting x with `(x0 - u)`
before integrating `u`, `dpmpp_2m_cfg_pp` by extrapolating `(x0 - u)` against
`u_prev`. Omni's first version extrapolated only `x0`, which leaves
`alpha_t*kappa*(d x0 - d u)` as an unearned guidance term: with `cond == uncond`
(where CFG++ must be a no-op) the order-2 path deviated by **16%**, and on the
CFG toy `cfg=1` behaved like standard `cfg≈4-5` instead of `≈2`. Both are fixed
by applying the same kernel average to `u` (`hist_u`), disabling the corrector in
CFG++ mode, and restoring the uncond value after the corrector's extra
evaluation. `cfg_pp_strength` (0..1) then interpolates the noise estimate
between the guided prediction (plain sampling) and the unconditional one
(reference CFG++). The measured effective-cfg table is in the README.

## 6. Flow-specific samplers and *RF* variants

`euler_ancestral_RF` (240), `dpm_2_ancestral_RF` (372), `dpmpp_2s_ancestral_RF`
(754) exist because the ancestral noise maths differ for flow models:
`downstep_ratio`, `renoise_coeff = sqrt(sigma_t^2 - sigma_down^2 * alpha_t^2/alpha_down^2)`.
`sample_dpmpp_sde`, `seeds_*`, `exp_heun_2_x0`, `er_sde`, `sa_solver` use the
half-logSNR maps and are therefore already family-correct. Everything else
(`euler`, `dpmpp_2m`, `lms`, `ipndm`, `deis`, `res_multistep`, `heun`, `dpm_2`)
hard-codes the eps `h` and/or `alpha=1`. Omni's core contribution is doing the
whole thing in the family-correct `lambda` coordinate, which measured 10x better
than `dpmpp_2m` on a flow model with a uniform-sigma schedule (0.052 vs 0.525 at
6 steps).

## 7. Video / autoregressive

`ar_video` (1913) — not a diffusion integrator at all: block-by-block denoising
with KV caches for Causal-WAN-style models, 5-D latents only. Out of scope for a
general sampler; Omni leaves it alone.

## 8. What Omni-Hybrid actually recombines

| component | source | why |
|---|---|---|
| exponential step in `lambda` time | dpmpp_2m_sde, exp_heun, seeds | exact for both families |
| `h` from half-logSNR (flow-correct) | dpmpp_sde / seeds | fixes eps-only `log(sigma_s/sigma_t)` |
| variable-step x0 averaging (AB2, `h/2`) | dpmpp_2m | best measured accuracy/robustness |
| exact-kernel x0 average (φ/κ) | res_multistep | available, measured *worse* on eps -> not default |
| Adams-Bashforth in sigma, order 1-4 | ipndm / deis | optional `integrator="ab"`; reproduces ipndm exactly on uniform schedules |
| step-ratio + large-step guard | new (motivated by dpmpp_3m_sde failures) | 0.44 -> 0.10 on eps/uniform |
| gated SDE with exponential drift | dpmpp_2m_sde + sa_solver tau gate | noise where it helps, none where it hurts |
| Brownian-tree correlated noise | dpmpp_sde | temporally coherent noise |
| CFG++ uncond-as-noise | euler_cfg_pp, res_multistep_cfg_pp | less guidance saturation |
| trapezoidal corrector on step 0 | heun / sa_solver PECE | 20% error reduction for +1 NFE |
| order reduction on the last real step | sa_solver | guards the huge-h tail |
| churn at high sigma | euler / heun (Karras) | optional exploration |
| final `sigma=0` extrapolation | cfgpp_ud10_ab | available, no measured gain |
| sigma=1/sigma=0 guards | dpmpp_sde | `logit`/`log` domain |

Rejected after measurement: extra NFE everywhere (heun/2S/3m), 3rd-order
multistep (§9), per-step adaptive stepping (breaks previews/denoise).

## 9. Why no 3rd order

A 3rd-order multistep variant was implemented (quadratic history fit with the
exact kernel moments `kappa_1`, `kappa_2`, both verified against quadrature to
1e-10). Measured convergence on the synthetic ODE: **2.09** (cold start) versus
**2.87** when the history is seeded exactly. The first step has no history and is
necessarily first order; its `O(h^2)` local error dominates the total at every
practical step count, so the 3rd-order tail buys nothing. It was removed rather
than shipped as an unverified knob. (`tools/probe_orders.py` reproduces this.)

## 10. Honest limitations

* The analytic benchmark is a Gaussian mixture, not ImageNet. It measures ODE
  discretization error and mode fidelity, not aesthetics. The real-image test
  (`tools/sd_bench.py`) is the check that the conclusion transfers.
* `eta > 0` always increases distance to the deterministic path; that is not an
  error, it is the point of an SDE sampler. The benchmark reports mode accuracy
  alongside, and Omni's default is deterministic (`eta=0`).
* `cfg_pp` costs an extra model call per step when `cfg == 1` (ComfyUI's
  `disable_cfg1_optimization`); this is inherent to CFG++.
* Tanh/sigma assumptions: for models whose `model_sampling` is neither `CONST`
  nor eps-like (e.g. `COSMOS_RFLOW`, `IMG_TO_IMG*`), Omni falls back to
  `alpha=1`, exactly like every other built-in sampler. `IMG_TO_IMG` models
  ignore the scheduler anyway.
