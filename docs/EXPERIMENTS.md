# Experiments: leads that were tested and rejected

Every change to the sampler has to measurably improve it. This file records the
leads taken from the installed custom-node packs, what was measured, and why
nothing from them shipped. The code for each experiment lives on the branch
`experiments/integrator-variants` (commit `e7be0c4`); `main` stays at the
shipped sampler.

---

## 1. UniPC-fm PECE reuse (ComfyUI_RH_FramePack, `uni_pc_fm.py:91-107`)

**Idea.** Evaluate each node at the *end* of the previous step and reuse that
evaluation, so one model call serves as both the corrector of step *i* and the
starting derivative of step *i+1* — a corrected step at 1 NFE/node instead of
`corrector="first"` giving one correction per run.

**Implementation.** `pece=True`: the predictor uses the AB2 `x0`, the step uses
the trapezoidal average `0.5*(x0(sigma_i) + x0(sigma_{i+1}))`, and the
evaluation at `sigma_{i+1}` is carried into the next iteration. For `eta>0` the
carried evaluation is taken at the actual noisy state instead.

**Analytic result (3 seeds, 160 samples, mean relative error):**

| case | n | current (corr=first) | corr=none | PECE |
|---|---|---|---|---|
| eps/karras | 8 | 6.19e-2 | 6.08e-2 | **3.98e-2** |
| eps/karras | 12 | 2.45e-2 | 2.41e-2 | **5.46e-3** |
| eps/karras | 20 | 7.16e-3 | 7.05e-3 | **3.38e-3** |
| flow/uniform | 8 | 2.87e-2 | 3.09e-2 | 3.41e-2 |
| flow/uniform | 12 | 9.20e-3 | 1.07e-2 | 9.07e-3 |
| flow/uniform | 20 | 1.72e-3 | 2.46e-3 | 2.06e-3 |

Up to 4.5x better on eps at one *fewer* NFE, neutral on flow.

**Real-model result (SDXL, 2 seeds, SSIM against an 80-step dpmpp_2m reference,
cfg 5, karras):**

| steps | current (corr=first) | corr=none | PECE | NFE |
|---|---|---|---|---|
| 12 | **0.7250** | 0.7036 | 0.6932 | 13 / 12 / 12 |
| 20 | **0.7849** | 0.7730 | 0.7843 | 21 / 20 / 20 |

**Verdict: not adopted.** At equal NFE PECE is a wash (slightly worse at 12,
slightly better at 20, both inside the seed spread) and it loses to the shipped
default at both step counts. The analytic gain does not transfer because the
reused evaluation is taken at the *predictor's* state, not at the corrected one,
so on a noisy real denoiser the corrector's benefit is only partial — while on a
smooth analytic `x0` the averaged evaluation wins clearly. That is a genuine
property of the PEC scheme, not a bug in this implementation.

---

## 2. Spectrally coloured SDE noise (comfyui-colorednoisediffusionsampling, `spectral.py:42-110`)

**Idea.** Shape the injected SDE noise as `1/f^beta` (pink/brown) with unit
per-step variance instead of white.

**Result (SDXL, 12 steps, cfg 5, eta 1.0, seed 42, SSIM vs 80-step reference):**

| noise | SSIM | high-frequency energy |
|---|---|---|
| white | 0.3174 | 0.1011 |
| pink (1/f) | 0.3166 | 0.0522 |
| brown (1/f^2) | 0.3290 | 0.0475 |

**Verdict: rejected.** The colouring works (high-frequency energy halves) but
buys no fidelity — it smooths the image. Structurally: the SDE's stationary
distribution at sigma is white noise, and the step preserves that marginal
exactly (verified: variance ratio 0.997/1.015). Colour the injected noise and the
state is no longer a valid `p(x_sigma)` sample, so the denoiser sees off-manifold
input. This is why that pack bakes the noise into its own loop with per-type
renormalisation and a gamma matrix.

---

## 3. Exponential Adams-Nørsett (RES4LYF, `rk_coefficients_beta.py:1616-1672`)

Already covered by measurement: AB-Nørsett is the phi-weighted history average,
i.e. `x0_average="exact"` in this sampler. Measured against the shipped `ab2`
weight (`h/2`) on eps/karras 6/12/20:

```
exact-kernel  0.236 / 0.0350 / 0.0092
ab2 (shipped) 0.149 / 0.0254 / 0.0072
```

and it matched `res_multistep` to four digits. Worse — the phi-weighted average
over-weights the slope and amplifies curvature error at large steps. No further
test needed.

---

## 4. Per-sample linearisation (ComfyUI-AFloPS, `aflops_core.py:77-127`)

**Not tested.** It needs a per-sample state path (a coefficient fitted from the
last two steps, clamped to ±1) inside the step loop. The cheap version is only
worth it if that coefficient is stable across steps, which is doubtful for a
learned denoiser; it is listed here as the one lead that was not run.

---

## Reproduction

The three harness scripts live on the experiment branch, not on `main`. Run from
the repository root, with the ComfyUI venv python (`$env:COMFY_DIR` points at
your ComfyUI checkout; see [CONTRIBUTING.md](../CONTRIBUTING.md)):

```powershell
git switch experiments/integrator-variants
& python tools/pece_check.py          # analytic PECE comparison
& python tools/sd_pece_ab.py          # SDXL, equal NFE
& python tools/noise_color_check.py   # coloured noise
git switch main
```
