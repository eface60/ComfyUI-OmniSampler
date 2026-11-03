# Guidance-gap shaping (experiment branch)

**Status: tested on three model families; the hypothesis that motivated the
shaping is not supported.** Branch `experiments/gap-shaping`. Nothing here
changes a release default or a release sampler name; the knobs default to
neutral, and a neutral set installs no hook (bit-identical output).

## The observation

Krea2 turbo (distilled, so trained to run at cfg 1), 12 steps:

| resolution | plain cfg 1 | CFG++ strength 0.5 | CFG++ strength 1.0 |
|---|---|---|---|
| 1024x1408 | fine | fine | fine |
| 2048x2944 | artifacts | artifacts | **no artifacts, severe burn/oversaturation** |

At cfg 1 the guided target *is* the conditional prediction, so `cfg_pp_strength`
has a very literal meaning: **the fraction of the cond/uncond gap that becomes
guidance**. There is no cfg-1 "no-op" mode: `tools/cfgpp_check.py` measures that
cfg 1 at strength 1 already looks like plain cfg 2.5 at 12 steps. On a model
distilled for cfg 1 that extra guidance is the burn, and at strength 0.5 half of
the gap is pushed into the noise estimate instead, which is where the artifacts
come from. Both settings are the same gap routed to two places.

Both resolutions run 12 steps, so step placement was ruled out early. The
question this branch was built around: does the gap itself change with
resolution -- is it bigger, or more tonal, when the model is out of distribution?

## What the sweep measured

Three families, one prompt, seed 7, 12 steps, `simple` scheduler, the
`gap_stats` probe only (the image is unchanged). `krea2` is the official turbo
int8 (cfg 1, 16 channels, VAE ratio 8), `moss` the base int8 (cfg 3.5, 128
channels, ratio 16), `anima` base v1.0 (cfg 4, 16 channels, ratio 8). 26 records
in `outputs/gap_sweep/records.jsonl`; reproduce with
[`tools/gap_sweep.py`](../tools/gap_sweep.py) and read with
[`tools/gap_report.py`](../tools/gap_report.py).

**1. The relative gap does not grow with resolution -- it shrinks.** Mean
`||delta|| / ||uncond||` over the trajectory (magnitude divided back by cfg, so
a cfg-1 and a cfg-3.5 model are comparable):

| area (MP) | krea2 | moss | anima |
|---|---|---|---|
| 0.26 | 0.867 | 0.380 | 0.123 |
| 0.59 | 0.609 | -- | -- |
| 1.05 | 0.513 | 0.216 | 0.109 |
| 1.44 | 0.514 | -- | -- |
| 1.98 | 0.490 | -- | -- |
| 4.19 | 0.439 | 0.149 | 0.103 |
| 6.03 | 0.469 | -- | -- |
| **power law** | **area^-0.19** | **area^-0.34** | **area^-0.07** |

All three fall (R2 0.82-0.99). Krea2 at 2048x2944 is 9% *below* its 1024x1408
value, and its tone share (`dc_ratio`) falls 0.192 -> 0.153. The one size that is
badly over-guided is the *small* one: 0.26 MP is 1.7x the 1 MP gap on krea2 and
moss, 1.1x on anima.

**2. The absolute impulse shrinks too.** Latent RMS of the delta, and of the
uncond prediction it is measured against:

| family | u_rms (1 MP) | d_rms (1 MP) | d_rms (4 MP) | d_rms (6 MP) |
|---|---|---|---|---|
| krea2 | 0.533 | 0.319 | 0.251 | 0.239 |
| moss | 0.646 | 0.316 | 0.227 | -- |
| anima | 0.472 | 0.183 | 0.156 | -- |

The uncond prediction's magnitude is essentially flat in area (0.47-0.65 across
all three -- the only quantity that is nearly model-independent), while the
delta loses a fifth to a quarter of its amplitude from 1 MP to 4-6 MP.

**3. The step weighting makes it worse, not better.** What the trajectory
actually receives is `alpha_t * delta`, and `alpha_t = 1 - sigma` is ~0 at the
first step. Summed over 12 steps, krea2 at 1024x1408 receives 0.507 and at
2048x2944 receives 0.351 -- the burning run gets **31% less** effective guidance.

The gap is also strongly front-loaded (step 0 is 2.5-4.2x the run mean, the last
step 0.02-0.06 of it), and the one resolution-linked excess is in that very first
step: krea2's step-0 `rms_ratio` rises 1.28 (1 MP) -> 1.82 (4 MP) -> 1.99 (6 MP)
and its step-0 tone 0.80 -> 1.32. That spike is almost inert, because the step's
`alpha` is 0.000.

**4. What is universal.** The delta's spectrum: energy fraction below 2/5/10/20%
of the short side, at 1 MP, is 0.55/0.41/0.26/0.16 (krea2), 0.47/0.36/0.25/0.16
(moss), 0.57/0.44/0.30/0.19 (anima) -- a coefficient of variation of 0.10-0.12.
So a band expressed as a *fraction of the short side* is meaningful across
models, and that part of the design holds.

**5. What is not universal.** The magnitude: 0.51 / 0.22 / 0.11 at 1 MP (4.7x
spread), and the area exponent (-0.07 to -0.34). `par_share` (the delta's
component along the conditional prediction) is 0.28 / 0.12 / 0.12. There is no
absolute constant that can be right for all three, so **an automatic rule can
only be relative to the run's own measurement** -- a universal threshold does not
exist in this data.

## What this means

* The shaping cannot fix the 6 MP burn through the delta: at 6 MP the gap is
  smaller than at 1.4 MP in relative terms, smaller in absolute terms, and 31%
  smaller after the `alpha_t` weighting. Suppressing it further would flatten the
  image, not cure the burn.
* The remaining explanations for the 6 MP symptom are outside the guidance
  delta: the VAE decode at 6 MP (tiling/statistics), attention over ~94k tokens
  at a size the model was not trained on, or the model's own out-of-distribution
  content. The branch's next measurement should be plain vs CFG++ at the same
  seed and size with image metrics -- if plain burns too, no sampler-side
  guidance change can be the fix.
* The knobs are still available and still verified, but they are not a resolution
  fix. The part of the design the sweep does support is the band as a fraction of
  the short side; the cap's tolerance remains a judgement call.

## The fix under test

Leave the *amount* of guidance (cfg) alone and shape *what the gap does*, in x0
space, in the post-CFG hook:

| knob | transform |
|---|---|
| `gap_lf` | remove the low-frequency band of the delta (band width as a fraction of the short side); `gap_lf_keep` keeps a fraction of it |
| `gap_cap` | per-pixel cap `|delta| <= cap * |uncond|`, direction preserved |
| `gap_parallel` | keep that fraction of the delta's component along the conditional prediction (APG-style at 0) |

Applied in the order project -> split -> cap. The hook receives a denoised
prediction on every model family, so the same code covers eps and flow models,
and it covers plain `cfg > 1`, the forced-unconditional path at `cfg = 1`, and
CFG++ (CFG++ then uses the shaped delta as its signal and the raw uncond as its
noise). No extra model call.

## How to run

```powershell
$env:COMFY_DIR = "G:\AI\comfy"

# invariants and the tone/structure demonstration (no model needed)
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\gap_shape_check.py

# the cross-model sweep behind the tables above (probe only, no image change)
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\gap_sweep.py --families krea2,moss,anima
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\gap_report.py

# real model, image metrics (images + metrics in outputs\gap_shaping)
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\gap_shape_bench.py --res 1024x1408 --arms plain,cfgpp,hf,cap
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\gap_shape_bench.py --res 2048x2944 --arms plain,cfgpp,hf,cap
```

In the node: rows `N: gap HF (5%)`, `O: gap cap 0.2`, `P: gap HF + cap`,
`Q: gap probe (no shaping)`; the same settings are the registered sampler name
`omni_hybrid_gap` (CFG++ 1.0 + `gap_lf 0.05` + `gap_cap 0.2`) for a stock
KSampler workflow.

## Cost and caveats

* Any gap feature at `cfg = 1` forces the unconditional pass, so the model batch
  doubles per step -- the same cost CFG++ already pays at cfg 1. At `cfg > 1` it
  is free.
* The sweep is three checkpoints, one prompt, one seed (a second seed moved
  krea2's mean by 1%), 12 steps (8/12/20 gave 0.553/0.513/0.474 on krea2), one
  scheduler, and square sizes plus one aspect. "Universal" here means "the same
  qualitative law across these three families", not a claim about every model.
  The falsifiable statement is in the tables: another family that *grows* with
  area, or a spectrum that does not sit near the 0.10 CV band, would break it.
* The knob values in the preset are **not measured optima**; they are starting
  points for the image-metric arms, and the tables above are the reason to expect
  little from them at 6 MP.

## Verification

`tools/gap_shape_check.py` covers what the feature must not do and what the
transform does: neutral is bit-identical, `cond == uncond` stays a no-op with
every knob on, the cap's bound and direction, the split's band algebra, the
projection, 5-D latents, and an end-to-end tone-vs-structure demonstration
through ComfyUI's real post-CFG hook. The existing suites (`verify_math`,
`cfgpp_check`, `test_cross_features`, `verify_spatial`, `test_node`,
`test_node_parity`) pass unchanged.
