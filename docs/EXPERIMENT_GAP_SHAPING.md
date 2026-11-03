# Guidance-gap shaping (experiment branch)

**Status: experimental, unmeasured on a real model.** Nothing here changes a
release default or a release sampler name; the knobs default to neutral, and a
neutral set installs no hook (bit-identical output). Branch:
`experiments/gap-shaping`.

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

Both resolutions run 12 steps, so step placement is not the difference. The
difference has to be in the gap itself.

## Hypothesis

**H1 (tone).** At out-of-training sizes the gap is dominated by low-frequency
disagreement -- exposure, colour, contrast -- and that is what burns. The
high-frequency part is structure and texture, which is the part that keeps the
OOD sample coherent.

**H2 (magnitude).** The gap's magnitude relative to the prediction grows with
latent area (the model is less certain, so its conditional and unconditional
branches disagree harder), so one fixed guidance schedule cannot be right at
both 1.4 MP and 6 MP.

Prediction, measurable with the `gap_stats` row without changing a pixel:
`rms_ratio` (gap RMS over prediction RMS) and `lf_share` (energy fraction below
5% of the short side) are both higher at 2048x2944 than at 1024x1408 at matched
step indices.

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

This is deliberately not "lower cfg": lowering cfg scales the whole delta by one
number, while the hypothesis says the tone part is harmful and the structure
part is useful, so the two can be separated.

## The experiment

Same prompt, seed, 12 steps, cfg 1, `simple` scheduler. Run at 1024x1408 and at
2048x2944.

| arm | settings | what it answers |
|---|---|---|
| `plain` | -- | baseline artifacts, no guidance at all |
| `probe` | `gap_stats` | is the gap bigger / more low-frequency at 6 MP? (**H1/H2**, output unchanged) |
| `cfgpp` | `cfg_pp`, strength 1.0 | the burn to fix |
| `hf` | `cfg_pp 1.0`, `gap_lf 0.05` | does removing the tone band keep the clean image and kill the burn? (**H1**) |
| `cap` | `cfg_pp 1.0`, `gap_cap 0.2` | does bounding the overshoot do it? (**H2**) |
| `hf_cap` | both | the combination the preset `omni_hybrid_gap` uses |
| `project` | `gap_parallel 0` | contrast-only component removed |
| `cfgpp_cfg2` | `cfg_pp 1.0`, cfg 2 | the gain-matched control from `cfgpp_check.py` |

Read: saved PNGs, the reference-free metrics the runner prints (`sat` mean
chroma, `clip` fraction of pixels at the rail, `hf` Laplacian energy in flat
regions, `lum`), and the per-step `gap_series`.

**What would confirm it.** `probe` separates 1024 from 2048 on `rms_ratio` or
`lf_share`; `hf` or `cap` removes the burn while keeping the artifact-free look
of `cfgpp`; `hf_cap` is better than either alone.

**What would falsify it.** If the gap statistics are the same at both sizes, the
resolution effect is not in the guidance gap (next suspects: VAE decode tiling
at 6 MP, attention/splitting at 94k tokens, or the model being simply out of
distribution). If no shaping arm changes the burn, the burn is not the low
frequency/magnitude of the delta.

## How to run

```powershell
$env:COMFY_DIR = "G:\AI\comfy"

# invariants and the tone/structure demonstration (no model needed)
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\gap_shape_check.py

# real model: 1.4 MP first, then the 6 MP case (images + metrics in outputs\gap_shaping)
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\gap_shape_bench.py --res 1024x1408
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\gap_shape_bench.py --res 2048x2944 --arms probe,cfgpp,hf,cap,hf_cap
```

In the node: rows `N: gap HF (5%)`, `O: gap cap 0.2`, `P: gap HF + cap`,
`Q: gap probe (no shaping)`; the same settings are the registered sampler name
`omni_hybrid_gap` (CFG++ 1.0 + `gap_lf 0.05` + `gap_cap 0.2`) for a stock
KSampler workflow.

## Cost and caveats

* Any gap feature at `cfg = 1` forces the unconditional pass, so the model batch
  doubles per step -- the same cost CFG++ already pays at cfg 1. At `cfg > 1` it
  is free.
* The knob values above are **starting points, not measurements**. The branch
  exists to produce the measurement; do not promote any of them to a default
  before the matrix above says which one wins.
* `gap_lf` removes tone from the *guidance*, not from the image: a prompt that
  depends on a global colour cast ("a red-lit room") will lose some of it,
  because that part of the request now comes from the unconditional prediction.
  `gap_lf_keep` is the damping knob for that.
* The feature shapes guidance; it cannot add information the model does not have
  at 6 MP. If the model is simply out of distribution there, generating at
  1024-1408 and upscaling remains the reliable route, and that comparison is part
  of the matrix on purpose.

## Verification

`tools/gap_shape_check.py` covers what the feature must not do and the one thing
it is for: neutral is bit-identical, `cond == uncond` stays a no-op with every
knob on, the cap's bound and direction, the split's band algebra, the projection,
5-D latents, and an end-to-end tone-vs-structure demonstration through ComfyUI's
real post-CFG hook. The existing suites (`verify_math`, `cfgpp_check`,
`test_cross_features`, `verify_spatial`, `test_node`, `test_node_parity`) pass
unchanged.
