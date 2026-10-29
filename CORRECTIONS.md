# Corrections: two measurement bugs that invalidated the real-model numbers

This file records errors found *after* v1.1/v1.2 were tagged. The toy/analytic
results and the algebraic checks stand; the real-model image numbers did not.

## 1. Decoded images had the wrong shape (768x61, 768x1)

**Symptom.** Krea2 outputs were 768x61, Anima outputs 768x1. User-visible.

**Cause.** Two separate mistakes in `tools/modern_bench.py`:

* the Wan 2.1 VAE returns `(B, T, H, W, C)`, but `to_hwc` assumed
  `(B, C, T, H, W)` and indexed axis 2 (`[:, :, 0]`), which is the *height* axis.
  On a `(1,1,768,768,3)` decode that produces a `(1 x 768 x 3)` image.
* Krea2's latent format is `Wan21` (a video format, `latent_dimensions == 3`), so
  it needs a 5-D latent. Only Anima was given one, so Krea2 was decoded from a
  4-D latent and the VAE produced 61 rows.

**Fix.** The latent rank now comes from `latent_format.latent_dimensions`, and
`to_hwc` squeezes leading singletons and handles `(B,T,H,W,C)`, `(B,C,T,H,W)`,
`(B,C,H,W)` and `(B,H,W,C)`. Verified: 768x768 RGB out, `assert`ed in
`tools/latent_bench.py`.

**Consequence.** Every Krea2 and Anima number in the v1.1 notes (corrector
comparison, CFG++ at cfg=1, skip/flow verification) was computed on malformed
images and is void. MOSS (2-D Flux2 format) decoded correctly.

## 2. There is no converged high-NFE reference to measure against

**Symptom.** Rankings flipped with step count, and different samplers "won" at
adjacent step counts by large margins. The user suggested comparing latents
instead of decoded pixels; that exposed the real cause.

**Measurements** (relative L2 between final latents, same seed/prompt/model):

| model | 20 -> 40 steps | 40 -> 60 | 60 -> 100 | 40 -> 80 |
|---|---|---|---|---|
| Krea2 (turbo, cfg 1) | 3.6e-1 | 7.4e-2 | 2.6e-1 | - |
| MOSS (cfg 3.5) | 4.0e-1 | 2.7e-1 | - | - |
| Anima (cfg 4) | 3.6e-1 | 1.2e-1 | - | - |
| SDXL (eps, cfg 5) | 3.4e-1 (euler) | - | - | 4.1e-1 (euler) |

Run-to-run determinism is exact (0.0), so these are real differences. Visual
check of SDXL euler@40 vs euler@80 with the same seed: **different
compositions** (the fox is turned 3/4 in one and front-facing in the other), not
the same image with different detail. The guided trajectory bifurcates into
different basins depending on the discretisation, so "distance to a high-NFE
reference" measures *which mode you landed in*, not integration accuracy.

**Consequence.** All reference-based rankings on these models (SSIM/rel-L2 to an
80/60/40-step run) are not accuracy measurements. That includes the flow-model
evidence behind v1.1's `corrector="auto"` default and the Krea2 side of v1.2.

## What was done about it

* `corrector` default reverted to `first` (the v1.0 behaviour). `auto` is still
  available and documented as a heuristic, not a measured win.
* Real-model numbers in README/CHANGELOG are marked void; the analytic toy
  numbers (exact ODE reference, no mode ambiguity at the endpoint) stand.

## What a valid metric would look like

Per-step *local* error: for one interval `[sigma_i, sigma_{i+1}]`, integrate the
same starting latent with many fine substeps (e.g. 16 Euler steps) and compare
the sampler's single step against that local solution. The interval is tiny, so
the mode is fixed, and the number measures what a sampler is supposed to do
(approximate the ODE) rather than which basin it reaches. Cost is ~16x per
measured step, so it is affordable on a handful of steps.

Until that is implemented and validated, I do not claim any real-model accuracy
improvement over v1.0 — only the algebraic fixes (CFG++ guidance weighting,
churn geometry) and the ppm `euler_gamma` equivalence (2.4e-7).
