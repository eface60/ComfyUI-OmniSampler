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

## Per-model failure modes at 6 MP

The author's notes on the same three checkpoints at 2048x2944, which the sweep
numbers agree with:

| model | 6 MP behaviour | cause | sampler leverage |
|---|---|---|---|
| Krea2 turbo (distilled, cfg 1) | burns at any effective cfg > 1; artifacts without guidance | a distilled checkpoint is trained to *reproduce* a guided output at cfg 1, so re-adding guidance oversaturates; 6 MP is also outside its training sizes | none from strength: any CFG++ strength > 0 adds guidance (below) |
| Anima base 1.0 | artifacts at cfg 1 *and* cfg > 1, no burn | 6 MP native is out of distribution; the artifacts are not guidance-shaped | none via guidance |
| MOSS base | fine | in distribution | -- |

**Why Krea2 has no non-burning CFG++ setting.** At cfg 1 with strength `s` the
step's extra guidance term is `s * alpha_t * e^-h * delta`, and the state
difference between re-noising with the unconditional and the conditional
prediction is exactly the same term:

```
sigma_t * (eps_u - eps_c) = sigma_t * alpha_s / sigma_s * delta = alpha_t * e^-h * delta
```

So "use the unconditional noise" and "add guidance" are not two things a knob
can trade off: `s = 0` is plain sampling, and every `s > 0` is guidance in
proportion to `s`. A checkpoint that burns at any cfg > 1 therefore has exactly
one non-burning CFG++ setting, and it is the one that no longer does the thing
that removed the artifacts.

This also reframes the sweep's main result: the 6 MP delta is *smaller* than the
1.4 MP one, so the burn is not more guidance -- it is a **lower tolerance for the
same guidance** at a size the model was not trained on. Which is why no
delta-magnitude or delta-spectrum transform in this branch can remove it.

## The fix under test

Leave the *amount* of guidance (cfg) alone and shape *what the gap does*, in x0
space, in the post-CFG hook:

| knob | transform |
|---|---|
| `gap_lf` | remove the low-frequency band of the delta (band width as a fraction of the short side); `gap_lf_keep` keeps a fraction of it |
| `gap_cap` | per-pixel cap `|delta| <= cap * |uncond|`, direction preserved |
| `gap_parallel` | keep that fraction of the delta's component along the conditional prediction (APG-style at 0) |
| `gap_parallel_local` | the same, per pixel and only on the hot tail: remove up to that fraction of the contrast push in the top decile of the step's own push distribution (the PD-AOS self-normalisation, applied to the delta) |

Applied in the order project -> split -> cap. The hook receives a denoised
prediction on every model family, so the same code covers eps and flow models,
and it covers plain `cfg > 1`, the forced-unconditional path at `cfg = 1`, and
CFG++ (CFG++ then uses the shaped delta as its signal and the raw uncond as its
noise). No extra model call.

## First real result (Krea2, 6 MP, author's run)

The local gate at `0.6` on every step, CFG++ 1.0, cfg 1, 2048x2944:

* **the ending burn is nearly completely gone** (the thing this whole branch was
  after), detail kept;
* but the structure glitches early -- "colour blobs that had to be resolved
  creatively" -- and the model recovers them over the rest of the run.

That is the expected failure of a *per-step* gate. It recomputes its decile
every step, and early in a trajectory the top decile is not the clipping tail:
it is composition-sized regions, so relieving their contrast fights the
structure being laid down, and the order-2 history carries the shape forward.
Late in the run the same decile *is* the highlight tail, which is what the
relief is for.

`gap_local_start` fixes that by keeping the early steps bit-identical: relief
ramps in only after a fraction of the **steps** (not of logSNR -- the extreme
sigmas eat most of the lambda span, so 30% of logSNR is step 2 of 12). The A/B
row is `S: gap local APG late (0.6 from 30%)`; `0.25-0.40` is the range.

Next: run `S` at 6 MP. If the early blobs are gone and the end burn stays
suppressed, the ramp becomes part of the recommendation, and the remaining
question is only how much relief the mid-trajectory needs.

## Sensors: what the sampler can measure about itself

All three are relative, because the sweep showed no absolute threshold transfers
across models (the gap's scale differs 4.7x between families, and the burn
threshold differs by model).

| sensor | what it is | what it sees | what it cannot see |
|---|---|---|---|
| `rev` | `||x0_t - x0_(t-1)|| / ||x0_t||`, the model's own revision rate | the **phase**: high while it is still deciding structure, low once the frame has settled. This is what `gap_local_start` hardcodes, measured instead of assumed. | a still-burning image can be settled, so it cannot see the burn |
| `oor` | fraction of the guided prediction beyond `3 * u_rms` per channel | the **burn**: how much of this x0 is outside the range the model itself produces. `u` is the prior, so the unit is the model's own sigma. | clipping in the decoded image (no VAE in the sampler); it is a proxy, not a saturation count |
| `local_mean` / `local_frac` / `local_ref` | the gate's own activity: mean gate, fraction of the frame above half, and the push quantile it used | the **where**: how much of the frame the relief touched and at what push level | whether those pixels are highlights or texture |

They cannot measure "error" in the accuracy sense: there is no target, and the
decoded image is not available inside the sampler. They measure self-consistency
and out-of-range-ness, which is the closest thing to "unwanted behaviour" that
exists at step time.

### Control rules to test, in order

1. **Start from `rev`, not from a fixed fraction.** `gap_local_start` assumes
   the structure phase ends at 30% of the steps; the measured version is "relief
   on once `rev` falls below a fraction of its early peak" (or its running
   median). One run's `gap_rev` series says whether that curve is clean enough to
   threshold. This is a **feed-forward** rule -- the relief is computed from the
   step's own measurement, never integrated over the run -- so it cannot
   oscillate the way a controller with memory would.
2. **Amount from `oor`.** Run plain (`s = 0`, `gap_stats`) once on the same
   model/size/prompt; its `gap_oor` series is the baseline for that model at that
   size. Then a CFG++ run can scale the relief by how far above that baseline it
   is: no baseline excess, no relief. That is the two-run calibration again, but
   on a quantity that is directly the symptom rather than on the gap.
3. **Per-pixel phase.** The natural next step if the global `rev` is not enough:
   a per-pixel revision map, normalised by the frame's median, so the relief is
   off where structure is still forming and on where it has settled. That is the
   PD-AOS jump map used in the opposite direction to its own.

Deliberately *not* proposed: an integrating controller on `oor` (it would
feedback into the thing it measures) and any absolute threshold.

## What is still worth testing

1. **`gap_parallel_local` (local APG) at 6 MP.** First run: the end burn is
   nearly gone and the detail is kept, but the early structure glitches; see
   the result section above. The next render is the late-only variant
   (`S: gap local APG late (0.6 from 30%)`, or `gap_local_start` 0.25-0.40),
   which leaves the first steps untouched. This is the live candidate.
2. **Plain at 6 MP** (no guidance at all). The author reports it is "decent" but
   still on the over-burned side; if it burns too, the model itself is
   over-contrast at that size and the local gate is the only thing that can
   subtract contrast without removing the structure.
3. **Two-stage workflow** instead of 6 MP native: base at 1024x1408 with CFG++
   (known good), latent upscale, then a *plain* `cfg 1` pass at 0.35-0.5 denoise
   at 6 MP. The detail pass has a short trajectory (little guidance) and, being
   plain, adds none.

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
