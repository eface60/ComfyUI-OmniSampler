# PD-AOS spatial mechanisms: what worked, what did not, and what was ported

**Scope.** PD-AOS working tree at `comfyui-research-samplers/pd_aos` (junctioned
into ComfyUI's `custom_nodes/pd_aos`), commit `27db101`
(branch `ab-audit-2026-10-19`), engine `research_samplers/aflops.py`. The three
mechanisms the operator asked about are:

1. **per-pixel CFG** — the stage-2 per-region guidance map,
2. **per-pixel order** — the `order_pixel_adaptive` ladder and the
   `second_order_per_pixel` mask,
3. **local fields** — the `lf_*` block/fixel step modulation.

Every number below comes from an artifact produced by the scripts in
`tools/pd_aos_audit/` (corpus scan) or `tools/verify_spatial.py` /
`tools/spatial_ab.py` (ports). The re-runnable commands are at the end.

---

## 1. What the code actually does

### 1.1 Per-pixel CFG: an 8x8 region map, late-only, on top of a scalar ramp

- The map is built by `_cfg_schedule_prior` from the prompt probe's
  `gap["prior"]` (`_gap_outputs`, `aflops.py`, symbol `out["prior"]`). The prior
  is a channel-RMS map of `x0_cond - x0_uncond` at one mid-probe step,
  normalised by its p90 and pooled to 16x16, then to the probe latent.
  On the corpus runs the cond probe runs at 16 px, whose MOSS latent is 8x8:
  **the "per-pixel" map is 64 regions of 128x128 pixels at 1024²** (logged
  `h=8 w=8 n=64` in `cond_probe_profiles`).
- Stage 2 normalises that map again to [0,1] (`_cfg_schedule_scale`'s caller,
  `_pn = ((_pn - _pmin) / (_pmax - _pmin))`) and sets
  `map = step_scale * (0.85 + 0.15 * prior)`. The post-CFG hook divides by the
  step's guidance scale, so the realised ratio is **0.85..1.0 of the guidance
  delta** — a 15% floor, never less.
- Stage 1 is a scalar ramp: `scale = base * (0.5 + 0.5*smoothstep(...))`, where
  progress is `t = 1 - s/sigma_max` and the ramp starts at `t = 0.7`
  (`_CFG_SCHEDULE_START`). Stage 2 is only set when the ramp's smoothstep is at
  least 0.5 (`_CFG_SCHEDULE_STAGE2_FROM`).

### 1.2 Per-pixel order: real per-pixel x0 selection, one global step coefficient

- `_pixel_order_bound` / `_pixel_innovations` choose, per pixel, an order in
  [1..P] from the innovation ladder and/or the Lagrange remainder
  (`order_by_bound`). `_mid_x0` builds the candidate x0 at the step midpoint for
  each order; the chosen candidate is blended with the next-lower one by a
  per-pixel history cosine (`order_pixel_blend`).
- The **step itself** uses one scalar decay rate:
  `lam_raw = (delta_v . delta_x) / ||delta_x||²` over the whole flattened latent
  (`_aflops_step`). The per-pixel part is the x0 estimate (and the
  `second_order_per_pixel` mask on the slope term), not the exponential
  coefficient.
- The engine's own docstring for `_pixel_order_bound` records that the shipped
  innovation criterion "on a real model settles near order 1 — `ord_want`
  1.03–1.23".

### 1.3 Local fields: a bounded per-pixel multiplier on the drift term

- `_lf_step_field` produces `m_off` (and a refresh `rho`); the step application is
  `x = x0 + (1 + m_off) * (s2/s) * (x_input - x0) + (x - x1)`.
- The amplitude is `amp = 0.15 * clamp`, with `clamp` in [0.10, 0.48]
  (`lf_gains` build in `_aflops_engine_body`), so `|m_off|` is bounded by about
  0.015–0.072; the modal typical value is much smaller (below).

---

## 2. Corpus measurements (2,210 valid runs)

`tools/pd_aos_audit/scan_corpus.py` reads every
`%COMFY_DIR%\output\aflops\*\log_*.json` (override with `--root`)
and validates the fields before using them (range checks, map ordering, `m_min/m_max`
bounds; eight injected-defect controls all rejected). It counted:

| mechanism | statistic | value |
|---|---|---|
| per-pixel CFG | runs with stage 2 installed | 691 |
| | of those, runs with at least one logged map | 330 (47.8%) |
| | mapped steps per stage-2 run | p50 **0**, p90 2 (max 7) |
| | map min/max over 475 logged maps | **0.850** at p10, p50 and p90 |
| | uniform maps | 0 |
| per-pixel order | `ord_mean` over 61,515 steps | p10 1.000, p50 **1.144**, p90 1.945 |
| | `second_order_px` over 55,094 steps | p50 **4.6%**, p90 25.8% |
| local fields | mean `abs(lf_m)` per step | p50 **0.25%**, p90 0.45%, max 3.25% |
| | runs with any step above 1% | 796 / 2,210 |
| | `rho_max` per step | p50 0.37%, p90 1.70% |

The map's range is exactly 0.85 by construction (the floor); the spatial content
is *where* the floor sits, not how deep it is. The per-pixel order map is real
and spatially non-uniform (the earlier `rig_order_spatial_census` measured
non-uniformity on 339/369 ladder-active steps), but the admitted order is
mostly 1 and the second-order term reaches a median 4.6% of pixels. The local
field's typical steering is 0.25% of one term of the step.

---

## 3. Direct A/B, and the instrument that could not resolve it

`tools/pd_aos_audit/pd_aos_ab.py` runs the shipped Autotuner + Sampler on one
model, same prompt/seed/schedule, changing one switch per arm. Two arms are the
same configuration (`base`, `base2`) as the reproducibility control.

| model / schedule | control `base` vs `base2` | `ppo_off` vs base | `lf_off` vs base |
|---|---|---|---|
| MOSS, 12 steps, 1024², simple | latent rel 9.3e-2; image RMS 0.00999 | image RMS 0.01039 | 0.00987 |
| Krea2, 12 steps, 1024², simple | latent rel 3.7e-2; image RMS 0.00546 | 0.00355 | 0.01488 |

**The control failed on both models**: PD-AOS's own runs are not reproducible
within one session on these checkpoints (step 2 already differs at 5e-4
relative, then amplifies). The `ppo_off` and `lf_off` distances are at or below
that floor on MOSS and on the same order on Krea2, so **this instrument cannot
attribute a visible difference to either switch**. (`lf_off`'s Krea2 number is
confounded: its step-1 `x0` already differs, i.e. the switch changed the probe
profile, not only the step field.)

Per-pixel CFG was measured a different way, because it is inert unless the run
reaches the map. On a 12-step **simple** MOSS schedule the last sigma is 0.314,
so `t = 1 - s/sigma_max` never exceeds 0.686: the stage-1 ramp never starts,
`cfg_eff` stays at `0.5*base` for all 12 steps, and stage 2 never sets a map.
The two arms are **bit-identical** (`latent abe06def2ef71554` for both) while
`stage1` changes the image by 0.0595 RMS against the no-schedule base. That is
the largest effect in this audit and it is the *unintended* one: on short
"simple" schedules the documented "half early, full late" is half throughout.

On a 12-step **Karras** MOSS schedule the ramp does start (1.75 -> 3.50) and
stage 2 maps steps 6–12 at 0.85*cfg. The stage-1 vs stage-2 image difference is
**0.00161 RMS (0.16% of range, max 3.9%)** — real, late, and small, which is
what a 15% scaling of a guidance delta that is itself a small part of the
prediction should look like.

---

## 4. Verdicts

### 4.1 Per-pixel CFG — real, but not per-pixel, often inert, and small

**Worked as intended?** Partly. The post-CFG composition, the 0.85 floor and the
per-step map are wired and act (330 runs, maps logged, 0.16% image effect on the
one schedule where it fires). It is **not** per-pixel: it is an 8x8 region map
from a 16 px prompt probe. It is **inert on short simple schedules** because the
ramp's start is tied to `1 - s/sigma_max`; the map needs the ramp's smoothstep
above 0.5, which never happens when the schedule ends above `0.3*sigma_max`.
It also cannot act at cfg 1 (no cond/uncond gap, so no prior); the corpus has
0 stage-2 runs at cfg<=1, which is consistent with that path never producing a
map.

**Worth porting?** Only in corrected form, and as a deliberate visual option,
not as an accuracy improvement. The corrected form keeps the idea (calm the
low-gap regions late) and fixes the progress frame: use the trajectory fraction
over the actual sigma span, not `1 - s/sigma_max`. That is what
`cfg_map`/`cfg_map_floor`/`cfg_map_start` in `omni_core.py` do. The map is
measured from the run's own mid-trajectory cond/uncond gap, so there is no
probe, no cache, and no extra NFE; it requires cfg > 1 and refuses CFG++.

### 4.2 Per-pixel order — real, but its effect was not a win

**Worked as intended?** The mechanism is real: the order map is spatially
varying, and `second_order_per_pixel` masks the slope term per pixel. But the
admitted order is mostly 1 (median `ord_mean` 1.144; second-order on 4.6% of
pixels), the step's exponential coefficient is global, and the engine's own
criterion describes this as the typical real-model outcome. As a result the
shipped switch is not distinguishable from run-to-run noise in the PD-AOS A/B
above.

**Worth porting?** Ported in the reduced form omni can support: a per-pixel
Lagrange-remainder test on the order-2 x0 extrapolation (`px_order`). It costs
no model call and OFF is bit-identical. But the measurement is a **caution, not
a win**: on Krea2 at 8 steps (512²), SSIM to a 40-step reference was

| setting | SSIM |
|---|---|
| plain omni | 0.8620 |
| `px_order_tol=0.05` | 0.8477 |
| 0.2 | 0.8550 |
| 0.5 | 0.8587 |
| 1.0 | 0.8592 |
| 2.0 | 0.8617 |
| 5.0 | 0.8608 |

(One process, one reference, so the rows are comparable; an earlier run of the
same sweep reproduced the 0.05 row to 1e-4 but its reference differed, which is
why this table is quoted instead.)

A tight gate turns most pixels into order 1 and loses 0.014 SSIM; a loose gate
is neutral. The shipped default is therefore `px_order_tol=2.0`, the neutral
value, and `px_order=False`; a user who wants the gate must choose how strict it
is, and the tooltip says what the sweep found.

### 4.3 Local fields — real but too small to carry

**Worked as intended?** The field is computed and applied every step, but its
typical amplitude is 0.25% of the drift term (p90 0.45%), the refresh is 0.37%
of the state, and the PD-AOS A/B could not separate `lf_off` from the run's own
drift. The subsystem depends on probe-derived gains; without the probe stack it
has no inputs. Several of its internals are known-defective from the PD-AOS
audit (border padding in the detail map, probe/run `d` mismatch, a vacuous floor
conjunct, dwell reshape for batch>1), none of which changes the conclusion: the
typical effect is far below what the eye or a metric can resolve.

**Worth porting?** No. There is no self-contained version that preserves the
mechanism (the gains are probe measurements), and the measured amplitude does
not justify porting the probe stack for it.

---

## 5. What was ported, and how it is proven

Both additions live in `omni_core.py`, are appended to both nodes' widget lists,
and are **OFF by default**:

| option | default | what it does |
|---|---|---|
| `px_order` | False | per-pixel Lagrange-remainder gate on the order-2 x0 extrapolation |
| `px_order_tol` | 2.0 | allowance in units of the doubled Lagrange bound (see `_px_order_mask`) before a pixel falls back to order 1 |
| `cfg_map` | False | late per-region guidance map, measured from the run's own cond/uncond gap |
| `cfg_map_floor` | 0.85 | guidance fraction kept in the lowest-gap region |
| `cfg_map_start` | 0.5 | trajectory fraction where the map is measured |
| `cfg_map_pool` | 8 | region grid (8x8) |

`tools/verify_spatial.py` proves, with numbers:

- OFF is **bit-identical** to `git show v1.4.2:omni_core.py` on the toy problem
  (`maxdiff 0.000e+00`);
- `px_order` with an infinite tolerance is bit-identical (negative control), the
  tight value reaches its site (2 steps, mean 0.0 on the toy mixture), the same
  value reproduces bit-exactly, and the analytic checks separate a linear
  trajectory (mask 1) from a quadratic one (mask 0) and curved from flat pixels
  (0.000 vs 1.000 before the blur);
- `cfg_map` builds on a fake CFG model (6 steps), changes the path, reproduces
  bit-exactly, floor 1 is bit-identical, cfg 1 does not build, and the map
  scales the delta rather than the prediction (`torch.allclose` to the algebra).

The existing `tools/verify_math.py` and `tools/test_node.py` still pass, i.e.
the OFF path did not move.

`tools/spatial_ab.py` is the operator's A/B: same prompt/seed/schedule, arms
`base`, `base2` (reproducibility control), `pxo`, `cfgm`, `both`, with
`LAST_STATS` printed and latent/PNG outputs saved. On Krea2 (8 steps, 512²) the
base arm is bit-reproducible (control distance 0.0), which is what makes the
`px_order` numbers in §4.2 admissible.

On MOSS (8 steps, 512², cfg 3.5) the same control is bit-identical, `cfgm`
builds its map at step 3 and applies it on 6 steps, and the decoded image moves
by **0.01424 RMS (SSIM 0.9824 to base)**; `both` moves 0.01190 RMS
(SSIM 0.9841). That is a real, measurable effect from the corrected map — the
operator's eye is the instrument for whether it is the one they remember.

---

## 6. What was deliberately not ported

- **The scalar guidance ramp (stage 1).** It is the part that visibly changes an
  image, but it is global, not per-pixel, and its progress frame is the bug
  above. Porting it would carry the bug or silently change the sampler's default
  guidance; if the operator wants it, it should be a separate, correctly-framed
  feature with its own A/B.
- **Local fields** (§4.3).
- **The PD-AOS probe stack** (model/cond probes, profile cache, tolerance and
  cap derivation). Nothing here needs it: the ported map is measured from the
  run's own predictions.

## 7. Reproduce

Run from the repository root, with the ComfyUI venv python (`$env:COMFY_DIR`
points at your ComfyUI checkout; see [CONTRIBUTING.md](../CONTRIBUTING.md)):

```powershell
$py = python
# corpus (2,210 runs; controls must all be REJECTED)
& $py -X utf8 tools/pd_aos_audit/scan_corpus.py --control
& $py -X utf8 tools/pd_aos_audit/scan_corpus.py
# PD-AOS direct A/B (one model, same prompt/seed; control arm included)
& $py -X utf8 tools/pd_aos_audit/pd_aos_ab.py krea2 --steps 12 --res 1024 --arms base,base2,ppo_off,lf_off
& $py -X utf8 tools/pd_aos_audit/pd_aos_ab.py moss --steps 12 --res 1024 --scheduler karras --arms cfg_stage1,cfg_stage2
# port connectivity + real-model A/B
& $py -X utf8 tools/verify_spatial.py
& $py -X utf8 tools/spatial_ab.py krea2 --steps 8 --res 512 --arms base,base2,pxo
& $py -X utf8 tools/spatial_ab.py moss --steps 8 --res 512 --arms base,base2,cfgm
```

## 8. Not done

- No multi-seed averaging: every A/B above is one seed. The Krea2 omni control
  was bit-exact, so its differences are real for that seed; the PD-AOS controls
  were not, so those rows are labelled unresolved rather than compared.
- No operator-eye A/B: §4.2's SSIM is an instrument for maths, not appearance.
  `tools/spatial_ab.py` exists so that decision can be made on images.
- No test of `cfg_map` on a second non-distilled model at another step count;
  one MOSS run is a connectivity pre-flight and a first effect size, not a
  population.
