# A/B testing the open questions

`Omni-Hybrid KSampler (A/B)` is a full sampler node: model / positive /
negative / latent / seed / steps / cfg / scheduler / denoise, every Omni knob,
and a `variant` switch that selects a preset row. It returns the latent plus a
`config` string (also printed to the console) so two runs can be told apart in
the metadata. The same `variant` switch and knob set are on
`Omni-Hybrid Sampler (select)` for SamplerCustom / SamplerCustomAdvanced; that
node returns the SAMPLER instead of a latent, so it has no config string.

Pick `custom (use knobs)` to drive the knobs yourself; any named preset overrides
only the settings it mentions, so the rest keeps your knob values.

## Presets

| variant | what it changes | status |
|---|---|---|
| `A: shipped (v1.3.1)` | nothing — the defaults | the version that is validated algebraically and on the analytic harness |
| `B: no corrector` | `corrector=none` | open question: the step-0 corrector helps eps, unclear on flow |
| `C: skip every 2nd eval` | `skip_every=2, skip_order=2` | experimental: ~40% fewer model calls, quality unproven |
| `C2: skip every 2nd, order 3` | `skip_every=2, skip_order=3` | experimental, sharper extrapolation |
| `D: ppm gamma churn` | `churn_mode=gamma, s_churn=2.0` | reproduces comfyui-ppm `euler_gamma` (2.4e-7) |
| `E: CFG++ mild (0.5)` | `cfg_pp=True, cfg_pp_strength=0.5` | measured: cfg≈1.5x |
| `F: CFG++ full (1.0)` | `cfg_pp=True, cfg_pp_strength=1.0` | reference CFG++, measured: cfg≈2.5x |
| `G: order 1` | `order=1` | first-order baseline |
| `H: exact kernel` | `x0_average=exact` | measured worse than `ab2` |
| `I: SDE eta 0.3` | `eta=0.3` | stochastic; different image by design |
| `J: no gate` | `min_step_ratio=0` | disables the step-ratio guard |
| `K: per-pixel order` | `px_order=True` | experimental PD-AOS port; a tight tolerance cost 0.014 SSIM on Krea2 @8, a loose one was neutral (see [the PD-AOS audit](PD_AOS_AUDIT.md)) |
| `L: region CFG map` | `cfg_map=True` | experimental PD-AOS port; needs cfg > 1, moves MOSS @8 by 0.0142 image RMS (SSIM 0.9824) |
| `M: author pref (cfgpp+churn+eta+px)` | `cfg_pp=True, s_churn=100, churn_mode=gamma, eta=1.0, px_order=True` | the registered `omni_hybrid_cfgpp_author_pref` name as one row, so the combination can be compared with its parts at equal NFE; chosen by eye, no benchmark covers it |
| `N: gap HF (5%)` | `gap_lf=0.05, gap_lf_keep=0.0` | experiment branch only: move the delta's low-frequency band back to the unconditional prediction ([EXPERIMENT_GAP_SHAPING.md](EXPERIMENT_GAP_SHAPING.md)); unmeasured |
| `O: gap cap 0.2` | `gap_cap=0.2` | experiment branch only: per-pixel delta cap at 0.2x the local unconditional magnitude; unmeasured |
| `P: gap HF + cap` | `gap_lf=0.05, gap_cap=0.2` | experiment branch only: the combination the `omni_hybrid_gap` name uses; unmeasured |
| `Q: gap probe (no shaping)` | `gap_stats=True` | experiment branch only: measures and prints the per-step cond/uncond gap, output unchanged; at cfg = 1 the uncond pass is forced (2x batch) |

## Suggested way to compare

1. Fix the seed, steps, cfg and scheduler.
2. Queue variant A, then the variant you want to compare.
3. Each run prints `evals=measured/estimated`; compare at **equal NFE**, not
   equal step count — `skip_every=2` at 12 steps costs about as much as plain
   at 7.
4. The `config` string lands in the PNG metadata, so you can audit later which
   row produced which image.

There is no reference-based accuracy metric for these models yet (see
[CORRECTIONS.md](CORRECTIONS.md): the high-NFE trajectory bifurcates, so "closer
to 80 steps" measures mode choice, not quality). For now the judgement is
visual, which is exactly what this node is for.

## What the experiments already ruled out

`momentum`, `bongmath`, `antithetic drift`, `distance consensus`, `stork`,
guidance patches (CADS/TCFG/WaveletCFG/SMC/DCW), AFloPS' per-sample `c`, and the
flow-space AB2 integrator all lost on the analytic harness or in latent space.
They are not in the node. `skip_every` and `corrector` are the two that are
still genuinely open; `corrector_cfgpp` extends the corrector question to the
CFG++ path (it is opt-in and unmeasured — see [TECHNICAL.md](TECHNICAL.md#cfg-and-what-cfg-means)).
