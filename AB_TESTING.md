# A/B testing the open questions

`Omni-Hybrid KSampler (A/B)` is a full sampler node: model / positive /
negative / latent / seed / steps / cfg / scheduler / denoise, every Omni knob,
and a `variant` switch that selects a preset row. It returns the latent plus a
`config` string (also printed to the console) so two runs can be told apart in
the metadata.

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

## Suggested way to compare

1. Fix the seed, steps, cfg and scheduler.
2. Queue variant A, then the variant you want to compare.
3. Each run prints `evals~N`; compare at **equal NFE**, not equal step count —
   `skip_every=2` at 12 steps costs about as much as plain at 7.
4. The `config` string lands in the PNG metadata, so you can audit later which
   row produced which image.

There is no reference-based accuracy metric for these models yet (see
`CORRECTIONS.md`: the high-NFE trajectory bifurcates, so "closer to 80 steps"
measures mode choice, not quality). For now the judgement is visual, which is
exactly what this node is for.

## What the experiments already ruled out

`momentum`, `bongmath`, `antithetic drift`, `distance consensus`, `stork`,
guidance patches (CADS/TCFG/WaveletCFG/SMC/DCW), AFloPS' per-sample `c`, and the
flow-space AB2 integrator all lost on the analytic harness or in latent space.
They are not in the node. `skip_every` and `corrector` are the two that are
still genuinely open.
