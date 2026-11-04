# ComfyUI-OmniSampler

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-2ea44f)](https://github.com/comfyanonymous/ComfyUI)
[![Latest tag](https://img.shields.io/badge/tag-v1.5.5-informational)](CHANGELOG.md)

One diffusion sampler for ComfyUI that works correctly on **both** kinds of model
it can run: the older epsilon-prediction models (SDXL and friends) and the
flow-matching models that most current checkpoints use.

It recombines the parts of ComfyUI's 45 built-in samplers that measured best, and
fixes the step size that `dpmpp_2m`, `ipndm`, `res_multistep` and friends compute
for epsilon models only — a mismatch that costs them accuracy on flow models.

It installs as a normal custom node: four extra names in the stock sampler
dropdowns (this branch adds a fifth, the experimental `omni_hybrid_gap`), plus
two nodes if you want control over every knob. Every default is backed by a
measurement, and the scripts that produced them ship with the repo.

## Install

```powershell
cd path\to\ComfyUI\custom_nodes
git clone https://github.com/eface60/ComfyUI-OmniSampler.git
```

Restart ComfyUI. No extra packages are needed: the sampler uses the `torch`,
`numpy` and `comfy` modules ComfyUI already loads, and nothing inside ComfyUI is
modified on disk — the sampler names are appended to ComfyUI's in-memory
registries at import time.

## Usage

### Pick a sampler name

In **KSampler**, **KSamplerAdvanced** or **KSamplerSelect**, choose one of the
four names where you would normally choose `dpmpp_2m`:

| name | what it is | when to pick it |
|---|---|---|
| **`omni_hybrid`** | the measured defaults: deterministic, second order, one extra evaluation on the first step | start here |
| `omni_hybrid_sde` | the same integrator with SDE noise in the middle of the trajectory (`eta=0.5`) | when you want more variation between seeds |
| `omni_hybrid_cfg_pp` | CFG++ guidance: the unconditional prediction becomes the noise estimate | normal-cfg models (cfg > 1), not distilled cfg-1 checkpoints |
| `omni_hybrid_cfgpp_author_pref` | the author's everyday setting: CFG++, gamma churn at maximum, `eta=1`, per-pixel order gate | a preference, not a measured optimum — try it if you like the look |
| `omni_hybrid_gap` | **experiment branch only**: CFG++ 1.0 with the contrast component at 60% and the hot tail flattened after 30% of the steps (`gap_parallel 0.6`, `gap_parallel_local 0.6`, `gap_local_start 0.3`) | the 6 MP Krea2 burn fix ([the experiment](docs/EXPERIMENT_GAP_SHAPING.md)); judged harmless at 1.4 MP and on MOSS, not part of a release |

Steps, cfg, scheduler and denoise work exactly as they did before; the sampler
only replaces the step update.

### Or use a node

| node | use it for | returns |
|---|---|---|
| **Omni-Hybrid KSampler (A/B)** | the whole sampling graph, plus a `variant` switch that flips between preset rows for A/B runs | `LATENT`, plus a `config` string (also printed) with the exact evaluation count |
| **Omni-Hybrid Sampler (select)** | SamplerCustom / SamplerCustomAdvanced | `SAMPLER` |

Both nodes expose the same knobs and the same tooltips; the difference is what a
node is allowed to own — a SAMPLER object cannot own model, conditioning, latent,
seed, steps, cfg, scheduler or denoise.

### A sane first run

- Normal-cfg models (cfg 4-7): use the defaults, change nothing else.
- Distilled cfg=1 models (turbo/lightning): keep cfg 1 and use plain
  `omni_hybrid` — CFG++ adds guidance, and those models were trained without it.
- Cost: one extra model evaluation per run versus `euler` (the step-0
  corrector); otherwise the same number of evaluations.

### Tuning

Every knob has a tooltip in ComfyUI. The
[technical reference](docs/TECHNICAL.md#parameters) documents all of them, their
defaults, and the measurement behind each default.

## Known limitations

- CFG++ at cfg 1 costs a second model pass per step — that is inherent to the
  method, not a bug.
- Flow-model support is verified analytically and on small flow models, not yet on
  a very large flow checkpoint.
- Real-model comparisons are relative to high-step runs that are not fully
  converged; the [technical reference](docs/TECHNICAL.md#measured-results)
  explains what that does and does not show.

## Documentation

- **[Technical reference](docs/TECHNICAL.md)** — how the integrator works, every
  knob, all benchmarks, and the caveats.
- [Evidence and audits](docs/README.md) — the built-in sampler analysis, the
  experiments that were rejected, the corrections, the PD-AOS audit.
- [CHANGELOG.md](CHANGELOG.md) — what changed in each version and why.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to run the harness, and what a change
  needs.
- [tools/README.md](tools/README.md) — every verification and benchmark script,
  and what it proves.

## License

[MIT](LICENSE).
