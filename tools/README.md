# Benchmark and verification harness

These scripts are how every number in the README was produced. They are never
imported by ComfyUI; `omni_core.py`, `omni_nodes.py` and `__init__.py` are the
whole runtime.

## Running them

The scripts need ComfyUI's own venv (they import `torch`, `numpy` and `comfy`).
They locate ComfyUI from `COMFY_DIR` and fall back to the path the author's
machine used, so a clone elsewhere only needs one variable:

```powershell
$env:COMFY_DIR = "C:\path\to\ComfyUI"
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\verify_math.py
```

| variable | default | used for |
|---|---|---|
| `COMFY_DIR` | `G:\AI\comfy` | the ComfyUI checkout to import (`comfy`, `comfy.k_diffusion`, …) |
| `OMNI_WORK` | the parent directory of this repository | where the `omni_sampler` package is imported from |
| `PD_AOS_DIR` | `G:\AI\comfyui-research-samplers` | the PD-AOS working tree, only for `tools/pd_aos_audit/` |
| `AFLOPS_OUT` | `$COMFY_DIR\output\aflops` | where the PD-AOS run-log corpus is read from |

## Verification (no model weights needed)

| script | what it proves |
|---|---|
| [`verify_math.py`](verify_math.py) | every formula in `omni_core`: exact-step trajectory, the ODE identity, SDE marginals, quadrature kernel, convergence order, CFG++ algebra, model-family detection, equality with `dpmpp_2m`'s first step |
| [`cfgpp_check.py`](cfgpp_check.py) | CFG++ invariants, the match with ComfyUI's `euler_cfg_pp`, and the effective-guidance table printed in the README |
| [`test_node.py`](test_node.py) | the names appear in the stock dropdowns, both nodes resolve them, and `KSAMPLER.sample()` lands within 4-6% of the analytic reference on both families |
| [`test_node_parity.py`](test_node_parity.py) | both nodes expose the same knobs from one definition, every widget has a tooltip naming every combo choice, defaults agree with `omni_core.DEFAULTS`, the widget order stays a prefix (so saved workflows keep their values), and every knob reaches `sample_omni` |
| [`verify_spatial.py`](verify_spatial.py) | `px_order` / `cfg_map` OFF is bit-identical to the previous release, ON reaches its site, and the map/mask algebra is correct |
| [`test_eta_seed.py`](test_eta_seed.py) | `eta > 0` accepts negative seeds, reproduces the same latent for the same seed, and never builds a noise sampler when `eta=0` |
| [`test_node_emptylatent.py`](test_node_emptylatent.py) | the A/B node adapts a plain empty latent the way `common_ksampler` does |
| [`probe_orders.py`](probe_orders.py) | the exact-kernel multistep coefficients and why 3rd order was not shipped |
| [`toy.py`](toy.py) | the analytic test problem (Gaussian mixture, both families, DOP853 reference) the other scripts import |
| [`toy_bench.py`](toy_bench.py) | every built-in ComfyUI sampler against omni on an exact ODE - the table in the README |
| [`tune_integrator.py`](tune_integrator.py) | `exp` vs Adams-Bashforth, and against `ipndm` |
| [`tune2.py`](tune2.py) | the remaining defaults: `lower_order_to_end`, `final_extrapolate`, `corrector` |

## Real-model benchmarks (need the checkpoints installed in ComfyUI)

| script | what it measures |
|---|---|
| [`sd_bench.py`](sd_bench.py) | SDXL image comparison against a high-step reference |
| [`sd_cfgpp.py`](sd_cfgpp.py) | CFG++ at cfg=1 on a real SDXL model |
| [`modern_bench.py`](modern_bench.py) | Krea2 / MOSS / Anima in one process (int8 checkpoints are only reproducible within a session) |
| [`flow_bench.py`](flow_bench.py) | PixelDiT, a real flow-matching model |
| [`latent_bench.py`](latent_bench.py) | the same comparison in latent space, with decoded previews optional |
| [`local_metric.py`](local_metric.py) | per-interval local ODE error - the only mode-free real-model metric here |
| [`spatial_ab.py`](spatial_ab.py) | `px_order` / `cfg_map` A/B with a repeated-base control |
| [`test_node_ab.py`](test_node_ab.py) | the full A/B node on a real flow model: four presets, wall time, latent distance |
| [`nfe_audit.py`](nfe_audit.py) | real model evaluations and wall time per variant |
| [`time_breakdown.py`](time_breakdown.py) | model evaluations vs sampler overhead |
| [`time_coldwarm.py`](time_coldwarm.py) | cold vs warm vs previews, to separate load time from sampling time |
| [`cfgpp_cost.py`](cfgpp_cost.py) | the cost of CFG++ at cfg=1 vs cfg>1 |
| [`cfgpp_forwards.py`](cfgpp_forwards.py) | whether CFG++ runs the unconditional pass twice at cfg>1 |
| [`cfgpp_repeat.py`](cfgpp_repeat.py) | interleaved repeats, medians - the timing noise the single-shot numbers hid |
| [`ppm_gamma_check.py`](ppm_gamma_check.py) | comfyui-ppm's `euler_gamma` (and its CFG++ branch) against omni |
| [`diag_steps.py`](diag_steps.py) | per-step divergence between omni and ComfyUI's samplers |
| [`verify_winners.py`](verify_winners.py) | GPU check of two candidate variants on a modern flow model |

## PD-AOS audit

| script | what it does |
|---|---|
| [`pd_aos_audit/scan_corpus.py`](pd_aos_audit/scan_corpus.py) | scans the PD-AOS run-log corpus, validating fields before use; `--control` injects defects and requires them to be rejected |
| [`pd_aos_audit/pd_aos_ab.py`](pd_aos_audit/pd_aos_ab.py) | runs the shipped PD-AOS sampler and omni on one model/seed, in-process, with a control arm |

Findings from these land in [docs/PD_AOS_AUDIT.md](../docs/PD_AOS_AUDIT.md).

## Conventions

* A script prints what it measured and exits non-zero if a check failed; a green
  run must be able to go red.
* Numbers are only quoted from a re-runnable script; a number whose script
  changed has to be measured again.
* Real-model comparisons run in one process. Byte-identical reruns across
  sessions are not available for the int8 checkpoints, so a cross-session A/B
  measures noise as much as it measures the sampler.
