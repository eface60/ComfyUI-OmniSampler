# Benchmark and verification harness

These scripts are how every number in [docs/TECHNICAL.md](../docs/TECHNICAL.md)
was produced. They are never imported by ComfyUI; `omni_core.py`,
`omni_nodes.py` and `__init__.py` are the whole runtime.

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
| [`cfgpp_check.py`](cfgpp_check.py) | CFG++ invariants (including the no-op with `px_order` and `corrector_cfgpp` on), the match with ComfyUI's `euler_cfg_pp`, and the effective-guidance table in [docs/TECHNICAL.md](../docs/TECHNICAL.md#cfg-and-what-cfg-means) |
| [`test_cross_features.py`](test_cross_features.py) | the pairs no single-feature test covers: `order` clamped on `exp`, `ab` + churn falls back to `exp`, `final_extrapolate` receives the churned evaluation sigmas, and karras churn never evaluates a flow model above `sigma_max` |
| [`test_node.py`](test_node.py) | the names appear in the stock dropdowns, all three nodes resolve them, and `KSAMPLER.sample()` lands within 4-6% of the analytic reference on both families |
| [`test_node_parity.py`](test_node_parity.py) | the advanced node exposes the A/B KSampler knobs minus the preset switch, the simple node exposes only the named settings, every widget has a tooltip naming every combo choice, defaults agree with `omni_core.DEFAULTS`, the widget order stays a prefix, and every knob reaches `sample_omni` |
| [`verify_spatial.py`](verify_spatial.py) | `px_order` / `cfg_map` OFF is bit-identical to the previous release, ON reaches its site, and the map/mask algebra is correct |
| [`gap_shape_check.py`](gap_shape_check.py) | guidance-gap shaping: neutral is bit-identical, `cond == uncond` stays a no-op with every knob on, the cap bound/direction, the split's band algebra, the global and local (PD-AOS-style) projection gates, 5-D latents, and a tone-vs-structure demonstration through ComfyUI's real post-CFG hook |
| [`gap_sweep.py`](gap_sweep.py) | the cross-model guidance-gap sweep (Krea2 / MOSS / Anima, resolution ladder, `gap_stats` only, no image change); appends one JSON record per config |
| [`gap_report.py`](gap_report.py) | reads a sweep: power law of the gap in area per family, the relative-response table, the spectral profile, the normalised per-step shape, and the step-count rows -- the tables in [docs/EXPERIMENT_GAP_SHAPING.md](../docs/EXPERIMENT_GAP_SHAPING.md) |
| [`test_eta_seed.py`](test_eta_seed.py) | `eta > 0` accepts negative seeds, reproduces the same latent for the same seed, and never builds a noise sampler when `eta=0` |
| [`test_node_emptylatent.py`](test_node_emptylatent.py) | the A/B node adapts a plain empty latent the way `common_ksampler` does |
| [`probe_orders.py`](probe_orders.py) | the exact-kernel multistep coefficients and why 3rd order was not shipped |
| [`toy.py`](toy.py) | the analytic test problem (Gaussian mixture, both families, DOP853 reference) the other scripts import |
| [`toy_bench.py`](toy_bench.py) | every built-in ComfyUI sampler against omni on an exact ODE - the table in [docs/TECHNICAL.md](../docs/TECHNICAL.md#measured-results) |
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
| [`local_metric.py`](local_metric.py) | per-interval local ODE error - the only mode-free real-model metric here; the refined reference uses the endpoint node and `max(--substeps, ceil(h/0.2))` substeps, named deterministic variants (`first,none,cfg_pp,...`), and a repeat gate that refuses to write numbers unless the first arm and its reference are bit-identical |
| [`spatial_ab.py`](spatial_ab.py) | `px_order` / `cfg_map` A/B with a repeated-base control |
| [`gap_shape_bench.py`](gap_shape_bench.py) | the guidance-gap matrix at a chosen resolution (default 1024x1408): plain / CFG++ / HF / cap / both / project, with reference-free burn + grain metrics and the per-step gap statistics; `--res 2048x2944` for the 6 MP case |
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
