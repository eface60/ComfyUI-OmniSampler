# Contributing

Thanks for looking. This is a measurement-driven sampler, so the bar for a
change is not "it seems better" but "here is the number, and here is the script
that produces it".

## Ground rules

1. **A behaviour change ships with a measurement.** If you change a default or
   add a knob, say in the pull request which measurement justifies it, and
   commit the script that produces that measurement. If a change cannot be
   measured, say so explicitly and keep the old behaviour as the default.
2. **A finding lives in a script plus its output, not in prose.** If your change
   contradicts a number in `docs/`, update that number or record the correction —
   do not leave two versions of the same claim.
3. **Do not silently rename a widget.** ComfyUI stores widget values
   positionally, so inserting, removing or renaming a widget re-maps saved
   workflows. Append new widgets at the end; keep `variant` preset names stable.
4. **A new knob must be provably reachable and provably inert when off.** ON
   enters the new code path and changes the quantity it claims to change; OFF is
   bit-identical to the previous release.

## Development setup

The runtime needs nothing beyond what ComfyUI already installs (`torch`,
`numpy`, the `comfy` package). The harness needs ComfyUI's own venv:

```powershell
# from the repository root
$env:COMFY_DIR = "C:\path\to\ComfyUI"

# the checks that need no model weights
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\verify_math.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\cfgpp_check.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\test_cross_features.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\test_node.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\test_node_parity.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\verify_spatial.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\gap_shape_check.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\test_eta_seed.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\test_churn_eta.py
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\noise_cost.py --check
& "$env:COMFY_DIR\venv\Scripts\python.exe" tools\probe_orders.py
```

All eleven must pass, and `test_node_parity.py` must pass **whenever you touch a
widget**, because it is the only thing that catches a knob that exists on one
node and not the other, or a tooltip that stopped describing its combo.

[`tools/README.md`](tools/README.md) lists every script, what it proves, and the
environment variables they read. Real-model benchmarks additionally need the
checkpoints they name installed in ComfyUI; they are not part of a normal pull
request.

## Where things go

| if you have… | put it in |
|---|---|
| a change to the integrator | `omni_core.py`, plus a check in `tools/verify_math.py` |
| a change to a node, a widget or a tooltip | `omni_nodes.py`, plus parity coverage in `tools/test_node_parity.py` |
| a bug found by a measurement | `docs/CORRECTIONS.md` (and mark the affected numbers in `docs/TECHNICAL.md`) |
| a lead that did not ship | `docs/EXPERIMENTS.md` |
| a change users will notice | `CHANGELOG.md`, under a version heading with a date |
| a new benchmark | `tools/`, with a docstring saying what it measures and how to read the result |

Versions are tagged `vMAJOR.MINOR[.PATCH]` and `CHANGELOG.md` is newest-first.

## Style

* Python 3.10+, `from __future__ import annotations`, standard library plus
  `torch`/`numpy` only.
* Docstrings explain *why* an implementation looks the way it does; the formula
  and the measured reason are both worth a line.
* Comments must stay true. If you rename or remove an identifier, grep the files
  you touched for it — a comment naming something that no longer exists is worse
  than no comment.
* No formatter is enforced; keep lines under ~100 characters and match the
  surrounding code.

## Reporting a bug

Include what you supplied (model family, sampler name or preset, steps, cfg,
scheduler, seed), what came out (the `config` string from the A/B node is the
shortest complete record, and the console prints it), and what you expected
instead. If the sampler raised, the full traceback plus the ComfyUI version.
