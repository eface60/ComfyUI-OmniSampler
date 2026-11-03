# Documentation

The [README](../README.md) is the entry point and the
[technical reference](TECHNICAL.md) holds the details; the files below carry the
evidence behind both. Every number is backed by a script in [`tools/`](../tools)
that can be re-run.

| document | what it answers |
|---|---|
| [TECHNICAL.md](TECHNICAL.md) | What the integrator does differently, every knob and its default, all benchmarks, and the caveats behind them. |
| [ANALYSIS.md](ANALYSIS.md) | What each of ComfyUI's 45 built-in samplers actually does, and which parts Omni-Hybrid reuses — with the measured number for every component that was rejected. |
| [AB_TESTING.md](AB_TESTING.md) | The A/B node's preset rows, how to compare two runs at equal NFE, and which knobs are still genuinely open questions. |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Leads taken from other custom-node packs, what was measured for each, and why nothing from them shipped. |
| [CORRECTIONS.md](CORRECTIONS.md) | Two measurement bugs that invalidated the real-model numbers, what that leaves standing, and what a valid metric has to look like. |
| [PD_AOS_AUDIT.md](PD_AOS_AUDIT.md) | An audit of a third-party "per-pixel" sampler's spatial mechanisms: what they really do, what was ported, and what was deliberately left out. |
| [EXPERIMENT_GAP_SHAPING.md](EXPERIMENT_GAP_SHAPING.md) | The experiment branch for the 6 MP Krea2 turbo burn: the hypothesis (tone vs magnitude in the cond/uncond gap), the shaping knobs, the matrix, and what would falsify it. |

## How to read the numbers

* The **analytic** harness ([`tools/toy_bench.py`](../tools/toy_bench.py)) has an
  exact reference integrated by DOP853, so its relative errors are real
  integration errors.
* The **real-model** tables are reference-based (SSIM against a high-step run of
  another sampler). Those models have no converged reference: the guided
  trajectory bifurcates, so a difference in that metric is mostly *which mode the
  run landed in*, not integration accuracy. See
  [CORRECTIONS.md](CORRECTIONS.md#2-there-is-no-converged-high-nfe-reference-to-measure-against).
* The **Krea2 and Anima** numbers from the v1.1 round were additionally computed
  on malformed decodes and are void (
  [CORRECTIONS.md](CORRECTIONS.md#1-decoded-images-had-the-wrong-shape-768x61-768x1)).
* The **local-error** metric ([`tools/local_metric.py`](../tools/local_metric.py))
  is the only mode-free real-model instrument. It now refuses to write numbers
  unless its first arm and refined reference repeat bit-exactly; MOSS and Krea2
  both failed that gate at 10 steps, so their local-error rows are not quotable
  ([CORRECTIONS.md](CORRECTIONS.md)).
