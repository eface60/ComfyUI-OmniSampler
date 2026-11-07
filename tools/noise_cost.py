"""What the ``eta`` noise draw costs per step, with no checkpoint loaded.

The sampler asks the noise object for one unit-variance field per step:
``noise_sampler(sigma_s, sigma_t)``.  Three implementations answer that call:

* ``white``     -- ``torch.randn`` on the latent's device (comfy's
  ``default_noise_sampler``).  The sampler's default.
* ``brownian``  -- ``BrownianTreeNoiseSampler`` over torchsde's dyadic
  ``BrownianTree``, built on the CPU (comfy's default for the class).
* ``brownian-gpu`` -- the same tree built with ``cpu=False``, so the query runs
  on the latent's device instead of on the CPU followed by a host-to-device copy.
  This is what ``noise_mode=brownian`` now does.

Why this tool exists: loading a 12 GB checkpoint makes an end-to-end A/B noisy
enough to hide a few hundred milliseconds, and a warm ComfyUI process can keep a
model resident that changes the numbers.  This script loads no model at all, so
the timings depend only on the noise implementation, the latent size and the
step count.

Read it as: ``ms/step`` is what the SDE adds on top of the model call for each
step where ``eta_i > 0``.  ``total`` is that cost for a whole 10-step run at the
given resolution (the number a user would notice).

    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\noise_cost.py
    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\noise_cost.py --check
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings

import torch

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
WORK = os.environ.get(
    "OMNI_WORK", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, COMFY)
sys.path.insert(0, WORK)

from omni_sampler import omni_core as core  # noqa: E402

# The last interval starts exactly at the tree's t0, and float32 rounding puts the
# query a hair below it; torchsde clamps and warns.  Expected, not a finding.
warnings.filterwarnings("ignore", message="Should have ta>=t0")

# Krea2 / MOSS / Anima all use a 16-channel latent at 1/8 spatial downscale.
CHANNELS = 16
RATIO = 8


def make_sigmas(steps, sigma_max=1.0, sigma_min=0.03):
    """A flow-model schedule (uniform in sigma); only the intervals matter here."""
    return [sigma_max + (sigma_min - sigma_max) * i / steps for i in range(steps + 1)]


def build(mode, x, sigmas, seed=7):
    positive = [s for s in sigmas if s > 0.0]
    if mode == "white":
        return core.default_noise_sampler(x, seed=seed)
    return core.BrownianTreeNoiseSampler(
        x, min(positive), max(positive), seed=seed, cpu=(mode == "brownian")
    )


def draw_all(sampler, sigmas, steps, device):
    """Exactly how ``sample_omni`` calls it: one draw per step interval.

    ``sample_omni`` moves ``sigmas`` to the latent's device first, and the two
    samplers read the device of those scalars, so passing CPU tensors here would
    measure a different code path (a device copy per step for the GPU tree).
    """
    out = None
    for i in range(steps):
        out = sampler(torch.tensor(sigmas[i], device=device),
                      torch.tensor(sigmas[i + 1], device=device))
    return out


def time_mode(mode, res, steps, device, repeats=3):
    hw = res // RATIO
    x = torch.zeros([1, CHANNELS, hw, hw], device=device)
    sigmas = make_sigmas(steps)
    times = []
    live = 0
    for _ in range(repeats):
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        sampler = build(mode, x, sigmas)
        build_t = time.perf_counter() - t0
        torch.cuda.synchronize() if device.type == "cuda" else None

        t0 = time.perf_counter()
        draw_all(sampler, sigmas, steps, device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.perf_counter() - t0)
        if device.type == "cuda":
            # live, not peak: the caching allocator keeps a previous mode's freed
            # blocks, so a process-wide peak reports the last big arm, not this one.
            live = torch.cuda.memory_allocated() - base
        del sampler
    return min(times), build_t, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="1024,1536,2048",
                    help="comma-separated pixel resolutions (default 1024,1536,2048)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cpu", action="store_true", help="run on CPU (smoke test only)")
    ap.add_argument("--check", action="store_true",
                    help="distribution check instead of timing")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu else "cuda")
    if not args.cpu:
        torch.cuda.init()

    if args.check:
        return check(device)

    print(f"device={device} steps={args.steps} latent={CHANNELS}ch @1/{RATIO}")
    print(f"{'res':>6} {'mode':>14} {'ms/step':>9} {'total':>8} {'build':>7} {'peak':>7}")

    for res in (int(r) for r in args.res.split(",")):
        for mode in ("white", "brownian", "brownian-gpu"):
            per_step, build_t, peak = time_mode(mode, res, args.steps, device)
            print(f"{res:>6} {mode:>14} {per_step / args.steps * 1e3:>9.1f} "
                  f"{per_step * 1e3:>7.0f}ms {build_t * 1e3:>6.1f}ms "
                  f"{peak / 2 ** 20:>6.0f}MB")
    return 0


def check(device):
    """Both implementations must return unit-variance, mutually uncorrelated draws.

    ``sample_omni`` queries one disjoint interval per step, so the only thing that
    matters statistically is that each draw is N(0, 1) and that draws for
    different steps are independent.  A Brownian path satisfies that by
    construction (non-overlapping increments of a Brownian motion are
    independent), which is why ``white`` is not a degradation -- only a different
    realisation of the same distribution.

    The moments are taken over draws AND elements (one pooled sample), not over
    per-draw means: the mean of 2000 unit draws has std 1/sqrt(2000), so averaging
    first would report 0.02 for any correct sampler.
    """
    steps, draws = 4, 400
    sigmas = make_sigmas(steps)
    shape = (1, 4, 8, 8)
    x = torch.zeros(shape, device=device)

    for mode in ("white", "brownian", "brownian-gpu"):
        acc = {"n": 0, "sum": 0.0, "sq": 0.0, "s0": 0.0, "s1": 0.0, "c01": 0.0,
               "sq0": 0.0, "sq1": 0.0}
        for seed in range(draws):
            sampler = build(mode, x, sigmas, seed=seed)
            z0 = sampler(torch.tensor(sigmas[0]), torch.tensor(sigmas[1])).double().cpu()
            z1 = sampler(torch.tensor(sigmas[1]), torch.tensor(sigmas[2])).double().cpu()
            acc["n"] += z0.numel() + z1.numel()
            acc["sum"] += z0.sum().item() + z1.sum().item()
            acc["sq"] += (z0 * z0).sum().item() + (z1 * z1).sum().item()
            acc["s0"] += z0.sum().item()
            acc["s1"] += z1.sum().item()
            acc["sq0"] += (z0 * z0).sum().item()
            acc["sq1"] += (z1 * z1).sum().item()
            acc["c01"] += (z0 * z1).sum().item()

        n = acc["n"]
        mean = acc["sum"] / n
        std = (acc["sq"] / n - mean ** 2) ** 0.5
        m0, m1 = acc["s0"] / (n / 2), acc["s1"] / (n / 2)
        v0 = acc["sq0"] / (n / 2) - m0 ** 2
        v1 = acc["sq1"] / (n / 2) - m1 ** 2
        corr = (acc["c01"] / (n / 2) - m0 * m1) / (v0 * v1) ** 0.5
        verdict = "ok" if abs(mean) < 0.02 and abs(std - 1.0) < 0.03 and abs(corr) < 0.03 else "BAD"
        print(f"{mode:>14}: mean={mean:+.4f} std={std:.4f} corr(step0,step1)={corr:+.4f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
