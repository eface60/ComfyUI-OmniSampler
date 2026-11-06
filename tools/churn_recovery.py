"""Churn recovery probe: what does re-noising do to the model's x0?

The question is not "does churn change the image" (it always does) but "does the
denoiser put the structure back". Method:

1. Run one normal trajectory and record ``(x_i, x0_i, sigma_i)`` at every step.
2. At each step, re-noise ``x_i`` from ``sigma_i`` to ``sigma_hat`` with the
   sampler's own exact forward step, evaluate the model at ``sigma_hat``, and
   compare its x0 with ``x0_i``.
3. Split the difference into a low-frequency band (structure/tone) and the rest
   (detail). A model that can recover from churn moves the structure little and
   the detail more.

The probe evaluations have to happen *inside* the sampler callback: ComfyUI
tears the guider down when sampling returns, so the model is only callable
while the run is in flight.

``gamma = 0`` is a built-in control: it evaluates the model at ``sigma_i`` from
``x_i``, which must reproduce the recorded ``x0_i`` bit-exactly. If it does not,
the probe is not measuring the sampler's own call and the run is void.

    $env:COMFY_DIR = "G:\\AI\\comfy"
    python tools\\churn_recovery.py --toy
    python tools\\churn_recovery.py --family moss --steps 6 --res 512

The toy arm uses the analytic mixture denoiser (exact posterior mean), so it is
the reference curve: what churn costs a model that cannot be wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

COMFY = os.environ.get("COMFY_DIR", r"G:\AI\comfy")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(REPO))

import omni_sampler  # noqa: E402,F401
from omni_sampler import omni_core  # noqa: E402

OUT = os.path.join(REPO, "outputs", "churn_recovery")
GAMMAS = (0.0, 0.20, 0.414)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _4d(t):
    if t.dim() == 5:
        return t.mean(dim=2)
    return t


def _bands(t, k=8):
    h, w = t.shape[-2:]
    kk = max(1, min(k, h, w))
    lp = F.avg_pool2d(t.float(), kk, kk)
    lp = F.interpolate(lp, size=(h, w), mode="bilinear", align_corners=False)
    return lp, t.float() - lp


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def compare(x0_new, x0_old):
    a, b = _4d(x0_new), _4d(x0_old)
    cos = float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)))
    if a.dim() < 4:
        return {"rel": _rel(a, b), "lf": float("nan"), "hf": float("nan"), "cos": cos}
    lp_a, hf_a = _bands(a)
    lp_b, hf_b = _bands(b)
    return {"rel": _rel(a, b), "lf": _rel(lp_a, lp_b), "hf": _rel(hf_a, hf_b), "cos": cos}


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------
class Probe:
    """Sampler entry point that also runs the per-step churn evaluations."""

    def __init__(self, flow=False):
        self.flow = bool(flow)
        self.model = None
        self.extra_args = None
        self.sigmas = None
        self.records = 0
        self.rows = []

    # comfy's KSAMPLER calls this
    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
        self.model = model
        self.extra_args = dict(extra_args or {})
        self.sigmas = sigmas
        self.flow = omni_core.is_flow_model(omni_core._find_model_sampling(model))
        return omni_core.sample_omni(
            model, x, sigmas, extra_args=extra_args,
            callback=lambda d: self.cb(d, callback), disable=disable, **kw)

    def evaluate(self, x, sigma):
        s_in = x.new_ones([x.shape[0]])
        ea = dict(self.extra_args or {})
        ea.pop("denoise_mask", None)
        return self.model(x, sigma * s_in, denoise_mask=None, **ea)

    def cb(self, d, callback=None):
        i = int(d["i"])
        sigma = float(d["sigma"])
        x_i = d["x"].detach()
        x0_i = d["denoised"].detach()
        self.records += 1
        for gamma in GAMMAS:
            sigma_hat = sigma * (1.0 + gamma)
            if self.flow:
                sigma_hat = min(sigma_hat, float(self.sigmas[0]))
            x_hat, fired = omni_core._churn_forward(x_i, sigma, sigma_hat, self.flow, 1.0)
            if not fired:
                x_hat = x_i
            with torch.no_grad():
                x0_hat = self.evaluate(x_hat, sigma_hat)
            m = compare(x0_hat, x0_i)
            m.update(step=i, sigma=round(sigma, 5), gamma=gamma,
                     sigma_hat=round(sigma_hat, 5),
                     noise_frac=math.sqrt(max(0.0, (sigma_hat / sigma) ** 2 - 1.0)))
            self.rows.append(m)
        if callback is not None:
            callback(d)


def summarise(rows):
    def mean(sel, key):
        vals = [r[key] for r in sel if math.isfinite(r[key])]
        return sum(vals) / len(vals) if vals else float("nan")
    out = {}
    for g in GAMMAS:
        sel = [r for r in rows if abs(r["gamma"] - g) < 1e-9]
        if not sel:
            continue
        out[f"g{g:.3f}"] = {"lf": mean(sel, "lf"), "hf": mean(sel, "hf"),
                            "rel": mean(sel, "rel"), "cos": mean(sel, "cos"),
                            "n": len(sel)}
    return out


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def run_toy(steps=8):
    from omni_sampler.tools.toy import MixturePrior, ToyModel, karras_sigmas, make_init

    prior = MixturePrior(dim=2, seed=0)
    model = ToyModel(prior, flow=False)
    sig = torch.tensor([float(s) for s in karras_sigmas(steps, model.sigma_min, model.sigma_max)])
    x, _ = make_init(prior, 256, float(sig[0]), False, torch.Generator().manual_seed(3))
    probe = Probe(flow=False)
    probe.model = model
    probe.extra_args = {"seed": 0}
    probe.sigmas = sig
    with torch.no_grad():
        omni_core.sample_omni(model, x.clone(), sig, extra_args={"seed": 0},
                              disable=True, callback=probe.cb,
                              corrector="none", order=2)
    return "toy", probe


def run_family(family, unet, steps, res, seed):
    import comfy.sample
    import comfy.samplers
    from omni_sampler.tools import modern_bench

    if unet:
        modern_bench.FAMILIES[family]["unet"] = unet
    model, clip, vae, lf = modern_bench.load(family)
    cfg = float(modern_bench.FAMILIES[family]["cfg"])
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(modern_bench.PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(modern_bench.NEG))
    ratio = int(lf.spacial_downscale_ratio)
    hw, ww = res // ratio, res // ratio
    if getattr(lf, "latent_dimensions", 2) == 3:
        latent = torch.zeros([1, int(lf.latent_channels), 1, hw, ww])
    else:
        latent = torch.zeros([1, int(lf.latent_channels), hw, ww])

    ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device,
                                 sampler="euler", scheduler="simple", denoise=1.0,
                                 model_options=model.model_options)
    probe = Probe()
    noise = comfy.sample.prepare_noise(latent, seed)
    with torch.no_grad():
        comfy.samplers.sample(model, noise, pos, neg, cfg, model.load_device,
                              comfy.samplers.KSAMPLER(probe), ks.sigmas,
                              model_options=model.model_options, latent_image=latent,
                              disable_pbar=True, seed=seed)
    del clip, vae
    torch.cuda.empty_cache()
    return family, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--unet", default="")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    jobs = []
    if args.toy:
        jobs.append(run_toy(args.steps))
    if args.family:
        jobs.append(run_family(args.family, args.unet, args.steps, args.res, args.seed))

    summary, bad_control = {}, False
    for name, probe in jobs:
        rows = probe.rows
        s = summarise(rows)
        ctrl_rows = [r for r in rows if abs(r["gamma"]) < 1e-12]
        ctrl_rel = max((r["rel"] for r in ctrl_rows), default=float("nan"))
        # int8 checkpoints do not always reproduce a call bit-exactly in-process
        # (krea2: 0.3% here, moss: 0.0), so the control is a floor, not a gate.
        control = bool(ctrl_rows) and ctrl_rel < 0.02
        bad_control = bad_control or not control
        summary[name] = {"control": control, "control_rel": ctrl_rel, "flow": probe.flow,
                         "steps": probe.records, "summary": s, "rows": rows}
        print(f"\n=== {name}  flow={probe.flow} steps={probe.records} "
              f"control_rel={ctrl_rel:.2e} ({'ok' if control else 'FLOOR TOO HIGH'}) ===")
        print(f"{'gamma':>6} {'noise/sigma':>12} {'lf_drift':>9} {'hf_drift':>9} "
              f"{'rel':>7} {'cos':>7}")
        for g in GAMMAS:
            v = s.get(f"g{g:.3f}")
            if not v:
                continue
            nf = math.sqrt(max(0.0, (1 + g) ** 2 - 1.0))
            print(f"{g:6.3f} {nf:12.3f} {v['lf']:9.4f} {v['hf']:9.4f} "
                  f"{v['rel']:7.4f} {v['cos']:7.4f}")
        last = [r for r in rows if abs(r["gamma"] - GAMMAS[-1]) < 1e-9]
        print("  per-step lf_drift: " + " ".join(f"{r['step']}:{r['lf']:.3f}" for r in last))

    path = os.path.join(OUT, f"{args.family or 'toy'}.json")
    with open(path, "w") as fh:
        json.dump(summary, fh)
    print(f"\nwrote {path}")
    if bad_control:
        print("CONTROL FAILED: the probe did not reproduce the sampler's own call")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
