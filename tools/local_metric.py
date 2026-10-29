"""Mode-free accuracy metric: per-interval local ODE error.

With these models, comparing to a high-NFE reference measures which basin the
trajectory fell into, not integration accuracy (see CORRECTIONS.md).  Inside a
single interval [sigma_i, sigma_{i+1}] the basin is already decided, so the
sampler's step can be checked against a fine sub-solve from the same x_i.

Implementation avoids re-plumbing the guider: it re-runs the *same* sampler on a
refined sub-schedule starting from the captured x_i, and compares the denoised
(x0) predictions at the shared endpoint node.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

COMFY = r"G:\AI\comfy"
WORK = r"G:\AI\agent working folders\generic"
sys.path.insert(0, COMFY)
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "omni_sampler"))

import omni_core  # noqa: E402
import omni_sampler  # noqa: E402,F401
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
from omni_sampler.tools.modern_bench import FAMILIES, load  # noqa: E402

OUT = os.path.join(WORK, "omni_sampler", "outputs")
PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"
SEED = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--intervals", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=16)
    ap.add_argument("--variants", default="first,none")
    args = ap.parse_args()

    model, clip, vae, lf = load(args.family)
    flow = omni_core.is_flow_model(model.get_model_object("model_sampling"))
    cfg_scale = FAMILIES[args.family]["cfg"]
    hw = args.res // lf.spacial_downscale_ratio
    latent = (torch.zeros([1, lf.latent_channels, 1, hw, hw])
              if getattr(lf, "latent_dimensions", 2) == 3 else
              torch.zeros([1, lf.latent_channels, hw, hw]))
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    def sampler_for(corrector, order=2):
        return comfy.samplers.KSAMPLER(
            lambda m, x, s, **kw: omni_core.sample_omni(m, x, s, corrector=corrector,
                                                        order=order, **kw))

    def run(sigmas, x_start, sampler, record):
        s0 = float(sigmas[0])
        noise = (x_start / s0).detach()

        def cb(step, x0, x, total):
            record.append((x.detach().clone(), x0.detach().clone()))

        torch.manual_seed(SEED)
        with torch.no_grad():
            comfy.samplers.sample(
                model, noise, pos, neg, cfg_scale, model.load_device, sampler,
                torch.as_tensor(sigmas, device=model.load_device), model_options=model.model_options,
                latent_image=latent, callback=cb, disable_pbar=True, seed=SEED)

    base_sig = comfy.samplers.KSampler(
        model, steps=args.steps, device=model.load_device, sampler="euler",
        scheduler="simple", denoise=1.0, model_options=model.model_options).sigmas

    rows = []
    for variant in args.variants.split(","):
        rec = []
        run([float(s) for s in base_sig], latent, sampler_for(variant), rec)
        errs = []
        start = 0 if flow else 1   # eps max_denoise distorts a substep run at sigma_max
        for i in range(start, min(args.intervals, len(rec) - 1) + 1):
            s_i = float(base_sig[i])
            s_next = float(base_sig[i + 1])
            x_i = rec[i][0]
            l0 = omni_core.half_log_snr(omni_core.clamp_sigma(s_i, flow), flow)
            l1 = omni_core.half_log_snr(omni_core.clamp_sigma(s_next, flow), flow)
            sub = [omni_core.sigma_of_lambda(l0 + (l1 - l0) * k / args.substeps, flow)
                   for k in range(args.substeps + 1)]
            sub[0] = s_i
            fine_rec = []
            run(sub + [0.0], x_i, sampler_for("none", order=1), fine_rec)
            den_fine = fine_rec[args.substeps - 1][1]
            den_coarse = rec[i + 1][1]
            errs.append(float((den_coarse - den_fine).norm() / den_fine.norm()))
        mean = sum(errs) / len(errs) if errs else float("nan")
        rows.append(dict(variant=variant, mean_local=mean,
                         max_local=max(errs) if errs else None, per_interval=errs))
        print(f"  {variant:6s} intervals={len(errs)} mean={mean:.4e} "
              f"max={max(errs) if errs else float('nan'):.4e} " + " ".join(f"{e:.1e}" for e in errs))

    with open(os.path.join(OUT, f"localerr_{args.family}.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()

