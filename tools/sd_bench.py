"""Real-image comparison on an SDXL checkpoint (EPS family).

Runs the same seed/prompt through several samplers at a few step counts and
measures how close each result is to a heavily oversampled reference run of the
*same* model (the ODE solution).  That is the standard convergence proxy: it
does not prove aesthetic quality, but it measures integration error on a real
network instead of an analytic toy.

Usage (from the ComfyUI directory):
    venv\\Scripts\\python.exe "...\\tools\\sd_bench.py" [--ckpt PATH] [--steps 12,20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

COMFY = r"G:\AI\comfy"
PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, COMFY)
sys.path.insert(0, PARENT)

import omni_sampler  # noqa: E402,F401  registers the omni_* sampler names
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths
import comfy.model_management

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, sharp focus, 85mm photo"
NEG = "blurry, low quality, watermark, text"
CFG = 5.0
SEED = 42
RES = 1024
SCHED = "karras"


def ssim(a, b, win=11, sigma=1.5):
    """Gaussian-window SSIM on (B,C,H,W) tensors in [0,1]."""
    ch = a.shape[1]
    coords = torch.arange(win, dtype=torch.float32) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    k = (g.t() @ g).unsqueeze(0).unsqueeze(0).expand(ch, 1, win, win)
    pad = win // 2
    mu1 = torch.nn.functional.conv2d(a, k, padding=pad, groups=ch)
    mu2 = torch.nn.functional.conv2d(b, k, padding=pad, groups=ch)
    s11 = torch.nn.functional.conv2d(a * a, k, padding=pad, groups=ch) - mu1 ** 2
    s22 = torch.nn.functional.conv2d(b * b, k, padding=pad, groups=ch) - mu2 ** 2
    s12 = torch.nn.functional.conv2d(a * b, k, padding=pad, groups=ch) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", default="12,20")
    ap.add_argument("--ref-steps", type=int, default=80)
    ap.add_argument("--reso", type=int, default=RES)
    args = ap.parse_args()

    if args.ckpt is None:
        cands = []
        for root, _, files in os.walk(os.path.join(COMFY, "models", "checkpoints")):
            for f in files:
                if f.endswith((".safetensors", ".ckpt")) and "SUPIR" not in f:
                    p = os.path.join(root, f)
                    if os.path.getsize(p) > 3 * 1024 ** 3:
                        cands.append(p)
        args.ckpt = sorted(cands)[0]
    print(f"checkpoint: {args.ckpt}")

    t0 = time.time()
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        args.ckpt, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings")
    )
    print(f"loaded in {time.time() - t0:.1f}s")

    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    def run(sampler_name, steps, seed=SEED):
        torch.manual_seed(seed)
        noise = comfy.sample.prepare_noise(
            torch.zeros([1, 4, args.reso // 8, args.reso // 8]), seed
        )
        t = time.time()
        samples = comfy.sample.sample(
            model, noise, steps, CFG, sampler_name, SCHED, positive, negative,
            torch.zeros([1, 4, args.reso // 8, args.reso // 8]),
            seed=seed, disable_pbar=True,
        )
        dt = time.time() - t
        with torch.no_grad():
            img = vae.decode(samples.to(model.load_device))
        img = ((img + 1.0) / 2.0).clamp(0, 1).detach().cpu()[0].permute(2, 0, 1).unsqueeze(0)
        return img, dt

    ref, ref_dt = run("dpmpp_2m", args.ref_steps)
    print(f"reference ({args.ref_steps} steps dpmpp_2m) in {ref_dt:.1f}s")
    save_img(ref, f"reference_{args.ref_steps}step")

    rows = []
    names = ["euler", "dpmpp_2m", "res_multistep", "ipndm", "heun", "dpmpp_2m_sde", "omni_hybrid", "omni_hybrid_sde"]
    for steps in [int(s) for s in args.steps.split(",")]:
        for name in names:
            img, dt = run(name, steps)
            s = ssim(img, ref)
            l2 = float((img - ref).pow(2).mean().sqrt())
            rows.append({"sampler": name, "steps": steps, "ssim": s, "rmse": l2, "seconds": dt})
            save_img(img, f"{name}_{steps}step")
            print(f"  {name:16s} {steps:3d} steps  ssim={s:.4f}  rmse={l2:.4f}  {dt:6.1f}s")

    with open(os.path.join(OUT, "sd_bench.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nimages + json in {OUT}")


def save_img(t, name):
    arr = (t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, f"{name}.png"))


if __name__ == "__main__":
    main()

