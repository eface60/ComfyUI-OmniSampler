"""Measure whether spectrally coloured SDE noise helps or hurts.

Same seed/prompt/steps, only the noise colour differs (beta = 0 white, 1 pink,
2 brown; unit variance in all cases). Metric: SSIM to the 80-step reference.
Theory: the SDE assumes white noise, so colouring should push the state off the
manifold the denoiser was trained on.
"""

from __future__ import annotations

import os
import sys

import torch
from PIL import Image
import numpy as np

COMFY = r"G:\AI\comfy"
PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, COMFY)
sys.path.insert(0, PARENT)

import omni_sampler  # noqa: E402,F401
import comfy.sample
import comfy.sd
import comfy.samplers
import folder_paths
from omni_sampler.tools.sd_bench import ssim

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, sharp focus, 85mm photo"
NEG = "blurry, low quality, watermark, text"
SEED, RES, STEPS = 42, 1024, 12


def main():
    ref_img = Image.open(os.path.join(OUT, "reference_80step.png")).convert("RGB")
    ref = torch.from_numpy(np.asarray(ref_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    ck = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(COMFY, "models", "checkpoints"))
          for f in fs if f.endswith(".safetensors") and "SUPIR" not in f
          and os.path.getsize(os.path.join(r, f)) > 3 * 1024 ** 3][0]
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        ck, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    latent = torch.zeros([1, 4, RES // 8, RES // 8])

    print(f"{'noise':10s} {'beta':>4s} {'ssim':>7s} {'hf_ratio':>9s}")
    for name, beta in (("white", 0.0), ("pink", 1.0), ("brown", 2.0)):
        sampler = comfy.samplers.ksampler(
            "omni_hybrid", {"eta": 1.0, "noise_mode": "white", "noise_color": beta, "corrector": "none"}
        )
        torch.manual_seed(SEED)
        noise = comfy.sample.prepare_noise(latent, SEED)
        ks = comfy.samplers.KSampler(model, steps=STEPS, device=model.load_device, sampler="euler",
                                     scheduler="karras", denoise=1.0, model_options=model.model_options)
        with torch.no_grad():
            samples = comfy.samplers.sample(
                model, noise, pos, neg, 5.0, model.load_device, sampler, ks.sigmas,
                model_options=model.model_options, latent_image=latent, disable_pbar=True, seed=SEED)
            img = vae.decode(samples.to(model.load_device))
        img = ((img + 1.0) / 2.0).clamp(0, 1).detach().cpu()[0]
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        t = img.permute(2, 0, 1).unsqueeze(0)
        Image.fromarray((img.numpy() * 255).astype(np.uint8)).save(os.path.join(OUT, f"noise_{name}.png"))
        # crude high-frequency energy ratio: |laplacian| / |image|
        lap = t[:, :, 1:-1, 1:-1] * 4 - t[:, :, :-2, 1:-1] - t[:, :, 2:, 1:-1] - t[:, :, 1:-1, :-2] - t[:, :, 1:-1, 2:]
        hf = float(lap.abs().mean() / t.abs().mean())
        print(f"{name:10s} {beta:4.1f} {ssim(t, ref):7.4f} {hf:9.4f}")


if __name__ == "__main__":
    main()
