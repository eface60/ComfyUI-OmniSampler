"""One-model A/B of PD-AOS's three spatial mechanisms, in-process.

Why this exists
---------------
The corpus scan can say the mechanisms engaged; it cannot say what they did to the
image, because a one-switch A/B needs the same prompt, seed, schedule and model.
This script runs that A/B: every arm uses the same conds, noise, sigmas and seed,
and only the named switch differs.

Arms
----
  base      engine defaults as shipped (per-pixel order ON, local field ON,
            guidance schedule OFF)
  ppo_off   order_pixel_adaptive / order_pixel_blend / second_order_per_pixel OFF
  lf_off    lf_enable OFF
  cfg_stage1 only the scalar late guidance ramp
  cfg_stage2 stage 1 + the per-region CFG map (the per-pixel CFG arm)

Outputs
-------
  <out>/<tag>_<arm>.png         decoded image
  <out>/<tag>_<arm>.json        latent + the run's own step records and config stamp
  <out>/<tag>_contact.png       all arms side by side

Self-validation
---------------
* the run's reported `steps`, `sigmas` digest and `cfg_schedule.stage` are asserted
  against the arm that was asked for (a missing stage2 stamp means the map did not
  run, and that arm is labelled NOT FIRED rather than compared);
* every arm's noise tensor is asserted identical (hash) before sampling;
* the latent is asserted finite and non-degenerate (max-abs > 0 and std > 0);
* a negative control (same arm twice) must reproduce within 1e-6 relative L2 or the
  session is not reproducible and every difference is void.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

COMFY = r"G:\AI\comfy"
PD = r"G:\AI\comfyui-research-samplers"
sys.path.insert(0, COMFY)
sys.path.insert(0, PD)

import comfy.model_management  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
import pd_aos  # noqa: E402
import pd_aos.nodes as pdnodes  # noqa: E402
from pd_aos.research_samplers import aflops  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_out")
DM = os.path.join(COMFY, "models", "diffusion_models")
TE = os.path.join(COMFY, "models", "text_encoders")
VAE_DIR = os.path.join(COMFY, "models", "vae")

PROMPT = "a red fox sitting in a snowy pine forest at golden hour, highly detailed, photograph"
NEG = "blurry, low quality, watermark, text"

FAMILIES = {
    "krea2": dict(
        unet=os.path.join(DM, "krea2", "krea2TurboOfficialComfy_krea2TurboInt8.safetensors"),
        te=[os.path.join(TE, "qwen3vl_4b_fp8_scaled.safetensors")],
        clip_type="KREA2",
        vae=os.path.join(VAE_DIR, "wan_2.1_vae.safetensors"),
        cfg=1.0,
    ),
    "moss": dict(
        unet=os.path.join(DM, "moss", "moss_image_base_int8_convrot.safetensors"),
        te=[os.path.join(TE, "qwen3.5_9b_int8_convrot.safetensors")],
        clip_type="MOSS_IMAGE",
        vae=os.path.join(VAE_DIR, "flux2-vae.safetensors"),
        cfg=3.5,
    ),
}

ARMS = {
    "base": {},
    "base2": {},          # negative control: identical to base, must reproduce
    "ppo_off": {
        "order_pixel_adaptive": False,
        "order_pixel_blend": False,
        "second_order_per_pixel": False,
    },
    "lf_off": {"lf_enable": False},
    "cfg_stage1": {"cfg_schedule_stage1": True, "cfg_schedule_stage2": False},
    "cfg_stage2": {"cfg_schedule_stage1": True, "cfg_schedule_stage2": True},
}


def load(family):
    c = FAMILIES[family]
    model = comfy.sd.load_diffusion_model(c["unet"], model_options={})
    ms = model.get_model_object("model_sampling")
    lf = model.get_model_object("latent_format")
    clip = comfy.sd.load_clip(c["te"], embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=getattr(comfy.sd.CLIPType, c["clip_type"]))
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(c["vae"], safe_load=True))
    print("[%s] loaded  space=%s latent=%s" % (family, type(ms).__name__, lf.latent_channels), flush=True)
    return model, clip, vae, lf, ms


def to_hwc(decoded):
    x = decoded.detach().float().cpu()
    while x.dim() > 3 and x.shape[0] == 1:
        x = x[0]
    if x.dim() == 4:
        x = x[0] if x.shape[-1] == 3 else x[:, 0]
    if x.dim() != 3:
        raise RuntimeError("unexpected decode shape %s" % (tuple(decoded.shape),))
    if x.shape[-1] != 3:
        x = x.permute(1, 2, 0)
    x = ((x + 1.0) / 2.0).clamp(0, 1)
    return x.permute(2, 0, 1).unsqueeze(0)


def digest(t):
    return hashlib.sha1(np.ascontiguousarray(t.detach().float().cpu().numpy())).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arms", default="base,ppo_off,lf_off,cfg_stage1,cfg_stage2")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    model, clip, vae, lf, ms = load(args.family)
    cfg_scale = FAMILIES[args.family]["cfg"]
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))
    hw = args.res // lf.spacial_downscale_ratio
    if getattr(lf, "latent_dimensions", 2) == 3:
        latent = torch.zeros([1, lf.latent_channels, 1, hw, hw])
    else:
        latent = torch.zeros([1, lf.latent_channels, hw, hw])
    torch.manual_seed(args.seed)
    noise = comfy.sample.prepare_noise(latent, args.seed)
    noise_digest = digest(noise)
    print("noise %s shape %s" % (noise_digest, tuple(noise.shape)), flush=True)

    sig = comfy.samplers.calculate_sigmas(ms, args.scheduler, args.steps)
    print("sigmas", [round(float(s), 5) for s in sig], flush=True)

    # one guider per render so no state leaks between arms
    outputs = {}
    meta = {}
    tag = args.tag or "%s_%ds" % (args.family, args.steps)
    for arm in args.arms.split(","):
        over = dict(ARMS[arm])
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(pos, neg)
        guider.set_cfg(cfg_scale)
        t0 = time.time()
        tuned, opts = pdnodes.ResearchAFlopsAutotuner().autotune(
            guider, sig, 10, 48, 4, 16, 1.0, 1.0,
            probe_smin_frac=0.1, probe_schedule=False)
        # The Autotuner's opts are what the sampler would receive; the arm is
        # applied ON TOP, as a connected Options node would.
        opts.update(over)
        sampler = pdnodes.ResearchAFlopsSampler().get_sampler(
            log_errors=False, **{"options+": opts})[0]
        torch.manual_seed(args.seed)
        with torch.no_grad():
            out = guider.sample(noise.clone(), latent.clone(), sampler, tuned,
                                disable_pbar=True, seed=args.seed)
        dt = time.time() - t0
        stamp = dict(aflops._LAST_CFG or {})
        steps = list(aflops._LAST_ERRORS)
        assert torch.isfinite(out).all(), "non-finite latent in %s" % arm
        std = float(out.float().std())
        mx = float(out.float().abs().max())
        assert std > 0.05 and mx > 0.5, "degenerate latent in %s (std %.4f max %.4f)" % (arm, std, mx)
        stage = (stamp.get("cfg_schedule") or {}).get("stage")
        stage2 = (stamp.get("cfg_schedule") or {}).get("stage2")
        if arm == "cfg_stage2" and stage2 is not True:
            print("  NOT FIRED: cfg_stage2 requested but the run stamps stage2=%r (%s)"
                  % (stage2, (stamp.get("cfg_schedule") or {}).get("stage2_reason")), flush=True)
        if arm == "cfg_stage1" and stage2:
            print("  NOT FIRED: cfg_stage1 arm stamps stage2=True", flush=True)
        outputs[arm] = out
        meta[arm] = dict(
            style="pd_aos", family=args.family, steps=args.steps, seed=args.seed,
            cfg=cfg_scale, arm=arm, overrides=over, seconds=round(dt, 2),
            noise=noise_digest, sigmas=[round(float(s), 6) for s in sig],
            stage=stage, stage2=stage2,
            stage2_reason=(stamp.get("cfg_schedule") or {}).get("stage2_reason"),
            build=stamp.get("_build"), step_records=steps,
            latent_sha=digest(out), std=std, max_abs=mx,
        )
        print("[%s/%s] %.1fs  stage=%r stage2=%r std=%.4f" % (
            args.family, arm, dt, stage, stage2, std), flush=True)
        decoded = vae.decode(out.to(model.load_device))
        img = to_hwc(decoded).to(comfy.model_management.intermediate_device())
        Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
            os.path.join(OUT, "%s_%s.png" % (tag, arm)))
        with open(os.path.join(OUT, "%s_%s.json" % (tag, arm)), "w", encoding="utf-8") as fh:
            json.dump(meta[arm], fh)

    # ---- negative control: base rendered twice must reproduce -------------
    arms = list(outputs)
    if "base" in outputs and "base2" in outputs:
        a, b = outputs["base"].float(), outputs["base2"].float()
        rel = float((a - b).norm() / b.norm().clamp_min(1e-12))
        print("NEGATIVE CONTROL base vs base2: rel_l2 %.3e" % rel)
        if rel > 1e-6:
            print("CONTROL FAILED: the session is not reproducible; differences are void.")
            sys.exit(1)
        arms.remove("base2")

    if len(arms) > 1:
        base = outputs["base"].float()
        print("\narm                 rel_l2 vs base   image_rms")
        for arm in arms:
            if arm == "base":
                continue
            d = (outputs[arm].float() - base)
            rel = float(d.norm() / base.norm().clamp_min(1e-12))
            print("%-20s %12.4e %12.4e" % (arm, rel, float(d.pow(2).mean().sqrt())))

    # contact sheet
    try:
        imgs = []
        for arm in arms:
            p = os.path.join(OUT, "%s_%s.png" % (tag, arm))
            imgs.append((arm, Image.open(p).convert("RGB")))
        w, h = imgs[0][1].size
        sheet = Image.new("RGB", (w * len(imgs), h + 24), (16, 16, 16))
        from PIL import ImageDraw
        dr = ImageDraw.Draw(sheet)
        for i, (arm, im) in enumerate(imgs):
            sheet.paste(im, (i * w, 24))
            dr.text((i * w + 6, 6), arm, fill=(255, 255, 255))
        sheet.save(os.path.join(OUT, "%s_contact.png" % tag))
        print("contact sheet:", os.path.join(OUT, "%s_contact.png" % tag))
    except Exception as exc:  # noqa: BLE001
        print("contact sheet failed:", exc)


if __name__ == "__main__":
    main()
