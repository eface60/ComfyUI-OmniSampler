"""Report on a gap sweep: is the guidance gap a universal function of the run?

Reads outputs/gap_sweep/records.jsonl and answers three questions with numbers:

1. **Area response.**  For every family, rms_ratio / dc_ratio / p99_ratio as a
   power of latent area, and the *relative* response R(res) = value(res) /
   value(1024x1024).  If R is the same across families, the resolution law is
   model-independent and only one per-model scale factor is needed.
2. **Spectrum.**  The low-frequency energy profile at five bands, per family,
   and how far the profiles spread.
3. **Trajectory shape.**  rms_ratio per step, normalised by the run's own mean.
   If the normalised curves collapse, the *shape* of the gap over the trajectory
   is universal even when its magnitude is not.

    & "$env:COMFY_DIR\\venv\\Scripts\\python.exe" tools\\gap_report.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(REPO, "outputs", "gap_sweep", "records.jsonl")


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("failed") or not rec.get("summary"):
                continue
            rows.append(rec)
    return rows


def fit_power(points):
    """Least-squares slope and R^2 of log(y) = a + p*log(area)."""
    pts = [(math.log(a), math.log(y)) for a, y in points if a > 0 and y > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    if sxx <= 0:
        return None
    p = sxy / sxx
    a = my - p * mx
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    ss_res = sum((y - (a + p * x)) ** 2 for x, y in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return p, r2


def rel_spread(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return sd / mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=DEFAULT)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = load(args.records)
    if not rows:
        raise SystemExit(f"no usable records in {args.records}")

    # One record per (family, res, steps, seed); the LAST wins, so a re-run with
    # more fields supersedes an earlier one.
    cells = {}
    for r in rows:
        key = (r["family"], r["res"], r.get("steps"), r.get("seed"))
        cells[key] = r
    families = sorted({k[0] for k in cells})
    print(f"{len(rows)} records, {len(cells)} unique cells, families={families}")

    print("\n=== 1. area response (steps=%d, seed=%d) ===" % (args.steps, args.seed))
    print(f"{'family':8s} {'res':11s} {'area':>6s} {'rms':>7s} {'dc':>7s} "
          f"{'p99':>6s} {'lf02':>6s} {'lf05':>6s} {'par':>6s} {'sec':>6s}")
    per_family = {}
    for fam in families:
        pts_rms, pts_dc = [], []
        for (f, res, steps, seed), r in sorted(cells.items()):
            if f != fam or steps != args.steps or seed != args.seed:
                continue
            s = r["summary"]
            area = r["area"]
            per_family.setdefault(fam, {})[res] = s
            pts_rms.append((area, s.get("rms_ratio_raw") or 0.0))
            pts_dc.append((area, s.get("dc_ratio_raw") or 0.0))
            print(f"{fam:8s} {res:11s} {area:6.2f} "
                  f"{s.get('rms_ratio_raw') or 0:7.4f} {s.get('dc_ratio_raw') or 0:7.4f} "
                  f"{s.get('p99_ratio_raw') or 0:6.3f} {s.get('lf_02') or 0:6.3f} "
                  f"{s.get('lf_05') or 0:6.3f} {s.get('par_share') or 0:6.3f} "
                  f"{r.get('sec', 0):6.1f}")
        fit = fit_power(pts_rms)
        fit_dc = fit_power(pts_dc)
        if fit:
            line = f"    rms ~ area^{fit[0]:+.2f} (R2={fit[1]:.3f})"
            if fit_dc:
                line += f"   dc ~ area^{fit_dc[0]:+.2f} (R2={fit_dc[1]:.3f})"
            print(line)

    print("\n=== 2. relative response R(res) = value(res) / value(1024x1024) ===")
    ref = "1024x1024"
    print(f"{'res':11s} " + " ".join(f"{f:>10s}" for f in families) + "   spread(CV)")
    all_res = sorted({k[1] for k in cells}, key=lambda r: _area(r))
    for res in all_res:
        vals, cells_txt = [], []
        for fam in families:
            v = per_family.get(fam, {}).get(res, {}).get("rms_ratio_raw")
            base = per_family.get(fam, {}).get(ref, {}).get("rms_ratio_raw")
            if v is not None and base:
                rr = v / base
                vals.append(rr)
                cells_txt.append(f"{rr:10.3f}")
            else:
                cells_txt.append(f"{'-':>10s}")
        cv = rel_spread(vals)
        print(f"{res:11s} " + " ".join(cells_txt) +
              (f"   {cv:6.3f}" if cv is not None else ""))

    print("\n=== 3. spectrum at 1024x1024 (energy fraction below band) ===")
    print(f"{'family':8s} {'band 1%':>8s} {'2%':>6s} {'5%':>6s} {'10%':>6s} {'20%':>6s}")
    for fam in families:
        s = per_family.get(fam, {}).get(ref, {})
        print(f"{fam:8s} {s.get('lf_01', 0):8.3f} {s.get('lf_02', 0):6.3f} "
              f"{s.get('lf_05', 0):6.3f} {s.get('lf_10', 0):6.3f} {s.get('lf_20', 0):6.3f}")
    for band in ("lf_01", "lf_02", "lf_05", "lf_10", "lf_20"):
        cv = rel_spread([per_family.get(f, {}).get(ref, {}).get(band) for f in families])
        if cv is not None:
            print(f"    {band}: spread(CV)={cv:.3f}")

    print("\n=== 4. trajectory shape: rms_ratio per step / mean (1024x1024) ===")
    series = {}
    for fam in families:
        for (f, res, steps, seed), r in cells.items():
            if f == fam and res == ref and steps == args.steps and seed == args.seed:
                vals = [m.get("rms_ratio") for _, m in r.get("series", [])]
                vals = [v for v in vals if v is not None]
                if vals:
                    mean = sum(vals) / len(vals)
                    series[fam] = [v / mean for v in vals]
                break
    if series:
        n = min(len(v) for v in series.values())
        print("step      " + " ".join(f"{f:>8s}" for f in series))
        for i in range(n):
            print(f"{i:4d}      " + " ".join(f"{series[f][i]:8.3f}" for f in series))
        for i in range(n):
            cv = rel_spread([series[f][i] for f in series])
            if cv is not None:
                print(f"   step {i}: CV={cv:.3f}  "
                      f"range={min(series[f][i] for f in series):.2f}-"
                      f"{max(series[f][i] for f in series):.2f}")

    print("\n=== 5. step-count sensitivity (krea2, 1024x1024, all seeds) ===")
    for (f, res, steps, seed), r in sorted(cells.items(), key=lambda kv: (kv[0][2] or 0, kv[0][3])):
        if f == "krea2" and res == ref and seed == args.seed:
            s = r["summary"]
            print(f"    steps={steps:3d}  rms={s.get('rms_ratio_raw') or 0:.4f}  "
                  f"lf05={s.get('lf_05') or 0:.3f}  par={s.get('par_share') or 0:.3f}")


def _area(res):
    try:
        w, h = (int(v) for v in res.lower().split("x"))
        return w * h
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
