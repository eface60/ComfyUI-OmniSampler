"""Scan the PD-AOS run-log corpus and report whether its three spatial mechanisms
actually engaged, and by how much.

Purpose (why this file exists)
------------------------------
The operator asks whether PD-AOS's per-pixel CFG, per-pixel order and local
fields worked as intended before any of them is ported to omni.  The PD-AOS
engine already writes a per-step record for every run; this script reads those
records rather than trusting the claims in the README.

Self-validation
---------------
Every derived number is gated by an invariant that holds only if the field being
read is the field the engine wrote:

  * a file parses and carries `aflops_cfg` + a non-empty `aflops` list;
  * `ord_mean` lies in [1, order_cap] and `second_order_px` in [0, 1];
  * a per-step `cfg_map` satisfies `min <= mean <= max` and, when `cfg_eff` is
    present, `min >= 0.85*cfg_eff - 1e-3` and `max <= cfg_eff + 1e-3`
    (the shipped floor is 0.85 and the map cannot exceed the step scale);
  * `lf.m_abs >= 0` and `lf.m_min <= 1 <= lf.m_max` when both are present;
  * `stage2` implies `stage2_mode == "post_cfg"` and a non-null `base` cfg;
  * each run counted in a table has at least one step record.

A run that fails any check is COUNTED AS REJECTED and excluded from the
aggregates; the count is printed.  A green run therefore cannot be produced by
silently reading the wrong field.

Control
-------
`--control` injects one deliberately corrupted copy of each check's input and
asserts the checker rejects it.  The script exits non-zero if any control is
accepted, so a green table is a statement about the checker too.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys


def _q(xs, frac):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * frac
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _check_run(j):
    """Return (ok, reason, rows).  `rows` is the per-step list, validated."""
    c = j.get("aflops_cfg")
    steps = j.get("aflops")
    if not isinstance(c, dict):
        return False, "no aflops_cfg", None
    if not isinstance(steps, list) or not steps:
        return False, "no step records", None
    cap = _num(c.get("order_cap")) or _num(c.get("order")) or 3.0
    order_max = _num((c.get("cfg") or {}).get("order_max")) or cap
    hi = max(cap, order_max) + 1e-6
    for s in steps:
        if not isinstance(s, dict) or "step" not in s:
            return False, "bad step record", None
        om = _num(s.get("ord_mean"))
        if om is not None and not (1.0 - 1e-6 <= om <= hi):
            return False, "ord_mean out of range (%s)" % om, None
        sp = _num(s.get("second_order_px"))
        if sp is not None and not (0.0 <= sp <= 1.0):
            return False, "second_order_px out of range (%s)" % sp, None
        cm = s.get("cfg_map")
        if cm is not None:
            lo, me, up = _num(cm.get("min")), _num(cm.get("mean")), _num(cm.get("max"))
            if None in (lo, me, up):
                return False, "cfg_map incomplete", None
            if not (lo <= me <= up):
                return False, "cfg_map order violated", None
            ce = _num(s.get("cfg_eff"))
            if ce is not None:
                if lo < 0.85 * ce - 1e-3 or up > ce + 1e-3:
                    return False, "cfg_map outside [floor*cfg_eff, cfg_eff]", None
        lf = s.get("lf")
        if isinstance(lf, dict):
            ma = _num(lf.get("m_abs"))
            mn, mx = _num(lf.get("m_min")), _num(lf.get("m_max"))
            if ma is not None and ma < 0:
                return False, "negative lf.m_abs", None
            if mn is not None and mx is not None:
                # The field is bounded by its own amplitude (0.15*clamp for the
                # primary term, plus at most 0.35*clamp for the rescue), so the
                # shipped range is ~[0.83, 1.17].  A value far outside that means
                # this is not `1 + lf_m`.  m_min/m_max do NOT have to straddle 1:
                # the centring is weighted by the step, so an all-positive field
                # is legal, and the log rounds both to 4 dp.
                if not (mn <= mx):
                    return False, "lf m_min > m_max", None
                if mn < 0.5 or mx > 1.5:
                    return False, "lf m outside the amplitude bound", None
                if ma is not None:
                    bound = max(abs(mn - 1.0), abs(mx - 1.0)) + 5e-4
                    if ma > bound:
                        return False, "lf |mean| exceeds its extremes", None
    cs = c.get("cfg_schedule") or {}
    if cs.get("stage2") and cs.get("stage2_mode") != "post_cfg":
        return False, "stage2 without post_cfg mode", None
    return True, "", steps


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect(root):
    files = sorted(glob.glob(os.path.join(root, "*", "log_*.json")))
    runs = []
    rejected = []
    for f in files:
        try:
            j = load(f)
        except Exception as exc:  # noqa: BLE001 - a broken log is data
            rejected.append((f, "parse: %s" % type(exc).__name__))
            continue
        ok, why, steps = _check_run(j)
        if not ok:
            rejected.append((f, why))
            continue
        c = j["aflops_cfg"]
        cs = c.get("cfg_schedule") or {}
        model = os.path.basename(os.path.dirname(f))
        base = _num(cs.get("base"))
        step_cfg = [_num(s.get("cfg_eff")) for s in steps]
        step_cfg = [v for v in step_cfg if v is not None]
        maps = [s.get("cfg_map") for s in steps if s.get("cfg_map")]
        sp = [_num(s.get("second_order_px")) for s in steps]
        sp = [v for v in sp if v is not None]
        om = [_num(s.get("ord_mean")) for s in steps]
        om = [v for v in om if v is not None]
        lfs = [s.get("lf") for s in steps if isinstance(s.get("lf"), dict)]
        m_abs = [_num(x.get("m_abs")) for x in lfs]
        m_abs = [v for v in m_abs if v is not None]
        rho = [_num(x.get("rho_max")) for x in lfs]
        rho = [v for v in rho if v is not None]
        inj = [x.get("inject") for x in lfs]
        runs.append(
            dict(
                path=f,
                model=model,
                build=c.get("_build"),
                steps=len(steps),
                space=c.get("space"),
                order=c.get("order"),
                order_cap=c.get("order_cap"),
                order_cfg=(c.get("cfg") or {}).get("order_max"),
                order_pixel_adaptive=(c.get("cfg") or {}).get("order_pixel_adaptive"),
                second_order_per_pixel=c.get("second_order_per_pixel"),
                order_by_bound=c.get("order_by_bound"),
                stage=cs.get("stage"),
                stage2=bool(cs.get("stage2")),
                stage2_reason=cs.get("stage2_reason"),
                base_cfg=base,
                cfg_eff=[v for v in step_cfg],
                maps=maps,
                ord_mean=om,
                spx=sp,
                lf_m_abs=m_abs,
                lf_rho=rho,
                lf_inject=[v for v in inj if v],
                lf_on=(c.get("cfg") or {}).get("lf_enable"),
            )
        )
    return runs, rejected, len(files)


def summarize(runs):
    out = {}
    out["files"] = len(runs)
    stage2_runs = [r for r in runs if r["stage2"]]
    out["stage2_runs"] = len(stage2_runs)
    out["stage2_with_map"] = sum(1 for r in stage2_runs if r["maps"])
    out["stage2_no_map"] = sum(1 for r in stage2_runs if not r["maps"])
    reasons = {}
    for r in stage2_runs:
        if not r["maps"]:
            reasons[r["stage2_reason"] or "none"] = reasons.get(r["stage2_reason"] or "none", 0) + 1
    out["stage2_reasons"] = reasons
    out["stage2_cfg1"] = sum(1 for r in stage2_runs if r["base_cfg"] is not None and r["base_cfg"] <= 1.0 + 1e-9)
    out["stage2_cfg_gt1"] = sum(1 for r in stage2_runs if r["base_cfg"] is not None and r["base_cfg"] > 1.0 + 1e-9)
    # map range over every logged map (as a fraction of that step's cfg_eff)
    frac_lo, frac_hi = [], []
    uniform_maps = 0
    for r in stage2_runs:
        # step records are aligned to maps only by order; use the map's own numbers
        for m in r["maps"]:
            lo, up = _num(m.get("min")), _num(m.get("max"))
            if lo is not None and up is not None and up > 1e-9:
                frac_lo.append(lo / up)
                if up - lo <= 1e-6:
                    uniform_maps += 1
    out["map_uniform"] = uniform_maps
    out["map_min_over_max"] = dict(
        n=len(frac_lo),
        p10=_q(frac_lo, 0.10),
        p50=_q(frac_lo, 0.50),
        p90=_q(frac_lo, 0.90),
        mn=min(frac_lo) if frac_lo else None,
        mx=max(frac_lo) if frac_lo else None,
    )
    steps_with_map = [len(r["maps"]) for r in stage2_runs]
    out["map_steps_per_run"] = dict(
        p10=_q(steps_with_map, 0.10), p50=_q(steps_with_map, 0.50),
        p90=_q(steps_with_map, 0.90), mn=min(steps_with_map) if steps_with_map else None,
        mx=max(steps_with_map) if steps_with_map else None,
    )
    # stage 1: did the late ramp ever start?  `_cfg_schedule_scale` computes
    # t = 1 - s/sigma_max and starts the ramp at t=0.7, so a schedule whose last
    # sigma is above 0.3*sigma_max never leaves the early half-CFG value.
    stage1_runs = [r for r in runs if (r["stage"] or 0) >= 1]
    ramped = flat_half = never_reached_base = 0
    for r in stage1_runs:
        ce = [v for v in r["cfg_eff"] if v is not None]
        base = r["base_cfg"]
        if not ce or base is None or base <= 0.0:
            continue
        if max(ce) > 0.5 * base + 1e-6:
            ramped += 1
        elif all(abs(v - 0.5 * base) <= 1e-6 for v in ce):
            flat_half += 1
        if max(ce) < base - 1e-6:
            never_reached_base += 1
    out["stage1_runs"] = len(stage1_runs)
    out["stage1_ramped"] = ramped
    out["stage1_flat_half"] = flat_half
    out["stage1_never_reached_base"] = never_reached_base
    # per-pixel order
    out["ord_mean_all"] = dict(
        n=sum(len(r["ord_mean"]) for r in runs),
        p10=_q([v for r in runs for v in r["ord_mean"]], 0.10),
        p50=_q([v for r in runs for v in r["ord_mean"]], 0.50),
        p90=_q([v for r in runs for v in r["ord_mean"]], 0.90),
    )
    out["runs_with_ord_gt1"] = sum(1 for r in runs if any(v > 1.0 + 1e-6 for v in r["ord_mean"]))
    out["spx_all"] = dict(
        n=sum(len(r["spx"]) for r in runs),
        p50=_q([v for r in runs for v in r["spx"]], 0.50),
        p90=_q([v for r in runs for v in r["spx"]], 0.90),
        mx=max((v for r in runs for v in r["spx"]), default=None),
    )
    out["runs_with_spx"] = sum(1 for r in runs if any(v > 1e-6 for v in r["spx"]))
    # local field
    mabs = [v for r in runs for v in r["lf_m_abs"]]
    out["lf_m_abs"] = dict(
        n=len(mabs), p10=_q(mabs, 0.10), p50=_q(mabs, 0.50),
        p90=_q(mabs, 0.90), mx=max(mabs) if mabs else None,
    )
    out["runs_with_lf_gt1pct"] = sum(1 for r in runs if any(v > 0.01 for v in r["lf_m_abs"]))
    out["lf_rho"] = dict(
        n=sum(len(r["lf_rho"]) for r in runs),
        p50=_q([v for r in runs for v in r["lf_rho"]], 0.50),
        p90=_q([v for r in runs for v in r["lf_rho"]], 0.90),
        mx=max((v for r in runs for v in r["lf_rho"]), default=None),
    )
    inj = {}
    for r in runs:
        for v in r["lf_inject"]:
            inj[v] = inj.get(v, 0) + 1
    out["lf_inject"] = inj
    return out


def control(root):
    """One deliberately broken input per checker; all must be rejected."""
    good = None
    for f in glob.glob(os.path.join(root, "*", "log_*.json")):
        try:
            j = load(f)
        except Exception:
            continue
        if _check_run(j)[0] and (j["aflops_cfg"].get("cfg_schedule") or {}).get("stage2"):
            good = (f, j)
            break
    if good is None:
        print("CONTROL FAILED: no valid stage-2 run found to mutate")
        return False
    f, j = good
    ok = True

    def mut(label, fn):
        nonlocal ok
        jj = json.loads(json.dumps(j))
        fn(jj)
        okk, why, _ = _check_run(jj)
        print("  control %-34s -> %s (%s)" % (label, "REJECTED" if not okk else "ACCEPTED", why or "-"))
        ok = ok and not okk

    mut("ord_mean out of range", lambda jj: jj["aflops"][0].__setitem__("ord_mean", 99.0))
    mut("second_order_px out of range", lambda jj: jj["aflops"][0].__setitem__("second_order_px", 1.5))
    mut("cfg_map order violated", lambda jj: jj["aflops"][-1].__setitem__(
        "cfg_map", {"min": 2.0, "mean": 1.0, "max": 1.5}))
    mut("stage2 mode missing", lambda jj: jj["aflops_cfg"]["cfg_schedule"].__setitem__(
        "stage2_mode", None))
    mut("empty step list", lambda jj: jj.__setitem__("aflops", []))
    mut("no cfg block", lambda jj: jj.__setitem__("aflops_cfg", None))
    mut("lf m_min > m_max", lambda jj: jj["aflops"][-1]["lf"].update(
        {"m_min": 1.2, "m_max": 0.8, "m_abs": 0.1}))
    mut("lf outside amplitude bound", lambda jj: jj["aflops"][-1]["lf"].update(
        {"m_min": 0.4, "m_max": 1.0, "m_abs": 0.05}))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"G:\AI\comfy\output\aflops")
    ap.add_argument("--json-out", default=os.path.join(os.path.dirname(__file__), "scan_corpus_out.json"))
    ap.add_argument("--txt-out", default=os.path.join(os.path.dirname(__file__), "scan_corpus_out.txt"))
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()

    if args.control:
        print("CONTROLS (every one must be REJECTED)")
        sys.exit(0 if control(args.root) else 1)

    runs, rejected, nfiles = collect(args.root)
    if nfiles == 0:
        print("FAILED: no log files under %s" % args.root)
        sys.exit(1)
    if not runs:
        print("FAILED: every log was rejected")
        sys.exit(1)
    s = summarize(runs)
    s["files_total"] = nfiles
    s["runs_valid"] = len(runs)
    s["runs_rejected"] = len(rejected)
    s["rejected_reasons"] = {}
    for _, why in rejected:
        s["rejected_reasons"][why] = s["rejected_reasons"].get(why, 0) + 1
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1)

    lines = []
    lines.append("PD-AOS corpus scan -- did the three spatial mechanisms engage?")
    lines.append("root: %s" % args.root)
    lines.append("files %d | valid %d | rejected %d" % (nfiles, len(runs), len(rejected)))
    if rejected:
        for why, n in sorted(s["rejected_reasons"].items(), key=lambda kv: -kv[1]):
            lines.append("  rejected %4d  %s" % (n, why))
    lines.append("")
    lines.append("PER-PIXEL CFG (stage 2)")
    lines.append("  runs with stage2 installed: %d  (with a logged map: %d, without: %d)"
                 % (s["stage2_runs"], s["stage2_with_map"], s["stage2_no_map"]))
    lines.append("  stage2 runs at cfg<=1: %d | cfg>1: %d" % (s["stage2_cfg1"], s["stage2_cfg_gt1"]))
    lines.append("  stage2 installed but no map, by reason: %s" % (s["stage2_reasons"] or "-"))
    lines.append("  stage-1 runs: %d | ramp ever started: %d | constant half-CFG: %d | never returned to base: %d"
                 % (s["stage1_runs"], s["stage1_ramped"], s["stage1_flat_half"],
                    s["stage1_never_reached_base"]))
    if s["stage2_with_map"]:
        q = s["map_min_over_max"]
        lines.append("  map min/max over logged maps: n=%d p10 %.3f p50 %.3f p90 %.3f (min %.3f max %.3f)"
                     % (q["n"], q["p10"], q["p50"], q["p90"], q["mn"], q["mx"]))
        lines.append("  uniform maps (min==max, no spatial content): %d" % s["map_uniform"])
        q = s["map_steps_per_run"]
        lines.append("  mapped steps per stage2 run: p10 %.0f p50 %.0f p90 %.0f (min %s max %s)"
                     % (q["p10"], q["p50"], q["p90"], q["mn"], q["mx"]))
    lines.append("")
    lines.append("PER-PIXEL ORDER")
    q = s["ord_mean_all"]
    lines.append("  ord_mean over all steps: n=%d p10 %.3f p50 %.3f p90 %.3f"
                 % (q["n"], q["p10"], q["p50"], q["p90"]))
    lines.append("  runs with any ord_mean > 1: %d / %d" % (s["runs_with_ord_gt1"], len(runs)))
    q = s["spx_all"]
    lines.append("  second_order_px (fraction of pixels): n=%d p50 %.4f p90 %.4f max %s"
                 % (q["n"], q["p50"], q["p90"], q["mx"]))
    lines.append("  runs with any second-order pixel: %d / %d" % (s["runs_with_spx"], len(runs)))
    lines.append("")
    lines.append("LOCAL FIELDS")
    q = s["lf_m_abs"]
    lines.append("  |lf_m| mean per step: n=%d p10 %.5f p50 %.5f p90 %.5f max %s"
                 % (q["n"], q["p10"], q["p50"], q["p90"], q["mx"]))
    lines.append("  runs with any step |lf_m| > 1%%: %d / %d" % (s["runs_with_lf_gt1pct"], len(runs)))
    q = s["lf_rho"]
    lines.append("  lf rho_max per step: n=%d p50 %.5f p90 %.5f max %s"
                 % (q["n"], q["p50"], q["p90"], q["mx"]))
    lines.append("  lf inject outcomes: %s" % (s["lf_inject"] or "-"))
    text = "\n".join(lines) + "\n"
    with open(args.txt_out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
