#!/usr/bin/env python3
"""
Compare the reachable space of a rigid segment between two constructs -
typically WT (one linker) versus tagged (two loops flanking a rigid insert).

The question is not "are these two ensembles different" but "which region of
TM-placement space can each construct reach". Three reportables:

  reachable fraction   what proportion of attempted placements closed
  reach envelope       convex hull volume and max reach of the TM midpoint
  orientational freedom  spread of TM axis directions

Two facts about the sampler drive the whole design.

First, within a placement the rigid-body geometry is *constant* - the loop MC
moves torsions, and the TM hangs off a jump, so all n_traj models at one
placement share one TM position. The unit of analysis is therefore the
placement, not the model, and everything here collapses to one row per
placement. An n_traj of 10 gives you ten attempts at closure, not ten
geometries.

Second, both hull volume and directional coverage grow with sample size, so
comparing a 20-placement run against a 35-placement run is confounded by n
alone. Every envelope statistic here is computed on equal-n subsamples,
repeated, with bootstrap intervals over placements.

    ./compare_ensembles.py \\
        --wt  wt/ensemble.out  wt/metrics.json  wt.yaml \\
        --tag tag/ensemble.out tag/metrics.json tagged.yaml \\
        --segment tm --closure 0.5 --out compare/

Descriptors are cached to <out>/descriptors_{wt,tag}.json, so re-running the
statistics with a different --closure needs neither PyRosetta nor the silent
files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Geometry, in a frame built on the anchor domain
# --------------------------------------------------------------------------- #

def anchor_frame(pose, a_start: int, a_mid: int, a_stop: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthonormal frame from three anchor CAs, by Gram-Schmidt.

    Everything is expressed in this frame rather than in global coordinates, so
    the comparison does not depend on the two input structures having been
    superimposed on FnIII-3 beforehand. Origin is the anchor midpoint CA.
    """
    p0 = ca(pose, a_start)
    p1 = ca(pose, a_mid)
    p2 = ca(pose, a_stop)

    e1 = p2 - p0
    e1 /= np.linalg.norm(e1)
    t = p1 - p0
    e2 = t - np.dot(t, e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-6:
        raise SystemExit("Anchor CAs are collinear - pick a different mid residue")
    e2 /= n2
    e3 = np.cross(e1, e2)
    return p1, np.vstack([e1, e2, e3])       # rows are basis vectors


def ca(pose, i: int) -> np.ndarray:
    v = pose.residue(i).xyz("CA")
    return np.array([v.x, v.y, v.z])


def descriptors_for(pose, layout, seg_name: str) -> dict:
    """
    Reach vector and axis direction of the compared segment, in the anchor frame.

    The axis is kept signed (N -> C). Folding it into 0-90 degrees, as
    segment_tilt() does for the membrane angle, throws away half the
    orientational information, which is exactly what we are trying to measure.
    """
    a_lo, a_hi = layout["anchor"]
    origin, basis = anchor_frame(pose, a_lo, (a_lo + a_hi) // 2, a_hi)

    attach = layout["attachment"]
    s_lo, s_hi = layout["segment"]
    mid = (s_lo + s_hi) // 2

    to_local = lambda p: basis @ (p - origin)

    a = to_local(ca(pose, attach))
    m = to_local(ca(pose, mid))
    u = to_local(ca(pose, s_hi)) - to_local(ca(pose, s_lo))
    u = u / np.linalg.norm(u)

    reach = m - a
    return {"reach": reach.tolist(),
            "r": float(np.linalg.norm(reach)),
            "axis": u.tolist(),
            "mid": m.tolist()}


# --------------------------------------------------------------------------- #
# Reading a run
# --------------------------------------------------------------------------- #

def layout_from_yaml(path: str, seg_name: str) -> dict:
    segs = yaml.safe_load(open(path))["segments"]
    rigid = [s for s in segs if s["kind"] == "rigid"]
    flex = [s for s in segs if s["kind"] == "flexible"]
    target = [s for s in rigid if s["name"] == seg_name]
    if not target:
        raise SystemExit(f"No rigid segment named '{seg_name}' in {path}")
    return {"anchor": (int(rigid[0]["start"]), int(rigid[0]["stop"])),
            "attachment": int(flex[0]["start"]) - 1,
            "segment": (int(target[0]["start"]), int(target[0]["stop"])),
            "chain": str(segs[0].get("chain", "A"))}


def collapse_to_placements(models: List[dict], closure: float) -> Dict[int, dict]:
    """
    One row per placement.

    A placement counts as reachable if *any* of its trajectories closed, which
    is the existential quantifier the research question needs: can this TM
    position be bridged at all, not does it usually get bridged.
    """
    by_p: Dict[int, List[dict]] = {}
    for m in models:
        by_p.setdefault(int(m["placement"]), []).append(m)

    out = {}
    for p, rows in by_p.items():
        best = min(rows, key=lambda r: r["chainbreak"])
        out[p] = {"placement": p,
                  "n_traj": len(rows),
                  "best_chainbreak": best["chainbreak"],
                  "reachable": bool(best["chainbreak"] <= closure),
                  "best_tag": best["tag"],
                  "dof": {k: v for k, v in best.items()
                          if k.endswith(("_dx", "_dy", "_dz", "_tilt",
                                         "_tilt_azimuth", "_spin"))}}
    return out


def compute_descriptors(silent: str, metrics: str, config: str,
                        seg_name: str, closure: float) -> dict:
    """Decode one representative pose per placement and measure it."""
    meta = json.load(open(metrics))
    placements = collapse_to_placements(meta["models"], closure)
    layout = layout_from_yaml(config, seg_name)

    import pyrosetta
    pyrosetta.init("-mute all")
    from pyrosetta.rosetta.core.io.silent import SilentFileData
    try:
        from pyrosetta.rosetta.core.io.silent import SilentFileOptions
        sfd = SilentFileData(SilentFileOptions())
    except ImportError:
        sfd = SilentFileData()
    sfd.read_file(silent)

    first = True
    for p, row in sorted(placements.items()):
        pose = pyrosetta.Pose()
        sfd.get_structure(row["best_tag"]).fill_pose(pose)
        if first:
            info = pose.pdb_info()
            if info is not None:
                a, _ = layout["anchor"]
                if not info.pdb2pose(layout["chain"], a):
                    print("  [numbering] PDBInfo does not cover the YAML "
                          "numbering; treating YAML numbers as pose numbers")
            first = False
        row.update(descriptors_for(pose, layout, seg_name))

    return {"source": {"silent": silent, "metrics": metrics, "config": config},
            "meta": {k: v for k, v in meta.items() if k != "models"},
            "closure": closure,
            "placements": [placements[p] for p in sorted(placements)]}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def hull_volume(points: np.ndarray) -> Optional[float]:
    if len(points) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(points).volume)
    except Exception:                                              # noqa: BLE001
        return None                     # degenerate/coplanar, or no scipy


def spherical_spread(axes: np.ndarray) -> float:
    """
    1 - |mean resultant vector|. 0 = all axes parallel, 1 = maximally dispersed.

    Preferred over counting occupied cells on a sphere, which is dominated by
    sample size at n = 20.
    """
    if len(axes) == 0:
        return float("nan")
    return float(1.0 - np.linalg.norm(axes.mean(axis=0)))


def envelope_stats(rows: List[dict]) -> dict:
    if not rows:
        return {"n": 0}
    mid = np.array([r["mid"] for r in rows])
    axes = np.array([r["axis"] for r in rows])
    r = np.array([r["r"] for r in rows])
    return {"n": len(rows),
            "hull_volume": hull_volume(mid),
            "max_reach": float(r.max()),
            "mean_reach": float(r.mean()),
            "position_spread": float(np.sqrt(((mid - mid.mean(0)) ** 2).sum(1).mean())),
            "axis_spread": spherical_spread(axes),
            "centroid": mid.mean(0).tolist()}


def equal_n(rows_a: List[dict], rows_b: List[dict], reps: int,
            rng: random.Random) -> Tuple[dict, dict, int]:
    """
    Envelope statistics on equal-sized subsamples of both sets.

    Hull volume and directional coverage both grow with n, so a construct that
    simply reached more placements would look like it had a larger envelope
    even if its per-placement geometry were identical. Subsampling to the
    smaller n removes that.
    """
    n = min(len(rows_a), len(rows_b))
    if n < 4:
        return {"n": n}, {"n": n}, n

    acc: Tuple[List[dict], List[dict]] = ([], [])
    for _ in range(reps):
        for k, rows in enumerate((rows_a, rows_b)):
            acc[k].append(envelope_stats(rng.sample(rows, n)))

    def mean_of(dicts, key):
        vals = [d[key] for d in dicts if d.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    keys = ["hull_volume", "max_reach", "mean_reach", "position_spread", "axis_spread"]
    return ({"n": n, **{k: mean_of(acc[0], k) for k in keys}},
            {"n": n, **{k: mean_of(acc[1], k) for k in keys}}, n)


def bootstrap_diff(rows_a: List[dict], rows_b: List[dict], key: str,
                   reps: int, rng: random.Random) -> Optional[Tuple[float, float]]:
    """Percentile CI on (b - a) for one envelope statistic, resampling placements."""
    n = min(len(rows_a), len(rows_b))
    if n < 4:
        return None
    diffs = []
    for _ in range(reps):
        sa = envelope_stats([rng.choice(rows_a) for _ in range(n)])
        sb = envelope_stats([rng.choice(rows_b) for _ in range(n)])
        if sa.get(key) is None or sb.get(key) is None:
            continue
        diffs.append(sb[key] - sa[key])
    if len(diffs) < reps // 4:
        return None
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #

def dof_key(row: dict, nd: int = 3) -> Optional[tuple]:
    d = row.get("dof") or {}
    if not d:
        return None
    return tuple(round(d[k], nd) for k in sorted(d))


def pair_up(a: List[dict], b: List[dict]) -> Optional[List[Tuple[dict, dict]]]:
    """
    Match placements between runs by their drawn DOF.

    Returns None unless most placements match, which is the signal that the two
    runs did not share a placement list. Without a shared list the comparison
    is unpaired and much weaker - see the note printed in that case.
    """
    ka = {dof_key(r): r for r in a if dof_key(r)}
    kb = {dof_key(r): r for r in b if dof_key(r)}
    shared = set(ka) & set(kb)
    if not shared or len(shared) < 0.5 * min(len(ka), len(kb)):
        return None
    return [(ka[k], kb[k]) for k in sorted(shared, key=lambda t: t)]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def report(dw: dict, dt: dict, args) -> dict:
    rng = random.Random(args.seed)
    pw, pt = dw["placements"], dt["placements"]
    rw = [r for r in pw if r["reachable"]]
    rt = [r for r in pt if r["reachable"]]

    out: dict = {"closure_threshold": args.closure, "segment": args.segment}

    print("=" * 68)
    print(f"Reachable space of '{args.segment}'   (closure: chainbreak <= {args.closure})")
    print("=" * 68)

    # ---- 1. reachable fraction ------------------------------------------- #
    print("\n1. REACHABLE FRACTION")
    for name, p, r in (("WT", pw, rw), ("tagged", pt, rt)):
        lo, hi = wilson(len(r), len(p))
        print(f"   {name:<7} {len(r):3d}/{len(p):3d} = {len(r)/len(p):.3f}  "
              f"[{lo:.3f}, {hi:.3f}]")
        out[f"reachable_{name}"] = {"k": len(r), "n": len(p), "ci": [lo, hi]}

    pairs = pair_up(pw, pt)
    if pairs:
        b = sum(1 for x, y in pairs if x["reachable"] and not y["reachable"])
        c = sum(1 for x, y in pairs if y["reachable"] and not x["reachable"])
        p_val = mcnemar_exact(b, c)
        print(f"   paired on {len(pairs)} shared placements: "
              f"{c} gained by the insert, {b} lost, exact McNemar p = {p_val:.4f}")
        out["paired"] = {"n": len(pairs), "gained": c, "lost": b, "p": p_val}
    else:
        print("   NOT PAIRED: the two runs do not share a placement list, so "
              "this comparison\n   is unpaired and much weaker. Generate the "
              "placements once and feed the same\n   file to both runs.")
        out["paired"] = None

    attempts = dw["meta"].get("placement_attempts")
    kept = dw["meta"].get("placements")
    if attempts and kept and attempts > kept:
        print(f"\n   CAVEAT: the WT run drew {attempts} placements but recorded "
              f"{kept}. Placements\n   rejected by placement_is_viable never "
              "reach metrics.json, so the fraction above\n   is conditioned on "
              "that pre-filter rather than measuring it. Disable the\n   "
              "pre-filter and record failures to get the real denominator.")
        out["denominator_warning"] = True

    # ---- 2 & 3. envelope and orientation --------------------------------- #
    ew, et, n = equal_n(rw, rt, args.reps, rng)
    print(f"\n2. REACH ENVELOPE   (equal-n subsamples, n = {n}, {args.reps} reps)")
    rows = [("hull volume (A^3)", "hull_volume", "{:.0f}"),
            ("max reach (A)", "max_reach", "{:.2f}"),
            ("mean reach (A)", "mean_reach", "{:.2f}"),
            ("position spread (A)", "position_spread", "{:.2f}")]
    print(f"   {'':<22}{'WT':>12}{'tagged':>12}{'delta':>12}   95% CI on delta")
    for label, key, fmt in rows:
        _print_row(label, key, fmt, ew, et, rw, rt, args, rng, out)

    print("\n3. ORIENTATIONAL FREEDOM")
    _print_row("axis spread (0-1)", "axis_spread", "{:.3f}",
               ew, et, rw, rt, args, rng, out)

    if ew.get("centroid") is None and rw and rt:
        pass
    if rw and rt:
        cw = np.array(envelope_stats(rw)["centroid"])
        ct = np.array(envelope_stats(rt)["centroid"])
        shift = float(np.linalg.norm(ct - cw))
        print(f"\n   centroid shift WT -> tagged: {shift:.2f} A "
              "(displacement, not spread - a shifted envelope of the same size "
              "is\n   a different finding from an enlarged one)")
        out["centroid_shift"] = shift

    out["envelope_wt"], out["envelope_tagged"] = ew, et

    # ---- naive ceiling ---------------------------------------------------- #
    if args.linker_residues_wt and args.linker_residues_tag:
        bw = 3.3 * (args.linker_residues_wt + 1)
        bt = 3.3 * (args.linker_residues_tag + 1)
        print(f"\n4. AGAINST THE FREELY-JOINTED CEILING")
        print(f"   WT     max reach {ew.get('max_reach', float('nan')):.1f} A "
              f"vs ceiling {bw:.1f} A  ({ew.get('max_reach', 0)/bw:.0%})")
        print(f"   tagged max reach {et.get('max_reach', float('nan')):.1f} A "
              f"vs ceiling {bt:.1f} A  ({et.get('max_reach', 0)/bt:.0%})")
        print("   If tagged sits close to its ceiling, the extra reach is just "
              "extra residues\n   and the insert's rigidity is not doing much.")
        out["ceiling"] = {"wt": bw, "tagged": bt}

    return out


def _print_row(label, key, fmt, ew, et, rw, rt, args, rng, out):
    a, b = ew.get(key), et.get(key)
    if a is None or b is None:
        print(f"   {label:<22}{'n/a':>12}{'n/a':>12}")
        return
    ci = bootstrap_diff(rw, rt, key, args.reps, rng)
    ci_s = f"[{ci[0]:+.3g}, {ci[1]:+.3g}]" if ci else "n/a"
    crosses = "" if (ci and (ci[0] > 0 or ci[1] < 0)) else "   (CI includes 0)"
    print(f"   {label:<22}{fmt.format(a):>12}{fmt.format(b):>12}"
          f"{fmt.format(b - a):>12}   {ci_s}{crosses}")
    out.setdefault("deltas", {})[key] = {"wt": a, "tagged": b, "ci": ci}


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wt", nargs=3, metavar=("SILENT", "METRICS", "YAML"))
    p.add_argument("--tag", nargs=3, metavar=("SILENT", "METRICS", "YAML"))
    p.add_argument("--descriptors", nargs=2, metavar=("WT_JSON", "TAG_JSON"),
                   help="Skip the silent files and reuse cached descriptors")
    p.add_argument("--segment", default="tm", help="Rigid segment to compare")
    p.add_argument("--closure", type=float, default=0.5,
                   help="chainbreak at or below which a placement counts as "
                        "reached (default 0.5)")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--linker-residues-wt", type=int)
    p.add_argument("--linker-residues-tag", type=int,
                   help="Total flexible residues, for the freely-jointed ceiling")
    p.add_argument("--out", default="compare")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.descriptors:
        dw = json.load(open(args.descriptors[0]))
        dt = json.load(open(args.descriptors[1]))
        for d in (dw, dt):                       # re-apply the closure threshold
            for r in d["placements"]:
                r["reachable"] = bool(r["best_chainbreak"] <= args.closure)
    else:
        if not (args.wt and args.tag):
            raise SystemExit("Give --wt and --tag, or --descriptors")
        dw = compute_descriptors(*args.wt, args.segment, args.closure)
        dt = compute_descriptors(*args.tag, args.segment, args.closure)
        json.dump(dw, open(outdir / "descriptors_wt.json", "w"), indent=1)
        json.dump(dt, open(outdir / "descriptors_tag.json", "w"), indent=1)

    res = report(dw, dt, args)
    json.dump(res, open(outdir / "comparison.json", "w"), indent=1)
    print(f"\nWritten to {outdir}/")


if __name__ == "__main__":
    main()