#!/usr/bin/env python3
"""
Filter a no-closure ensemble down to a compact, non-redundant, energetically
favourable, geometrically usable set of solutions.

Input is a run directory from ensemble_sampling_noclosure.py: a silent file
(default `ensemble.out`) plus its `metrics.json` and the YAML construct
definition. Output is a new silent file containing only the survivors, plus a
CSV report and a short log explaining how many were lost at each stage.

Three filters, applied in this order (the order matters):

  1. ORIENTATION  - discard solutions whose mobile segment points the wrong way
     to link with something below it. For the F3<->alpha stage, "below" means
     -z (toward the membrane); if the alpha tag's downstream attachment end
     points upward (+z, back toward F3), no amount of TM sampling will place a
     membrane helix from there, so it is dead weight. Cheap and independent of
     the others, so it goes first.

  2. ENERGY       - keep only the most favourable solutions, by absolute
     threshold, by top fraction, or by top N (whichever you specify).

  3. SIMILARITY   - greedy clustering by RMSD over the mobile residues; from
     each cluster keep a single representative. Because energy filtering has
     already run and the survivors are sorted best-energy-first, the
     representative kept is always the lowest-energy member of its cluster.
     This is last because it is the most expensive (pairwise-ish) and there is
     no point clustering structures you were going to drop anyway.

The orientation geometry is the only part specific to the membrane problem;
read its docstring before trusting the defaults on a differently oriented
input. Convention (same as the sampler): membrane normal = +z, bilayer
mid-plane at z = 0, "below / toward the membrane" = -z.

    python filter_ensemble.py --dir ensemble_noclosure/ \
        --energy-percentile 40 --rmsd 2.0 --orient-cone 90 \
        --mobile-segment alpha --out filtered/

The mobile segment whose exit direction is judged defaults to the LAST
flexible segment's downstream rigid neighbour (i.e. the thing the loop hands
off to). For the F3--linker--alpha construct that is `alpha`; name it
explicitly with --mobile-segment if the auto-pick is wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import pyrosetta
from pyrosetta.rosetta.core.io.silent import (
    SilentFileData, SilentFileOptions, BinarySilentStruct,
)
from pyrosetta.rosetta.core.scoring import ScoreType


MEMBRANE_NORMAL = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# Config / metrics loading (mirrors analyse_ensemble.py)
# --------------------------------------------------------------------------- #

def load_metrics(run_dir: Path) -> dict:
    with open(run_dir / "metrics.json") as fh:
        return json.load(fh)


def load_config(run_dir: Path, metrics: dict,
                override: Optional[Path] = None) -> dict:
    if override is not None:
        with open(override) as fh:
            return yaml.safe_load(fh)
    cfg_path = Path(metrics["config"])
    for c in (cfg_path, run_dir / cfg_path.name, run_dir.parent / cfg_path.name):
        if c.is_file():
            with open(c) as fh:
                return yaml.safe_load(fh)
    raise FileNotFoundError(
        f"Cannot locate the YAML config (tried {cfg_path} and {run_dir}). "
        "Pass --config explicitly.")


def parse_score_table(silent_path: Path) -> List[dict]:
    """One dict per SCORE line; numeric columns as floats, tag as string."""
    columns: Optional[List[str]] = None
    rows: List[dict] = []
    with open(silent_path) as fh:
        for line in fh:
            if not line.startswith("SCORE:"):
                continue
            parts = line.split()
            if columns is None:
                columns = parts[1:]
                continue
            row: dict = {}
            for col, val in zip(columns, parts[1:]):
                try:
                    row[col] = float(val)
                except ValueError:
                    row[col] = val
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Segment lookups
# --------------------------------------------------------------------------- #

def segments_by_name(cfg: dict) -> Dict[str, dict]:
    return {s["name"]: s for s in cfg["segments"]}


def flexible_segments(cfg: dict) -> List[dict]:
    return [s for s in cfg["segments"] if s["kind"] == "flexible"]


def mobile_residues(cfg: dict) -> Tuple[List[int], str]:
    """
    Residues that actually moved in the no-closure run: the flexible loops plus
    every rigid segment downstream of the first loop (they ride along on the
    torsions). Equivalently: everything from the first flexible residue to the
    end of the chain. RMSD and orientation are judged over these.
    """
    flex = flexible_segments(cfg)
    if not flex:
        raise SystemExit("No flexible segments in config.")
    first_loop_start = min(s["start"] for s in flex)
    chains = {s.get("chain", "A") for s in cfg["segments"]}
    if len(chains) != 1:
        raise SystemExit(f"Multi-chain constructs not supported: {chains}")
    chain = next(iter(chains))
    last = max(s["stop"] for s in cfg["segments"])
    return list(range(first_loop_start, last + 1)), chain


def pick_mobile_segment(cfg: dict, override: Optional[str]) -> dict:
    """
    The rigid segment whose exit orientation we judge. Default: the rigid
    segment immediately downstream of the last flexible segment (what the loop
    hands off to). For F3--linker--alpha that's `alpha`.
    """
    byname = segments_by_name(cfg)
    if override:
        if override not in byname:
            raise SystemExit(f"--mobile-segment {override!r} not in config "
                             f"(have: {sorted(byname)})")
        seg = byname[override]
        if seg["kind"] != "rigid":
            raise SystemExit(f"{override!r} is not a rigid segment.")
        return seg
    last_loop = max(flexible_segments(cfg), key=lambda s: s["stop"])
    downstream = [s for s in cfg["segments"]
                  if s["kind"] == "rigid" and s["start"] > last_loop["stop"]]
    if not downstream:
        raise SystemExit(
            "No rigid segment downstream of the last flexible loop; "
            "specify --mobile-segment explicitly.")
    return min(downstream, key=lambda s: s["start"])


# --------------------------------------------------------------------------- #
# Silent access - full poses (needed for energy + RMSD + re-writing)
# --------------------------------------------------------------------------- #

def read_all_poses(silent_path: Path) -> Tuple[SilentFileData, List[str]]:
    opts = SilentFileOptions()
    sfd = SilentFileData(opts)
    sfd.read_file(str(silent_path))
    return sfd, list(sfd.tags())


def pose_from_sfd(sfd: SilentFileData, tag: str):
    pose = pyrosetta.Pose()
    sfd.get_structure(tag).fill_pose(pose)
    return pose


def ca_coords(pose, pdb_residues: List[int], chain: str) -> np.ndarray:
    """(N,3) array of CA coordinates for the given PDB-numbered residues."""
    pi = pose.pdb_info()
    want = set(pdb_residues)
    xyz = []
    for i in range(1, pose.total_residue() + 1):
        if int(pi.number(i)) in want and pi.chain(i) == chain:
            ca = pose.residue(i).xyz("CA")
            xyz.append((ca.x, ca.y, ca.z))
    return np.asarray(xyz, dtype=float)


# --------------------------------------------------------------------------- #
# Filter 1: orientation
# --------------------------------------------------------------------------- #

def exit_vector(pose, seg: dict, chain: str) -> np.ndarray:
    """
    Unit vector describing which way the mobile segment's downstream end points,
    measured from the anchor side of the construct.

    Defined as (CA_last - CA_first) of the segment: for a helix declared N->C
    down the chain, this points from the attachment end toward the free /
    downstream end. If the segment hands off further downstream, that free end
    is where the next loop (alpha->TM) would have to start, so its z-direction
    is exactly what decides whether a membrane helix can plausibly follow.
    """
    pi = pose.pdb_info()
    first = last = None
    for i in range(1, pose.total_residue() + 1):
        if pi.chain(i) != chain:
            continue
        n = int(pi.number(i))
        if n == seg["start"]:
            r = pose.residue(i).xyz("CA")
            first = np.array([r.x, r.y, r.z])
        if n == seg["stop"]:
            r = pose.residue(i).xyz("CA")
            last = np.array([r.x, r.y, r.z])
    if first is None or last is None:
        raise SystemExit(f"Segment {seg['name']} endpoints not found in pose.")
    v = last - first
    nrm = np.linalg.norm(v)
    if nrm < 1e-9:
        raise SystemExit(f"Segment {seg['name']} has a degenerate axis.")
    return v / nrm


def orientation_angle_from_down(vec: np.ndarray) -> float:
    """
    Angle in degrees between `vec` and the DOWNWARD membrane normal (-z).
    0 deg = points straight down (ideal for handing off to a TM below).
    180 deg = points straight up (back toward F3; useless).
    """
    down = -MEMBRANE_NORMAL
    cos = float(np.clip(np.dot(vec, down), -1.0, 1.0))
    return math.degrees(math.acos(cos))


# --------------------------------------------------------------------------- #
# Filter 2: energy - handled inline in main (needs the whole population)
# --------------------------------------------------------------------------- #

def energy_of(pose, sfxn) -> float:
    return float(sfxn(pose))


# --------------------------------------------------------------------------- #
# Filter 3: similarity - greedy RMSD clustering
# --------------------------------------------------------------------------- #

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Minimal-RMSD superposition of two equal-length (N,3) point sets.

    Superposing here rather than taking raw RMSD is the right choice: two
    linker conformations that differ only by where the whole downstream body
    sits in the lab frame can be the SAME conformation you don't want twice, or
    genuinely different geometries - superposition over the mobile residues
    tells them apart by shape, not by absolute placement. If instead you want
    to treat different lab-frame placements as distinct (because the TM lands
    somewhere different), skip the superposition and use raw RMSD; pass
    --no-superpose.
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    Prot = Pc @ U
    diff = Prot - Qc
    return float(np.sqrt((diff * diff).sum() / len(P)))


def raw_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    diff = P - Q
    return float(np.sqrt((diff * diff).sum() / len(P)))


def greedy_cluster(order: List[str],
                   coords: Dict[str, np.ndarray],
                   cutoff: float, superpose: bool) -> List[str]:
    """
    Walk `order` (already sorted best-energy-first). Keep a structure iff it is
    at least `cutoff` RMSD from every structure already kept. Returns the kept
    tags (the cluster representatives).

    O(k * n) where k = number kept: cheap when the cutoff is tight enough that
    k stays small, which is the regime you actually run this in.
    """
    rmsd = kabsch_rmsd if superpose else raw_rmsd
    kept: List[str] = []
    kept_coords: List[np.ndarray] = []
    for tag in order:
        c = coords[tag]
        if all(rmsd(c, kc) >= cutoff for kc in kept_coords):
            kept.append(tag)
            kept_coords.append(c)
    return kept


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_filtered_silent(out_path: Path, sfd: SilentFileData,
                          tags: List[str], extra: Dict[str, Dict[str, float]]):
    opts = SilentFileOptions()
    out = SilentFileData(opts)
    for tag in tags:
        ss = sfd.get_structure(tag)          # reuse the stored struct verbatim
        for k, v in extra.get(tag, {}).items():
            ss.add_energy(k, float(v))
        out.write_silent_struct(ss, str(out_path))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True,
                   help="Run directory from ensemble_sampling_noclosure.py")
    p.add_argument("--silent", default="ensemble.out",
                   help="Silent file name inside --dir")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (defaults to what metrics.json stored)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: <dir>/filtered)")
    p.add_argument("--out-silent", default="filtered.out",
                   help="Filtered silent file name inside --out")

    # Orientation
    p.add_argument("--mobile-segment", default=None,
                   help="Rigid segment whose exit direction is judged "
                        "(default: rigid segment downstream of the last loop)")
    p.add_argument("--orient-cone", type=float, default=90.0,
                   help="Keep a solution only if its exit vector is within this "
                        "many degrees of straight-down (-z). 90 = any downward "
                        "component; smaller = stricter. Set to 180 to disable.")

    # Energy - pick ONE; if several given, the most restrictive wins.
    p.add_argument("--energy-threshold", type=float, default=None,
                   help="Absolute REU cutoff: keep poses with total_score below "
                        "this.")
    p.add_argument("--energy-percentile", type=float, default=None,
                   help="Keep the best this-percent by total_score (e.g. 40 = "
                        "best 40%%).")
    p.add_argument("--energy-top-n", type=int, default=None,
                   help="Keep the best N by total_score.")

    # Similarity
    p.add_argument("--rmsd", type=float, default=2.0,
                   help="Similarity cutoff (A) over mobile residues. Two "
                        "survivors are never closer than this. 0 disables "
                        "clustering.")
    p.add_argument("--no-superpose", action="store_true",
                   help="Use raw RMSD (lab-frame) instead of superposed RMSD, "
                        "so different placements of the downstream body count "
                        "as different even at identical loop shape.")

    p.add_argument("--use-stored-score", action="store_true",
                   help="Trust the total_score already in the silent file "
                        "instead of re-scoring each pose with ref2015. Faster; "
                        "use only if you didn't change the score function.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.dir
    silent_path = run_dir / args.silent
    if not silent_path.is_file():
        raise SystemExit(f"Silent file not found: {silent_path}")

    out_dir = args.out or (run_dir / "filtered")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_silent = out_dir / args.out_silent
    if out_silent.exists():
        raise SystemExit(f"{out_silent} already exists - move or delete it first "
                         "(silent files are appended to).")

    metrics = load_metrics(run_dir)
    cfg = load_config(run_dir, metrics, args.config)
    mob_res, chain = mobile_residues(cfg)
    mob_seg = pick_mobile_segment(cfg, args.mobile_segment)

    score_rows = parse_score_table(silent_path)
    all_tags = [str(r["description"]) for r in score_rows]
    stored_score = {str(r["description"]): float(r.get("total_score", float("nan")))
                    for r in score_rows}

    print(f"Input     : {len(all_tags)} structures from {silent_path}")
    print(f"Mobile    : residues {mob_res[0]}-{mob_res[-1]} (chain {chain})")
    print(f"Orient on : segment {mob_seg['name']} "
          f"({mob_seg['start']}-{mob_seg['stop']}), "
          f"cone {args.orient_cone:g} deg from -z")

    pyrosetta.init(f"-mute all -constant_seed -jran {args.seed or 1}")
    sfd, tags = read_all_poses(silent_path)
    sfxn = pyrosetta.create_score_function("ref2015")

    # Decode every pose once; compute energy + exit angle + mobile CA coords.
    energy: Dict[str, float] = {}
    angle: Dict[str, float] = {}
    coords: Dict[str, np.ndarray] = {}
    for tag in tags:
        pose = pose_from_sfd(sfd, tag)
        energy[tag] = (stored_score[tag] if args.use_stored_score
                       else energy_of(pose, sfxn))
        angle[tag] = orientation_angle_from_down(exit_vector(pose, mob_seg, chain))
        coords[tag] = ca_coords(pose, mob_res, chain)

    n0 = len(tags)

    # ---- Filter 1: orientation ----
    if args.orient_cone >= 180.0:
        surviving = list(tags)
        print(f"[1] orientation : disabled")
    else:
        surviving = [t for t in tags if angle[t] <= args.orient_cone]
        print(f"[1] orientation : {len(surviving)}/{n0} kept "
              f"(exit within {args.orient_cone:g} deg of down)")
    if not surviving:
        raise SystemExit("Orientation filter removed everything - loosen "
                         "--orient-cone.")

    # ---- Filter 2: energy ----
    scores = np.array([energy[t] for t in surviving])
    keep_energy = set(surviving)
    reasons = []
    if args.energy_threshold is not None:
        keep_energy &= {t for t in surviving if energy[t] < args.energy_threshold}
        reasons.append(f"< {args.energy_threshold:g} REU")
    if args.energy_percentile is not None:
        cut = float(np.percentile(scores, args.energy_percentile))
        keep_energy &= {t for t in surviving if energy[t] <= cut}
        reasons.append(f"best {args.energy_percentile:g}% (<= {cut:.2f} REU)")
    if args.energy_top_n is not None:
        best_n = set(sorted(surviving, key=lambda t: energy[t])[:args.energy_top_n])
        keep_energy &= best_n
        reasons.append(f"top {args.energy_top_n}")
    surviving = [t for t in surviving if t in keep_energy]
    if reasons:
        print(f"[2] energy      : {len(surviving)} kept ({'; '.join(reasons)})")
    else:
        print(f"[2] energy      : no energy filter set - all {len(surviving)} kept")
    if not surviving:
        raise SystemExit("Energy filter removed everything - loosen the cutoff.")

    # ---- Filter 3: similarity ----
    if args.rmsd <= 0:
        kept = surviving
        print(f"[3] similarity  : disabled")
    else:
        order = sorted(surviving, key=lambda t: energy[t])   # best energy first
        kept = greedy_cluster(order, coords, args.rmsd,
                              superpose=not args.no_superpose)
        print(f"[3] similarity  : {len(kept)} representatives from "
              f"{len(surviving)} (RMSD >= {args.rmsd:g} A, "
              f"{'raw' if args.no_superpose else 'superposed'})")

    # Order final set best-energy-first.
    kept = sorted(kept, key=lambda t: energy[t])

    # ---- Write outputs ----
    extra = {t: {"filter_energy": energy[t], "exit_angle_down": angle[t]}
             for t in kept}
    write_filtered_silent(out_silent, sfd, kept, extra)

    report = out_dir / "filter_report.csv"
    with open(report, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "total_score", "exit_angle_down_deg", "kept"])
        keptset = set(kept)
        for t in sorted(tags, key=lambda t: energy[t]):
            w.writerow([t, f"{energy[t]:.3f}", f"{angle[t]:.2f}",
                        int(t in keptset)])

    with open(out_dir / "filter_summary.json", "w") as fh:
        json.dump({
            "input_silent": str(silent_path),
            "n_input": n0,
            "n_output": len(kept),
            "orient_cone_deg": args.orient_cone,
            "mobile_segment": mob_seg["name"],
            "energy_threshold": args.energy_threshold,
            "energy_percentile": args.energy_percentile,
            "energy_top_n": args.energy_top_n,
            "rmsd_cutoff": args.rmsd,
            "superposed": not args.no_superpose,
            "kept_tags": kept,
        }, fh, indent=2)

    print(f"\nKept {len(kept)}/{n0} structures.")
    print(f"  silent : {out_silent}")
    print(f"  report : {report}")
    print(f"  extract: extract_pdbs -in:file:silent {out_silent} "
          f"-in:file:tags {kept[0] if kept else '<tag>'}")


if __name__ == "__main__":
    main()