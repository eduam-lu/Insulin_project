#!/usr/bin/env python3
"""orient_bodies.py

Canonically orient rigid bodies in a construct before ensemble sampling.

Reads the same segments YAML as ensemble_sampling.py, with an added `orient:`
block on any rigid segment whose long axis should be constrained relative to
the membrane normal (z axis). Writes a PDB where each specified body has been
rotated (shortest-arc, preserving N->C direction) and optionally translated.

Flexible segments are left where they end up -- the resulting linker regions
will be broken. That is deliberate: ensemble_sampling.py already closes them
via MC + KIC, and any closure done here would be immediately overwritten.

YAML addition per rigid segment (any not listed is left alone):

    orient:
      axis: parallel        # or 'perpendicular' -- to the z axis
      translate:            # optional; each key optional
        x: 0.0              # if absent, x is not translated
        y: 0.0
        z: 0.0

Input and output can be either PDB or mmCIF; the format is inferred from the
file extension (.pdb / .ent -> PDB; .cif / .mmcif -> mmCIF).

Usage:
    python orient_bodies.py --pdb in.pdb  --config construct.yaml --out oriented.pdb
    python orient_bodies.py --pdb in.cif  --config construct.yaml --out oriented.pdb
    python orient_bodies.py --pdb in.cif  --config construct.yaml --out oriented.cif

Depends on: numpy, pyyaml, biopython.
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterator, Tuple

import numpy as np
import yaml
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Residue import Residue


# --------------------------------------------------------------------------- #
# I/O format dispatch
# --------------------------------------------------------------------------- #

def parser_for(path: str):
    """BioPython structure parser chosen from the file extension."""
    lower = path.lower()
    if lower.endswith((".cif", ".mmcif")):
        return MMCIFParser(QUIET=True)
    if lower.endswith((".pdb", ".ent")):
        return PDBParser(QUIET=True)
    raise ValueError(
        f"unknown structure format for {path!r} "
        "(expected .pdb, .ent, .cif or .mmcif)"
    )


def writer_for(path: str):
    """BioPython structure writer chosen from the file extension."""
    lower = path.lower()
    if lower.endswith((".cif", ".mmcif")):
        return MMCIFIO()
    if lower.endswith((".pdb", ".ent")):
        return PDBIO()
    raise ValueError(
        f"unknown structure format for {path!r} "
        "(expected .pdb, .ent, .cif or .mmcif)"
    )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def long_axis_and_centroid(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """PCA principal direction of `coords` (N x 3), oriented N->C.

    Returns (unit axis, centroid, elongation ratio s1/s2).
    """
    if coords.shape[0] < 3:
        raise ValueError("need at least 3 CA atoms to define a long axis")
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, singular, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0] / np.linalg.norm(Vt[0])
    n_to_c = coords[-1] - coords[0]
    if np.dot(axis, n_to_c) < 0:
        axis = -axis
    ratio = float(singular[0] / singular[1]) if singular[1] > 1e-9 else float("inf")
    return axis, centroid, ratio


def shortest_arc_rotation(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping unit v_from onto unit v_to via the shortest arc."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    c = float(np.dot(v_from, v_to))
    if c > 0.9999:
        return np.eye(3)
    if c < -0.9999:
        # 180 deg rotation about any axis perpendicular to v_from
        perp = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    v = np.cross(v_from, v_to)
    s = float(np.linalg.norm(v))
    v_hat = v / s
    K = np.array([[0.0, -v_hat[2], v_hat[1]],
                  [v_hat[2], 0.0, -v_hat[0]],
                  [-v_hat[1], v_hat[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


# --------------------------------------------------------------------------- #
# Residue selection & transform
# --------------------------------------------------------------------------- #

def iter_residues(model, chain_id: str, start: int, stop: int) -> Iterator[Residue]:
    """Yield polymer residues on `chain_id` with PDB number in [start, stop]."""
    if chain_id not in model:
        raise KeyError(f"chain {chain_id!r} not in structure")
    for res in model[chain_id].get_residues():
        het_flag, resnum, _icode = res.get_id()
        if het_flag != " ":
            continue  # skip HETATMs and waters
        if start <= resnum <= stop:
            yield res


def collect_ca(model, chain_id: str, start: int, stop: int) -> np.ndarray:
    coords = [res["CA"].get_coord() for res in iter_residues(model, chain_id, start, stop)
              if "CA" in res]
    if not coords:
        raise ValueError(f"no CA atoms found for {chain_id}/{start}-{stop}")
    return np.asarray(coords, dtype=float)


def transform_body(model, chain_id: str, start: int, stop: int,
                   R: np.ndarray, delta: np.ndarray, pivot: np.ndarray) -> None:
    """Apply x -> R (x - pivot) + pivot + delta to every atom in the range."""
    for res in iter_residues(model, chain_id, start, stop):
        for atom in res.get_atoms():
            x = np.asarray(atom.get_coord(), dtype=float)
            atom.set_coord(R @ (x - pivot) + pivot + delta)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path: str):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if "segments" not in cfg:
        raise ValueError("config must have a top-level 'segments' key")
    orientations = []
    for s in cfg["segments"]:
        if s.get("kind") != "rigid":
            continue
        orient = s.get("orient")
        if orient is None:
            continue
        axis = orient.get("axis")
        if axis not in ("parallel", "perpendicular"):
            raise ValueError(
                f"{s['name']}: orient.axis must be 'parallel' or 'perpendicular' "
                f"(got {axis!r})"
            )
        translate = orient.get("translate") or {}
        bad = set(translate) - {"x", "y", "z"}
        if bad:
            raise ValueError(f"{s['name']}: unknown translate keys {sorted(bad)}")
        orientations.append({
            "name": s["name"],
            "chain": s.get("chain", "A"),
            "start": int(s["start"]),
            "stop": int(s["stop"]),
            "axis": axis,
            "translate": {k: float(v) for k, v in translate.items()},
        })
    return cfg, orientations


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def report_gaps(model, cfg) -> None:
    """CA-CA gap across each flexible segment vs. its max extended reach."""
    segments = cfg["segments"]
    print("\nLinker gaps after reorientation:")
    for i, s in enumerate(segments):
        if s.get("kind") != "flexible":
            continue
        prev_r = segments[i - 1] if i > 0 else None
        next_r = segments[i + 1] if i + 1 < len(segments) else None
        if prev_r is None or next_r is None:
            print(f"  {s['name']}: cannot bound (missing flanking rigid segment)")
            continue
        chain = s.get("chain", "A")
        try:
            a = collect_ca(model, chain, prev_r["stop"], prev_r["stop"])[0]
            b = collect_ca(model, chain, next_r["start"], next_r["start"])[0]
        except (KeyError, ValueError, IndexError) as exc:
            print(f"  {s['name']}: could not measure ({exc})")
            continue
        gap = float(np.linalg.norm(b - a))
        n_res = int(s["stop"]) - int(s["start"]) + 1
        reach = 3.3 * (n_res + 1)
        marker = "OK" if gap < 0.9 * reach else "TIGHT/UNREACHABLE"
        print(f"  {s['name']:12s} {n_res:3d} res  gap {gap:6.1f} A  "
              f"max reach {reach:6.1f} A  [{marker}]")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pdb", required=True,
                   help="Input structure (.pdb, .ent, .cif or .mmcif)")
    p.add_argument("--config", required=True,
                   help="Construct YAML (same file used by ensemble_sampling.py)")
    p.add_argument("--out", required=True,
                   help="Output structure (.pdb, .ent, .cif or .mmcif); "
                        "format inferred from the extension")
    args = p.parse_args()

    cfg, orientations = load_config(args.config)
    if not orientations:
        print("No rigid segments have an 'orient:' block -- nothing to do.",
              file=sys.stderr)
        return 1

    struct = parser_for(args.pdb).get_structure("in", args.pdb)
    model = struct[0]

    print(f"Reorienting {len(orientations)} rigid segments:\n")
    for body in orientations:
        coords = collect_ca(model, body["chain"], body["start"], body["stop"])
        input_axis, centroid, elong = long_axis_and_centroid(coords)

        if body["axis"] == "parallel":
            # Pick +z or -z, whichever is nearer -- preserves N->C direction.
            target = np.array([0.0, 0.0, 1.0 if input_axis[2] >= 0 else -1.0])
        else:  # perpendicular: project input axis onto xy plane
            xy = np.array([input_axis[0], input_axis[1], 0.0])
            if np.linalg.norm(xy) < 1e-6:
                target = np.array([1.0, 0.0, 0.0])
            else:
                target = xy / np.linalg.norm(xy)

        R = shortest_arc_rotation(input_axis, target)

        # Rotate about the body's own centroid, so it stays put; translation
        # (if any) is applied on top.
        delta = np.zeros(3)
        for i, k in enumerate("xyz"):
            if k in body["translate"]:
                delta[i] = body["translate"][k] - centroid[i]

        angle_before = float(np.degrees(np.arccos(min(1.0, abs(input_axis[2])))))
        if body["axis"] == "parallel":
            axis_desc = f"long axis {angle_before:5.1f} deg from z, now aligned"
        else:
            axis_desc = f"long axis {angle_before:5.1f} deg from z (want 90), now in xy plane"

        c_before = [float(x) for x in centroid]
        c_after = [c_before[i] + float(delta[i]) for i in range(3)]
        trans_note = "" if not body["translate"] else ""
        elong_note = "" if elong >= 1.5 else f"  [WARN: elongation ratio {elong:.2f}, orientation poorly defined]"

        print(f"  {body['name']} ({body['chain']}/{body['start']}-{body['stop']}):")
        print(f"    {axis_desc}{elong_note}")
        print(f"    centroid ({c_before[0]:6.1f},{c_before[1]:6.1f},{c_before[2]:6.1f})"
              f" -> ({c_after[0]:6.1f},{c_after[1]:6.1f},{c_after[2]:6.1f})")

        transform_body(model, body["chain"], body["start"], body["stop"],
                       R, delta, centroid)

    report_gaps(model, cfg)

    io = writer_for(args.out)
    io.set_structure(struct)
    io.save(args.out)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())