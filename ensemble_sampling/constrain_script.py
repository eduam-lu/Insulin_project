#!/usr/bin/env python3
"""orient_bodies.py

Canonically orient rigid bodies in a construct before ensemble sampling.

Reads the same segments YAML as ensemble_sampling.py, with an added `orient:`
block on any rigid segment whose long axis should be constrained relative to
the membrane normal (z axis). Writes a PDB where each specified body has been
rotated (shortest-arc) and optionally translated.

Flexible segments are left where they end up -- the resulting linker regions
will be broken. That is deliberate: ensemble_sampling.py already closes them
via MC + KIC, and any closure done here would be immediately overwritten.

YAML addition per rigid segment (any not listed is left alone):

    orient:
      axis: parallel        # or 'perpendicular' -- to the z axis
      head: N               # optional; 'N' or 'C'. Named terminus ends up at
                            # the '+' end of the chosen axis (higher z for
                            # parallel; along the input's xy projection for
                            # perpendicular). Absent = preserve input N->C
                            # direction (old behaviour, sign-blind).
      tilt: 15.0            # optional; degrees the long axis deviates from
                            # the target axis after alignment. 0 = exactly on
                            # axis.
      tilt_azimuth: 0.0     # optional; degrees, which way the body leans.
                            # parallel:      0 -> N->C tips toward +x,
                            #               90 -> toward +y.
                            # perpendicular: 0 -> N->C tips toward +z (out of
                            #                     the membrane plane),
                            #               90 -> leans within the plane.
      translate:            # optional; each key optional. RELATIVE shift in A
        x: 0.0              # from the body's input position; absent = no
        y: 0.0              # shift on that axis. (x: 10 moves +10 A, it does
        z: 0.0              # not move the centroid *to* x = 10.)

Order of operations per body: align to the target axis, apply tilt, apply the
relative translation. Because both rotations are taken about the body's own
input centroid, that ordering is equivalent to any other -- the result is a
single rotation plus a single displacement.

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


def rodrigues(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rotation matrix of `degrees` about `axis` (right-handed)."""
    axis = axis / np.linalg.norm(axis)
    t = np.radians(degrees)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def tilt_rotation(target: np.ndarray, axis_kind: str,
                  tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Rotation tipping `target` by `tilt_deg` toward the azimuth direction.

    Azimuth frames, both measured as a rotation from `u` toward `w`:

    'parallel'      target is +-z, so the frame is the lab xy plane and does
                    not depend on `head`: u = +x, w = +y. Azimuth 0 leans the
                    N->C direction toward +x, 90 toward +y.
    'perpendicular' target lies in xy, so the frame is built on the body:
                    u = +z, w = target x z. Azimuth 0 leans out of the
                    membrane plane toward +z, 90 leans within the plane, to
                    the body's right seen from +z.
    """
    if abs(tilt_deg) < 1e-9:
        return np.eye(3)
    target = target / np.linalg.norm(target)

    if axis_kind == "parallel":
        # Lab frame, deliberately not derived from target: otherwise head: N
        # (target = -z) would mirror the handedness of the azimuth.
        u = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])
        u = ref - float(np.dot(ref, target)) * target
        if np.linalg.norm(u) < 1e-6:  # target happened to be parallel to ref
            ref = np.array([0.0, 1.0, 0.0])
            u = ref - float(np.dot(ref, target)) * target
        u /= np.linalg.norm(u)
        w = np.cross(target, u)
    az = np.radians(azimuth_deg)
    lean = np.cos(az) * u + np.sin(az) * w
    return rodrigues(np.cross(target, lean), tilt_deg)


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
        head = orient.get("head")
        if head is not None and head not in ("N", "C"):
            raise ValueError(
                f"{s['name']}: orient.head must be 'N' or 'C' if given "
                f"(got {head!r})"
            )
        unknown = set(orient) - {"axis", "head", "tilt", "tilt_azimuth",
                                 "translate"}
        if unknown:
            raise ValueError(f"{s['name']}: unknown orient keys {sorted(unknown)}")
        tilt = float(orient.get("tilt") or 0.0)
        if not -90.0 <= tilt <= 90.0:
            raise ValueError(
                f"{s['name']}: orient.tilt must be within +/-90 deg "
                f"(got {tilt}); use head: to flip the body instead")
        azimuth = float(orient.get("tilt_azimuth") or 0.0)
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
            "head": head,
            "tilt": tilt,
            "tilt_azimuth": azimuth,
            "translate": {k: float(v) for k, v in translate.items()},
        })
    return cfg, orientations


# --------------------------------------------------------------------------- #
# Target axis
# --------------------------------------------------------------------------- #

def target_axis(input_axis: np.ndarray, axis_kind: str, head) -> np.ndarray:
    """Unit target direction for the body's N->C axis after rotation.

    axis_kind='parallel'
        head=None  -> sign of input_axis.z is preserved (old behaviour)
        head='N'   -> N-term at +z (target = -z; N->C points down)
        head='C'   -> C-term at +z (target = +z; N->C points up)

    axis_kind='perpendicular'
        Base direction is the input axis projected onto xy (or +x as fallback).
        head=None  -> N->C aligns with +base (old behaviour)
        head='N'   -> N-term at the +base end (target = -base)
        head='C'   -> C-term at the +base end (target = +base)
    """
    if axis_kind == "parallel":
        if head is None:
            return np.array([0.0, 0.0, 1.0 if input_axis[2] >= 0 else -1.0])
        return np.array([0.0, 0.0, -1.0 if head == "N" else 1.0])

    # perpendicular
    xy = np.array([input_axis[0], input_axis[1], 0.0])
    n_xy = float(np.linalg.norm(xy))
    base = np.array([1.0, 0.0, 0.0]) if n_xy < 1e-6 else xy / n_xy
    if head is None:
        return base
    return -base if head == "N" else base


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

        target = target_axis(input_axis, body["axis"], body["head"])
        R_align = shortest_arc_rotation(input_axis, target)
        R_tilt = tilt_rotation(target, body["axis"],
                               body["tilt"], body["tilt_azimuth"])
        R = R_tilt @ R_align
        final_axis = R @ input_axis

        # Both rotations are about the body's own centroid, so it stays put;
        # the translation is a relative shift applied on top.
        delta = np.zeros(3)
        for i, k in enumerate("xyz"):
            if k in body["translate"]:
                delta[i] = body["translate"][k]

        angle_before = float(np.degrees(np.arccos(min(1.0, abs(input_axis[2])))))
        head_note = ""
        if body["head"] is not None:
            end_desc = "+z end" if body["axis"] == "parallel" else "+xy end"
            head_note = f", {body['head']}-term forced to {end_desc}"
        tilt_note = ""
        if abs(body["tilt"]) > 1e-9:
            tilt_note = (f", then tilted {body['tilt']:.1f} deg "
                         f"(azimuth {body['tilt_azimuth']:.1f})")
        if body["axis"] == "parallel":
            axis_desc = (f"long axis {angle_before:5.1f} deg from z, "
                         f"now aligned{head_note}{tilt_note}")
        else:
            axis_desc = (f"long axis {angle_before:5.1f} deg from z (want 90), "
                         f"now in xy plane{head_note}{tilt_note}")

        c_before = [float(x) for x in centroid]
        c_after = [c_before[i] + float(delta[i]) for i in range(3)]
        elong_note = "" if elong >= 1.5 else f"  [WARN: elongation ratio {elong:.2f}, orientation poorly defined]"

        print(f"  {body['name']} ({body['chain']}/{body['start']}-{body['stop']}):")
        print(f"    {axis_desc}{elong_note}")
        print(f"    centroid ({c_before[0]:6.1f},{c_before[1]:6.1f},{c_before[2]:6.1f})"
              f" -> ({c_after[0]:6.1f},{c_after[1]:6.1f},{c_after[2]:6.1f})")

        transform_body(model, body["chain"], body["start"], body["stop"],
                       R, delta, centroid)

        # Deviation from the requested target axis is exactly |tilt|, whatever
        # the azimuth, so this line catches any sign or frame mistake in the
        # tilt. The angle from z is reported too since that is the physically
        # meaningful one for a membrane body.
        from_target = float(np.degrees(np.arccos(
            min(1.0, abs(float(np.dot(final_axis, target)))))))
        from_z = float(np.degrees(np.arccos(
            min(1.0, abs(float(final_axis[2]))))))
        print(f"    after: {from_target:5.1f} deg off target axis "
              f"(requested {abs(body['tilt']):5.1f}), "
              f"{from_z:5.1f} deg from z")

        # Confirm the head/tail landed where requested. This is the line that
        # would have caught the F3-TM overlap bug: if `head` is set and the
        # terminus is not on the correct side of the centroid along the axis,
        # something is off (usually a degenerate or nearly-degenerate PCA).
        n_ca = model[body["chain"]][body["start"]]["CA"].get_coord()
        c_ca = model[body["chain"]][body["stop"]]["CA"].get_coord()
        if body["axis"] == "parallel":
            print(f"    after: N-term CA z={float(n_ca[2]):+7.2f}, "
                  f"C-term CA z={float(c_ca[2]):+7.2f}")
        else:
            # Project each terminus onto the target xy direction and report
            # the signed offset from the centroid.
            n_off = float(np.dot(np.asarray(n_ca) - np.asarray(c_after), target))
            c_off = float(np.dot(np.asarray(c_ca) - np.asarray(c_after), target))
            print(f"    after: N-term along +axis={n_off:+7.2f}, "
                  f"C-term along +axis={c_off:+7.2f}")

        if body["head"] is not None:
            # Sanity check on the request
            if body["axis"] == "parallel":
                ok = (float(n_ca[2]) > float(c_ca[2])) if body["head"] == "N" \
                    else (float(c_ca[2]) > float(n_ca[2]))
            else:
                n_off = float(np.dot(np.asarray(n_ca) - np.asarray(c_after), target))
                c_off = float(np.dot(np.asarray(c_ca) - np.asarray(c_after), target))
                ok = (n_off > c_off) if body["head"] == "N" else (c_off > n_off)
            if not ok:
                print(f"    WARN: {body['head']}-term did not end up on the "
                      "requested end. Check the elongation ratio above; a "
                      "poorly-defined long axis can make head enforcement "
                      "unreliable.")

    report_gaps(model, cfg)

    io = writer_for(args.out)
    io.set_structure(struct)
    io.save(args.out)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())