#!/usr/bin/env python3
"""
Visualise and analyse the ensemble in a silent file written by
ensemble_sampling.py, without handing PyMOL 10,000 full-atom copies of the
same anchor domain.

FnIII-3 is identical in every structure - it is the FoldTree root and never
moves - so it is shipped once, and only the mobile part is shipped per model.

The ensemble has two levels of structure, and the analysis follows them:

    placement   one rigid-body draw for the mobile segment(s).  Everything
                that is a property of the placement alone - the drawn DOF
                (dx, dy, dz, tilt, spin) and the loop gap it opens up - is
                constant across its trajectories.
    trajectory  one independent MC closure run started from that placement.
                Whether the loop actually closed is a property of the
                trajectory, not of the placement.

So "is this placement reachable?" is not a yes/no read off one structure; it
is the fraction of independent trajectories from that placement that closed.
That fraction is the reachability probability, and it is what the final plot
colours by.

Written into --out:

  ref.pdb                       one full-atom structure, for context
  view.pml                      every selected model                 (as before)
  reachable.pml                 only the trajectories that closed
  placements.pml                one representative trajectory per placement
  reachable_placements.pml      the same, restricted to reachable placements
  placements.csv                per-placement table: DOF, gap, n_reach, p
  reachability.png              p over placement (dx, dy, dz), plus p vs gap
  summary.png                   score-table plots (--png)

Each .pml has its own <name>_ensemble.pdb / <name>_axes.pdb beside it, except
view.pml which keeps the original ensemble.pdb / axes.pdb names.

    ./visualise_ensemble.py --silent ensemble/ensemble.out \
        --metrics ensemble/metrics.json --config tagged.yaml \
        --reach-metric chainbreak --reach-max 0.1 \
        --out viz/ --png

    pymol viz/reachable_placements.pml

Reachability is computed from --metrics when given, because metrics.json
records every trajectory that was run, including any that passes_filters()
rejected and that therefore never reached the silent file. Using the silent
file alone would silently shrink the denominator and inflate p.

Score columns available to --filter / --sort-by / --color-by / --reach-metric
are whatever passes_filters() and the placement DOF put there: total_score,
chainbreak, gap_<loop>, tilt_<seg>, z_<seg>, <seg>_dx/_dy/_dz/_tilt/_spin,
placement, traj, snapshot. Run with --list-columns to see them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml


# --------------------------------------------------------------------------- #
# 1. The SCORE: table  (plain text - no PyRosetta needed for any of this)
# --------------------------------------------------------------------------- #

def _looks_like_header(fields: List[str]) -> bool:
    """
    A header row has a non-numeric first column and ends in 'description'.
    Both conditions, because a data row also ends in 'description' (its tag
    goes there) and stray text lines also start non-numeric.
    """
    if fields[-1] != "description":
        return False
    try:
        float(fields[0])
        return False
    except ValueError:
        return True


def read_score_table(path: str) -> Tuple[List[str], List[dict]]:
    """
    Parse the SCORE: lines out of a silent file.

    Silent files are appended to, so a header can appear more than once (and,
    if columns ever changed between runs, differ). Rows are matched against the
    most recent header and dropped if the width does not match.
    """
    header: Optional[List[str]] = None
    rows: List[dict] = []
    dropped = 0

    with open(path) as fh:
        for line in fh:
            if not line.startswith("SCORE:"):
                continue
            fields = line.split()[1:]
            if not fields:
                continue
            if _looks_like_header(fields):
                header = fields
                continue
            if header is None or len(fields) != len(header):
                dropped += 1
                continue
            row = {}
            for key, val in zip(header, fields):
                try:
                    row[key] = float(val)
                except ValueError:
                    row[key] = val
            rows.append(row)

    if dropped:
        print(f"  [scores] skipped {dropped} malformed/mismatched SCORE lines")
    if header is None:
        raise SystemExit(f"No SCORE: header found in {path} - is it a silent file?")
    return header, rows


def read_metrics(path: str) -> Tuple[dict, List[dict]]:
    """
    Read metrics.json into the same row shape as the score table.

    The only reshaping is 'tag' -> 'description', so that a metrics record and
    a SCORE: row can be handed to the same functions.
    """
    with open(path) as fh:
        blob = json.load(fh)
    records = []
    for rec in blob.get("models", []):
        row = dict(rec)
        row["description"] = row.pop("tag")
        records.append(row)
    if not records:
        raise SystemExit(f"{path} contains no models")
    return blob, records


def select_rows(rows: List[dict], expr: Optional[str], sort_by: Optional[str],
                top: Optional[int], stride: int, max_models: Optional[int],
                label: str = "select") -> List[dict]:
    """Filter -> sort -> take top -> stride -> thin evenly to max_models."""
    n0 = len(rows)

    if expr:
        try:
            rows = [r for r in rows if eval(expr, {"__builtins__": {}}, r)]
        except NameError as exc:
            raise SystemExit(f"--filter refers to an unknown column: {exc}")
        print(f"  [{label}] filter '{expr}': {len(rows)}/{n0} kept")

    if sort_by:
        if rows and sort_by not in rows[0]:
            raise SystemExit(f"--sort-by '{sort_by}' is not a column in this file")
        rows = sorted(rows, key=lambda r: r[sort_by])

    if top:
        rows = rows[:top]
    if stride > 1:
        rows = rows[::stride]

    # Thin evenly rather than truncating, so a sorted list still spans its range
    # and an unsorted one still spans the whole run.
    if max_models and len(rows) > max_models:
        step = len(rows) / max_models
        rows = [rows[int(i * step)] for i in range(max_models)]
        print(f"  [{label}] thinned to {len(rows)} models (--max-models)")

    return rows


# --------------------------------------------------------------------------- #
# 2. Reachability: trajectories -> placements
# --------------------------------------------------------------------------- #

class Placement:
    """
    One rigid-body draw, summarised over its trajectories.

    n_traj    trajectories run from this placement
    n_reach   of those, how many closed (metric <= threshold)
    p         n_reach / n_traj - the reachability probability
    rep       the representative trajectory row: the best-closing one that
              actually has a structure in the silent file, or None if none of
              them do
    ref       any row from the placement, for reading placement-level columns
              (the drawn DOF and the loop gap, which do not vary within it)
    """

    __slots__ = ("idx", "n_traj", "n_reach", "p", "rep", "ref")

    def __init__(self, idx, n_traj, n_reach, rep, ref):
        self.idx = idx
        self.n_traj = n_traj
        self.n_reach = n_reach
        self.p = n_reach / n_traj if n_traj else 0.0
        self.rep = rep
        self.ref = ref


def collapse_trajectories(rows: List[dict], metric: str) -> Dict[Tuple[int, int], dict]:
    """
    One row per (placement, traj): the snapshot with the lowest metric.

    In recover_low mode there is one snapshot per trajectory and this is a
    no-op. In Boltzmann mode it takes the best-closed snapshot as the
    trajectory's verdict, which is the same convention recover_low applies
    internally.
    """
    for key in ("placement", "traj", metric):
        if rows and key not in rows[0]:
            raise SystemExit(f"Column '{key}' is missing - "
                             "was this file written by ensemble_sampling.py?")
    best: Dict[Tuple[int, int], dict] = {}
    for r in rows:
        k = (int(r["placement"]), int(r["traj"]))
        if k not in best or r[metric] < best[k][metric]:
            best[k] = r
    return best


def summarise_placements(rows: List[dict], metric: str, threshold: float,
                         available: Optional[set]) -> List[Placement]:
    """
    Collapse every trajectory to its best snapshot, then every placement to a
    count of how many of its trajectories closed.

    `available` is the set of tags that exist as structures. It constrains
    which trajectory can be the drawable representative; it does NOT constrain
    the count, because a trajectory that was run and rejected still happened.
    """
    best = collapse_trajectories(rows, metric)

    by_placement: Dict[int, List[dict]] = defaultdict(list)
    for (p_idx, _), row in best.items():
        by_placement[p_idx].append(row)

    out: List[Placement] = []
    for p_idx in sorted(by_placement):
        trajs = by_placement[p_idx]
        n_reach = sum(1 for r in trajs if r[metric] <= threshold)
        drawable = [r for r in trajs
                    if available is None or r["description"] in available]
        rep = min(drawable, key=lambda r: r[metric]) if drawable else None
        out.append(Placement(p_idx, len(trajs), n_reach, rep, trajs[0]))
    return out


def find_dof_prefix(row: dict, wanted: Optional[str]) -> str:
    """
    Which rigid segment's translation to use as the placement's coordinates.

    Auto-detected from the <seg>_dx/_dy/_dz columns when there is exactly one
    mobile segment, which is the common case (fn3 anchored, tm free).
    """
    prefixes = sorted({k[:-3] for k in row if k.endswith("_dx")})
    if wanted:
        if wanted not in prefixes:
            raise SystemExit(f"--dof-segment '{wanted}' has no _dx/_dy/_dz "
                             f"columns; available: {', '.join(prefixes) or 'none'}")
        return wanted
    if not prefixes:
        raise SystemExit("No <seg>_dx columns in the score table - nothing to "
                         "plot placements against")
    if len(prefixes) > 1:
        raise SystemExit(f"Several mobile segments ({', '.join(prefixes)}); "
                         "pick one with --dof-segment")
    return prefixes[0]


def placement_xyz(pl: Placement, prefix: str) -> Tuple[float, float, float]:
    r = pl.ref
    return (float(r.get(f"{prefix}_dx", 0.0)),
            float(r.get(f"{prefix}_dy", 0.0)),
            float(r.get(f"{prefix}_dz", 0.0)))


# --------------------------------------------------------------------------- #
# 3. Decoding structures out of the silent file
# --------------------------------------------------------------------------- #

def open_silent(path: str):
    from pyrosetta.rosetta.core.io.silent import SilentFileData
    try:
        from pyrosetta.rosetta.core.io.silent import SilentFileOptions
        sfd = SilentFileData(SilentFileOptions())
    except ImportError:                       # older builds took no arguments
        sfd = SilentFileData()
    sfd.read_file(path)
    return sfd


def fill(sfd, tag: str):
    """Decode one tag into a pose."""
    import pyrosetta
    pose = pyrosetta.Pose()
    ss = sfd.get_structure(tag)
    try:
        ss.fill_pose(pose)
    except Exception:                                              # noqa: BLE001
        from pyrosetta.rosetta.core.chemical import ChemicalManager
        rts = ChemicalManager.get_instance().residue_type_set("fa_standard")
        ss.fill_pose(pose, rts)
    return pose


def anchor_rmsd(pose, ref, lo: int, hi: int) -> float:
    """CA RMSD over the anchor, without superposition - a drift check."""
    tot = 0.0
    for i in range(lo, hi + 1):
        d = pose.residue(i).xyz("CA") - ref.residue(i).xyz("CA")
        tot += d.length_squared()
    return math.sqrt(tot / (hi - lo + 1))


def superimpose_on_anchor(pose, ref, lo: int, hi: int) -> None:
    from pyrosetta.rosetta.core.id import AtomID, AtomID_Map_AtomID
    from pyrosetta.rosetta.core.pose import initialize_atomid_map
    from pyrosetta.rosetta.core.scoring import superimpose_pose

    amap = AtomID_Map_AtomID()
    initialize_atomid_map(amap, pose, AtomID.BOGUS_ATOM_ID())
    for i in range(lo, hi + 1):
        ca = pose.residue(i).atom_index("CA")
        amap.set(AtomID(ca, i), AtomID(ref.residue(i).atom_index("CA"), i))
    superimpose_pose(pose, ref, amap)


# --------------------------------------------------------------------------- #
# 4. Which residues are worth drawing  (from the same YAML the sampler used)
# --------------------------------------------------------------------------- #

class Layout:
    """
    anchor      : (start, stop) of the root rigid segment - drawn once
    mobile      : (start, stop) spanning everything from the first flexible
                  segment to the end - drawn per model
    axis_segs   : rigid segments carrying a dof block, drawn as axis lines
    All in PDB numbering as written in the YAML; converted to pose numbering
    against the first decoded pose.
    """

    def __init__(self, segments: List[dict]):
        rigid = [s for s in segments if s["kind"] == "rigid"]
        flex = [s for s in segments if s["kind"] == "flexible"]
        if not rigid or not flex:
            raise SystemExit("Config needs at least one rigid and one flexible segment")

        self.chain = str(segments[0].get("chain", "A"))
        self.anchor = (int(rigid[0]["start"]), int(rigid[0]["stop"]))
        self.mobile = (int(flex[0]["start"]), int(segments[-1]["stop"]))
        self.axis_segs = [(s["name"], int(s["start"]), int(s["stop"]))
                          for s in rigid if s.get("dof")]


def load_layout(path: str) -> Layout:
    with open(path) as fh:
        return Layout(yaml.safe_load(fh)["segments"])


def resolve_numbering(pose, lo: int, hi: int, chain: str, label: str,
                      warned: List[bool]) -> Tuple[int, int]:
    """
    Map PDB numbering onto pose numbering.

    Silent files do not always carry PDBInfo through intact. If the lookup
    fails, fall back to reading the YAML numbers as pose numbers, which is
    correct whenever the input PDB was numbered from 1, and say so loudly
    rather than silently drawing the wrong residues.
    """
    info = pose.pdb_info()
    if info is not None:
        a, b = info.pdb2pose(chain, lo), info.pdb2pose(chain, hi)
        if a and b:
            return a, b
    if not warned:
        print("  [numbering] WARNING: PDBInfo missing or does not cover "
              f"{label} {chain}{lo}-{hi}; treating YAML numbers as pose numbers. "
              "Check ref.pdb covers the residues you expect.")
        warned.append(True)
    if hi > pose.total_residue():
        raise SystemExit(f"{label}: residue {hi} is past the end of the pose "
                         f"({pose.total_residue()} residues)")
    return lo, hi


# --------------------------------------------------------------------------- #
# 5. Writing the light-weight PDBs by hand
# --------------------------------------------------------------------------- #

def pdb_atom(serial: int, name: str, resname: str, chain: str, resnum: int,
             x: float, y: float, z: float, bfac: float,
             element: str = "C", het: bool = False) -> str:
    rec = "HETATM" if het else "ATOM  "
    nm = name if len(name) >= 4 else " " + name.ljust(3)
    return (f"{rec}{serial:5d} {nm:4s} {resname:>3s} {chain:1s}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{bfac:6.2f}          {element:>2s}")


def rescale(values: Sequence[float]) -> Tuple[List[float], float, float]:
    """
    Map a metric onto 0-100 for the B-factor column.

    Not cosmetic: total_score runs to four figures and would overflow the
    6.2f B-factor field, silently corrupting the coordinate lines that follow.
    `spectrum b` in PyMOL only cares about relative values anyway.
    """
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [50.0] * len(values), lo, hi
    return [100.0 * (v - lo) / (hi - lo) for v in values], lo, hi


def write_ca_ensemble(path: Path, poses, bvals: List[float],
                      lo: int, hi: int, chain: str) -> int:
    """CA trace of residues lo-hi, one MODEL per pose."""
    natoms = 0
    with open(path, "w") as fh:
        for m, (pose, b) in enumerate(zip(poses, bvals), start=1):
            fh.write(f"MODEL     {m:4d}\n")
            info = pose.pdb_info()
            for serial, i in enumerate(range(lo, hi + 1), start=1):
                res = pose.residue(i)
                if not res.has("CA"):
                    continue
                v = res.xyz("CA")
                num = info.number(i) if info is not None else i
                ch = (info.chain(i) if info is not None else chain) or chain
                fh.write(pdb_atom(serial, "CA", res.name3()[:3], ch, num,
                                  v.x, v.y, v.z, b, "C") + "\n")
                natoms += 1
            fh.write("TER\nENDMDL\n")
    return natoms


def write_axes(path: Path, poses, bvals: List[float],
               segs: List[Tuple[str, int, int]]) -> int:
    """
    One 2-atom line per mobile rigid segment per model, all in ONE object.

    Deliberately not multi-state: the whole ensemble of helix axes is visible
    at once with no state juggling, which is the view that actually answers
    "where can the TM helix end up".
    """
    if 2 * len(poses) * len(segs) > 99999:
        raise SystemExit("Too many axis atoms for the 5-digit PDB serial field - "
                         "lower --max-models")

    atoms: List[str] = []
    conect: List[str] = []      # kept separate: CONECT belongs after the coordinates
    serial = 0
    resnum = 0
    for pose, b in zip(poses, bvals):
        for k, (name, lo, hi) in enumerate(segs):
            a, c = pose.residue(lo).xyz("CA"), pose.residue(hi).xyz("CA")
            resnum += 1
            rn = (resnum - 1) % 9999 + 1
            ch = chr(ord("A") + (k % 26))
            serial += 1
            i0 = serial
            atoms.append(pdb_atom(serial, "N1", "AXS", ch, rn,
                                  a.x, a.y, a.z, b, "N", het=True))
            serial += 1
            atoms.append(pdb_atom(serial, "C1", "AXS", ch, rn,
                                  c.x, c.y, c.z, b, "C", het=True))
            conect.append(f"CONECT{i0:5d}{serial:5d}")
    with open(path, "w") as fh:
        fh.write("\n".join(atoms + conect) + "\nEND\n")
    return serial


# --------------------------------------------------------------------------- #
# 6. The PyMOL scripts
# --------------------------------------------------------------------------- #

PML = """\
# Generated by analyse_ensemble.py
# {title}
# {n} models. B-factors carry {colour_by}, rescaled from
# [{cmin:.4g}, {cmax:.4g}] onto 0-100.

# These three must be set BEFORE loading, and are the difference between a
# responsive session and a hung one on a large multi-state file.
set defer_builds_mode, 3
set async_builds, 1
set ribbon_sampling, 1

# CA-only traces need this or ribbon/cartoon draws nothing at all.
set ribbon_trace_atoms, 1
set cartoon_trace_atoms, 1

load {ref}, anchor
load {ens}, {ens_obj}
{axes_load}
hide everything

show cartoon, anchor
color grey70, anchor

show ribbon, {ens_obj}
set all_states, 1, {ens_obj}
spectrum b, blue_white_red, {ens_obj}, minimum=0, maximum=100
set ribbon_width, 2

{axes_show}
{membrane}
bg_color white
set ray_opaque_background, 0
orient anchor
zoom all, 5

# All {n} states are drawn at once. To step through them instead:
#   set all_states, 0, {ens_obj}
#   mset 1 -{n}
#   mplay
"""

MEMBRANE_CGO = """
python
from pymol.cgo import BEGIN, END, TRIANGLE_STRIP, COLOR, VERTEX
from pymol import cmd
_h = {half:.1f}
_r = {extent:.1f}
_obj = []
for _z in (-_h, _h):
    _obj += [BEGIN, TRIANGLE_STRIP, COLOR, 0.85, 0.85, 0.60,
             VERTEX, -_r, -_r, _z, VERTEX, -_r, _r, _z,
             VERTEX,  _r, -_r, _z, VERTEX,  _r, _r, _z, END]
cmd.load_cgo(_obj, "membrane")
cmd.set("cgo_transparency", 0.7, "membrane")
python end
"""


def view_filenames(name: str) -> Tuple[str, str, str]:
    """view.pml keeps the original filenames; everything else is prefixed."""
    if name == "view":
        return "view.pml", "ensemble.pdb", "axes.pdb"
    return f"{name}.pml", f"{name}_ensemble.pdb", f"{name}_axes.pdb"


def write_view(outdir: Path, name: str, title: str, poses, bvals: List[float],
               m_lo: int, m_hi: int, chain: str,
               axis_segs: List[Tuple[str, int, int]],
               colour_by: str, cmin: float, cmax: float,
               slab: Optional[float], extent: float) -> None:
    """One .pml plus the light-weight PDBs it loads."""
    pml_name, ens_name, axes_name = view_filenames(name)
    obj = "ens_" + ("all" if name == "view" else name)

    natoms = write_ca_ensemble(outdir / ens_name, poses, bvals, m_lo, m_hi, chain)

    have_axes = bool(axis_segs)
    if have_axes:
        write_axes(outdir / axes_name, poses, bvals, axis_segs)

    axes_obj = "axes_" + ("all" if name == "view" else name)
    axes_load = f"load {axes_name}, {axes_obj}" if have_axes else ""
    axes_show = (f"show sticks, {axes_obj}\nset stick_radius, 0.25, {axes_obj}\n"
                 f"spectrum b, blue_white_red, {axes_obj}, minimum=0, maximum=100\n"
                 ) if have_axes else ""
    membrane = MEMBRANE_CGO.format(half=slab / 2.0, extent=extent) if slab else ""

    (outdir / pml_name).write_text(
        PML.format(n=len(poses), title=title, colour_by=colour_by,
                   cmin=cmin, cmax=cmax, ref="ref.pdb", ens=ens_name,
                   ens_obj=obj, axes_load=axes_load, axes_show=axes_show,
                   membrane=membrane))

    print(f"  [write] {pml_name}: {len(poses)} models, {natoms} CA atoms"
          f"{', axes' if have_axes else ''}")


# --------------------------------------------------------------------------- #
# 7. Per-placement table and the reachability plot
# --------------------------------------------------------------------------- #

def write_placement_csv(path: Path, placements: List[Placement], prefix: str,
                        metric: str, threshold: float) -> None:
    dof_cols = [c for c in (f"{prefix}_dx", f"{prefix}_dy", f"{prefix}_dz",
                            f"{prefix}_tilt", f"{prefix}_tilt_azimuth",
                            f"{prefix}_spin")
                if c in placements[0].ref]
    gap_cols = sorted(c for c in placements[0].ref if c.startswith("gap_"))

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["placement", "n_traj", f"n_reach({metric}<={threshold:g})",
                    "p_reach", f"best_{metric}", "representative"]
                   + dof_cols + gap_cols)
        for pl in placements:
            best = pl.rep[metric] if pl.rep is not None else ""
            w.writerow([pl.idx, pl.n_traj, pl.n_reach, f"{pl.p:.3f}",
                        f"{best:.4f}" if best != "" else "",
                        pl.rep["description"] if pl.rep is not None else ""]
                       + [pl.ref.get(c, "") for c in dof_cols]
                       + [pl.ref.get(c, "") for c in gap_cols])
    print(f"  [write] {path.name}: {len(placements)} placements")


def write_reachability_png(path: Path, placements: List[Placement], prefix: str,
                           metric: str, threshold: float) -> None:
    """
    Left: every placement at its drawn displacement (dx, dy, dz), coloured by
    the fraction of its trajectories that closed.

    Right: the same p against the loop gap the placement opens. The gap is
    fixed by the placement and is the obvious geometric predictor of closure,
    so this panel is the sanity check on the left one - if p is simply a
    function of gap, the 3D structure is telling you nothing extra.

    Note that tilt and spin are collapsed here: two placements can sit at the
    same (dx, dy, dz) with different helix orientations.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
    except ImportError:
        print("  [png] matplotlib not available - skipping reachability.png")
        return

    xs, ys, zs, ps = [], [], [], []
    for pl in placements:
        x, y, z = placement_xyz(pl, prefix)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ps.append(pl.p)

    gap_cols = sorted(c for c in placements[0].ref if c.startswith("gap_"))

    fig = plt.figure(figsize=(12, 5.5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=ps, cmap="viridis", vmin=0.0, vmax=1.0,
                    s=70, edgecolors="black", linewidths=0.4, depthshade=False)
    for pl, x, y, z in zip(placements, xs, ys, zs):
        ax.text(x, y, z, f" {pl.idx}", fontsize=6, color="0.3")
    ax.set_xlabel(f"{prefix} dx (A)")
    ax.set_ylabel(f"{prefix} dy (A)")
    ax.set_zlabel(f"{prefix} dz (A)")
    ax.set_title(f"placement reachability\n({metric} <= {threshold:g})", fontsize=10)
    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.12)
    cb.set_label("fraction of trajectories that closed")

    ax2 = fig.add_subplot(1, 2, 2)
    if gap_cols:
        col = gap_cols[0]
        gaps = [pl.ref.get(col, float("nan")) for pl in placements]
        ax2.scatter(gaps, ps, c=ps, cmap="viridis", vmin=0.0, vmax=1.0,
                    s=70, edgecolors="black", linewidths=0.4)
        ax2.set_xlabel(f"{col} (A)")
    else:
        ax2.scatter(range(len(ps)), ps, s=70)
        ax2.set_xlabel("placement")
    ax2.set_ylabel("p(reach)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)
    ax2.set_title("is reachability just distance?", fontsize=10)

    n_traj = placements[0].n_traj if placements else 0
    fig.suptitle(f"{len(placements)} placements x ~{n_traj} trajectories")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  [png] {path.name}")


def write_png(path: Path, rows: List[dict], all_rows: List[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [png] matplotlib not available - skipping")
        return

    tilt_cols = sorted(k for k in rows[0] if k.startswith("tilt_"))
    z_cols = sorted(k for k in rows[0] if k.startswith("z_"))
    gap_cols = sorted(k for k in rows[0] if k.startswith("gap_"))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    for c in tilt_cols:
        ax[0][0].hist([r[c] for r in all_rows], bins=40, alpha=0.5, label=c)
    ax[0][0].set_xlabel("tilt vs membrane normal (deg)")
    ax[0][0].set_ylabel("count")
    if tilt_cols:
        ax[0][0].legend(fontsize=7)

    if "chainbreak" in rows[0]:
        ax[0][1].hist([r["chainbreak"] for r in all_rows], bins=40, color="firebrick")
        ax[0][1].set_xlabel("chainbreak")
        ax[0][1].set_ylabel("count")
        ax[0][1].set_yscale("log")

    if tilt_cols and "total_score" in rows[0]:
        c = tilt_cols[-1]
        ax[1][0].scatter([r[c] for r in all_rows],
                         [r["total_score"] for r in all_rows],
                         s=4, alpha=0.3, color="grey", label="all")
        ax[1][0].scatter([r[c] for r in rows], [r["total_score"] for r in rows],
                         s=8, alpha=0.8, color="tab:blue", label="selected")
        ax[1][0].set_xlabel(c)
        ax[1][0].set_ylabel("total_score")
        ax[1][0].legend(fontsize=7)

    for c in gap_cols + z_cols:
        ax[1][1].hist([r[c] for r in all_rows], bins=40, alpha=0.5, label=c)
    ax[1][1].set_xlabel("loop gap / segment z (A)")
    ax[1][1].set_ylabel("count")
    if gap_cols or z_cols:
        ax[1][1].legend(fontsize=7)

    fig.suptitle(f"{len(all_rows)} structures, {len(rows)} selected for viewing")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  [png] {path.name}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--silent", required=True, help="Silent file from ensemble_sampling.py")
    p.add_argument("--metrics", help="metrics.json from the same run. Used for the "
                                     "reachability counts, because it also records "
                                     "trajectories that were filtered out and so "
                                     "never made it into the silent file.")
    p.add_argument("--config", help="The same construct YAML used for sampling. "
                                    "Without it, the whole chain is drawn per model "
                                    "and no axes are produced.")
    p.add_argument("--out", default="viz", help="Output directory")

    p.add_argument("--reach-metric", default="chainbreak",
                   help="Score column that decides whether a trajectory closed")
    p.add_argument("--reach-max", type=float, default=0.1,
                   help="A trajectory is reachable if its --reach-metric is at "
                        "or below this. Look at the chainbreak histogram in "
                        "summary.png before trusting the default.")
    p.add_argument("--min-reach-prob", type=float, default=0.0,
                   help="A placement counts as reachable if strictly more than "
                        "this fraction of its trajectories closed (default 0.0: "
                        "at least one)")
    p.add_argument("--dof-segment", help="Which rigid segment's dx/dy/dz to use "
                                         "as the placement coordinates "
                                         "(auto-detected if there is only one)")

    p.add_argument("--filter", help="Python expression over score columns, "
                                    "e.g. 'chainbreak < 5 and tilt_tm < 40'")
    p.add_argument("--sort-by", help="Score column to sort ascending by")
    p.add_argument("--top", type=int, help="Keep the first N after sorting")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-models", type=int, default=500,
                   help="Hard cap per view, thinned evenly (default 500 - a CA "
                        "trace of ~40 residues x 500 states is ~20k atoms, which "
                        "PyMOL handles without complaint)")

    p.add_argument("--color-by", default="total_score",
                   help="Score column written into the B-factor column of the "
                        "per-trajectory views. The per-placement views always "
                        "carry the reachability probability instead.")
    p.add_argument("--align", action="store_true",
                   help="Superimpose on the anchor before writing. Should be "
                        "unnecessary - the anchor is the FoldTree root - so the "
                        "drift check is on by default and this is the fix.")
    p.add_argument("--slab", type=float, default=30.0,
                   help="Membrane slab thickness drawn at z=0 (0 to omit)")
    p.add_argument("--slab-extent", type=float, default=60.0)
    p.add_argument("--png", action="store_true", help="Also write summary.png")
    p.add_argument("--list-columns", action="store_true",
                   help="Print the score columns and exit (no PyRosetta needed)")
    return p.parse_args()


def main():
    args = parse_args()

    header, silent_rows = read_score_table(args.silent)
    print(f"  [scores] {len(silent_rows)} structures, {len(header)} columns")

    if args.list_columns:
        for c in header:
            print("   ", c)
        return

    if not silent_rows:
        raise SystemExit("No structures in the score table")

    # ---- reachability is counted over every trajectory that was run --------- #
    if args.metrics:
        blob, stat_rows = read_metrics(args.metrics)
        n_rejected = sum(1 for r in stat_rows if not r.get("kept", True))
        print(f"  [metrics] {len(stat_rows)} trajectories recorded "
              f"({n_rejected} rejected by passes_filters, "
              f"mode={blob.get('mode')})")
    else:
        stat_rows = silent_rows
        print("  [metrics] no --metrics: counting reachability over the silent "
              "file only, which ignores any trajectory that was filtered out")

    metric, thresh = args.reach_metric, args.reach_max
    if metric not in stat_rows[0]:
        raise SystemExit(f"--reach-metric '{metric}' is not a column")

    tags_present = {r["description"] for r in silent_rows}
    placements = summarise_placements(stat_rows, metric, thresh, tags_present)
    prefix = find_dof_prefix(placements[0].ref, args.dof_segment)

    n_reach_pl = sum(1 for pl in placements if pl.p > args.min_reach_prob)
    n_traj_tot = sum(pl.n_traj for pl in placements)
    n_traj_reach = sum(pl.n_reach for pl in placements)
    print(f"  [reach] {metric} <= {thresh:g}: "
          f"{n_traj_reach}/{n_traj_tot} trajectories, "
          f"{n_reach_pl}/{len(placements)} placements "
          f"(p > {args.min_reach_prob:g})")
    print(f"  [reach] placement DOF read from '{prefix}_dx/_dy/_dz'")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    write_placement_csv(outdir / "placements.csv", placements, prefix,
                        metric, thresh)
    write_reachability_png(outdir / "reachability.png", placements, prefix,
                           metric, thresh)

    # ---- the four sets of rows to draw -------------------------------------- #
    view_rows = select_rows(silent_rows, args.filter, args.sort_by,
                            args.top, args.stride, args.max_models, "view")
    if not view_rows:
        raise SystemExit("Selection is empty - loosen --filter")

    reach_rows = select_rows([r for r in silent_rows if r[metric] <= thresh],
                             None, args.sort_by, None, 1, args.max_models,
                             "reachable")

    rep_rows = [pl.rep for pl in placements if pl.rep is not None]
    rep_probs = [pl.p for pl in placements if pl.rep is not None]
    reach_rep = [(pl.rep, pl.p) for pl in placements
                 if pl.rep is not None and pl.p > args.min_reach_prob]

    n_missing = sum(1 for pl in placements if pl.rep is None)
    if n_missing:
        print(f"  [reach] WARNING: {n_missing} placements have no structure in "
              "the silent file and are absent from the per-placement views")

    if args.png:
        write_png(outdir / "summary.png", view_rows, silent_rows)

    # ---- decode every tag any view needs, once ------------------------------ #
    import pyrosetta
    pyrosetta.init("-mute all")

    print(f"  [silent] reading {args.silent} ...")
    sfd = open_silent(args.silent)

    wanted, seen = [], set()
    for r in view_rows + reach_rows + rep_rows:
        tag = r["description"]
        if tag not in seen:
            seen.add(tag)
            wanted.append(tag)

    missing = [t for t in wanted if not sfd.has_tag(t)]
    if missing:
        raise SystemExit(f"{len(missing)} tags in the score table are not in the "
                         f"structure body (first: {missing[0]}) - truncated file?")

    poses = {t: fill(sfd, t) for t in wanted}
    ref = poses[view_rows[0]["description"]]
    print(f"  [silent] decoded {len(poses)} structures "
          f"({ref.total_residue()} residues each)")

    # ---- residue ranges ----------------------------------------------------- #
    warned: List[bool] = []
    if args.config:
        layout = load_layout(args.config)
        chain = layout.chain
        a_lo, a_hi = resolve_numbering(ref, *layout.anchor, chain, "anchor", warned)
        m_lo, m_hi = resolve_numbering(ref, *layout.mobile, chain, "mobile", warned)
        axis_segs = [(n,) + resolve_numbering(ref, lo, hi, chain, n, warned)
                     for n, lo, hi in layout.axis_segs]
    else:
        chain = "A"
        a_lo = a_hi = 0
        m_lo, m_hi = 1, ref.total_residue()
        axis_segs = []
        print("  [layout] no --config: drawing every residue of every model and "
              "no axes")

    if a_hi:
        drift = max(anchor_rmsd(p, ref, a_lo, a_hi) for p in poses.values())
        print(f"  [frame] max anchor CA RMSD to the reference: {drift:.3f} A")
        if args.align:
            for t, p in poses.items():
                if p is not ref:
                    superimpose_on_anchor(p, ref, a_lo, a_hi)
            print("  [frame] superimposed on the anchor")
        elif drift > 0.5:
            print("  [frame] WARNING: the anchor is not sitting still across "
                  "models, so the spread you are about to look at is partly "
                  "frame drift. Re-run with --align.")

    ref.dump_pdb(str(outdir / "ref.pdb"))

    # ---- the views ---------------------------------------------------------- #
    def by_score(rows: List[dict]) -> Tuple[List, List[float], str, float, float]:
        """B-factors from --color-by, for the per-trajectory views."""
        ps = [poses[r["description"]] for r in rows]
        if args.color_by not in rows[0]:
            return ps, [50.0] * len(rows), "nothing", 0.0, 0.0
        b, lo, hi = rescale([r[args.color_by] for r in rows])
        return ps, b, args.color_by, lo, hi

    common = dict(m_lo=m_lo, m_hi=m_hi, chain=chain, axis_segs=axis_segs,
                  slab=args.slab or None, extent=args.slab_extent)

    p_, b_, cb_, lo_, hi_ = by_score(view_rows)
    write_view(outdir, "view",
               f"every selected trajectory ({len(view_rows)} of {len(silent_rows)})",
               p_, b_, colour_by=cb_, cmin=lo_, cmax=hi_, **common)

    if reach_rows:
        p_, b_, cb_, lo_, hi_ = by_score(reach_rows)
        write_view(outdir, "reachable",
                   f"trajectories with {metric} <= {thresh:g} "
                   f"({len(reach_rows)} of {len(silent_rows)})",
                   p_, b_, colour_by=cb_, cmin=lo_, cmax=hi_, **common)
    else:
        print(f"  [write] no trajectory has {metric} <= {thresh:g} - "
              "reachable.pml not written")

    if rep_rows:
        write_view(outdir, "placements",
                   f"best trajectory per placement, coloured by p(reach) "
                   f"({len(rep_rows)} placements)",
                   [poses[r["description"]] for r in rep_rows],
                   [100.0 * p for p in rep_probs],
                   colour_by="p(reach)", cmin=0.0, cmax=1.0, **common)

    if reach_rep:
        write_view(outdir, "reachable_placements",
                   f"placements with p(reach) > {args.min_reach_prob:g}, "
                   f"coloured by p(reach) ({len(reach_rep)} placements)",
                   [poses[r["description"]] for r, _ in reach_rep],
                   [100.0 * p for _, p in reach_rep],
                   colour_by="p(reach)", cmin=0.0, cmax=1.0, **common)
    else:
        print(f"  [write] no placement has p(reach) > {args.min_reach_prob:g} - "
              "reachable_placements.pml not written")

    print(f"\nDone.  pymol {outdir / 'placements.pml'}")


if __name__ == "__main__":
    main()