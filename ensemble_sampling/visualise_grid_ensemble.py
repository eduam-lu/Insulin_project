#!/usr/bin/env python3
"""
Analyse a grid-mode ensemble sampling run.

Reads a run directory containing `metrics.json` and a Rosetta silent file
(default `ensemble.out`), produces a `viz/` subdirectory with:

  Data files (referenced by the PMLs):
    ref.pdb                     CA trace of the fixed anchor domain
    ensemble_all.pdb            multi-model, up to --ensemble-cap trajectories,
                                B-factor = chainbreak
    ensemble_reachable.pdb      subset with chainbreak < --close-threshold
    placements_all.pdb          best (lowest chainbreak) model per viable cell,
                                B-factor = closure fraction on a fixed 0-1 scale
    placements_reachable.pdb    subset where closure fraction >= --reach-threshold

  PyMOL scripts:
    view.pml, reachable.pml                 trajectories, coloured by chainbreak
    placements.pml, reachable_placements.pml  per-cell, coloured by closure fraction

  Plots:
    reachability.png            closure fraction over the grid (2D heatmap)
    chainbreak.png              mean chainbreak over the grid (log-colour)
    gap_vs_closure.png          pre-MC loop gap vs closure fraction
    summary.png                 2x2 combining the three plus a chainbreak histogram

  Table:
    placements.csv              per-cell aggregate metrics

The two thresholds have distinct roles: --close-threshold is per-model (is the
chainbreak low enough to call the loop closed?); --reach-threshold is per-cell
(is the closure fraction high enough to call the cell reachable?). Both are
rescan knobs -- change them and rerun without re-sampling.

Only supports 2D grids for now; will exit with a message otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pyrosetta
from pyrosetta.rosetta.core.io.silent import SilentFileData, SilentFileOptions


# --------------------------------------------------------------------------- #
# Constants shared with the sampler.
# --------------------------------------------------------------------------- #

MAX_SPAN_PER_RESIDUE = 3.3


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #

def load_metrics(run_dir: Path) -> dict:
    """Read metrics.json."""
    with open(run_dir / "metrics.json") as fh:
        return json.load(fh)


def load_config(run_dir: Path, metrics: dict,
                override: Optional[Path] = None) -> dict:
    """Load the YAML construct definition referenced in metrics.json.

    Tries the exact path stored in metrics.json first (that's what the sampler
    was invoked with), then falls back to the same filename inside `run_dir`
    and its parent -- so a moved run directory still works.
    """
    if override is not None:
        with open(override) as fh:
            return yaml.safe_load(fh)

    cfg_path = Path(metrics["config"])
    candidates = [cfg_path, run_dir / cfg_path.name,
                  run_dir.parent / cfg_path.name]
    for c in candidates:
        if c.is_file():
            with open(c) as fh:
                return yaml.safe_load(fh)
    raise FileNotFoundError(
        f"Cannot locate the YAML config. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nPass --config to point at it explicitly.")


def parse_score_table(silent_path: Path) -> List[dict]:
    """Return one dict per SCORE line, keyed by column name.

    Numeric columns become floats. The final column, `description`, is the
    silent-file tag and is left as a string.
    """
    columns: Optional[List[str]] = None
    rows: List[dict] = []
    with open(silent_path) as fh:
        for line in fh:
            if not line.startswith("SCORE:"):
                continue
            parts = line.split()
            if columns is None:
                columns = parts[1:]                # drop the "SCORE:" tag
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
# 2. Grid geometry
# --------------------------------------------------------------------------- #

def grid_axes(metrics: dict) -> Tuple[str, List[Tuple[str, List[float]]]]:
    """Return (segment_name, [(axis_name, values), ...])."""
    grid = metrics.get("grid") or {}
    if not grid:
        raise SystemExit(
            "metrics.json has no `grid` block. Was this a grid-mode run? "
            "(random-mode analysis is not implemented in this script.)")
    if len(grid) > 1:
        raise SystemExit(
            f"Multiple gridded segments not supported: {sorted(grid)}")
    seg_name = next(iter(grid))
    axes = [(a["axis"], list(a["values"])) for a in grid[seg_name]["axes"]]
    return seg_name, axes


def anchor_info(metrics: dict) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """Return (anchor_segment_name, anchor_position_xyz, fixed_dof_values)."""
    grid = metrics["grid"]
    seg_name = next(iter(grid))
    info = grid[seg_name]
    return info["anchor"], info["anchor_position"], info.get("fixed", {})


# --------------------------------------------------------------------------- #
# 3. Per-cell aggregation
# --------------------------------------------------------------------------- #

def aggregate_cells(metrics: dict, score_rows: List[dict],
                    close_threshold: float) -> Dict[int, dict]:
    """Compute per-cell statistics.

    Denominator for the closure fraction is `n_traj_run` from metrics.json
    (0 for non-viable cells, args.n_traj for viable). Numerator is the count
    of silent-file models per cell whose chainbreak is below the threshold.
    Cells with n_traj_run == 0 get NaN closure fraction (no attempt made).
    """
    seg_name = next(iter(metrics["grid"]))
    dx_key = f"{seg_name}_dx"
    dy_key = f"{seg_name}_dy"
    dz_key = f"{seg_name}_dz"
    xi_key = f"{seg_name}_grid_x_i"
    yi_key = f"{seg_name}_grid_y_i"

    cells: Dict[int, dict] = {}
    for prec in metrics["placements"]:
        p_idx = int(prec["placement"])
        cells[p_idx] = {
            "placement": p_idx,
            "viable": bool(prec["viable"]),
            "n_traj_run": int(prec["n_traj_run"]),
            "dx": float(prec.get(dx_key, 0.0)),
            "dy": float(prec.get(dy_key, 0.0)),
            "dz": float(prec.get(dz_key, 0.0)),
            "grid_x_i": int(prec.get(xi_key, -1)),
            "grid_y_i": int(prec.get(yi_key, -1)),
            "n_kept": 0,
            "n_closed": 0,
            "_chainbreaks": [],
            "best_chainbreak": float("inf"),
            "best_tag": None,
        }

    for row in score_rows:
        p_idx = int(row.get("placement", -1))
        if p_idx not in cells:
            continue
        cell = cells[p_idx]
        cb = float(row.get("chainbreak", float("inf")))
        cell["n_kept"] += 1
        cell["_chainbreaks"].append(cb)
        if cb < close_threshold:
            cell["n_closed"] += 1
        if cb < cell["best_chainbreak"]:
            cell["best_chainbreak"] = cb
            cell["best_tag"] = row.get("description")

    for cell in cells.values():
        n = cell["n_traj_run"]
        cell["closure_fraction"] = (cell["n_closed"] / n) if n > 0 else float("nan")
        cbs = cell["_chainbreaks"]
        cell["mean_chainbreak"] = float(np.mean(cbs)) if cbs else float("nan")
        cell["loop_gap"] = float(np.sqrt(
            cell["dx"]**2 + cell["dy"]**2 + cell["dz"]**2))
        del cell["_chainbreaks"]
        if cell["best_chainbreak"] == float("inf"):
            cell["best_chainbreak"] = float("nan")

    return cells


# --------------------------------------------------------------------------- #
# 4. Segment residue lookups (from the YAML config)
# --------------------------------------------------------------------------- #

def segment_residues(cfg: dict) -> Tuple[List[int], List[int], str]:
    """Return (anchor_residues, mobile_residues, chain).

    Anchor = every rigid segment WITHOUT a `dof` block (they are fixed by
    construction of the FoldTree). Mobile = flexible segments + rigid
    segments WITH a dof block.
    """
    anchor: List[int] = []
    mobile: List[int] = []
    chains = set()

    for seg in cfg["segments"]:
        chains.add(seg.get("chain", "A"))
        residues = list(range(seg["start"], seg["stop"] + 1))
        if seg["kind"] == "flexible":
            mobile.extend(residues)
        elif seg["kind"] == "rigid":
            if seg.get("dof"):
                mobile.extend(residues)
            else:
                anchor.extend(residues)

    if len(chains) != 1:
        # The sampler doesn't handle multi-chain constructs either; we mirror
        # that limitation rather than pretend it works.
        raise SystemExit(f"Multi-chain constructs not supported: chains={chains}")
    chain = next(iter(chains))
    return sorted(anchor), sorted(mobile), chain


# --------------------------------------------------------------------------- #
# 5. Silent file access
# --------------------------------------------------------------------------- #

def pose_ca_by_pdb_num(pose) -> Dict[int, Tuple[float, float, float]]:
    """{pdb_residue_number: (x, y, z)} for every CA in the pose."""
    out: Dict[int, Tuple[float, float, float]] = {}
    pi = pose.pdb_info()
    for i in range(1, pose.total_residue() + 1):
        ca = pose.residue(i).xyz("CA")
        out[int(pi.number(i))] = (float(ca.x), float(ca.y), float(ca.z))
    return out


def load_poses(silent_path: Path, wanted_tags: List[str]
               ) -> Dict[str, Dict[int, Tuple[float, float, float]]]:
    """Decode `wanted_tags` from the silent file into {tag: {pdb_num: xyz}}."""
    if not wanted_tags:
        return {}

    opts = SilentFileOptions()
    sfd = SilentFileData(opts)
    sfd.read_file(str(silent_path))

    wanted = set(wanted_tags)
    poses: Dict[str, Dict[int, Tuple[float, float, float]]] = {}
    for tag in sfd.tags():
        if tag not in wanted:
            continue
        ss = sfd.get_structure(tag)
        pose = pyrosetta.Pose()
        ss.fill_pose(pose)
        poses[tag] = pose_ca_by_pdb_num(pose)
    missing = wanted - set(poses)
    if missing:
        print(f"WARNING: {len(missing)} requested tags not found in silent "
              f"file (first few: {sorted(missing)[:5]})")
    return poses


# --------------------------------------------------------------------------- #
# 6. PDB writing
# --------------------------------------------------------------------------- #

def _atom_line(atom_num: int, chain: str, resnum: int,
               x: float, y: float, z: float, bfactor: float) -> str:
    """One ATOM record, CA of ALA, occupancy 1.00, element C."""
    return (
        f"ATOM  {atom_num:>5d}  CA  ALA {chain}{resnum:>4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C  \n"
    )


def write_single_model_pdb(path: Path,
                           coords: Dict[int, Tuple[float, float, float]],
                           residues: List[int], chain: str,
                           bfactor: float) -> None:
    """One structure, no MODEL/ENDMDL wrapping. For ref.pdb."""
    with open(path, "w") as fh:
        for i, resnum in enumerate(residues, start=1):
            if resnum not in coords:
                continue
            x, y, z = coords[resnum]
            fh.write(_atom_line(i, chain, resnum, x, y, z, bfactor))
        fh.write("END\n")


def write_multi_model_pdb(path: Path,
                          models: List[Tuple[Dict[int, Tuple[float, float, float]], float]],
                          residues: List[int], chain: str) -> int:
    """Write one MODEL block per (coords, bfactor) entry.

    Returns the number of MODEL blocks written (may be smaller than
    len(models) if some entries had no atoms for the requested residues).
    """
    n_written = 0
    with open(path, "w") as fh:
        for coords, bfactor in models:
            atoms = [(r, coords[r]) for r in residues if r in coords]
            if not atoms:
                continue
            n_written += 1
            fh.write(f"MODEL     {n_written:>4d}\n")
            for i, (resnum, (x, y, z)) in enumerate(atoms, start=1):
                fh.write(_atom_line(i, chain, resnum, x, y, z, bfactor))
            fh.write("ENDMDL\n")
        fh.write("END\n")
    return n_written


# --------------------------------------------------------------------------- #
# 7. PyMOL script generation
# --------------------------------------------------------------------------- #

_PML_HEADER = """\
# Auto-generated by analyse_ensemble.py
# {title}

# Multi-state performance settings
set defer_builds_mode, 3
set async_builds, 1
set ribbon_trace_atoms, 1

bg_color white
"""


def write_trajectory_pml(path: Path, ensemble_pdb: str, title: str) -> None:
    """PML for a trajectory ensemble, coloured by chainbreak (auto-scale)."""
    body = f"""
load ref.pdb, ref
load {ensemble_pdb}, ensemble

hide everything
show cartoon, ref
color grey70, ref

show ribbon, ensemble
set all_states, 1

# B-factor here = chainbreak. Rescaled per view, so blue/red mean
# relative-lowest/highest within this file, not absolute values.
spectrum b, blue_white_red, ensemble

orient ref
zoom ref, 15
"""
    path.write_text(_PML_HEADER.format(title=title) + body)


def write_placement_pml(path: Path, placements_pdb: str, title: str) -> None:
    """PML for placements, coloured by closure fraction on a fixed 0-1 scale."""
    body = f"""
load ref.pdb, ref
load {placements_pdb}, placements

hide everything
show cartoon, ref
color grey70, ref

show cartoon, placements
set all_states, 1

# B-factor here = closure fraction, fixed 0..1 scale so that colours ARE
# comparable across files and across runs (blue=0, white=0.5, red=1).
spectrum b, blue_white_red, placements, 0.0, 1.0

orient ref
zoom ref, 15
"""
    path.write_text(_PML_HEADER.format(title=title) + body)


# --------------------------------------------------------------------------- #
# 8. Grid array assembly for plots
# --------------------------------------------------------------------------- #

def build_grid_arrays(cells: Dict[int, dict], axes: List[Tuple[str, List[float]]]
                      ) -> Dict[str, object]:
    """Reshape the per-cell dict into 2D arrays over the grid."""
    if len(axes) != 2:
        raise SystemExit(
            f"Only 2D grids are supported in this analyser; found "
            f"{len(axes)} axes ({[a[0] for a in axes]}). "
            f"A 1D or 3D grid needs a different plot layout.")
    (x_name, x_vals), (y_name, y_vals) = axes
    nx, ny = len(x_vals), len(y_vals)

    closure = np.full((nx, ny), np.nan)
    chainbreak = np.full((nx, ny), np.nan)
    viable = np.zeros((nx, ny), dtype=bool)
    gap = np.full((nx, ny), np.nan)

    for cell in cells.values():
        i, j = cell["grid_x_i"], cell["grid_y_i"]
        if not (0 <= i < nx and 0 <= j < ny):
            continue
        viable[i, j] = cell["viable"]
        gap[i, j] = cell["loop_gap"]
        if cell["viable"] and cell["n_kept"] > 0:
            closure[i, j] = cell["closure_fraction"]
            chainbreak[i, j] = cell["mean_chainbreak"]

    return {
        "x_name": x_name, "y_name": y_name,
        "x_vals": np.asarray(x_vals), "y_vals": np.asarray(y_vals),
        "closure": closure, "chainbreak": chainbreak,
        "viable": viable, "gap": gap,
    }


# --------------------------------------------------------------------------- #
# 9. Plots
# --------------------------------------------------------------------------- #

def _grid_scatter(ax, x_vals, y_vals, Z_nx_ny,
                  reach_threshold=None, figsize_ax_inches=5.5,
                  fill=0.62, **kwargs):
    """Scatter-plot a 2D grid: one coloured circle per cell.

    Parameters
    ----------
    ax                : matplotlib Axes
    x_vals, y_vals    : 1-D coordinate arrays (length nx, ny)
    Z_nx_ny           : 2-D data array of shape (nx, ny); NaN = no data
    reach_threshold   : if given, cells with Z >= this get a bold white halo
    figsize_ax_inches : approximate axes width (inches) — used to auto-size dots
    fill              : fraction of the narrowest cell spacing each dot covers
    **kwargs          : forwarded to the coloured scatter call (cmap, vmin, vmax, …)

    Returns the PathCollection for colorbar attachment (or None if all NaN).
    """
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")  # shape (nx, ny)

    # Auto-size: diameter = fill * cell_spacing, area in scatter's "points^2"
    nx, ny = len(x_vals), len(y_vals)
    pts_per_cell = figsize_ax_inches * 72.0 / max(nx, ny)
    s = (pts_per_cell * fill) ** 2

    # Set tight limits with half-cell padding so dots don't touch the frame
    margin_x = (np.diff(x_vals).min() if nx > 1 else 1) * 0.6
    margin_y = (np.diff(y_vals).min() if ny > 1 else 1) * 0.6
    ax.set_xlim(x_vals[0] - margin_x, x_vals[-1] + margin_x)
    ax.set_ylim(y_vals[0] - margin_y, y_vals[-1] + margin_y)

    # Ghost circles for NaN / unsampled cells (grid structure stays visible)
    nan_mask = np.isnan(Z_nx_ny)
    if np.any(nan_mask):
        ax.scatter(X[nan_mask], Y[nan_mask],
                   s=s, marker='o',
                   facecolors='none', edgecolors='#cccccc', linewidths=0.8,
                   zorder=1)

    # Coloured circles for valid cells
    valid = ~nan_mask
    sc = None
    if np.any(valid):
        sc = ax.scatter(X[valid], Y[valid],
                        c=Z_nx_ny[valid],
                        s=s, marker='o',
                        edgecolors='#333333', linewidths=0.5,
                        zorder=2, **kwargs)

    # White halo ring for cells above reach_threshold
    if reach_threshold is not None:
        reach = valid & (Z_nx_ny >= reach_threshold)
        if np.any(reach):
            ax.scatter(X[reach], Y[reach],
                       s=s * 1.18, marker='o',
                       facecolors='none', edgecolors='white', linewidths=2.0,
                       zorder=3)

    return sc


def _crosshair(ax, colour="red"):
    ax.axhline(0, color=colour, linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(0, color=colour, linestyle=":", linewidth=1, alpha=0.7)


def plot_reachability(path: Path, grid: dict, z_fixed: Optional[float],
                      reach_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    closure = grid["closure"]

    sc = _grid_scatter(ax, grid["x_vals"], grid["y_vals"], closure,
                       reach_threshold=reach_threshold,
                       cmap="viridis", vmin=0, vmax=1)

    _crosshair(ax)
    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Closure fraction")

    n_reach = int(np.sum(~np.isnan(closure) & (closure >= reach_threshold)))
    n_valid = int(np.sum(~np.isnan(closure)))
    ax.set_xlabel(f"{grid['x_name']} offset from anchor (Å)")
    ax.set_ylabel(f"{grid['y_name']} offset from anchor (Å)")
    z_note = f", z = {z_fixed:+.1f} Å" if z_fixed is not None else ""
    ax.set_title(f"Closure fraction over grid{z_note}\n"
                 f"{n_reach}/{n_valid} cells reachable "
                 f"(≥{reach_threshold:g}, white ring)")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_chainbreak(path: Path, grid: dict, z_fixed: Optional[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cb = grid["chainbreak"]

    # LogNorm chokes on non-positive values; clip and mask.
    positive = np.where(cb > 0, cb, np.nan)
    if np.all(np.isnan(positive)):
        ax.text(0.5, 0.5, "no closed structures", ha="center", va="center",
                transform=ax.transAxes)
        sc = None
    else:
        vmin = max(np.nanmin(positive), 1e-3)
        vmax = np.nanmax(positive)
        norm = LogNorm(vmin=vmin, vmax=vmax) if vmax > vmin else None
        sc = _grid_scatter(ax, grid["x_vals"], grid["y_vals"], positive,
                           cmap="viridis_r", norm=norm)
        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Mean chainbreak (log)")

    _crosshair(ax)
    ax.set_xlabel(f"{grid['x_name']} offset from anchor (Å)")
    ax.set_ylabel(f"{grid['y_name']} offset from anchor (Å)")
    z_note = f", z = {z_fixed:+.1f} Å" if z_fixed is not None else ""
    ax.set_title(f"Mean chainbreak per cell{z_note}")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap_vs_closure(path: Path, cells: Dict[int, dict],
                        max_span: Optional[float],
                        reach_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    viable = [c for c in cells.values() if c["viable"] and c["n_kept"] > 0]
    if not viable:
        ax.text(0.5, 0.5, "no viable cells with structures",
                ha="center", va="center", transform=ax.transAxes)
    else:
        gaps = np.array([c["loop_gap"] for c in viable])
        fracs = np.array([c["closure_fraction"] for c in viable])
        sc = ax.scatter(gaps, fracs, c=fracs, cmap="viridis", vmin=0, vmax=1,
                        s=40, edgecolors="black", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="Closure fraction")

    if max_span is not None:
        ax.axvline(max_span, color="red", linestyle="--", linewidth=1,
                   label=f"linker reach limit ({max_span:.1f} Å)")
    ax.axhline(reach_threshold, color="grey", linestyle=":", linewidth=1,
               label=f"reach threshold ({reach_threshold:g})")

    ax.set_xlabel("Pre-MC loop gap = |dx, dy, dz| (Å)")
    ax.set_ylabel("Closure fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Closure fraction vs starting loop gap")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_summary(path: Path, grid: dict, cells: Dict[int, dict],
                 score_rows: List[dict], close_threshold: float,
                 reach_threshold: float, z_fixed: Optional[float],
                 max_span: Optional[float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # A: reachability scatter
    ax = axes[0, 0]
    closure = grid["closure"]
    sc = _grid_scatter(ax, grid["x_vals"], grid["y_vals"], closure,
                       reach_threshold=reach_threshold,
                       figsize_ax_inches=5.0,
                       cmap="viridis", vmin=0, vmax=1)
    _crosshair(ax)
    if sc is not None:
        plt.colorbar(sc, ax=ax, label="Closure fraction")
    ax.set_xlabel(f"{grid['x_name']} (Å)")
    ax.set_ylabel(f"{grid['y_name']} (Å)")
    ax.set_title("Closure fraction")
    ax.set_aspect("equal")

    # B: chainbreak scatter
    ax = axes[0, 1]
    cb = grid["chainbreak"]
    positive = np.where(cb > 0, cb, np.nan)
    if np.any(~np.isnan(positive)):
        vmin = max(np.nanmin(positive), 1e-3)
        vmax = np.nanmax(positive)
        norm = LogNorm(vmin=vmin, vmax=vmax) if vmax > vmin else None
        sc = _grid_scatter(ax, grid["x_vals"], grid["y_vals"], positive,
                           figsize_ax_inches=5.0,
                           cmap="viridis_r", norm=norm)
        if sc is not None:
            plt.colorbar(sc, ax=ax, label="Mean chainbreak (log)")
    _crosshair(ax)
    ax.set_xlabel(f"{grid['x_name']} (Å)")
    ax.set_ylabel(f"{grid['y_name']} (Å)")
    ax.set_title("Mean chainbreak")
    ax.set_aspect("equal")

    # C: gap vs closure
    ax = axes[1, 0]
    viable = [c for c in cells.values() if c["viable"] and c["n_kept"] > 0]
    if viable:
        gaps = np.array([c["loop_gap"] for c in viable])
        fracs = np.array([c["closure_fraction"] for c in viable])
        ax.scatter(gaps, fracs, c=fracs, cmap="viridis", vmin=0, vmax=1,
                   s=30, edgecolors="black", linewidths=0.3)
    if max_span is not None:
        ax.axvline(max_span, color="red", linestyle="--", linewidth=1,
                   label=f"reach limit ({max_span:.1f} Å)")
    ax.axhline(reach_threshold, color="grey", linestyle=":", linewidth=1,
               label=f"reach threshold ({reach_threshold:g})")
    ax.set_xlabel("Pre-MC loop gap (Å)")
    ax.set_ylabel("Closure fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Gap vs closure")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # D: chainbreak histogram across all silent-file models
    ax = axes[1, 1]
    cbs = [float(r["chainbreak"]) for r in score_rows if "chainbreak" in r]
    if cbs:
        # Log-x if the distribution spans orders of magnitude
        cbs_positive = [c for c in cbs if c > 0]
        if cbs_positive and max(cbs_positive) / min(cbs_positive) > 100:
            bins = np.logspace(np.log10(min(cbs_positive)),
                               np.log10(max(cbs_positive)), 40)
            ax.hist(cbs_positive, bins=bins)
            ax.set_xscale("log")
        else:
            ax.hist(cbs, bins=40)
    ax.axvline(close_threshold, color="red", linestyle="--", linewidth=1,
               label=f"close threshold ({close_threshold:g})")
    ax.set_xlabel("Chainbreak")
    ax.set_ylabel("Count (models)")
    ax.set_title("Chainbreak distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    z_note = f"z = {z_fixed:+.1f} Å, " if z_fixed is not None else ""
    fig.suptitle(f"Grid sweep summary  ({z_note}"
                 f"close < {close_threshold:g}, "
                 f"reach >= {reach_threshold:g})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 10. CSV
# --------------------------------------------------------------------------- #

def write_csv(path: Path, cells: Dict[int, dict]) -> None:
    fields = ["placement", "viable", "grid_x_i", "grid_y_i",
              "dx", "dy", "dz", "loop_gap",
              "n_traj_run", "n_kept", "n_closed", "closure_fraction",
              "mean_chainbreak", "best_chainbreak", "best_tag"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for cell in sorted(cells.values(), key=lambda c: c["placement"]):
            row = {k: cell.get(k, "") for k in fields}
            for k in ("dx", "dy", "dz", "loop_gap", "closure_fraction",
                      "mean_chainbreak", "best_chainbreak"):
                v = row[k]
                if isinstance(v, float):
                    row[k] = "" if np.isnan(v) else round(v, 4)
            w.writerow(row)


# --------------------------------------------------------------------------- #
# 11. Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, type=Path,
                   help="Ensemble run directory (contains metrics.json and "
                        "the silent file).")
    p.add_argument("--silent", default="ensemble.out",
                   help="Silent file name inside --dir")
    p.add_argument("--config", type=Path, default=None,
                   help="Explicit path to the YAML construct definition "
                        "(defaults to what's stored in metrics.json).")
    p.add_argument("--out", default="viz",
                   help="Output directory name inside --dir")

    p.add_argument("--close-threshold", type=float, default=1.0,
                   help="A model is 'closed' if its chainbreak is below this. "
                        "Rosetta chainbreak units; lower means better closure.")
    p.add_argument("--reach-threshold", type=float, default=0.02,
                   help="A cell is 'reachable' if its closure fraction is at "
                        "least this. Prior work found 0.1 too permissive; "
                        "0.01-0.05 is a more informative range.")

    p.add_argument("--ensemble-cap", type=int, default=200,
                   help="Max MODELs in the trajectory ensemble PDBs. Random "
                        "subsample if the run produced more. Set to 0 to "
                        "include everything (may be slow to load in PyMOL).")
    p.add_argument("--span-slack", type=float, default=0.9,
                   help="Same knob as the sampler; used only to draw the "
                        "reach line on gap_vs_closure.png.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the ensemble subsample")
    return p.parse_args()


def linker_reach_limit(cfg: dict, span_slack: float) -> Optional[float]:
    """Return the reach limit if there is exactly one flexible segment."""
    flex = [s for s in cfg["segments"] if s["kind"] == "flexible"]
    if len(flex) != 1:
        return None
    seg = flex[0]
    length = seg["stop"] - seg["start"] + 1
    return MAX_SPAN_PER_RESIDUE * (length + 1) * span_slack


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    run_dir = args.dir
    silent_path = run_dir / args.silent
    if not silent_path.is_file():
        raise SystemExit(f"Silent file not found: {silent_path}")

    print(f"Reading   : {run_dir}")
    metrics = load_metrics(run_dir)
    cfg = load_config(run_dir, metrics, args.config)

    if metrics.get("placement_mode") != "grid":
        raise SystemExit(
            f"placement_mode is {metrics.get('placement_mode')!r}; "
            "this analyser only handles grid runs.")

    seg_name, axes = grid_axes(metrics)
    anchor_name, anchor_pos, fixed_dof = anchor_info(metrics)
    z_fixed = fixed_dof.get("z")

    print(f"Grid      : {seg_name} anchored on {anchor_name} at "
          f"({anchor_pos['x']:.2f}, {anchor_pos['y']:.2f}, {anchor_pos['z']:.2f})")
    for name, values in axes:
        print(f"            axis {name}: {len(values)} values "
              f"[{values[0]}..{values[-1]}]")

    score_rows = parse_score_table(silent_path)
    print(f"Silent    : {len(score_rows)} models")

    cells = aggregate_cells(metrics, score_rows, args.close_threshold)
    n_viable = sum(1 for c in cells.values() if c["viable"])
    n_reach = sum(1 for c in cells.values()
                  if c["viable"] and not np.isnan(c["closure_fraction"])
                  and c["closure_fraction"] >= args.reach_threshold)
    print(f"Cells     : {len(cells)} total, {n_viable} viable, "
          f"{n_reach} reachable (closure >= {args.reach_threshold:g})")

    # ---- PyRosetta init (needed for silent decoding) ----
    pyrosetta.init(f"-mute all -constant_seed -jran {args.seed or 1}")

    # Anchor domain: read from the input PDB (always available; sampler stored
    # its path in metrics.json). This is preferred over decoding a silent
    # structure because the silent file might be empty if all cells failed.
    ref_pose = pyrosetta.pose_from_pdb(str(metrics["pdb"]))
    anchor_res, mobile_res, chain = segment_residues(cfg)
    ref_coords = pose_ca_by_pdb_num(ref_pose)

    # ---- Decide which structures to decode ----
    all_tags = [str(r["description"]) for r in score_rows]
    reachable_row_idx = [i for i, r in enumerate(score_rows)
                         if float(r.get("chainbreak", float("inf"))) < args.close_threshold]

    if args.ensemble_cap and len(all_tags) > args.ensemble_cap:
        sample_idx = sorted(rng.sample(range(len(all_tags)), args.ensemble_cap))
    else:
        sample_idx = list(range(len(all_tags)))

    reach_capped = [i for i in reachable_row_idx]
    if args.ensemble_cap and len(reach_capped) > args.ensemble_cap:
        reach_capped = sorted(rng.sample(reach_capped, args.ensemble_cap))

    best_tags_by_cell = {p: c["best_tag"] for p, c in cells.items()
                         if c["best_tag"] is not None}

    tags_needed: List[str] = []
    tags_needed.extend(str(score_rows[i]["description"]) for i in sample_idx)
    tags_needed.extend(str(score_rows[i]["description"]) for i in reach_capped)
    tags_needed.extend(best_tags_by_cell.values())
    tags_needed = list(dict.fromkeys(tags_needed))       # preserve order, dedup

    print(f"Decoding  : {len(tags_needed)} unique structures from the silent file")
    poses = load_poses(silent_path, tags_needed)

    # ---- Output setup ----
    out_dir = run_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- ref.pdb ----
    write_single_model_pdb(out_dir / "ref.pdb", ref_coords, anchor_res, chain,
                           bfactor=0.0)

    # ---- ensemble PDBs ----
    def _row_entries(indices):
        entries = []
        for i in indices:
            tag = str(score_rows[i]["description"])
            if tag not in poses:
                continue
            bfactor = float(score_rows[i].get("chainbreak", 0.0))
            entries.append((poses[tag], bfactor))
        return entries

    n_all = write_multi_model_pdb(out_dir / "ensemble_all.pdb",
                                  _row_entries(sample_idx),
                                  mobile_res, chain)
    n_reach = write_multi_model_pdb(out_dir / "ensemble_reachable.pdb",
                                    _row_entries(reach_capped),
                                    mobile_res, chain)
    print(f"Wrote     : ensemble_all.pdb ({n_all} MODELs), "
          f"ensemble_reachable.pdb ({n_reach} MODELs)")

    # ---- placement PDBs ----
    placement_entries = []
    placement_entries_reachable = []
    for p_idx in sorted(cells):
        cell = cells[p_idx]
        tag = cell["best_tag"]
        if tag is None or tag not in poses:
            continue
        frac = cell["closure_fraction"]
        if np.isnan(frac):
            continue
        placement_entries.append((poses[tag], frac))
        if frac >= args.reach_threshold:
            placement_entries_reachable.append((poses[tag], frac))

    n_p_all = write_multi_model_pdb(out_dir / "placements_all.pdb",
                                    placement_entries, mobile_res, chain)
    n_p_reach = write_multi_model_pdb(out_dir / "placements_reachable.pdb",
                                      placement_entries_reachable,
                                      mobile_res, chain)
    print(f"Wrote     : placements_all.pdb ({n_p_all} MODELs), "
          f"placements_reachable.pdb ({n_p_reach} MODELs)")

    # ---- PMLs ----
    write_trajectory_pml(out_dir / "view.pml", "ensemble_all.pdb",
                         "All trajectories, coloured by chainbreak")
    write_trajectory_pml(out_dir / "reachable.pml", "ensemble_reachable.pdb",
                         f"Reachable trajectories (chainbreak < "
                         f"{args.close_threshold:g}), coloured by chainbreak")
    write_placement_pml(out_dir / "placements.pml", "placements_all.pdb",
                        "Best model per viable cell, coloured by closure fraction")
    write_placement_pml(out_dir / "reachable_placements.pml",
                        "placements_reachable.pdb",
                        f"Best model per reachable cell (closure >= "
                        f"{args.reach_threshold:g}), coloured by closure fraction")

    # ---- Plots ----
    grid = build_grid_arrays(cells, axes)
    reach_limit = linker_reach_limit(cfg, args.span_slack)

    plot_reachability(out_dir / "reachability.png", grid, z_fixed,
                      args.reach_threshold)
    plot_chainbreak(out_dir / "chainbreak.png", grid, z_fixed)
    plot_gap_vs_closure(out_dir / "gap_vs_closure.png", cells, reach_limit,
                        args.reach_threshold)
    plot_summary(out_dir / "summary.png", grid, cells, score_rows,
                 args.close_threshold, args.reach_threshold,
                 z_fixed, reach_limit)

    # ---- CSV ----
    write_csv(out_dir / "placements.csv", cells)

    print(f"Done      : viz written to {out_dir}/")


if __name__ == "__main__":
    main()