#!/usr/bin/env python3
"""
Ensemble sampling of flexible linkers connecting rigid bodies.

The construct is described in a YAML file as an ordered list of segments that
alternate rigid / flexible, beginning and ending with a rigid segment:

    [ rigid ]--flexible--[ rigid ]--flexible--[ rigid ]--...

Rigid segments declare how far they may move (translate per axis, tilt, spin).
Flexible segments are what the Monte Carlo torsion sampling acts on, and are
what has to close back up after the rigid bodies have moved.

Nothing is hard-coded to a particular number of segments, so a construct with
one flexible linker and a construct with two flexible loops around an inserted
helix both run through the same code.

Stages
------
1. Read the construct definition from YAML.
2. Build a FoldTree with a jump to each rigid segment and a cutpoint inside
   each flexible segment.
3. Enumerate rigid-body placements. Each mobile rigid segment declares a
   placement mode:
     random  Draw n placements uniformly within the declared per-axis limits;
             retry viability failures up to n * max_attempts_factor times.
             Translations are lab-frame deltas from the input pose.
     grid    Iterate the Cartesian product of the declared grid axes; run
             every cell once and record non-viable cells with zero
             trajectories. Translations are absolute positions of the moved
             segment's N-terminal CA, expressed relative to the C-terminal CA
             of an `anchor` rigid segment (signed z: negative = below anchor).
4. For each placement, run Monte Carlo torsion sampling over the flexible
   segments (perturb -> close -> repack -> Metropolis), analogous to the
   SetTorsion + PackRotamers + GenericMonteCarlo block in symm_torsions.xml.
5. Filter surviving configurations (PLACEHOLDER - criteria not yet decided).

Parallelism (grid mode only)
----------------------------
--cores N runs grid cells in N worker processes, one cell per task. Each cell
is seeded from (--seed, cell index) and writes its own silent file under
<out>/parts/; the parent merges them into --silent-name (single header) and
writes metrics.json in placement order. Output does not depend on N.
Random mode stays serial with one shared RNG, as before.

Convention assumed throughout: the membrane normal is the lab-frame z axis and
the bilayer mid-plane is at z = 0. If your input PDB is not oriented that way
(e.g. it came straight out of AlphaFold), pre-orient it before running this, or
set MEMBRANE_NORMAL below to the correct vector.

    python sample_linker_ensemble.py --pdb tagged.pdb --config tagged.yaml \
        --n-placements 50 --n-traj 20 --trials 500 --out ensemble/

    python sample_linker_ensemble.py --pdb tagged.pdb --config grid.yaml \
        --n-traj 20 --trials 500 --cores 16 --out grid_ensemble/

Example YAML
------

# Random placements within a 5D box:

segments:
  - name: fn3
    kind: rigid
    start: 1
    stop: 95
    chain: A

  - name: linker
    kind: flexible
    start: 96
    stop: 113
    chain: A

  - name: tm
    kind: rigid
    start: 114
    stop: 136
    chain: A
    dof:
      translate: {x: 15.0, y: 15.0}
      tilt: 20.0
      spin: 180.0

# Grid over (x, y) with fixed z below the F3 anchor:

  - name: tm
    kind: rigid
    start: 114
    stop: 136
    chain: A
    dof:
      mode: grid
      anchor: fn3                            # segment whose C-term CA is the origin
      translate:
        x: {min: -20, max: 20, step: 2}      # dict -> grid axis
        y: {min: -20, max: 20, step: 2}      # dict -> grid axis
        z: -15.0                             # scalar -> fixed, signed offset
      tilt: 0.0
      spin: 0.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
import random
import shutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

import pyrosetta
from pyrosetta.rosetta.core.kinematics import FoldTree, MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.core.select.residue_selector import (
    NeighborhoodResidueSelector, ResidueIndexSelector,
)
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
from pyrosetta.rosetta.protocols import rigid
from pyrosetta.rosetta.protocols.loops import Loop, Loops
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.moves import MonteCarlo


# --------------------------------------------------------------------------- #
# Global conventions
# --------------------------------------------------------------------------- #

MEMBRANE_NORMAL = (0.0, 0.0, 1.0)

# Max C-alpha span per residue for an extended chain, used for the
# "can this loop physically reach?" pre-check. ~3.3-3.5 A is standard.
MAX_SPAN_PER_RESIDUE = 3.3


# --------------------------------------------------------------------------- #
# 1. Construct definition (from YAML)
# --------------------------------------------------------------------------- #

@dataclass
class DOF:
    """Placement policy for one rigid segment.

    Random mode (default): tx/ty/tz/tilt/spin are max magnitudes for uniform
    sampling. Translations are lab-frame deltas from the input pose; tilt is
    an angle away from the segment's own long axis; spin is about that axis.

    Grid mode: any of {x, y, z, tilt, spin} may be given as a scalar (fixed
    value) or as a dict {min, max, step} (grid axis). The Cartesian product
    of the grid axes is enumerated, with scalars held constant. The (x, y, z)
    values are absolute positions of the moved segment's N-terminal CA,
    expressed relative to the C-terminal CA of `anchor` -- signed, so
    z: -15 means "attachment atom 15 A below the anchor". Tilt and spin
    keep their perturbation-from-input semantics.
    """
    mode: str = "random"
    anchor: Optional[str] = None
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    tilt: float = 0.0
    spin: float = 0.0
    # {"x"|"y"|"z"|"tilt"|"spin": (min, max, step)}. Grid mode only.
    grid_axes: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DOF":
        if not d:
            return cls()
        bad = set(d) - {"translate", "tilt", "spin", "mode", "anchor"}
        if bad:
            raise ValueError(f"Unknown dof keys: {sorted(bad)}")

        mode = d.get("mode", "random")
        if mode not in ("random", "grid"):
            raise ValueError(f"dof.mode must be 'random' or 'grid', got {mode!r}")

        fixed = {"x": 0.0, "y": 0.0, "z": 0.0, "tilt": 0.0, "spin": 0.0}
        grid_axes: Dict[str, Tuple[float, float, float]] = {}

        def parse_axis(name: str, value) -> None:
            if isinstance(value, dict):
                if mode != "grid":
                    raise ValueError(
                        f"dof.{name} is a grid spec ({value}) but dof.mode "
                        f"is {mode!r}; set 'mode: grid' or use a scalar")
                extra = set(value) - {"min", "max", "step"}
                missing = {"min", "max", "step"} - set(value)
                if extra or missing:
                    raise ValueError(
                        f"dof.{name} grid axis: unknown keys {sorted(extra)}, "
                        f"missing keys {sorted(missing)}")
                mn, mx, st = float(value["min"]), float(value["max"]), float(value["step"])
                if st <= 0:
                    raise ValueError(f"dof.{name}.step must be > 0, got {st}")
                if mx < mn:
                    raise ValueError(f"dof.{name}: max < min ({mn}..{mx})")
                grid_axes[name] = (mn, mx, st)
            else:
                fixed[name] = float(value)

        trans = d.get("translate") or {}
        bad_ax = set(trans) - {"x", "y", "z"}
        if bad_ax:
            raise ValueError(f"Unknown translate axes: {sorted(bad_ax)}")
        for k, v in trans.items():
            parse_axis(k, v)
        if "tilt" in d:
            parse_axis("tilt", d["tilt"])
        if "spin" in d:
            parse_axis("spin", d["spin"])

        anchor = d.get("anchor")
        if mode == "grid":
            if not anchor:
                raise ValueError("grid mode requires 'anchor: <segment name>'")
            if not grid_axes:
                raise ValueError(
                    "grid mode requires at least one axis given as "
                    "{min, max, step}; got only scalars")
        elif anchor:
            raise ValueError("'anchor' is only meaningful in grid mode")

        return cls(
            mode=mode, anchor=anchor,
            tx=fixed["x"], ty=fixed["y"], tz=fixed["z"],
            tilt=fixed["tilt"], spin=fixed["spin"],
            grid_axes=grid_axes,
        )

    @property
    def is_mobile(self) -> bool:
        return (any((self.tx, self.ty, self.tz, self.tilt, self.spin))
                or bool(self.grid_axes))


@dataclass
class Segment:
    """One segment of the construct. start/stop are inclusive PDB residue numbers."""
    name: str
    kind: str          # "rigid" | "flexible"
    start: int
    stop: int
    chain: str = "A"
    dof: DOF = field(default_factory=DOF)

    def __post_init__(self):
        if self.kind not in ("rigid", "flexible"):
            raise ValueError(f"{self.name}: kind must be 'rigid' or 'flexible'")
        if self.stop < self.start:
            raise ValueError(f"{self.name}: stop < start ({self.start}-{self.stop})")
        if self.kind == "flexible" and self.dof.is_mobile:
            raise ValueError(
                f"{self.name}: flexible segments are sampled by torsion MC, not "
                "rigid-body moves - remove the dof block"
            )

    @property
    def length(self) -> int:
        return self.stop - self.start + 1

    @property
    def mid(self) -> int:
        return (self.start + self.stop) // 2

    def residues(self) -> List[int]:
        return list(range(self.start, self.stop + 1))


@dataclass
class Construct:
    segments: List[Segment]

    @property
    def rigid(self) -> List[Segment]:
        return [s for s in self.segments if s.kind == "rigid"]

    @property
    def flexible(self) -> List[Segment]:
        return [s for s in self.segments if s.kind == "flexible"]

    def cutpoints(self) -> List[int]:
        return [s.mid for s in self.flexible]

    def loop_residues(self) -> List[int]:
        out: List[int] = []
        for s in self.flexible:
            out.extend(s.residues())
        return out

    def blocks(self) -> List[Tuple[Segment, int, int]]:
        """
        The cutpoints partition the chain into one block per rigid segment.
        Each rigid segment carries the trailing half of the loop before it and
        the leading half of the loop after it, so moving a block moves a whole
        physically sensible unit.
        """
        cuts = sorted(self.cutpoints())
        lo, hi = self.segments[0].start, self.segments[-1].stop
        bounds, prev = [], lo
        for c in cuts:
            bounds.append((prev, c))
            prev = c + 1
        bounds.append((prev, hi))
        return [(r, b[0], b[1]) for r, b in zip(self.rigid, bounds)]

    def validate(self) -> None:
        kinds = [s.kind for s in self.segments]
        if not kinds:
            raise ValueError("No segments defined")
        if kinds[0] != "rigid" or kinds[-1] != "rigid":
            raise ValueError("The construct must begin and end with a rigid segment")
        for a, b in zip(kinds, kinds[1:]):
            if a == b:
                raise ValueError(f"Two consecutive '{a}' segments - they must alternate")
        for a, b in zip(self.segments, self.segments[1:]):
            if b.start != a.stop + 1:
                raise ValueError(
                    f"Gap or overlap: {a.name} ends {a.stop}, {b.name} starts {b.start}"
                )
        if not self.flexible:
            raise ValueError("No flexible segments - nothing to sample")

        # Mixed placement modes across mobile segments would require iterating
        # a grid *and* drawing random placements simultaneously; not worth the
        # complexity until a use case appears.
        mobile = [s for s in self.rigid if s.dof.is_mobile]
        modes = {s.dof.mode for s in mobile}
        if len(modes) > 1:
            raise ValueError(
                f"All mobile rigid segments must share the same dof.mode; "
                f"got {sorted(modes)}")

        rigid_names = {s.name for s in self.rigid}
        by_name = {s.name: s for s in self.rigid}
        for s in mobile:
            if s.dof.mode != "grid":
                continue
            if s.dof.anchor not in rigid_names:
                raise ValueError(
                    f"{s.name}: dof.anchor '{s.dof.anchor}' is not a declared "
                    f"rigid segment (options: {sorted(rigid_names)})")
            if s.dof.anchor == s.name:
                raise ValueError(f"{s.name}: dof.anchor cannot be self")
            if by_name[s.dof.anchor].dof.is_mobile:
                raise ValueError(
                    f"{s.name}: anchor '{s.dof.anchor}' is mobile; grid "
                    f"coordinates need a fixed reference point")


def load_construct(path: str) -> Construct:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = Construct([
        Segment(name=e["name"], kind=e["kind"],
                start=int(e["start"]), stop=int(e["stop"]),
                chain=str(e.get("chain", "A")),
                dof=DOF.from_dict(e.get("dof")))
        for e in raw["segments"]
    ])
    cfg.validate()
    return cfg


def to_pose_numbering(pose, cfg: Construct) -> Construct:
    """Translate the YAML's PDB numbering into pose numbering, once at startup."""
    info = pose.pdb_info()

    def idx(chain: str, resnum: int, name: str) -> int:
        i = info.pdb2pose(chain, resnum)
        if i == 0:
            raise ValueError(f"Residue {chain}{resnum} of segment '{name}' not in pose")
        return i

    return Construct([
        Segment(name=s.name, kind=s.kind,
                start=idx(s.chain, s.start, s.name),
                stop=idx(s.chain, s.stop, s.name),
                chain=s.chain, dof=s.dof)
        for s in cfg.segments
    ])


# --------------------------------------------------------------------------- #
# 2. FoldTree
# --------------------------------------------------------------------------- #

def build_fold_tree(pose, cfg: Construct) -> Dict[str, int]:
    """
    Root on the first rigid segment, with a jump out to every other rigid
    segment and a chain break at the midpoint of every flexible segment.

    The root is fixed by construction: everything is expressed relative to it,
    so a dof block on the first rigid segment would only translate the whole
    system and is ignored. Put the dof on the segments you want to move
    relative to it.

    Star topology, so the mobile bodies are independent of one another -
    perturbing one does not drag the others along.

    Returns {segment_name: jump_id}.
    """
    blocks = cfg.blocks()
    root_seg, root_lo, root_hi = blocks[0]
    root = root_seg.mid

    ft = FoldTree()
    ft.clear()
    ft.add_edge(root, root_lo, -1)
    ft.add_edge(root, root_hi, -1)

    jumps: Dict[str, int] = {}
    for jid, (rseg, blo, bhi) in enumerate(blocks[1:], start=1):
        ft.add_edge(root, rseg.mid, jid)
        ft.add_edge(rseg.mid, blo, -1)
        ft.add_edge(rseg.mid, bhi, -1)
        jumps[rseg.name] = jid

    ft.reorder(root)
    if not ft.check_fold_tree():
        raise RuntimeError("Invalid FoldTree - check the segment ranges in the YAML")
    pose.fold_tree(ft)

    # Cutpoint variants are required for the chainbreak score term to register.
    # TODO: verify the exact helper name in your PyRosetta build; alternatives
    # are core.pose.correctly_add_cutpoint_variants(pose) or
    # protocols.loops.add_single_cutpoint_variant(pose, loop).
    for s in cfg.flexible:
        pyrosetta.rosetta.protocols.loops.add_single_cutpoint_variant(
            pose, Loop(s.start, s.stop, s.mid))

    return jumps


def build_loops(cfg: Construct) -> Loops:
    loops = Loops()
    for s in cfg.flexible:
        loops.add_loop(Loop(s.start, s.stop, s.mid))
    return loops


# --------------------------------------------------------------------------- #
# 3. Rigid-body placement sampling
# --------------------------------------------------------------------------- #

def ca_xyz(pose, resid: int):
    return pose.residue(resid).xyz("CA")


def segment_axis(pose, seg: Segment) -> Tuple[float, float, float]:
    """
    Long axis of a rigid segment, as the first-CA -> last-CA unit vector.

    Fine for a straight helix or a compact domain. For a kinked helix this is
    skewed by the kink - if that matters, fit the axis over a narrower,
    well-defined sub-range instead.
    """
    a, b = ca_xyz(pose, seg.start), ca_xyz(pose, seg.stop)
    v = (b.x - a.x, b.y - a.y, b.z - a.z)
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-9:
        raise ValueError(f"{seg.name}: degenerate axis")
    return (v[0] / n, v[1] / n, v[2] / n)


def segment_centroid(pose, lo: int, hi: int) -> Tuple[float, float, float]:
    sx = sy = sz = 0.0
    n = 0
    for i in range(lo, hi + 1):
        res = pose.residue(i)
        for a in range(1, res.natoms() + 1):
            v = res.xyz(a)
            sx += v.x
            sy += v.y
            sz += v.z
            n += 1
    return (sx / n, sy / n, sz / n)


def perpendicular_to(axis: Tuple[float, float, float],
                     azimuth_deg: float) -> Tuple[float, float, float]:
    """A unit vector perpendicular to `axis`, at a given azimuth around it."""
    ref = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.95 else (1.0, 0.0, 0.0)

    def cross(u, v):
        return (u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])

    def unit(u):
        n = math.sqrt(sum(c * c for c in u))
        return (u[0] / n, u[1] / n, u[2] / n)

    e1 = unit(cross(axis, ref))
    e2 = unit(cross(axis, e1))
    t = math.radians(azimuth_deg)
    return (math.cos(t) * e1[0] + math.sin(t) * e2[0],
            math.cos(t) * e1[1] + math.sin(t) * e2[1],
            math.cos(t) * e1[2] + math.sin(t) * e2[2])


def translate_jump(pose, jump_id: int, axis: Tuple[float, float, float],
                   distance: float) -> None:
    """
    Translate the downstream partner of `jump_id` along `axis` by `distance`.

    NOTE / CAVEAT: RigidBodyTransMover's axis convention has bitten people
    before - it is worth confirming empirically (translate by a large known
    distance, dump the PDB, measure) that this moves in the LAB frame and not
    in the jump's local stub frame. If it turns out to be stub-local, replace
    this with an explicit pose.set_jump() using a rotation-corrected vector.
    """
    if abs(distance) < 1e-9:
        return
    mover = rigid.RigidBodyTransMover(pose, jump_id)
    mover.trans_axis(xyzVec(*axis))
    mover.step_size(distance)
    mover.apply(pose)


def rotate_jump(pose, jump_id: int, axis: Tuple[float, float, float],
                angle_deg: float, center: Tuple[float, float, float]) -> None:
    """
    Rotate the downstream partner about `axis` through `center`.

    Rotating about the origin rather than the body's own centre would fling it
    across the box - placements that look plausible in the log and absurd in
    PyMOL.

    TODO: RigidBodyDeterministicSpinMover is the right class for this, but its
    exact name/signature has moved between Rosetta versions. Confirm against
    your build, or compose the rotation into the jump's RT directly.
    """
    if abs(angle_deg) < 1e-9:
        return
    spin = rigid.RigidBodyDeterministicSpinMover()
    spin.rb_jump(jump_id)
    spin.spin_axis(xyzVec(*axis))
    spin.rot_center(xyzVec(*center))
    spin.angle_magnitude(float(angle_deg))
    spin.apply(pose)


def _axis_values(mn: float, mx: float, step: float) -> List[float]:
    """Inclusive linspace from mn to mx at intervals of step.

    Uses explicit `mn + i * step` rather than numpy.arange so the endpoints
    land on exact multiples of step and downstream `.index()` lookups match.
    """
    n = int(round((mx - mn) / step)) + 1
    return [mn + i * step for i in range(n)]


def anchor_position(pose, cfg: Construct, anchor_name: str):
    """CA of the C-terminal residue of the named rigid segment.

    This is the origin of the grid coordinate frame for any mobile segment
    that names it as `anchor`: (dx, dy, dz) = (0, 0, 0) means "attachment CA
    sits exactly at this atom", and z is a signed lab-frame offset from it
    (negative -> below, in the F3/TM case that means towards the membrane).
    """
    for s in cfg.rigid:
        if s.name == anchor_name:
            return ca_xyz(pose, s.stop)
    raise KeyError(anchor_name)


def draw_random_placement(mobiles: List[Segment],
                          rng: random.Random) -> Dict[str, Dict[str, float]]:
    """One random draw within each mobile segment's declared dof limits.

    Values are perturbations of the input pose: (dx, dy, dz) are lab-frame
    translations, tilt is a rotation about a perpendicular to the segment
    axis, spin is about the axis itself. Tilt is drawn uniform in
    [0, tilt_max]; not an isotropic draw on the sphere (which would weight
    by sin), but that deliberately over-samples small tilts around the
    upright reference state.
    """
    drawn: Dict[str, Dict[str, float]] = {}
    for seg in mobiles:
        d = seg.dof
        drawn[seg.name] = {
            "dx": rng.uniform(-d.tx, d.tx) if d.tx else 0.0,
            "dy": rng.uniform(-d.ty, d.ty) if d.ty else 0.0,
            "dz": rng.uniform(-d.tz, d.tz) if d.tz else 0.0,
            "tilt": rng.uniform(0.0, d.tilt) if d.tilt else 0.0,
            "tilt_azimuth": rng.uniform(0.0, 360.0) if d.tilt else 0.0,
            "spin": rng.uniform(-d.spin, d.spin) if d.spin else 0.0,
        }
    return drawn


# The internal placement dict keys that each DOF axis writes to.
_AXIS_TO_KEY = {"x": "dx", "y": "dy", "z": "dz", "tilt": "tilt", "spin": "spin"}


def enumerate_grid_placements(mobiles: List[Segment]):
    """Yield one placement dict per cell of the joint grid.

    Each mobile segment's gridded axes vary; its scalar dof values are held
    fixed. Iteration order is a Cartesian product across all axes of all
    mobile segments, in declaration order (last axis varies fastest).

    For grid segments (dx, dy, dz) are absolute coords (see apply_placement).
    Tilt/spin keep their perturbation semantics. `tilt_azimuth` is deterministic
    at 0.0 in grid mode -- if you later grid tilt itself, decide whether to
    also grid the azimuth or just pick a fixed direction and record that here.
    """
    axis_specs: List[Tuple[str, str, List[float]]] = []
    for seg in mobiles:
        for axis, (mn, mx, step) in seg.dof.grid_axes.items():
            axis_specs.append((seg.name, axis, _axis_values(mn, mx, step)))

    if not axis_specs:
        return   # no grid axes declared -- caller shouldn't have picked grid mode

    index_ranges = [range(len(vals)) for _, _, vals in axis_specs]
    for combo in itertools.product(*index_ranges):
        placement: Dict[str, Dict[str, float]] = {}
        for seg in mobiles:
            d = seg.dof
            placement[seg.name] = {
                "dx": d.tx, "dy": d.ty, "dz": d.tz,
                "tilt": d.tilt, "tilt_azimuth": 0.0, "spin": d.spin,
            }
        for (segname, axis, values), idx in zip(axis_specs, combo):
            placement[segname][_AXIS_TO_KEY[axis]] = values[idx]
            # Grid index rides along in the placement so it surfaces as a
            # SCORE: column later and the analyser can group by cell directly
            # without floating-point comparisons.
            placement[segname][f"grid_{axis}_i"] = float(idx)
        yield placement


def apply_placement(pose, cfg: Construct, jumps: Dict[str, int],
                    placement: Dict[str, Dict[str, float]],
                    anchor_positions: Dict[str, object]) -> None:
    """Apply a placement to the mobile rigid segments, in place.

    Order per segment: spin (about own axis) -> tilt (about a perpendicular)
    -> translate. Random-mode translations are lab-frame deltas from the
    input pose. Grid-mode translations move the segment's N-terminal CA to
    an absolute target: anchor_position + (dx, dy, dz), where dz is signed.
    Because tilt/spin rotate the attachment CA off its input location, the
    translation for grid mode is measured *after* the rotations are applied.
    """
    block_of = {r.name: (lo, hi) for r, lo, hi in cfg.blocks()}

    for seg in cfg.rigid:
        if seg.name not in placement:
            continue
        params = placement[seg.name]
        jid = jumps[seg.name]
        own_axis = segment_axis(pose, seg)
        centre = segment_centroid(pose, *block_of[seg.name])

        if params.get("spin"):
            rotate_jump(pose, jid, own_axis, params["spin"], centre)
        if params.get("tilt"):
            tilt_axis = perpendicular_to(own_axis, params.get("tilt_azimuth", 0.0))
            rotate_jump(pose, jid, tilt_axis, params["tilt"], centre)

        if seg.dof.mode == "grid":
            ap = anchor_positions[seg.dof.anchor]
            current = ca_xyz(pose, seg.start)
            tx = ap.x + params["dx"] - current.x
            ty = ap.y + params["dy"] - current.y
            tz = ap.z + params["dz"] - current.z
        else:
            tx, ty, tz = params["dx"], params["dy"], params["dz"]

        dist = math.sqrt(tx * tx + ty * ty + tz * tz)
        if dist > 1e-9:
            translate_jump(pose, jid, (tx / dist, ty / dist, tz / dist), dist)


def loop_gap(pose, loop: Segment) -> float:
    """CA-CA distance between the rigid residues flanking a flexible segment."""
    return (ca_xyz(pose, loop.start - 1) - ca_xyz(pose, loop.stop + 1)).norm()


def loop_is_spannable(pose, loop: Segment, slack: float = 0.9) -> bool:
    """
    Cheap pre-check: can a loop of this length physically bridge the gap after
    the rigid-body move? Rejecting here is far cheaper than discovering it
    after a failed closure run.

    `slack` < 1.0 keeps a margin, since a maximally extended loop is neither
    closable in practice nor conformationally interesting.
    """
    return loop_gap(pose, loop) <= MAX_SPAN_PER_RESIDUE * (loop.length + 1) * slack


def placement_is_viable(pose, cfg: Construct, slack: float = 0.9) -> bool:
    return all(loop_is_spannable(pose, s, slack) for s in cfg.flexible)


# --------------------------------------------------------------------------- #
# 4. Monte Carlo torsion sampling over the flexible segments
# --------------------------------------------------------------------------- #

def make_score_function(chainbreak_weight: float = 1.0):
    sfxn = pyrosetta.create_score_function("ref2015")
    sfxn.set_weight(ScoreType.chainbreak, chainbreak_weight)
    sfxn.set_weight(ScoreType.linear_chainbreak, chainbreak_weight)
    return sfxn


def make_loop_movemap(cfg: Construct) -> MoveMap:
    """Backbone + sidechain freedom in the flexible segments only; everything
    else fixed.

    Mirrors the MoveMap logic in your FastRelax block (freeze the whole protein,
    then re-enable the linker), but expressed per-residue.
    """
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)
    for resid in cfg.loop_residues():
        mm.set_bb(resid, True)
        mm.set_chi(resid, True)
    return mm


def make_repack_mover(sfxn, cfg: Construct):
    """
    Repack sidechains, restricted to the flexible segments and their
    neighbours. Repacking the entire receptor every MC trial would dominate
    runtime.
    """
    loop_idx = cfg.loop_residues()
    sel = ResidueIndexSelector(",".join(str(i) for i in loop_idx))
    nbr = NeighborhoodResidueSelector(sel, 6.0, True)

    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    tf.push_back(operation.OperateOnResidueSubset(
        operation.PreventRepackingRLT(), nbr, True))   # True = flip the subset

    packer = PackRotamersMover(sfxn)
    packer.task_factory(tf)
    return packer


def perturb_loop_torsions(pose, cfg: Construct, rng: random.Random,
                          sigma: float, n_residues: int = 2) -> None:
    """
    Perturb phi/psi on a few randomly chosen loop residues.

    Your XML randomised every torsion in the linker on every trial
    (angle="random"). That is very aggressive once a chain break has to be
    closed - a Gaussian perturbation on a small subset gives an MC chain that
    actually accepts moves. Set sigma large (or switch to rng.uniform(-180,180))
    if you want the original behaviour.
    """
    candidates = cfg.loop_residues()
    for resid in rng.sample(candidates, min(n_residues, len(candidates))):
        pose.set_phi(resid, pose.phi(resid) + rng.gauss(0.0, sigma))
        pose.set_psi(resid, pose.psi(resid) + rng.gauss(0.0, sigma))


def make_closure_movers(loops: Loops, movemap: MoveMap):
    """
    CCD closure movers, one per flexible segment.

    TODO: KIC generally samples more native-like and more diverse loop
    conformations than CCD, and is the better choice for ensemble generation.
    Swap in GeneralizedKIC, or instantiate LoopModeler from an XML snippet via
    protocols.rosetta_scripts.XmlObjects.static_get_mover(...) to reuse the
    RosettaScripts syntax you already have.
    """
    from pyrosetta.rosetta.protocols.loops.loop_closure.ccd import CCDLoopClosureMover
    return [CCDLoopClosureMover(loops[i], movemap)
            for i in range(1, loops.num_loop() + 1)]


def mc_torsion_sampling(pose, cfg: Construct, sfxn, closers, packer,
                        rng: random.Random, trials: int, temperature: float,
                        sigma: float, n_perturb: int,
                        boltzmann_n: Optional[int] = None, burnin: int = 0):
    """
    Metropolis MC over loop torsions: perturb -> close -> repack -> accept/reject.

    A generator, so structures are handed back one at a time and can be written
    straight to disk instead of piling up in memory.

    Two output modes:

    boltzmann_n = None (default)
        Yields a single pose per trajectory: the lowest-energy one found.
        Equivalent in spirit to GenericMonteCarlo with recover_low="1". Each
        trajectory contributes one local minimum - good structures, but not a
        thermodynamic ensemble.

    boltzmann_n = N
        Yields a snapshot every N accepted moves, after `burnin` trials. This
        samples the Metropolis distribution at `temperature` rather than
        collecting minima, so it is the right mode if you want the *shape* of
        the linker conformational distribution rather than a set of good
        models. Yields many more structures per unit compute.

        Two things it buys you and one it costs: consecutive snapshots are
        correlated (raise N until they are not - check by plotting an
        autocorrelation of some geometric observable), the count per trajectory
        varies with the acceptance rate rather than being fixed, and the
        distribution is only meaningful if the chain has equilibrated, which is
        what burnin is for.
    """
    mc = MonteCarlo(pose, sfxn, temperature)
    accepted = 0

    for t in range(trials):
        perturb_loop_torsions(pose, cfg, rng, sigma, n_perturb)
        for closer in closers:
            closer.apply(pose)
        packer.apply(pose)
        # boltzmann() returns whether the move was accepted
        was_accepted = mc.boltzmann(pose)

        if boltzmann_n and was_accepted and t >= burnin:
            accepted += 1
            if accepted % boltzmann_n == 0:
                yield pose.clone()

    if not boltzmann_n:
        mc.recover_low(pose)
        yield pose


# --------------------------------------------------------------------------- #
# 5. Filtering  (PLACEHOLDER)
# --------------------------------------------------------------------------- #

def segment_tilt(pose, seg: Segment) -> float:
    """Unsigned angle in degrees between a rigid segment's axis and the
    membrane normal."""
    ax = segment_axis(pose, seg)
    n = MEMBRANE_NORMAL
    dot = sum(a * b for a, b in zip(ax, n))
    nn = math.sqrt(sum(c * c for c in n))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, dot / nn))))
    return min(ang, 180.0 - ang)


def chainbreak_score(pose, sfxn) -> float:
    sfxn(pose)
    e = pose.energies().total_energies()
    return float(e[ScoreType.chainbreak] + e[ScoreType.linear_chainbreak])


def passes_filters(pose, cfg: Construct, sfxn) -> Tuple[bool, dict]:
    """
    PLACEHOLDER. Returns (keep?, metrics_dict).

    Everything is currently computed and reported but nothing is rejected, so
    you can run the sampling first, look at the distributions, and only then
    decide where the thresholds belong.

    Criteria to decide on:
      - chainbreak: did the loops actually close? (this one is not optional)
      - tilt vs membrane normal (single-pass TM helices are typically well
        under ~40 deg)
      - membrane depth/span: does the hydrophobic stretch sit across a ~30 A
        slab without the hydrophobic core protruding?
      - total score / clash check
      - has a domain been pushed into the bilayer?
      - TODO: if you move to RosettaMP, replace the geometric checks with
        AddMembraneMover + a membrane-aware score function (franklin2019)
    """
    metrics = {
        "total_score": float(sfxn(pose)),
        "chainbreak": chainbreak_score(pose, sfxn),
    }
    for s in cfg.flexible:
        metrics[f"gap_{s.name}"] = round(loop_gap(pose, s), 3)
    for s in cfg.rigid:
        metrics[f"tilt_{s.name}"] = round(segment_tilt(pose, s), 2)
        metrics[f"z_{s.name}"] = round(ca_xyz(pose, s.mid).z, 2)

    keep = True  # TODO: apply real thresholds
    return keep, metrics


# --------------------------------------------------------------------------- #
# Output: one silent file, or one PDB per model
# --------------------------------------------------------------------------- #

class SilentWriter:
    """
    Collect every structure into a single silent file instead of one PDB each.

    BinarySilentStruct (Cartesian coordinates) rather than ProteinSilentStruct
    (torsions + ideal geometry): closure leaves non-ideal geometry at the
    cutpoints by construction, and a torsion-only representation would quietly
    regularise away exactly the chainbreak geometry the filter measures.

    Every metric and every drawn placement DOF is attached with add_energy(),
    so they become SCORE: columns riding along inside the same file. That is
    the real reason to use this format - the coordinates and their metadata
    can no longer drift apart. Recover the table with

        grep "^SCORE:" ensemble.out

    and individual structures with

        extract_pdbs -in:file:silent ensemble.out -in:file:tags model_003_007

    Structures are appended one at a time rather than accumulated and flushed
    at the end, so memory stays flat and a killed job still leaves a usable
    partial file.
    """

    def __init__(self, path, sfxn, verify: bool = True):
        from pyrosetta.rosetta.core.io.silent import SilentFileData, BinarySilentStruct

        # SilentFileOptions is required in newer builds and absent in older
        # ones, where SilentFileData() took no arguments.
        try:
            from pyrosetta.rosetta.core.io.silent import SilentFileOptions
            self.opts = SilentFileOptions()
            self.sfd = SilentFileData(self.opts)
            self._new = lambda pose, tag: BinarySilentStruct(self.opts, pose, tag)
        except ImportError:
            self.opts = None
            self.sfd = SilentFileData()
            self._new = lambda pose, tag: BinarySilentStruct(pose, tag)

        self.path = str(path)
        self.sfxn = sfxn
        self.verify = verify
        self.n = 0

    def add(self, pose, tag: str, energies: Dict[str, float]) -> None:
        ss = self._new(pose, tag)
        for k, v in energies.items():
            ss.add_energy(k, float(v))
        if self.verify and self.n == 0:
            self._round_trip_check(ss, pose)
        self.sfd.write_silent_struct(ss, self.path)
        self.n += 1

    def _round_trip_check(self, ss, original) -> None:
        """
        Encode/decode the first structure and confirm the chainbreak term is
        unchanged. Cutpoint variants travel in the annotated sequence and
        should survive, but if they do not, every structure in the file would
        score differently than it did at generation time - much better to find
        that out on structure one than after a production run.
        """
        self.verify = False
        try:
            back = pyrosetta.Pose()
            ss.fill_pose(back)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [silent] WARNING: could not rebuild a pose from the "
                  f"silent struct ({exc}). Some builds need an explicit "
                  f"residue type set passed to fill_pose().")
            return

        before = chainbreak_score(original, self.sfxn)
        after = chainbreak_score(back, self.sfxn)
        delta = abs(before - after)
        print(f"  [silent] round-trip chainbreak {before:.4f} -> {after:.4f} "
              f"({'OK' if delta < 1e-3 else 'MISMATCH'})")
        if delta >= 1e-3:
            print("  [silent] cutpoint variants may not be surviving the "
                  "encoding - verify before trusting extracted structures.")


# --------------------------------------------------------------------------- #
# Run context and per-placement work (shared by serial and parallel paths)
# --------------------------------------------------------------------------- #

# Per-cell silent files live here until they are merged into --silent-name.
PARTS_DIRNAME = "parts"


def init_rosetta(seed: int) -> None:
    pyrosetta.init(
        f"-ex1 -ex2aro -use_input_sc -mute all -constant_seed -jran {seed or 1}"
    )


@dataclass
class RunContext:
    """Everything a placement needs that is built once per process.

    PyRosetta objects (poses, score functions, movers) cannot be pickled, so
    the parent never ships this to a worker: each worker builds its own copy
    from the same `args` in its pool initializer.
    """
    args: argparse.Namespace
    outdir: Path
    reference: object
    cfg: Construct
    jumps: Dict[str, int]
    sfxn: object
    closers: list
    packer: object
    mobiles: List[Segment]
    placement_mode: str
    anchor_positions: Dict[str, object]
    burnin: int


def build_context(args: argparse.Namespace) -> RunContext:
    """Load the input, build fold tree / score function / movers. No output."""
    reference = pyrosetta.pose_from_pdb(args.pdb)
    cfg = to_pose_numbering(reference, load_construct(args.config))
    jumps = build_fold_tree(reference, cfg)
    loops = build_loops(cfg)

    sfxn = make_score_function(args.chainbreak_weight)
    movemap = make_loop_movemap(cfg)
    closers = make_closure_movers(loops, movemap)
    packer = make_repack_mover(sfxn, cfg)

    mobiles = [s for s in cfg.rigid if s.dof.is_mobile]
    placement_mode = mobiles[0].dof.mode if mobiles else "random"

    # Anchors are required to be non-mobile (checked in Construct.validate),
    # so their C-terminal CA is unchanged by any placement and can be
    # measured once from the reference pose.
    anchor_positions: Dict[str, object] = {}
    if placement_mode == "grid":
        for s in mobiles:
            if s.dof.anchor and s.dof.anchor not in anchor_positions:
                anchor_positions[s.dof.anchor] = anchor_position(
                    reference, cfg, s.dof.anchor)

    burnin = (args.burnin if args.burnin is not None
              else (args.trials // 5 if args.boltzmann_n else 0))

    return RunContext(args=args, outdir=Path(args.out), reference=reference,
                      cfg=cfg, jumps=jumps, sfxn=sfxn, closers=closers,
                      packer=packer, mobiles=mobiles,
                      placement_mode=placement_mode,
                      anchor_positions=anchor_positions, burnin=burnin)


def make_placement_record(args, p_idx: int, spec: Dict[str, Dict[str, float]],
                          viable: bool, gap_note: str = "",
                          n_traj_run: Optional[int] = None,
                          extra: Optional[dict] = None) -> dict:
    if n_traj_run is None:
        n_traj_run = args.n_traj if viable else 0
    prec = {"placement": p_idx, "viable": bool(viable), "n_traj_run": n_traj_run}
    for name, params in spec.items():
        prec.update({f"{name}_{k}": round(float(v), 3)
                     for k, v in params.items()})
    if gap_note:
        prec["gap_note"] = gap_note
    if extra:
        prec.update(extra)
    return prec


def run_trajectories(ctx: RunContext, placed, p_idx: int,
                     spec: Dict[str, Dict[str, float]], rng: random.Random,
                     writer: Optional[SilentWriter]) -> Tuple[List[dict], int]:
    """Run n_traj MC trajectories on `placed`; write kept structures.

    Returns (model records, number of structures kept). Records are returned
    rather than appended to shared state so this works inside a worker.
    """
    args = ctx.args
    records: List[dict] = []
    n_kept = 0
    for t_idx in range(args.n_traj):
        start = placed.clone()
        for s_idx, pose in enumerate(mc_torsion_sampling(
                start, ctx.cfg, ctx.sfxn, ctx.closers, ctx.packer, rng,
                trials=args.trials,
                temperature=args.temperature,
                sigma=args.sigma,
                n_perturb=args.n_perturb,
                boltzmann_n=args.boltzmann_n,
                burnin=ctx.burnin)):
            keep, m = passes_filters(pose, ctx.cfg, ctx.sfxn)
            tag = (f"model_{p_idx:03d}_{t_idx:03d}_{s_idx:04d}"
                   if args.boltzmann_n else f"model_{p_idx:03d}_{t_idx:03d}")

            rec = {"tag": tag, "placement": p_idx, "traj": t_idx,
                   "snapshot": s_idx, "kept": keep, **m}
            for name, params in spec.items():
                rec.update({f"{name}_{k}": round(float(v), 3)
                            for k, v in params.items()})
            records.append(rec)

            if keep:
                if writer is not None:
                    # Everything numeric in the record becomes a SCORE:
                    # column, so metrics and placement DOF live inside the
                    # silent file alongside the coordinates.
                    energies = {k: float(v) for k, v in rec.items()
                                if k != "tag" and isinstance(v, (int, float, bool))}
                    writer.add(pose, tag, energies)
                else:
                    pose.dump_pdb(str(ctx.outdir / f"{tag}.pdb"))
                n_kept += 1
    return records, n_kept


# --------------------------------------------------------------------------- #
# Grid mode: one task per cell, optionally across worker processes
# --------------------------------------------------------------------------- #

def cell_seed(seed: int, p_idx: int) -> int:
    """Deterministic per-cell seed in [1, 2**31 - 2].

    Hashing (seed, p_idx) rather than e.g. seed + p_idx keeps neighbouring
    cells from getting related streams. Because the seed depends only on the
    cell, a cell's structures are the same whichever worker runs it and in
    whatever order - so --cores 1 and --cores 36 give the same ensemble.
    (str seeds go through SHA-512 in `random`, so PYTHONHASHSEED is irrelevant.)
    """
    return random.Random(f"{seed}:{p_idx}").randrange(1, 2**31 - 1)


_ROSETTA_RESEED_WARNED = False


def reseed_rosetta(seed: int) -> None:
    """Reseed Rosetta's global RNG (used by the packer and MonteCarlo).

    Without this, each worker's Rosetta stream would depend on which cells it
    happened to run before, breaking core-count independence.
    """
    global _ROSETTA_RESEED_WARNED
    try:
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)
    except AttributeError as exc:
        if not _ROSETTA_RESEED_WARNED:
            print(f"WARNING: could not reseed Rosetta's RNG ({exc}). Grid "
                  "results will depend on --cores and scheduling.", flush=True)
            _ROSETTA_RESEED_WARNED = True


def run_grid_cell(ctx: RunContext, p_idx: int, spec: Dict[str, Dict[str, float]],
                  verify_silent: bool) -> dict:
    """Run one viable grid cell into its own silent file (or PDBs)."""
    args = ctx.args
    seed = cell_seed(args.seed, p_idx)
    rng = random.Random(seed)
    reseed_rosetta(seed)

    part_path = None
    writer = None
    if args.silent:
        part_path = ctx.outdir / PARTS_DIRNAME / f"p{p_idx:05d}.out"
        if part_path.exists():          # never append to a stale part
            part_path.unlink()
        writer = SilentWriter(part_path, ctx.sfxn, verify=verify_silent)

    placed = ctx.reference.clone()
    apply_placement(placed, ctx.cfg, ctx.jumps, spec, ctx.anchor_positions)
    records, n_kept = run_trajectories(ctx, placed, p_idx, spec, rng, writer)

    # SilentWriter only creates the file on the first add(), so a cell that
    # kept nothing has no part file.
    return {"p_idx": p_idx, "records": records, "n_kept": n_kept,
            "part": str(part_path) if (part_path is not None and n_kept) else None,
            "error": None}


def _safe_run_cell(ctx: RunContext, task) -> dict:
    """Wrap run_grid_cell so one failing cell doesn't take down the run.

    A failed cell's records are discarded and its part file (possibly
    partial) is not merged, so the silent file never holds structures that
    metrics.json doesn't know about.
    """
    p_idx, spec, verify = task
    try:
        return run_grid_cell(ctx, p_idx, spec, verify)
    except Exception:                                              # noqa: BLE001
        return {"p_idx": p_idx, "records": [], "n_kept": 0, "part": None,
                "error": traceback.format_exc()}


_WORKER_CTX: Optional[RunContext] = None


def _worker_init(args: argparse.Namespace) -> None:
    global _WORKER_CTX
    init_rosetta(args.seed)
    _WORKER_CTX = build_context(args)


def _worker_run_cell(task) -> dict:
    return _safe_run_cell(_WORKER_CTX, task)


def merge_silent_parts(parts: List[Path], dest: Path) -> Tuple[int, bool]:
    """Concatenate per-cell silent files into `dest` with a single header.

    Each part was written as a fresh file, so each starts with its own header
    (SEQUENCE:, the SCORE: column-name line, any REMARK lines). The first part
    is copied verbatim; for the rest, everything before their first *data*
    SCORE: line (the second SCORE: line) is dropped. Plain `cat` would leave
    repeated column-name lines, which parse_score_table() in the analysis
    scripts would read as data rows.

    Written to a .tmp file and renamed, so an interrupted merge never leaves a
    half-written ensemble.out. Returns (structures written, headers consistent).
    """
    n_structs = 0
    header_cols: Optional[List[str]] = None
    consistent = True
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "w") as out:
        for i, part in enumerate(parts):
            n_score = 0
            with open(part) as fh:
                for line in fh:
                    if line.startswith("SCORE:"):
                        n_score += 1
                        if n_score == 1:
                            cols = line.split()
                            if header_cols is None:
                                header_cols = cols
                            elif cols != header_cols:
                                consistent = False
                        else:
                            n_structs += 1
                    if i > 0 and n_score < 2:
                        continue
                    if not line.endswith("\n"):
                        line += "\n"
                    out.write(line)
    tmp.replace(dest)
    return n_structs, consistent


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", required=True, help="Input structure")
    p.add_argument("--config", required=True, help="Construct definition YAML")
    p.add_argument("--out", default="ensemble", help="Output directory")

    p.add_argument("--n-placements", type=int, default=20,
                   help="Random mode only: number of viable placements to keep. "
                        "Ignored in grid mode (every grid cell is visited once).")
    p.add_argument("--n-traj", type=int, default=10,
                   help="MC trajectories per placement (both modes)")
    p.add_argument("--trials", type=int, default=500, help="MC trials per trajectory")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=25.0,
                   help="Std dev (deg) of per-residue torsion perturbation")
    p.add_argument("--n-perturb", type=int, default=2,
                   help="Loop residues perturbed per MC trial")
    p.add_argument("--boltzmann-n", type=int, default=None,
                   help="If set, snapshot the pose every N accepted moves "
                        "instead of keeping only the lowest-energy structure "
                        "per trajectory. Samples the Metropolis distribution "
                        "rather than collecting minima; yields many more "
                        "structures. Raise N to decorrelate consecutive "
                        "snapshots.")
    p.add_argument("--burnin", type=int, default=None,
                   help="MC trials to discard before snapshotting, so the "
                        "chain has equilibrated. Only used with --boltzmann-n; "
                        "defaults to trials/5.")
    p.add_argument("--chainbreak-weight", type=float, default=1.0)
    p.add_argument("--span-slack", type=float, default=0.9,
                   help="Fraction of maximal loop extension allowed (0-1)")
    p.add_argument("--max-attempts-factor", type=int, default=20,
                   help="Random mode only: give up after n-placements * this "
                        "many rejected draws")
    p.add_argument("--silent", action=argparse.BooleanOptionalAction, default=True,
                   help="Write all structures into one silent file (default). "
                        "Use --no-silent for one PDB per model.")
    p.add_argument("--silent-name", default="ensemble.out",
                   help="Silent file name, inside --out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cores", type=int, default=1,
                   help="Grid mode only: worker processes, one grid cell per "
                        "task. Each worker loads its own PyRosetta, so budget "
                        "memory per core. Results do not depend on this value.")
    p.add_argument("--keep-parts", action="store_true",
                   help="Grid mode: keep the per-cell silent files in "
                        f"<out>/{PARTS_DIRNAME}/ after a successful merge")

    g = p.add_argument_group(
        "grid sharding",
        "Split one grid across independent runs (e.g. SLURM array tasks on "
        "separate nodes). Shards are disjoint contiguous windows of the same "
        "cell enumeration, so no two shards ever run the same cell and the "
        "union of all shards is the whole grid. Cell indices, per-cell seeds "
        "and structure tags are global, so a sharded campaign yields exactly "
        "the same ensemble as one unsharded run.")
    g.add_argument("--n-shards", type=int, default=1,
                   help="Split the grid into this many equal-sized shards.")
    g.add_argument("--shard", type=int, default=0,
                   help="Which shard this run executes (0-based). Set from "
                        "$SLURM_ARRAY_TASK_ID.")
    g.add_argument("--cell-start", type=int, default=None,
                   help="Explicit lower bound (inclusive) of the global cell "
                        "index window. Overrides --shard/--n-shards.")
    g.add_argument("--cell-stop", type=int, default=None,
                   help="Explicit upper bound (exclusive) of the global cell "
                        "index window. Overrides --shard/--n-shards.")
    return p.parse_args()


def shard_window(n_cells: int, args: argparse.Namespace) -> Tuple[int, int]:
    """Resolve this run's [start, stop) window into the global cell list.

    Explicit --cell-start/--cell-stop win if either is given. Otherwise the
    grid is cut into --n-shards contiguous windows; the first
    (n_cells % n_shards) shards take one extra cell, so the windows tile the
    grid exactly with no gap, no overlap and a size spread of at most one.
    """
    if args.cell_start is not None or args.cell_stop is not None:
        start = 0 if args.cell_start is None else args.cell_start
        stop = n_cells if args.cell_stop is None else args.cell_stop
        if not 0 <= start <= stop <= n_cells:
            raise SystemExit(
                f"--cell-start/--cell-stop [{start}, {stop}) is not inside "
                f"[0, {n_cells}) for this grid.")
        return start, stop

    if args.n_shards < 1:
        raise SystemExit("--n-shards must be >= 1")
    if not 0 <= args.shard < args.n_shards:
        raise SystemExit(
            f"--shard must be in [0, {args.n_shards}), got {args.shard}")

    base, rem = divmod(n_cells, args.n_shards)
    start = args.shard * base + min(args.shard, rem)
    stop = start + base + (1 if args.shard < rem else 0)
    return start, stop


def main():
    args = parse_args()
    if args.cores < 1:
        raise SystemExit("--cores must be >= 1")
    rng = random.Random(args.seed)     # random mode only; grid seeds per cell

    init_rosetta(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(args)
    reference, cfg, jumps = ctx.reference, ctx.cfg, ctx.jumps
    mobiles, placement_mode = ctx.mobiles, ctx.placement_mode
    anchor_positions, burnin = ctx.anchor_positions, ctx.burnin

    if placement_mode != "grid" and args.cores > 1:
        raise SystemExit("--cores > 1 is only supported in grid mode.")

    sharded = (args.n_shards > 1 or args.cell_start is not None
               or args.cell_stop is not None)
    if placement_mode != "grid" and sharded:
        raise SystemExit(
            "--n-shards/--cell-start/--cell-stop are only supported in grid "
            "mode. Random mode has no fixed cell enumeration to partition; "
            "give each run a different --seed instead.")

    print("Rigid    : " + ", ".join(
        f"{s.name} {s.start}-{s.stop}"
        + ("" if s.name not in jumps else f" (jump {jumps[s.name]})")
        + ("" if s.dof.is_mobile or s.name not in jumps else " [no dof]")
        for s in cfg.rigid))
    print("Flexible : " + ", ".join(f"{s.name} {s.start}-{s.stop} cut {s.mid}"
                                    for s in cfg.flexible))
    for s in cfg.flexible:
        reach = MAX_SPAN_PER_RESIDUE * (s.length + 1) * args.span_slack
        print(f"           {s.name}: {s.length} res, max usable span {reach:.1f} A, "
              f"gap in input {loop_gap(reference, s):.1f} A")
    print(reference.fold_tree())

    if args.boltzmann_n:
        print(f"Sampling : Boltzmann snapshots every {args.boltzmann_n} accepted "
              f"moves, after {burnin} burn-in trials (T = {args.temperature})")
        if burnin >= args.trials:
            raise SystemExit("--burnin is >= --trials; nothing would be sampled")
    else:
        print("Sampling : lowest-energy structure per trajectory (recover_low)")

    silent_path = outdir / args.silent_name
    parts_dir = outdir / PARTS_DIRNAME
    writer = None
    if args.silent:
        if silent_path.exists():
            raise SystemExit(
                f"{silent_path} already exists. Silent files are appended to, so "
                "a rerun would mix old and new structures - move or delete it first."
            )
        if placement_mode == "grid":
            if parts_dir.exists() and any(parts_dir.iterdir()):
                raise SystemExit(
                    f"{parts_dir} is not empty (left over from an earlier run?) "
                    "- move or delete it first.")
            parts_dir.mkdir(exist_ok=True)
            print(f"Output   : silent file {silent_path} "
                  f"(per-cell parts in {parts_dir}/, merged at the end)")
        else:
            writer = SilentWriter(silent_path, ctx.sfxn)
            print(f"Output   : silent file {silent_path}")
    else:
        print(f"Output   : one PDB per model in {outdir}/")

    # ----- Placement setup -----------------------------------------------
    if placement_mode == "grid":
        grid_placements = list(enumerate_grid_placements(mobiles))
        n_cells_total = len(grid_placements)
        cell_start, cell_stop = shard_window(n_cells_total, args)
        # Keep the GLOBAL index as p_idx: it drives cell_seed(), the model
        # tags and the metrics records, so a cell behaves identically no
        # matter which shard happens to run it.
        shard_cells = list(enumerate(grid_placements))[cell_start:cell_stop]
        print(f"Placement: grid, {n_cells_total} cells")
        if sharded:
            print(f"Shard    : {args.shard + 1}/{args.n_shards} -> global "
                  f"cells [{cell_start}, {cell_stop}) = {len(shard_cells)} "
                  f"cells on this node")
        if not shard_cells:
            raise SystemExit(
                f"This shard's window [{cell_start}, {cell_stop}) is empty "
                f"({n_cells_total} cells over {args.n_shards} shards). "
                "Use fewer shards than cells.")
        for s in mobiles:
            if not s.dof.grid_axes:
                continue
            axes_desc = "; ".join(
                f"{ax} [{mn}..{mx} step {st}, {len(_axis_values(mn, mx, st))} pts]"
                for ax, (mn, mx, st) in s.dof.grid_axes.items())
            fixed_desc = ", ".join(
                f"{k}={v}" for k, v in [
                    ("x", s.dof.tx), ("y", s.dof.ty), ("z", s.dof.tz),
                    ("tilt", s.dof.tilt), ("spin", s.dof.spin)]
                if k not in s.dof.grid_axes)
            anchor_seg = next(x for x in cfg.rigid if x.name == s.dof.anchor)
            ap = anchor_positions[s.dof.anchor]
            print(f"           {s.name}: axes {axes_desc}")
            if fixed_desc:
                print(f"           {s.name}: fixed {fixed_desc}")
            print(f"           {s.name}: anchor {s.dof.anchor} "
                  f"(CA of res {anchor_seg.stop} at "
                  f"{ap.x:.2f}, {ap.y:.2f}, {ap.z:.2f})")
    else:
        max_attempts = max(args.n_placements * args.max_attempts_factor,
                           args.n_placements)
        print(f"Placement: random, target {args.n_placements} placements "
              f"(up to {max_attempts} attempts)")

    records: List[dict] = []
    placement_records: List[dict] = []
    n_written = 0

    # Compact per-placement log: show the DOF keys that are non-zero for at
    # least one placement in this run (grid axes always qualify; fixed
    # scalars only if non-zero). Keeps the log narrow in the common case
    # where only x, y vary.
    log_keys = {}
    for s in mobiles:
        keys = []
        for axis, key in _AXIS_TO_KEY.items():
            if axis in s.dof.grid_axes or getattr(
                    s.dof, {"x": "tx", "y": "ty", "z": "tz",
                            "tilt": "tilt", "spin": "spin"}[axis]):
                keys.append(key)
        log_keys[s.name] = keys or ["dx", "dy", "dz"]

    def log_placement(p_idx: int, total: int,
                      spec: Dict[str, Dict[str, float]], body: str) -> None:
        parts = []
        for name, params in spec.items():
            parts.append(" ".join(f"{k}={params[k]:+.2f}" for k in log_keys[name]
                                  if k in params))
        head = f"[placement {p_idx:>4d}/{total}]"
        print(f"{head} {' | '.join(parts)}  {body}", flush=True)

    # ----- Main loop -----------------------------------------------------
    failed_cells: List[int] = []
    merge_ok = True
    if placement_mode == "grid":
        total = n_cells_total

        # Viability is cheap geometry, so the parent screens every cell in
        # its own shard and only viable ones are dispatched. This also fixes
        # which cell runs the silent round-trip check (the first viable one
        # in this shard) up front.
        viable_cells = []
        for p_idx, spec in shard_cells:
            placed = reference.clone()
            apply_placement(placed, cfg, jumps, spec, anchor_positions)
            if not placement_is_viable(placed, cfg, args.span_slack):
                gaps = "; ".join(f"{s.name} gap {loop_gap(placed, s):.1f}A"
                                 for s in cfg.flexible)
                placement_records.append(make_placement_record(
                    args, p_idx, spec, viable=False, gap_note=gaps))
                log_placement(p_idx, total, spec, f"not viable ({gaps})")
                continue
            viable_cells.append((p_idx, spec))

        tasks = [(p_idx, spec, i == 0) for i, (p_idx, spec) in enumerate(viable_cells)]
        specs = dict(viable_cells)
        n_workers = min(args.cores, len(tasks))
        print(f"Running  : {len(tasks)} viable cells "
              f"on {max(n_workers, 1)} process(es)", flush=True)

        pool = None
        if n_workers > 1:
            # spawn, not fork: each worker starts clean and runs its own
            # pyrosetta.init, instead of inheriting Rosetta's global state.
            mp = multiprocessing.get_context("spawn")
            pool = mp.Pool(n_workers, initializer=_worker_init, initargs=(args,))
            results = pool.imap_unordered(_worker_run_cell, tasks, chunksize=1)
        else:
            results = (_safe_run_cell(ctx, t) for t in tasks)

        part_files: List[Path] = []
        done = 0
        try:
            for res in results:
                done += 1
                p_idx = res["p_idx"]
                spec = specs[p_idx]
                progress = f"[{done}/{len(tasks)} done]"
                seed_info = {"seed": cell_seed(args.seed, p_idx)}
                if res["error"]:
                    failed_cells.append(p_idx)
                    last = res["error"].strip().splitlines()[-1]
                    placement_records.append(make_placement_record(
                        args, p_idx, spec, viable=True, n_traj_run=0,
                        extra={**seed_info, "error": last}))
                    log_placement(p_idx, total, spec, f"FAILED: {last}  {progress}")
                    print(res["error"], flush=True)
                    continue
                placement_records.append(make_placement_record(
                    args, p_idx, spec, viable=True, extra=seed_info))
                records.extend(res["records"])
                n_written += res["n_kept"]
                if res["part"]:
                    part_files.append(Path(res["part"]))
                log_placement(p_idx, total, spec,
                              f"{args.n_traj} trajectories, {res['n_kept']} "
                              f"structures  {progress}")
        except BaseException:
            if pool is not None:
                pool.terminate()
                pool.join()
            raise
        else:
            if pool is not None:
                pool.close()
                pool.join()

        # ----- Merge per-cell silent files -------------------------------
        if args.silent:
            part_files.sort()            # p00000.out, p00001.out, ... = serial order
            n_merged, consistent = (merge_silent_parts(part_files, silent_path)
                                    if part_files else (0, True))
            merge_ok = consistent and n_merged == n_written
            if not consistent:
                print("WARNING: per-cell silent files have different SCORE "
                      "columns; the merged SCORE table may be misaligned.")
            if n_merged != n_written:
                print(f"WARNING: merged {n_merged} structures but workers "
                      f"reported {n_written}.")
            if not merge_ok or args.keep_parts:
                print(f"Parts kept in {parts_dir}/")
            else:
                shutil.rmtree(parts_dir)
    else:
        kept_placements = 0
        attempts = 0
        while kept_placements < args.n_placements and attempts < max_attempts:
            attempts += 1
            placed = reference.clone()
            spec = draw_random_placement(mobiles, rng)
            apply_placement(placed, cfg, jumps, spec, anchor_positions)

            if not placement_is_viable(placed, cfg, args.span_slack):
                continue

            p_idx = kept_placements
            placement_records.append(make_placement_record(
                args, p_idx, spec, viable=True))
            cell_records, n_kept = run_trajectories(ctx, placed, p_idx, spec,
                                                    rng, writer)
            records.extend(cell_records)
            n_written += n_kept
            kept_placements += 1
            log_placement(p_idx, args.n_placements, spec,
                          f"{args.n_traj} trajectories, {n_kept} structures "
                          f"({kept_placements}/{args.n_placements})")

    # Grid results arrive in completion order; put everything back in
    # placement order so metrics.json looks the same as a serial run.
    placement_records.sort(key=lambda r: r["placement"])
    records.sort(key=lambda r: (r["placement"], r["traj"], r["snapshot"]))

    # ----- Write metrics.json -------------------------------------------
    metrics_out: Dict[str, object] = {
        "config": args.config, "pdb": args.pdb, "seed": args.seed,
        "mode": "boltzmann" if args.boltzmann_n else "recover_low",
        "placement_mode": placement_mode,
        "boltzmann_n": args.boltzmann_n, "burnin": burnin,
        "temperature": args.temperature, "trials": args.trials,
        "n_traj": args.n_traj,
        "cores": args.cores,
        "placement_count": len(placement_records),
        "placements": placement_records,
        "models": records,
    }
    if placement_mode == "grid":
        metrics_out["rng"] = "per_placement"
        metrics_out["failed_placements"] = sorted(failed_cells)
        metrics_out["shard"] = {
            "index": args.shard, "n_shards": args.n_shards,
            "cell_start": cell_start, "cell_stop": cell_stop,
            "n_cells_total": n_cells_total,
            "n_cells_this_shard": len(shard_cells),
        }
        metrics_out["grid"] = {
            s.name: {
                "anchor": s.dof.anchor,
                "anchor_position": {
                    "x": float(anchor_positions[s.dof.anchor].x),
                    "y": float(anchor_positions[s.dof.anchor].y),
                    "z": float(anchor_positions[s.dof.anchor].z),
                },
                "axes": [
                    {"axis": ax, "min": mn, "max": mx, "step": st,
                     "values": _axis_values(mn, mx, st)}
                    for ax, (mn, mx, st) in s.dof.grid_axes.items()
                ],
                "fixed": {k: v for k, v in [
                    ("x", s.dof.tx), ("y", s.dof.ty), ("z", s.dof.tz),
                    ("tilt", s.dof.tilt), ("spin", s.dof.spin)]
                    if k not in s.dof.grid_axes},
            }
            for s in mobiles if s.dof.grid_axes
        }
    else:
        metrics_out["rng"] = "shared"
        metrics_out["placement_attempts"] = attempts

    with open(outdir / "metrics.json", "w") as fh:
        json.dump(metrics_out, fh, indent=2)

    if args.silent:
        print(f"\nDone. {n_written} structures written to {silent_path}")
        print(f"      scores : grep '^SCORE:' {silent_path}")
        print(f"      extract: extract_pdbs -in:file:silent {silent_path} "
              f"-in:file:tags <tag> [<tag> ...]")
    else:
        print(f"\nDone. {n_written} structures written to {outdir}/")

    if placement_mode == "random" and kept_placements < args.n_placements:
        print("Hit the placement attempt limit: the dof limits are probably "
              "pushing the loops past what they can span. Reduce the "
              "translation maxima, or check the reach budget above.")

    if failed_cells:
        print(f"\n{len(failed_cells)} placement(s) failed: {sorted(failed_cells)}. "
              "They are in metrics.json with n_traj_run = 0 and an 'error' field.")
    if failed_cells or not merge_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()