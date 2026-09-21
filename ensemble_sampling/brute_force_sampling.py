#!/usr/bin/env python3
"""
No-closure ensemble sampling of flexible linkers.

Alternative to ensemble_sampling.py. That script severs each flexible segment
at a cutpoint, moves the rigid bodies as independent jumps, and closes the
break back up with CCD/KIC. Here there is NO cutpoint and NO closure: the chain
is one unbroken kinematic entity, and torsional Monte Carlo on the loop
residues moves everything downstream as a rigid consequence of the backbone
change. Every pose that comes out is therefore closed by construction - there
is no chainbreak term to satisfy, because there is no break.

    [ F3 ]===linker===[ TM ]
      ^ root         ^ dragged along by the linker torsions

The price is that you do NOT get to place the TM where you want and ask the
loop to reach it. Instead the TM lands wherever the sampled linker torsions
put it. Most draws will be orientations incompatible with membrane binding;
you keep the ones that are (filtering is a later conversation - passes_filters
here is the same permissive placeholder as in the closure script).

This is deliberately the "measure the success rate" tool. Expect a low hit
rate; that is the number you are trying to obtain.

Two-loop / alpha-tag strategy (as discussed): sample the F3<->alpha loop with
THIS script, take the best-scoring / membrane-compatible solutions as fixed
input structures, then sample the alpha<->TM loop with the closure script
(ensemble_sampling.py), which can be told to close onto a target TM placement.
To do that here, mark the alpha<->TM loop's residues as part of a rigid
segment (or simply omit them from the flexible list) so this run only moves the
F3<->alpha torsions.

    python ensemble_sampling_noclosure.py --pdb tagged.pdb --config tagged.yaml \
        --n-traj 200 --trials 1000 --boltzmann-n 5 --out ensemble_noclosure/

Reuses, unchanged in spirit, from ensemble_sampling.py:
  - the YAML construct definition and pose-numbering translation
  - the loop movemap and neighbourhood repack task
  - the Gaussian torsion perturbation
  - passes_filters and the SilentWriter

Dropped entirely:
  - build_fold_tree / cutpoints / add_single_cutpoint_variant
  - build_loops / CCDLoopClosureMover / make_closure_movers
  - sample_placement and the whole rigid-body DOF machinery (dof blocks in the
    YAML are ignored here; a warning is printed if any are present)
  - the chainbreak score terms (there is no break to score)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

import pyrosetta
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.core.select.residue_selector import (
    NeighborhoodResidueSelector, ResidueIndexSelector,
)
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.moves import MonteCarlo


# --------------------------------------------------------------------------- #
# Global conventions
# --------------------------------------------------------------------------- #

MEMBRANE_NORMAL = (0.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 1. Construct definition (from YAML) - same schema as the closure script,
#    minus any meaning for the dof block.
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    """One segment. start/stop are inclusive PDB residue numbers."""
    name: str
    kind: str          # "rigid" | "flexible"
    start: int
    stop: int
    chain: str = "A"
    has_dof: bool = False   # only tracked so we can warn that it's ignored

    def __post_init__(self):
        if self.kind not in ("rigid", "flexible"):
            raise ValueError(f"{self.name}: kind must be 'rigid' or 'flexible'")
        if self.stop < self.start:
            raise ValueError(f"{self.name}: stop < start ({self.start}-{self.stop})")

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

    def loop_residues(self) -> List[int]:
        out: List[int] = []
        for s in self.flexible:
            out.extend(s.residues())
        return out

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


def load_construct(path: str) -> Construct:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = Construct([
        Segment(name=e["name"], kind=e["kind"],
                start=int(e["start"]), stop=int(e["stop"]),
                chain=str(e.get("chain", "A")),
                has_dof=bool(e.get("dof")))
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
                chain=s.chain, has_dof=s.has_dof)
        for s in cfg.segments
    ])


# --------------------------------------------------------------------------- #
# 2. FoldTree: the trivial linear one. No jumps, no cutpoints.
# --------------------------------------------------------------------------- #

def set_linear_fold_tree(pose) -> None:
    """
    Use a simple peptide-edge FoldTree spanning the whole pose, rooted at
    residue 1. This is what a freshly loaded pose already has, but we assert it
    explicitly so that a pose carrying a jump/cutpoint tree from an earlier
    protocol is reset.

    With this tree, setting phi/psi on a loop residue propagates through every
    residue downstream of it: the entire C-terminal portion (including the TM
    body) rotates about that backbone bond. That propagation is exactly what
    keeps the chain closed without a closure move.
    """
    from pyrosetta.rosetta.core.kinematics import FoldTree
    ft = FoldTree(pose.total_residue())   # single peptide edge 1..N
    if not ft.check_fold_tree():
        raise RuntimeError("Failed to build a linear FoldTree")
    pose.fold_tree(ft)


# --------------------------------------------------------------------------- #
# 3. Score function (no chainbreak terms - there is no break)
# --------------------------------------------------------------------------- #

def make_score_function():
    return pyrosetta.create_score_function("ref2015")


# --------------------------------------------------------------------------- #
# 4. Movemap + repack, restricted to the flexible segments
#    (identical to the closure script)
# --------------------------------------------------------------------------- #

def make_loop_movemap(cfg: Construct) -> MoveMap:
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)
    for resid in cfg.loop_residues():
        mm.set_bb(resid, True)
        mm.set_chi(resid, True)
    return mm


def make_repack_mover(sfxn, cfg: Construct):
    loop_idx = cfg.loop_residues()
    sel = ResidueIndexSelector(",".join(str(i) for i in loop_idx))
    nbr = NeighborhoodResidueSelector(sel, 6.0, True)

    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    tf.push_back(operation.OperateOnResidueSubset(
        operation.PreventRepackingRLT(), nbr, True))

    packer = PackRotamersMover(sfxn)
    packer.task_factory(tf)
    return packer


# --------------------------------------------------------------------------- #
# 5. Torsion perturbation + MC
# --------------------------------------------------------------------------- #

def perturb_loop_torsions(pose, cfg: Construct, rng: random.Random,
                          sigma: float, n_residues: int) -> None:
    """
    Perturb phi/psi on a few randomly chosen loop residues.

    Set sigma large (or use --uniform) to recover the aggressive
    angle="random" behaviour of the original XML. With no closure to satisfy,
    large perturbations are fine mechanically - they just move the downstream
    body a long way, which lowers the acceptance rate but is not otherwise a
    problem. Small Gaussian steps explore locally; large/uniform steps explore
    the whole orientational space of the TM at the cost of accepting rarely.
    """
    candidates = cfg.loop_residues()
    for resid in rng.sample(candidates, min(n_residues, len(candidates))):
        if sigma is None:   # uniform mode
            pose.set_phi(resid, rng.uniform(-180.0, 180.0))
            pose.set_psi(resid, rng.uniform(-180.0, 180.0))
        else:
            pose.set_phi(resid, pose.phi(resid) + rng.gauss(0.0, sigma))
            pose.set_psi(resid, pose.psi(resid) + rng.gauss(0.0, sigma))


def mc_torsion_sampling(pose, cfg: Construct, sfxn, packer,
                        rng: random.Random, trials: int, temperature: float,
                        sigma: Optional[float], n_perturb: int,
                        boltzmann_n: Optional[int] = None, burnin: int = 0):
    """
    Metropolis MC over loop torsions: perturb -> repack -> accept/reject.

    No closure step. A generator; yields poses one at a time.

    boltzmann_n = None : one pose per trajectory, the lowest-energy one found
                         (recover_low). Good models, not a thermodynamic set.
    boltzmann_n = N    : a snapshot every N accepted moves after burnin. Samples
                         the Metropolis distribution - the right mode for asking
                         "what fraction of the accessible ensemble is
                         membrane-compatible", which is the success rate you
                         want. Consecutive snapshots are correlated; raise N to
                         decorrelate.
    """
    mc = MonteCarlo(pose, sfxn, temperature)
    accepted = 0

    for t in range(trials):
        perturb_loop_torsions(pose, cfg, rng, sigma, n_perturb)
        packer.apply(pose)
        was_accepted = mc.boltzmann(pose)

        if boltzmann_n and was_accepted and t >= burnin:
            accepted += 1
            if accepted % boltzmann_n == 0:
                yield pose.clone()

    if not boltzmann_n:
        mc.recover_low(pose)
        yield pose


# --------------------------------------------------------------------------- #
# 6. Filtering (same permissive placeholder as the closure script)
# --------------------------------------------------------------------------- #

def ca_xyz(pose, resid: int):
    return pose.residue(resid).xyz("CA")


def segment_axis(pose, seg: Segment) -> Tuple[float, float, float]:
    a, b = ca_xyz(pose, seg.start), ca_xyz(pose, seg.stop)
    v = (b.x - a.x, b.y - a.y, b.z - a.z)
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-9:
        raise ValueError(f"{seg.name}: degenerate axis")
    return (v[0] / n, v[1] / n, v[2] / n)


def segment_tilt(pose, seg: Segment) -> float:
    ax = segment_axis(pose, seg)
    n = MEMBRANE_NORMAL
    dot = sum(a * b for a, b in zip(ax, n))
    nn = math.sqrt(sum(c * c for c in n))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, dot / nn))))
    return min(ang, 180.0 - ang)


def loop_gap(pose, loop: Segment) -> float:
    return (ca_xyz(pose, loop.start - 1) - ca_xyz(pose, loop.stop + 1)).norm()


def passes_filters(pose, cfg: Construct, sfxn) -> Tuple[bool, dict]:
    """
    PLACEHOLDER - reports geometry, rejects nothing. Run first, look at the
    distributions, decide thresholds later. Note there is deliberately no
    chainbreak metric here: the chain is never broken.

    The gap_<loop> values are informational only (they measure the flanking
    CA-CA distance, which in no-closure mode is just wherever the torsions put
    the ends - not a closure residual).
    """
    metrics = {"total_score": float(sfxn(pose))}
    for s in cfg.flexible:
        metrics[f"gap_{s.name}"] = round(loop_gap(pose, s), 3)
    for s in cfg.rigid:
        metrics[f"tilt_{s.name}"] = round(segment_tilt(pose, s), 2)
        metrics[f"z_{s.name}"] = round(ca_xyz(pose, s.mid).z, 2)

    keep = True   # TODO: real thresholds (tilt, membrane span, clash, score)
    return keep, metrics


# --------------------------------------------------------------------------- #
# 7. Output: silent file (Binary) or one PDB per model
# --------------------------------------------------------------------------- #

class SilentWriter:
    """
    One silent file for the whole run, metrics attached as SCORE: columns.
    BinarySilentStruct so coordinates are stored verbatim.

        grep "^SCORE:" ensemble.out
        extract_pdbs -in:file:silent ensemble.out -in:file:tags model_000_0003
    """

    def __init__(self, path, sfxn):
        from pyrosetta.rosetta.core.io.silent import SilentFileData, BinarySilentStruct
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
        self.n = 0

    def add(self, pose, tag: str, energies: Dict[str, float]) -> None:
        ss = self._new(pose, tag)
        for k, v in energies.items():
            ss.add_energy(k, float(v))
        self.sfd.write_silent_struct(ss, self.path)
        self.n += 1


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", required=True, help="Input structure")
    p.add_argument("--config", required=True, help="Construct definition YAML")
    p.add_argument("--out", default="ensemble_noclosure", help="Output directory")

    p.add_argument("--n-traj", type=int, default=100,
                   help="Independent MC trajectories from the input structure")
    p.add_argument("--trials", type=int, default=1000, help="MC trials per trajectory")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=25.0,
                   help="Std dev (deg) of per-residue torsion perturbation")
    p.add_argument("--uniform", action="store_true",
                   help="Ignore --sigma; draw phi/psi uniformly in [-180,180] "
                        "each perturbed residue (original angle='random' style). "
                        "Broadest exploration, lowest acceptance.")
    p.add_argument("--n-perturb", type=int, default=2,
                   help="Loop residues perturbed per MC trial")
    p.add_argument("--boltzmann-n", type=int, default=None,
                   help="Snapshot every N accepted moves instead of keeping "
                        "only the lowest-energy pose per trajectory. Use this "
                        "for the success-rate measurement.")
    p.add_argument("--burnin", type=int, default=None,
                   help="Trials to discard before snapshotting. Only used with "
                        "--boltzmann-n; defaults to trials/5.")
    p.add_argument("--silent", action=argparse.BooleanOptionalAction, default=True,
                   help="Write all structures into one silent file (default).")
    p.add_argument("--silent-name", default="ensemble.out")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    pyrosetta.init(
        f"-ex1 -ex2aro -use_input_sc -mute all -constant_seed -jran {args.seed or 1}"
    )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    reference = pyrosetta.pose_from_pdb(args.pdb)
    cfg = to_pose_numbering(reference, load_construct(args.config))

    set_linear_fold_tree(reference)

    if any(s.has_dof for s in cfg.rigid):
        print("NOTE: dof blocks in the YAML are ignored in no-closure mode - "
              "rigid bodies are not moved independently; they follow the linker "
              "torsions. The TM lands wherever the sampled loop puts it.")

    print("Rigid    : " + ", ".join(f"{s.name} {s.start}-{s.stop}" for s in cfg.rigid))
    print("Flexible : " + ", ".join(f"{s.name} {s.start}-{s.stop}" for s in cfg.flexible))
    print(reference.fold_tree())

    sigma = None if args.uniform else args.sigma
    burnin = (args.burnin if args.burnin is not None
              else (args.trials // 5 if args.boltzmann_n else 0))
    if args.boltzmann_n:
        print(f"Sampling : Boltzmann snapshots every {args.boltzmann_n} accepted "
              f"moves, after {burnin} burn-in trials (T = {args.temperature}, "
              f"perturb = {'uniform' if args.uniform else f'gauss {sigma} deg'})")
        if burnin >= args.trials:
            raise SystemExit("--burnin is >= --trials; nothing would be sampled")
    else:
        print("Sampling : lowest-energy structure per trajectory (recover_low)")

    sfxn = make_score_function()
    packer = make_repack_mover(sfxn, cfg)

    silent_path = outdir / args.silent_name
    if args.silent:
        if silent_path.exists():
            raise SystemExit(
                f"{silent_path} already exists. Silent files are appended to, so "
                "a rerun would mix old and new structures - move or delete it first."
            )
        writer = SilentWriter(silent_path, sfxn)
        print(f"Output   : silent file {silent_path}")
    else:
        writer = None
        print(f"Output   : one PDB per model in {outdir}/")

    records: List[dict] = []
    n_written = 0

    for t_idx in range(args.n_traj):
        start = reference.clone()
        n_from_traj = 0

        for s_idx, pose in enumerate(mc_torsion_sampling(
                start, cfg, sfxn, packer, rng,
                trials=args.trials,
                temperature=args.temperature,
                sigma=sigma,
                n_perturb=args.n_perturb,
                boltzmann_n=args.boltzmann_n,
                burnin=burnin)):

            keep, m = passes_filters(pose, cfg, sfxn)
            tag = (f"model_{t_idx:03d}_{s_idx:04d}"
                   if args.boltzmann_n else f"model_{t_idx:03d}")

            rec = {"tag": tag, "traj": t_idx, "snapshot": s_idx, "kept": keep, **m}
            records.append(rec)

            if keep:
                if writer is not None:
                    energies = {k: float(v) for k, v in rec.items()
                                if k != "tag" and isinstance(v, (int, float, bool))}
                    writer.add(pose, tag, energies)
                else:
                    pose.dump_pdb(str(outdir / f"{tag}.pdb"))
                n_written += 1
                n_from_traj += 1

        print(f"[traj {t_idx}] {n_from_traj} structures "
              f"({t_idx + 1}/{args.n_traj} trajectories)")

    with open(outdir / "metrics.json", "w") as fh:
        json.dump({"config": args.config, "pdb": args.pdb, "seed": args.seed,
                   "mode": "boltzmann" if args.boltzmann_n else "recover_low",
                   "closure": "none",
                   "boltzmann_n": args.boltzmann_n, "burnin": burnin,
                   "perturb": "uniform" if args.uniform else f"gauss_{sigma}",
                   "temperature": args.temperature, "trials": args.trials,
                   "n_traj": args.n_traj, "models": records}, fh, indent=2)

    if writer is not None:
        print(f"\nDone. {n_written} structures written to {silent_path}")
        print(f"      scores : grep '^SCORE:' {silent_path}")
        print(f"      extract: extract_pdbs -in:file:silent {silent_path} "
              f"-in:file:tags <tag> [<tag> ...]")
    else:
        print(f"\nDone. {n_written} structures written to {outdir}/")


if __name__ == "__main__":
    main()