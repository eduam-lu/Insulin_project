#!/usr/bin/env python3
"""
Benchmark the per-trial cost of the MC torsion sampling inner loop.

Mirrors the exact mover stack used in ensemble_sampling.py
(CCD closure + PackRotamers) and times only the trial loop,
not PyRosetta init or PDB loading.

Usage
-----
    python benchmark_trial.py --pdb tagged.pdb --config tagged.yaml

    # More trials for tighter estimates, or quicker smoke test:
    python benchmark_trial.py --pdb tagged.pdb --config tagged.yaml \\
        --warmup 30 --bench-trials 100

Output
------
    === Benchmark  (100 trials, 30 warmup) ===
    Full trial   :  843 ± 112 ms/trial
      CCD closure:  231 ±  18 ms
      PackRotamers:  598 ±  99 ms
      MC overhead :   14 ±   3 ms

    t_trial estimate : 0.843 s/trial
    → to reach 75,000 CPU·h (2 constructs): P × T × N ≈ 319,olean,olean
      e.g.  P=1000  T=150  N=2130
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import os
import time
from pathlib import Path

# 1. Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path of the parent directory (one level up)
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent directory to the beginning of sys.path
sys.path.insert(0, parent_dir)
# ── import helpers from ensemble_sampling (no side-effects on import) ─────── #
# ensemble_sampling.py must be on the Python path (or in the same directory).
try:
    from ensemble_sampling_grid import (
        load_construct,
        to_pose_numbering,
        build_fold_tree,
        build_loops,
        make_score_function,
        make_loop_movemap,
        make_closure_movers,
        make_repack_mover,
        draw_random_placement,
        enumerate_grid_placements,
        apply_placement,
        anchor_position,
        placement_is_viable,
        perturb_loop_torsions,
    )
except ImportError as e:
    sys.exit(
        f"Could not import from ensemble_sampling_grid.py: {e}\n"
        "Make sure this script is run from the same directory as "
        "ensemble_sampling_grid.py."
    )

import pyrosetta
from pyrosetta.rosetta.protocols.moves import MonteCarlo


# ── CLI ───────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark MC trial cost")
    p.add_argument("--pdb",    required=True, help="Input PDB file")
    p.add_argument("--config", required=True, help="Construct YAML file")
    p.add_argument("--warmup",       type=int, default=50,
                   help="Trials to run before timing starts (default: 50)")
    p.add_argument("--bench-trials", type=int, default=200,
                   help="Timed trials (default: 200)")
    p.add_argument("--temperature",  type=float, default=2.0)
    p.add_argument("--sigma",        type=float, default=25.0)
    p.add_argument("--n-perturb",    type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--target-hours", type=float, default=75000,
                   help="CPU-hour target to back-calculate P×T×N (default: 75000)")
    p.add_argument("--constructs",   type=int,   default=2,
                   help="Number of constructs (WT + tagged) for back-calculation")
    return p.parse_args()


# ── timing helper ─────────────────────────────────────────────────────────── #

def time_block(fn, n: int):
    """Run fn() n times; return list of wall-clock durations in seconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def report(label: str, times: list[float], width: int = 14):
    mean = statistics.mean(times) * 1000
    sd   = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
    print(f"  {label:<{width}}: {mean:6.1f} ± {sd:5.1f} ms")


# ── main ──────────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    # ── init ──────────────────────────────────────────────────────────────── #
    print("Initialising PyRosetta …", flush=True)
    pyrosetta.init(
        f"-ex1 -ex2aro -use_input_sc -mute all "
        f"-constant_seed -jran {args.seed}"
    )

    # ── load pose and build movers ─────────────────────────────────────────── #
    print(f"Loading {args.pdb} …", flush=True)
    reference = pyrosetta.pose_from_pdb(args.pdb)
    cfg       = to_pose_numbering(reference, load_construct(args.config))

    jumps    = build_fold_tree(reference, cfg)
    loops    = build_loops(cfg)
    sfxn     = make_score_function()
    movemap  = make_loop_movemap(cfg)
    closers  = make_closure_movers(loops, movemap)
    packer   = make_repack_mover(sfxn, cfg)

    # ── find a viable starting placement ──────────────────────────────────── #
    mobiles = [s for s in cfg.rigid if s.dof.is_mobile]
    placement_mode = mobiles[0].dof.mode if mobiles else "random"

    # Grid mode needs anchor CA positions measured once from the reference.
    anchor_positions = {}
    if placement_mode == "grid":
        for s in mobiles:
            if s.dof.anchor and s.dof.anchor not in anchor_positions:
                anchor_positions[s.dof.anchor] = anchor_position(
                    reference, cfg, s.dof.anchor)

    print(f"Finding a viable placement (mode: {placement_mode}) …", flush=True)
    placed = None

    if placement_mode == "grid":
        # Walk the grid until we hit a viable cell. For benchmarking we don't
        # care which cell, just that CCD has a real closure problem to solve.
        for i, placement in enumerate(enumerate_grid_placements(mobiles)):
            candidate = reference.clone()
            apply_placement(candidate, cfg, jumps, placement, anchor_positions)
            if placement_is_viable(candidate, cfg):
                placed = candidate
                print(f"  found viable grid cell #{i + 1}")
                break
    else:
        for attempt in range(5000):
            candidate = reference.clone()
            placement = draw_random_placement(mobiles, rng)
            apply_placement(candidate, cfg, jumps, placement, anchor_positions)
            if placement_is_viable(candidate, cfg):
                placed = candidate
                print(f"  found after {attempt + 1} attempt(s)")
                break

    if placed is None:
        sys.exit("Could not find a viable placement. Check your DOF limits "
                 "(z offset, tilt, spin) — they may be pushing the loops past "
                 "what they can span.")

    # ── warm-up  (not timed) ──────────────────────────────────────────────── #
    print(f"Warming up ({args.warmup} trials) …", flush=True)
    wp   = placed.clone()
    w_mc = MonteCarlo(wp, sfxn, args.temperature)
    for _ in range(args.warmup):
        perturb_loop_torsions(wp, cfg, rng, args.sigma, args.n_perturb)
        for closer in closers:
            closer.apply(wp)
        packer.apply(wp)
        w_mc.boltzmann(wp)

    # ── benchmark: full trial ─────────────────────────────────────────────── #
    print(f"Timing full MC trial ({args.bench_trials} trials) …", flush=True)
    bp     = placed.clone()
    b_mc   = MonteCarlo(bp, sfxn, args.temperature)

    def full_trial():
        perturb_loop_torsions(bp, cfg, rng, args.sigma, args.n_perturb)
        for closer in closers:
            closer.apply(bp)
        packer.apply(bp)
        b_mc.boltzmann(bp)

    t_full = time_block(full_trial, args.bench_trials)

    # ── benchmark: CCD only ───────────────────────────────────────────────── #
    print("Timing CCD closure only …", flush=True)
    cp = placed.clone()

    def ccd_only():
        perturb_loop_torsions(cp, cfg, rng, args.sigma, args.n_perturb)
        for closer in closers:
            closer.apply(cp)

    t_ccd = time_block(ccd_only, args.bench_trials)

    # ── benchmark: PackRotamers only ──────────────────────────────────────── #
    print("Timing PackRotamers only …", flush=True)
    pp = placed.clone()

    def pack_only():
        packer.apply(pp)

    t_pack = time_block(pack_only, args.bench_trials)

    # ── MC overhead (boltzmann + perturbation) ────────────────────────────── #
    # Approximate: full - CCD - pack
    t_overhead = [f - c - pk
                  for f, c, pk in zip(t_full, t_ccd, t_pack)]

    # ── report ────────────────────────────────────────────────────────────── #
    mean_full = statistics.mean(t_full)

    print()
    print("=" * 52)
    print(f"  Benchmark  ({args.bench_trials} trials, {args.warmup} warmup)")
    print("=" * 52)
    report("Full trial",    t_full)
    report("  CCD closure", t_ccd)
    report("  PackRotamers",t_pack)
    report("  MC overhead", t_overhead)
    print("-" * 52)
    print(f"  t_trial estimate : {mean_full:.3f} s/trial")
    print(f"                     {mean_full*1000:.1f} ms/trial")

    # back-calculate P×T×N required for target CPU hours
    target  = args.target_hours
    C       = args.constructs
    req_ptn = int(target * 3600 / (C * mean_full))
    print()
    print(f"  → to reach {target:,.0f} CPU·h ({C} construct{'s' if C>1 else ''}):")
    print(f"    C × P × T × N  ≈  {req_ptn:,} trial-calls total")
    print(f"    i.e. P × T × N ≈  {req_ptn // C:,} per construct")
    print()

    # a few example breakdowns
    print("  Example parameter combinations:")
    examples = [
        (200, 150), (500, 100), (500, 150),
        (1000, 100), (1000, 150), (1500, 100),
    ]
    per_construct = req_ptn // C
    header = f"  {'P':>6}  {'T':>5}  {'N':>6}   CPU·h"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for P, T in examples:
        N = round(per_construct / (P * T) / 100) * 100  # round to nearest 100
        if N < 100:
            continue
        h = C * P * T * N * mean_full / 3600
        print(f"  {P:>6}  {T:>5}  {N:>6}   {h:,.0f}")

    print("=" * 52)


if __name__ == "__main__":
    main()