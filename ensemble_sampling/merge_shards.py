#!/usr/bin/env python3
"""Merge sharded grid runs into one silent file + one metrics.json.

Each shard ran a disjoint window of the same global cell enumeration and
wrote its own <out>/ensemble.out and <out>/metrics.json. This script:

  1. Verifies the shards tile the grid exactly - every global cell index
     covered once, none twice, none missing. This is the check that the
     campaign is actually complete; run it before you analyse anything.
  2. Concatenates the silent files, keeping a single header.
  3. Merges the metrics.json files in global placement order.

    python merge_shards.py --shards grid/shard* --out grid/merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", nargs="+", required=True,
                   help="Shard output directories (shell glob is fine)")
    p.add_argument("--out", required=True, help="Merged output directory")
    p.add_argument("--silent-name", default="ensemble.out",
                   help="Silent file name inside each shard dir")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="Merge even if cells are missing (prints what is "
                        "missing; use when some shards are still queued)")
    return p.parse_args()


def main():
    args = parse_args()
    shard_dirs = sorted(Path(d) for d in args.shards)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- Load and check coverage ---------------------------------------
    metrics = []
    for d in shard_dirs:
        mpath = d / "metrics.json"
        if not mpath.exists():
            raise SystemExit(f"{mpath} missing - did that shard finish?")
        with open(mpath) as fh:
            metrics.append((d, json.load(fh)))

    if not metrics:
        raise SystemExit("No shards given.")

    shards = [m.get("shard") for _, m in metrics]
    if any(s is None for s in shards):
        raise SystemExit("A metrics.json has no 'shard' block - that run was "
                         "made by an unsharded/older version of the sampler.")

    totals = {s["n_cells_total"] for s in shards}
    if len(totals) != 1:
        raise SystemExit(f"Shards disagree on grid size: {sorted(totals)}. "
                         "They were not run against the same YAML.")
    n_total = totals.pop()

    seen: dict[int, Path] = {}
    dupes: list[tuple[int, Path, Path]] = []
    for (d, _), s in zip(metrics, shards):
        for c in range(s["cell_start"], s["cell_stop"]):
            if c in seen:
                dupes.append((c, seen[c], d))
            else:
                seen[c] = d
    missing = sorted(set(range(n_total)) - set(seen))

    print(f"Grid     : {n_total} cells total")
    print(f"Shards   : {len(shard_dirs)}")
    print(f"Covered  : {len(seen)} cells")
    if dupes:
        head = "; ".join(f"cell {c} in {a.name} and {b.name}"
                         for c, a, b in dupes[:5])
        raise SystemExit(f"OVERLAP: {len(dupes)} cell(s) run by more than one "
                         f"shard ({head}). Refusing to merge.")
    if missing:
        head = ", ".join(str(c) for c in missing[:20])
        msg = (f"MISSING: {len(missing)} cell(s) never ran: {head}"
               + (" ..." if len(missing) > 20 else ""))
        if not args.allow_incomplete:
            raise SystemExit(msg + "\nRe-run those shards, or pass "
                                   "--allow-incomplete to merge anyway.")
        print(msg)
    else:
        print("Coverage : complete - every cell run exactly once")

    # Surface per-cell failures, which coverage alone would not reveal.
    failed = sorted(c for _, m in metrics for c in m.get("failed_placements", []))
    if failed:
        print(f"WARNING  : {len(failed)} cell(s) failed during sampling: "
              f"{failed[:20]}{' ...' if len(failed) > 20 else ''}")

    # ----- Merge silent files --------------------------------------------
    dest = outdir / args.silent_name
    if dest.exists():
        raise SystemExit(f"{dest} already exists - move or delete it first.")

    n_structs = 0
    header_written = False
    header_cols = None
    inconsistent = False
    with open(dest, "w") as out:
        for d in shard_dirs:
            spath = d / args.silent_name
            if not spath.exists():
                print(f"note: {spath} missing (shard kept no structures?)")
                continue
            with open(spath) as fh:
                for line in fh:
                    if line.startswith("SEQUENCE:"):
                        if header_written:
                            continue
                        out.write(line)
                        continue
                    if line.startswith("SCORE:") and "description" in line:
                        cols = line.split()
                        if header_cols is None:
                            header_cols = cols
                            out.write(line)
                            header_written = True
                        else:
                            if cols != header_cols:
                                inconsistent = True
                            continue
                        continue
                    if line.startswith("SCORE:"):
                        n_structs += 1
                    out.write(line)

    if inconsistent:
        print("WARNING  : shards have different SCORE columns; the merged "
              "SCORE table may be misaligned.")

    # ----- Merge metrics --------------------------------------------------
    base = dict(metrics[0][1])
    placements, models = [], []
    for _, m in metrics:
        placements.extend(m.get("placements", []))
        models.extend(m.get("models", []))
    placements.sort(key=lambda r: r["placement"])
    models.sort(key=lambda r: (r["placement"], r["traj"], r["snapshot"]))

    base.pop("shard", None)
    base["placements"] = placements
    base["models"] = models
    base["placement_count"] = len(placements)
    base["failed_placements"] = failed
    base["merged_from"] = [str(d) for d in shard_dirs]
    base["n_cells_total"] = n_total
    base["missing_placements"] = missing

    with open(outdir / "metrics.json", "w") as fh:
        json.dump(base, fh, indent=2)

    print(f"\nDone. {n_structs} structures -> {dest}")
    print(f"      {len(placements)} placement records -> {outdir}/metrics.json")


if __name__ == "__main__":
    main()