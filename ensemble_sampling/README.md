# Ensemble sampling

This folder contains a PyRosetta pipeline for sampling conformational ensembles of the linker that connects the FnIII-3 domain and the transmembrane (TM) helix of the insulin receptor. FnIII-3 is held fixed as the anchor. The TM helix is treated as a mobile rigid body that is placed at different positions, and for each placement the flexible linker is sampled by Monte Carlo (MC) torsion moves with loop closure. The pipeline asks which TM positions the linker can actually reach, and how the insertion of the ALFA-tag changes this.

All scripts share one construct definition file (YAML), so the same file describes the construct from orientation through sampling to visualisation.

## Contents

- [Folder structure](#folder-structure)
- [Typical workflow](#typical-workflow)
- [Conventions](#conventions)
- [`constrain_script.py`](#constrain_scriptpy)
- [`ensemble_sampling_grid.py`](#ensemble_sampling_gridpy)
- [`visualise_grid_ensemble.py`](#visualise_grid_ensemblepy)
- [`visualise_ensemble.py`](#visualise_ensemblepy)
- [`ensemble_yamls/`: construct definition files](#ensemble_yamls-construct-definition-files)
- [`test_and_diagnosis/`](#test_and_diagnosis)

## Folder structure

```
ensemble_sampling/
├── constrain_script.py           # Orient rigid bodies relative to the membrane normal
├── ensemble_sampling_grid.py     # Sampler (grid mode and random mode)
├── visualise_grid_ensemble.py    # Analysis and visualisation of grid-mode runs
├── visualise_ensemble.py         # Analysis and visualisation of random-mode runs
├── ensemble_yamls/               # Construct definition files
├── inputs/                       # Input structures (.pdb / .cif)
└── test_and_diagnosis/           # Smoke tests, example outputs, legacy prototypes
```

## Typical workflow

```bash
conda activate pyrosetta

# 1. Orient FnIII-3 and TM relative to the membrane normal (z axis)
python constrain_script.py \
    --pdb inputs/construct.cif \
    --config ensemble_yamls/construct.yaml \
    --out inputs/construct_oriented.pdb

# 2. Sample the ensemble
python ensemble_sampling_grid.py \
    --pdb inputs/construct_oriented.pdb \
    --config ensemble_yamls/construct.yaml \
    --out runs/construct_grid \
    --n-traj 10 --trials 500

# 3. Analyse and write PyMOL sessions + plots
python visualise_grid_ensemble.py --dir runs/construct_grid     # grid mode
# or
python visualise_ensemble.py \
    --silent runs/construct_random/ensemble.out \
    --metrics runs/construct_random/metrics.json \
    --config ensemble_yamls/construct.yaml \
    --out runs/construct_random/viz --png                        # random mode

# 4. Copy the viz/ directory to a local machine and open in PyMOL
tar czf viz.tar.gz -C runs/construct_grid viz
pymol viz/placements.pml
```

Sampling runs on the server. Only the lightweight `viz/` directory needs to be downloaded for inspection.

## Conventions

- **Membrane frame.** The membrane normal is the lab-frame z axis and the bilayer midplane is at z = 0. Input structures that come straight from a predictor (e.g. AlphaFold, Chai) are not oriented this way, which is what `constrain_script.py` is for.
- **Residue numbering.** All residue numbers in the YAML are PDB (author) numbering on the given chain. They are converted to Rosetta pose numbering internally.
- **Anchor.** The first rigid segment in the YAML is the root of the FoldTree and never moves. Everything else is expressed relative to it.
- **Units.** Distances in Å, angles in degrees, chainbreak in Rosetta energy units.

---

## `constrain_script.py`

Pre-processing step that puts rigid bodies in a canonical orientation relative to the membrane normal before sampling. For example, it can align the TM helix with z and lay FnIII-3 flat. It reads the same YAML as the sampler and acts only on rigid segments that carry an `orient:` block.

> The script's internal docstring still refers to it by its former name, `orient_bodies.py`.

### How to run it

```bash
python constrain_script.py \
    --pdb    <input structure> \
    --config <construct YAML> \
    --out    <output structure>
```

| Argument   | Required | Description |
|------------|----------|-------------|
| `--pdb`    | yes | Input structure. `.pdb` / `.ent` are read as PDB, `.cif` / `.mmcif` as mmCIF. |
| `--config` | yes | Construct YAML (the same file used by the sampler). |
| `--out`    | yes | Output structure. The format is inferred from the extension, so you can convert CIF → PDB in the same step. |

Dependencies: `numpy`, `pyyaml`, `biopython` (no PyRosetta needed).

The script exits with code 1 if no rigid segment has an `orient:` block.

### How it works

For each rigid segment with an `orient:` block:

1. **Long axis.** The CA coordinates of the segment are centred and decomposed by SVD (PCA). The first principal component is the long axis, oriented so that it points from N- to C-terminus. The ratio of the first two singular values (elongation ratio) is reported; below 1.5 the body is not elongated enough for its axis to be well defined, and a warning is printed.
2. **Target direction.** Depends on `axis` and `head`:

   | `axis` | `head` | Resulting orientation |
   |---|---|---|
   | `parallel` | not set | Long axis along z, keeping the sign of the input (N→C still points up if it pointed up before) |
   | `parallel` | `N` | N-terminus at the +z end (N→C points down, −z) |
   | `parallel` | `C` | C-terminus at the +z end (N→C points up, +z) |
   | `perpendicular` | not set | Long axis in the xy plane, along the input axis' xy projection (or +x if the axis was exactly along z) |
   | `perpendicular` | `N` / `C` | As above, with the named terminus at the + end of that direction |

3. **Rotation.** The shortest-arc rotation mapping the current axis onto the target is applied to every atom of the segment, about the segment's own CA centroid. The body therefore rotates in place and its centroid does not move.
4. **Translation (optional).** Each key given under `translate:` sets the **absolute** coordinate of the centroid on that axis. For example, `z: 0.0` moves the centroid onto the membrane midplane. Axes that are not listed are left untouched.
5. **Checks.** The script prints the angle between the axis and z before rotation, the centroid before/after, and the z (or in-plane) position of both termini afterwards. If `head` was requested and the terminus did not end up on the requested side, a warning is printed (usually a sign of a poorly defined long axis).

After all bodies have been moved, the script reports, for every flexible segment, the CA–CA gap between its flanking rigid residues and compares it with the maximum reach of a fully extended loop, `3.3 Å × (n_residues + 1)`. Gaps above 90 % of that reach are flagged `TIGHT/UNREACHABLE`.

Flexible segments are **not** moved, so linkers are broken in the output. This is intentional: the sampler rebuilds and closes them.

---

## `ensemble_sampling_grid.py`

The main sampler. It supports two ways of placing the mobile rigid body (the TM helix):

- **Grid mode:** a deterministic sweep. Each placement is a cell of a Cartesian grid, expressed as the absolute position of the TM attachment atom relative to the FnIII-3 anchor. Every cell is visited once.
- **Random mode:** placements are drawn uniformly within limits, as perturbations of the input pose.

For each placement, several independent MC trajectories sample the linker torsions and try to close the chain.

### How to run it

```bash
python ensemble_sampling_grid.py \
    --pdb <input structure> \
    --config <construct YAML> \
    [--out ensemble] \
    [--n-placements 20] \
    [--n-traj 10] \
    [--trials 500] \
    [--temperature 2.0] \
    [--sigma 25.0] \
    [--n-perturb 2] \
    [--boltzmann-n N] \
    [--burnin N] \
    [--chainbreak-weight 1.0] \
    [--span-slack 0.9] \
    [--max-attempts-factor 20] \
    [--silent | --no-silent] \
    [--silent-name ensemble.out] \
    [--seed 0]
```

The placement mode is not a command-line option; it comes from `dof.mode` in the YAML.

| Argument | Default | Mode | Description |
|---|---|---|---|
| `--pdb` | *required* | both | Input structure, ideally the output of `constrain_script.py`. |
| `--config` | *required* | both | Construct YAML. |
| `--out` | `ensemble` | both | Output directory (created if missing). |
| `--n-placements` | `20` | random | Number of **viable** placements to collect. Ignored in grid mode. |
| `--n-traj` | `10` | both | Independent MC trajectories per placement. |
| `--trials` | `500` | both | MC trials per trajectory. |
| `--temperature` | `2.0` | both | Metropolis temperature (kT in Rosetta energy units). |
| `--sigma` | `25.0` | both | Standard deviation (degrees) of the Gaussian perturbation applied to φ and ψ. |
| `--n-perturb` | `2` | both | Number of linker residues perturbed per MC trial. |
| `--boltzmann-n` | not set | both | If set, save a snapshot every N **accepted** moves instead of only the lowest-energy structure per trajectory (see [Output modes](#output-modes-recover_low-vs-boltzmann)). |
| `--burnin` | `trials/5` with `--boltzmann-n`, else `0` | both | MC trials discarded before snapshots are taken. Must be smaller than `--trials`. |
| `--chainbreak-weight` | `1.0` | both | Weight of the `chainbreak` and `linear_chainbreak` terms added to `ref2015`. |
| `--span-slack` | `0.9` | both | Fraction of the maximum loop extension allowed in the viability pre-check. |
| `--max-attempts-factor` | `20` | random | Stop after `n_placements × factor` draws, even if not enough viable placements were found. |
| `--silent` / `--no-silent` | `--silent` | both | Write all structures to one silent file (default), or one PDB per model. **The visualisation scripts require the silent file.** |
| `--silent-name` | `ensemble.out` | both | Name of the silent file inside `--out`. |
| `--seed` | `0` | both | Seed for the Python RNG and for Rosetta (`-jran`). Same seed + same inputs = same ensemble. |

> If the silent file already exists in `--out`, the script refuses to start. Silent files are appended to, so a rerun would mix structures from two runs. Move or delete the old file first.

### How it works

#### 1. Construct definition

The YAML is parsed into an ordered list of segments and validated (see [validation rules](#validation-rules)). YAML residue numbers are converted to pose numbering using the input structure's PDB info. Any `orient:` blocks are ignored here; they are only used by `constrain_script.py`.

#### 2. FoldTree

A star-shaped FoldTree is built:

- The root is the middle residue of the **first rigid segment** (FnIII-3).
- Every other rigid segment is connected to the root by its own jump, attached at its middle residue. The mobile bodies are therefore independent of each other.
- A cutpoint is placed at the middle residue of every flexible segment, and cutpoint variants are added so the chainbreak score term is active.

The cutpoints split the chain into one block per rigid segment. Each block contains the rigid segment plus the adjacent half of each neighbouring linker, so moving a jump moves the rigid body together with "its" half of the linker.

#### 3. Rigid-body placement

Each mobile rigid segment is transformed in this order: **spin → tilt → translate**. Rotations are applied about the centroid of the segment's block, not about the origin.

- **Spin** rotates about the segment's own axis, defined as the vector from its first CA to its last CA.
- **Tilt** rotates about an axis perpendicular to the segment axis. The direction of that perpendicular is set by `tilt_azimuth`.
- **Translation** depends on the mode:

| | Random mode | Grid mode |
|---|---|---|
| Placements | Uniform draws: `dx, dy, dz ∈ [−max, +max]`, `spin ∈ [−max, +max]`, `tilt ∈ [0, max]`, `tilt_azimuth ∈ [0, 360)` | Every cell of the Cartesian product of the grid axes; scalar values held fixed; `tilt_azimuth` fixed at 0 |
| Meaning of `x, y, z` | Lab-frame **displacement** from the input position | **Absolute** position of the mobile segment's N-terminal CA, relative to the C-terminal CA of the `anchor` segment |
| Translation applied | `(dx, dy, dz)` directly | After the rotations, the N-terminal CA is moved to `anchor_CA + (dx, dy, dz)` |
| Meaning of tilt / spin | Perturbation from the input orientation | Same (perturbation from the input orientation) |

In grid mode, `(0, 0, 0)` means the TM's first residue sits exactly on FnIII-3's last residue, and `z` is **signed** (`z: -15` = 15 Å below the anchor). The anchor position is measured once from the input structure, so the grid geometry is reproducible regardless of where the TM started. The grid is enumerated with the last-declared axis varying fastest.

#### 4. Viability pre-check

Before any MC is run, each placement is checked: for every flexible segment, the CA–CA distance between its flanking rigid residues must be at most

```
3.3 Å × (n_residues + 1) × span_slack
```

Placements that fail cannot be bridged by the linker.

- **Random mode:** the placement is discarded and a new one is drawn.
- **Grid mode:** the cell is recorded as non-viable in `metrics.json` (with the gaps in `gap_note`) and zero trajectories are run for it.

#### 5. Monte Carlo torsion sampling

For each viable placement, `--n-traj` independent trajectories are started from a copy of the placed pose. Each trajectory runs `--trials` iterations of:

1. **Perturb:** pick `--n-perturb` random linker residues and add Gaussian noise (σ = `--sigma`) to their φ and ψ.
2. **Close:** run CCD loop closure (`CCDLoopClosureMover`) on every flexible segment. Only linker backbone and side chains are allowed to move.
3. **Repack:** repack side chains of the linker residues and all residues within 6 Å of them (no design).
4. **Accept/reject:** Metropolis criterion at `--temperature` using `ref2015` + chainbreak terms.

The score function is `ref2015` with `chainbreak` and `linear_chainbreak` set to `--chainbreak-weight`. PyRosetta is initialised with `-ex1 -ex2aro -use_input_sc`.

##### Output modes: `recover_low` vs Boltzmann

| | Default (`recover_low`) | `--boltzmann-n N` |
|---|---|---|
| Structures per trajectory | 1: the lowest-energy pose visited | Variable: one every N accepted moves after the burn-in |
| What the ensemble represents | A set of local minima (good models) | Samples of the Metropolis distribution at the given temperature (shape of the conformational distribution) |
| Tag format | `model_PPP_TTT` | `model_PPP_TTT_SSSS` |
| Caveats | None | Consecutive snapshots are correlated (increase N); requires equilibration (burn-in) |

`PPP` = placement index, `TTT` = trajectory index, `SSSS` = snapshot index.

#### 6. Metrics and filtering

For every structure, the following metrics are computed:

| Metric | Description |
|---|---|
| `total_score` | Total `ref2015` + chainbreak score |
| `chainbreak` | `chainbreak` + `linear_chainbreak`: how well the loop closed (lower is better, ~0 = closed) |
| `gap_<flexible>` | CA–CA distance between the rigid residues flanking that linker |
| `tilt_<rigid>` | Angle (0–90°) between the segment axis (first CA → last CA) and the membrane normal |
| `z_<rigid>` | z coordinate of the CA of the segment's middle residue |

The filtering step (`passes_filters`) is currently a **placeholder**: all metrics are computed but every structure is kept (`kept = true`). Thresholds (e.g. chainbreak cutoff, TM tilt, membrane depth, clashes) are to be decided after inspecting the distributions.

### Outputs

```
<out>/
├── ensemble.out      # Rosetta binary silent file: coordinates + per-structure SCORE columns
└── metrics.json      # Run parameters, every placement and every trajectory
```

With `--no-silent`, `ensemble.out` is replaced by one `<tag>.pdb` per structure.

#### The silent file (`ensemble.out`)

All structures of the run are stored in a single Rosetta **binary silent file**. Binary silent structs store Cartesian coordinates rather than torsions, which preserves the non-ideal geometry at the cutpoints that the chainbreak metric measures. On the first structure, the script encodes and decodes the pose and checks that the chainbreak score is unchanged (printed as `round-trip chainbreak ... OK`).

Structures are appended as they are produced, so memory use stays flat and a job that is killed still leaves a readable partial file.

Every numeric value associated with a structure is stored as a `SCORE:` column next to its coordinates, so metrics and structures cannot get out of sync. The columns are:

| Column(s) | Description |
|---|---|
| `placement`, `traj`, `snapshot` | Indices identifying the structure |
| `kept` | Filter verdict (currently always 1) |
| `total_score`, `chainbreak`, `gap_<flexible>`, `tilt_<rigid>`, `z_<rigid>` | Metrics (see above) |
| `<seg>_dx`, `<seg>_dy`, `<seg>_dz`, `<seg>_tilt`, `<seg>_tilt_azimuth`, `<seg>_spin` | Placement parameters of each mobile segment |
| `<seg>_grid_<axis>_i` | Grid mode only: integer index of the cell along each grid axis |
| `description` | The structure tag (last column) |

Useful commands:

```bash
# Score table
grep "^SCORE:" ensemble.out

# Extract structures with Rosetta binaries (if available)
extract_pdbs -in:file:silent ensemble.out -in:file:tags model_012_003 model_012_004
```

Or with PyRosetta:

```python
import pyrosetta
from pyrosetta.rosetta.core.io.silent import SilentFileData, SilentFileOptions

pyrosetta.init("-mute all")
sfd = SilentFileData(SilentFileOptions())
sfd.read_file("ensemble.out")

pose = pyrosetta.Pose()
sfd.get_structure("model_012_003").fill_pose(pose)
pose.dump_pdb("model_012_003.pdb")
```

#### The metrics file (`metrics.json`)

`metrics.json` is the complete record of the run. Unlike the silent file, it also contains placements that were not viable and trajectories rejected by the filter. It is the file the visualisation scripts use to count reachability, so that failed attempts are included in the denominator.

```jsonc
{
  "config": "ensemble_yamls/construct.yaml",   // YAML used
  "pdb": "inputs/construct_oriented.pdb",      // input structure used
  "seed": 0,
  "mode": "recover_low",                       // or "boltzmann"
  "placement_mode": "grid",                    // or "random"
  "boltzmann_n": null,
  "burnin": 0,
  "temperature": 2.0,
  "trials": 500,
  "n_traj": 10,
  "placement_count": 25,

  "placements": [                              // one entry per placement / grid cell
    {
      "placement": 0,
      "viable": true,
      "n_traj_run": 10,                        // 0 for non-viable cells
      "tm_dx": -8.0, "tm_dy": -8.0, "tm_dz": 0.0,
      "tm_tilt": 0.0, "tm_tilt_azimuth": 0.0, "tm_spin": 0.0,
      "tm_grid_x_i": 0.0, "tm_grid_y_i": 0.0   // grid mode only
      // "gap_note": "linker gap 52.3A"        // only for non-viable cells
    }
  ],

  "models": [                                  // one entry per structure produced
    {
      "tag": "model_000_000",
      "placement": 0, "traj": 0, "snapshot": 0,
      "kept": true,
      "total_score": -130.81, "chainbreak": 0.058,
      "gap_linker": 23.9,
      "tilt_fn3": 89.6, "z_fn3": -17.2,
      "tilt_tm": 56.6, "z_tm": -0.07,
      "tm_dx": -8.0, "tm_dy": -8.0, "tm_dz": 0.0, "...": "..."
    }
  ],

  // grid mode only
  "grid": {
    "tm": {
      "anchor": "fn3",
      "anchor_position": {"x": 1.2, "y": -3.4, "z": 0.5},
      "axes": [
        {"axis": "x", "min": -8.0, "max": 8.0, "step": 4.0, "values": [-8.0, -4.0, 0.0, 4.0, 8.0]},
        {"axis": "y", "min": -8.0, "max": 8.0, "step": 4.0, "values": [-8.0, -4.0, 0.0, 4.0, 8.0]}
      ],
      "fixed": {"z": 0.0, "tilt": 0.0, "spin": 0.0}
    }
  },

  // random mode only
  "placement_attempts": 22                     // total draws needed to get n_placements viable ones
}
```

---

## `visualise_grid_ensemble.py`

Analysis and visualisation of **grid-mode** runs. It aggregates trajectories per grid cell, computes how often the linker closed in each cell, and writes PyMOL sessions, plots and a table.

### How to run it

```bash
python visualise_grid_ensemble.py \
    --dir <run directory> \
    [--silent ensemble.out] \
    [--config <construct YAML>] \
    [--out viz] \
    [--close-threshold 1.0] \
    [--reach-threshold 0.02] \
    [--ensemble-cap 200] \
    [--span-slack 0.9] \
    [--seed 0]
```

| Argument | Default | Description |
|---|---|---|
| `--dir` | *required* | Run directory containing `metrics.json` and the silent file. |
| `--silent` | `ensemble.out` | Silent file name inside `--dir`. |
| `--config` | from `metrics.json` | YAML construct file. By default, the path stored in `metrics.json` is tried, then the same file name inside `--dir` and its parent directory. |
| `--out` | `viz` | Output directory, created **inside** `--dir`. |
| `--close-threshold` | `1.0` | Per structure: a model is *closed* if `chainbreak < threshold`. |
| `--reach-threshold` | `0.02` | Per cell: a cell is *reachable* if its closure fraction ≥ threshold. Values of 0.01–0.05 have been more informative than 0.1. |
| `--ensemble-cap` | `200` | Maximum number of models in the trajectory PDBs (random subsample above that). `0` = no cap (slow in PyMOL). |
| `--span-slack` | `0.9` | Same meaning as in the sampler; only used to draw the reach limit on `gap_vs_closure.png`. |
| `--seed` | `0` | Seed for the random subsample. |

Both thresholds only affect analysis, so they can be changed and the script rerun without resampling.

**Requirements and limitations**

- The run must be in grid mode (`placement_mode: grid`), with **one** gridded segment gridded over exactly **two axes, `x` and `y`**. Grids over `z`, tilt or spin, or 3D grids, are not supported yet.
- Single-chain constructs only.
- The input structure path stored in `metrics.json` (`pdb`) must still exist, because the anchor domain is read from it.
- Closure fractions assume the default `recover_low` output mode (one structure per trajectory).

### How it works

1. **Load.** Reads `metrics.json`, the YAML and the `SCORE:` table of the silent file (plain text parsing, no decoding).
2. **Aggregate per cell.** For each grid cell:
   - `n_closed` = number of structures in the silent file with `chainbreak < --close-threshold`.
   - `closure_fraction = n_closed / n_traj_run`, where `n_traj_run` comes from `metrics.json`. Non-viable cells (`n_traj_run = 0`) get `NaN`.
   - `mean_chainbreak`, `best_chainbreak` and the tag of the best (lowest chainbreak) model are recorded.
   - `loop_gap = √(dx² + dy² + dz²)`: the distance between the anchor CA and the TM attachment CA before MC, i.e. the gap the linker has to span.
3. **Split residues.** From the YAML, *anchor* residues are all rigid segments without a `dof` block; *mobile* residues are all flexible segments plus rigid segments with a `dof` block.
4. **Decode structures.** Only the structures needed for the output are decoded from the silent file: a random subsample of all models, a subsample of the closed models, and the best model of every cell.
5. **Write PDBs.** The anchor is written once (`ref.pdb`); the mobile part of each model is written as a CA trace, one `MODEL` per structure, with a metric stored in the B-factor column. This keeps the files small enough for PyMOL.
6. **Write PyMOL scripts, plots and CSV.**

### Output structure

```
<dir>/viz/
├── ref.pdb                     # CA trace of the anchor (fixed) residues, from the input structure
├── ensemble_all.pdb            # CA traces of mobile residues, ≤ --ensemble-cap models; B-factor = chainbreak
├── ensemble_reachable.pdb      # Same, only models with chainbreak < --close-threshold
├── placements_all.pdb          # Best model of every viable cell; B-factor = closure fraction (0–1)
├── placements_reachable.pdb    # Same, only cells with closure fraction ≥ --reach-threshold
├── view.pml                    # Loads ref + ensemble_all
├── reachable.pml               # Loads ref + ensemble_reachable
├── placements.pml              # Loads ref + placements_all
├── reachable_placements.pml    # Loads ref + placements_reachable
├── reachability.png            # Closure fraction over the (x, y) grid
├── chainbreak.png              # Mean chainbreak over the grid (log colour scale)
├── gap_vs_closure.png          # Pre-MC loop gap vs closure fraction, with the linker reach limit
├── summary.png                 # 2×2 panel: the three plots above + chainbreak histogram
└── placements.csv              # One row per grid cell
```

**PyMOL colouring.** All sessions show the anchor as a grey cartoon and use a blue–white–red spectrum on the B-factor, but the scales differ:

| Session | Colour encodes | Scale |
|---|---|---|
| `view.pml`, `reachable.pml` | chainbreak | Rescaled per file (blue = lowest, red = highest **within that file**); not comparable across files |
| `placements.pml`, `reachable_placements.pml` | closure fraction | Fixed 0–1 (blue = 0, white = 0.5, red = 1); comparable across files and runs |

**Plots.** Grid plots show `x` and `y` in Å relative to the anchor CA, with a red dotted crosshair at (0, 0). The fixed `z` value is given in the title.

**`placements.csv` columns.**

| Column | Description |
|---|---|
| `placement` | Cell index |
| `viable` | Whether the cell passed the viability pre-check |
| `grid_x_i`, `grid_y_i` | Integer grid indices |
| `dx`, `dy`, `dz` | Cell position relative to the anchor (Å) |
| `loop_gap` | Distance the linker has to span (Å) |
| `n_traj_run` | Trajectories run in this cell |
| `n_kept` | Structures present in the silent file for this cell |
| `n_closed` | Structures with `chainbreak < --close-threshold` |
| `closure_fraction` | `n_closed / n_traj_run` (empty for non-viable cells) |
| `mean_chainbreak`, `best_chainbreak` | Chainbreak statistics |
| `best_tag` | Tag of the lowest-chainbreak model (extract it from the silent file for full-atom inspection) |

---

## `visualise_ensemble.py`

Analysis and visualisation of **random-mode** runs. It works on two levels: trajectories are summarised into placements, and each placement gets a reachability probability, `p(reach)` = fraction of its trajectories that closed.

### How to run it

```bash
python visualise_ensemble.py \
    --silent <run>/ensemble.out \
    [--metrics <run>/metrics.json] \
    [--config <construct YAML>] \
    [--out viz] \
    [--reach-metric chainbreak] \
    [--reach-max 0.1] \
    [--min-reach-prob 0.0] \
    [--dof-segment <segment name>] \
    [--filter "<python expression>"] \
    [--sort-by <column>] \
    [--top N] \
    [--stride 1] \
    [--max-models 500] \
    [--color-by total_score] \
    [--align] \
    [--slab 30.0] \
    [--slab-extent 60.0] \
    [--png] \
    [--list-columns]
```

| Argument | Default | Description |
|---|---|---|
| `--silent` | *required* | Silent file from the sampler. |
| `--metrics` | not set | `metrics.json` from the same run. **Strongly recommended:** reachability is then counted over every trajectory that was run, including ones rejected by the filter. Without it, only the silent file is used, which can inflate `p(reach)`. |
| `--config` | not set | Construct YAML. Without it, the whole chain is drawn for every model and no helix axes are drawn. |
| `--out` | `viz` | Output directory. |
| `--reach-metric` | `chainbreak` | Score column that decides whether a trajectory closed. |
| `--reach-max` | `0.1` | A trajectory is closed if `reach-metric ≤ reach-max`. Check the chainbreak histogram in `summary.png` before relying on this. |
| `--min-reach-prob` | `0.0` | A placement is reachable if `p(reach) >` this value (default: at least one trajectory closed). |
| `--dof-segment` | auto | Which mobile segment's `dx/dy/dz` define placement coordinates. Only needed if more than one segment is mobile. |
| `--filter` | not set | Python expression over score columns used to select models for `view.pml`, e.g. `"chainbreak < 5 and tilt_tm < 40"`. |
| `--sort-by` | not set | Sort selected models (ascending) by this column. |
| `--top` | not set | Keep the first N models after sorting. |
| `--stride` | `1` | Keep every N-th model. |
| `--max-models` | `500` | Hard cap on models per view; models are thinned evenly across the list rather than truncated. |
| `--color-by` | `total_score` | Column written to the B-factor of the per-trajectory views (`view`, `reachable`). |
| `--align` | off | Superimpose every model on the anchor. Normally unnecessary (the anchor is the FoldTree root); use it if the drift check warns. |
| `--slab` | `30.0` | Thickness (Å) of the membrane slab drawn around z = 0. `0` = no slab. |
| `--slab-extent` | `60.0` | Half-width (Å) of the slab planes in x and y. |
| `--png` | off | Also write `summary.png`. |
| `--list-columns` | off | Print the available score columns and exit (no PyRosetta needed). |

Model selection for `view.pml` is applied in the order: filter → sort → top → stride → max-models.

### How it works

1. **Read the score table** from the silent file (plain text).
2. **Count reachability** (from `metrics.json` if given):
   - Each trajectory is collapsed to its best snapshot (lowest `--reach-metric`). In `recover_low` mode there is only one.
   - A trajectory is closed if its metric ≤ `--reach-max`.
   - For each placement: `p(reach) = n_closed / n_traj`. The representative model of the placement is its best trajectory that is present in the silent file.
3. **Select models** for the four views: all selected models, closed models, one representative per placement, and one representative per reachable placement.
4. **Decode** every needed structure from the silent file once.
5. **Layout from the YAML:** the *anchor* is the first rigid segment, drawn once; the *mobile* part is everything from the first flexible segment to the end, drawn per model; rigid segments with a `dof` block are also drawn as axis lines (first CA → last CA).
6. **Drift check:** the maximum CA RMSD of the anchor across models is printed. Above 0.5 Å a warning suggests `--align`.
7. **Write** PDBs, PyMOL scripts, CSV and plots.

### Output structure

```
viz/
├── ref.pdb                              # One full-atom model (context)
├── view.pml                             # All selected models
├── ensemble.pdb                         #   CA traces of mobile residues, B-factor = --color-by
├── axes.pdb                             #   Axis lines of mobile rigid segments (with --config)
├── reachable.pml                        # Only closed trajectories
├── reachable_ensemble.pdb
├── reachable_axes.pdb
├── placements.pml                       # One representative per placement, B-factor = p(reach)
├── placements_ensemble.pdb
├── placements_axes.pdb
├── reachable_placements.pml             # Representatives of reachable placements only
├── reachable_placements_ensemble.pdb
├── reachable_placements_axes.pdb
├── placements.csv                       # Per-placement table
├── reachability.png                     # p(reach) in 3D (dx, dy, dz) + p(reach) vs loop gap
└── summary.png                          # With --png: tilt, chainbreak, score vs tilt, gap/z histograms
```

`reachable.pml` and `reachable_placements.pml` are skipped if nothing passes the thresholds.

**PyMOL sessions.** The anchor is a grey cartoon, the ensemble is drawn as ribbons with all states shown at once, the axis lines are sticks, and the membrane is shown as two transparent planes at z = ±slab/2. B-factors are stored rescaled to 0–100 and coloured blue–white–red:

| Session | Colour encodes | Scale |
|---|---|---|
| `view.pml`, `reachable.pml` | `--color-by` (default `total_score`) | Rescaled per file, not comparable across files |
| `placements.pml`, `reachable_placements.pml` | `p(reach)` | Fixed 0–1, comparable across files |

To step through states instead of showing all at once, the `.pml` files contain the commands as comments (`set all_states, 0` + `mplay`).

**`placements.csv` columns.** `placement`, `n_traj`, `n_reach(<metric><=<threshold>)`, `p_reach`, `best_<metric>`, `representative` (tag), the placement parameters (`<seg>_dx`, `_dy`, `_dz`, `_tilt`, `_tilt_azimuth`, `_spin`) and the loop gap(s) (`gap_<flexible>`).

**`reachability.png`.** Left: every placement at its `(dx, dy, dz)`, coloured by `p(reach)`. Right: `p(reach)` against the loop gap. If `p(reach)` is fully explained by the gap, the 3D position adds no information beyond distance. Note that tilt and spin are not shown, so two placements at the same position can differ in helix orientation.

---

## `ensemble_yamls/`: construct definition files

A construct is a top-level `segments:` list that alternates rigid and flexible segments, starting and ending with a rigid one:

```
[ rigid ]──flexible──[ rigid ]──flexible──[ rigid ] ...
```

### Segment keys

| Key | Required | Used by | Description |
|---|---|---|---|
| `name` | yes | all | Segment name. Used in metric names (`gap_<name>`, `tilt_<name>`, `<name>_dx`, ...) and as `anchor` reference. |
| `kind` | yes | all | `rigid` or `flexible`. |
| `start` | yes | all | First residue (PDB numbering, inclusive). |
| `stop` | yes | all | Last residue (PDB numbering, inclusive). |
| `chain` | no (`A`) | all | Chain ID. |
| `orient` | no | `constrain_script.py` | Orientation relative to the membrane normal (rigid segments only). Ignored by the sampler. |
| `dof` | no | sampler, visualisers | Degrees of freedom for placement (rigid segments only, not the first one). A rigid segment with a `dof` block is considered **mobile**. |

### `orient` block

```yaml
orient:
  axis: parallel        # required: parallel | perpendicular (to z)
  head: N               # optional: N | C (which terminus ends at the + end of the axis)
  translate:            # optional: absolute centroid coordinates; each key optional
    x: 0.0
    y: 0.0
    z: 0.0
```

See [`constrain_script.py`](#how-it-works) for the exact behaviour.

### `dof` block: random mode

```yaml
dof:
  mode: random          # optional, random is the default
  translate:            # max displacement per axis (Å), drawn in [-v, +v]; each key optional
    x: 8.0
    y: 8.0
    z: 4.0
  tilt: 15.0            # max tilt (deg), drawn in [0, v], random azimuth
  spin: 180.0           # max spin about own axis (deg), drawn in [-v, +v]
```

Omitted values are 0 (no movement along that degree of freedom).

### `dof` block: grid mode

```yaml
dof:
  mode: grid
  anchor: fn3                               # required: a non-mobile rigid segment
  translate:
    x: {min: -8.0, max: 8.0, step: 4.0}     # dict   -> grid axis
    y: {min: -8.0, max: 8.0, step: 4.0}     # dict   -> grid axis
    z: -10.0                                # scalar -> fixed value (signed; negative = below anchor)
  tilt: 0.0                                 # scalar or {min, max, step}
  spin: 0.0                                 # scalar or {min, max, step}
```

- Any of `x`, `y`, `z`, `tilt`, `spin` can be a scalar (fixed) or `{min, max, step}` (grid axis, inclusive of both ends).
- At least one axis must be a grid axis.
- `x`, `y`, `z` are **absolute** positions of the segment's N-terminal CA relative to the C-terminal CA of `anchor`. `tilt` and `spin` are perturbations from the input orientation.
- Number of cells = product of the number of values on each grid axis. Total trajectories = cells × `--n-traj`.
- To scan several heights, either run one job per `z` value (current practice, compatible with `visualise_grid_ensemble.py`) or grid `z` as well (supported by the sampler, not yet by the visualiser).

### Validation rules

The sampler stops with an error if any of these is violated:

- The construct begins and ends with a rigid segment, and kinds alternate.
- Segments are contiguous: each `start` equals the previous `stop + 1`.
- `stop ≥ start` for every segment.
- There is at least one flexible segment.
- Flexible segments have no `dof` block.
- All mobile segments use the same `dof.mode`.
- Grid mode: `anchor` is given, names an existing rigid segment, is not the segment itself, and is not mobile.
- `anchor` is only allowed in grid mode; `{min, max, step}` values are only allowed in grid mode.
- Grid axes: `step > 0`, `max ≥ min`, and exactly the keys `min`, `max`, `step`.
- Unknown keys under `dof` or `translate` are rejected.

Additional notes:

- Do not put a `dof` block on the **first** rigid segment. It is the FoldTree root and cannot be moved.
- Always leave a space after the colon: `z:15.0` is not read as `z: 15.0` and the value is silently lost.
- Large translations are easy to overdo. `z: ±15` is half a bilayer, and a modest tilt of a ~45 Å TM helix displaces its ends considerably. Reasonable random-mode starting values are `x: 8, y: 8, z: 4`, `tilt: 15`, `spin: 180`, depending on the gap in the input structure.

### Example 1: grid sweep of the TM below FnIII-3 (WT, single linker)

```yaml
segments:
  - name: fn3
    kind: rigid
    start: 91
    stop: 183
    chain: B
    orient:
      axis: parallel

  - name: linker
    kind: flexible
    start: 184
    stop: 197
    chain: B

  - name: tm
    kind: rigid
    start: 198
    stop: 218
    chain: B
    orient:
      axis: parallel
      head: N
      translate:
        z: 0.0
    dof:
      mode: grid
      anchor: fn3
      translate:
        x: {min: -8.0, max: 8.0, step: 4.0}
        y: {min: -8.0, max: 8.0, step: 4.0}
        z: 0.0
      tilt: 0.0
      spin: 0.0
```

### Example 2: random placements

```yaml
segments:
  - name: fn3
    kind: rigid
    start: 91
    stop: 183
    chain: B

  - name: linker
    kind: flexible
    start: 184
    stop: 197
    chain: B

  - name: tm
    kind: rigid
    start: 198
    stop: 218
    chain: B
    dof:
      translate: {x: 8.0, y: 8.0, z: 4.0}
      tilt: 15.0
      spin: 180.0
```

### Example 3: tagged construct with two linkers (illustrative numbering)

The inserted ALFA helix is its own rigid segment, flanked by two flexible linkers. Both mobile segments must use the same mode.

```yaml
segments:
  - name: fn3
    kind: rigid
    start: 91
    stop: 183
    chain: B

  - name: linker1
    kind: flexible
    start: 184
    stop: 189
    chain: B

  - name: alfa
    kind: rigid
    start: 190
    stop: 203
    chain: B
    dof:
      translate: {x: 4.0, y: 4.0, z: 4.0}
      tilt: 30.0
      spin: 180.0

  - name: linker2
    kind: flexible
    start: 204
    stop: 223
    chain: B

  - name: tm
    kind: rigid
    start: 224
    stop: 244
    chain: B
    dof:
      translate: {x: 8.0, y: 8.0, z: 4.0}
      tilt: 15.0
      spin: 180.0
```

---

## `test_and_diagnosis/`

| File | Description |
|---|---|
| `test_grid.py` | Smoke tests for grid mode (YAML parsing, grid enumeration, validation errors, placement geometry). |
| `grid_trial_1009.yaml` | Small 5 × 5 grid (x, y ∈ [−8, 8], step 4, z = 0) used for quick test runs. |
| `metrics.json` | Example output of an early random-mode run (20 placements × 10 trajectories). Predates grid support, so it lacks `placement_mode` and the per-placement records. |
| `symm_torsions.xml` | Original RosettaScripts prototype: symmetric setup, random φ/ψ of the linker + repack in a `GenericMonteCarlo` (T = 2.0, 1000 trials, `recover_low`), followed by `FastRelax` of the linker. |
| `loop.xml` | RosettaScripts prototype that rebuilds the linker with `LoopModeler` (only `fix_loop` is active). |
| `run.sh` | Runs `symm_torsions.xml` with MPI `rosetta_scripts` (36 processes, `nstruct 1000`) on `chain_break.pdb` with a symmetry definition. |
| `gen_chain_break.sh` | Builds `chain_break.pdb` from `relaxed.pdb` by inserting a `TER` record between residues 2638 and 2639. |
