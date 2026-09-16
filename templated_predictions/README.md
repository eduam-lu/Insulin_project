# Templated predictions

This folder generates structure predictions of the insulin receptor (IR) constructs templated on experimental cryo-EM structures. Two predictors are used, each with its own script and its own template mechanism:

- **AlphaFold 3** (`templated_AF.py`): templates are built manually — the script aligns each chain's sequence onto a resolved template chain itself and writes the alignment directly into the AF3 input JSON.
- **Chai-1** (`templated_chai.py`): templates are handed to Chai-1's own template pipeline via a manifest CSV and a small "fake BLAST hit" (m8) file; Chai does its own alignment internally.

Both scripts take one fasta per prediction job (a job can be a monomer, a protomer/heterodimer, or a tetramer/dimer-of-dimers) and produce one structure prediction per job.

## Contents

- [Folder structure](#folder-structure)
- [`templated_AF.py`](#templated_afpy)
- [`templated_chai.py`](#templated_chaipy)
- [`af_fastas/`](#af_fastas)
- [`chai_fastas/`](#chai_fastas)
- [`manifests/`](#manifests)
- [Additional contents](#additional-contents)

## Folder structure

```
templated_predictions/
├── templated_AF.py             # AF3 templated-prediction script
├── templated_chai.py           # Chai-1 templated-prediction script
├── run_af3_wrapper_0109.sh     # Bash wrapper: loops templated_AF.py over af_fastas/, sequentially
├── run_chai_fastas.sh          # SLURM array wrapper: runs templated_chai.py over chai_fastas/
├── af_fastas/                  # AF3-format fastas, one per prediction job
├── chai_fastas/                # Chai-format fastas, one per prediction job
├── manifests/                  # Per-job template manifest CSVs (used by templated_chai.py)
├── templates/                  # Per-structure, single-chain-extracted template cifs
│   ├── 7SL1_templates/         #   A_temp.cif, B_temp.cif, ... (one per chain of PDB 7SL1)
│   ├── 8DTL_templates/
│   ├── 8EYX_templates/
│   └── chai_cache/             # Runtime cache where templated_chai.py stages cifs for chai_lab (created on first run)
├── originals/                  # Unmodified, as-downloaded template PDB cifs (7SL1.cif, 8DTL.cif, 8EYX.cif)
└── envs/                       # af3-templating.yaml, chai.yaml (see main README)
```

`af_fastas/` and `chai_fastas/` describe the **same** prediction jobs (same PDB templates, same monomer/protomer/dimer composition) in the two fasta conventions each script expects. The template cif referenced by a given job is the same physical file either way — only how it's *pointed to* differs (inline in the AF3 fasta header vs. a manifest row for Chai).

---

## `templated_AF.py`

### Brief description

Generates AlphaFold 3 input JSONs with per-chain structural templates, and (optionally) launches AF3 on them.

For each chain that has a template, the script extracts the **resolved-only** sequence of the requested chain from the template cif (`sequence_extractor`), globally aligns it against the query sequence (`aligner`, using gemmi's BLOSUM62 aligner), and writes the resulting `queryIndices`/`templateIndices` plus the template mmCIF straight into the AF3 JSON's `templates` block. This is a manual reimplementation of AF3's template-alignment step — no other tool is involved.

Records sharing a `file_ID` are grouped into one AF3 job:

| Chains in the job | Wrapper called |
|---|---|
| 1 | `generate_templated_AF_json_monomer` |
| 2 | `generate_templated_AF_json_protomer` |
| 4 | `generate_templated_AF_json_dimer` |

All three delegate to the same builder (`_generate_templated_AF_json`), so the templating/alignment logic lives in one place. Other chain counts raise an error (add another wrapper if needed).

If a chain's `template_cif_path` was extracted as a single chain from a larger deposited structure, it commonly loses the header fields (`_pdbx_database_status.`, `_pdbx_audit_revision_history.`) that carry the release date — and AF3 refuses templates without one ("The structure must have a release date."). If `template_original_cif_path` is given for that chain, the script grafts those header categories from the original, unextracted cif onto the template before use (`graft_release_date_metadata`).

Finally, the script prints (and, with `--execute`, runs) the AF3 command for every generated JSON (`run_af`). One call to `run_alphafold.py` is made per JSON file, since `--json_path` takes a single file, not a directory.

### How to run it

```bash
python templated_AF.py \
    --fasta-path <fasta> \
    --model-dir <AF3 model params dir> \
    --db-dir <AF3 genetic/public databases dir> \
    [--json-output-dir af3_inputs] \
    [--af-output-dir af3_outputs] \
    [--json-schema-version 1] \
    [--conda-env-path /home/ingemar/anaconda3/envs/alphafold3] \
    [--af3-script-path /mnt/data/alphafold3/run_alphafold.py] \
    [--num-diffusion-samples 1] \
    [--execute]
```

| Argument | Default | Description |
|---|---|---|
| `--fasta-path` | *required* | Fasta using the header convention described in [`af_fastas/`](#af_fastas). |
| `--model-dir` | *required* | AF3 model parameters directory. |
| `--db-dir` | *required* | AF3 genetic/public databases directory. |
| `--json-output-dir` | `af3_inputs` | Where the generated AF3 input JSONs (and grafted template cifs, under `grafted_templates/`) are written. |
| `--af-output-dir` | `af3_outputs` | Passed through as AF3's own `--output_dir`. |
| `--json-schema-version` | `1` | AF3 input JSON `version` field: `1` embeds the template mmCIF text inline (`mmcif`); `2+` references it by path (`mmcifPath`) instead. Must match what your installed AF3 build accepts — `1` was confirmed against a real "unsupported version" error on the workstation this was developed on, so don't assume the upstream docs' current default applies to your install. |
| `--conda-env-path` | `/home/ingemar/anaconda3/envs/alphafold3` | Conda env (`-p`) AF3 itself runs in. **Update this for your machine.** |
| `--af3-script-path` | `/mnt/data/alphafold3/run_alphafold.py` | Path to AF3's `run_alphafold.py`. **Update this for your machine.** |
| `--num-diffusion-samples` | `1` | AF3's `--num_diffusion_samples`. |
| `--execute` | off | Without this flag, the AF3 command(s) are printed but not run — JSON generation is cheap, AF3 runs are not, so this is a deliberate speed bump against firing off an expensive run by accident (e.g. inside a loop). |

Dependencies: `gemmi`, `pandas` (see `envs/af3-templating.yaml`; no AF3 install needed just to generate JSONs, only to `--execute`).

---

## `templated_chai.py`

### Brief description

Generates the per-chain template inputs Chai-1 expects and runs a Chai-1 prediction.

Chai-1's own fasta parser only accepts plain `>protein|name=chainX` headers — adding extra fields to smuggle in template info raises a hard error — so template info lives in a separate manifest CSV instead (see [`manifests/`](#manifests)). For each chain with a manifest row, the script:

1. Stages the referenced template cif into a cache directory as `<IDENTIFIER>.cif.gz`, the exact name and location `chai_lab`'s own downloader checks before trying to fetch a structure from RCSB — so this is what makes it use the given structure instead.
2. Writes one row of a `custom.m8` file per templated chain (chain ID, `<IDENTIFIER>_<template_chain_id>`, and BLAST-like alignment coordinates). `subject_end` is deliberately set larger than the template's real length; Chai slices the template sequence as `seq[start:end]` and Python silently clips an out-of-range end to the actual length, so this always requests "the whole resolved template" without the script having to pre-parse the cif just to find its length.
3. Builds the rest of the feature context as usual (MSA via the ColabFold server or a local directory, ESM embeddings), then attaches a template context built from the `custom.m8` file and the staged cifs.
4. Runs folding.

Chai aligns query to template itself (via kalign), so unlike the AF3 script there is no separate sequence-extraction/alignment step here — the script only has to point Chai at the right cif and chain.

A chain absent from the manifest runs untemplated (no error).

### How to run it

```bash
python templated_chai.py \
    --fasta-path <fasta> \
    --template-manifest <manifest.csv> \
    [--output-dir outputs] \
    [--cif-cache-dir my_templates] \
    [--use-msa-server | --msa-directory <dir>] \
    [--num-trunk-recycles 3] \
    [--num-diffn-timesteps 200] \
    [--num-diffn-samples 5]
```

| Argument | Default | Description |
|---|---|---|
| `--fasta-path` | *required* | Chai-format fasta (see [`chai_fastas/`](#chai_fastas)). |
| `--template-manifest` | *required* | CSV: `chain_id,template_cif_path,template_chain_id` (see [`manifests/`](#manifests)). |
| `--output-dir` | `outputs` | Where predictions (and `custom.m8`) are written. |
| `--cif-cache-dir` | `my_templates` | Where template cifs are staged for `chai_lab` to find. Re-staged on every call — not skip-if-exists — so edits to the source cif are always picked up. |
| `--use-msa-server` / `--msa-directory` | `--use-msa-server` (mutually exclusive) | Use the ColabFold MSA server, or precomputed `.aligned.pqt` MSAs from a local directory. |
| `--num-trunk-recycles` | `3` | Chai-1 trunk recycles. |
| `--num-diffn-timesteps` | `200` | Diffusion timesteps. |
| `--num-diffn-samples` | `5` | Number of diffusion samples (structures) generated. |

Dependencies: `chai_lab`, `pandas`, a CUDA-capable GPU (see `envs/chai.yaml`).

---

## `af_fastas/`

### Contents

One fasta per AF3 prediction job. Files are named `af_<template>_<composition>.fa`, e.g. `af_7sl1_alpha.fa` (monomer, alpha subunit only), `af_7sl1_protomer.fa` (alpha + beta, heterodimer), `af_7sl1_dimer.fa` (the physiological (αβ)₂ tetramer, i.e. two protomers). The same naming is used across the three template structures currently in the project (`7sl1`, `8dtl`, `8eyx`), so there are 9 fastas in total (3 templates × 3 compositions).

### Fasta format

```
>file_ID|chain_id|template_cif_path|template_chain_id|template_original_cif_path
SEQUENCE...
```

| Field | Required | Description |
|---|---|---|
| `file_ID` | yes | Groups records into one AF3 job / one output JSON. All records with the same `file_ID` become one job's chains, in file order. |
| `chain_id` | yes | The AF3 entity `"id"`, e.g. `A`, `B`, `C`, `D`. |
| `template_cif_path` | no | Path to the template mmCIF for this chain. Leave empty to skip templating this chain. |
| `template_chain_id` | no | Chain ID **inside that cif** to use as the template (not necessarily the same letter as `chain_id`). |
| `template_original_cif_path` | no | Path to the original, unextracted PDB cif `template_cif_path` was pulled from. Only needed if `template_cif_path` is missing release-date header metadata (see `graft_release_date_metadata` above); leave blank if it already has it. |

Fields are separated by `|` and trail off if omitted (e.g. a fully untemplated chain can just be `>file_ID|chain_id`). One record per chain; consecutive records with the same `file_ID` become one job. Number of records sharing a `file_ID` must be 1, 2 or 4 (monomer/protomer/dimer).

Example (heterodimer, both chains templated on the same PDB entry):

```
>IR_7sl1_protomer|A|templates/7SL1_templates/A_temp.cif|A|originals/7SL1.cif
HLYPGEVCPG...
>IR_7sl1_protomer|B|templates/7SL1_templates/B_temp.cif|A|originals/7SL1.cif
SLEEVGNVTA...
```

---

## `chai_fastas/`

### Contents

One fasta per Chai-1 prediction job, mirroring the same jobs as `af_fastas/` (same template/composition combinations, same `chai_<template>_<composition>.fa` naming — e.g. `chai_7sl1_protomer.fa`), but in Chai's native header format and without any template information inline (templating is handled entirely by the paired manifest CSV, see below).

### Fasta format

```
>protein|name=chainA
SEQUENCE...
>protein|name=chainB
SEQUENCE...
```

This is exactly the header format `chai_lab`'s own fasta parser (`chai_lab.data.dataset.inference_dataset.read_inputs`) requires — any deviation (e.g. extra `|`-delimited fields) raises a hard `ValueError`. `name=` gives the chain ID used both by Chai internally and as the `chain_id` key expected in the paired manifest CSV. The number of `>protein` records in the file is the job's stoichiometry (1 = monomer, 2 = protomer, 4 = tetramer, ...), inferred automatically — no separate per-stoichiometry file naming is required by the script itself (though this project's files follow the same `_alpha` / `_protomer` / `_dimer` convention as `af_fastas/` for consistency).

---

## `manifests/`

### Contents

One CSV per Chai prediction job, named to match its fasta (e.g. `chai_7sl1_protomer.csv` pairs with `chai_fastas/chai_7sl1_protomer.fa`). Each row gives the template for one chain of that job; a chain with no row runs untemplated.

### Manifest format

```csv
chain_id,template_cif_path,template_chain_id
chainA,/path/to/templates/7SL1_templates/A_temp.cif,A
chainB,/path/to/templates/7SL1_templates/B_temp.cif,A
```

| Column | Description |
|---|---|
| `chain_id` | Must exactly match the `name=` value of a chain in the paired fasta. |
| `template_cif_path` | Path to the template mmCIF for this chain. One cif file per template chain (not the "one shared cif, multiple chains" pattern) — each row gets its own `custom.m8` line and its own staged, cached cif. |
| `template_chain_id` | Chain ID **inside that cif** to use as the template. |

Paths in this project's manifests are absolute (HPC cluster paths, e.g. `/home/x_eduam/insulin_project/templates/...`) — update them to match wherever the repo is checked out, or pass relative paths from the working directory `templated_chai.py` is run from.

---

## Additional contents

- **`templates/`** — one subfolder per template PDB entry (`7SL1_templates/`, `8DTL_templates/`, `8EYX_templates/`), each containing one single-chain-extracted cif per chain (`A_temp.cif`, `B_temp.cif`, ...). These are the files referenced by both `af_fastas/` headers and `manifests/` rows. `templates/chai_cache/` is a runtime cache directory `templated_chai.py` stages cifs into for `chai_lab` to find (created on first run, not checked in).
- **`originals/`** — the unmodified, as-downloaded PDB mmCIFs for each template entry (`7SL1.cif`, `8DTL.cif`, `8EYX.cif`), kept so `templated_AF.py` can graft release-date header metadata back onto single-chain extractions from `templates/` that are missing it.
- **`run_af3_wrapper_0109.sh`** / **`run_chai_fastas.sh`** — wrapper scripts that loop the two prediction scripts over their respective fasta folders. The AF3 wrapper runs jobs sequentially in a single process (via `conda run`); the Chai wrapper is a SLURM array job (one array task per fasta, with per-job manifest lookup and a `.done` marker to skip completed jobs on rerun). Neither is required to use the scripts directly — they document how the batch predictions in this project were actually launched, and are a starting point for launching new ones.
