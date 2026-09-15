# Insulin Project – André Lab

This repository contains the computational analyses performed for the Insulin Project in the André Lab (Lund University). The project studies how inserting a peptide tag (ALFA-tag) between the FnIII-3 domain and the transmembrane (TM) helix of the mouse insulin receptor (IR-A) affects the structure and conformational landscape of that region.

## Repository structure

```
.
├── templated_predictions/   # Templated Chai structure predictions of WT and tagged IR constructs
├── ensemble_sampling/       # PyRosetta sampling of FnIII-3–TM linker conformational ensembles
├── envs/                    # Conda environment files (.yaml)
└── README.md
```

- **`templated_predictions/`** contains the structure prediction workflow for the wild-type and ALFA-tagged mouse IR-A constructs. Predictions are run with Chai-1 using structural templates: each input FASTA is paired with a CSV template manifest and submitted as a SLURM job array on an HPC cluster. See the folder's README for inputs, outputs and how to launch a run.
- **`ensemble_sampling/`** contains a YAML-driven PyRosetta pipeline that samples conformational ensembles of the FnIII-3–TM linker. FnIII-3 is kept fixed as the anchor and the TM helix is treated as a mobile rigid body. Its placement is either sampled randomly or enumerated on a deterministic grid, and the flexible linker is sampled by Monte Carlo torsion moves with CCD loop closure. The folder also contains scripts to analyse the resulting ensembles and visualise them in PyMOL. See the folder's README for details.

## Installation

All analyses were run in conda environments defined by the `.yaml` files in `envs/`. You need [conda](https://docs.conda.io/) or [mamba](https://mamba.readthedocs.io/) installed.

Clone the repository:

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
```

Create the environments:

```bash
# Environment for templated structure predictions (Chai-1)
conda env create -f envs/chai.yaml

# Environment for ensemble sampling and analysis (PyRosetta)
conda env create -f envs/pyrosetta.yaml
```

Activate the environment you need before running the scripts in the corresponding folder:

```bash
conda activate chai        # for templated_predictions/
conda activate pyrosetta   # for ensemble_sampling/
```

> **Note on PyRosetta:** PyRosetta requires a license, which is free for academic and non-commercial use. It can be obtained from [RosettaCommons](https://www.pyrosetta.org/home/licensing-pyrosetta). If the environment creation fails at the PyRosetta step, check that your license credentials are set up for the PyRosetta conda channel.

> **Note on GPUs:** Chai-1 predictions require a CUDA-capable GPU. The prediction scripts were run on an HPC cluster with SLURM.

## Citation / contact

If you use this code, please contact the André Lab or cite the associated publication (in preparation).
