#!/usr/bin/env bash
#SBATCH --job-name=chai-templated
##SBATCH -A berzelius-20XX-NNN     # <-- uncomment and set your project account
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-19%4
#SBATCH --output=/home/x_eduam/insulin_project/templated_runs_1408/slurm-%A_%a.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/x_eduam/insulin_project}"
FASTA_DIR="${FASTA_DIR:-$PROJECT_ROOT/chai_fastas}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/templated_runs_1408}"
MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_ROOT/manifests}"
CIF_CACHE_DIR="${CIF_CACHE_DIR:-$PROJECT_ROOT/templates/chai_cache}"
PY_SCRIPT="${PY_SCRIPT:-$PROJECT_ROOT/templated_chai.py}"
CONDA_ENV="${CONDA_ENV:-chai}"
FORCE="${FORCE:-0}"

export CHAI_DOWNLOADS_DIR="${CHAI_DOWNLOADS_DIR:-$PROJECT_ROOT/chai_downloads}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$FASTA_DIR" ]] || die "FASTA dir not found: $FASTA_DIR"
[[ -f "$PY_SCRIPT" ]] || die "Chai script not found: $PY_SCRIPT"
mkdir -p "$OUTPUT_ROOT" "$CIF_CACHE_DIR" "$CHAI_DOWNLOADS_DIR"

shopt -s nullglob
raw_fastas=("$FASTA_DIR"/*.fa "$FASTA_DIR"/*.fasta)
shopt -u nullglob
((${#raw_fastas[@]})) || die "no FASTA files in $FASTA_DIR"
mapfile -t fasta_files < <(printf '%s\n' "${raw_fastas[@]}" | sort)
unset raw_fastas

if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]] \
   && (( ${#fasta_files[@]} > SLURM_ARRAY_TASK_COUNT )); then
  echo "WARNING: ${#fasta_files[@]} FASTAs but only $SLURM_ARRAY_TASK_COUNT array tasks;" \
       "$(( ${#fasta_files[@]} - SLURM_ARRAY_TASK_COUNT )) will NOT be processed" >&2
fi

pick_manifest() {
  local stem="$1" dataset="${1#chai_}" c
  for c in "$MANIFEST_DIR/$dataset.csv" "$MANIFEST_DIR/$stem.csv" \
           "$MANIFEST_DIR/chai_$dataset.csv"; do
    [[ -f "$c" ]] && { printf '%s\n' "$c"; return 0; }
  done
  return 1
}

setup_env() {
  set +u                                  # NSC wrapper + conda.sh both read unset vars
  module load Miniforge3/24.7.1-2-hpc1-bdist
  local base; base="$(conda info --base)"
  source "$base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  set -u
  python3 -c 'import torch; assert torch.cuda.is_available(), "no CUDA device"'
}

run_one() {
  local fasta_path="$1"
  local name="${fasta_path##*/}"
  local stem="${name%.*}"
  local out="$OUTPUT_ROOT/$stem" log="$OUTPUT_ROOT/$stem.log" manifest rc=0

  if [[ -f "$out/.done" && "$FORCE" != "1" ]]; then
    echo "SKIP (already done): $stem"; return 0
  fi

  manifest="$(pick_manifest "$stem")" \
    || { echo "FAILED: $stem — no manifest in $MANIFEST_DIR" >&2; return 1; }

  mkdir -p "$out"
  echo "Starting: $stem"
  echo "  manifest: $manifest"
  echo "  output:   $out"
  echo "  log:      $log"

  python3 "$PY_SCRIPT" \
    --fasta-path "$fasta_path" \
    --template-manifest "$manifest" \
    --output-dir "$out" \
    --cif-cache-dir "$CIF_CACHE_DIR" \
    >"$log" 2>&1 || rc=$?

  if ((rc)); then
    echo "FAILED: $stem (exit $rc) — see $log" >&2
    return "$rc"
  fi

  touch "$out/.done"
  echo "Finished: $stem"
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="$SLURM_ARRAY_TASK_ID"
  if (( idx >= ${#fasta_files[@]} )); then
    echo "index $idx past end (${#fasta_files[@]} FASTAs), nothing to do"
    exit 0
  fi
  setup_env
  run_one "${fasta_files[$idx]}"
else
  setup_env
  failed=()
  for f in "${fasta_files[@]}"; do
    run_one "$f" || failed+=("${f##*/}")
  done
  if ((${#failed[@]})); then
    printf 'FAILED (%d): %s\n' "${#failed[@]}" "${failed[*]}" >&2
    exit 1
  fi
  echo "All Chai runs completed."
fi