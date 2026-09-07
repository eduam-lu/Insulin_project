#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run templated_AF.py over all fastas in a directory,
# one-by-one, under the af3-templating conda environment.

FASTAS_DIR="/home/eduardo/insulin_project/af_fastas_0109"
RUNS_ROOT="/home/eduardo/insulin_project/templated_runs_0109"
TEMPLATED_SCRIPT="/home/eduardo/insulin_project/templated_AF.py"
MODEL_DIR="/mnt/data/alphafold3/models"
DB_DIR="/mnt/data/alphafold3/alphafold_databases"
ENV_NAME="af3-templating"

mkdir -p "${RUNS_ROOT}"
shopt -s nullglob

# Accept several fasta extensions
FASTAS=("${FASTAS_DIR}"/*.fa "${FASTAS_DIR}"/*.fasta "${FASTAS_DIR}"/*.fa* )

if [ ${#FASTAS[@]} -eq 0 ]; then
  echo "No FASTA files found in ${FASTAS_DIR}" >&2
  exit 1
fi

for fasta in "${FASTAS[@]}"; do
  if [ ! -f "$fasta" ]; then
    continue
  fi
  base=$(basename "$fasta")
  name="${base%.*}"
  outdir="${RUNS_ROOT}/${name}"
  mkdir -p "$outdir"
  log="$outdir/run.log"

  echo "=== Starting $base -> $outdir ==="
  echo "Logging to $log"

  # Build the command to run inside the conda env
  CMD=(python3 "$TEMPLATED_SCRIPT" \
    --fasta-path "$fasta" \
    --af-output-dir "$outdir" \
    --json-output-dir "$outdir" \
    --model-dir "$MODEL_DIR" \
    --db-dir "$DB_DIR" \
    --execute)

  # Use `conda run` to execute inside the named environment; wrap in nohup
  # so the process survives a hangup. Start it in background, capture pid,
  # then wait so runs are strictly sequential.
  nohup conda run -n "$ENV_NAME" --no-capture-output "${CMD[@]}" >"$log" 2>&1 &
  pid=$!
  echo "Started PID $pid; waiting for completion..."
  wait "$pid"
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "Run for $base failed (exit $exit_code). See $log" >&2
    # stop here so user can inspect logs; comment out `exit` to continue on failure
    exit $exit_code
  fi
  echo "Completed $base"
  echo
done

echo "All runs completed. Outputs under ${RUNS_ROOT}"