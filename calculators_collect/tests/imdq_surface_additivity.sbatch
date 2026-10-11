#!/bin/bash
#SBATCH --job-name=imdq-surface
#SBATCH --partition=256G56c
#SBATCH --qos=qos1h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%1
#SBATCH --output=imdq/logs/imdq-surface-%A_%a.out
#SBATCH --error=imdq/logs/imdq-surface-%A_%a.err

set -euo pipefail
source scripts/env.sh
cd "$ROOT/imdq"
export PYTHONPATH="$PWD:$ROOT/scripts:$ROOT/inputs:$ROOT/vendor"
models=(Haldane FeS)
grids=(41 81)
"$PYTHON" scripts/validate_imdq_surface_additivity.py \
    --model "${models[$SLURM_ARRAY_TASK_ID]}" \
    --grid "${grids[$SLURM_ARRAY_TASK_ID]}"
