#!/bin/bash
#SBATCH --job-name=imdq-fes121
#SBATCH --partition=256G56c
#SBATCH --qos=qos1h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=imdq/logs/imdq-fes121-%j.out
#SBATCH --error=imdq/logs/imdq-fes121-%j.err

set -euo pipefail
source scripts/env.sh
cd "$ROOT/imdq"
export PYTHONPATH="$PWD:$ROOT/scripts:$ROOT/inputs:$ROOT/vendor"
"$PYTHON" scripts/run_imdq.py --model FeS --grids 121
