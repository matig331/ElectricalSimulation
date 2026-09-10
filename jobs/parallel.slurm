#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM -- PARALLEL culture export (the deliverable).
# One culture per core; each core builds its cells once, then runs its cultures.
#
# CONCURRENCY: set it in ONE place -- --cpus-per-task below. Everything follows.
#   Submit:  sbatch jobs/parallel.slurm
#   Override without editing:  sbatch --cpus-per-task=24 jobs/parallel.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=culture_par
#SBATCH --nodes=1
#SBATCH --ntasks=1
# <-- CONCURRENCY
#SBATCH --cpus-per-task=48
# <-- from the dry run estimate
#SBATCH --time=24:00:00
# <-- EDIT: your partition
#SBATCH --partition=batch
#SBATCH --output=culture_par.log

set -euo pipefail

REPO=""
for cand in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." "$PWD" "$PWD/.." "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"; do
    if [ -n "$cand" ] && [ -f "$cand/culture_export.py" ]; then
        REPO="$(cd "$cand" && pwd)"; break
    fi
done
if [ -z "$REPO" ]; then echo "FATAL: repo not found" >&2; exit 1; fi
cd "$REPO"

source jobs/env_setup.sh

# Concurrency: -v NPROC=... / --export overrides the resource request above.
bash jobs/run_parallel.sh "${NPROC:-}" "${NCULT:-}"

echo "finished: $(date)"
