#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM -- FULL RUN: the activation-probability deliverable.
# Run jobs/dryrun.slurm FIRST and set the time below from the number it prints.
# Submit:  sbatch jobs/culture_export.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=culture_export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# <-- EDIT from the dry run's estimate
#SBATCH --time=24:00:00
# <-- EDIT: your partition name
#SBATCH --partition=batch
#SBATCH --output=culture_export.log

set -euo pipefail

# Locate the repo robustly. A scheduler copies the script to a spool directory, so
# the script's own path is unreliable; and sbatch may have been run from either the
# repo root OR from jobs/. Try each candidate and pick the one that actually
# contains the code. Quoted throughout: the real path contains a space.
REPO=""
for cand in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." "$PWD" "$PWD/.." "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"; do
    if [ -n "$cand" ] && [ -f "$cand/culture_export.py" ]; then
        REPO="$(cd "$cand" && pwd)"
        break
    fi
done
if [ -z "$REPO" ]; then
    echo "FATAL: could not locate the repo (culture_export.py not found)." >&2
    echo "       Tried SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-unset}' and the script directory." >&2
    exit 1
fi
cd "$REPO"

source jobs/env_setup.sh

# Parameters come from config.py (n_cultures, network_half_um, use_hpc_count,
# density_per_mm2, culture_distance_mode). Echo them so the log records exactly
# what this run used -- the log is the provenance record for the thesis.
python -c "
from config import CFG
n = CFG.n_neurons_effective()
print('n_cultures       :', CFG.n_cultures)
print('neurons/culture  :', n)
print('layers           :', CFG.layers_um)
print('distance_mode    :', CFG.culture_distance_mode)
print('i0_uA            :', CFG.i0_uA)
print('total sims       :', CFG.n_cultures * n * len(CFG.layers_um))
"

SECONDS=0
python culture_export.py
echo "---------------------------------------------------------------"
echo "elapsed: ${SECONDS} s"
wc -l culture_Pactivation.csv
echo "finished: $(date)"

# Expected on success: culture_Pactivation.csv (the deliverable the network
# consumes) and culture_Pmap.pdf, both in $REPO.
