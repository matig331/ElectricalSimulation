#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM -- DRY RUN: 1 culture x 20 neurons x 3 layers = 60 simulations.
# Purpose: prove the pipeline runs end-to-end and MEASURE the per-simulation
# cost, so the full job's walltime can be sized from data instead of guessed.
# Submit:  sbatch jobs/dryrun.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=estim_dryrun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --output=dryrun.log

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
N_CULTURES=1
N_NEURONS=20
echo "DRY RUN: ${N_CULTURES} culture x ${N_NEURONS} neurons x 3 layers = $((N_CULTURES*N_NEURONS*3)) sims"

SECONDS=0
python -c "
import culture_export as ce
ce.culture_export(n_cultures=${N_CULTURES},
                  neurons_per_culture=${N_NEURONS},
                  csv_path='dryrun_Pactivation.csv')
"
ELAPSED=$SECONDS
NSIM=$((N_CULTURES*N_NEURONS*3))

echo "---------------------------------------------------------------"
echo "elapsed        : ${ELAPSED} s for ${NSIM} simulations"
python - <<PY
from config import CFG
elapsed, nsim = ${ELAPSED}, ${NSIM}
per = elapsed / nsim if nsim else 0
print("per simulation : %.3f s" % per)
total = CFG.n_cultures * CFG.n_neurons_effective() * len(CFG.layers_um)
h = per * total / 3600.0
print("your config.py : n_cultures=%d, use_hpc_count=%s, n_neurons_effective()=%d, layers=%d -> %d sims"
      % (CFG.n_cultures, CFG.use_hpc_count, CFG.n_neurons_effective(), len(CFG.layers_um), total))
print("full run at YOUR current config.py: %.1f h  -> request %.0f h walltime (2x margin)"
      % (h, max(1, round(2*h))))
PY
echo "---------------------------------------------------------------"
wc -l dryrun_Pactivation.csv dryrun_Pdepolarization.csv dryrun_Phyperpolarization.csv
head -2 dryrun_Pactivation.csv
echo "dry run finished: $(date)"
