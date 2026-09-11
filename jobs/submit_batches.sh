#!/bin/bash
# ---------------------------------------------------------------------------
# Submit several jobs, each owning a DISJOINT block of culture indices, each with
# its OWN job name and OWN log file. Together they add up to more total cultures
# instead of recomputing the same ones.
#
# USAGE
#   bash jobs/submit_batches.sh <n_jobs> <cultures_per_job> [cores_per_job] [pbs|slurm] [tag]
#
# EXAMPLES
#   bash jobs/submit_batches.sh 4 60  48 pbs        # 4 jobs x 60 cultures = 240 total
#   bash jobs/submit_batches.sh 4 240 48 pbs        # 4 jobs x 240 cultures = 960 total
#   bash jobs/submit_batches.sh 4 60  48 pbs run2   # same, names/logs tagged 'run2'
#
# WHY NOT DIFFERENT SEEDS PER JOB
#   Culture c is ALWAYS seeded default_rng(cfg.seed + c) -- see culture_worker.py.
#   Job k gets CULTURE_OFFSET = k * cultures_per_job, so job 0 draws seeds
#   cfg.seed+0.., job 1 draws cfg.seed+cultures_per_job.., etc: the random streams
#   are ALREADY disjoint, and the culture LABELS are unique so culture_merge.py can
#   verify no double-counting. Hand-setting a different cfg.seed per job instead is
#   actively dangerous: seed=60/offset=0 gives the SAME stream as seed=0/offset=60,
#   a duplicate the merge cannot detect (the labels differ, the data does not).
#   So: vary the OFFSET (this script), never the seed.
#
# PER-JOB IDENTITY
#   name : <tag>_off<OFFSET>          (visible in qstat/squeue)
#   log  : <tag>_off<OFFSET>.log      (one per job -- no clobbering)
#   parts: results_<model>/parts_off<OFFSET>/   (one per job and model)
#   csvs : results_<model>/off<OFFSET>/culture_P*.csv
#   Combine every job's output at the end with: bash jobs/merge_all.sh
#   <model> = config.cell_model, passed to each job as EXPECT_MODEL. Existing parts
#   holding data are never overwritten unless OVERWRITE=1 (checked before submitting).
# ---------------------------------------------------------------------------
set -euo pipefail

N_JOBS="${1:?usage: submit_batches.sh <n_jobs> <cultures_per_job> [cores] [pbs|slurm] [tag]}"
CULT_PER_JOB="${2:?usage: submit_batches.sh <n_jobs> <cultures_per_job> [cores] [pbs|slurm] [tag]}"
CORES="${3:-48}"
SCHED="${4:-pbs}"
TAG="${5:-culture}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source jobs/results_layout.sh
MODEL="$(cfg_cell_model)"

PER_CORE=$(( (CULT_PER_JOB + CORES - 1) / CORES ))
echo "jobs              : $N_JOBS"
echo "cultures per job  : $CULT_PER_JOB"
echo "cores per job     : $CORES   (~$PER_CORE culture(s) per core)"
echo "TOTAL cultures    : $((N_JOBS * CULT_PER_JOB))"
echo "cell model        : $MODEL  -> results_${MODEL}/"
for ((k=0; k<N_JOBS; k++)); do
    if parts_have_data "results_${MODEL}/parts_off$((k * CULT_PER_JOB))" && [ "${OVERWRITE:-0}" != "1" ]; then
        echo "FATAL: results_${MODEL}/parts_off$((k * CULT_PER_JOB))/ already holds data;" \
             "set OVERWRITE=1 to replace it." >&2
        exit 1
    fi
done
if [ $((CULT_PER_JOB % CORES)) -ne 0 ]; then
    echo "note: $CULT_PER_JOB is not a multiple of $CORES -- some cores get one culture"
    echo "      more than others (still fully used, just slightly unbalanced)."
fi
python -c "
from config import CFG
n = CFG.n_neurons_effective()
print('neurons/culture   :', n)
print('sims per job      :', $CULT_PER_JOB * n * len(CFG.layers_um))
print('sims per core     :', $PER_CORE * n * len(CFG.layers_um))
print('sims TOTAL        :', $N_JOBS * $CULT_PER_JOB * n * len(CFG.layers_um))
"
echo

for ((k=0; k<N_JOBS; k++)); do
    OFFSET=$((k * CULT_PER_JOB))
    NAME="${TAG}_off${OFFSET}"
    LOG="${NAME}.log"
    if [ "$SCHED" = "pbs" ]; then
        JID=$(qsub -N "$NAME" -o "$LOG" \
                   -l "nodes=1:ppn=${CORES}" \
                   -v "CULTURE_OFFSET=${OFFSET},NCULT=${CULT_PER_JOB},NPROC=${CORES},EXPECT_MODEL=${MODEL},OVERWRITE=${OVERWRITE:-0}" \
                   jobs/parallel.pbs)
    elif [ "$SCHED" = "slurm" ]; then
        JID=$(sbatch --job-name="$NAME" --output="$LOG" \
                     --cpus-per-task="${CORES}" \
                     --export="ALL,CULTURE_OFFSET=${OFFSET},NCULT=${CULT_PER_JOB},NPROC=${CORES},EXPECT_MODEL=${MODEL},OVERWRITE=${OVERWRITE:-0}" \
                     --parsable jobs/parallel.slurm)
    else
        echo "FATAL: unknown scheduler '$SCHED' (use 'pbs' or 'slurm')" >&2
        exit 1
    fi
    echo "  job $k : name=$NAME  cultures ${OFFSET}..$((OFFSET+CULT_PER_JOB-1))  log=$LOG  -> $JID"
done

echo
echo "Watch:    qstat -u \$USER      (or squeue -u \$USER)"
echo "Combine:  bash jobs/merge_all.sh      # once ALL jobs have finished"
