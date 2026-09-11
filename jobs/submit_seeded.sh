#!/bin/bash
# ---------------------------------------------------------------------------
# Submit several NAMED jobs, each an independent replicate run: same local
# culture range (0..N-1), but each its own seed -- so each job's "culture 0"
# is a genuinely different random draw, not the same one recomputed.
# culture_merge.py tracks (seed, culture) as the true identity, so merging
# all of them afterward correctly treats every job as its own set of
# statistical replicates (see culture_merge.py's OUTPUT SCHEMA note).
#
# USAGE
#   bash jobs/submit_seeded.sh <cultures_per_job> <cores_per_job> <pbs|slurm> \
#        name1:seed1 [name2:seed2 ...]
#
# Env: WALLTIME = walltime for every job, e.g. WALLTIME=48:00:00 (HH:MM:SS).
#      Overrides the #PBS -l walltime / #SBATCH --time line in parallel.pbs/.slurm,
#      so you don't have to edit those files per submission. Unset -> whatever the
#      job script already specifies.
#      QUEUE    = queue/partition override, e.g. QUEUE=long
#      OVERWRITE=1 = allow re-using a name whose results_<model>/parts_<name>/
#                 already holds data (default: refuse BEFORE submitting anything)
#      FLUSH_EVERY = neurons per flush+fsync block (default: 25, same as calling
#                 run_parallel.sh directly with nothing set -- this only needs to be
#                 set explicitly to CHANGE it)
#      NEURONS_PER_CULTURE = TIMING-ONLY override, small (e.g. 20) -- for a concurrency
#                 dry run through the real scheduler, NOT for an actual campaign (see
#                 run_parallel.sh and HPC_RUN_COMPLETE.md, "Walltime")
#
# MODEL: taken from config.cell_model at submission and passed to every job as
#      EXPECT_MODEL, so a job refuses to start if config.py changed while it queued.
#      Outputs go to results_<model>/ (see jobs/results_layout.sh); each job logs to
#      logs/<model>_<name>.log (one log per job -- they used to share one file).
#
# EXAMPLE -- your 4 jobs, different name and seed:
#   bash jobs/submit_seeded.sh 20 48 pbs run1:1000 run2:2000 run3:3000 run4:4000
#
#   with an explicit walltime and queue:
#   WALLTIME=48:00:00 QUEUE=long bash jobs/submit_seeded.sh 384 48 pbs \
#        run1:1000 run2:2000 run3:3000 run4:4000
#
# Each job gets its own PBS/SLURM job name (qsub -N / sbatch --job-name,
# which overrides the #PBS -N / #SBATCH --job-name line already in the
# script), its own results_<model>/parts_<name>/ directory and its own
# results_<model>/<name>/ outputs -- so none can collide or clobber another,
# whether they run sequentially or genuinely at the same time.
#
# Once ALL have finished (or hit the walltime):  bash jobs/merge_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail

CULT_PER_JOB="${1:?usage: submit_seeded.sh <cultures_per_job> <cores_per_job> <pbs|slurm> name:seed [...]}"
CORES="${2:?usage: submit_seeded.sh <cultures_per_job> <cores_per_job> <pbs|slurm> name:seed [...]}"
SCHED="${3:?usage: submit_seeded.sh <cultures_per_job> <cores_per_job> <pbs|slurm> name:seed [...]}"
shift 3
if [ "$#" -eq 0 ]; then
    echo "FATAL: no name:seed pairs given, e.g. run1:1000 run2:2000" >&2
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source jobs/results_layout.sh
MODEL="$(cfg_cell_model)"

echo "Submitting $# named replicate job(s), $CULT_PER_JOB cultures each, $CORES cores each ($SCHED)."
echo "cell model: $MODEL  -> results_${MODEL}/ , logs in logs/"
echo

# ---- guards for EVERY name before the FIRST submission ---------------------------
declare -A SEEN_NAMES SEEN_SEEDS
for pair in "$@"; do
    NAME="${pair%%:*}"
    SEED="${pair#*:}"
    if [ "$NAME" = "$pair" ] || [ -z "$SEED" ]; then
        echo "FATAL: '$pair' is not NAME:SEED" >&2; exit 1
    fi
    if [ -n "${SEEN_NAMES[$NAME]:-}" ]; then
        echo "FATAL: name '$NAME' given twice -- outputs would collide." >&2; exit 1
    fi
    if [ -n "${SEEN_SEEDS[$SEED]:-}" ]; then
        echo "FATAL: seed '$SEED' given twice (jobs '${SEEN_SEEDS[$SEED]}' and '$NAME') -- " \
             "they would simulate IDENTICAL cultures, not independent replicates." >&2
        exit 1
    fi
    SEEN_NAMES[$NAME]=1; SEEN_SEEDS[$SEED]=$NAME
    if parts_have_data "results_${MODEL}/parts_${NAME}" && [ "${OVERWRITE:-0}" != "1" ]; then
        echo "FATAL: results_${MODEL}/parts_${NAME}/ already holds data -- a job named" \
             "'$NAME' would delete it. Pick a new name, or set OVERWRITE=1." >&2
        exit 1
    fi
done
mkdir -p logs

# ---- submit ----------------------------------------------------------------------
for pair in "$@"; do
    NAME="${pair%%:*}"
    SEED="${pair#*:}"
    LOG="logs/${MODEL}_${NAME}.log"
    VARS="JOB_SEED=${SEED},JOB_NAME=${NAME},NCULT=${CULT_PER_JOB},NPROC=${CORES},EXPECT_MODEL=${MODEL},OVERWRITE=${OVERWRITE:-0},FLUSH_EVERY=${FLUSH_EVERY:-25}"
    [ -n "${NEURONS_PER_CULTURE:-}" ] && VARS="${VARS},NEURONS_PER_CULTURE=${NEURONS_PER_CULTURE}"

    if [ "$SCHED" = "pbs" ]; then
        EXTRA=()
        [ -n "${WALLTIME:-}" ] && EXTRA+=(-l "walltime=${WALLTIME}")
        [ -n "${QUEUE:-}" ]    && EXTRA+=(-q "${QUEUE}")
        JID=$(qsub -N "$NAME" ${EXTRA[@]+"${EXTRA[@]}"} -o "$LOG" \
                    -l "nodes=1:ppn=${CORES}" \
                    -v "$VARS" \
                    jobs/parallel.pbs)
    elif [ "$SCHED" = "slurm" ]; then
        EXTRA=()
        [ -n "${WALLTIME:-}" ] && EXTRA+=(--time="${WALLTIME}")
        [ -n "${QUEUE:-}" ]    && EXTRA+=(--partition="${QUEUE}")
        JID=$(sbatch --job-name="$NAME" ${EXTRA[@]+"${EXTRA[@]}"} --output="$LOG" \
                     --cpus-per-task="${CORES}" \
                     --export="ALL,${VARS}" \
                     --parsable jobs/parallel.slurm)
    else
        echo "FATAL: unknown scheduler '$SCHED' (use 'pbs' or 'slurm')" >&2
        exit 1
    fi
    echo "  $NAME (seed=$SEED) -> $JID   log: $LOG"
done

echo
echo "All jobs submitted. Check with: qstat -u \$USER   (or squeue -u \$USER)"
echo "Once ALL finish, combine with:  bash jobs/merge_all.sh"
