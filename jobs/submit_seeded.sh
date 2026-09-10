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
# script), its own parts_<name>/ directory, and its own
# culture_Pactivation_<name>.csv -- so none of the 4 can collide or clobber
# another, whether they run sequentially or genuinely at the same time.
#
# Once all 4 have finished:  bash jobs/merge_all.sh
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

echo "Submitting $# named replicate job(s), $CULT_PER_JOB cultures each, $CORES cores each ($SCHED)."
echo

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

    if [ "$SCHED" = "pbs" ]; then
        EXTRA=()
        [ -n "${WALLTIME:-}" ] && EXTRA+=(-l "walltime=${WALLTIME}")
        [ -n "${QUEUE:-}" ]    && EXTRA+=(-q "${QUEUE}")
        JID=$(qsub -N "$NAME" "${EXTRA[@]}" \
                    -l "nodes=1:ppn=${CORES}" \
                    -v "JOB_SEED=${SEED},JOB_NAME=${NAME},NCULT=${CULT_PER_JOB},NPROC=${CORES}" \
                    jobs/parallel.pbs)
    elif [ "$SCHED" = "slurm" ]; then
        EXTRA=()
        [ -n "${WALLTIME:-}" ] && EXTRA+=(--time="${WALLTIME}")
        [ -n "${QUEUE:-}" ]    && EXTRA+=(--partition="${QUEUE}")
        JID=$(sbatch --job-name="$NAME" "${EXTRA[@]}" \
                     --cpus-per-task="${CORES}" \
                     --export="ALL,JOB_SEED=${SEED},JOB_NAME=${NAME},NCULT=${CULT_PER_JOB},NPROC=${CORES}" \
                     --parsable jobs/parallel.slurm)
    else
        echo "FATAL: unknown scheduler '$SCHED' (use 'pbs' or 'slurm')" >&2
        exit 1
    fi
    echo "  $NAME (seed=$SEED) -> $JID"
done

echo
echo "All jobs submitted. Check with: qstat -u \$USER   (or squeue -u \$USER)"
echo "Once ALL finish, combine with:  bash jobs/merge_all.sh"
