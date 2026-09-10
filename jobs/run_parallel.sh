#!/bin/bash
# ---------------------------------------------------------------------------
# Fan out culture_worker.py across NPROC cores, then merge THIS job's own parts.
#
# Interactive:
#     bash jobs/run_parallel.sh 8 20                      # 8 cores, cultures 0-19
#     CULTURE_OFFSET=20 bash jobs/run_parallel.sh 8 20     # offset split: cultures 20-39
#     JOB_SEED=1000 JOB_NAME=run1 bash jobs/run_parallel.sh 8 20   # seed-differentiated replicate
#
# Args: $1 = concurrency (default $NPROC, else scheduler-given, else all cores)
#       $2 = number of cultures THIS JOB runs (default config.n_cultures)
# Env:  CULTURE_OFFSET = first GLOBAL culture index this job owns (default 0)
#       JOB_SEED       = RNG base override (default: config.seed) -- set this to
#                         get an INDEPENDENT REPLICATE job (see below)
#       JOB_NAME       = human label used for the parts dir / output CSV name and
#                         (if you set it in your qsub/sbatch call too) the job name
#       PARTS_DIR      = force the parts directory name (overrides the automatic
#                         naming below)
#
# TWO WAYS TO SPLIT WORK ACROSS JOBS -- pick ONE per set of jobs, don't mix:
#
#   (a) OFFSET split (jobs/submit_batches.sh): one shared seed, each job owns a
#       DISJOINT range of culture indices. Together they add up to one bigger
#       total. Use this to scale UP the statistics of a single run.
#
#   (b) SEED split (jobs/submit_seeded.sh): each job runs the SAME local culture
#       range (0..N-1) but with its OWN seed -- independent replicate runs, each
#       named. Use this when you explicitly want separate, individually
#       identifiable runs (e.g. 4 named replicates) rather than one pooled total.
#       culture_merge.py tracks the (seed, culture) pair as the true identity, so
#       merging replicates never confuses "culture 0 of run1" with "culture 0 of
#       run2" -- each gets its own row in the per-culture statistics.
#
# Either way the parts directory is job-scoped (named after JOB_NAME, or the
# offset/seed if no name is given), so 'rm -rf' below can never touch another
# job's in-progress output even if several jobs run at the same time.
#
# Re-submitting the SAME (offset, seed) combination re-simulates the SAME
# cultures (deterministic -- harmless to correctness) but overwrites that job's
# own parts/CSV. If you ever hand culture_merge.py two directories that DO share
# a (seed, culture) pair, it refuses to merge rather than silently double-
# counting it.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- concurrency ------------------------------------------------------------
if [ "${1:-}" != "" ]; then
    CONC="$1"
elif [ "${NPROC:-}" != "" ]; then
    CONC="$NPROC"
elif [ "${PBS_NP:-}" != "" ]; then
    CONC="$PBS_NP"
elif [ "${SLURM_CPUS_PER_TASK:-}" != "" ]; then
    CONC="$SLURM_CPUS_PER_TASK"
else
    CONC="$(nproc)"
fi

# ---- offset / seed / how many cultures THIS job runs ------------------------
OFFSET="${CULTURE_OFFSET:-0}"
SEED="${JOB_SEED:-}"          # empty -> culture_worker.py falls back to config.seed
NAME="${JOB_NAME:-}"          # empty -> naming falls back to offset/seed below
if [ "${2:-}" != "" ]; then
    NCULT="$2"
else
    NCULT="$(python -c 'from config import CFG; print(CFG.n_cultures)')"
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ---- job-scoped naming: NAME > SEED > OFFSET, always collision-safe --------
if [ -n "$NAME" ]; then
    TAG="$NAME"
elif [ -n "$SEED" ]; then
    TAG="off${OFFSET}_seed${SEED}"
else
    TAG="off${OFFSET}"
fi
PARTS="${PARTS_DIR:-parts_${TAG}}"

echo "concurrency : $CONC core(s)"
echo "cultures    : $NCULT (local ids ${OFFSET}..$((OFFSET+NCULT-1)))"
echo "seed        : ${SEED:-<config.seed, unset>}"
echo "job tag     : $TAG"
echo "flush every : ${FLUSH_EVERY:-25} neurons"
python -c "
from config import CFG
n = CFG.n_neurons_effective()
print('neurons/cult:', n)
print('layers      :', len(CFG.layers_um))
print('total sims  :', $NCULT * n * len(CFG.layers_um))
"

# ---- build the worker task list ---------------------------------------------
rm -rf "$PARTS"; mkdir -p "$PARTS"
python - "$NCULT" "$CONC" "$PARTS" "$OFFSET" > "$PARTS/tasks.txt" <<'PY'
import sys
from culture_worker import split_ids
n_cult, conc, parts, offset = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
for w, ids in enumerate(split_ids(n_cult, conc)):
    ids = [i + offset for i in ids]
    spec = ",".join(str(i) for i in ids)
    print("%s\t%s/part_%03d.csv" % (spec, parts, w))
PY
NTASK=$(wc -l < "$PARTS/tasks.txt")
echo "workers     : $NTASK (each builds its cells once, then runs its cultures)"
echo "---------------------------------------------------------------"

# ---- run them ----------------------------------------------------------------
export _WORKER_SEED="$SEED" _WORKER_NAME="$NAME" _WORKER_FLUSH="${FLUSH_EVERY:-25}"
START=$SECONDS
if ! tr '\t' ' ' < "$PARTS/tasks.txt" | xargs -P "$CONC" -L1 bash -c \
        'python culture_worker.py --ids "$0" --out "$1" \
            ${_WORKER_SEED:+--seed "$_WORKER_SEED"} \
            ${_WORKER_NAME:+--job-name "$_WORKER_NAME"} \
            --flush-every "$_WORKER_FLUSH"'; then
    echo "FATAL: at least one worker failed -- NOT merging." >&2
    exit 1
fi
ELAPSED=$((SECONDS - START))
echo "---------------------------------------------------------------"
echo "all workers finished in ${ELAPSED} s"

# ---- merge THIS job's own parts into its OWN csv -- never overwrites another
#      job's output. Run jobs/merge_all.sh once every job has finished to
#      combine all of them into the single final culture_Pactivation.csv.
OUT_CSV="culture_Pactivation_${TAG}.csv"
python culture_merge.py --parts "$PARTS" --out "$OUT_CSV"
echo "wall time: ${ELAPSED} s on ${CONC} core(s)"
echo "output    : $OUT_CSV"
