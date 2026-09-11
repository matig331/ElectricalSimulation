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
#       EXPECT_MODEL   = set by the submitters: abort if config.cell_model differs
#                         (config.py edited while the job sat in the queue)
#       OVERWRITE=1    = allow re-running a tag whose parts already hold data
#                         (otherwise the job refuses instead of deleting them)
#       NEURONS_PER_CULTURE = override neurons/culture (default: config.n_neurons_effective(),
#                         the biological count, e.g. 1700). For CONCURRENCY TIMING ONLY: a small
#                         value (e.g. 20, matching jobs/dryrun.pbs) lets `$CONC` workers each run
#                         a few neurons under REAL multi-process contention in minutes instead of
#                         hours, so the concurrent s/sim can be checked against the solo number
#                         from dryrun.pbs BEFORE sizing a multi-day submission (see
#                         HPC_RUN_COMPLETE.md, "Walltime"). Never set this for a real campaign --
#                         it silently shrinks every culture, it does not just speed up a test.
#
# RESULTS LAYOUT (jobs/results_layout.sh): everything goes under
# results_<cell_model>/, so a soma_only job can never touch full_active data:
#   results_<model>/parts_<tag>/part_NNN.csv   raw worker output
#   results_<model>/<tag>/culture_P*.csv       this job's merge (CSV only)
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
# Either way the parts directory is job- AND model-scoped, so the cleanup below
# can never touch another job's (or the other model's) output, even when several
# jobs run at the same time.
#
# Re-running the SAME tag re-simulates the SAME cultures (deterministic). Because a
# finished run can hold 100+ h of compute, the job now REFUSES to delete existing
# parts unless OVERWRITE=1. If you ever hand culture_merge.py two directories that
# share a (seed, culture) pair, it refuses to merge rather than double-count.
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/results_layout.sh"

# ---- cell model: decides the results tree --------------------------------------
MODEL="$(cfg_cell_model)"
if [ -n "${EXPECT_MODEL:-}" ] && [ "$EXPECT_MODEL" != "$MODEL" ]; then
    echo "FATAL: this job was submitted for cell_model=$EXPECT_MODEL but config.py now says" \
         "$MODEL (edited while queued?). Refusing to run the wrong model." >&2
    exit 1
fi
RESULTS="results_${MODEL}"

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
PARTS="${PARTS_DIR:-${RESULTS}/parts_${TAG}}"
OUTDIR="${RESULTS}/${TAG}"

echo "cell model  : $MODEL   -> $RESULTS/"
echo "concurrency : $CONC core(s)"
echo "cultures    : $NCULT (local ids ${OFFSET}..$((OFFSET+NCULT-1)))"
echo "seed        : ${SEED:-<config.seed, unset>}"
echo "job tag     : $TAG"
echo "flush every : ${FLUSH_EVERY:-25} neurons"
[ -n "${NEURONS_PER_CULTURE:-}" ] && echo "neurons/cult: ${NEURONS_PER_CULTURE} (NEURONS_PER_CULTURE override -- NOT the biological count; concurrency-timing runs only)"
python -c "
from config import CFG
n = ${NEURONS_PER_CULTURE:-0} or CFG.n_neurons_effective()
print('neurons/cult:', n)
print('layers      :', len(CFG.layers_um))
print('total sims  :', $NCULT * n * len(CFG.layers_um))
"

# ---- never silently delete finished work ----------------------------------------
if parts_have_data "$PARTS"; then
    if [ "${OVERWRITE:-0}" != "1" ]; then
        echo "FATAL: $PARTS already holds simulated data. Re-running tag '$TAG' would" \
             "DELETE it. Use a new JOB_NAME, move the directory away, or set OVERWRITE=1." >&2
        exit 1
    fi
    echo "OVERWRITE=1: deleting existing $PARTS"
fi

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
export _WORKER_NEURONS="${NEURONS_PER_CULTURE:-}"
START=$SECONDS
if ! tr '\t' ' ' < "$PARTS/tasks.txt" | xargs -P "$CONC" -L1 bash -c \
        'python culture_worker.py --ids "$0" --out "$1" \
            ${_WORKER_SEED:+--seed "$_WORKER_SEED"} \
            ${_WORKER_NAME:+--job-name "$_WORKER_NAME"} \
            ${_WORKER_NEURONS:+--neurons "$_WORKER_NEURONS"} \
            --flush-every "$_WORKER_FLUSH"'; then
    echo "FATAL: at least one worker failed -- NOT merging." >&2
    exit 1
fi
ELAPSED=$((SECONDS - START))
echo "---------------------------------------------------------------"
echo "all workers finished in ${ELAPSED} s"

# ---- merge THIS job's own parts into its OWN three CSVs (no figures: rendering
#      must not eat the end of the walltime). Once every job has finished -- or hit
#      its walltime, in which case this step never runs -- use jobs/merge_all.sh,
#      which reads the raw parts of ALL jobs and also renders the figures.
python culture_merge.py --parts "$PARTS" --out-dir "$OUTDIR" --no-figures \
       --expect-model "$MODEL"
echo "wall time: ${ELAPSED} s on ${CONC} core(s)"
echo "output    : $OUTDIR/culture_P{activation,depolarization,hyperpolarization}.csv"
