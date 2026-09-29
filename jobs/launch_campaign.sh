#!/bin/bash
# ---------------------------------------------------------------------------
# Launch a culture campaign SIZED FROM config.py: TARGET_NEURONS neurons (each simulated once
# per layer), in jobs of CULT_PER_JOB cultures x config.n_neurons_effective() neurons, one
# culture per core. Checks everything it can BEFORE the first qsub, prints the plan, appends
# plan + submissions to results_<model>/campaign_<PREFIX>.log.
#
#   bash jobs/launch_campaign.sh              # plan, checks, submit
#   DRY=1 bash jobs/launch_campaign.sh        # plan and checks only -- nothing is submitted
#   FIRST=8 bash jobs/launch_campaign.sh      # resume: jobs 8..J only (qsub stopped part-way)
#
# Env (default):
#   PREFIX=ftd             job names PREFIX01, PREFIX02, ...; parts in results_<model>/parts_<name>
#   TARGET_NEURONS=2000000 neurons wanted; the job count is rounded UP to whole jobs
#   CULT_PER_JOB=48        cultures per job        CORES=48   cores per job (one culture each)
#   SEED0=50000            seed of job 1; job i gets SEED0 + (i-1) x SEED_STEP
#   SEED_STEP=1000         culture c of seed s draws from default_rng(s + c): the step must be
#                          >= CULT_PER_JOB, or two jobs simulate the same cultures under two
#                          identities (a merge cannot see it). Also checked against every
#                          earlier run of the same model on disk.
#   FLUSH_EVERY=10         neurons per written + fsync'ed block
#   WALLTIME=24:00:00      QUEUE=cpu
#   S_SIM=4.2              single-core s per simulation for the CURRENT protocol on the cluster
#   CONTENTION=2.25        worst slowdown of CORES workers sharing a node (full-active campaign)
#   FORCE=1                submit even if the predicted worst case exceeds WALLTIME
#   EXPECT_MODEL=full_tuned  refuse if config.cell_model differs
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source jobs/results_layout.sh

PREFIX="${PREFIX:-ftd}"
TARGET_NEURONS="${TARGET_NEURONS:-2000000}"
CULT_PER_JOB="${CULT_PER_JOB:-48}"
CORES="${CORES:-48}"
SEED0="${SEED0:-50000}"
SEED_STEP="${SEED_STEP:-1000}"
FLUSH_EVERY="${FLUSH_EVERY:-10}"
WALLTIME="${WALLTIME:-24:00:00}"
QUEUE="${QUEUE:-cpu}"
S_SIM="${S_SIM:-4.2}"
CONTENTION="${CONTENTION:-2.25}"
EXPECT_MODEL="${EXPECT_MODEL:-full_tuned}"
FIRST="${FIRST:-1}"

case "$PREFIX" in
    ""|*[!A-Za-z0-9_]*) echo "FATAL: PREFIX='$PREFIX' (letters, digits, _ only)" >&2; exit 1 ;;
esac
MODEL="$(cfg_cell_model)"
if [ "$MODEL" != "$EXPECT_MODEL" ]; then
    echo "FATAL: config.cell_model is '$MODEL', expected '$EXPECT_MODEL' -- nothing submitted" >&2
    exit 1
fi
RES="results_${MODEL}"
mkdir -p "$RES" logs
LOG="$RES/campaign_${PREFIX}.log"

# ---- the plan: sizes, seeds, streams, predicted time -- all checks before any qsub ---------
PLAN="$(python - "$MODEL" "$PREFIX" "$TARGET_NEURONS" "$CULT_PER_JOB" "$CORES" "$SEED0" \
        "$SEED_STEP" "$WALLTIME" "$S_SIM" "$CONTENTION" "$FIRST" "${FORCE:-0}" "$RES" <<'PY'
import csv, glob, math, os, sys
(model, prefix, target, cpj, cores, seed0, step, walltime, s_sim, cont, first, force,
 res) = sys.argv[1:]
target, cpj, cores, seed0, step, first = (int(v) for v in (target, cpj, cores, seed0, step, first))
s_sim, cont, force = float(s_sim), float(cont), force == "1"
from config import CFG
import field as F
from culture_export import placement_frame, pre_stim_split
elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
span, cen = placement_frame(CFG, elec, sign)
base, settle = pre_stim_split(CFG)
N = int(CFG.n_neurons_effective())
L = len(CFG.layers_um)
err = []
if CFG.use_hpc_count is not True:
    err.append("config.use_hpc_count is not True: the culture size would be the LOCAL test count")
if cpj < 1 or cores < 1:
    err.append("CULT_PER_JOB and CORES must be >= 1")
if cpj % cores:
    err.append("CULT_PER_JOB (%d) is not a multiple of CORES (%d): some cores would idle while "
               "others run a second culture" % (cpj, cores))
if step < cpj:
    err.append("SEED_STEP (%d) < CULT_PER_JOB (%d): jobs would share random streams" % (step, cpj))
J = int(math.ceil(target / float(cpj * N)))
if first < 1 or first > J:
    err.append("FIRST=%d outside 1..%d" % (first, J))
width = max(2, len(str(J)))
names = ["%s%0*d" % (prefix, width, i) for i in range(1, J + 1)]
seeds = [seed0 + (i - 1) * step for i in range(1, J + 1)]
new = {}
for nm, s in zip(names, seeds):
    for c in range(cpj):
        new[s + c] = nm
# streams already simulated for this model: seed (first data row) + culture ids (tasks.txt)
clash = []
for d in sorted(glob.glob(os.path.join(res, "parts_*"))):
    tag = os.path.basename(d)[len("parts_"):]
    if tag in names:
        continue                      # submit_seeded.sh itself refuses a name that holds data
    seed = None
    for f in sorted(glob.glob(os.path.join(d, "part_*.csv"))):
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd, None)
            row = next(rd, None)
        if head and row and "seed" in head and len(row) == len(head):
            seed = int(float(row[head.index("seed")]))
            break
    if seed is None:
        continue
    ids = set()
    tp = os.path.join(d, "tasks.txt")
    if os.path.exists(tp):
        for line in open(tp):
            spec = line.split("\t")[0].strip()
            ids.update(int(x) for x in spec.split(",") if x.strip())
    else:
        for f in glob.glob(os.path.join(d, "part_*.csv")):
            with open(f, newline="") as fh:
                rd = csv.reader(fh)
                head = next(rd, None)
                ic = head.index("culture") if head and "culture" in head else None
                for r in rd:
                    if ic is not None and len(r) > ic and r[ic].strip():
                        ids.add(int(float(r[ic])))
    hit = sorted(set(seed + c for c in ids) & set(new))
    if hit:
        clash.append("%s (seed %d) shares %d random stream(s) with %s"
                     % (d, seed, len(hit), new[hit[0]]))
if clash:
    err.append("new cultures would repeat earlier draws of this model: " + "; ".join(clash[:4])
               + " -- choose another SEED0")
h, m, sec = (int(x) for x in walltime.split(":"))
wall_h = h + m / 60.0 + sec / 3600.0
sims_core = (cpj // cores if cpj % cores == 0 else -(-cpj // cores)) * N * L
solo_h = sims_core * s_sim / 3600.0
worst_h = solo_h * cont * 1.2
if worst_h > wall_h and not force:
    err.append("predicted worst case %.1f h per job (%d sims per core x %.1f s x %.2f contention "
               "x 1.2) exceeds WALLTIME %s: lower config.neurons_per_culture or CULT_PER_JOB, "
               "raise WALLTIME, or FORCE=1" % (worst_h, sims_core, s_sim, cont, walltime))
print("model         : %s" % model)
print("culture       : %d neurons, somata in +/-%g um around (%g, %g) um, %d layers %s"
      % (N, span, cen[0], cen[1], L, tuple(CFG.layers_um)))
print("protocol      : one pulse per neuron -- %g ms rest (%g variable-step + %g fixed-dt), "
      "%g uA x %g ms per phase, %g ms after (window fraction %g)"
      % (base + settle, settle, base, CFG.i0_uA, CFG.phase_dur_ms, CFG.bump_ms,
         CFG.bump_culture_fraction))
print("campaign      : %d jobs x %d cultures x %d neurons = %d neurons, %d simulations"
      % (J, cpj, N, J * cpj * N, J * cpj * N * L))
print("jobs          : %s .. %s, seeds %d .. %d (step %d)%s"
      % (names[0], names[-1], seeds[0], seeds[-1], step,
         "" if first == 1 else "; submitting %s .. %s only" % (names[first - 1], names[-1])))
print("time per job  : %d sims per core -> %.1f h at %.1f s/sim, %.1f h worst case (x%.2f "
      "contention, +20%%); walltime %s" % (sims_core, solo_h, s_sim, worst_h, cont, walltime))
print("CPU-hours     : %.0f at %.1f s/sim" % (J * cpj * N * L * s_sim / 3600.0, s_sim))
for e in err:
    print("FATAL: " + e)
print("PAIRS " + " ".join("%s:%d" % (n, s) for n, s in list(zip(names, seeds))[first - 1:]))
PY
)"
PAIRS="$(printf '%s\n' "$PLAN" | sed -n 's/^PAIRS //p')"
{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S')  launch_campaign.sh  code $(git log -1 --format='%h %s' 2>/dev/null || echo '?')"
    printf '%s\n' "$PLAN" | grep -v '^PAIRS '
} | tee -a "$LOG"
if printf '%s\n' "$PLAN" | grep -q '^FATAL'; then
    echo "nothing submitted" | tee -a "$LOG" >&2
    exit 1
fi

# ---- not twice: none of these names may already be queued or running ------------------------
RE="$(printf '%s\n' $PAIRS | sed 's/:.*//' | paste -sd'|' -)"
if qstat -u "${USER:-$(id -un)}" 2>/dev/null | grep -qE " (${RE}) "; then
    echo "FATAL: some of these jobs are already queued or running -- not submitting twice" \
        | tee -a "$LOG" >&2
    exit 1
fi

if [ "${DRY:-0}" = "1" ]; then
    echo "DRY=1: would run WALLTIME=$WALLTIME QUEUE=$QUEUE FLUSH_EVERY=$FLUSH_EVERY" \
         "bash jobs/submit_seeded.sh $CULT_PER_JOB $CORES pbs <$(printf '%s\n' $PAIRS | wc -l) name:seed pairs>" \
        | tee -a "$LOG"
    exit 0
fi

# shellcheck disable=SC2086
WALLTIME="$WALLTIME" QUEUE="$QUEUE" FLUSH_EVERY="$FLUSH_EVERY" \
    bash jobs/submit_seeded.sh "$CULT_PER_JOB" "$CORES" pbs $PAIRS 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
