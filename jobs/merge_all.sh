#!/bin/bash
# ---------------------------------------------------------------------------
# Combine the RAW parts of every job of ONE cell model into the final three
# deliverables + figures. Run once all jobs from jobs/submit_seeded.sh and/or
# jobs/submit_batches.sh have finished OR hit their walltime (a killed job never
# runs its own per-job merge; this reads its raw parts directly).
#
#   bash jobs/merge_all.sh                  # model = config.cell_model
#   MODEL=full_active bash jobs/merge_all.sh
#   bash jobs/merge_all.sh --no-figures     # CSVs only (faster)
#   PARTS='results_full_tuned/parts_ftc*' bash jobs/merge_all.sh --no-figures
#
# Env: MODEL = cell model (default config.cell_model)
#      PARTS = glob of the parts directories to merge (QUOTE it). Default
#              results_<model>/parts_* -- except for full_tuned, where it is REQUIRED:
#              that tree also holds the test runs (dry runs, ft1000_*), whose cultures are
#              not biological size and must not be pooled with a campaign.
#      OUT   = output folder (default merged_<model>)
#
# in : $PARTS/part_*.csv
# out: $OUT/culture_P{activation,depolarization,hyperpolarization}.csv
#      $OUT/culture_Pmap_<outcome>.pdf
#   -> then: python culture_statistics.py --input $OUT --output stats_<model>
# merged_<model>/ is a SIBLING of results_<model>/ on purpose: pointing
# culture_statistics.py at results_<model>/ can then never count a job twice.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source jobs/results_layout.sh

if [ -z "${MODEL:-}" ]; then
    MODEL="$(cfg_cell_model)"
fi
case "$MODEL" in
    soma_only|full_active|full_tuned) ;;
    *) echo "FATAL: MODEL='$MODEL' (expected soma_only, full_active or full_tuned)" >&2; exit 1 ;;
esac

if [ -z "${PARTS:-}" ] && [ "$MODEL" = "full_tuned" ]; then
    echo "FATAL: for full_tuned give the campaign's parts explicitly, e.g." >&2
    echo "         PARTS='results_full_tuned/parts_ftc*' bash jobs/merge_all.sh" >&2
    echo "       results_full_tuned/ also holds test runs that must not be pooled with it:" >&2
    for d in results_full_tuned/parts_*/; do
        [ -d "$d" ] && echo "         $d" >&2
    done
    exit 1
fi
PARTS="${PARTS:-results_${MODEL}/parts_*}"
OUT="${OUT:-merged_${MODEL}}"

N=0
for d in $PARTS; do
    [ -d "$d" ] && N=$((N + 1))
done
if [ "$N" -eq 0 ]; then
    echo "FATAL: no directory matches PARTS='$PARTS' in $REPO -- nothing to merge." >&2
    exit 1
fi
echo "model $MODEL: $N job director$([ "$N" -eq 1 ] && echo y || echo ies) matching '$PARTS' -> $OUT"

python culture_merge.py --parts "$PARTS" --out-dir "$OUT" --expect-model "$MODEL" "$@"
echo "next: python culture_statistics.py --input $OUT --output stats_${MODEL}"
