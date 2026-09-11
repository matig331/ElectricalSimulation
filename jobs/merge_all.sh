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
#
# in : results_<model>/parts_*/part_*.csv
# out: merged_<model>/culture_P{activation,depolarization,hyperpolarization}.csv
#      merged_<model>/culture_Pmap_<outcome>.pdf
#   -> then: python culture_statistics.py --input merged_<model> --output stats_<model>
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
    soma_only|full_active) ;;
    *) echo "FATAL: MODEL='$MODEL' (expected soma_only or full_active)" >&2; exit 1 ;;
esac

N=0
for d in "results_${MODEL}"/parts_*/; do
    [ -d "$d" ] && N=$((N + 1))
done
if [ "$N" -eq 0 ]; then
    echo "FATAL: no results_${MODEL}/parts_*/ directories in $REPO -- nothing to merge." >&2
    exit 1
fi
echo "model $MODEL: $N job director$([ "$N" -eq 1 ] && echo y || echo ies) under results_${MODEL}/"

python culture_merge.py --parts "results_${MODEL}/parts_*" --out-dir "merged_${MODEL}" \
       --expect-model "$MODEL" "$@"
echo "next: python culture_statistics.py --input merged_${MODEL} --output stats_${MODEL}"
