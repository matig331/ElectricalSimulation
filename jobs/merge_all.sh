#!/bin/bash
# ---------------------------------------------------------------------------
# Combine every job's parts directory (offset-split AND/OR seed-differentiated
# replicates) into the single final deliverable. Run this once all jobs from
# jobs/submit_batches.sh and/or jobs/submit_seeded.sh have finished.
#
#   bash jobs/merge_all.sh
#   bash jobs/merge_all.sh --no-figures     # skip the PDF, CSV only (faster)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

N=$(ls -d parts_* 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then
    echo "FATAL: no parts_*/ directories found in $REPO -- nothing to merge." >&2
    exit 1
fi
echo "found $N job director$([ "$N" -eq 1 ] && echo y || echo ies): $(ls -d parts_*)"

python culture_merge.py --parts "parts_*" --out culture_Pactivation.csv "$@"
