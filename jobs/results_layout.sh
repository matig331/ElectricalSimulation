#!/bin/bash
# ---------------------------------------------------------------------------
# Shared results layout + guards. SOURCED (never executed) by run_parallel.sh,
# merge_all.sh, submit_seeded.sh and submit_batches.sh, so the launcher, the job
# and the merger always agree on where a model's data lives.
#
#   results_<model>/parts_<tag>/part_NNN.csv   raw worker output (all outcomes per row)
#   results_<model>/<tag>/culture_P*.csv       per-job merge (written only if the job
#                                              finished inside its walltime)
#   merged_<model>/culture_P*.csv + PDFs       ALL jobs of that model (jobs/merge_all.sh)
#                                              -> the input of culture_statistics.py
#
# <model> = config.cell_model ("soma_only" | "full_active" | "full_tuned"). One tree per
# model, so a
# job of one model can never delete, overwrite or be merged with the other's data.
# The preliminary full-active campaign predates this layout: its parts_run*/ dirs in
# the repo root are left untouched (merge them with culture_merge.py directly).
# ---------------------------------------------------------------------------

cfg_cell_model() {
    # Print config.cell_model, validated. Fails loudly: the model decides where data goes.
    local m
    # 2>/dev/null: config.py reports its ESTIM_* overrides on stderr, and this captures
    # stdout only. Keep it that way -- a diagnostic leaking into $m breaks every job.
    if ! m="$(python -c 'from config import CFG; print(CFG.cell_model)' 2>/dev/null)"; then
        echo "FATAL: could not read config.cell_model -- is the NEURON env active and" \
             "config.py up to date (it must define cell_model)?" >&2
        return 1
    fi
    case "$m" in
        soma_only|full_active|full_tuned) echo "$m" ;;
        *) echo "FATAL: config.cell_model='$m' (expected soma_only, full_active or"\
                " full_tuned)" >&2
           return 1 ;;
    esac
}

parts_have_data() {
    # Exit 0 if directory $1 holds at least one part_*.csv with a data row after its
    # header. Reads at most 100 kB per file, so it is cheap even on 100 MB parts.
    local d="$1" f n
    [ -d "$d" ] || return 1
    for f in "$d"/part_*.csv; do
        [ -f "$f" ] || continue
        n="$(head -c 100000 "$f" | awk 'END{print NR}')"
        if [ "${n:-0}" -ge 2 ]; then
            return 0
        fi
    done
    return 1
}
