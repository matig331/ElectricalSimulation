"""
culture_maps.py -- quick-look P maps for ONE outcome, drawn from an already merged folder.

culture_merge.py can draw these maps itself, but only while merging and one outcome after the
other. This draws the same figure for a single outcome from the merged CSVs, so the three
outcomes can be rendered in parallel (jobs/analysis.pbs) from ONE frozen merge: every figure
then describes the same snapshot, even while simulation jobs are still appending parts.

Input : <input>/culture_P<outcome>.csv, as written by culture_merge.py (one outcome column).
Output: <out-dir>/culture_Pmap_<outcome>.pdf

Same inputs as the merge's own figure: same columns, same row order, 'culture' = the global id,
span taken from the data, i0 and n_pulses taken from the data (smoke_test_culture_maps.py checks
this array for array against culture_merge.merge(make_figures=True)).

    python culture_maps.py --input merged_dir --outcome activation --out-dir figures
"""
import argparse
import os

import numpy as np

from culture_export import OUTCOME_FILES, OUTCOME_COLS, PLOT_COLUMNS, render_outcome_set
from culture_statistics import (DEFAULT_DV_THRESHOLD_MV, DV_COL, POLARIZATION_OUTCOMES,
                                polarization_labels)


def merged_csv_path(input_path, outcome):
    """The merged CSV of `outcome`: `input_path` itself if it is a file, else the file
    culture_P<outcome>.csv inside the folder."""
    if os.path.isfile(input_path):
        return input_path
    stem, _col = OUTCOME_FILES[outcome]
    path = os.path.join(input_path, "culture_%s.csv" % stem)
    if not os.path.isfile(path):
        raise SystemExit("no merged %s file: %s (run culture_merge.py first)" % (outcome, path))
    return path


def load_plot_array(csv_path, outcome, dv_threshold_mV=0.0):
    """Array with columns PLOT_COLUMNS (the layout render_outcome_set expects): the chosen
    outcome's column filled, the other two all-nan (so only this outcome is drawn).
    dv_threshold_mV > 0 relabels depolarization / hyperpolarization by |DeltaV_end| (the same
    rule as culture_statistics.polarization_labels); 0 keeps the CSV's sign-only labels.
    Returns (arr, i0_uA, n_pulses)."""
    import pandas as pd
    _stem, col = OUTCOME_FILES[outcome]
    base = [c for c in PLOT_COLUMNS if c not in OUTCOME_COLS]
    # round_trip parsing = the same doubles Python's float() gives the merge for the same text
    theta = float(dv_threshold_mV or 0.0)
    extra = [DV_COL, "phase2_outcome"] if (theta > 0 and outcome in POLARIZATION_OUTCOMES) else []
    df = pd.read_csv(csv_path, usecols=base + [col, "i0_uA", "n_pulses"] + extra,
                     dtype={"i0_uA": str, "n_pulses": str}, float_precision="round_trip")
    if df.empty:
        raise SystemExit("%s has no rows" % csv_path)
    arr = np.full((len(df), len(PLOT_COLUMNS)), np.nan)
    for j, c in enumerate(PLOT_COLUMNS):
        if c in base:
            arr[:, j] = df[c].to_numpy(float)
    arr[:, PLOT_COLUMNS.index(col)] = df[col].to_numpy(float)
    if extra:
        arr[:, PLOT_COLUMNS.index(col)] = polarization_labels(
            outcome, df[DV_COL].to_numpy(float),
            df["phase2_outcome"].astype(str).eq("activation").to_numpy(), theta)
    # same choices as culture_merge.merge (first i0 seen; smallest n_pulses as text)
    i0 = float(df["i0_uA"].iloc[0])
    n_pulses = int(float(sorted(set(df["n_pulses"]))[0]))
    return arr, i0, n_pulses


def render_one(input_path, outcome, out_dir=None, bin_um=8.0,
               dv_threshold_mV=DEFAULT_DV_THRESHOLD_MV):
    """Draw culture_Pmap_<outcome>.pdf; returns its path."""
    from config import CFG
    import field as F
    csv_path = merged_csv_path(input_path, outcome)
    out_dir = out_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)
    arr, i0, n_pulses = load_plot_array(csv_path, outcome, dv_threshold_mV)
    elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
    # span from the data, not config: the figure matches what was actually simulated
    span = float(max(np.abs(arr[:, 1]).max(), np.abs(arr[:, 2]).max()))
    pdfs = render_outcome_set(arr, out_dir, "culture_", span, bin_um, i0, n_pulses, CFG,
                              elec, sign)
    print("[maps] %s: %d rows -> %s" % (outcome, len(arr), pdfs[outcome]))
    return pdfs[outcome]


def main():
    ap = argparse.ArgumentParser(description="Quick-look P maps for one outcome, from merged CSVs.")
    ap.add_argument("--input", required=True, help="merged folder (or one merged CSV)")
    ap.add_argument("--outcome", required=True, choices=sorted(OUTCOME_FILES))
    ap.add_argument("--out-dir", default=None, help="default: next to the merged CSV")
    ap.add_argument("--bin-um", type=float, default=8.0)
    ap.add_argument("--dv-threshold", type=float, default=DEFAULT_DV_THRESHOLD_MV,
                    help="mV, polarization outcomes only; 0 = sign-only labels (default %(default)s)")
    a = ap.parse_args()
    render_one(a.input, a.outcome, a.out_dir, a.bin_um, a.dv_threshold)


if __name__ == "__main__":
    main()
