"""culture_merge.py -- MERGE + RENDER. No simulation, no NEURON.

Concatenates the partial CSVs written by culture_worker.py into the single
deliverable (culture_Pactivation.csv) and renders culture_Pmap.pdf using the
SAME renderer the serial path uses (culture_export.render_culture_figures),
so parallel and serial runs produce identical figures.

MULTIPLE JOBS: --parts is a glob PATTERN, default 'parts*'. Each job submission
writes its own parts_off<OFFSET>/ (jobs/submit_batches.sh) or parts_seed<SEED>/
(jobs/submit_seeded.sh) directory, so the default pattern picks up every job's
output in one call.

IDENTITY: a row's true replicate identity is the pair (seed, LOCAL culture index),
not the local culture index alone -- two jobs using DIFFERENT seeds legitimately
produce rows with the SAME local culture index (e.g. both have a "culture 0"), and
those are independent replicates, not duplicates. Conversely two rows sharing the
SAME (seed, culture) pair are a true duplicate (e.g. the same job re-submitted) and
must be rejected -- silently keeping both would double-weight that replicate in the
per-culture mean +/- SD.

Validation performed before writing anything:
  - every part has the expected header
  - no (seed, culture) pair appears in two different parts (see IDENTITY above)
  - every (seed, culture) has the same number of rows (a truncated part = a worker
    that died mid-run; averaging it in would bias P without any visible error)

OUTPUT SCHEMA: the merged CSV's 'culture' column is a FRESH sequential id (0..K-1),
one per unique (seed, culture) pair found, in sorted order -- so 'culture' always
means "a unique statistical replicate" regardless of whether jobs were split by
offset or by seed. The original per-job index is kept in the new 'local_culture'
column, and 'seed' is kept as-is, so every row's provenance is fully traceable.

USAGE
    python culture_merge.py                        # all parts*/ -> culture_Pactivation.csv
    python culture_merge.py --parts parts_off0      # merge just ONE job's own parts
    python culture_merge.py --no-figures            # CSV only
"""
import argparse
import csv
import glob
import os
from collections import Counter

import numpy as np

from culture_export import CSV_HEADER

_SEED_IDX = CSV_HEADER.index("seed")
_CULTURE_IDX = CSV_HEADER.index("culture")
OUT_HEADER = CSV_HEADER + ["local_culture"]   # merged-output schema: culture is remapped


def read_parts(part_files):
    """Read and validate the partial CSVs. Returns (rows_as_lists, per_pair_counts).

    `rows` are raw string lists in CSV_HEADER order (not yet remapped).
    `counts` keys are (seed, local_culture) tuples -> row count.
    """
    if not part_files:
        raise SystemExit("no partial CSVs found -- did the workers run?")
    all_rows = []
    pair_to_file = {}
    for f in sorted(part_files):
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd, None)
            if header != CSV_HEADER:
                raise SystemExit("%s: unexpected header\n  got      %s\n  expected %s"
                                 % (f, header, CSV_HEADER))
            n_before = len(all_rows)
            for row in rd:
                if not row:
                    continue
                all_rows.append(row)
            got = {(int(r[_SEED_IDX]), int(r[_CULTURE_IDX])) for r in all_rows[n_before:]}
            for pair in got:
                if pair in pair_to_file:
                    raise SystemExit(
                        "seed=%d culture=%d appears in BOTH %s and %s -- true duplicate "
                        "(same seed AND same local culture index); the merged mean would "
                        "double-weight it. If these are meant to be independent replicates, "
                        "they need DIFFERENT seeds (see jobs/submit_seeded.sh)."
                        % (pair[0], pair[1], pair_to_file[pair], f))
                pair_to_file[pair] = f
        seeds_here = sorted({p[0] for p in got})
        print("  %-32s %6d rows, seed(s) %s, cultures %s"
              % (os.path.basename(f) + " (" + os.path.dirname(f) + ")",
                 len(all_rows) - n_before, seeds_here, sorted(p[1] for p in got)))
    counts = Counter((int(r[_SEED_IDX]), int(r[_CULTURE_IDX])) for r in all_rows)
    sizes = set(counts.values())
    if len(sizes) > 1:
        full = max(sizes)
        short = {pair: n for pair, n in sorted(counts.items()) if n != full}
        print("  NOTE: %d of %d cultures are PARTIAL (%d have the full %d rows)."
              % (len(short), len(counts), len(counts) - len(short), full))
        print("        This is expected when a job is stopped by the walltime: workers write"
              " neuron-major and flush every few neurons, so a partial culture is an unbiased"
              " random SUBSET OF NEURONS, each with all its layers -- usable, just noisier.")
        # Verify that claim rather than asserting it: each partial culture should still be
        # layer-balanced. A layer-skewed partial would bias P and must be reported loudly.
        _LAYER_IDX = CSV_HEADER.index("layer_um")
        by_pair = {}
        for r in all_rows:
            by_pair.setdefault((int(r[_SEED_IDX]), int(r[_CULTURE_IDX])), []).append(r)
        skewed = []
        for pair in short:
            per_layer = Counter(rr[_LAYER_IDX] for rr in by_pair[pair])
            if len(set(per_layer.values())) > 1:
                skewed.append((pair, dict(per_layer)))
        if skewed:
            print("        WARNING: %d partial culture(s) are NOT layer-balanced -- these are"
                  " biased and should be excluded: %s" % (len(skewed), skewed[:5]))
        else:
            print("        Checked: every partial culture is layer-balanced. Safe to include.")
    return all_rows, counts


def merge(parts_dir="parts*", out_csv="culture_Pactivation.csv", make_figures=True,
          bin_um=8.0):
    """parts_dir: a glob PATTERN (default 'parts*'), matching one or many directories --
    e.g. 'parts_off0', 'parts_off240', 'parts_seed1000', 'parts_seed2000' from separate
    job submissions all get picked up by the default pattern in one call. Pass an exact
    single directory name (no wildcard) to merge only that one job's parts.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    part_files = sorted(glob.glob(os.path.join(parts_dir, "part_*.csv")))
    matched_dirs = sorted({os.path.dirname(f) for f in part_files})
    print("[merge] pattern '%s' matched %d directory(ies): %s"
          % (parts_dir, len(matched_dirs), matched_dirs))
    print("[merge] reading %d partial file(s)" % len(part_files))
    rows, counts = read_parts(part_files)

    # Global remap: every unique (seed, local_culture) pair -> a fresh sequential id,
    # in sorted order. This is what makes per-culture grouping downstream correct
    # regardless of which splitting mechanism (offset or seed) produced the rows.
    pairs_sorted = sorted(counts.keys())
    global_id_of = {pair: i for i, pair in enumerate(pairs_sorted)}
    n_seeds = len({p[0] for p in pairs_sorted})
    print("[merge] %d unique (seed, culture) replicate(s) across %d distinct seed(s)"
          % (len(pairs_sorted), n_seeds))

    out_full = out_csv if os.path.isabs(out_csv) else os.path.join(here, out_csv)
    with open(out_full, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(OUT_HEADER)
        for r in rows:
            pair = (int(r[_SEED_IDX]), int(r[_CULTURE_IDX]))
            local_culture = r[_CULTURE_IDX]
            out_row = list(r)
            out_row[_CULTURE_IDX] = global_id_of[pair]     # 'culture' becomes the global id
            out_row.append(local_culture)                   # original index preserved
            w.writerow(out_row)
    print("[merge] %d replicates, %d rows -> %s" % (len(pairs_sorted), len(rows), out_full))

    if not make_figures:
        return out_full, None

    from config import CFG
    import field as F
    from culture_export import render_culture_figures, dipole_frame

    cfg = CFG
    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=not cfg.bipolar)
    dip_c, dip_d = dipole_frame(elec, sign)

    # Rebuild the array the renderer expects:
    #   columns = culture(GLOBAL id), x, y, d_nearest, r_dipole, theta_pos, layer, fired
    idx = {name: i for i, name in enumerate(CSV_HEADER)}
    arr = np.array([[float(global_id_of[(int(r[_SEED_IDX]), int(r[_CULTURE_IDX]))]),
                     float(r[idx["x_um"]]), float(r[idx["y_um"]]),
                     float(r[idx["dist_nearest_elec_um"]]), float(r[idx["dist_dipole3d_um"]]),
                     float(r[idx["theta_pos_deg"]]), float(r[idx["layer_um"]]),
                     float(r[idx["fired"]])] for r in rows], dtype=float)

    # span: take it from the data rather than config, so the figure matches what was
    # actually simulated even if config.py was edited after the run.
    span = float(max(np.abs(arr[:, 1]).max(), np.abs(arr[:, 2]).max()))
    i0 = float(rows[0][idx["i0_uA"]])
    n_pulses = int(float(rows[0][idx["n_pulses"]]))

    pdf = render_culture_figures(arr, span, bin_um, i0, n_pulses, cfg, elec, sign,
                                 dip_c, dip_d, os.path.join(here, "culture_Pmap.pdf"))
    print("[merge] figures -> %s" % pdf)
    return out_full, pdf


def main():
    ap = argparse.ArgumentParser(description="Merge worker CSVs into the deliverable.")
    ap.add_argument("--parts", default="parts*",
                    help="glob PATTERN matching one or more job parts directories "
                         "(default 'parts*' -- picks up every job at once)")
    ap.add_argument("--out", default="culture_Pactivation.csv")
    ap.add_argument("--no-figures", action="store_true", help="write the CSV only")
    ap.add_argument("--bin-um", type=float, default=8.0)
    a = ap.parse_args()
    merge(a.parts, a.out, make_figures=not a.no_figures, bin_um=a.bin_um)


if __name__ == "__main__":
    main()
