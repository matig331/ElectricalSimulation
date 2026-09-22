"""culture_merge.py -- MERGE + (optional) RENDER. No simulation, no NEURON.

Merges the partial CSVs written by culture_worker.py into the three deliverables that
culture_statistics.py consumes, in ONE output directory:
    <out-dir>/culture_Pactivation.csv          (0/1 column 'fired')
    <out-dir>/culture_Pdepolarization.csv      (0/1 column 'depolarized')
    <out-dir>/culture_Phyperpolarization.csv   (0/1 column 'hyperpolarized')
Each file has every identity/provenance column plus exactly ONE outcome column. Optional
quick-look maps culture_Pmap_<outcome>.pdf are rendered by culture_export.render_outcome_set.

MULTIPLE JOBS: --parts is a glob PATTERN of parts directories, e.g.
'results_soma_only/parts_*' (quote it). Each job writes its own parts directory under
results_<cell_model>/ (jobs/run_parallel.sh), so one pattern picks up every job of ONE model.

IDENTITY: a row's true replicate identity is the pair (seed, LOCAL culture index), not the
local culture index alone -- two jobs using DIFFERENT seeds legitimately produce rows with the
SAME local culture index (e.g. both have a "culture 0"), and those are independent
replicates, not duplicates. Conversely the SAME (seed, culture) pair in two different parts is
a true duplicate (e.g. the same job re-submitted) and is rejected -- keeping both would
double-weight that replicate.

VALIDATION (pass 1, streaming, before anything is written):
  - every part has the SAME known header: the current schema (culture_export.CSV_HEADER) or
    the legacy full-active one (LEGACY_CSV_HEADER, read-only: activation file only). The two
    are never merged together.
  - one cell_model and one stimulus amplitude across all parts (never pool models).
  - no (seed, culture) pair in two parts; no (seed, culture, neuron, layer) twice.
  - empty part (worker died before its header) -> skipped with a NOTE.
  - a last row cut by a kill (no newline / wrong field count) -> dropped with a NOTE.
  - a neuron missing some layer (a block cut by a kill) -> dropped with a NOTE, so every
    merged neuron has ALL its layers and partial cultures are unbiased subsets of neurons.
    More than 5% incomplete neurons means a structural problem (e.g. different layers per
    job) -> refused.

OUTPUT SCHEMA: 'culture' is a FRESH sequential id (0..K-1), one per unique (seed, culture)
pair, in sorted order; the original per-job index is kept in 'local_culture', 'seed' as-is.
Memory is O(number of neurons) for the checks, not O(rows x columns).

USAGE
    python culture_merge.py --parts 'results_soma_only/parts_*' --out-dir merged_soma_only
    python culture_merge.py --parts results_soma_only/parts_runB1 --out-dir results_soma_only/runB1 --no-figures
    python culture_merge.py --parts 'parts_run*' --out-dir merged_full_active   # legacy parts
"""
import argparse
import csv
import glob
import os
from array import array
from collections import Counter

import numpy as np

from culture_export import (CELL_MODELS, CSV_HEADER, LEGACY_CSV_HEADER, OUTCOME_CSV_HEADER,
                            OutcomeWriter,
                            PLOT_COLUMNS)

SCHEMAS = {tuple(CSV_HEADER): "current",
           tuple(OUTCOME_CSV_HEADER): "legacy_outcomes_no_kinetics",
           tuple(LEGACY_CSV_HEADER): "legacy_full_active"}
# LEGACY_CSV_HEADER is a prefix of CSV_HEADER, so these indices hold for BOTH schemas
_CULTURE, _NEURON, _LAYER, _I0, _PULSES, _SEED = (
    CSV_HEADER.index(k) for k in ("culture", "neuron", "layer_um", "i0_uA", "n_pulses", "seed"))
_MODEL = CSV_HEADER.index("cell_model")          # current schema only


class Scan(object):
    """Result of scan_parts(): the validated inventory of a set of parts."""

    def __init__(self):
        self.header, self.schema, self.cell_model = None, None, None
        self.files, self.notes = [], []
        self.layer_bit = {}        # layer string -> bit
        self.mask = {}             # (seed, culture, neuron) -> OR of layer bits seen
        self.counts = Counter()    # (seed, culture) -> KEPT rows (complete neurons only)
        self.i0, self.pulses, self.models = set(), set(), set()
        self.full_mask = 0
        self.tail = set()          # (seed, culture, neuron) of the LAST row of each part
        self.n_rows = 0


def _iter_rows(path, n_cols, notes):
    """Yield the complete data rows of one part (its header is skipped; the caller validated
    it). A LAST row that is not newline-terminated or has the wrong number of fields is a
    write cut by a kill -> dropped and noted. A wrong-length row anywhere else -> SystemExit."""
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb") as fb:
        fb.seek(size - 1)
        ends_nl = fb.read(1) in (b"\n", b"\r")
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        prev, prev_line = None, 0
        for row in rd:
            if not row:
                continue
            if prev is not None:
                if len(prev) != n_cols:
                    raise SystemExit("%s line %d: %d fields, expected %d -- corrupt part"
                                     % (path, prev_line, len(prev), n_cols))
                yield prev
            prev, prev_line = row, rd.line_num
        if prev is not None:
            if ends_nl and len(prev) == n_cols:
                yield prev
            else:
                notes.append("%s: dropped a truncated last row (line %d) -- write cut by a kill"
                             % (path, prev_line))


def scan_parts(part_files, expect_model=None):
    """PASS 1 (streaming): validate every part and build the inventory. SystemExit on any
    violation listed in the module docstring. Returns a Scan."""
    if not part_files:
        raise SystemExit("no partial CSVs found -- did the workers run? (check --parts)")
    s = Scan()
    pair_file = {}
    for f in sorted(part_files):
        if os.path.getsize(f) == 0:
            s.notes.append("%s: empty file (worker died before writing its header) -- skipped" % f)
            continue
        with open(f, newline="") as fh:
            header = next(csv.reader(fh), None)
        schema = SCHEMAS.get(tuple(header or ()))
        if schema is None:
            raise SystemExit("%s: unexpected header (%d columns)\n  got      %s\n  "
                             "expected one of the %d known schemas:\n    current (%d cols) %s"
                             "\n    legacy_outcomes_no_kinetics (%d cols)\n    "
                             "legacy_full_active (%d cols)"
                             % (f, len(header or ()), header, len(SCHEMAS), len(CSV_HEADER),
                                CSV_HEADER, len(OUTCOME_CSV_HEADER), len(LEGACY_CSV_HEADER)))
        if s.schema is None:
            s.schema, s.header = schema, list(header)
        elif schema != s.schema:
            raise SystemExit("%s is a %s part but earlier parts are %s -- full-active (legacy) and "
                             "current parts are different models and are never merged together"
                             % (f, schema, s.schema))
        s.files.append(f)
        pairs_here, n_here, key = set(), 0, None
        i_out = [s.header.index(k) for k in ("fired", "depolarized", "hyperpolarized")
                 if k in s.header]
        for r in _iter_rows(f, len(s.header), s.notes):
            vals = [r[i] for i in i_out]
            if any(v not in ("0", "1") for v in vals) or sum(v == "1" for v in vals) > 1:
                raise SystemExit("%s: seed=%s culture=%s neuron=%s layer=%s has outcome flags %s "
                                 "(fired, depolarized, hyperpolarized): each must be 0/1 and at "
                                 "most one may be 1" % (f, r[_SEED], r[_CULTURE], r[_NEURON],
                                                        r[_LAYER], vals))
            seed, c, n = int(r[_SEED]), int(r[_CULTURE]), int(r[_NEURON])
            pair = (seed, c)
            if pair not in pairs_here:
                if pair in pair_file:
                    raise SystemExit(
                        "seed=%d culture=%d appears in BOTH %s and %s -- true duplicate (same "
                        "seed AND same local culture index); the merged statistics would "
                        "double-weight it. If these are meant to be independent replicates, "
                        "they need DIFFERENT seeds (see jobs/submit_seeded.sh)."
                        % (seed, c, pair_file[pair], f))
                pairs_here.add(pair)
            bit = s.layer_bit.setdefault(r[_LAYER], 1 << len(s.layer_bit))
            key = (seed, c, n)
            m = s.mask.get(key, 0)
            if m & bit:
                raise SystemExit("%s: seed=%d culture=%d neuron=%d layer=%s appears twice"
                                 % (f, seed, c, n, r[_LAYER]))
            s.mask[key] = m | bit
            s.i0.add(r[_I0])
            s.pulses.add(r[_PULSES])
            if s.schema == "current":
                s.models.add(r[_MODEL])
            n_here += 1
        if key is not None:
            s.tail.add(key)                    # only this neuron can be cut by a kill
        for p in pairs_here:
            pair_file[p] = f
        s.n_rows += n_here
        seeds = sorted({p[0] for p in pairs_here})
        cults = sorted(p[1] for p in pairs_here)
        print("  %-44s %8d rows | seed(s) %s | %d culture(s)%s"
              % (os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f)), n_here,
                 seeds, len(cults), (" %d..%d" % (cults[0], cults[-1])) if cults else ""))
    if s.schema is None:
        raise SystemExit("every part is empty -- nothing to merge")
    if s.schema == "current":
        if len(s.models) != 1:
            raise SystemExit("parts mix cell models %s -- never pool models; merge each model's "
                             "parts separately" % sorted(s.models))
        s.cell_model = next(iter(s.models))
    else:
        s.cell_model = "full_active"
    if expect_model is not None and s.cell_model != expect_model:
        raise SystemExit("parts are cell_model=%s but --expect-model %s was requested (a job run "
                         "with a different config.cell_model?) -- refusing to merge"
                         % (s.cell_model, expect_model))
    if len(s.i0) != 1:
        raise SystemExit("parts mix stimulus amplitudes %s uA -- P is defined at ONE amplitude"
                         % sorted(s.i0))
    if len(s.pulses) > 1:
        s.notes.append("n_pulses differs across parts %s (stim_duration_s changed between jobs?) "
                       "-- an annotation column only, merged anyway" % sorted(s.pulses))
    s.full_mask = (1 << len(s.layer_bit)) - 1
    n_layers = len(s.layer_bit)
    n_incomplete, n_bad = 0, 0
    for key, m in s.mask.items():
        if m == s.full_mask:
            s.counts[key[:2]] += n_layers
        else:
            n_incomplete += 1
            n_bad += key not in s.tail
    if n_bad:
        # rows are written neuron-major and in order, so a kill (or a partially flushed buffer)
        # can only truncate the LAST neuron of a part; an incomplete neuron anywhere else means
        # the parts do not share the same layers -- exact at any scale, unlike a % threshold
        raise SystemExit("%d neuron(s) lack some of the layers %s and are NOT the last neuron of "
                         "their part file -- a killed job can only truncate the final neuron. "
                         "Were jobs run with different layers_um? Merge them separately."
                         % (n_bad, sorted(s.layer_bit)))
    if n_incomplete:
        s.notes.append("%d part(s) end with a neuron cut by a kill (some layers missing) -- dropped, "
                       "so every merged neuron has all %d layers" % (n_incomplete, n_layers))
    sizes = set(s.counts.values())
    if len(sizes) > 1:
        full = max(sizes)
        n_short = sum(1 for v in s.counts.values() if v != full)
        s.notes.append("%d of %d cultures are PARTIAL (%d have the full %d rows): expected when a "
                       "job hits the walltime. Workers write neuron-major blocks and only complete "
                       "neurons are kept, so a partial culture is an unbiased random subset of "
                       "neurons -- usable, just noisier."
                       % (n_short, len(s.counts), len(s.counts) - n_short, full))
    return s


def read_parts(part_files, expect_model=None):
    """Convenience for tests / small data: (kept rows as string lists, per-(seed,culture) kept
    row counts). merge() itself streams and never holds all rows."""
    s = scan_parts(part_files, expect_model)
    rows = []
    for f in s.files:
        for r in _iter_rows(f, len(s.header), []):
            if s.mask[(int(r[_SEED]), int(r[_CULTURE]), int(r[_NEURON]))] == s.full_mask:
                rows.append(r)
    for n in s.notes:
        print("  NOTE: " + n)
    return rows, s.counts


def merge(parts_pattern, out_dir, make_figures=True, bin_um=8.0, expect_model=None):
    """Validate + merge every part_*.csv under the directories matching `parts_pattern` into
    <out_dir>/culture_P{activation,depolarization,hyperpolarization}.csv (legacy parts:
    activation only). Returns ({outcome: csv}, {outcome: pdf} or None)."""
    out_dir = str(out_dir)
    if out_dir.lower().endswith(".csv"):
        raise SystemExit("--out-dir must be a DIRECTORY (the culture_P*.csv files are created "
                         "inside it), got %r" % out_dir)
    part_files = sorted(glob.glob(os.path.join(parts_pattern, "part_*.csv")))
    matched = sorted({os.path.dirname(f) for f in part_files})
    print("[merge] pattern '%s' matched %d directory(ies): %s" % (parts_pattern, len(matched), matched))
    if os.path.abspath(out_dir) in {os.path.abspath(d) for d in matched}:
        raise SystemExit("--out-dir must not be one of the parts directories")
    s = scan_parts(part_files, expect_model)
    for n in s.notes:
        print("  NOTE: " + n)
    pairs_sorted = sorted(s.counts)
    gid = {pair: i for i, pair in enumerate(pairs_sorted)}
    print("[merge] model=%s schema=%s | %d replicate(s) (seed, culture) across %d seed(s)"
          % (s.cell_model, s.schema, len(pairs_sorted), len({p[0] for p in pairs_sorted})))

    out_header = s.header + ["local_culture"]
    act_path = os.path.join(out_dir, "culture_Pactivation.csv")
    cols = ([array("d") for _ in PLOT_COLUMNS] if make_figures else None)
    pidx = [s.header.index(k) if k in s.header else None for k in PLOT_COLUMNS]
    n_written = 0
    with OutcomeWriter(out_header, act_path) as ow:
        for f in s.files:
            for r in _iter_rows(f, len(s.header), []):
                pair = (int(r[_SEED]), int(r[_CULTURE]))
                if s.mask[pair + (int(r[_NEURON]),)] != s.full_mask:
                    continue
                out = list(r)
                out[_CULTURE] = gid[pair]              # 'culture' becomes the global id
                out.append(r[_CULTURE])                # original local index preserved
                ow.write(out)
                n_written += 1
                if cols is not None:
                    cols[0].append(float(gid[pair]))
                    for k in range(1, len(PLOT_COLUMNS)):
                        j = pidx[k]
                        cols[k].append(float(r[j]) if j is not None else float("nan"))
    paths = ow.paths
    print("[merge] %d rows -> %s" % (n_written, ", ".join(paths[k] for k in sorted(paths))))
    if not make_figures or not n_written:
        return paths, None

    from config import CFG
    import field as F
    from culture_export import render_outcome_set
    elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
    arr = np.column_stack([np.frombuffer(a, dtype=float) for a in cols])
    # span from the data, not config: the figure matches what was actually simulated
    span = float(max(np.abs(arr[:, 1]).max(), np.abs(arr[:, 2]).max()))
    i0 = float(next(iter(s.i0)))
    n_pulses = int(float(sorted(s.pulses)[0]))
    pdfs = render_outcome_set(arr, out_dir, "culture_", span, bin_um, i0, n_pulses, CFG,
                              elec, sign)
    for k in sorted(pdfs):
        print("[merge] figure -> %s" % pdfs[k])
    return paths, pdfs


def main():
    ap = argparse.ArgumentParser(description="Merge worker CSVs into the three deliverables.")
    ap.add_argument("--parts", required=True,
                    help="glob PATTERN of parts directories, e.g. 'results_soma_only/parts_*' "
                         "(quote it so the shell does not expand it)")
    ap.add_argument("--out-dir", required=True,
                    help="output DIRECTORY for culture_P{activation,depolarization,"
                         "hyperpolarization}.csv")
    ap.add_argument("--no-figures", action="store_true", help="write the CSVs only")
    ap.add_argument("--bin-um", type=float, default=8.0)
    ap.add_argument("--expect-model", choices=CELL_MODELS, default=None,
                    help="refuse to merge unless the parts are this cell_model (the job scripts "
                         "pass config.cell_model)")
    a = ap.parse_args()
    merge(a.parts, a.out_dir, make_figures=not a.no_figures, bin_um=a.bin_um,
          expect_model=a.expect_model)


if __name__ == "__main__":
    main()
