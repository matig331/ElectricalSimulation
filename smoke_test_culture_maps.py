"""smoke_test_culture_maps.py -- offline (no NEURON) test of culture_maps.py.

    python smoke_test_culture_maps.py        # expect the last line: 'All smoke tests passed.'

[1] for every outcome, culture_maps hands the renderer EXACTLY the inputs culture_merge's own
    figure path hands it (same arrays bit for bit, same span / bin / i0 / n_pulses / electrodes)
[2] a real render writes a non-empty PDF
[3] an outcome file that does not exist fails loudly (legacy full-active data: activation only)
"""
import csv
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import culture_export as CE
import culture_merge
import culture_maps

FAILURES = []


def check(name, cond, detail=""):
    print("  %-70s %s" % (name, "PASS" if cond else "FAIL " + str(detail)), flush=True)
    if not cond:
        FAILURES.append(name)


def write_part(path, seed, cultures, n_neurons, layers, rng):
    """Valid current-schema rows: exclusive outcomes, every neuron at every layer."""
    labels = {"activation": (1, 0, 0), "depol": (0, 1, 0), "hyperpol": (0, 0, 1),
              "neutral": (0, 0, 0)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CE.CSV_HEADER)
        for c in cultures:
            for n in range(n_neurons):
                x, y = rng.uniform(-300, 300, 2)
                for L in layers:
                    lab = rng.choice(list(labels))
                    f, d, h = labels[lab]
                    r = float(np.hypot(x, y))
                    w.writerow([c, n, "60308", int(L), round(x, 2), round(y, 2), round(r, 2),
                                round(r, 2), round(r + 1.0, 2), round(rng.uniform(0, 90), 1),
                                round(rng.uniform(0, 180), 1), 2, 50.0, f, seed, "soma_only",
                                -83.18, 0.0896, round(rng.normal(), 6), lab, d, h])


def capture(fn):
    """Run fn() with render_outcome_maps replaced by a recorder; returns {outcome: kwargs}."""
    calls = {}
    real = CE.render_outcome_maps

    def fake(pdf_path, tag, outcome, **ctx):
        calls[tag] = dict(ctx, outcome=np.asarray(outcome, float).copy())
        return pdf_path
    CE.render_outcome_maps = fake
    try:
        fn()
    finally:
        CE.render_outcome_maps = real
    return calls


tmp = tempfile.mkdtemp(prefix="stmaps_")
try:
    rng = np.random.default_rng(3)
    write_part(os.path.join(tmp, "parts_a", "part_000.csv"), 1000, [0, 1], 30, (40, 80), rng)
    write_part(os.path.join(tmp, "parts_b", "part_000.csv"), 2000, [0, 1, 2], 30, (40, 80), rng)
    pattern = os.path.join(tmp, "parts_*")
    merged = os.path.join(tmp, "merged")

    print("\n[1] culture_maps == the merge's own figure inputs")
    ref = capture(lambda: culture_merge.merge(pattern, merged, make_figures=True,
                                              expect_model="soma_only"))
    check("merge drew all three outcomes", sorted(ref) == sorted(CE.OUTCOME_FILES), sorted(ref))
    for outcome in sorted(CE.OUTCOME_FILES):
        got = capture(lambda: culture_maps.render_one(merged, outcome, os.path.join(tmp, "o"),
                                                          dv_threshold_mV=0.0))
        a, b = ref[outcome], got.get(outcome, {})
        same = sorted(a) == sorted(b)
        bad = []
        for k in a:
            if k not in b:
                continue
            va, vb = a[k], b[k]
            if k == "cfg":
                ok = va is vb
            elif isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                ok = np.array_equal(np.asarray(va), np.asarray(vb), equal_nan=True)
            else:
                ok = va == vb
            if not ok:
                bad.append(k)
        check("%s: identical renderer inputs (%d arrays/params)" % (outcome, len(a)),
              same and not bad, bad or (sorted(a), sorted(b)))

    print("\n[2] real render")
    pdf = culture_maps.render_one(merged, "hyperpolarization", os.path.join(tmp, "real"))
    check("PDF written and non-trivial (%d bytes)" % os.path.getsize(pdf),
          os.path.getsize(pdf) > 5000)

    print("\n[2b] |DeltaV| threshold relabels the polarization maps (culture_statistics rule)")
    import pandas as pd
    from culture_statistics import polarization_labels
    for outcome, stem in (("depolarization", "Pdepolarization"), ("hyperpolarization", "Phyperpolarization")):
        f = os.path.join(merged, "culture_%s.csv" % stem)
        arr, _i0, _np = culture_maps.load_plot_array(f, outcome, 1.0)
        d = pd.read_csv(f)
        want = polarization_labels(outcome, d["deltaVm_end_phase2_mV"].to_numpy(float),
                                   d["phase2_outcome"].eq("activation").to_numpy(), 1.0)
        col = CE.OUTCOME_FILES[outcome][1]
        check("%s: map labels == polarization_labels(|dV| >= 1 mV)" % outcome,
              np.array_equal(arr[:, CE.PLOT_COLUMNS.index(col)], want))

    print("\n[3] missing outcome file")
    os.remove(os.path.join(merged, "culture_Pdepolarization.csv"))
    try:
        culture_maps.render_one(merged, "depolarization")
        check("missing merged file -> refused", False)
    except SystemExit:
        check("missing merged file -> refused", True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if FAILURES:
    print("\nFAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("\nAll smoke tests passed.")
