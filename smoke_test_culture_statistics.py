"""smoke_test_culture_statistics.py -- OFFLINE test of culture_statistics.py (no NEURON).

Builds worker-format parts with the REAL culture_export.build_row, merges them with the REAL
culture_merge.merge (exactly what the HPC produces), then checks culture_statistics against
hand-computed answers:

  [1] contract: culture_statistics.OUTCOMES == culture_export.OUTCOME_FILES
  [2] Wilson 95% interval == textbook value (k=3, n=10 -> [0.1078, 0.6032])
  [3] end to end on merged HPC output: exact P per distance bin, one outcome per file,
      P(act)+P(dep)+P(hyp) <= 1, every output file present, cell model recorded
  [4] guards: two cell models -> refused; per-job + all-jobs outputs in one --input ->
      duplicate refused; different seeds with the same local culture -> NOT duplicates;
      legacy full-active files still analysable (Part A); zip-style local files (no seed)
      accepted with file-scoped ids; a raw file carrying several outcome columns is safe

    python smoke_test_culture_statistics.py      # expect: 'All smoke tests passed.'
"""
import csv
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import culture_statistics as CS
import culture_export as CE
import culture_merge

FAILURES = []


def check(name, cond, detail=""):
    print("  %-70s %s" % (name, "PASS" if cond else "FAIL " + str(detail)))
    if not cond:
        FAILURES.append(name)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
        return False
    except exc:
        return True


# ---------------------------------------------------------------- [1] contract
print("\n[1] outcome contract between culture_export and culture_statistics")
ok = all(CS.OUTCOMES[k] == ("culture_%s*.csv" % stem, col)
         for k, (stem, col) in CE.OUTCOME_FILES.items()) and \
    sorted(CS.OUTCOMES) == sorted(CE.OUTCOME_FILES)
check("same outcome names, file patterns and 0/1 columns", ok, (CS.OUTCOMES, CE.OUTCOME_FILES))

# ------------------------------------------------------------------- [2] Wilson
print("\n[2] Wilson interval")
lo, hi = CS.wilson_interval(np.array([3]), np.array([10]))
check("k=3, n=10 -> [0.1078, 0.6032]", abs(lo[0] - 0.1078) < 5e-4 and abs(hi[0] - 0.6032) < 5e-4,
      (lo, hi))
lo0, hi0 = CS.wilson_interval(np.array([0]), np.array([20]))
check("k=0 -> lower bound exactly 0, upper > 0", lo0[0] == 0.0 and hi0[0] > 0.0)

# ------------------------------------------------------- synthetic HPC output
PREP = dict(cell_model="soma_only", v_rest=-83.18, ctrl_drift=0.0896)


def outcome_for(dist):
    """Deterministic ground truth: fire < 50 um, depol 50-100, hyperpol 100-150, neutral beyond."""
    if dist < 50:
        return dict(fired=1, dv_end=5.0, outcome="activation", depolarized=0, hyperpolarized=0)
    if dist < 100:
        return dict(fired=0, dv_end=0.2, outcome="depol", depolarized=1, hyperpolarized=0)
    if dist < 150:
        return dict(fired=0, dv_end=-0.2, outcome="hyperpol", depolarized=0, hyperpolarized=1)
    return dict(fired=0, dv_end=0.0, outcome="neutral", depolarized=0, hyperpolarized=0)


def write_part(path, seed, cultures, n=40, layers=(40, 80, 120), prep=PREP):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CE.CSV_HEADER)
        for c in cultures:
            for i in range(n):
                dist = 5.0 * i                     # 0..195 um, 10 somata per 50-um band
                for lay in layers:
                    w.writerow(CE.build_row(c, i, "60308", lay, dist, 0.0, dist, dist, dist,
                                            45.0, 90.0, 36, 50.0, seed, prep, outcome_for(dist)))


tmp = tempfile.mkdtemp()
try:
    res = os.path.join(tmp, "results_soma_only")
    write_part(os.path.join(res, "parts_runS1", "part_000.csv"), 1000, [0, 1])
    write_part(os.path.join(res, "parts_runS2", "part_000.csv"), 2000, [0, 1])  # same local ids
    # per-job merges (what run_parallel.sh writes) and the all-jobs merge (merge_all.sh)
    for tag in ("runS1", "runS2"):
        culture_merge.merge(os.path.join(res, "parts_" + tag), os.path.join(res, tag),
                            make_figures=False, expect_model="soma_only")
    merged = os.path.join(tmp, "merged_soma_only")
    culture_merge.merge(os.path.join(res, "parts_*"), merged, make_figures=False,
                        expect_model="soma_only")

    # ------------------------------------------------------------- [3] end to end
    print("\n[3] end to end on merged HPC output")
    out = CS.analyze_cultures(merged, output_dir=os.path.join(tmp, "stats"), outcome="all",
                              distance_bin_um=50.0, angle_bin_deg=15, orientation_bin_deg=15,
                              xy_bin_um=50.0,
                              dv_threshold_mV=0.0)
    check("three analyses produced", sorted(out) == sorted(CS.OUTCOMES))
    want = {"activation": [1, 0, 0, 0], "depolarization": [0, 1, 0, 0],
            "hyperpolarization": [0, 0, 1, 0]}
    tot = None
    for name, paths in out.items():
        s1 = pd.read_csv(os.path.join(paths["output_dir"], "summary_1d_distance.csv"))
        s1 = s1.sort_values("x")
        p = s1["P_fire"].to_numpy()
        check("%-17s exact P per 50-um band %s" % (name, want[name]),
              np.allclose(p, want[name]), p)
        check("%-17s n per band = 10 somata x 3 layers x 4 cultures = 120" % name,
              set(s1["n_neurons"]) == {120}, set(s1["n_neurons"]))
        tot = p if tot is None else tot + p
        mdf = pd.read_csv(paths["merged"])
        check("%-17s cell_model recorded, 4 distinct cultures" % name,
              set(mdf["cell_model"]) == {"soma_only"} and mdf["culture_global"].nunique() == 4)
        for f in ("statistics_%s.pdf" % name, "summary_2d_xy.csv",
                  "summary_1d_distance_by_morphology.csv"):
            check("%-17s wrote %s" % (name, f), os.path.exists(os.path.join(paths["output_dir"], f)))
    check("P(act)+P(dep)+P(hyp) <= 1 in every band (mutually exclusive)", np.all(tot <= 1 + 1e-12))

    # per-job directory tree (results_<model>/) -> same answer as the all-jobs merge
    out_pj = CS.analyze_one_outcome(res, "activation", os.path.join(tmp, "stats_pj"),
                                    distance_bin_um=50.0)
    n_pj = len(pd.read_csv(out_pj["merged"]))
    n_all = len(pd.read_csv(out["activation"]["merged"]))
    check("results_<model>/ (per-job files) == merged_<model>/ (all jobs): same N",
          n_pj == n_all == 2 * 2 * 40 * 3, (n_pj, n_all))

    # ---------------------------------------------------------------- [4] guards
    print("\n[4] guards")
    both = os.path.join(tmp, "both")               # per-job AND all-jobs outputs in one tree
    shutil.copytree(res, os.path.join(both, "results_soma_only"))
    shutil.copytree(merged, os.path.join(both, "merged_soma_only"))
    check("per-job + all-jobs outputs under one --input -> duplicate refused",
          raises(ValueError, CS.analyze_one_outcome, both, "activation",
                 os.path.join(tmp, "s_both")))

    mixed = os.path.join(tmp, "mixed")
    write_part(os.path.join(mixed, "results_full_active", "parts_runB1", "part_000.csv"), 1000,
               [0], prep=dict(PREP, cell_model="full_active"))
    culture_merge.merge(os.path.join(mixed, "results_full_active", "parts_runB1"),
                        os.path.join(mixed, "fa"), make_figures=False)
    shutil.copytree(os.path.join(res, "runS1"), os.path.join(mixed, "so"))
    check("soma_only + full_active under one --input -> refused (never pooled)",
          raises(ValueError, CS.analyze_one_outcome, mixed, "activation", os.path.join(tmp, "s_mx")))

    # legacy full-active merged file (the preliminary dataset, Part A): no cell_model column
    leg = os.path.join(tmp, "legacy")
    os.makedirs(leg)
    with open(os.path.join(leg, "culture_Pactivation_run1.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CE.LEGACY_CSV_HEADER + ["local_culture"])
        for i in range(40):
            w.writerow([0, i, "60308", 40, 5.0 * i, 0, 5.0 * i, 5.0 * i, 5.0 * i, 45, 90, 16, 50,
                        int(5.0 * i < 50), 1000, 0])
    lo_ = CS.analyze_one_outcome(leg, "activation", os.path.join(tmp, "s_leg"), distance_bin_um=50.0)
    ldf = pd.read_csv(lo_["merged"])
    check("legacy full-active file analysable, model inferred = full_active",
          set(ldf["cell_model"]) == {"full_active"} and len(ldf) == 40)

    # zip-style local soma-only export: soma_only_rest_mV, no seed/cell_model columns
    zl = os.path.join(tmp, "ziplocal")
    os.makedirs(zl)
    zcols = ["culture", "neuron", "morphology", "layer_um", "x_um", "y_um",
             "dist_nearest_elec_um", "dist_center_um", "dist_dipole3d_um", "theta_orient_deg",
             "theta_pos_deg", "n_pulses", "i0_uA", "soma_only_rest_mV",
             "deltaVm_end_phase2_mV", "phase2_outcome", "fired"]
    with open(os.path.join(zl, "culture_Pactivation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(zcols)
        for i in range(10):
            w.writerow([0, i, "60308", 40, i, 0, i, i, i, 1, 1, 36, 50, -83.1, 0.1,
                        "depol", 0])
    zo = CS.analyze_one_outcome(zl, "activation", os.path.join(tmp, "s_zip"))
    check("zip-style local file accepted, tagged soma_only_v0_scalar_rest (never pooled with sham-referenced data), file-scoped ids",
          set(pd.read_csv(zo["merged"])["cell_model"]) == {"soma_only_v0_scalar_rest"})

    # a raw file carrying ALL outcome columns must not produce a duplicate 'fired' column
    raw = os.path.join(tmp, "raw")
    os.makedirs(raw)
    shutil.copy(os.path.join(res, "parts_runS1", "part_000.csv"),
                os.path.join(raw, "culture_Pdepolarization_raw.csv"))
    ro = CS.analyze_one_outcome(raw, "depolarization", os.path.join(tmp, "s_raw"),
                                distance_bin_um=50.0,
                              dv_threshold_mV=0.0)
    rdf = pd.read_csv(ro["merged"])
    check("raw multi-outcome file: analysed column is 'depolarized', no clash",
          list(rdf.columns).count("fired") == 1 and
          int(rdf["fired"].sum()) == 2 * 10 * 3)

    # [5] --dedupe-identical: batch A and batch B were run with the SAME seeds (1000..4000),
    #     so their overlapping local cultures are the same draws simulated twice
    print("\n[5] --dedupe-identical (two batches, same seeds)")
    def legacy_file(path, fired_of):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(CE.LEGACY_CSV_HEADER + ["local_culture"])
            for i in range(40):
                w.writerow([0, i, "60308", 40, 5.0 * i, 0, 5.0 * i, 5.0 * i, 5.0 * i, 45, 90,
                            16, 50, fired_of(i), 1000, 0])
    ab = os.path.join(tmp, "batchAB")
    legacy_file(os.path.join(ab, "run1", "culture_Pactivation.csv"), lambda i: int(i < 10))
    legacy_file(os.path.join(ab, "runB1", "culture_Pactivation.csv"), lambda i: int(i < 10))
    check("same-seed batches without the flag -> refused",
          raises(ValueError, CS.analyze_one_outcome, ab, "activation", os.path.join(tmp, "s_ab0")))
    do = CS.analyze_one_outcome(ab, "activation", os.path.join(tmp, "s_ab1"),
                                distance_bin_um=50.0, dedupe_identical=True)
    ddf = pd.read_csv(do["merged"])
    check("with --dedupe-identical: one copy kept (N=40, not 80), P unchanged",
          len(ddf) == 40 and int(ddf["fired"].sum()) == 10, (len(ddf), int(ddf["fired"].sum())))
    ab2 = os.path.join(tmp, "batchAB_conflict")
    legacy_file(os.path.join(ab2, "run1", "culture_Pactivation.csv"), lambda i: int(i < 10))
    legacy_file(os.path.join(ab2, "runB1", "culture_Pactivation.csv"), lambda i: int(i < 11))
    check("duplicates that DISAGREE are refused even with --dedupe-identical",
          raises(ValueError, CS.analyze_one_outcome, ab2, "activation", os.path.join(tmp, "s_ab2"),
                 dedupe_identical=True))

    # [6] |DeltaV_end| threshold for depolarization / hyperpolarization
    print("\n[6] --dv-threshold (magnitude rule for the polarization outcomes)")
    import re as _re
    import pandas as pd
    def pol_rows():
        rows = []
        spec = ([("activation", 120.0)] * 10 + [("depol", 5.0)] * 20 + [("depol", 1.5)] * 20 +
                [("depol", 0.01)] * 20 + [("hyperpol", -3.0)] * 20 + [("neutral", 0.0)] * 10)
        for i, (lab, dv) in enumerate(spec):
            rows.append(dict(culture=0, neuron=i, morphology="60308", layer_um=40, x_um=10.0 * i,
                             y_um=0.0, dist_nearest_elec_um=10.0 * i, dist_center_um=10.0 * i,
                             dist_dipole3d_um=10.0 * i + 5, theta_orient_deg=45.0,
                             theta_pos_deg=90.0, n_pulses=2, i0_uA=50.0, seed=1000,
                             cell_model="soma_only", deltaVm_end_phase2_mV=dv,
                             phase2_outcome=lab, local_culture=0,
                             depolarized=int(lab == "depol"), hyperpolarized=int(lab == "hyperpol")))
        return pd.DataFrame(rows)
    pdir = os.path.join(tmp, "pol"); os.makedirs(pdir, exist_ok=True)
    P = pol_rows()
    P.drop(columns=["hyperpolarized"]).to_csv(os.path.join(pdir, "culture_Pdepolarization.csv"), index=False)
    P.drop(columns=["depolarized"]).to_csv(os.path.join(pdir, "culture_Phyperpolarization.csv"), index=False)
    def npos(outcome, theta):
        r = CS.analyze_one_outcome(pdir, outcome, os.path.join(tmp, "pol_%s_%g" % (outcome, theta)),
                                   distance_bin_um=100.0, dv_threshold_mV=theta)
        return int(pd.read_csv(r["merged"])["fired"].sum()), r
    got = {th: npos("depolarization", th)[0] for th in (0, 1, 2)}
    check("depolarization: sign-only 60, >=1 mV 40, >=2 mV 20 (spikes never counted)",
          got == {0: 60, 1: 40, 2: 20}, got)
    got = {th: npos("hyperpolarization", th)[0] for th in (0, 1, 5)}
    check("hyperpolarization: sign-only 20, >=1 mV 20, >=5 mV 0", got == {0: 20, 1: 20, 5: 0}, got)
    _, r = npos("depolarization", 1)
    a = pd.read_csv(os.path.join(r["output_dir"], "summary_1d_abs_dv_distance.csv"))
    check("|DeltaV| summary covers every non-firing soma (n = 90)", int(a["n"].sum()) == 90, int(a["n"].sum()))
    pages = len(_re.findall(rb"/Type\s*/Page[^s]", open(r["pdf"], "rb").read()))
    check("polarization PDF has the extra |DeltaV| page (6 pages)", pages == 6, pages)
    pl = CS.polarization_labels
    dv, act = [2.0, -2.0, 0.5, float("nan"), 3.0], [0, 0, 0, 0, 1]
    check("polarization_labels: rule, NaN and spikes",
          list(pl("depolarization", dv, act, 1.0)) == [1, 0, 0, 0, 0]
          and list(pl("hyperpolarization", dv, act, 1.0)) == [0, 1, 0, 0, 0])
    check("theta <= 0 refused by polarization_labels", raises(ValueError, pl, "depolarization", dv, act, 0.0))
    bad = P.copy(); bad.loc[70, "deltaVm_end_phase2_mV"] = 5.0      # labelled hyperpol, dV positive
    bdir = os.path.join(tmp, "pol_bad"); os.makedirs(bdir, exist_ok=True)
    bad.drop(columns=["hyperpolarized"]).to_csv(os.path.join(bdir, "culture_Pdepolarization.csv"), index=False)
    check("labels inconsistent with DeltaV -> refused",
          raises(ValueError, CS.analyze_one_outcome, bdir, "depolarization", os.path.join(tmp, "pb"),
                 dv_threshold_mV=1.0))
    ndir = os.path.join(tmp, "pol_nodv"); os.makedirs(ndir, exist_ok=True)
    P.drop(columns=["hyperpolarized", "deltaVm_end_phase2_mV", "phase2_outcome"]).to_csv(
        os.path.join(ndir, "culture_Pdepolarization.csv"), index=False)
    check("no DeltaV column: threshold refused, sign-only (0) accepted",
          raises(ValueError, CS.analyze_one_outcome, ndir, "depolarization", os.path.join(tmp, "pn1"),
                 dv_threshold_mV=1.0)
          and CS.analyze_one_outcome(ndir, "depolarization", os.path.join(tmp, "pn2"),
                                     dv_threshold_mV=0.0) is not None)

    # [7] edge bins: the maximum belongs to the last bin (no one-sample bin beyond it)
    print("\n[7] edge bins")
    n = 50
    E = pd.DataFrame(dict(culture=0, neuron=range(n), morphology="60308", layer_um=40,
                          x_um=np.linspace(-500, 500, n), y_um=np.linspace(-500, 500, n),
                          dist_nearest_elec_um=50.0, dist_center_um=50.0, dist_dipole3d_um=50.0,
                          theta_orient_deg=np.linspace(0, 90, n), theta_pos_deg=np.linspace(0, 180, n),
                          n_pulses=2, i0_uA=50.0, fired=0, seed=1000, cell_model="soma_only",
                          local_culture=0))
    edir = os.path.join(tmp, "edges"); os.makedirs(edir, exist_ok=True)
    E.to_csv(os.path.join(edir, "culture_Pactivation.csv"), index=False)
    r = CS.analyze_one_outcome(edir, "activation", os.path.join(tmp, "edges_out"),
                               angle_bin_deg=5.0, orientation_bin_deg=5.0, xy_bin_um=10.0)
    for name, top in (("y", 500.0), ("theta_pos", 180.0), ("theta_orient", 90.0)):
        sm = pd.read_csv(os.path.join(r["output_dir"], "summary_1d_%s.csv" % name))
        check("%s: all %d samples binned, last bin ends at %g (not beyond)" % (name, n, top),
              int(sm["n_neurons"].sum()) == n and float(sm["bin_right"].max()) == top,
              (int(sm["n_neurons"].sum()), float(sm["bin_right"].max())))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("All smoke tests passed.")
