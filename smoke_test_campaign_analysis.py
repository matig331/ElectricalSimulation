"""smoke_test_campaign_analysis.py -- OFFLINE checks (no NEURON) of the analysis path at the
size of the full_tuned campaign (6.12 M rows): what changed so that it fits in memory, and
that nothing it computes changed.

    python smoke_test_campaign_analysis.py            # ~1-2 min, ends with ALL PASSED
    python smoke_test_campaign_analysis.py --big      # + a 1.2 M-row time / memory check

  1  plot_bump_statistics.load(): the column reader (csv module and pandas) returns exactly
     the arrays of the previous dict-per-row loader, cell for cell, on rows with blanks,
     'nan' strings and a last row cut by a kill
  2  rows outside the kinetics subsample (bump_culture_fraction < 1) are left out and counted,
     instead of being counted as "no bump"
  3  directory input; the refusals (no kinetics columns, header only, nothing measured)
  4  _pick(): at most max_n points, a seeded subset of the mask, all of them when max_n = 0
  5  main(): every scatter holds <= MAX_SCATTER points, the PDF stays small, the summary CSV
     equals a direct numpy computation and does not depend on how many points are drawn
  6  culture_statistics: the kinetics columns are no longer read, every statistic is identical;
     culture_merge reads the model of the 22-column soma_only parts again (it had labelled
     them full_active since the kinetics columns were added, and refused to merge them)
  7  jobs/merge_all.sh and jobs/analysis.pbs refuse a full_tuned merge without PARTS, and
     merge_all.sh merges exactly the directories PARTS matches
  8  pure ASCII, LF line endings in every file of this delivery
Exit status 1 on any failure (the job scripts can gate on it).
"""
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plot_bump_statistics as PBS            # noqa: E402
from synthetic_campaign import write_parts    # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def same(a, b):
    """Arrays equal cell for cell, nan == nan, same dtype kind."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a.astype(float), b.astype(float), equal_nan=True))
    return bool(np.array_equal(a, b))


# ------------------------------------------------------------------ the previous loader
def legacy_load(path):
    """plot_bump_statistics.load() as it was before this change (dict per row), verbatim except
    that it also returns dv_t0 so the measured rows can be selected for the comparison."""
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*Pactivation*.csv"))) or \
            sorted(glob.glob(os.path.join(path, "*.csv")))
        path = cands[0]
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    get = lambda k: np.array([PBS._f(r.get(k, "")) for r in rows])
    if "dexp_data_peak_mV" in rows[0]:
        data_peak, measured = get("dexp_data_peak_mV"), True
    elif "bump_data_peak_mV" in rows[0]:
        data_peak, measured = get("bump_data_peak_mV"), True
    else:
        data_peak, measured = get("dexp_peak_mV"), False
    return dict(path=path, n=len(rows),
                dist=get(PBS.DIST_COL), theta_pos=get("theta_pos_deg"),
                theta_or=get("theta_orient_deg"), layer=get("layer_um"),
                morph=np.array([r.get("morphology", "") for r in rows]),
                dv_end=get("deltaVm_end_phase2_mV"),
                peak=get("dexp_peak_mV"), t_peak=get("dexp_t_peak_ms"),
                tau_r=get("dexp_tau_rise_ms"), tau_d=get("dexp_tau_decay_ms"),
                r2=get("dexp_r2"), ok=get("dexp_fit_ok") > 0.5,
                data_peak=data_peak, measured_peak=measured,
                t_1e=get("early_t_1e_ms"), tau_m=get("early_tau_ms"),
                early_ok=get("early_fit_ok") > 0.5, dv_t0=get("dexp_dv_t0_mV"))


ARRAY_KEYS = ("dist", "theta_pos", "theta_or", "layer", "morph", "dv_end", "peak", "t_peak",
              "tau_r", "tau_d", "r2", "ok", "data_peak", "t_1e", "tau_m", "early_ok")


def one_part(tmp, name, n_cultures, n_neurons, fraction=1.0, seed=10000):
    """Path of a single raw part (the worker schema) with the given culture layout."""
    files = write_parts(os.path.join(tmp, name), n_cultures, n_neurons, seed, n_workers=1,
                        measured_fraction=fraction)
    return files[0]


def damage(path):
    """Blank and 'nan' cells in measured rows, then a last row cut in the middle of a field
    (no newline): what a part being written, or a hand-edited file, can hold."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    ip, it, idd = (head.index(c) for c in ("dexp_peak_mV", "dexp_tau_rise_ms",
                                          "dist_dipole3d_um"))
    rows[3][ip] = ""
    rows[5][it] = "nan"
    rows[7][idd] = "NaN"
    rows[9][head.index("morphology")] = ""
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
        # cut inside the early_* block, before dexp_dv_t0_mV: the row has no DeltaV at t0
        cut = ",".join(rows[1][:head.index("early_t_1e_ms") + 1])
        fh.write(cut[:-2])
    return len(rows) - 1 + 1          # data rows including the cut one


# ------------------------------------------------------------------------------- tests
def test_loader(tmp):
    print("1  loader: csv module and pandas == the previous dict-per-row loader")
    path = one_part(tmp, "parts_eq", n_cultures=3, n_neurons=150)
    n_written = damage(path)
    old = legacy_load(path)
    new_csv = PBS.load(path, use_pandas=False)
    import importlib.util
    if importlib.util.find_spec("pandas") is not None:
        new_pd = PBS.load(path, use_pandas=True)
    else:
        new_pd = None
        print("  note  pandas not installed: only the csv-module path is compared")
    keep = np.isfinite(old["dv_t0"])
    check(old["n"] == n_written, "legacy loader saw all %d rows" % n_written)
    check(new_csv["n"] == int(keep.sum()) and new_csv["n_unmeasured"] == int((~keep).sum()),
          "new loader: %d measured + %d without DeltaV at t0 (the cut row)"
          % (new_csv["n"], new_csv["n_unmeasured"]))
    bad = [k for k in ARRAY_KEYS if not same(old[k][keep], new_csv[k])]
    check(not bad, "csv-module path identical to the legacy loader on %d measured rows%s"
          % (int(keep.sum()), (" -- differ: %s" % bad) if bad else ""))
    if new_pd is not None:
        bad = [k for k in ARRAY_KEYS if not same(new_csv[k], new_pd[k])]
        check(not bad, "pandas path identical to the csv-module path%s"
              % ((" -- differ: %s" % bad) if bad else ""))
        check(new_pd["n"] == new_csv["n"] and new_pd["n_unmeasured"] == new_csv["n_unmeasured"],
              "pandas path: same counts")
    check(new_csv["measured_peak"] is True, "measured-peak column recognised")
    check(new_csv["morph"].dtype.kind == "U" and "" in set(new_csv["morph"]),
          "morphology kept as text, a blank cell stays blank")


def test_unmeasured(tmp):
    print("2  cultures outside the kinetics subsample are left out, not counted as 'no bump'")
    path = one_part(tmp, "parts_frac", n_cultures=12, n_neurons=40, fraction=0.5)
    from culture_export import culture_has_kinetics
    meas_c = [c for c in range(12) if culture_has_kinetics(10000, c, 0.5)]
    n_meas = len(meas_c) * 40 * 3
    d = PBS.load(path)
    old = legacy_load(path)
    check(0 < len(meas_c) < 12, "the fraction split the cultures (%d of 12 measured)"
          % len(meas_c))
    check(d["n"] == n_meas and d["n_unmeasured"] == 12 * 120 - n_meas,
          "n = %d measured rows, %d unmeasured left out" % (d["n"], d["n_unmeasured"]))
    frac_new = d["ok"].mean()
    frac_old = old["ok"].mean()
    check(abs(frac_new - old["ok"][np.isfinite(old["dv_t0"])].mean()) < 1e-12
          and frac_old < frac_new,
          "accepted fraction %.3f over measured rows (the old loader diluted it to %.3f)"
          % (frac_new, frac_old))


def test_inputs(tmp):
    print("3  directory input and refusals")
    d = os.path.join(tmp, "merged_dir")
    os.makedirs(d)
    src = one_part(tmp, "parts_dir", n_cultures=1, n_neurons=10)
    shutil.copy(src, os.path.join(d, "culture_Pactivation.csv"))
    shutil.copy(src, os.path.join(d, "aaa_other.csv"))
    check(PBS.load(d)["path"].endswith("culture_Pactivation.csv"),
          "a folder resolves to its *Pactivation*.csv")

    def refused(path, why):
        try:
            PBS.load(path)
        except SystemExit as exc:
            check(True, "refused %s: %s" % (why, str(exc)[:70]))
            return
        check(False, "should have refused " + why)

    no_kin = os.path.join(tmp, "no_kin.csv")
    with open(no_kin, "w") as fh:
        fh.write("culture,neuron,dist_dipole3d_um\n0,0,10.0\n")
    refused(no_kin, "a short-window file")
    head_only = os.path.join(tmp, "head_only.csv")
    with open(src) as fi, open(head_only, "w") as fo:
        fo.write(fi.readline())
    refused(head_only, "a header-only file")
    none_meas = one_part(tmp, "parts_none", n_cultures=2, n_neurons=5, fraction=0.0)
    refused(none_meas, "a file with no measured row")


def test_pick():
    print("4  _pick: bounded, reproducible, inside the mask")
    rng = np.random.default_rng(3)
    mask = rng.random(100000) < 0.3
    a = PBS._pick(mask, 1000, seed=7)
    b = PBS._pick(mask, 1000, seed=7)
    check(a.size == 1000 and np.array_equal(a, b), "1000 of %d, same seed -> same subset"
          % int(mask.sum()))
    check(bool(mask[a].all()) and np.all(np.diff(a) > 0), "all inside the mask, sorted, unique")
    check(PBS._pick(mask, 0).size == int(mask.sum()), "max_n = 0 -> every point")
    small = np.zeros(50, bool)
    small[[3, 9]] = True
    check(list(PBS._pick(small, 1000)) == [3, 9], "fewer points than max_n -> all of them")


def _scatter_sizes(fig):
    return [len(c.get_offsets()) for ax in fig.axes for c in ax.collections
            if hasattr(c, "get_offsets") and c.get_offsets() is not None
            and len(c.get_offsets()) > 1]


def test_main(tmp):
    print("5  main(): bounded scatters, small PDF, summary == direct numpy")
    parts = os.path.join(tmp, "parts_main")
    write_parts(parts, n_cultures=8, n_neurons=2500, seed=12000, n_workers=2)
    merged = os.path.join(tmp, "merged_main")
    import culture_merge
    culture_merge.merge(parts, merged, make_figures=False, expect_model="full_tuned")
    act = os.path.join(merged, "culture_Pactivation.csv")
    t0 = time.time()
    pdf, out_csv = PBS.main(act, os.path.join(tmp, "b.pdf"), os.path.join(tmp, "b.csv"),
                            verbose=False)
    dt = time.time() - t0
    d = PBS.load(act)
    check(d["n"] == 8 * 2500 * 3, "read all %d rows of the merge" % d["n"])
    size_mb = os.path.getsize(pdf) / 1e6
    check(size_mb < 3.0, "PDF %.2f MB for %d rows (the previous version: ~23 MB per M rows)"
          % (size_mb, d["n"]))
    import matplotlib.pyplot as plt
    edges = np.arange(0.0, 800.0, 40.0)
    sizes = []
    for fig in (PBS.page_amplitude(d, edges, 0.05), PBS.page_shape(d, edges),
                PBS.page_geometry(d)):
        sizes += _scatter_sizes(fig)
        plt.close(fig)
    check(sizes and max(sizes) <= PBS.MAX_SCATTER,
          "largest scatter %d points <= MAX_SCATTER %d" % (max(sizes), PBS.MAX_SCATTER))
    with open(out_csv) as fh:
        summ = dict((r["quantity"], r) for r in csv.DictReader(fh))
    ok = d["ok"]
    med = np.median(d["peak"][ok][np.isfinite(d["peak"][ok])])
    check(abs(float(summ["bump_peak_mV"]["median"]) - round(float(med), 5)) < 1e-12
          and int(summ["bump fit accepted"]["n"]) == int(ok.sum()),
          "summary: median peak %s mV over %s accepted fits, as numpy computes it"
          % (summ["bump_peak_mV"]["median"], summ["bump fit accepted"]["n"]))
    _p2, csv_all = PBS.main(act, os.path.join(tmp, "b_all.pdf"), os.path.join(tmp, "b_all.csv"),
                            verbose=False, max_scatter=500)
    with open(out_csv) as f1, open(csv_all) as f2:
        check(f1.read() == f2.read(), "summary CSV independent of the points drawn")
    print("        (%d rows in %.1f s)" % (d["n"], dt))


def test_statistics(tmp):
    print("6  culture_statistics: kinetics columns not read, statistics identical")
    import culture_statistics as CS
    import culture_merge
    parts = os.path.join(tmp, "parts_stats")
    write_parts(parts, n_cultures=4, n_neurons=300, seed=13000, n_workers=2)
    merged = os.path.join(tmp, "merged_stats")
    culture_merge.merge(parts, merged, make_figures=False, expect_model="full_tuned")
    act = os.path.join(merged, "culture_Pactivation.csv")
    from pathlib import Path
    df = CS.load_and_merge([Path(act)], "fired")
    kin = [c for c in df.columns if str(c).startswith(("early_", "dexp_"))]
    with open(act) as fh:
        head = next(csv.reader(fh))
    check(not kin and all(c in df.columns for c in head
                          if not c.startswith(("early_", "dexp_"))),
          "load_and_merge keeps the %d other columns and none of the 24 kinetics ones"
          % len([c for c in head if not c.startswith(("early_", "dexp_"))]))
    outs = {}
    saved = CS.KINETICS_PREFIXES
    for tag, pref in (("new", saved), ("all_columns", ())):
        CS.KINETICS_PREFIXES = pref
        try:
            out = os.path.join(tmp, "stats_" + tag)
            CS.analyze_cultures(merged, output_dir=out, outcome="activation",
                                distance_bin_um=5, angle_bin_deg=5, orientation_bin_deg=5,
                                xy_bin_um=10)
            outs[tag] = out
        finally:
            CS.KINETICS_PREFIXES = saved
    names = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(outs["new"], "activation", "summary_*.csv")))
    diff = [n for n in names if open(os.path.join(outs["new"], "activation", n)).read()
            != open(os.path.join(outs["all_columns"], "activation", n)).read()]
    check(names and not diff, "%d summary tables identical to reading every column%s"
          % (len(names), (" -- differ: %s" % diff) if diff else ""))


def test_merge_models(tmp):
    print("6b culture_merge: the model of every schema that records one")
    import culture_merge
    from culture_export import OUTCOME_CSV_HEADER, LEGACY_CSV_HEADER, CSV_HEADER
    src = one_part(tmp, "parts_src", n_cultures=2, n_neurons=10)
    with open(src, newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    im = CSV_HEADER.index("cell_model")
    for tag, header, model in (("soma22", OUTCOME_CSV_HEADER, "soma_only"),
                               ("legacy15", LEGACY_CSV_HEADER, None)):
        d = os.path.join(tmp, "parts_" + tag)
        os.makedirs(d)
        with open(os.path.join(d, "part_000.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for r in rows:
                r = list(r)
                r[im] = model or r[im]
                w.writerow(r[:len(header)])
        expect = model or "full_active"
        try:
            culture_merge.merge(d, os.path.join(tmp, "merged_" + tag), make_figures=False,
                                expect_model=expect)
            got = True
        except SystemExit as exc:
            got = str(exc)[:90]
        check(got is True, "%d-column parts merge as %s%s"
              % (len(header), expect, "" if got is True else " -- " + str(got)))
    try:
        culture_merge.merge(os.path.join(tmp, "parts_soma22"), os.path.join(tmp, "m_x"),
                            make_figures=False, expect_model="full_tuned")
        check(False, "soma_only parts must not merge as full_tuned")
    except SystemExit:
        check(True, "soma_only parts refused under --expect-model full_tuned")


def test_job_guards(tmp):
    print("7  merge_all.sh / analysis.pbs: full_tuned needs PARTS; PARTS selects exactly")
    repo = os.path.join(tmp, "repo")
    os.makedirs(os.path.join(repo, "jobs"))
    for f in ("culture_merge.py", "culture_export.py", "bump_kinetics.py", "field.py",
              "config.py"):
        shutil.copy(os.path.join(HERE, f), repo)
    for f in ("merge_all.sh", "analysis.pbs", "results_layout.sh"):
        shutil.copy(os.path.join(HERE, "jobs", f), os.path.join(repo, "jobs"))
    cfg = os.path.join(repo, "config.py")
    src = open(cfg).read()
    src = src.replace('    cell_model: str = "soma_only"', '    cell_model: str = "full_tuned"', 1)
    open(cfg, "w").write(src)
    with open(os.path.join(repo, "jobs", "env_setup.sh"), "w") as fh:
        fh.write("export MPLBACKEND=Agg\n")          # the real one activates conda + NEURON
    res = os.path.join(repo, "results_full_tuned")
    write_parts(os.path.join(res, "parts_ftc01"), 2, 20, 10000, n_workers=2)
    write_parts(os.path.join(res, "parts_ftc02"), 2, 20, 11000, n_workers=2)
    write_parts(os.path.join(res, "parts_dryrun_ft"), 2, 20, 10000, n_workers=1)
    shim = os.path.join(tmp, "bin")                  # `python` = this interpreter
    os.makedirs(shim, exist_ok=True)
    if not os.path.exists(os.path.join(shim, "python")):
        os.symlink(sys.executable, os.path.join(shim, "python"))
    env = dict(os.environ, PBS_O_WORKDIR=repo, PATH=shim + os.pathsep + os.environ["PATH"])
    for k in ("PARTS", "MERGED", "MODEL", "OUT"):
        env.pop(k, None)
    r = subprocess.run(["bash", "jobs/merge_all.sh", "--no-figures"], cwd=repo, env=env,
                       capture_output=True, text=True)
    check(r.returncode != 0 and "parts_dryrun_ft" in r.stderr,
          "merge_all.sh without PARTS refuses and lists the test runs")
    r = subprocess.run(["bash", "jobs/merge_all.sh", "--no-figures"], cwd=repo,
                       env=dict(env, PARTS="results_full_tuned/parts_ftc*"),
                       capture_output=True, text=True)
    act = os.path.join(repo, "merged_full_tuned", "culture_Pactivation.csv")
    n_rows = sum(1 for _ in open(act)) - 1 if os.path.exists(act) else -1
    check(r.returncode == 0 and n_rows == 2 * 2 * 20 * 3,
          "PARTS='...parts_ftc*' merges the 2 campaign jobs only (%d rows)" % n_rows)
    r = subprocess.run(["bash", "jobs/analysis.pbs"], cwd=repo, env=env, capture_output=True,
                       text=True)
    check(r.returncode != 0 and "PARTS=results_full_tuned/parts_ftc" in r.stderr,
          "analysis.pbs without PARTS or MERGED refuses")


def test_ascii():
    print("8  pure ASCII, LF only")
    files = ["plot_bump_statistics.py", "culture_statistics.py", "culture_merge.py",
             "synthetic_campaign.py", "smoke_test_campaign_analysis.py",
             "smoke_test_culture_maps.py", "jobs/merge_all.sh", "jobs/analysis.pbs"]
    bad = []
    for f in files:
        b = open(os.path.join(HERE, f), "rb").read()
        if b"\r" in b or any(x > 127 for x in b):
            bad.append(f)
    check(not bad, "%d files%s" % (len(files), (" -- not clean: %s" % bad) if bad else ""))


def test_big(tmp):
    print("9  --big: 1.2 M rows through the loader")
    import resource
    parts = os.path.join(tmp, "parts_big")
    write_parts(parts, n_cultures=16, n_neurons=25000, seed=20000, n_workers=4)
    rows = []
    for f in sorted(glob.glob(os.path.join(parts, "part_*.csv"))):
        rows.append(f)
    big = os.path.join(tmp, "big.csv")
    with open(big, "w") as fo:
        for k, f in enumerate(rows):
            with open(f) as fi:
                head = fi.readline()
                if k == 0:
                    fo.write(head)
                shutil.copyfileobj(fi, fo)
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    t0 = time.time()
    d = PBS.load(big)
    dt = time.time() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    check(d["n"] == 16 * 25000 * 3, "%d rows in %.1f s, peak memory %.0f MB (was %.0f)"
          % (d["n"], dt, r1, r0))


def main():
    tmp = tempfile.mkdtemp(prefix="smoke_campaign_analysis_")
    try:
        test_loader(tmp)
        test_unmeasured(tmp)
        test_inputs(tmp)
        test_pick()
        test_main(tmp)
        test_statistics(tmp)
        test_merge_models(tmp)
        test_job_guards(tmp)
        test_ascii()
        if "--big" in sys.argv:
            test_big(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("-" * 70)
    if FAILS:
        print("FAILED: %d check(s)" % len(FAILS))
        for m in FAILS:
            print("   - " + m)
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
