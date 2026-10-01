"""smoke_test_culture_states.py -- offline (no NEURON) test of culture_states_hpc.py against the pipeline.

    python smoke_test_culture_states.py        # expect the last line: 'All smoke tests passed.'

culture_states_hpc.py is SELF-CONTAINED on purpose (numpy, pandas, matplotlib, morphio: it runs
anywhere), so it carries COPIES of a few pieces of the pipeline. This test pins every copy to its
source of truth: a change to the campaign code that the copy does not follow fails here, instead of
silently mis-drawing a figure.
[1] pure ASCII; imports with no project module; its built-in self-test passes
[2] assign_morphologies / draws == culture_export.assign_morphologies / culture_draws, bit for bit,
    in the CURRENT soma square of config (placement_frame) and in the LEGACY one (before 2026-09-28)
[3] infer_params recovers the culture size, half side and centre of the square from a campaign table,
    also for a culture cut short (a job still running): checked on the rotations, not only the positions
[4] states == the labels of culture_statistics (activation; polarization_labels at 1 mV)
[5] the electrode outlines == field.default_array(); ELEC_UM == config.electrode_um
[6] slicing == slicer.py (synthetic trees; the real eyal_archive too when it is found); the specimen
    file is the one the campaign picks; the rotation convention == rich_footprint._placed (the simulation)
[7] end to end on the three tables a campaign leaves on disk -- a raw worker part, culture_merge's
    merge of two jobs (`culture` renumbered, `local_culture` kept) and culture_statistics'
    merged_activation.csv: culture names, rotations == culture_draws theta, a figure with all six
    morphologies drawn and the counts of the table; a raw part whose last row is half-written
[8] jobs/culture_states.pbs: bash -n, ASCII, LF only; the self-test count it and the HOWTO quote
"""
import ast
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import culture_states_hpc as CS
import culture_export as CE
import culture_merge
import culture_statistics
import field as F
import synthetic_campaign
from config import CFG

FAILURES = []
SCRIPT = os.path.join(HERE, "culture_states_hpc.py")
PROJECT_MODULES = ("config", "culture_export", "culture_merge", "culture_statistics", "field",
                   "slicer", "morphologies", "bump_kinetics", "rich_footprint")


def check(name, cond, detail=""):
    print("  %-78s %s" % (name, "PASS" if cond else "FAIL " + str(detail)), flush=True)
    if not cond:
        FAILURES.append(name)


def geometry():
    elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
    dip_c, dip_d = CE.dipole_frame(elec, sign)
    return elec, sign, CE.electrode_center(elec), CE.dipole_axis_deg(elec, sign), dip_c, dip_d


def campaign_draws(seed, c, n, legacy=False):
    """culture_export.culture_draws of culture c in config's frame (legacy=True: before 2026-09-28)."""
    elec, sign, center, axis, dip_c, dip_d = geometry()
    span, pc = CE.placement_frame(CFG, elec, sign, legacy=legacy)
    d = CE.culture_draws(seed, c, n, len(CFG.morphologies), span, elec, center, dip_c, dip_d, axis,
                         CFG.h_soma_um, place_center=pc)
    return d, float(span), (float(pc[0]) + 0.0, float(pc[1]) + 0.0)


def campaign_rows(seed, c, n, legacy=False):
    """Worker rows (CSV_HEADER order, synthetic outcomes) whose placement IS the campaign's."""
    d, _span, _pc = campaign_draws(seed, c, n, legacy)
    rows = synthetic_campaign.culture_rows(seed, c, n, measured=False)
    H = CE.CSV_HEADER
    for k, r in enumerate(rows):
        i = k // len(synthetic_campaign.LAYERS)
        r[H.index("morphology")] = CFG.morphologies[d["midx"][i]]
        r[H.index("x_um")] = repr(round(float(d["pos"][i, 0]), 2))
        r[H.index("y_um")] = repr(round(float(d["pos"][i, 1]), 2))
        r[H.index("theta_orient_deg")] = repr(round(float(d["th_or"][i]), 1))
        r[H.index("dist_dipole3d_um")] = repr(round(float(d["r_dip"][i]), 2))
    return rows


def write_part(path, blocks):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CE.CSV_HEADER)
        for rows in blocks:
            w.writerows(rows)


ASC = """("CellBody"
  (Color Red)
  (CellBody)
  (  -5.0   0.0  0.0  0.1)
  (   0.0   5.0  0.0  0.1)
  (   5.0   0.0  0.0  0.1)
  (   0.0  -5.0  0.0  0.1)
)

( (Color Green)
  (Dendrite)
  (   0.0   5.0  0.0  1.0)
  (  {a:.1f}  10.0  2.0  1.0)
  (  40.0  15.0  4.0  1.0)
  (
    (  60.0  20.0  30.0  1.0)
    (  80.0  30.0  {b:.1f}  1.0)
  |
    (  60.0  10.0  5.0  1.0)
    (  90.0   5.0  8.0  1.0)
  )
)

( (Color Blue)
  (Apical)
  (   0.0   5.0  0.0  1.5)
  (   0.0  60.0  15.0  1.5)
  (   0.0 120.0  {c:.1f}  1.5)
)

( (Color Red)
  (Axon)
  (   0.0  -5.0  0.0  0.5)
  (   0.0 -50.0  0.0  0.5)
)
"""


def write_archive(root):
    """eyal_archive/specimen_<id>/morphology.asc for the six specimens: small synthetic trees whose
    sections cross the 40 / 80 / 120 um slabs differently (and one axon, never kept)."""
    for k, m in enumerate(CFG.morphologies):
        p = os.path.join(root, "eyal_archive", "specimen_%s" % m, "morphology.asc")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(ASC.format(a=20.0 + 3 * k, b=25.0 + 9 * k, c=18.0 + 8 * k))
    return os.path.join(root, "eyal_archive")


tmp = tempfile.mkdtemp(prefix="smoke_cstates_")
try:
    print("\n[1] self-contained module and its built-in self-test")
    src = open(SCRIPT, "rb").read()
    check("culture_states_hpc.py is pure ASCII, LF only",
          all(b < 128 for b in src) and b"\r" not in src)
    iso = os.path.join(tmp, "iso")
    os.makedirs(iso)
    shutil.copy(SCRIPT, iso)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    probe = ("import sys; import culture_states_hpc; "
             "print(sorted(m for m in sys.modules if m in %r))" % (PROJECT_MODULES,))
    r = subprocess.run([sys.executable, "-c", probe], cwd=iso, env=env, capture_output=True, text=True)
    check("imports alone (no project module on the path, none loaded)",
          r.returncode == 0 and r.stdout.strip() == "[]", r.stdout + r.stderr)
    r = subprocess.run([sys.executable, SCRIPT, "--selftest"], cwd=tmp, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    m = re.match(r"^(\d+)/(\d+) passed$", last)
    check("--selftest: every built-in check passes (%s)" % last,
          r.returncode == 0 and m is not None and m.group(1) == m.group(2), r.stdout[-2000:] + r.stderr[-2000:])

    print("\n[2] the draws are the campaign's (culture_export.culture_draws)")
    rng = np.random.default_rng(3)
    same = True
    for n, n_morph in ((1, 6), (7, 6), (2000, 6), (1700, 3)):
        a = CS.assign_morphologies(n, n_morph, np.random.default_rng(int(rng.integers(1 << 30))))
        b = CE.assign_morphologies(n, n_morph, np.random.default_rng(int(rng.integers(1 << 30))))
        same &= a.shape == b.shape and sorted(a.tolist()) == sorted(b.tolist())
    for s in (0, 5, 50013):
        same &= np.array_equal(CS.assign_morphologies(37, 6, np.random.default_rng(s)),
                               CE.assign_morphologies(37, 6, np.random.default_rng(s)))
    check("assign_morphologies == culture_export.assign_morphologies", same)
    for legacy in (False, True):
        N = CFG.n_neurons_effective() if not legacy else 1700
        ok, what = True, ""
        for seed, c in ((0, 0), (50000, 7), (1000, 719), (69000, 47)):
            ref, span, pc = campaign_draws(seed, c, N, legacy)
            midx, pos, th = CS.draws(seed, c, N, len(CFG.morphologies), span, pc)
            ok &= (np.array_equal(midx, ref["midx"]) and np.array_equal(pos, ref["pos"])
                   and np.array_equal(th, ref["theta"]))
            what = "+/-%g um around (%g, %g), N=%d" % (span, pc[0], pc[1], N)
        check("draws == culture_draws, %s square: %s" % ("legacy" if legacy else "current", what), ok)
    axis = geometry()[3]
    check("ORIENT_AXIS_DEG == culture_export.dipole_axis_deg() = %g" % axis, CS.ORIENT_AXIS_DEG == axis)
    th = np.random.default_rng(4).uniform(0.0, 360.0, 20000)
    check("fold_orientation == culture_export.rel_orientation_deg",
          np.array_equal(CS.fold_orientation(th), CE.rel_orientation_deg(th, axis)))

    print("\n[3] the square and the culture size are recovered from a table")
    for legacy, seed, c in ((False, 50000, 3), (True, 1000, 11)):
        N = CFG.n_neurons_effective() if not legacy else 1700
        d, span, pc = campaign_draws(seed, c, N, legacy)
        T = pd.DataFrame({"culture_global": "S%d_C%d" % (seed, c), "seed": seed, "neuron": np.arange(N),
                          "morphology": [CFG.morphologies[i] for i in d["midx"]],
                          "x_um": np.round(d["pos"][:, 0], 2), "y_um": np.round(d["pos"][:, 1], 2),
                          "theta_orient_deg": [round(float(v), 1) for v in d["th_or"]]})
        got = CS.infer_params(T)
        check("infer_params -> %s  (%s square)" % (got, "legacy" if legacy else "current"),
              got == (N, len(CFG.morphologies), span, pc), "expected %s" % ((N, 6, span, pc),))
        th = CS.rotations_from_seed(T, None, N, len(CFG.morphologies), span, pc)
        check("  rotations == culture_draws theta", np.array_equal(th, d["theta"]))
        try:
            CS.rotations_from_seed(T, None, N, len(CFG.morphologies), span, (pc[0] + 1.0, pc[1]))
            check("  a centre 1 um off is refused", False)
        except ValueError:
            check("  a centre 1 um off is refused", True)

    N = CFG.n_neurons_effective()
    cut = None
    for seed in range(50000, 71000, 1000):           # a culture whose first N-10 positions are also those
        for c in range(48):                         # of a culture of N-10: only the rotations differ
            a_ = campaign_draws(seed, c, N - 10)[0]["pos"]
            if np.array_equal(a_, campaign_draws(seed, c, N)[0]["pos"][:N - 10]):
                cut = (seed, c)
                break
        if cut:
            break
    check("found a culture where N-10 neurons reproduce every position (%s)" % (cut,), cut is not None)
    if cut:
        d, span, pc = campaign_draws(cut[0], cut[1], N)
        for keep in (N - 10, 1200):
            T = pd.DataFrame({"culture_global": "S%d_C%d" % cut, "seed": cut[0], "neuron": np.arange(keep),
                              "morphology": [CFG.morphologies[i] for i in d["midx"][:keep]],
                              "x_um": np.round(d["pos"][:keep, 0], 2), "y_um": np.round(d["pos"][:keep, 1], 2),
                              "theta_orient_deg": [round(float(v), 1) for v in d["th_or"][:keep]]})
            got = CS.infer_params(T)
            th = CS.rotations_from_seed(T, None, N, len(CFG.morphologies), span, pc)
            check("  culture cut at %d of %d neurons: infer_params -> %s, rotations == culture_draws"
                  % (keep, N, got), got == (N, 6, span, pc) and np.array_equal(th, d["theta"][:keep]))
            if keep == N - 10:
                try:
                    CS.rotations_from_seed(T, None, N - 10, len(CFG.morphologies), span, pc)
                    check("  N-10 (positions match, rotations do not) is refused", False)
                except ValueError:
                    check("  N-10 (positions match, rotations do not) is refused", True)

    print("\n[4] outcome rules == culture_statistics")
    rng = np.random.default_rng(11)
    dv = rng.normal(0.0, 3.0, 5000)
    dv[:40] = np.nan
    dv[40:60] = np.repeat([1.0, -1.0, 0.999999, -0.999999], 5)
    fired = rng.random(5000) < 0.1
    act, dep, hyp = CS.states(dv, fired.astype(int), 1.0)
    check("activated == spike", np.array_equal(act, fired))
    check("depolarized == polarization_labels(depolarization, 1 mV)",
          np.array_equal(dep, culture_statistics.polarization_labels("depolarization", dv, fired, 1.0) == 1))
    check("hyperpolarized == polarization_labels(hyperpolarization, 1 mV)",
          np.array_equal(hyp, culture_statistics.polarization_labels("hyperpolarization", dv, fired, 1.0) == 1))

    print("\n[5] electrodes")
    elec, _sign = F.default_array(pitch_um=CFG.pitch_um)
    mine = CS.electrodes()
    check("outlines drawn == field.default_array() (config.electrodes %s)"
          % ("set" if getattr(CFG, "electrodes", None) else "None"),
          sorted(map(tuple, np.round(mine, 6))) == sorted(map(tuple, np.round(elec, 6))),
          "culture_states_hpc draws the default 3+3 array at pitch 60 um: update electrodes()")
    check("ELEC_UM == config.electrode_um", CS.ELEC_UM == CFG.electrode_um, (CS.ELEC_UM, CFG.electrode_um))

    print("\n[6] slicing == slicer.py")
    try:
        import morphio  # noqa: F401
        import slicer
    except ImportError as exc:
        slicer = None
        check("morphio + slicer importable (needed for the morphologies)", False, exc)
    archive = write_archive(tmp)
    if slicer is not None:
        ascs = [(m, os.path.join(archive, "specimen_%s" % m, "morphology.asc")) for m in CFG.morphologies]
        try:
            import morphologies
            real = morphologies.all_morphologies()
        except Exception as exc:                                     # noqa: BLE001
            real = []
            print("  NOTE: no real eyal_archive found (%s): synthetic trees only" % type(exc).__name__)
        ok, n_sec = True, 0
        for name, asc in ascs + list(real):
            sc1, S1 = CS._read_tree(asc)
            sc2, S2 = slicer._read_tree(asc)
            ok &= np.array_equal(sc1, sc2) and sorted(S1) == sorted(S2)
            for L in CFG.layers_um:
                ok &= CS.kept_ids(S1, sc1[2], L, CFG.slice_thresh) == slicer.kept_ids(S2, sc2[2], L, CFG.slice_thresh)
                for i in S1:
                    ok &= CS._frac_in_slab(S1[i]["pts"], sc1[2], L) == slicer._frac_in_slab(S2[i]["pts"], sc2[2], L)
                    n_sec += 1
                mine = sorted(map(lambda p: p.round(9).tobytes(), CS.sliced_polylines(asc, L, CFG.slice_thresh)))
                ref = sorted(p.round(9).tobytes() for p, _t in slicer.sliced_polylines(asc, L, CFG.slice_thresh)
                             if len(p) >= 2)
                ok &= mine == ref
        check("kept sections, in-slab fractions, polylines == slicer (%d trees, %d section x layer)"
              % (len(ascs) + len(real), n_sec), ok)
        K = [CS.kept_ids(CS._read_tree(ascs[0][1])[1], CS._read_tree(ascs[0][1])[0][2], L) for L in CFG.layers_um]
        check("synthetic trees are informative (kept sets differ across layers)", K[0] < K[1] < K[2] or K[0] < K[2],
              [len(k) for k in K])
    decoy = os.path.join(archive, "old", "specimen_60308", "morphology.asc")   # sorts before specimen_*
    os.makedirs(os.path.dirname(decoy))
    shutil.copy(os.path.join(archive, "specimen_60308", "morphology.asc"), decoy)
    try:
        import morphologies
        campaign = {m: next(p for name, p in morphologies.all_morphologies(archive) if m in name)
                    for m in CFG.morphologies}
        mine = {m: CS.find_asc(m, archive) for m in CFG.morphologies}
        check("find_asc == the file the campaign uses (morphologies.find_one_morphology rule)",
              mine == campaign, (mine, campaign))
    except ImportError as exc:
        check("morphologies importable", False, exc)
    shutil.rmtree(os.path.join(archive, "old"))
    stub_neuron = "neuron" not in sys.modules       # rich_footprint imports NEURON; this test stays offline
    if stub_neuron:
        sys.modules["neuron"] = types.ModuleType("neuron")
        sys.modules["neuron"].h = None
    try:
        import rich_footprint as RF
        rng = np.random.default_rng(8)
        pts3, sc = rng.normal(0.0, 80.0, (60, 3)), np.array([2.5, -1.5, 3.0])
        RF.segment_coords = lambda cell: (pts3, None)
        RF._soma_center = lambda cell, coords, refs: sc
        ok = True
        for x, y, t in ((0.0, 0.0, 0.0), (10.0, -40.0, 37.5), (-150.25, 120.5, 271.3)):
            placed, _refs = RF._placed(None, (x, y), t)
            ok &= np.allclose(placed[:, :2], CS.rotate_translate(pts3[:, :2] - sc[:2], x, y, t), rtol=0, atol=1e-9)
        check("rotate_translate == rich_footprint._placed (placement + rotation of the simulation)", ok)
    except ImportError as exc:
        check("rich_footprint importable with a stub NEURON", False, exc)
    finally:
        if stub_neuron:
            del sys.modules["neuron"]

    print("\n[7] end to end on the tables of a campaign (current square, two jobs)")
    N = 150
    res = os.path.join(tmp, "results_full_tuned")
    for seed, tag in ((50000, "ftdA"), (51000, "ftdB")):
        write_part(os.path.join(res, "parts_" + tag, "part_000.csv"), [campaign_rows(seed, c, N) for c in (0, 1)])
    merged_dir = os.path.join(tmp, "merged")
    with contextlib.redirect_stdout(io.StringIO()):
        culture_merge.merge(os.path.join(res, "parts_ftd*"), merged_dir, make_figures=False)
        stats = culture_statistics.analyze_one_outcome(merged_dir, "activation", os.path.join(tmp, "stats"))
    tables = (("raw worker part", os.path.join(res, "parts_ftdB", "part_000.csv"), "S51000_C1"),
              ("culture_merge merge of 2 jobs", os.path.join(merged_dir, "culture_Pactivation.csv"), "S51000_C1"),
              ("culture_statistics merged_activation", str(stats["merged"]), "full_tuned_S51000_C1"))
    ref, span, pc = campaign_draws(51000, 1, N)
    for label, path, name in tables:
        names = sorted({k[0] for k in CS.list_cultures(path, 1000)})
        want = sorted(name.replace("S51000_C1", "S%d_C%d" % (s, c)) for s in (50000, 51000) for c in (0, 1))
        if label == "raw worker part":
            want = ["S51000_C0", "S51000_C1"]
        check("%s: culture names %s" % (label, names), names == want, want)
        d = CS._read_filtered(path, 80, name, False, 97).sort_values("neuron").reset_index(drop=True)
        if label.startswith("culture_merge"):
            gid = sorted(int(x) for x in d["culture"].unique())
            lid = sorted(int(x) for x in d["local_culture"].unique())
            check("  (the merge renumbered it: culture = %s, local_culture = %s)" % (gid, lid),
                  gid == [3] and lid == [1])
        got = CS.infer_params(d)
        check("  infer_params -> %s" % (got,), got == (N, 6, span, pc))
        th = CS.rotations_from_seed(d, None, N, 6, span, pc)
        check("  rotations == culture_draws theta", np.allclose(th, ref["theta"][d["neuron"].to_numpy(int)],
                                                                 rtol=0, atol=0))
        out = io.StringIO()
        stem = os.path.join(tmp, "fig", re.sub(r"\W", "_", label))
        with contextlib.redirect_stdout(out):
            files, cnt = CS.main(["--input", path, "--layer", "80", "--culture", name, "--morph-dir", archive,
                                  "--require-all-morph", "--out", stem])
        log = out.getvalue()
        f = d["phase2_outcome"].astype(str).eq("activation").to_numpy()
        v = d["deltaVm_end_phase2_mV"].to_numpy(float)
        want_cnt = (int(f.sum()), int((~f & (v >= 1.0)).sum()), int((~f & (v <= -1.0)).sum()))
        drawn = re.search(r"morphologies drawn for (\[.*?\])", log)
        check("  figure: all 6 morphologies, centre (%g, %g), counts %s" % (pc[0], pc[1], cnt),
              drawn is not None and len(ast.literal_eval(drawn.group(1))) == 6 and "NOTE" not in log
              and ("centre (%g, %g) um" % pc) in log and cnt == want_cnt
              and os.path.getsize(files[0]) > 30000 and os.path.getsize(files[1]) > 5000, log)

    running = os.path.join(tmp, "running_part.csv")
    shutil.copy(os.path.join(res, "parts_ftdB", "part_000.csv"), running)
    with open(running, "a") as fh:                  # a row cut while the worker writes it: no newline
        fh.write(",".join(str(v) for v in campaign_rows(51000, 2, 1)[0])[:60])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        names = sorted({k[0] for k in CS.list_cultures(running, 1000)})
        files, cnt = CS.main(["--input", running, "--layer", "80", "--culture", "S51000_C1", "--morph-dir", archive,
                              "--require-all-morph", "--out", os.path.join(tmp, "fig", "running")])
    log = out.getvalue()
    check("raw part with a half-written last row: row dropped with a NOTE, figure with all morphologies",
          names == ["S51000_C0", "S51000_C1"] and "last row is incomplete" in log
          and "morphologies drawn for" in log and "not drawn" not in log, log)

    print("\n[8] jobs/culture_states.pbs")
    pbs = os.path.join(HERE, "jobs", "culture_states.pbs")
    raw = open(pbs, "rb").read()
    check("ASCII, LF only", all(b < 128 for b in raw) and b"\r" not in raw)
    r = subprocess.run(["bash", "-n", pbs], capture_output=True, text=True)
    check("bash -n", r.returncode == 0, r.stderr)
    howto = open(os.path.join(HERE, "HOWTO_culture_states.md"), encoding="utf-8").read()
    check("the job and the HOWTO quote the self-test result '%s'" % last,
          last in raw.decode("ascii") and last in howto)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if FAILURES:
    print("\nFAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("\nAll smoke tests passed.")
