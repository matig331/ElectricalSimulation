"""smoke_test_soma_only.py -- NEURON-backed, END-TO-END test of the soma-only culture export.

Needs the compiled mechanisms (nrnivmodl rich_mech) and eyal_archive/. Run it from the repo
root, on the cluster in an INTERACTIVE session (not the login node) or on any machine with NEURON:

    python smoke_test_soma_only.py        # expect the last line: 'All smoke tests passed.'

Uses ONE morphology x ONE layer (config.morphologies[0], config.layers_um[0]) and ~50 short
simulations, so it takes minutes, not hours. Every check prints PASS/FAIL:

  [1] biophysics   soma_only: active channels ONLY on the soma (dendrites + axon/AIS = pas);
                   full_active contrast: the AIS carries the Eyal na/kv
  [2] rest         settled rest converged (1500 vs 3000 ms); fast_imem left OFF
  [3] sham         I = 0 -> DeltaV_end == 0 exactly -> 'neutral'; the init drift is reported
  [4] far field    sham-referenced DeltaV is ODD in I and LINEAR in |I| (stimulus-only)
  [5] near field   at least one near-electrode placement fires (soma-only is excitable)
  [6] run length   detail=True does not lengthen the run (post = post_ms, not 1000 ms)
  [7] worker       run_worker: CSV_HEADER rows, model + seed recorded, exclusive outcomes,
                   placements == culture_draws, bit-identical on re-run
  [8] serial==par  culture_export() (serial) writes the same rows as the worker (same seed)
  [9] merge+stats  culture_merge -> 3 files -> culture_statistics: P == raw counts

The measured single-process cost per simulation is printed: use it (x the slowdown seen under
full-node concurrency) to size WALLTIME. Nothing is written outside a temporary directory.
"""
import csv
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURES = []


def check(name, cond, detail=""):
    print("  %-74s %s" % (name, "PASS" if cond else "FAIL " + str(detail)), flush=True)
    if not cond:
        FAILURES.append(name)


from neuron import h
import neuron
from config import CFG
import culture_export as CE
import field as F
from rich_cell import settled_resting_voltage
from rich_footprint import spikes_at
from morphologies import find_one_morphology
from slicer import reduced_asc

MORPH = str(CFG.morphologies[0])
LAYER = float(CFG.layers_um[0])
I0 = float(CFG.i0_uA)
ACTIVE = ("NaTa_t", "Nap_Et2", "K_Pst", "K_Tst", "SKv3_1", "SK_E2", "Ca_LVAst", "Ca_HVA",
          "Ih", "Im", "CaDynamics_E2", "na", "kv")
T_START = time.time()
print("[smoke_test_soma_only] morphology %s, layer %d um, i0 %.0f uA, NEURON %s"
      % (MORPH, int(LAYER), I0, neuron.__version__), flush=True)


def sections(cell):
    """Every section of the cell, the stylized axon included, de-duplicated by name."""
    out = {}
    for s in list(cell.all) + list(getattr(cell, "axon", [])):
        out[s.name()] = s
    return list(out.values())


def mechs(sec):
    return {m for m in ACTIVE if h.ismembrane(m, sec=sec)}


# ------------------------------------------------------------------ [1] biophysics
print("\n[1] biophysics of the two cell models")
asc = reduced_asc(find_one_morphology(MORPH), LAYER, out_path="_sst_%d.asc" % os.getpid())
try:
    so = CE.build_cell_for_model(asc, "soma_only")
    fa = CE.build_cell_for_model(asc, "full_active")
finally:
    if os.path.exists(asc):
        os.remove(asc)
so_soma, so_other, n_other = set(), set(), 0
for s in sections(so):
    if "soma" in s.name():
        so_soma |= mechs(s)
    else:
        so_other |= mechs(s)
        n_other += 1
check("soma_only: soma carries the Rich set (NaTa_t, SKv3_1, Ih, ...)",
      {"NaTa_t", "SKv3_1", "Ih", "K_Pst"} <= so_soma, sorted(so_soma))
check("soma_only: %d non-soma sections carry NO active channel" % n_other,
      not so_other and n_other > 0, sorted(so_other))
check("soma_only: every section has the leak (pas)",
      all(h.ismembrane("pas", sec=s) for s in sections(so)))
fa_axon = set().union(*[mechs(s) for s in sections(fa) if "axon" in s.name()] or [set()])
check("full_active contrast: the AIS carries Eyal na/kv", {"na", "kv"} <= fa_axon, sorted(fa_axon))
del so, fa

# ------------------------------------------------------------------ [2] rest
print("\n[2] settled rest")
t0 = time.time()
prep = CE.prepare_cell(MORPH, LAYER, CFG, "soma_only", tag="_sst_")
V_REST_PREP = prep["v_rest"]
t_prep = time.time() - t0
v3000 = settled_resting_voltage(prep["cell"], tstop_ms=3000.0, dt_ms=CFG.dt_ms)
check("rest converged: |V(1500 ms) - V(3000 ms)| < 0.01 mV (%.4f vs %.4f)"
      % (prep["v_rest"], v3000), abs(prep["v_rest"] - v3000) < 0.01)
try:
    fim = int(h.cvode.use_fast_imem())
except Exception:
    fim = -1
check("fast_imem left OFF after prepare_cell (it slows every sim)", fim == 0, fim)
print("  prepare_cell took %.1f s (build + 1500 ms rest + sham)" % t_prep)

# ------------------------------------------------------------------ [2b] morphology distinctness
print("\n[2b] a second morphology is a genuinely different cell (catches a stale/duplicate archive)")
if len(CFG.morphologies) >= 2:
    morph2 = str(CFG.morphologies[1])
    prep2 = CE.prepare_cell(morph2, LAYER, CFG, "soma_only", tag="_sst2_")
    check("%r and %r give DIFFERENT resting potentials (%.4f vs %.4f mV) -- if these ever match "
          "to several decimals, treat it as a red flag: check that the archive's specimen "
          "folders hold genuinely distinct .asc files, not copies of one another"
          % (MORPH, morph2, prep["v_rest"], prep2["v_rest"]),
          abs(prep2["v_rest"] - prep["v_rest"]) > 1e-4, (prep["v_rest"], prep2["v_rest"]))
    del prep2
    import gc; gc.collect()
else:
    print("  (skipped: config.morphologies has only 1 entry)")

# ------------------------------------------------------------------ [3] sham
print("\n[3] sham reference")
z = CE.simulate_neuron(prep, (400.0, 0.0), 0.0, 0.0)
check("I = 0 -> DeltaV_end == 0 exactly", z["dv_end"] == 0.0, z["dv_end"])
check("I = 0 -> outcome 'neutral', no spike", z["outcome"] == "neutral" and z["fired"] == 0)
check("init drift finite and reported: %+.6f mV (removed by the sham)" % prep["ctrl_drift"],
      np.isfinite(prep["ctrl_drift"]))

# ------------------------------------------------------------------ [4] far field
print("\n[4] far field: the response must be stimulus-only (odd + linear)")
worst_odd = worst_lin = 0.0
n_tested = n_flip = 0
t0, n_sim = time.time(), 0
for xy in ((450.0, 100.0), (-400.0, -200.0), (0.0, 480.0)):
    for th in (0.0, 135.0):
        a = CE.simulate_neuron(prep, xy, th, I0)["dv_end"]
        b = CE.simulate_neuron(prep, xy, th, -I0)["dv_end"]
        c = CE.simulate_neuron(prep, xy, th, 2.0 * I0)["dv_end"]
        n_sim += 3
        if abs(a) < 1e-5:
            continue
        n_tested += 1
        worst_odd = max(worst_odd, abs(a + b) / abs(a))
        worst_lin = max(worst_lin, abs(c / a - 2.0))
        if np.sign(a) != np.sign(a + prep["ctrl_drift"]):
            n_flip += 1
s_per_sim = (time.time() - t0) / max(n_sim, 1)
check("at least 3 far-field responses large enough to test (got %d)" % n_tested, n_tested >= 3)
check("ODD in I: max |dV(+I) + dV(-I)| / |dV| < 1e-2 (got %.2e)" % worst_odd, worst_odd < 1e-2)
check("LINEAR: max |dV(2I)/dV(I) - 2| < 2e-2 (got %.2e)" % worst_lin, worst_lin < 2e-2)
print("  info: %d of %d tested placements would get the OPPOSITE label under the old "
      "scalar-rest reference" % (n_flip, n_tested))
print("  measured cost: %.2f s per simulation (single process)" % s_per_sim)

# ------------------------------------------------------------------ [5] near field
print("\n[5] near field")
fired = 0
for xy in ((30.0, -30.0), (-30.0, -30.0), (30.0, 30.0), (-30.0, 30.0), (30.0, -90.0),
           (-30.0, -90.0)):
    for th in (0.0, 90.0, 180.0, 270.0):
        fired += CE.simulate_neuron(prep, xy, th, I0)["fired"]
        if fired:
            break
    if fired:
        break
check("a soma on an electrode fires at %.0f uA (soma-only is excitable)" % I0, fired >= 1,
      "no placement on the electrodes fired: P(activation) would be ~0 everywhere")

# ------------------------------------------------------------------ [6] run length
print("\n[6] run length")
_n, _vm, tw, _vw = spikes_at(prep["cell"], (400.0, 0.0), 0.0, i0_uA=0.0, detail=True,
                            pre_end_ms=0.0, v_init_mV=prep["v_rest"])
check("detail=True ends post_ms (6 ms) after the pulse, not 1000 ms (got %.3f)" % tw[-1],
      abs(tw[-1] - 6.0) < 0.03)

# ------------------------------------------------------------------ [6b] CellPool
print("\n[6b] CellPool (the drivers' one-live-cell cache)")
del prep
import gc; gc.collect()
L2 = float(CFG.layers_um[-1])


def _n_sections_for(morph, layer):
    """Ground truth: build (morph, layer) OUTSIDE the pool, count its sections, destroy it.
    Different layers legitimately keep different amounts of dendrite on a real morphology (a
    thicker slab retains more), so their section counts need NOT match each other -- comparing
    two different layers directly is the wrong check. Each key gets its own reference instead."""
    asc = reduced_asc(find_one_morphology(morph), layer,
                      out_path="_sst_ref_%d.asc" % os.getpid())
    try:
        c = CE.build_cell_for_model(asc, "soma_only")
        n = sum(1 for _ in h.allsec())
    finally:
        if os.path.exists(asc):
            os.remove(asc)
    del c
    gc.collect()
    return n


n_ref_L1 = _n_sections_for(MORPH, LAYER)
n_ref_L2 = _n_sections_for(MORPH, L2)
with CE.CellPool(CFG, "soma_only") as pool:
    a1 = pool.simulate(MORPH, LAYER, (450.0, 100.0), 135.0, I0)
    n1 = sum(1 for _ in h.allsec())
    check("pool section count for L1 matches an independent build (%d)" % n1, n1 == n_ref_L1,
          (n1, n_ref_L1))
    b1 = pool.simulate(MORPH, L2, (450.0, 100.0), 135.0, I0)
    n2 = sum(1 for _ in h.allsec())
    check("switching to L2: old cell's sections gone, only L2's remain (%d vs ref %d)" % (n2, n_ref_L2),
          n2 == n_ref_L2, (n2, n_ref_L2))
    a2 = pool.simulate(MORPH, LAYER, (450.0, 100.0), 135.0, I0)
    n3 = sum(1 for _ in h.allsec())
    check("switching back to L1: section count returns to the L1 reference (%d)" % n3,
          n3 == n_ref_L1, (n3, n_ref_L1))
    check("exactly 3 builds (L1, L2, L1 again -- no cache across the switch)", pool.n_builds == 3,
          pool.n_builds)
    info = pool.info(MORPH, LAYER)
    check("rebuilt cell reproduces the result bit for bit (dV %+.6f)" % a1["dv_end"],
          a1 == a2, (a1, a2))
    check("pool rest == prepare_cell rest for the same morphology x layer",
          abs(info["v_rest"] - V_REST_PREP) < 1e-9, (info["v_rest"], V_REST_PREP))
gc.collect()
check("closing the pool destroys its cell (no section left)", sum(1 for _ in h.allsec()) == 0,
      sum(1 for _ in h.allsec()))

# ------------------------------------------------------------------ [7]-[9] end to end
tmp = tempfile.mkdtemp(prefix="sst_")
try:
    CFG.morphologies = [MORPH]                      # in-process only: 1 cell to prepare
    CFG.layers_um = (LAYER,)
    import culture_worker as CW
    print("\n[7] worker (2 cultures x 3 neurons x 1 layer, seed 1000)")
    p1 = CW.run_worker([0, 1], os.path.join(tmp, "a", "part_000.csv"), neurons_per_culture=3,
                       seed=1000, quiet=True)
    p2 = CW.run_worker([0, 1], os.path.join(tmp, "b", "part_000.csv"), neurons_per_culture=3,
                       seed=1000, quiet=True)
    with open(p1, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rows = [r for r in rd if r]
    D = [dict(zip(header, r)) for r in rows]
    check("header == culture_export.CSV_HEADER", header == CE.CSV_HEADER)
    check("6 rows (2 cultures x 3 neurons x 1 layer)", len(rows) == 6, len(rows))
    check("cell_model = soma_only and seed = 1000 in every row",
          all(d["cell_model"] == "soma_only" and d["seed"] == "1000" for d in D))
    lab = {"activation": (1, 0, 0), "depol": (0, 1, 0), "hyperpol": (0, 0, 1), "neutral": (0, 0, 0)}
    check("outcomes exclusive and consistent with phase2_outcome",
          all(lab[d["phase2_outcome"]] == (int(d["fired"]), int(d["depolarized"]),
                                           int(d["hyperpolarized"])) for d in D))
    check("v_rest_mV == the rest of this morphology x layer (%.4f)" % V_REST_PREP,
          all(abs(float(d["v_rest_mV"]) - V_REST_PREP) < 1e-3 for d in D))
    elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
    dc, dd = CE.dipole_frame(elec, sign)
    ok = True
    for c in (0, 1):
        dr = CE.culture_draws(1000, c, 3, 1, CFG.span_half_um(), elec, CE.electrode_center(elec),
                              dc, dd, CE.dipole_axis_deg(elec, sign), CFG.h_soma_um)
        for d in D:
            if int(d["culture"]) == c:
                i = int(d["neuron"])
                ok &= float(d["x_um"]) == round(float(dr["pos"][i, 0]), 2)
                ok &= float(d["y_um"]) == round(float(dr["pos"][i, 1]), 2)
    check("placements == culture_draws(seed + culture) (the RNG contract)", bool(ok))
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        check("re-running the same seed gives a bit-identical part file", f1.read() == f2.read())

    print("\n[8] serial path == parallel path")
    sp = CE.culture_export(n_cultures=2, neurons_per_culture=3, layers=(LAYER,), seed=1000,
                           csv_path=os.path.join(tmp, "serial", "culture_Pactivation.csv"),
                           make_figures=False)
    same = True
    for path, col in zip(sp, CE.OUTCOME_COLS):
        with open(path, newline="") as fh:
            S = list(csv.DictReader(fh))
        same &= len(S) == len(D)
        for s, d in zip(S, D):
            same &= all(s[k] == d[k] for k in s)
    check("culture_export() writes exactly the worker's rows, all three files", bool(same))

    print("\n[9] merge -> statistics")
    import culture_merge
    paths, _ = culture_merge.merge(os.path.join(tmp, "a"), os.path.join(tmp, "merged"),
                                   make_figures=False, expect_model="soma_only")
    check("merge wrote the three deliverable files",
          sorted(paths) == sorted(CE.OUTCOME_FILES) and all(os.path.exists(p) for p in paths.values()))
    try:
        import culture_statistics as CS
    except ImportError as e:
        check("culture_statistics importable (needs pandas)", False, e)
    else:
        res = CS.analyze_cultures(os.path.join(tmp, "merged"), os.path.join(tmp, "stats"),
                                  outcome="all", distance_bin_um=50.0)
        good = True
        for name, (_stem, col) in CE.OUTCOME_FILES.items():
            import pandas as pd
            m = pd.read_csv(res[name]["merged"])
            good &= int(m["fired"].sum()) == sum(int(d[col]) for d in D) and len(m) == len(D)
            good &= os.path.getsize(res[name]["pdf"]) > 1000
        check("statistics: N and positives per outcome == the raw rows; PDFs written", bool(good))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\nelapsed %.0f s" % (time.time() - T_START))
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("All smoke tests passed.")
