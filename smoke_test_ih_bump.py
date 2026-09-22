"""smoke_test_ih_bump.py -- NEURON-backed test of the long-window Ih bump measurement.

Needs the compiled mechanisms (nrnivmodl rich_mech) and eyal_archive/. Run from the repo root,
on a compute node (not the login node):

    python smoke_test_ih_bump.py            # expect the last line: 'All smoke tests passed.'

  [1] bump_ms = 0 leaves the short window untouched (same spikes, same DeltaV at pulse end)
  [2] the bump trace is on the requested fixed grid and starts at the end of phase 2
  [3] CVODE is not left active for the next caller
  [4] a real bump is found and fitted on the tuned full-channel model
  [5] ATTRIBUTION: removing Ih abolishes it; removing SK_E2 (the other slow candidate) does not
  [6] cost: the long window is a small multiple of the short one, not the ~57x of fixed dt
"""
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron import h

from bump_kinetics import fit_ih_bump
from config import CFG
from morphologies import find_one_morphology
from rich_cell import all_sections, build_rich_cell, settled_resting_voltage, tune_leak_isopotential
from rich_footprint import spikes_at
from slicer import reduced_asc

FAILURES = []
BUMP_MS = 800.0
BUMP_DT = 0.5
PROBE = (150.0, -30.0)          # a placement that neither fires nor sits in the far field


def check(name, cond, detail=""):
    print("  %-68s %s" % (name, "PASS" if cond else "FAIL " + str(detail)), flush=True)
    if not cond:
        FAILURES.append(name)


def tuned_cell(asc, kill=None):
    """Rich's own model (passive axon), leak-tuned to its own settled rest. `kill` zeroes one
    conductance before tuning, so the control is tuned to ITS own resting state, not the intact one."""
    cell = build_rich_cell(asc, axon_active=False)
    if kill is not None:
        mech, attr = kill
        for s in all_sections(cell):
            if h.ismembrane(mech, sec=s):
                for seg in s:
                    setattr(seg, attr, 0.0)
    vt = settled_resting_voltage(cell, tstop_ms=3000.0, dt_ms=CFG.dt_ms, v_init_mV=-75.0)
    tune_leak_isopotential(cell, vt)
    return cell, vt


MORPH = str(CFG.morphologies[0])
LAYER = float(CFG.layers_um[-1])
print("[smoke_test_ih_bump] morphology %s, layer %d um, bump window %g ms"
      % (MORPH, int(LAYER), BUMP_MS))
asc = reduced_asc(find_one_morphology(MORPH), LAYER, out_path="_sib_%d.asc" % os.getpid())

try:
    cell, v_target = tuned_cell(asc)
    print("  tuned to its own settled rest: %.4f mV" % v_target)

    print("\n[1]-[3] the long window does not disturb the short one")
    short = spikes_at(cell, PROBE, 0.0, i0_uA=CFG.i0_uA, detail=True, pre_end_ms=2.0,
                      v_init_mV=v_target)
    long_ = spikes_at(cell, PROBE, 0.0, i0_uA=CFG.i0_uA, detail=True, pre_end_ms=2.0,
                      v_init_mV=v_target, bump_ms=BUMP_MS, bump_dt_ms=BUMP_DT)
    check("bump mode returns 6 values (n, Vmax, tw, vw, tb, vb)", len(long_) == 6, len(long_))
    check("same spike count with and without the bump window (%d)" % short[0],
          short[0] == long_[0], (short[0], long_[0]))
    i0s = int(np.argmin(np.abs(short[2])))
    i0l = int(np.argmin(np.abs(long_[2])))
    d_end = abs(float(short[3][i0s]) - float(long_[3][i0l]))
    check("DeltaV at the end of phase 2 unchanged (%.2e mV, the pulse is still fixed-dt)" % d_end,
          d_end < 1e-6, d_end)
    tb, vb = long_[4], long_[5]
    step = np.diff(tb)
    check("bump trace on the requested %.1f ms grid (%d samples, step %.4f .. %.4f)"
          % (BUMP_DT, tb.size, step.min(), step.max()),
          tb.size > 100 and abs(step.max() - BUMP_DT) < 1e-6 and abs(step.min() - BUMP_DT) < 1e-6)
    check("bump trace reaches the requested window (%.1f ms, t=0 is the pulse end)" % tb[-1],
          abs(tb[-1] - BUMP_MS) < 2 * BUMP_DT and tb[0] < 0.0, (float(tb[0]), float(tb[-1])))
    check("CVODE switched back off for the next caller", int(h.cvode_active()) == 0)

    print("\n[4] a real bump, measured against the no-stimulus sham")
    sham = spikes_at(cell, PROBE, 0.0, i0_uA=0.0, detail=True, pre_end_ms=2.0,
                     v_init_mV=v_target, bump_ms=BUMP_MS, bump_dt_ms=BUMP_DT)
    check("the sham is flat -- the tuned cell does not drift (max |dV| %.2e mV)"
          % np.abs(sham[5] - v_target).max(), np.abs(sham[5] - v_target).max() < 0.01)
    dv = vb - sham[5]
    fit = fit_ih_bump(tb, dv, t_skip_ms=7.0, t_search_ms=BUMP_MS)
    print("      peak %+.4f mV at %.1f ms | tau_rise %.1f ms | tau_decay %.1f ms | r2 %.4f/%.4f"
          % (fit["peak_mV"], fit["time_to_peak_ms"], fit["tau_rise_ms"], fit["tau_decay_ms"],
             fit["r2_rise"], fit["r2_decay"]))
    check("a bump of at least 0.1 mV is present", abs(fit["peak_mV"]) > 0.1, fit["peak_mV"])
    check("it peaks well after the direct transient (>30 ms), i.e. it is the slow one",
          fit["time_to_peak_ms"] > 30.0, fit["time_to_peak_ms"])
    check("both taus fitted and the fit passes its gates", fit["fit_ok"] == 1,
          {k: fit[k] for k in ("tau_rise_ms", "tau_decay_ms", "r2_rise", "r2_decay")})
    check("the window is long enough: tau_decay is well inside it (%.1f vs %.0f ms)"
          % (fit["tau_decay_ms"], BUMP_MS), fit["tau_decay_ms"] < 0.5 * BUMP_MS)
    del cell
    gc.collect()

    print("\n[5] attribution: the bump is Ih")
    peaks = {}
    for label, kill in (("intact", None), ("Ih removed", ("Ih", "gIhbar_Ih")),
                        ("SK_E2 removed", ("SK_E2", "gSK_E2bar_SK_E2"))):
        c, vt = tuned_cell(asc, kill)
        st = spikes_at(c, PROBE, 0.0, i0_uA=CFG.i0_uA, detail=True, v_init_mV=vt,
                       bump_ms=BUMP_MS, bump_dt_ms=BUMP_DT)
        sh = spikes_at(c, PROBE, 0.0, i0_uA=0.0, detail=True, v_init_mV=vt,
                       bump_ms=BUMP_MS, bump_dt_ms=BUMP_DT)
        peaks[label] = fit_ih_bump(st[4], st[5] - sh[5], t_skip_ms=7.0, t_search_ms=BUMP_MS)["peak_mV"]
        print("      %-14s peak %+8.4f mV" % (label, peaks[label]))
        del c
        gc.collect()
    check("removing Ih abolishes the bump (%.4f -> %.4f mV)" % (peaks["intact"], peaks["Ih removed"]),
          abs(peaks["Ih removed"]) < 0.2 * abs(peaks["intact"]))
    check("removing SK_E2 does not (%.4f mV) -- so it is not the Ca-activated K current"
          % peaks["SK_E2 removed"],
          abs(peaks["SK_E2 removed"] - peaks["intact"]) < 0.1 * abs(peaks["intact"]))

    print("\n[6] cost of the long window")
    cell, v_target = tuned_cell(asc)
    def timeit(**kw):
        ts = []
        for _ in range(3):
            t0 = time.time()
            spikes_at(cell, PROBE, 0.0, i0_uA=CFG.i0_uA, detail=True, v_init_mV=v_target, **kw)
            ts.append(time.time() - t0)
        return min(ts)
    t_short = timeit()
    t_long = timeit(bump_ms=BUMP_MS, bump_dt_ms=BUMP_DT)
    print("      short %.3f s | %g ms bump %.3f s | ratio %.2fx" % (t_short, BUMP_MS, t_long,
                                                                    t_long / t_short))
    check("the %g ms window costs < 3x the short one (fixed dt would be ~57x)" % BUMP_MS,
          t_long / t_short < 3.0, t_long / t_short)
finally:
    for f in (asc,):
        if f and os.path.exists(f):
            os.remove(f)

print()
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("All smoke tests passed.")
