"""smoke_test_leak_tuning.py -- NEURON-backed test of the isopotential leak tuning.

Needs the compiled mechanisms (nrnivmodl rich_mech) and eyal_archive/. Run from the repo root:

    python smoke_test_leak_tuning.py        # expect the last line: 'All smoke tests passed.'

  [1] axon_active=False gives Rich's own model: no na/kv anywhere in the axon
  [2] the free (untuned) arbour is NOT isopotential -- the problem this fixes
  [3] after tuning, every segment holds v_target with no stimulus, and nothing fires by itself
  [4] the per-segment net membrane current at v_target is driven to ~0
  [5] tuning changes e_pas only: g_pas, Ra and cm are untouched (cable properties preserved)
  [6] excitability survives: the same placements spike before and after tuning
  [7] idempotent: tuning an already-tuned cell is a no-op
  [8] the tuning is temperature-DEPENDENT via the Nernst eca, so it must run at 37 C
  [9] an unknown mechanism is refused rather than silently skipped
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron import h

from config import CFG
import field as F
from active_cell import segment_coords, _soma_center
from morphologies import find_one_morphology
from rich_cell import (_segment_ionic_current, all_sections, build_rich_cell, domain_of,
                       settled_resting_voltage, tune_leak_isopotential)
from slicer import reduced_asc

FAILURES = []


def check(name, cond, detail=""):
    print("  %-68s %s" % (name, "PASS" if cond else "FAIL " + str(detail)), flush=True)
    if not cond:
        FAILURES.append(name)


def spikes_at_placement(cell, pos_xy, theta_deg, v_init, i0_uA=None, post_ms=6.0, dt=0.025):
    """One biphasic pulse at a placement; returns (n_spikes, Vmax). Local copy so the test does
    not depend on rich_footprint's signature."""
    i0 = CFG.i0_uA if i0_uA is None else float(i0_uA)
    coords, refs = segment_coords(cell)
    sc = _soma_center(cell, coords, refs)
    p = coords + (np.array([pos_xy[0], pos_xy[1], CFG.h_soma_um]) - sc)
    if theta_deg:
        r = np.radians(theta_deg)
        co, si = np.cos(r), np.sin(r)
        ox, oy = pos_xy
        p = np.column_stack([ox + co * (p[:, 0] - ox) - si * (p[:, 1] - oy),
                             oy + si * (p[:, 0] - ox) + co * (p[:, 1] - oy), p[:, 2]])
    elec, sign = F.default_array(monopolar=not CFG.bipolar)
    g = F.geom_factor(p, elec, sign, sigma_Sm=CFG.sigma_Sm, rmin_um=CFG.rmin_um)
    t = np.arange(0.0, 5.0 + 2 * CFG.phase_dur_ms + post_ms + dt, dt)
    cur = F.biphasic_current(t - 5.0, i0, CFG.phase_dur_ms, True, CFG.ramp_us, CFG.interphase_us)
    tvec = h.Vector(t)
    keep = [tvec]
    for sec in set(x.sec for x in refs):
        if not h.ismembrane("extracellular", sec=sec):
            sec.insert("extracellular")
    for k, seg in enumerate(refs):
        vv = h.Vector(g[k] * cur)
        vv.play(seg._ref_e_extracellular, tvec, True)
        keep.append(vv)
    vs = h.Vector().record(cell.soma[0](0.5)._ref_v)
    h.celsius = 37
    h.dt = dt
    h.finitialize(float(v_init))
    h.continuerun(t[-1])
    v = np.asarray(vs)
    return int(np.sum((v[:-1] < 0) & (v[1:] >= 0))), float(v.max())


MORPH = str(CFG.morphologies[0])
LAYER = float(CFG.layers_um[-1])
PLACEMENTS = ((30.0, -30.0, 0.0), (30.0, -30.0, 180.0), (60.0, -30.0, 270.0),
              (90.0, -30.0, 0.0), (120.0, -30.0, 90.0))
print("[smoke_test_leak_tuning] morphology %s, layer %d um" % (MORPH, int(LAYER)))

asc = reduced_asc(find_one_morphology(MORPH), LAYER, out_path="_slt_%d.asc" % os.getpid())
try:
    cell = build_rich_cell(asc, axon_active=False)

    print("\n[1] Rich's own model: passive axon")
    ax = [s for s in all_sections(cell) if domain_of(s.name()) == "axon"]
    act = [s.name() for s in ax
           if h.ismembrane("na", sec=s) or h.ismembrane("kv", sec=s)]
    check("axon_active=False -> no na/kv in any of the %d axon sections" % len(ax),
          len(ax) > 0 and not act, act[:3])
    check("the soma still carries the Rich active set",
          all(h.ismembrane(m, sec=cell.soma[0]) for m in ("NaTa_t", "SKv3_1", "Ih", "K_Pst")))

    print("\n[2] the untuned arbour is not isopotential")
    v_target = settled_resting_voltage(cell, tstop_ms=3000.0, dt_ms=CFG.dt_ms, v_init_mV=-75.0)
    h.finitialize(v_target)
    h.continuerun(3000.0)
    free = [seg.v for s in all_sections(cell) for seg in s]
    spread_before = max(free) - min(free)
    check("free settled somatic rest = %.4f mV" % v_target, np.isfinite(v_target))
    check("untuned spread is several mV (got %.3f mV over %.3f .. %.3f)"
          % (spread_before, min(free), max(free)), spread_before > 1.0)

    print("\n[3]-[5] tuning")
    g_before = [seg.g_pas for s in all_sections(cell) for seg in s]
    geom_before = [(s.Ra, s.cm, s.nseg) for s in all_sections(cell)]
    rep = tune_leak_isopotential(cell, v_target)
    g_after = [seg.g_pas for s in all_sections(cell) for seg in s]
    geom_after = [(s.Ra, s.cm, s.nseg) for s in all_sections(cell)]
    check("net |current| per segment driven to ~0 (%.2e -> %.2e mA/cm2)"
          % (rep["max_abs_net_before"], rep["max_abs_net_after"]),
          rep["max_abs_net_after"] < 1e-12, rep["max_abs_net_after"])
    check("g_pas untouched in all %d segments" % len(g_before), g_before == g_after)
    check("Ra / cm / nseg untouched in all %d sections" % len(geom_before),
          geom_before == geom_after)
    vs = h.Vector().record(cell.soma[0](0.5)._ref_v)   # record BEFORE finitialize
    h.finitialize(v_target)
    h.continuerun(2000.0)
    v = np.asarray(vs)
    tuned = [seg.v for s in all_sections(cell) for seg in s]
    spread_after = max(tuned) - min(tuned)
    check("after 2000 ms with no stimulus the whole arbour holds v_target "
          "(spread %.5f mV, was %.3f)" % (spread_after, spread_before), spread_after < 0.05,
          spread_after)
    check("somatic drift over 2000 ms < 0.01 mV (got %+.5f)" % (v[-1] - v[0]),
          abs(v[-1] - v[0]) < 0.01)
    check("no spontaneous spikes in 2000 ms",
          int(np.sum((v[:-1] < 0) & (v[1:] >= 0))) == 0)
    print("      e_pas %.2f .. %.2f mV (default -84.395); per domain:"
          % (rep["e_pas_min"], rep["e_pas_max"]))
    for d, (lo, med, hi, n) in sorted(rep["per_domain"].items()):
        print("        %-5s n=%4d  %8.2f / %8.2f / %8.2f mV (min/median/max)" % (d, n, lo, med, hi))

    print("\n[6] excitability survives the tuning")
    cell_u = build_rich_cell(asc, axon_active=False)
    vt_u = settled_resting_voltage(cell_u, tstop_ms=3000.0, dt_ms=CFG.dt_ms, v_init_mV=-75.0)
    before = [spikes_at_placement(cell_u, (x, y), th, vt_u) for x, y, th in PLACEMENTS]
    del cell_u
    after = [spikes_at_placement(cell, (x, y), th, v_target) for x, y, th in PLACEMENTS]
    check("at least one placement spikes (the model is excitable at %.0f uA)" % CFG.i0_uA,
          sum(b[0] for b in before) > 0, [b[0] for b in before])
    check("same spike counts before and after tuning: %s" % [b[0] for b in before],
          [b[0] for b in before] == [a[0] for a in after], [a[0] for a in after])
    dv = max(abs(b[1] - a[1]) for b, a in zip(before, after))
    check("peak Vm shifts by < 1 mV at every placement (max %.4f mV)" % dv, dv < 1.0)

    print("\n[7]-[8] idempotent and temperature-independent")
    e1 = [seg.e_pas for s in all_sections(cell) for seg in s]
    tune_leak_isopotential(cell, v_target)
    e2 = [seg.e_pas for s in all_sections(cell) for seg in s]
    check("re-tuning an already-tuned cell is a no-op",
          max(abs(a - b) for a, b in zip(e1, e2)) < 1e-9)
    # The tuning is NOT temperature-free: NEURON computes eca from cai/cao by Nernst, so the Ca
    # reversal moves with temperature and every Ca-bearing segment balances at a different e_pas.
    # It must therefore be done at the simulation temperature -- this guards the h.celsius line.
    cold = build_rich_cell(asc, axon_active=False)
    rep_cold = tune_leak_isopotential(cold, v_target, celsius=6.3)
    e_cold = [seg.e_pas for s in all_sections(cold) for seg in s]
    d_cold = max(abs(a - b) for a, b in zip(e1, e_cold))
    n_moved = sum(1 for a, b in zip(e1, e_cold) if abs(a - b) > 1e-6)
    check("tuning at the wrong temperature shifts e_pas in the Ca-bearing segments only "
          "(%d of %d, max %.4f mV)" % (n_moved, len(e1), d_cold),
          0 < n_moved < len(e1) and d_cold > 1e-3, (n_moved, d_cold))
    h.celsius = 37.0
    h.finitialize(v_target)
    h.fcurrent()
    resid = max(abs(seg.i_pas + _segment_ionic_current(seg, h.ismembrane("Ih", sec=s)))
                for s in all_sections(cold) for seg in s)
    check("a cell tuned at 6.3 C leaves a standing current at 37 C (%.2e mA/cm2)" % resid,
          resid > 1e-9, resid)
    check("tuning at 37 C leaves none (%.2e mA/cm2)" % rep["max_abs_net_after"],
          rep["max_abs_net_after"] < 1e-12)
    del cold, rep_cold

    print("\n[9] an unknown mechanism is refused")
    odd = build_rich_cell(asc, axon_active=False)
    odd.soma[0].insert("hh")                       # not in KNOWN_MECHS
    try:
        tune_leak_isopotential(odd, v_target)
        check("unknown mechanism raises rather than being skipped", False)
    except RuntimeError as e:
        check("unknown mechanism raises rather than being skipped", "hh" in str(e))
    del odd
finally:
    if os.path.exists(asc):
        os.remove(asc)

print()
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("All smoke tests passed.")
