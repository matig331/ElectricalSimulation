"""smoke_test_full_tuned.py -- the full_tuned campaign path end to end (needs NEURON).

    python smoke_test_full_tuned.py       # expect the last line: 'All smoke tests passed.'

Takes a few minutes. The offline halves (the fits, the CSV schema, the merge) are covered by
smoke_test_bump_kinetics.py and smoke_test_culture_parallel.py -- run those first, they are
instant. This one exercises what only NEURON can show.

  [1] the model is built as decided: full channel set, Rich's PASSIVE axon, leak tuned so the
      measured rest is imposed isopotentially
  [2] THE REBUILD INVARIANT. CellPool keeps one live cell and rebuilds it on every morphology
      switch, but caches rest and the sham per (morphology, layer). The per-build biophysical
      state must be re-applied on each rebuild, or the cell stops matching its own cached
      sham. Before apply_model_state() was split out of _rest_and_sham(), this produced a
      spurious 0.409 mV 'bump' with near-degenerate taus, identical at 370 and 500 um.
  [3] the long window does not disturb the outcome columns it shares a simulation with
  [4] kinetics land in the row, and a culture outside the subsample gets blanks, not garbage
  [5] soma_only through the same path has NO bump (Ih is dendritic) -- the negative control
"""
import sys

import numpy as np

import bump_kinetics as bk
import culture_export as ce
from config import CFG

FAILURES = []


def check(name, cond, detail=""):
    print("  %-70s %s" % (name, "PASS" if cond else "FAIL " + str(detail)))
    if not cond:
        FAILURES.append(name)


CFG.cell_model = "full_tuned"
CFG.use_hpc_count = False
CFG.n_neurons = 4
CFG.morphologies = ["60308", "130303"]
CFG.layers_um = (80.0,)
CFG.bump_ms = 400.0                      # short for speed; 800 in production
MORPHS = list(CFG.morphologies)
LAYER = 80.0
POS, THETA, I0 = (90.0, -30.0), 0.0, float(CFG.i0_uA)

print("\n[1] the model is built as decided")
pool = ce.CellPool(CFG, "full_tuned")
try:
    info = pool.info(MORPHS[0], LAYER)
    check("cell_model recorded as full_tuned", info["cell_model"] == "full_tuned", info["cell_model"])
    rep = info.get("leak_report")
    check("leak tuning ran and reports %s segments" % (rep or {}).get("n_segments"),
          rep is not None and rep["n_segments"] > 100, rep)
    check("net standing current driven to zero (%.2e -> %.2e mA/cm2)"
          % (rep["max_abs_net_before"], rep["max_abs_net_after"]),
          rep["max_abs_net_after"] < 1e-12 and rep["max_abs_net_before"] > 1e-6)
    check("v_target is the cell's OWN settled rest (%.4f mV), not the config scalar (%.4f)"
          % (rep["v_target_mV"], CFG.v_rest_mV),
          abs(rep["v_target_mV"] - info["v_rest"]) < 1e-9
          and abs(rep["v_target_mV"] - CFG.v_rest_mV) > 1e-3)
    check("the axon is passive: its e_pas is a single value (Rich's own axon)",
          "axon" in rep["per_domain"] and
          abs(rep["per_domain"]["axon"][0] - rep["per_domain"]["axon"][2]) < 1e-9,
          rep["per_domain"].get("axon"))
    check("the sham is quiescent and its long-window trace was stored",
          "t_grid_sham" in info and info["t_grid_sham"].size > 100,
          info.get("t_grid_sham", np.array([])).size)

    print("\n[2] THE REBUILD INVARIANT (the bug this test exists for)")
    a = pool.simulate(MORPHS[0], LAYER, POS, THETA, I0)
    n0 = pool.n_builds
    pool.simulate(MORPHS[1], LAYER, POS, THETA, I0)          # forces a rebuild...
    b = pool.simulate(MORPHS[0], LAYER, POS, THETA, I0)      # ...and back
    check("the pool really did rebuild (%d -> %d builds)" % (n0, pool.n_builds),
          pool.n_builds >= n0 + 2)
    check("DeltaV_end identical across the rebuild (%.9f vs %.9f mV)"
          % (a["dv_end"], b["dv_end"]), abs(a["dv_end"] - b["dv_end"]) < 1e-9,
          a["dv_end"] - b["dv_end"])
    ka, kb = a["kinetics"]["bump"], b["kinetics"]["bump"]
    check("bump peak identical across the rebuild (%.6f vs %.6f mV)"
          % (ka["peak_mV"], kb["peak_mV"]), abs(ka["peak_mV"] - kb["peak_mV"]) < 1e-6,
          ka["peak_mV"] - kb["peak_mV"])
    check("bump taus identical across the rebuild (%.3f/%.3f vs %.3f/%.3f ms)"
          % (ka["tau_rise_ms"], ka["tau_decay_ms"], kb["tau_rise_ms"], kb["tau_decay_ms"]),
          abs(ka["tau_rise_ms"] - kb["tau_rise_ms"]) < 1e-6
          and abs(ka["tau_decay_ms"] - kb["tau_decay_ms"]) < 1e-6)

    print("\n[3] the long window leaves the outcome columns alone")
    CFG.bump_ms = 0.0
    pool_s = ce.CellPool(CFG, "full_tuned")
    try:
        s = pool_s.simulate(MORPHS[0], LAYER, POS, THETA, I0)
        check("DeltaV_end with and without the long window (%.9f vs %.9f mV)"
              % (a["dv_end"], s["dv_end"]), abs(a["dv_end"] - s["dv_end"]) < 1e-6,
              a["dv_end"] - s["dv_end"])
        check("outcome label unchanged (%s)" % s["outcome"], s["outcome"] == a["outcome"])
        check("bump_ms = 0 leaves the kinetics blank, not wrong",
              s["kinetics"]["bump"]["fit_ok"] == 0
              and not np.isfinite(s["kinetics"]["bump"]["tau_rise_ms"]))
        blanks = bk.staged_row_values(s["kinetics"])
        check("...and the row still has all %d kinetics columns" % len(bk.STAGED_COLUMNS),
              len(blanks) == len(bk.STAGED_COLUMNS))
    finally:
        pool_s.close()
    CFG.bump_ms = 400.0

    print("\n[4] the bump itself, and the row it produces")
    kin = a["kinetics"]
    e, b_ = kin["early"], kin["bump"]
    check("bump measured: %+.3f mV at %.1f ms, tau_r %.1f tau_d %.1f, r2 %.5f, fit_ok %d"
          % (b_["peak_mV"], b_["t_peak_ms"], b_["tau_rise_ms"], b_["tau_decay_ms"],
             b_["r2"], b_["fit_ok"]),
          b_["fit_ok"] == 1 and b_["peak_mV"] > 0.3 and b_["r2"] > 0.99
          and 10.0 < b_["tau_rise_ms"] < 200.0 and b_["tau_decay_ms"] > b_["tau_rise_ms"])
    check("the direct relaxation is resolved on the fine grid (tau_m %.4f ms, t_1e %.4f ms)"
          % (e["tau_ms"], e["t_1e_ms"]),
          e["fit_ok"] == 1 and 0.01 < e["tau_ms"] < 5.0 and np.isfinite(e["t_1e_ms"]))
    check("the residual term matches -DeltaV(t0) (%+.4f vs %+.4f mV)"
          % (b_["offset_mV"], b_["offset_expected_mV"]),
          abs(b_["offset_mV"] - b_["offset_expected_mV"]) < 0.15 * max(0.1, abs(b_["dv_t0_mV"])))
    row = ce.build_row(0, 0, MORPHS[0], LAYER, POS[0], POS[1], 10.0, 20.0, 21.0, 30.0, 40.0,
                       1, I0, 0, pool.info(MORPHS[0], LAYER), a)
    check("build_row produces exactly len(CSV_HEADER) = %d values" % len(ce.CSV_HEADER),
          len(row) == len(ce.CSV_HEADER))
    i = ce.CSV_HEADER.index("dexp_tau_decay_ms")
    check("the row carries the fitted tau_decay (%s ms)" % row[i],
          abs(float(row[i]) - b_["tau_decay_ms"]) < 1e-2)

    print("\n[5] the subsample, and the soma_only negative control")
    check("fraction 1.0 -> every culture measured",
          all(ce.culture_has_kinetics(7, c, 1.0) for c in range(20)))
    check("fraction 0.0 -> none", not any(ce.culture_has_kinetics(7, c, 0.0) for c in range(20)))
    n = sum(ce.culture_has_kinetics(7, c, 0.5) for c in range(400))
    check("fraction 0.5 -> %d/400 cultures, and the draw is reproducible" % n,
          150 < n < 250 and all(ce.culture_has_kinetics(7, c, 0.5)
                                == ce.culture_has_kinetics(7, c, 0.5) for c in range(50)))
    sk = pool.simulate(MORPHS[0], LAYER, POS, THETA, I0, with_kinetics=False)
    check("with_kinetics=False leaves the kinetics blank but keeps the outcome",
          sk["kinetics"]["bump"]["fit_ok"] == 0 and abs(sk["dv_end"] - a["dv_end"]) < 1e-6)
finally:
    pool.close()

CFG.cell_model = "soma_only"
pool2 = ce.CellPool(CFG, "soma_only")
try:
    so = pool2.simulate(MORPHS[0], LAYER, POS, THETA, I0)
    b2 = so["kinetics"]["bump"]
    # Assert the MEASURED extremum, not the fitted one: with no bump to fit, the model is
    # rejected (fit_ok 0) and its peak_mV is the extremum of a curve that does not describe
    # the trace -- meaningless by construction. data_peak_mV is what the trace actually did.
    check("soma_only has no bump: measured %.4f mV at %.1f ms -- Ih is dendritic"
          % (b2["data_peak_mV"], b2["data_t_peak_ms"]), abs(b2["data_peak_mV"]) < 0.05,
          b2["data_peak_mV"])
    check("...and the fit is REJECTED rather than reporting taus (fit_ok %d, tau_d %.0f ms "
          "pinned to its bound)" % (b2["fit_ok"], b2["tau_decay_ms"]), b2["fit_ok"] == 0)
    check("...while the SAME code on full_tuned accepted the fit -- the control discriminates",
          b_["fit_ok"] == 1 and abs(b_["peak_mV"]) > 10 * abs(b2["data_peak_mV"]))
finally:
    pool2.close()

print()
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
    sys.exit(1)
print("All smoke tests passed.")
