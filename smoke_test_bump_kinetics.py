"""smoke_test_bump_kinetics.py -- OFFLINE test of bump_kinetics.py (no NEURON).

    python smoke_test_bump_kinetics.py      # expect the last line: 'All smoke tests passed.'

  [1] the two taus are recovered from traces built to the fitted model
  [2] sign handling: depolarising and hyperpolarising bumps
  [3] the early direct-polarisation transient is stepped over, not mistaken for the bump
  [4] guards: negligible amplitude, peak pinned to a window edge, too few samples, flat trace
  [5] a realistic double-exponential bump gives sensible, monotonic taus
  [6] the CSV row helper matches KINETICS_COLUMNS and survives nan

  --- the figure-grade fits (fit_exp_decay / fit_double_exp_from_zero / fit_post_pulse) ---
  [7]  fit_exp_decay recovers (amp, tau) from a clean single exponential, both signs
  [8]  fit_exp_decay stops at the bump upturn instead of fitting through it
  [9]  fit_double_exp_from_zero is pinned: yhat(t0) = DeltaV(t0) exactly, for any parameters
  [10] fit_post_pulse recovers all FIVE parameters from traces built to its model
  [11] the eval_* helpers reconstruct the fitted trace, and the two components sum to it
  [12] identifiability: swap_dsse_frac is large when the components separate, ~0 when not
  [13] fit_post_pulse under noise, and the fit_ok guards
  [14] the two new CSV row helpers match their column tuples
"""
import sys

import numpy as np

from bump_kinetics import (DEXP_COLUMNS, EXPDECAY_COLUMNS, KINETICS_COLUMNS,
                           POSTPULSE_COLUMNS, _dexp_design, dexp_row_values,
                           eval_double_exp_from_zero, eval_exp_decay, expdecay_row_values,
                           fit_double_exp_from_zero, fit_exp_decay, fit_ih_bump,
                           fit_post_pulse, kinetics_row_values, postpulse_row_values)

FAILURES = []


def check(name, cond, detail=""):
    print("  %-66s %s" % (name, "PASS" if cond else "FAIL " + str(detail)))
    if not cond:
        FAILURES.append(name)


def model_trace(peak, t_peak, tau_r, tau_d, t_skip=7.0, t_end=400.0, dt=0.5):
    """A trace built EXACTLY to the fitted model, so the taus must come back."""
    t = np.arange(0.0, t_end + dt, dt)
    dv = np.empty_like(t)
    rise = t < t_peak
    A = peak / np.exp(-(t_skip - t_skip) / tau_r)      # dv(t_skip) = 0
    dv[rise] = peak - A * np.exp(-(t[rise] - t_skip) / tau_r)
    dv[~rise] = peak * np.exp(-(t[~rise] - t_peak) / tau_d)
    return t, dv


print("\n[1] tau recovery")
for tau_r, tau_d in ((30.0, 150.0), (60.0, 400.0), (15.0, 80.0)):
    t, dv = model_trace(1.0, 100.0, tau_r, tau_d)
    f = fit_ih_bump(t, dv)
    check("tau_rise %.0f -> %.2f ms and tau_decay %.0f -> %.2f ms"
          % (tau_r, f["tau_rise_ms"], tau_d, f["tau_decay_ms"]),
          abs(f["tau_rise_ms"] - tau_r) / tau_r < 0.02
          and abs(f["tau_decay_ms"] - tau_d) / tau_d < 0.02 and f["fit_ok"] == 1,
          (f["tau_rise_ms"], f["tau_decay_ms"], f["fit_ok"]))
t, dv = model_trace(1.0, 100.0, 30.0, 150.0)
f = fit_ih_bump(t, dv)
check("peak amplitude and time recovered (%.4f mV at %.1f ms)" % (f["peak_mV"], f["time_to_peak_ms"]),
      abs(f["peak_mV"] - 1.0) < 1e-6 and abs(f["time_to_peak_ms"] - 100.0) < 0.6)
check("r2 of both fits > 0.999", f["r2_rise"] > 0.999 and f["r2_decay"] > 0.999,
      (f["r2_rise"], f["r2_decay"]))

print("\n[2] sign")
check("positive bump -> depol", f["sign"] == "depol")
t, dv = model_trace(-2.5, 80.0, 25.0, 200.0)
g = fit_ih_bump(t, dv)
check("negative bump -> hyperpol, peak %.3f mV, taus %.1f / %.1f ms"
      % (g["peak_mV"], g["tau_rise_ms"], g["tau_decay_ms"]),
      g["sign"] == "hyperpol" and abs(g["peak_mV"] + 2.5) < 1e-6
      and abs(g["tau_rise_ms"] - 25.0) / 25.0 < 0.02
      and abs(g["tau_decay_ms"] - 200.0) / 200.0 < 0.02, g)

print("\n[3] the early transient must not be mistaken for the bump")
t, dv = model_trace(1.0, 100.0, 30.0, 150.0)
dv = dv + 8.0 * np.exp(-t / 1.5)          # direct post-pulse polarisation, decays in a few ms
f_skip = fit_ih_bump(t, dv, t_skip_ms=7.0)
f_none = fit_ih_bump(t, dv, t_skip_ms=0.0)
check("with t_skip=7 the Ih peak is found (%.1f ms), not the transient" % f_skip["time_to_peak_ms"],
      abs(f_skip["time_to_peak_ms"] - 100.0) < 5.0, f_skip["time_to_peak_ms"])
check("with t_skip=0 the transient wins (%.1f ms) -- this is why t_skip exists"
      % f_none["time_to_peak_ms"], f_none["time_to_peak_ms"] < 5.0, f_none["time_to_peak_ms"])

print("\n[4] guards")
t, dv = model_trace(0.01, 100.0, 30.0, 150.0)
f0 = fit_ih_bump(t, dv, min_amp_mV=0.05)
check("negligible amplitude -> sign 'none', taus nan, fit_ok 0",
      f0["sign"] == "none" and not np.isfinite(f0["tau_rise_ms"]) and f0["fit_ok"] == 0)
t2 = np.arange(0.0, 400.5, 0.5)
f_edge = fit_ih_bump(t2, 1.0 - np.exp(-(t2 - 7.0) / 50.0))      # still rising at the window end
check("peak pinned to the window edge -> fit_ok 0", f_edge["fit_ok"] == 0, f_edge)
f_few = fit_ih_bump(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
check("too few samples -> all nan, no exception", f_few["fit_ok"] == 0
      and not np.isfinite(f_few["peak_mV"]))
f_flat = fit_ih_bump(t2, np.zeros_like(t2))
check("flat trace -> sign 'none', no exception", f_flat["sign"] == "none" and f_flat["fit_ok"] == 0)
f_mismatch = fit_ih_bump(np.arange(10.0), np.arange(5.0))
check("length mismatch -> all nan, no exception", f_mismatch["fit_ok"] == 0)

print("\n[5] realistic double-exponential bump")
prev = None
for td in (100.0, 200.0, 400.0):
    t = np.arange(0.0, 800.5, 0.5)
    dv = 1.2 * (1.0 - np.exp(-(t) / 40.0)) * np.exp(-t / td)
    f = fit_ih_bump(t, dv, t_search_ms=800.0)
    check("tau_decay grows with the generating constant (td=%.0f -> %.1f ms, peak at %.0f ms)"
          % (td, f["tau_decay_ms"], f["time_to_peak_ms"]),
          np.isfinite(f["tau_decay_ms"]) and (prev is None or f["tau_decay_ms"] > prev),
          f["tau_decay_ms"])
    prev = f["tau_decay_ms"]

print("\n[6] CSV row helper")
vals = kinetics_row_values(f)
check("one value per KINETICS_COLUMNS entry (%d)" % len(KINETICS_COLUMNS),
      len(vals) == len(KINETICS_COLUMNS))
vals_nan = kinetics_row_values(f_few)
check("a nan fit becomes empty CSV cells, not the string 'nan'",
      all(v == "" for v in vals_nan[:4]) and vals_nan[-1] == 0, vals_nan)



# ---------------------------------------------------------------------------
# The figure-grade fits. The model everything below is built from is
#
#     DeltaV(t) = P  * exp(-u/tau_m)                                   u = t - t0 >= 0
#               + Ab * ( exp(-u/tau_decay) - exp(-u/tau_rise) )
#
# with the bump term exactly 0 at u = 0 (the pinning), so DeltaV(t0) = P.
# ---------------------------------------------------------------------------
DT = 0.5
TT = np.arange(0.0, 800.0 + DT, DT)


def two_component(P, tau_m, Ab, tau_rise, tau_decay, t=TT):
    """A trace built EXACTLY to the fit_post_pulse model."""
    return P * np.exp(-t / tau_m) + Ab * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))


print("\n[7] fit_exp_decay on a clean single exponential (no bump present)")
for P, tau_m in ((-3.0, 12.0), (2.5, 20.0)):
    dv = P * np.exp(-TT / tau_m)
    e = fit_exp_decay(TT, dv)
    want = "hyperpol" if P < 0 else "depol"
    check("P %+.1f tau_m %.0f -> amp %+.4f tau %.4f sign %s r2 %.6f"
          % (P, tau_m, e["amp_mV"], e["tau_ms"], e["sign"], e["r2"]),
          abs(e["amp_mV"] - P) < 1e-6 and abs(e["tau_ms"] - tau_m) / tau_m < 1e-3
          and e["sign"] == want and e["peak_mV"] == P and e["t_peak_ms"] == 0.0
          and e["fit_ok"] == 1,
          (e["amp_mV"], e["tau_ms"], e["sign"], e["fit_ok"]))

print("\n[7b] the model-free 1/e time")
for P, tau_m in ((-3.0, 12.0), (2.5, 20.0), (-1.0, 3.0)):
    e = fit_exp_decay(TT, P * np.exp(-TT / tau_m))
    check("pure exponential: t_1e %.4f ms == tau %.1f ms (%.2e relative)"
          % (e["t_1e_ms"], tau_m, abs(e["t_1e_ms"] - tau_m) / tau_m),
          abs(e["t_1e_ms"] - tau_m) / tau_m < 2e-3, e["t_1e_ms"])
    check("pure exponential: amp_over_peak %.6f == 1" % e["amp_over_peak"],
          abs(e["amp_over_peak"] - 1.0) < 1e-6)
# A sum of two decays is what the real traces are. tau_ms must then move with the fitted
# window and t_1e_ms must not -- which is exactly why the figure quotes both. The DIRECTION
# tau_ms moves is not fixed: here the fast component dominates early so a narrower window
# gives a shorter tau, while on the measured traces the decay accelerates and it is the other
# way round (0.22 ms at floor 0.10, 0.47 at 0.50).
two = -1.5 * np.exp(-TT / 3.0) - 1.5 * np.exp(-TT / 30.0)
taus, t1es = [], []
for fl in (0.10, 0.30, 0.50):
    e = fit_exp_decay(TT, two, floor_fraction=fl)
    taus.append(e["tau_ms"]); t1es.append(e["t_1e_ms"])
check("two-component decay: tau_ms is window-dependent (%s), spread %.0f%%"
      % (", ".join("%.2f" % v for v in taus),
         100.0 * (max(taus) - min(taus)) / min(taus)),
      (max(taus) - min(taus)) / min(taus) > 0.25 and all(np.isfinite(taus)), taus)
check("two-component decay: t_1e_ms is NOT (%s)"
      % ", ".join("%.4f" % v for v in t1es), max(t1es) - min(t1es) < 1e-9, t1es)
check("...and a single exponential shows no such spread",
      abs(max(fit_exp_decay(TT, -3.0 * np.exp(-TT / 12.0), floor_fraction=f)["tau_ms"]
              for f in (0.10, 0.30, 0.50))
          - min(fit_exp_decay(TT, -3.0 * np.exp(-TT / 12.0), floor_fraction=f)["tau_ms"]
                for f in (0.10, 0.30, 0.50))) < 1e-6)

print("\n[8] fit_exp_decay stops where the bump starts")
dv = two_component(-3.0, 12.0, 2.1, 25.0, 180.0)
e = fit_exp_decay(TT, dv)
check("the fitted window ends well before the 40 ms cap (hi %.1f ms)" % e["t_fit_hi_ms"],
      np.isfinite(e["t_fit_hi_ms"]) and e["t_fit_hi_ms"] < 40.0, e["t_fit_hi_ms"])
cross = float(TT[int(np.argmax(dv >= 0.0))])          # DeltaV crosses back through zero here
check("window hi %.1f ms stops before the zero crossing at %.1f ms"
      % (e["t_fit_hi_ms"], cross), e["t_fit_hi_ms"] < cross, (e["t_fit_hi_ms"], cross))
check("it was the |DeltaV| floor that stopped it (last |DeltaV| %.4f ~ 10%% of 3 mV)"
      % abs(float(dv[int(round(e["t_fit_hi_ms"] / DT))])),
      abs(float(dv[int(round(e["t_fit_hi_ms"] / DT))])) < 0.11 * 3.0)
check("peak is read at the pulse end for a monotone relaxation (%.4f mV at %.2f ms)"
      % (e["peak_mV"], e["t_peak_ms"]),
      abs(e["peak_mV"] - float(dv[0])) < 1e-12 and e["t_peak_ms"] == 0.0)
check("the isolated early fit is BIASED by the bump (tau %.2f ms vs true 12 ms) -- "
      "this is why fit_post_pulse exists" % e["tau_ms"], e["tau_ms"] < 10.0, e["tau_ms"])

print("\n[9] fit_double_exp_from_zero is pinned at t0")
for t0 in (0.0, 7.0, 25.0):
    f = fit_double_exp_from_zero(TT, dv, t0_ms=t0, tau_offset_ms=12.0)
    curve = eval_double_exp_from_zero(TT, f)
    k = int(np.argmin(np.abs(TT - f["t_fit_lo_ms"])))
    check("t0 %.0f ms: model(t0) - DeltaV(t0) = %.2e mV (exactly zero by construction)"
          % (t0, curve[k] - f["dv_t0_mV"]), abs(curve[k] - f["dv_t0_mV"]) < 1e-12,
          curve[k] - f["dv_t0_mV"])
    check("t0 %.0f ms: offset_expected_mV = -DeltaV(t0) = %+.4f mV"
          % (t0, f["offset_expected_mV"]),
          abs(f["offset_expected_mV"] + f["dv_t0_mV"]) < 1e-12)
# the pinning holds for ARBITRARY parameters, not just the fitted ones
u0 = np.array([0.0])
for tr, td, to in ((5.0, 50.0, 3.0), (100.0, 2000.0, 400.0), (1.0, 1.05, 1.0)):
    X = _dexp_design(u0, tr, td, True, to)
    check("basis at u=0 is (0, 0) for tau_rise %.1f tau_decay %.1f tau_off %.1f"
          % (tr, td, to), float(np.max(np.abs(X))) == 0.0, X)

print("\n[10] fit_post_pulse recovers all five parameters")
CASES = ((-3.0, 12.0, 2.1, 25.0, 180.0),      # hyperpol + depolarising bump (the usual case)
         (-4.0, 8.0, 1.4, 60.0, 250.0),       # slower bump rise
         (2.0, 15.0, -0.9, 40.0, 300.0),      # depol + hyperpolarising bump
         (-1.0, 18.0, 3.0, 45.0, 400.0))      # bump dominates the trace
fits = []
for (P, tau_m, Ab, tr, td) in CASES:
    dvc = two_component(P, tau_m, Ab, tr, td)
    f = fit_post_pulse(TT, dvc)
    fits.append((f, dvc, (P, tau_m, Ab, tr, td)))
    rel = max(abs(f["P_mV"] - P) / abs(P), abs(f["tau_m_ms"] - tau_m) / tau_m,
              abs(f["Ab_mV"] - Ab) / abs(Ab), abs(f["tau_rise_ms"] - tr) / tr,
              abs(f["tau_decay_ms"] - td) / td)
    check("P %+.1f tau_m %.0f Ab %+.1f tau_r %.0f tau_d %.0f -> worst relative error %.2e"
          % (P, tau_m, Ab, tr, td, rel), rel < 5e-3 and f["fit_ok"] == 1,
          (f["P_mV"], f["tau_m_ms"], f["Ab_mV"], f["tau_rise_ms"], f["tau_decay_ms"],
           f["fit_ok"]))
f0, dv0c, _ = fits[0]
bump_true = 2.1 * (np.exp(-TT / 180.0) - np.exp(-TT / 25.0))
k = int(np.argmax(np.abs(bump_true)))
check("bump peak %+.4f mV at %.1f ms (true %+.4f at %.1f ms)"
      % (f0["bump_peak_mV"], f0["bump_t_peak_ms"], bump_true[k], TT[k]),
      abs(f0["bump_peak_mV"] - bump_true[k]) < 5e-3
      and abs(f0["bump_t_peak_ms"] - TT[k]) < 1.0)
check("DeltaV(t0) is assigned entirely to the polarisation: P %+.4f vs data %+.4f"
      % (f0["P_mV"], f0["dv_t0_mV"]), abs(f0["P_mV"] - f0["dv_t0_mV"]) < 5e-3)

print("\n[11] the eval_* helpers rebuild the trace")
for f, dvc, _p in fits:
    a = eval_exp_decay(TT, f["early"])
    b = eval_double_exp_from_zero(TT, f["bump"])
    check("max |DeltaV - (early + bump)| = %.2e mV" % float(np.nanmax(np.abs(dvc - (a + b)))),
          float(np.nanmax(np.abs(dvc - (a + b)))) < 1e-3)
check("the bump component is exactly 0 at t0 (%.2e mV)"
      % eval_double_exp_from_zero(TT, fits[0][0]["bump"])[0],
      abs(eval_double_exp_from_zero(TT, fits[0][0]["bump"])[0]) < 1e-12)
check("the polarisation component equals P at t0 (%.6f vs %.6f)"
      % (eval_exp_decay(TT, fits[0][0]["early"])[0], fits[0][0]["P_mV"]),
      abs(eval_exp_decay(TT, fits[0][0]["early"])[0] - fits[0][0]["P_mV"]) < 1e-12)

print("\n[12] identifiability diagnostic")
check("well-separated components -> swap_dsse_frac %.1f is large" % fits[1][0]["swap_dsse_frac"],
      fits[1][0]["swap_dsse_frac"] > 10.0, fits[1][0]["swap_dsse_frac"])
dv_nopol = two_component(0.0, 12.0, 2.1, 25.0, 180.0)       # no direct polarisation at all
f_np = fit_post_pulse(TT, dv_nopol)
check("no polarisation component -> swap_dsse_frac %.3g is ~0 (tau_m unidentifiable)"
      % f_np["swap_dsse_frac"], f_np["swap_dsse_frac"] < 1e-3, f_np["swap_dsse_frac"])
check("...while the bump itself is still recovered (tau_d %.2f, peak %+.4f mV)"
      % (f_np["tau_decay_ms"], f_np["bump_peak_mV"]),
      abs(f_np["tau_decay_ms"] - 180.0) / 180.0 < 0.05
      and abs(f_np["bump_peak_mV"] - bump_true[k]) < 0.05,
      (f_np["tau_decay_ms"], f_np["bump_peak_mV"]))

print("\n[13] noise and guards")
rng = np.random.RandomState(7)
dvn = two_component(-3.0, 12.0, 2.1, 25.0, 180.0) + rng.normal(0.0, 0.02, TT.size)
fn = fit_post_pulse(TT, dvn)
check("with 0.02 mV noise: tau_m %.2f, tau_r %.2f, tau_d %.2f, peak %+.4f mV, r2 %.4f"
      % (fn["tau_m_ms"], fn["tau_rise_ms"], fn["tau_decay_ms"], fn["bump_peak_mV"], fn["r2"]),
      abs(fn["tau_m_ms"] - 12.0) / 12.0 < 0.15 and abs(fn["tau_rise_ms"] - 25.0) / 25.0 < 0.15
      and abs(fn["tau_decay_ms"] - 180.0) / 180.0 < 0.10
      and abs(fn["bump_peak_mV"] - bump_true[k]) < 0.05 and fn["fit_ok"] == 1,
      (fn["tau_m_ms"], fn["tau_rise_ms"], fn["tau_decay_ms"], fn["fit_ok"]))
check("flat trace -> fit_ok 0, no exception",
      fit_post_pulse(TT, np.zeros_like(TT))["fit_ok"] == 0)
check("too few samples -> fit_ok 0, no exception",
      fit_post_pulse(TT[:3], np.zeros(3))["fit_ok"] == 0)
check("mismatched lengths -> fit_ok 0, no exception",
      fit_post_pulse(TT, np.zeros(7))["fit_ok"] == 0)
nan_tr = two_component(-3.0, 12.0, 2.1, 25.0, 180.0).copy()
nan_tr[10:20] = np.nan
check("nan samples are dropped, not propagated (tau_d %.2f)"
      % fit_post_pulse(TT, nan_tr)["tau_decay_ms"],
      abs(fit_post_pulse(TT, nan_tr)["tau_decay_ms"] - 180.0) / 180.0 < 0.05)
f_pin = fit_post_pulse(TT, two_component(-3.0, 12.0, 2.1, 25.0, 180.0),
                       bounds_decay_ms=(5.0, 60.0))      # true tau_d is outside the box
check("a tau pinned to a search bound -> fit_ok 0 (tau_d %.2f at bound 60)"
      % f_pin["tau_decay_ms"], f_pin["fit_ok"] == 0, f_pin["tau_decay_ms"])
check("fit_exp_decay on a flat trace -> fit_ok 0, sign 'none'",
      fit_exp_decay(TT, np.zeros_like(TT))["fit_ok"] == 0
      and fit_exp_decay(TT, np.zeros_like(TT))["sign"] == "none")
check("fit_double_exp_from_zero on a flat trace -> fit_ok 0, no exception",
      fit_double_exp_from_zero(TT, np.zeros_like(TT))["fit_ok"] == 0)
check("eval_* on an empty fit return all-nan, no exception",
      np.all(np.isnan(eval_exp_decay(TT, fit_exp_decay(TT[:3], np.zeros(3)))))
      and np.all(np.isnan(eval_double_exp_from_zero(TT, fit_double_exp_from_zero(TT[:3],
                                                                                 np.zeros(3))))))

print("\n[14] CSV row helpers for the new fits")
check("expdecay_row_values matches EXPDECAY_COLUMNS (%d)" % len(EXPDECAY_COLUMNS),
      len(expdecay_row_values(fits[0][0]["early"])) == len(EXPDECAY_COLUMNS))
check("dexp_row_values matches DEXP_COLUMNS (%d)" % len(DEXP_COLUMNS),
      len(dexp_row_values(fits[0][0]["bump"])) == len(DEXP_COLUMNS))
check("postpulse_row_values matches POSTPULSE_COLUMNS (%d)" % len(POSTPULSE_COLUMNS),
      len(postpulse_row_values(fits[0][0])) == len(POSTPULSE_COLUMNS))
bad = fit_post_pulse(TT[:3], np.zeros(3))
check("a failed fit becomes empty CSV cells, not the string 'nan'",
      all(v == "" for v in postpulse_row_values(bad)[:8]) and postpulse_row_values(bad)[-1] == 0,
      postpulse_row_values(bad))

print()
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("All smoke tests passed.")
