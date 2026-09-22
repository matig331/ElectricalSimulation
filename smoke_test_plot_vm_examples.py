"""smoke_test_plot_vm_examples.py -- END-TO-END test of plot_vm_examples.py (needs NEURON).

    python smoke_test_plot_vm_examples.py        # expect the last line: 'All smoke tests passed.'

Takes a few minutes: it builds two cells and runs real simulations. Everything that can be
tested WITHOUT NEURON (the fits themselves) lives in smoke_test_bump_kinetics.py -- run that
first, it is instant.

  [1] the two runs share their fixed-dt time base exactly, so DeltaV at 0.025 ms is a plain
      subtraction over the whole play_margin_ms window -- no interpolation anywhere
  [2] the uniform grid carries a sample AT the end of phase 2, and is identical in both runs
  [3] the sham really is placement-independent (with I = 0 the field is 0 everywhere)
  [4] positive/negative control: full_tuned has a ~1 mV bump, soma_only has none (Ih is
      dendritic), and the guards mark the latter fit_ok = 0 rather than inventing taus
  [5] the fitted bump reproduces the measured trace: component peak vs measured DeltaV
      extremum, and offset_mV vs -DeltaV(t0)
  [6] the pinned fit WITHOUT the residual term collapses -- the failure mode the residual
      term exists to prevent
  [7] staged and joint fit modes agree on the bump
  [8] the driver writes a PDF with one page per placement and a CSV with a matching header
"""
import os
import re
import sys

import numpy as np

import bump_kinetics as bk
import plot_vm_examples as P
from config import CFG

MORPH, LAYER = CFG.morphologies[0], float(CFG.layers_um[len(CFG.layers_um) // 2])
BUMP_MS = 400.0
PLACES = [(90.0, -30.0, 0.0), (-110.0, -110.0, 270.0)]

FAILURES = []


def check(name, cond, detail=""):
    print("  %-72s %s" % (name, "PASS" if cond else "FAIL " + str(detail)))
    if not cond:
        FAILURES.append(name)


proto = P.default_protocol(CFG, bump_ms=BUMP_MS)
phi = proto["phase_dur_ms"]

print("\n[build] %s L%d, bump window %.0f ms" % (MORPH, LAYER, BUMP_MS))
tuned = P.build_instance(MORPH, LAYER, "full_tuned", cfg=CFG)
sham_t = P.sham_reference(tuned, proto)
recs = [P.simulate_instance(tuned, proto, sham_t, (x, y), th) for (x, y, th) in PLACES]
fits = [P.analyse(r) for r in recs]

print("\n[1] the fine DeltaV is an exact subtraction")
for r, (x, y, th) in zip(recs, PLACES):
    k = r["t_dv_fine"].size
    check("(%+.0f,%+.0f) th=%.0f: %d shared fixed-dt samples, exact to %.1e ms"
          % (x, y, th, k, float(np.max(np.abs(r["t_dv_fine"] - r["t_fine_sham"][:k])))),
          k > 100 and float(np.max(np.abs(r["t_dv_fine"] - r["t_fine_sham"][:k]))) < 1e-9)
    span = float(r["t_dv_fine"][-1])
    check("   it reaches %.2f ms past the pulse (play_margin_ms = %.1f)"
          % (span, proto["play_margin_ms"]), span >= 0.9 * proto["play_margin_ms"], span)
    check("   and starts before the pulse (%.3f ms <= -2*phi = %.3f)"
          % (float(r["t_dv_fine"][0]), -2.0 * phi),
          float(r["t_dv_fine"][0]) <= -2.0 * phi + 1e-9)

print("\n[2] the uniform grid is anchored at the end of phase 2")
for r, (x, y, th) in zip(recs, PLACES):
    j = int(np.argmin(np.abs(r["t_grid"])))
    check("(%+.0f,%+.0f): closest grid sample to t=0 is %.2e ms away"
          % (x, y, float(r["t_grid"][j])), abs(float(r["t_grid"][j])) < 1e-9,
          r["t_grid"][j])
    check("   the grid spans [%.1f, %.1f] ms at %.2f ms"
          % (r["t_grid"][0], r["t_grid"][-1], proto["bump_dt_ms"]),
          abs(float(r["t_grid"][-1]) - BUMP_MS) < 2.0 * proto["bump_dt_ms"])
check("stim and sham share the uniform grid exactly",
      float(np.max(np.abs(recs[0]["t_grid"] - sham_t["t_grid"]))) < 1e-9)

print("\n[3] the sham is placement-independent")
_f1, _v1, _tw1, _vw1, _tb1, vb1 = P._run(tuned, proto, (0.0, 0.0), 0.0, 0.0)
_f2, _v2, _tw2, _vw2, _tb2, vb2 = P._run(tuned, proto, (200.0, -150.0), 137.0, 0.0)
check("two very different placements give the same zero-current trace (max %.2e mV)"
      % float(np.max(np.abs(vb1 - vb2))), float(np.max(np.abs(vb1 - vb2))) < 1e-9)

print("\n[4] positive and negative control")
for r, f, (x, y, th) in zip(recs, fits, PLACES):
    b = f["bump"]
    late = r["t_grid"] >= 10.0
    obs = float(r["dv"][late][int(np.argmax(np.abs(r["dv"][late])))])
    check("full_tuned (%+.0f,%+.0f): bump %+.3f mV @ %.1f ms, tau_r %.1f tau_d %.1f, r2 %.5f, ok=%d"
          % (x, y, b["peak_mV"], b["t_peak_ms"], b["tau_rise_ms"], b["tau_decay_ms"],
             b["r2"], b["fit_ok"]),
          b["fit_ok"] == 1 and abs(b["peak_mV"]) > 0.5 and b["r2"] > 0.99
          and 10.0 < b["tau_rise_ms"] < 200.0 and 50.0 < b["tau_decay_ms"] < 1000.0
          and b["tau_decay_ms"] > b["tau_rise_ms"],
          (b["peak_mV"], b["tau_rise_ms"], b["tau_decay_ms"], b["fit_ok"]))
    check("   the fitted component peak %+.3f matches the measured extremum %+.3f mV"
          % (b["peak_mV"], obs), abs(b["peak_mV"] - obs) < 0.1 * max(0.2, abs(obs)),
          (b["peak_mV"], obs))
    check("   the early fit resolves the fast relaxation (tau_m %.4f ms, t_1e %.4f ms, n=%d)"
          % (f["early"]["tau_ms"], f["early"]["t_1e_ms"], f["early"]["n"]),
          f["early"]["fit_ok"] == 1 and 0.01 < f["early"]["tau_ms"] < 5.0
          and f["early"]["n"] >= bk.MIN_FIT_POINTS, f["early"])

soma = P.build_instance(MORPH, LAYER, "soma_only", cfg=CFG)
sham_s = P.sham_reference(soma, proto)
r_s = P.simulate_instance(soma, proto, sham_s, (90.0, -30.0), 0.0)
f_s = P.analyse(r_s)
late = r_s["t_grid"] >= 10.0
obs_s = float(np.max(np.abs(r_s["dv"][late])))
check("soma_only: no bump (max |DeltaV| after 10 ms = %.4f mV) -- Ih is dendritic"
      % obs_s, obs_s < 0.05, obs_s)
check("   ...and the guards report it instead of inventing taus (fit_ok = %d)"
      % f_s["bump"]["fit_ok"], f_s["bump"]["fit_ok"] == 0)

print("\n[5] the pinning is consistent with the trace")
for r, f, (x, y, th) in zip(recs, fits, PLACES):
    b = f["bump"]
    curve = bk.eval_double_exp_from_zero(r["t_grid"], b)
    j = int(np.argmin(np.abs(r["t_grid"] - b["t_fit_lo_ms"])))
    check("(%+.0f,%+.0f): model(t0) - DeltaV(t0) = %.2e mV (zero by construction)"
          % (x, y, curve[j] - b["dv_t0_mV"]), abs(curve[j] - b["dv_t0_mV"]) < 1e-12)
    rel = abs(b["offset_mV"] - b["offset_expected_mV"]) / max(1e-9, abs(b["dv_t0_mV"]))
    check("   offset %+.4f vs -DeltaV(t0) %+.4f mV (%.1f%% apart)"
          % (b["offset_mV"], b["offset_expected_mV"], 100.0 * rel), rel < 0.10, rel)
    resid = float(np.nanmax(np.abs(r["dv"][r["t_grid"] >= 5.0]
                                   - curve[r["t_grid"] >= 5.0])))
    check("   max |DeltaV - model| after 5 ms = %.4f mV" % resid,
          resid < 0.05 * max(0.2, abs(b["peak_mV"])), resid)

print("\n[6] without the residual term the pinned fit collapses")
r, b = recs[0], fits[0]["bump"]
naive = bk.fit_double_exp_from_zero(r["t_grid"], r["dv"], t0_ms=0.0, free_offset=False)
check("no-residual fit: tau_r %.1f tau_d %.1f r2 %.4f ok=%d  vs  with: %.1f / %.1f / %.5f ok=%d"
      % (naive["tau_rise_ms"], naive["tau_decay_ms"], naive["r2"], naive["fit_ok"],
         b["tau_rise_ms"], b["tau_decay_ms"], b["r2"], b["fit_ok"]),
      naive["r2"] < b["r2"] - 0.05 and naive["fit_ok"] == 0 and b["fit_ok"] == 1,
      (naive["r2"], b["r2"]))

print("\n[7] staged and joint modes agree on the bump")
for r, f, (x, y, th) in zip(recs, fits, PLACES):
    jf = P.analyse(r, mode="joint")
    a, c = f["bump"], jf["bump"]
    dr = abs(a["tau_rise_ms"] - c["tau_rise_ms"]) / a["tau_rise_ms"]
    dd = abs(a["tau_decay_ms"] - c["tau_decay_ms"]) / a["tau_decay_ms"]
    dp = abs(a["peak_mV"] - c["peak_mV"]) / abs(a["peak_mV"])
    check("(%+.0f,%+.0f): tau_r %.1f/%.1f (%.1f%%), tau_d %.1f/%.1f (%.1f%%), peak %.3f/%.3f (%.1f%%)"
          % (x, y, a["tau_rise_ms"], c["tau_rise_ms"], 100 * dr, a["tau_decay_ms"],
             c["tau_decay_ms"], 100 * dd, a["peak_mV"], c["peak_mV"], 100 * dp),
          dr < 0.15 and dd < 0.15 and dp < 0.10, (dr, dd, dp))

print("\n[8] the driver writes a PDF and a CSV")
places = [(MORPH, LAYER, x, y, th) for (x, y, th) in PLACES]
pdf, csvp = P.main(cell_model="full_tuned", placements=places, bump_ms=BUMP_MS,
                   out_pdf="_smoke_pve.pdf", out_csv="_smoke_pve.csv", verbose=False)
check("PDF exists and is not empty (%d bytes)" % (os.path.getsize(pdf) if
                                                  os.path.exists(pdf) else 0),
      os.path.exists(pdf) and os.path.getsize(pdf) > 10000)
with open(csvp) as fh:
    lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
head = lines[0].split(",")
check("CSV header has %d columns and matches CSV_HEADER" % len(P.CSV_HEADER),
      head == list(P.CSV_HEADER), head[:5])
check("CSV has one row per placement (%d)" % len(PLACES), len(lines) - 1 == len(PLACES))
for ln in lines[1:]:
    check("   row has %d fields" % len(P.CSV_HEADER),
          len(ln.split(",")) == len(P.CSV_HEADER), ln[:80])
with open(pdf, "rb") as fh:
    blob = fh.read()
counts = [int(c) for c in re.findall(rb"/Count\s+(\d+)", blob)]
n_pages = max(counts) if counts else 0
check("the PDF page tree reports %d pages for %d placements" % (n_pages, len(PLACES)),
      n_pages == len(PLACES), counts)
for f in (pdf, csvp):
    if os.path.exists(f):
        os.remove(f)

print()
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
    sys.exit(1)
print("All smoke tests passed.")
