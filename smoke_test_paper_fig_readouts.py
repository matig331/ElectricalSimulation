"""smoke_test_paper_fig_readouts.py -- offline checks of the publication figure (no NEURON).

A synthetic record is built from KNOWN components with the same time bases the simulator
returns (fixed-dt prefix shared by stim and sham, sparse solver steps after it, the uniform
0.5 ms grid), then pushed through exactly what the real figure runs.

  [1] save_record / load_record round trip (arrays and metadata identical)
  [2] the fits recover the generating bump (tau_r, tau_d, peak) and a finite fast relaxation
  [3] what is DRAWN obeys the geometry it claims:
        panel C's two exponential terms sum to the fitted bump; fit and bump data start at
        exactly 0 at t0 while the lower square is DeltaV_end (the pinning); each tau marker
        sits on its term at 1/e; tau_r < t_b < tau_d; the t_1/e point lies on the data at
        |peak|/e; the reconstruction matches the trace; the DeltaV arrow spans the measured
        bump; the pulse is charge-balanced
  [4] the four files are written: PDF with TrueType fonts embedded (no Type 3), SVG with
      text kept as text, PNG 180 mm wide at 600 dpi, and the values file
  [5] no two labels overlap and no text leaves the page -- for a HYPERPOLARISING fast
      response with the peak after t0, and for a DEPOLARISING one with the peak AT t0;
      and a record with no bump at all still renders
  [6] every scale bar is exactly as long as its label says, and every dimension line
      spans exactly its readout (t_1/e in B; t_b, tau_r, tau_d in C, all from t0)
  [7] the example ranking helper and the CLI (suggest, plot)
  [8] the module source is pure ASCII

Run:   python smoke_test_paper_fig_readouts.py          (~15 s)
"""
import csv
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paper_fig_readouts as PF                           # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def section(t):
    print("\n" + t)


TRUE = dict(Ab=2.6, tau_r=50.0, tau_d=200.0)


def synthetic(sign=-1.0, peak_after_t0=True):
    """(rec, meta) on the simulator's time bases, from known fast + bump components.

    sign=-1: hyperpolarising fast response that keeps deepening ~0.06 ms after t0 (as in the
    real example); sign=+1 with peak_after_t0=False: depolarising, largest AT t0."""
    v0 = -74.3
    dt, margin = 0.025, 3.0
    t_pre = np.round(np.arange(-100.0, margin + 0.5 * dt, dt), 6)          # shared prefix
    t_fine = np.concatenate([t_pre, np.arange(margin + 2.0, 800.0 + 1e-9, 2.0)])
    t_fine_sham = np.concatenate([t_pre, np.arange(margin + 2.5, 800.0 + 1e-9, 2.5)])
    t_grid = np.round(np.arange(-100.5, 800.0 + 1e-9, 0.5), 6)

    def dv(t):
        t = np.asarray(t, float)
        out = np.zeros_like(t)
        post = t >= 0
        u = t[post]
        if peak_after_t0:
            fast = sign * (2.0 * np.exp(-u / 0.35) - 0.8 * np.exp(-u / 0.05))
        else:
            fast = sign * 1.3 * np.exp(-u / 0.30)
        bump = TRUE["Ab"] * (np.exp(-u / TRUE["tau_d"]) - np.exp(-u / TRUE["tau_r"]))
        out[post] = fast + bump
        f0 = float(sign * (1.2 if peak_after_t0 else 1.3))
        p1 = (t >= -0.5) & (t < -0.25)                 # a smooth in-pulse swing
        out[p1] = -sign * 2.5 * np.sin(np.pi * (t[p1] + 0.5) / 0.25) ** 2
        p2 = (t >= -0.25) & (t < 0)
        out[p2] = f0 * (t[p2] + 0.25) / 0.25
        return out

    rec = dict(t_fine=t_fine, v_fine=v0 + dv(t_fine),
               t_fine_sham=t_fine_sham, v_fine_sham=np.full(t_fine_sham.shape, v0),
               t_dv_fine=t_pre, dv_fine=dv(t_pre),
               t_grid=t_grid, v_grid=v0 + dv(t_grid),
               v_grid_sham=np.full(t_grid.shape, v0), dv=dv(t_grid))
    meta = dict(morph="synthetic", layer_um=80.0, cell_model="full_tuned", v_rest_mV=v0,
                x_um=-110.0, y_um=-110.0, theta_deg=270.0, r_um=155.6, fired=0,
                i0_uA=50.0, phase_dur_ms=0.25, ramp_us=100.0, interphase_us=0.0,
                anodic_first=True, dt_ms=dt, bump_ms=800.0, bump_dt_ms=0.5,
                play_margin_ms=margin, baseline_ms=100.0, early_floor=0.25, t0_ms=0.0,
                created="smoke")
    return rec, meta


tmp = tempfile.mkdtemp(prefix="pfr_")

# ---------------------------------------------------------------------------------------------
section("[1] save_record / load_record")
rec, meta = synthetic()
path = PF.save_record(os.path.join(tmp, "sub", "rec.npz"), rec, meta)
rec2, meta2 = PF.load_record(path)
check(all(np.array_equal(rec[k], rec2[k]) for k in PF.TRACE_KEYS), "arrays identical")
check(meta2 == meta, "metadata identical")

# ---------------------------------------------------------------------------------------------
section("[2] the fits recover the generating components")
q = PF.readouts(rec2, meta2)
v, b, e = q["values"], q["bump"], q["early"]
t_pk_true = TRUE["tau_r"] * TRUE["tau_d"] / (TRUE["tau_d"] - TRUE["tau_r"]) * np.log(
    TRUE["tau_d"] / TRUE["tau_r"])
pk_true = TRUE["Ab"] * (np.exp(-t_pk_true / TRUE["tau_d"]) - np.exp(-t_pk_true / TRUE["tau_r"]))
check(v["dexp_fit_ok"] == 1 and v["early_fit_ok"] == 1, "both fits accepted")
check(abs(v["dexp_tau_rise_ms"] / TRUE["tau_r"] - 1) < 0.03,
      "tau_r %.2f vs %.1f ms" % (v["dexp_tau_rise_ms"], TRUE["tau_r"]))
check(abs(v["dexp_tau_decay_ms"] / TRUE["tau_d"] - 1) < 0.03,
      "tau_d %.2f vs %.1f ms" % (v["dexp_tau_decay_ms"], TRUE["tau_d"]))
check(abs(v["dexp_peak_mV"] / pk_true - 1) < 0.02,
      "bump peak %.4f vs %.4f mV" % (v["dexp_peak_mV"], pk_true))
check(0.2 < v["early_tau_ms"] < 0.6 and np.isfinite(v["early_t_1e_ms"]),
      "fast relaxation: tau_m %.3f ms, t_1e %.3f ms" % (v["early_tau_ms"], v["early_t_1e_ms"]))
check(v["early_t_peak_ms"] > 0, "the fast peak is found AFTER t0 (%.3f ms)" % v["early_t_peak_ms"])

# ---------------------------------------------------------------------------------------------
section("[3] the drawn geometry")
from bump_kinetics import eval_bump_component                            # noqa: E402
cu = q["C_term_u"]
ref = eval_bump_component(b["t_fit_lo_ms"] + cu, b)
check(np.nanmax(np.abs(q["C_term_d"] + q["C_term_r"] - ref)) < 1e-12,
      "C: the two exponential terms sum to the fitted bump")
check(q["C_u"][0] == 0.0 and abs(q["C_fit"][0]) < 1e-12 and abs(q["C_data"][0]) < 1e-12,
      "C: fit and bump data both start at exactly 0 at t0 (the pinning)")
check(q["C_end"] == (0.0, v["deltaVm_end_phase2_mV"]),
      "C: the lower square is DeltaV_end (%.4f mV), the trace's value at t0" % q["C_end"][1])
for key, curve in (("C_tau_d", q["C_term_d"]), ("C_tau_r", q["C_term_r"])):
    tx, ty = q[key]
    on = float(np.interp(tx, cu, curve))
    check(abs(on - ty) < 1e-3 * abs(b["amp_mV"]) and abs(abs(ty) - abs(b["amp_mV"]) / np.e) < 1e-12,
          "%s marker on its term at 1/e of A_b (%.4f vs %.4f)" % (key, on, ty))
check(q["C_tau_r"][0] < q["C_peak"][0] < q["C_tau_d"][0],
      "tau_r < t_b < tau_d (%.1f < %.1f < %.1f ms)"
      % (q["C_tau_r"][0], q["C_peak"][0], q["C_tau_d"][0]))
ta, tb, ye = q["B_t1e"]
on = float(np.interp(tb, rec["t_dv_fine"], rec["dv_fine"]))
check(abs(ye - v["early_peak_mV"] / np.e) < 1e-12 and abs(on - ye) < 0.02 * abs(v["early_peak_mV"]),
      "t_1/e point on the data at |peak|/e (%.4f vs %.4f mV)" % (on, ye))
late = q["A_tm"] >= 5.0
va = np.interp(q["A_tm"][late], q["A_t"], q["A_v"])
rms = float(np.sqrt(np.mean((q["A_vm"][late] - va) ** 2)))
check(rms < 0.01 * abs(pk_true), "reconstruction matches the trace after 5 ms (RMS %.2e mV)" % rms)
t_a, y_a, y_b = q["A_dv_arrow"]
check(abs((y_b - y_a) - v["dexp_data_peak_mV"]) < 1e-6,
      "DeltaV arrow spans the measured bump (%.4f mV)" % (y_b - y_a))
cur = q["B_I"]
charge = float(np.trapezoid(cur, q["B_It"]) if hasattr(np, "trapezoid") else np.trapz(cur, q["B_It"]))
check(abs(charge) < 1e-3 * np.max(np.abs(cur)) * 0.5 and abs(np.max(np.abs(cur)) - 50.0) < 1e-9,
      "stimulus: +/-50 uA, charge-balanced (net %.2e uA ms)" % charge)

# ---------------------------------------------------------------------------------------------
section("[4] the files")
stem = os.path.join(tmp, "out", "fig")
paths, over, outside = PF.render(q, stem, verbose=False)
for ext in ("pdf", "svg", "png"):
    p = "%s.%s" % (stem, ext)
    check(os.path.isfile(p) and os.path.getsize(p) > 5000, "%s written (%d bytes)"
          % (os.path.basename(p), os.path.getsize(p) if os.path.isfile(p) else 0))
pdf = open(stem + ".pdf", "rb").read()
check(b"/FontFile2" in pdf and b"/Type3" not in pdf, "PDF: TrueType fonts embedded, no Type 3")
svg = open(stem + ".svg", encoding="utf-8").read()
check("<text" in svg, "SVG: text kept as text (editable)")
from matplotlib import image as mpimg                                     # noqa: E402
w_px = mpimg.imread(stem + ".png").shape[1]
check(abs(w_px - round(180.0 / 25.4 * 600)) <= 1, "PNG %d px wide = 180 mm at 600 dpi" % w_px)
vt = open(stem + "_values.txt").read()
check("dexp_tau_decay_ms" in vt and "early_t_1e_ms" in vt, "values file lists the readouts")

# ---------------------------------------------------------------------------------------------
section("[5] layout: no label collisions, nothing off the page")
check(not over and not outside, "hyperpolarising, peak after t0: %d overlaps, %d outside"
      % (len(over), len(outside)))
rec_p, meta_p = synthetic(sign=1.0, peak_after_t0=False)
q_p = PF.readouts(rec_p, meta_p)
check(abs(q_p["values"]["early_t_peak_ms"]) < 1e-9, "second case: the fast peak IS at t0")
with plt.rc_context(PF.paper_rc()):
    fig, axd = PF.make_figure(q_p)
    over_p, outside_p = PF.layout_problems(fig)
check(not over_p and not outside_p, "depolarising, peak at t0: %d overlaps, %d outside %s"
      % (len(over_p), len(outside_p), over_p + outside_p))

# a record with NO bump: the double exponential has nothing to fit -- the figure must still
# render (it would be a poor example, but a wrong neuron must not crash a cluster job)
TRUE_SAVED = dict(TRUE)
TRUE["Ab"] = 0.0
rec_n, meta_n = synthetic()
TRUE.update(TRUE_SAVED)
try:
    q_n = PF.readouts(rec_n, meta_n)
    with plt.rc_context(PF.paper_rc()):
        fig_n, _ = PF.make_figure(q_n)
        fig_n.canvas.draw()
    plt.close(fig_n)
    check(True, "no bump at all: renders (bump fit_ok %d)" % q_n["values"]["dexp_fit_ok"])
except Exception as exc:                                   # noqa: BLE001
    check(False, "no bump at all: raised %s: %s" % (type(exc).__name__, exc))

# ---------------------------------------------------------------------------------------------
section("[6] scale bars are as long as their labels say")
n_bars = 0
ok_all = True
for ax in fig.axes:
    for ln in ax.lines:
        gid = ln.get_gid() or ""
        if not gid.startswith("scalebar_"):
            continue
        n_bars += 1
        val = float(gid.split(":")[1])
        xs, ys = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        length = abs(xs[1] - xs[0]) if gid.startswith("scalebar_x") else abs(ys[1] - ys[0])
        labels = [t.get_text().replace(chr(0x2212), "-") for t in ax.texts]
        has_label = any(t.startswith("%g " % val) for t in labels)
        if not (abs(length - val) < 1e-9 * max(1.0, val) and has_label):
            ok_all = False
            print("     bar %s: length %.6g, labels %s" % (gid, length, labels))
check(n_bars >= 7 and ok_all, "%d scale bars, each exactly its labelled length" % n_bars)
vp_ = q_p["values"]
want = {"bracket:t_1e": (vp_["early_t_peak_ms"], vp_["early_t_peak_ms"] + vp_["early_t_1e_ms"]),
        "bracket:t_b": (0.0, vp_["dexp_t_peak_ms"] - q_p["bump"]["t_fit_lo_ms"]),
        "bracket:tau_d": (0.0, vp_["dexp_tau_decay_ms"]),
        "bracket:tau_r": (0.0, vp_["dexp_tau_rise_ms"])}
got = {}
for ax in fig.axes:
    for art in ax.texts:
        gid = art.get_gid() or ""
        if gid.startswith("bracket:"):
            got[gid] = (float(art.xyann[0]), float(art.xy[0]), float(art.xyann[1]), float(art.xy[1]))
for gid, (x0, x1) in want.items():
    g = got.get(gid)
    check(g is not None and abs(g[0] - x0) < 1e-9 and abs(g[1] - x1) < 1e-9 and g[2] == g[3],
          "%s spans %.3f -> %.3f ms, horizontal%s" % (gid, x0, x1, "" if g else " (MISSING)"))
plt.close(fig)

# ---------------------------------------------------------------------------------------------
section("[7] example ranking and the CLI")
ex_csv = os.path.join(tmp, "campaign_examples.csv")
cols = ["example", "morph", "layer_um", "x_um", "y_um", "r_um", "theta_deg", "i0_uA", "fired",
        "dexp_fit_ok", "early_fit_ok", "dexp_peak_mV", "dexp_r2", "early_peak_mV",
        "early_t_peak_ms", "dexp_dv_t0_mV", "dexp_t_peak_ms"]
rows = [
    [1, "60308", 80, 1, 2, 3, 10, 50, 0, 1, 1, 0.8, 0.999, -1.0, 0.10, -0.7, 95],   # good
    [2, "60308", 80, 1, 2, 3, 10, 50, 0, 1, 1, 1.5, 0.999, -1.2, 0.10, -0.9, 95],   # best
    [3, "60308", 80, 1, 2, 3, 10, 50, 0, 0, 1, 3.0, 0.700, -1.0, 0.10, -0.7, 95],   # bump rejected
    [4, "60308", 80, 1, 2, 3, 10, 50, 1, 1, 1, 2.0, 0.999, -1.0, 0.10, -0.7, 95],   # fired
    [5, "60308", 80, 1, 2, 3, 10, 50, 0, 1, 1, 1.5, 0.999, -1.2, 0.00, -1.2, 95],   # peak at t0
]
with open(ex_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    w.writerows(rows)
ranked = PF.suggest_examples(ex_csv)
order = [ex for _, ex, _ in ranked]
# scores: #2 1.5, #5 1.5 x 0.6 = 0.9 (peak at t0), #1 0.8 -- all at the same r2
check(order == [2, 5, 1],
      "ranking %s: rejected fit and firing cell excluded; peak at t0 ranks below the same "
      "bump with a distinct peak" % order)
pl = PF.placement_from_examples_csv(ex_csv, 2)
check(pl["morph"] == "60308" and pl["theta_deg"] == 10.0 and pl["i0_uA"] == 50.0,
      "placement read back from example 2")
check(PF._cli(["suggest", "--from-examples-csv", ex_csv]) == 0, "CLI: suggest")
check(PF._cli(["plot", "--data", path, "--out-stem", os.path.join(tmp, "cli", "fig")]) == 0
      and os.path.isfile(os.path.join(tmp, "cli", "fig.pdf")), "CLI: plot from the .npz")

# ---------------------------------------------------------------------------------------------
section("[8] ASCII source")
src = open(os.path.join(HERE, "paper_fig_readouts.py"), "rb").read()
check(all(c < 128 for c in src) and b"\r" not in src, "paper_fig_readouts.py: pure ASCII, LF only")

print("\n%s: %d failure(s)" % ("FAIL" if FAILS else "PASS", len(FAILS)))
for f in FAILS:
    print("  - " + f)
sys.exit(1 if FAILS else 0)
