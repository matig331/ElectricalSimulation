"""smoke_test_campaign_examples.py -- offline checks of plot_vm_examples --from-csv.

No NEURON: exercises the parts that decide WHICH neuron is drawn and WHERE it is placed,
which is where a silent error would put the wrong cell on a page with the right label.

  [1] placement recovery: rows written from culture_draws are regenerated exactly,
      including the raw rotation theta that the row does not store
  [2] a corrupted row (x off by 1 um, wrong morphology, wrong orientation) is SKIPPED,
      never re-placed
  [3] renumbered merged files: 'culture' renumbered, 'local_culture' kept -> still recovered
  [4] N inference from a truncated culture; a seed whose EVERY culture is truncated may be
      unrecoverable by inference, but no recovered row is ever wrong -- and passing N fixes it
  [5] row_fit_state: accepted / rejected / unmeasured
  [6] select_examples: counts, pools, distance spread, top-up, sort order, every mode
  [7] campaign_protocol: amplitude from the rows, mixed amplitudes refused, window default
  [8] load_campaign_rows: file, results directory, parts directory, missing column refused
  [9] campaign_check: identical -> agrees; DeltaV_end or tau off -> differs; unmeasured row
 [10] plot_gallery renders one page per `per_page` tiles
 [11] CLI: --from-csv with --fit joint is refused before anything is simulated

Run:   python smoke_test_campaign_examples.py          (~5 s)
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

import plot_vm_examples as P                                  # noqa: E402
from config import CFG                                        # noqa: E402
from culture_export import CSV_HEADER, culture_draws          # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def make_rows(seed, cultures, N, keep=None, rng_seed=0):
    """Rows exactly as build_row would round them, for the placements culture_draws makes.

    Kinetics columns are synthetic: accepted / rejected / unmeasured in a fixed pattern.
    Returns (rows, truth) with truth[(seed, c, i)] = raw theta.
    """
    ctx = P.draw_context(CFG)
    rng = np.random.default_rng(rng_seed)
    rows, truth = [], {}
    for c in cultures:
        d = culture_draws(seed, c, N, len(ctx["morphs"]), ctx["span"], ctx["elec"],
                          ctx["center"], ctx["dip_c"], ctx["dip_d"], ctx["axis"],
                          ctx["h_soma"])
        for i in range(N if keep is None else keep.get(c, N)):
            r = {k: "" for k in CSV_HEADER}
            r.update(culture=str(c), neuron=str(i), morphology=ctx["morphs"][int(d["midx"][i])],
                     layer_um="40", x_um=str(round(float(d["pos"][i, 0]), 2)),
                     y_um=str(round(float(d["pos"][i, 1]), 2)),
                     dist_dipole3d_um=str(round(float(d["r_dip"][i]), 2)),
                     theta_orient_deg=str(round(float(d["th_or"][i]), 1)),
                     i0_uA="50.0", seed=str(seed), cell_model="full_tuned",
                     deltaVm_end_phase2_mV=str(round(float(rng.normal(0, 0.5)), 6)))
            state = ("accepted", "rejected", "unmeasured")[(i + c) % 3]
            r["dexp_fit_ok"] = "1" if state == "accepted" else "0"
            if state != "unmeasured":
                r["dexp_dv_t0_mV"] = "0.1"
                r["dexp_data_peak_mV"] = str(round(float(rng.uniform(0, 0.3)), 6))
                r["dexp_peak_mV"] = r["dexp_data_peak_mV"]
                r["dexp_tau_decay_ms"] = "120.0"
            rows.append(r)
            truth[(seed, c, i)] = float(d["theta"][i])
            truth[("pos", seed, c, i)] = (float(d["pos"][i, 0]), float(d["pos"][i, 1]))
    return rows, truth


def section(t):
    print("\n" + t)


# ---------------------------------------------------------------------------------------------
section("[1] placement recovery")
rows, truth = make_rows(2000, [0, 1, 2], 20)
places, skipped = P.recover_placements(rows, CFG)
check(len(places) == 60 and not skipped, "60 / 60 rows recovered, 0 skipped (%d, %d)"
      % (len(places), len(skipped)))
err = max(abs(p["theta"] - truth[(2000, int(p["row"]["culture"]), int(p["row"]["neuron"]))])
          for p in places)
check(err == 0.0, "raw rotation theta regenerated bit for bit (max err %.3g deg)" % err)
check(all((p["x"], p["y"]) == truth[("pos", 2000, int(p["row"]["culture"]),
                                       int(p["row"]["neuron"]))] for p in places),
      "x/y are the exact draws, not the row's 0.01 um rounding")
check(all(abs(p["x"] - float(p["row"]["x_um"])) <= 0.005 + 1e-9 for p in places),
      "... and agree with the row's x to its rounding")

# ---------------------------------------------------------------------------------------------
section("[2] corrupted rows are skipped, not re-placed")
bad = [dict(r) for r in rows[:3]]
bad[0]["x_um"] = str(float(bad[0]["x_um"]) + 1.0)
bad[1]["morphology"] = [m for m in CFG.morphologies if m != bad[1]["morphology"]][0]
bad[2]["theta_orient_deg"] = str(float(bad[2]["theta_orient_deg"]) + 0.5)
pl2, sk2 = P.recover_placements(bad + rows[3:], CFG)
check(len(sk2) == 3 and len(pl2) == 57, "3 corrupted rows skipped, 57 recovered (%d, %d)"
      % (len(sk2), len(pl2)))

# ---------------------------------------------------------------------------------------------
section("[3] merged file with renumbered culture, local_culture kept")
ren = []
for r in rows:
    q = dict(r)
    q["local_culture"] = q["culture"]
    q["culture"] = str(int(q["culture"]) + 100)
    ren.append(q)
pl3, sk3 = P.recover_placements(ren, CFG)
check(len(pl3) == 60 and not sk3, "all recovered through local_culture (%d)" % len(pl3))

# ---------------------------------------------------------------------------------------------
section("[4] truncated cultures")
rows4, truth4 = make_rows(3000, [0, 1], 20, keep={1: 7})     # culture 1 cut after 7 neurons
pl4, sk4 = P.recover_placements(rows4, CFG)
check(len(pl4) == 27 and not sk4, "N taken from the complete culture: 27 / 27 (%d)" % len(pl4))
rows5, truth5 = make_rows(3100, [0, 1], 20, keep={0: 7, 1: 5})  # EVERY culture truncated
pl5, sk5 = P.recover_placements(rows5, CFG)
wrong = [p for p in pl5 if p["theta"] != truth5[(3100, int(p["row"]["culture"]),
                                                  int(p["row"]["neuron"]))]]
check(not wrong, "all cultures truncated, N inferred: %d recovered, %d skipped, 0 wrong "
      "(%d wrong)" % (len(pl5), len(sk5), len(wrong)))
pl5b, sk5b = P.recover_placements(rows5, CFG, n_per_culture=20)
check(len(pl5b) == 12 and not sk5b, "... with n_per_culture=20 given: 12 / 12 (%d)" % len(pl5b))

# ---------------------------------------------------------------------------------------------
section("[5] row_fit_state")
check(P.row_fit_state({"dexp_fit_ok": "1", "dexp_dv_t0_mV": "0.1"}) == "accepted", "accepted")
check(P.row_fit_state({"dexp_fit_ok": "0", "dexp_dv_t0_mV": "0.1"}) == "rejected", "rejected")
check(P.row_fit_state({"dexp_fit_ok": "0", "dexp_dv_t0_mV": ""}) == "unmeasured", "unmeasured")
check(P.row_fit_state({}) == "unmeasured", "legacy row without kinetics -> unmeasured")

# ---------------------------------------------------------------------------------------------
section("[6] select_examples")
states = [P.row_fit_state(p["row"]) for p in places]
n_ok, n_rej = states.count("accepted"), states.count("rejected")
print("  pools: %d accepted, %d rejected, %d unmeasured" % (n_ok, n_rej,
                                                            states.count("unmeasured")))
dist = lambda p: float(p["row"]["dist_dipole3d_um"])
sel = P.select_examples(places, 10, "stratified")
st = [P.row_fit_state(p["row"]) for p in sel]
check(len(sel) == 10, "10 selected (%d)" % len(sel))
check(st.count("accepted") == 6 and st.count("rejected") == 4,
      "6 accepted + 4 rejected (%d + %d)" % (st.count("accepted"), st.count("rejected")))
check("unmeasured" not in st, "no unmeasured row in the stratified pick")
check(len(set(id(p) for p in sel)) == 10, "no duplicates")
check([dist(p) for p in sel] == sorted(dist(p) for p in sel), "sorted by distance")
ok_d = np.array(sorted(dist(p) for p in places if P.row_fit_state(p["row"]) == "accepted"))
sel_ok_d = [dist(p) for p in sel if P.row_fit_state(p["row"]) == "accepted"]
check(min(sel_ok_d) <= np.percentile(ok_d, 20) and max(sel_ok_d) >= np.percentile(ok_d, 80),
      "accepted picks span the distance range (%.0f..%.0f of %.0f..%.0f um)"
      % (min(sel_ok_d), max(sel_ok_d), ok_d[0], ok_d[-1]))
rej_meas = sorted((abs(float(p["row"]["dexp_data_peak_mV"])) for p in places
                   if P.row_fit_state(p["row"]) == "rejected"), reverse=True)
sel_rej = sorted((abs(float(p["row"]["dexp_data_peak_mV"])) for p in sel
                  if P.row_fit_state(p["row"]) == "rejected"), reverse=True)
check(sel_rej == rej_meas[:4], "rejected picks are the 4 largest measured |bump|")
# n = 36 asks for round(0.6 * 36) = 22 accepted but only 20 exist -> 20 + 16
big = P.select_examples(places, 36, "stratified")
bst = [P.row_fit_state(p["row"]) for p in big]
check(len(big) == 36 and bst.count("accepted") == n_ok and bst.count("rejected") == 16,
      "accepted pool exhausted -> topped up with rejected (%d accepted + %d rejected)"
      % (bst.count("accepted"), bst.count("rejected")))
allp = P.select_examples(places, 500, "stratified")
check(len(allp) == n_ok + n_rej, "n > pool -> every measured row once (%d)" % len(allp))
check(all(P.row_fit_state(p["row"]) == "accepted"
          for p in P.select_examples(places, 5, "accepted")), "mode accepted")
check(all(P.row_fit_state(p["row"]) == "rejected"
          for p in P.select_examples(places, 5, "rejected")), "mode rejected")
nr = P.select_examples(places, 5, "nearest")
check([dist(p) for p in nr] == sorted(dist(p) for p in places)[:5], "mode nearest")
r1 = P.select_examples(places, 5, "random", seed=1)
r2 = P.select_examples(places, 5, "random", seed=1)
check([id(p) for p in r1] == [id(p) for p in r2] and len(r1) == 5, "mode random is seeded")
unm = [p for p in places if P.row_fit_state(p["row"]) == "unmeasured"]
check(len(P.select_examples(unm, 4, "stratified")) == 4,
      "nothing measured -> stratified spreads over every row")
try:
    P.select_examples(places, 3, "bogus")
    check(False, "unknown mode refused")
except ValueError:
    check(True, "unknown mode refused")

# ---------------------------------------------------------------------------------------------
section("[7] campaign_protocol")
pr = P.campaign_protocol(CFG, rows, bump_ms=300.0)
check(pr["i0_uA"] == 50.0 and pr["bump_ms"] == 300.0, "i0 from rows, window as passed")
check(pr["play_margin_ms"] == float(CFG.play_margin_ms)
      and pr["bump_dt_ms"] == float(CFG.bump_dt_ms)
      and pr["cvode_atol"] == float(CFG.cvode_atol), "grid / margin / atol from config")
check(P.campaign_protocol(CFG, rows)["bump_ms"] == float(CFG.bump_ms),
      "window defaults to config.bump_ms (%.0f)" % CFG.bump_ms)
mixed = [dict(r) for r in rows[:2]]
mixed[1]["i0_uA"] = "30.0"
try:
    P.campaign_protocol(CFG, mixed)
    check(False, "mixed amplitudes refused")
except SystemExit:
    check(True, "mixed amplitudes refused")

# ---------------------------------------------------------------------------------------------
section("[8] load_campaign_rows")
tmp = tempfile.mkdtemp(prefix="sce_")
path = os.path.join(tmp, "culture_Pactivation.csv")
with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in CSV_HEADER})
src, back = P.load_campaign_rows(path)
check(len(back) == 60, "file: 60 rows")
src2, back2 = P.load_campaign_rows(tmp)
check(src2 == path and len(back2) == 60, "directory -> finds *Pactivation*.csv")
pl8, _ = P.recover_placements(back, CFG)
check(len(pl8) == 60, "rows read back from disk still recover 60 / 60 (%d)" % len(pl8))
pdir = tempfile.mkdtemp(prefix="sce_parts_")
for k, chunk in enumerate((rows[:25], rows[25:])):
    with open(os.path.join(pdir, "part_%02d.csv" % k), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in chunk:
            w.writerow({k2: r[k2] for k2 in CSV_HEADER})
src3, back3 = P.load_campaign_rows(pdir)
check(len(back3) == 60 and "2 part files" in src3, "parts directory -> concatenates part_*.csv")
badp = os.path.join(tmp, "x_Pactivation_bad.csv")
with open(badp, "w", newline="") as fh:
    fh.write("culture,neuron\n0,1\n")
try:
    P.load_campaign_rows(badp)
    check(False, "file without placement columns refused")
except SystemExit:
    check(True, "file without placement columns refused")

# ---------------------------------------------------------------------------------------------
section("[9] campaign_check")
t = np.arange(-1.0, 3.0, 0.025)
t[np.argmin(np.abs(t))] = 0.0
dv = np.where(t < 0, 1.0, np.exp(-np.clip(t, 0, None) / 0.3)) * 0.5
rec = dict(t_dv_fine=t, dv_fine=dv)
bump = dict(fit_ok=1, peak_mV=0.2, tau_decay_ms=120.0)
row = dict(deltaVm_end_phase2_mV="0.5", dexp_fit_ok="1", dexp_dv_t0_mV="0.1",
           dexp_peak_mV="0.2", dexp_tau_decay_ms="120.0")
c1 = P.campaign_check(rec, {"bump": bump}, row)
check(c1["agrees"] == 1 and c1["d_dv_end"] < 1e-12, "identical -> agrees")
c2 = P.campaign_check(rec, {"bump": bump}, dict(row, deltaVm_end_phase2_mV="0.51"))
check(c2["agrees"] == 0, "DeltaV_end off by 0.01 mV -> differs")
c3 = P.campaign_check(rec, {"bump": bump}, dict(row, dexp_tau_decay_ms="150.0"))
check(c3["agrees"] == 0, "tau_decay off by 25 % -> differs")
c4 = P.campaign_check(rec, {"bump": bump}, dict(row, dexp_fit_ok="0"))
check(c4["agrees"] == 0, "accepted here, rejected in the campaign -> differs")
c5 = P.campaign_check(rec, {"bump": bump}, dict(row, dexp_fit_ok="0", dexp_dv_t0_mV=""))
check(c5["agrees"] == 1 and c5["row_state"] == "unmeasured",
      "unmeasured row -> only DeltaV_end compared")
c6 = P.campaign_check(rec, {"bump": bump}, dict(row, deltaVm_end_phase2_mV=""))
check(c6["agrees"] == 0, "missing DeltaV_end -> not counted as agreeing")

# ---------------------------------------------------------------------------------------------
section("[10] plot_gallery")
proto = dict(bump_ms=200.0, phase_dur_ms=0.2)
tg = np.arange(0.0, 200.0, 0.5)
items = []
for k in range(14):
    ok = k % 2 == 0
    shape = 0.2 * (np.exp(-tg / 100.0) - np.exp(-tg / 20.0))
    f = dict(bump=dict(fit_ok=int(ok), peak_mV=0.1, t_peak_ms=40.0, tau_rise_ms=20.0,
                       tau_decay_ms=100.0, r2=0.99, data_peak_mV=0.1),
             model_grid=shape)
    items.append(dict(idx=k + 1, morph="60303", layer=40.0,
                      rec=dict(t_grid=tg, dv=shape + 0.002 * np.sin(tg)), fit=f,
                      row=dict(dist_dipole3d_um="123.4"), check=dict(agrees=int(k != 3))))
figs = P.plot_gallery(items, proto, per_page=12, title="smoke")
check(len(figs) == 2, "14 tiles at 12 per page -> 2 pages (%d)" % len(figs))
out_pdf = os.path.join(tmp, "gallery.pdf")
from matplotlib.backends.backend_pdf import PdfPages      # noqa: E402
with PdfPages(out_pdf) as pdf:
    for f in figs:
        pdf.savefig(f)
check(os.path.getsize(out_pdf) > 10000, "gallery PDF written (%d bytes)" % os.path.getsize(out_pdf))

# ---------------------------------------------------------------------------------------------
section("[11] CLI guard")
try:
    P._cli(["--from-csv", path, "--fit", "joint"])
    check(False, "--fit joint with --from-csv refused")
except SystemExit as e:
    check(e.code == 2, "--fit joint with --from-csv refused before simulating (exit %s)" % e.code)

print("\n%s: %d failure(s)" % ("FAIL" if FAILS else "PASS", len(FAILS)))
for f in FAILS:
    print("  - " + f)
sys.exit(1 if FAILS else 0)
