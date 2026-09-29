"""smoke_test_protocol.py -- NEURON checks of the single-stimulation protocol and the soma
square (config.pre_stim_ms, bump_ms, placement_half_um, neurons_per_culture).

    python smoke_test_protocol.py            # ~2 min, one cell; ends with ALL PASSED
    python smoke_test_protocol.py --worker   # + one culture through culture_worker (~8 min)

  1  rich_footprint.spikes_at with pre_settle_ms = 0 is bit-identical to the function before the
     change (kept verbatim below): short and long window, detail on and off, one phase only
  2  the rest before the pulse is a steady state: the sham's soma moves < 1e-4 mV over it and
     does not spike
  3  the variable-step part of the rest changes nothing: DeltaV_end, the direct relaxation and
     the bump fit equal those of the same rest integrated entirely at the fixed dt
  4  one pulse: DeltaV is exactly 0 before the pulse; the output grid holds a sample AT the end
     of phase 2 and ends bump_ms after it; V at the end of phase 2 is bit-identical with and
     without the window after the pulse (the outcome columns never depend on it)
  5  cost of the campaign protocol against the earlier one on the same cell
  6  (--worker) culture_worker end to end: every soma inside the square, n_pulses = 1, kinetics
     measured on every row, culture_merge accepts the part
Exit status 1 on any failure.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from neuron import h                                             # noqa: E402
import field as F                                                # noqa: E402
import rich_footprint as RF                                      # noqa: E402
from rich_footprint import _placed, V_REST                       # noqa: E402,F401
from config import CFG                                           # noqa: E402
import culture_export as CE                                      # noqa: E402
from bump_kinetics import common_prefix_len, fit_staged          # noqa: E402

FAILS = []
MORPH, LAYER = "60308", 80.0
PLACES = [((40.0, -20.0), 30.0), ((110.0, 60.0), 200.0), ((-150.0, -120.0), 75.0),
          ((25.0, -70.0), 300.0)]


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def same(a, b):
    if isinstance(a, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _old_spikes_at(cell, pos_xy, theta_deg, i0_uA=50.0, phase_dur_ms=0.25,
              baseline_ms=5.0, post_ms=6.0, dt_ms=0.025,
              sigma_Sm=1.5, rmin_um=12.5, ramp_us=100.0, interphase_us=0.0, phase="both",
              detail=False, pre_end_ms=0.0, v_init_mV=V_REST,
              bump_ms=0.0, bump_dt_ms=0.5, cvode_atol=1e-6, play_margin_ms=1.0):
    """rich_footprint.spikes_at as it was before pre_settle_ms (git a7643f9), verbatim."""
    coords, refs = _placed(cell, pos_xy, theta_deg)
    elec, sign = F.default_array(monopolar=False)
    g = F.geom_factor(coords, elec, sign, sigma_Sm=sigma_Sm, rmin_um=rmin_um)
    t_on = baseline_ms
    post = post_ms      # detail=True must NOT lengthen the run (HPC cost)
    bump_ms = float(bump_ms or 0.0)
    t_end = t_on + 2 * phase_dur_ms                # END of the biphasic pulse (after phase 2)
    t_stop = t_end + (bump_ms if bump_ms > 0 else post)
    # play grid: the whole window in the classic mode, the pulse only in bump mode
    t = np.arange(0.0, (t_end + play_margin_ms if bump_ms > 0 else t_stop) + dt_ms, dt_ms)
    I = F.biphasic_current(t - t_on, i0_uA, phase_dur_ms, True, ramp_us, interphase_us)  # ramped
    if phase != "both":                                    # deliver only phase 1 (+) or phase 2 (-)
        I = I * F.phase_mask(t, t_on, phase_dur_ms, phase, interphase_us)
    tvec = h.Vector(t); keep = [tvec]
    for sec in set(s.sec for s in refs):
        if not h.ismembrane("extracellular", sec=sec):
            sec.insert("extracellular")
    for k, seg in enumerate(refs):
        vv = h.Vector(g[k] * I); vv.play(seg._ref_e_extracellular, tvec, True); keep.append(vv)
    vs = h.Vector().record(cell.soma[0](0.5)._ref_v)
    ts = h.Vector().record(h._ref_t) if (bump_ms > 0 and detail) else None
    if bump_ms > 0:                                # fixed output grid for the bump fit
        vb = h.Vector(); vb.record(cell.soma[0](0.5)._ref_v, bump_dt_ms)
        tb = h.Vector(); tb.record(h._ref_t, bump_dt_ms)
    h.celsius = 37; h.dt = dt_ms; h.tstop = t_stop
    try:
        h.finitialize(float(v_init_mV))
        h.continuerun(t[-1])                       # through the pulse, fixed dt
        if bump_ms > 0:
            for vv in keep[1:]:                    # stimulus is over: drop the play, zero the field
                vv.play_remove()
            for seg in refs:
                seg.e_extracellular = 0.0
            h.cvode_active(1); h.cvode.atol(float(cvode_atol))
            h.continuerun(t_stop)
    finally:
        h.cvode_active(0)                          # never leak the solver mode to the next call
    v = np.asarray(vs)
    nsp = int(np.sum((v[:-1] < 0) & (v[1:] >= 0)))
    if detail:
        t_rec = np.asarray(ts) if bump_ms > 0 else t
        t_start = t_end - max(0.0, float(pre_end_ms))
        m = t_rec >= t_start
        tw = (t_rec[m] - t_end).astype(float)      # t=0 is exactly END of phase 2
        vw = v[m].astype(float)
        if bump_ms > 0:
            # .copy(): np.asarray() on a live NEURON Vector is a VIEW into its buffer, and the
            # next spikes_at() call overwrites it -- silently zeroing an already returned bump
            # trace (caught by smoke_test_ih_bump).
            return (nsp, float(v.max()), tw, vw,
                    np.asarray(tb).copy() - t_end, np.asarray(vb).copy())
        return nsp, float(v.max()), tw, vw
    return nsp, float(v.max())


def protocol_kw(settle, base, bump_ms):
    return dict(bump_ms=bump_ms, bump_dt_ms=float(CFG.bump_dt_ms), cvode_atol=float(CFG.cvode_atol),
                play_margin_ms=float(CFG.play_margin_ms), baseline_ms=base, pre_settle_ms=settle)


def delta(stim, sham):
    """DeltaV_end, the staged fit and the fixed-dt prefix length, as simulate_neuron forms them."""
    tw, vw, tg, vg = stim[2], stim[3], stim[4], stim[5]
    tws, vws, tgs, vgs = sham[2], sham[3], sham[4], sham[5]
    dv_end = CE.v_at_end_of_phase2(tw, vw) - CE.v_at_end_of_phase2(tws, vws)
    k = common_prefix_len(tw, tws)
    dvg = vg - vgs if tgs.shape == tg.shape else vg - np.interp(tg, tgs, vgs)
    fit = fit_staged(tw[:k], vw[:k] - vws[:k], tg, dvg, t0_ms=float(CFG.bump_t0_ms),
                     early_floor=float(CFG.bump_early_floor))
    return dv_end, fit, k


def main():
    base, settle = CE.pre_stim_split(CFG)
    bump = float(CFG.bump_ms)
    print("protocol: %g ms rest (%g variable-step + %g fixed-dt), one pulse, %g ms after"
          % (base + settle, settle, base, bump))
    with CE.CellPool(CFG, "full_tuned") as pool:
        info = pool.info(MORPH, LAYER)
        pool._activate(MORPH, LAYER)
        cell, vr = pool._cell, info["v_rest"]

        print("1  pre_settle_ms = 0 is the earlier function, bit for bit")
        cases = [("short", dict()), ("short detail", dict(detail=True, pre_end_ms=2.0)),
                 ("long detail", dict(detail=True, bump_ms=800.0, bump_dt_ms=0.5,
                                      cvode_atol=1e-6, play_margin_ms=3.0)),
                 ("phase 1 only", dict(phase="p1"))]
        for name, kw in cases:
            ok = True
            for pos, th in PLACES[:2]:
                a = RF.spikes_at(cell, pos, th, i0_uA=50.0, v_init_mV=vr, **kw)
                b = _old_spikes_at(cell, pos, th, i0_uA=50.0, v_init_mV=vr, **kw)
                ok = ok and same(tuple(a), tuple(b))
            check(ok, "%-13s identical on %d placements" % (name, 2))

        print("2  the rest before the pulse is a steady state")
        kw = protocol_kw(settle, base, bump)
        sham = RF.spikes_at(cell, (0.0, 0.0), 0.0, i0_uA=0.0, detail=True, v_init_mV=vr,
                            pre_end_ms=base + settle + 2 * float(CFG.phase_dur_ms), **kw)
        pre = sham[2] < -2 * float(CFG.phase_dur_ms) - 1e-9
        move = float(np.max(np.abs(sham[3][pre] - vr))) if pre.any() else np.nan
        check(sham[0] == 0 and move < 1e-4,
              "sham: no spike, soma within %.1e mV of its rest over the %g ms before the pulse"
              % (move, base + settle))

        print("3  variable-step rest == the same rest at fixed dt")
        kw_fixed = protocol_kw(0.0, base + settle, bump)
        sh_c = RF.spikes_at(cell, (0.0, 0.0), 0.0, i0_uA=0.0, detail=True, v_init_mV=vr, **kw)
        sh_f = RF.spikes_at(cell, (0.0, 0.0), 0.0, i0_uA=0.0, detail=True, v_init_mV=vr,
                            **kw_fixed)
        worst_dv, worst_pk, worst_td, same_ok, t_c, t_f = 0.0, 0.0, 0.0, True, 0.0, 0.0
        for pos, th in PLACES:
            t0 = time.time()
            st_c = RF.spikes_at(cell, pos, th, i0_uA=50.0, detail=True, v_init_mV=vr, **kw)
            t_c += time.time() - t0
            t0 = time.time()
            st_f = RF.spikes_at(cell, pos, th, i0_uA=50.0, detail=True, v_init_mV=vr, **kw_fixed)
            t_f += time.time() - t0
            dc, fc, kc = delta(st_c, sh_c)
            df, ff, kf = delta(st_f, sh_f)
            worst_dv = max(worst_dv, abs(dc - df))
            bc, bf = fc["bump"], ff["bump"]
            same_ok = same_ok and st_c[0] == st_f[0] and bc["fit_ok"] == bf["fit_ok"] and kc == kf
            if bc["fit_ok"] and bf["fit_ok"]:
                worst_pk = max(worst_pk, abs(bc["peak_mV"] - bf["peak_mV"]) / abs(bf["peak_mV"]))
                worst_td = max(worst_td, abs(bc["tau_decay_ms"] - bf["tau_decay_ms"])
                               / bf["tau_decay_ms"])
        check(worst_dv < 1e-5, "DeltaV_end: worst difference %.1e mV over %d placements"
              % (worst_dv, len(PLACES)))
        check(same_ok and worst_pk < 0.005 and worst_td < 0.01,
              "same spikes, same accepted fits; bump peak within %.2f %%, tau_decay within "
              "%.2f %%" % (100 * worst_pk, 100 * worst_td))

        print("4  one pulse, and the grid the bump fit is anchored on")
        st = RF.spikes_at(cell, PLACES[1][0], PLACES[1][1], i0_uA=50.0, detail=True,
                          v_init_mV=vr, pre_end_ms=base + settle + 2 * float(CFG.phase_dur_ms),
                          **kw)
        k = common_prefix_len(st[2], sham[2])      # same steps through the rest, then fixed dt
        before = st[2][:k] < -2 * float(CFG.phase_dur_ms) - 1e-9
        dv_pre = (float(np.max(np.abs(st[3][:k][before] - sham[3][:k][before])))
                  if before.any() else np.nan)
        check(before.sum() > 10 and dv_pre == 0.0,
              "stimulated run and sham take the same %d steps before the pulse; DeltaV there "
              "exactly 0 (max %.1e mV)" % (int(before.sum()), dv_pre))
        tg = st[4]
        check(bool(np.any(np.abs(tg) < 1e-9)) and abs(tg[-1] - bump) < float(CFG.bump_dt_ms),
              "output grid has a sample AT the end of phase 2 and ends at %+.1f ms (bump_ms %g)"
              % (tg[-1], bump))
        same_end = True
        for pos, th in PLACES:
            st_l = RF.spikes_at(cell, pos, th, i0_uA=50.0, detail=True, v_init_mV=vr, **kw)
            st_s = RF.spikes_at(cell, pos, th, i0_uA=50.0, detail=True, v_init_mV=vr,
                                **dict(kw, bump_ms=0.0))
            same_end = same_end and (CE.v_at_end_of_phase2(st_l[2], st_l[3])
                                     == CE.v_at_end_of_phase2(st_s[2], st_s[3])
                                     and st_l[0] >= st_s[0])
        check(same_end, "V(end of phase 2) bit-identical with and without the window after the "
              "pulse (%d placements): the outcome columns do not depend on it" % len(PLACES))
        check(CFG.n_pulses_for_duration() == 1 and int(CFG.n_pulses) == 1,
              "n_pulses written to the rows = 1 (stim_duration_s %g s at %g Hz)"
              % (CFG.stim_duration_s, CFG.stim_freq_hz))

        print("5  cost")
        kw_old = protocol_kw(0.0, 5.0, 800.0)
        t0 = time.time()
        for pos, th in PLACES:
            RF.spikes_at(cell, pos, th, i0_uA=50.0, detail=True, v_init_mV=vr, **kw_old)
        t_o = time.time() - t0
        n = len(PLACES)
        print("        campaign protocol %.2f s/sim | same rest at fixed dt %.2f s/sim | earlier "
              "protocol (5 ms rest, 800 ms) %.2f s/sim -> x%.2f; cluster estimate %.1f s/sim "
              "(3.1 s measured there for the earlier one)"
              % (t_c / n, t_f / n, t_o / n, t_c / t_o, 3.1 * t_c / t_o))
        check(t_c < t_f, "variable-step rest cheaper than the fixed-dt one")

    if "--worker" in sys.argv:
        print("6  culture_worker end to end (18 cells: a few minutes)")
        import csv
        import culture_merge
        import culture_worker
        tmp = tempfile.mkdtemp(prefix="smoke_protocol_")
        try:
            saved = CFG.cell_model
            CFG.cell_model = "full_tuned"
            parts = os.path.join(tmp, "parts_x")
            out = culture_worker.run_worker([0], os.path.join(parts, "part_000.csv"),
                                            neurons_per_culture=4, seed=999, flush_every=2,
                                            quiet=True)
            CFG.cell_model = saved
            with open(out, newline="") as fh:
                rows = list(csv.DictReader(fh))
            elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
            span, cen = CE.placement_frame(CFG, elec, sign)
            xy = np.array([[float(r["x_um"]), float(r["y_um"])] for r in rows])
            check(len(rows) == 4 * len(CFG.layers_um), "%d rows (4 neurons x %d layers)"
                  % (len(rows), len(CFG.layers_um)))
            check(bool(np.all(np.abs(xy - cen) <= span + 0.01)),
                  "every soma inside +/-%g um around (%g, %g)" % (span, cen[0], cen[1]))
            check(all(r["n_pulses"] == "1" for r in rows), "n_pulses = 1 on every row")
            check(all(r["dexp_dv_t0_mV"] != "" for r in rows), "kinetics measured on every row")
            culture_merge.merge(parts, os.path.join(tmp, "merged"), make_figures=False,
                                expect_model="full_tuned")
            check(os.path.exists(os.path.join(tmp, "merged", "culture_Pactivation.csv")),
                  "culture_merge accepts the part")
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
