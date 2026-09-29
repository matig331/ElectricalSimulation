"""smoke_video_frames.py -- offline checks for video_frames.py (no NEURON).

NEURON is replaced by FakePool, whose traces have the SAME STRUCTURE as
rich_footprint.spikes_at(detail=True, bump_ms>0): a fixed-dt window starting PRE_END_MS before the
end of phase 2, then a uniform long-window grid; a sham with a +0.09 mV drift; a toy response
(cathode side depolarised, anode side hyperpolarised, spike when > 12 mV). REAL: culture_export
(culture_draws, v_at_end_of_phase2, phase2_end_outcome, dipole frame), field, config, the parallel
driver, the merge, and both renderers.

Checks
  1  frame schedule: -0.6 .. tmax, 25 us during the pulse, 163 frames for 200 ms
  2  one neuron: DeltaV(t0) = toy value (sham drift removed); pulse frames not frozen; fired_by_t
     switches at the spike; video DeltaV_end = campaign DeltaV_end
  3  a window that does not cover the frames is refused
  4  serial run: one frames file per culture, rows = cultures x neurons x layers x frames, same
     positions as culture_draws, consistency 0, exit code 0
  5  parallel run (2 processes, spawn) writes exactly the same data as the serial run
  6  make_prob_videos reads the output (no 'frozen' stop) and renders; a culture split over two
     files is still ONE culture (seed, culture identity)
  7  make_culture_videos lists and renders one culture with morphologies

Run:  python smoke_video_frames.py        (writes into ./_smoke_video_frames/)
"""
import glob
import os
import shutil
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or next(
    (d for d in (HERE, os.getcwd(), "/mnt/user-data/uploads")
     if os.path.isfile(os.path.join(d, "culture_export.py"))), HERE)
for p in (HERE, UPLOAD_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
OUT = os.path.join(HERE, "_smoke_video_frames")


def _common_prefix_len(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(a.size, b.size)
    neq = np.nonzero(~np.isclose(a[:n], b[:n], rtol=0.0, atol=1e-12))[0]
    return int(neq[0]) if neq.size else int(n)


if "bump_kinetics" not in sys.modules:
    try:
        import bump_kinetics  # noqa: F401   (real one, on the project machine)
    except Exception:
        m = types.ModuleType("bump_kinetics")
        m.STAGED_COLUMNS = (); m.common_prefix_len = _common_prefix_len
        m.empty_staged = lambda: {}; m.fit_staged = lambda *a, **k: {}
        m.staged_row_values = lambda *a, **k: []
        sys.modules["bump_kinetics"] = m
if "morphologies" not in sys.modules:
    try:
        import morphologies  # noqa: F401
    except Exception:
        asc = next(iter(sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.asc")))), "")
        mm = types.ModuleType("morphologies")
        mm.find_one_morphology = lambda name=None: asc
        mm.all_morphologies = lambda: []
        sys.modules["morphologies"] = mm

PHI, DT, MARGIN = 0.25, 0.025, 1.0


def toy_dv(x, y, theta):
    vx, vy = x, y + 30.0
    r = max(np.hypot(vx, vy), 10.0)
    s = 1.0 if vx > 0 else -1.0
    return 180.0 * s * (0.6 + 0.4 * np.cos(np.radians(theta))) / (r / 10.0) ** 1.2


class FakePool(object):
    """Same interface as video_frames.VideoPool.trace -> (out, sham, info)."""

    def __init__(self, tmax_ms=200.0, pre_end_ms=1.0):
        self.tmax, self.pre = float(tmax_ms), float(pre_end_ms)
        self.tw = np.round(np.arange(-self.pre, MARGIN + 1e-9, DT), 6)
        self.tg = np.round(np.arange(MARGIN + 0.5, self.tmax + 1e-9, 0.5), 6)

    def _sham(self):
        drift = lambda t: 0.09 * (t + self.pre + 5.0) / (self.pre + 5.0)
        vw, vg = -74.3 + drift(self.tw), -74.3 + drift(np.minimum(self.tg, 0.0))
        return (0, float(vw.max()), self.tw.copy(), vw, self.tg.copy(), vg)

    def trace(self, morph, layer, pos_xy, theta_deg, i0_uA):
        sh = self._sham()
        dv = toy_dv(pos_xy[0], pos_xy[1], theta_deg)
        shape = np.clip((self.tw + 2 * PHI) / (2 * PHI), 0.0, 1.0)
        shape[self.tw > 0] = np.exp(-self.tw[self.tw > 0] / 0.3)
        vw = sh[3] + dv * shape
        vg = sh[5] + dv * np.exp(-self.tg / 0.3) + 1.5 * (np.exp(-self.tg / 160.0) - np.exp(-self.tg / 22.0))
        nsp = 0
        if dv > 12.0:
            k = int(np.argmin(np.abs(self.tw - 0.3)))
            vw[k:k + 8] = 30.0
            nsp = 1
        k0 = int(np.argmin(np.abs(self.tw)))
        info = dict(v_rest=-74.3, v_sham_end=float(sh[3][k0]))
        return (nsp, float(max(vw.max(), vg.max())), self.tw.copy(), vw, self.tg.copy(), vg), sh, info


def report(tag, cond, msg):
    print(f"[{tag}] {msg}: {'OK' if cond else 'FAIL'}")
    return bool(cond)


def _read_sorted(pattern):
    import pandas as pd
    parts = [pd.read_csv(f) for f in sorted(glob.glob(pattern))]
    d = pd.concat(parts, ignore_index=True)
    return d.sort_values(["culture", "neuron", "layer", "t_ms"]).reset_index(drop=True)


def main():
    import video_frames as V
    ok = True
    shutil.rmtree(OUT, ignore_errors=True)
    os.environ["VIDEO_FRAMES_FAKE"] = "1"

    ft = V.frame_times(200.0)
    ok &= report("1", abs(ft[0] + 0.6) < 1e-9 and ft[-1] == 200.0 and ft.size == 163
                 and np.allclose(np.diff(ft[ft <= 0.4]), 0.025),
                 f"{ft.size} frames, {ft[0]} .. {ft[-1]} ms")

    fp = FakePool(200.0)
    out, sh, info = fp.trace("60308", 80, (60.0, -30.0), 180.0, 50.0)
    r = V.neuron_frames(out, sh, info["v_sham_end"], ft)
    k0 = int(np.argmin(np.abs(ft))); k1 = int(np.argmin(np.abs(ft + 0.375)))
    toy = toy_dv(60.0, -30.0, 180.0)
    c2 = (abs(r["dv"][k0] - toy) < 1e-6 and abs(r["dv"][k1] - r["dv"][k0]) > 0.1
          and abs(r["dv_end"] - r["dv_end_campaign"]) < 1e-9 and r["fired"] == 0)
    out2, sh2, info2 = fp.trace("60308", 80, (30.0, -30.0), 0.0, 50.0)
    r2 = V.neuron_frames(out2, sh2, info2["v_sham_end"], ft)
    fb = r2["fired_by_t"]
    c2 = c2 and r2["fired"] == 1 and fb[0] == 0 and fb[-1] == 1 and np.all(np.diff(fb) >= 0)
    ok &= report("2", c2, f"dV(t0) {r['dv'][k0]:+.3f} = toy {toy:+.3f}; mid phase 1 {r['dv'][k1]:+.3f}; "
                 f"spike at {r2['t_spike']} ms")

    refused = False
    try:
        V.neuron_frames(*FakePool(50.0).trace("60308", 80, (60.0, -30.0), 0.0, 50.0)[:2], -74.3, ft)
    except RuntimeError:
        refused = True
    ok &= report("3", refused, "window shorter than the frames is refused")

    ser = os.path.join(OUT, "serial")
    common = ["--n-cultures", "2", "--neurons", "30", "--span-um", "200", "--layers", "80",
              "--tmax-ms", "20", "--chunk", "10", "--seed", "7"]
    rc = V.main(common + ["--processes", "0", "--out", ser])
    fr = sorted(glob.glob(os.path.join(ser, "culture_frames_S7_C*.csv")))
    d = _read_sorted(os.path.join(ser, "culture_frames_S7_C*.csv"))
    nfr = V.frame_times(20.0).size
    import culture_export as CE
    from config import CFG
    g = V._geometry(CFG)
    dd = CE.culture_draws(7, 1, 30, len(CFG.morphologies), 200.0, g["elec"], g["center"],
                          g["dip"][0], g["dip"][1], g["axis"], CFG.h_soma_um,
                          place_center=g["place_center"])
    row = d[(d.culture == 1) & (d.neuron == 4)].iloc[0]
    import json
    man = json.load(open(os.path.join(ser, "video_manifest_S7.json")))
    ok &= report("4", rc == 0 and len(fr) == 2 and len(d) == 2 * 30 * 1 * nfr
                 and abs(row.x - round(dd["pos"][4, 0], 2)) < 1e-9
                 and abs(row.y - round(dd["pos"][4, 1], 2)) < 1e-9
                 and abs(row.theta_deg - round(dd["theta"][4], 2)) < 1e-9
                 and man["max_abs_dv_end_diff_vs_campaign_mV"] == 0.0,
                 f"{len(fr)} files, {len(d)} rows (= 2x30x1x{nfr}), positions = culture_draws")

    par = os.path.join(OUT, "parallel")
    rc = V.main(common + ["--processes", "2", "--out", par])
    dp = _read_sorted(os.path.join(par, "culture_frames_S7_C*.csv"))
    ok &= report("5", rc == 0 and dp.equals(d), "parallel (2 spawn workers) == serial, row for row")

    import make_prob_videos as P
    split = os.path.join(OUT, "split"); os.makedirs(split, exist_ok=True)
    shutil.copy(os.path.join(ser, "electrodes.csv"), split)
    c0 = d[d.culture == 0]
    half = c0.neuron.max() // 2
    c0[c0.neuron <= half].to_csv(os.path.join(split, "culture_frames_a.csv"), index=False)
    c0[c0.neuron > half].to_csv(os.path.join(split, "culture_frames_b.csv"), index=False)
    ld, _ = P.load_frames(os.path.join(split, "culture_frames_*.csv"))
    paths = P.main(["--frames", os.path.join(ser, "culture_frames_S7_C*.csv"), "--half", "220",
                    "--bin", "20", "--sigma", "30", "--min-weight", "1", "--fps", "20",
                    "--hold-end", "0", "--outdir", os.path.join(ser, "render")])
    ok &= report("6", ld.culture_id.nunique() == 1 and len(paths) == 1
                 and os.path.getsize(paths[0]) > 10_000,
                 "split culture = 1 culture; probability video rendered")

    import make_culture_videos as C
    lst = C.main(["--frames", os.path.join(ser, "culture_frames_S7_C*.csv"), "--list"])
    pv = C.main(["--frames", os.path.join(ser, "culture_frames_S7_C*.csv"), "--cultures", "1",
                 "--layer", "80", "--half", "220", "--n-morph", "5", "--fps", "20",
                 "--hold-end", "0", "--outdir", os.path.join(ser, "render")])
    ok &= report("7", lst == [] and len(pv) == 1 and os.path.getsize(pv[0]) > 10_000,
                 "culture listed and culture video rendered")

    print("\nSMOKE TEST:", "ALL PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)      # non-zero on failure, so a job gate can stop
