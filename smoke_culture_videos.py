"""
smoke_culture_videos.py -- offline checks for make_culture_videos.py (no NEURON needed).

Uses the project's real slicer.py / field.py / config.py and ONE real morphology file; a stub
`morphologies` module maps every specimen name to that file (so the three "specimens" share the
same shape here -- only their colour differs). Frames are synthetic, with theta_deg.

Checks
  1  rotate_translate: (10,0) rotated 90 deg and moved to (100,50) -> (100,60)
  2  field: single electrode matches 1/(2 pi sigma r); array signs follow the current
  3  wilson: interval contains p, inside [0,1]
  4  state_counts_vs_r equals a direct recount
  5  neuron_matrices keeps neuron/time alignment
  6  morphologies are read through the real slicer for all specimens
  7  two culture videos + snapshots are written
  8  old export without theta_deg -> renders somata only
  9  polarity: during phase 1 the cathode-side somata are more often depolarized than the
     anode-side ones, and the opposite in phase 2
 10  angle: theta convention (anode 0 deg, cathode 180 deg) and ensemble P(depol | theta)
     flips between phases
 11  soma gradient: +10 mV is red, -10 mV is blue, 0 is near white; state mode still renders

Paths: run it from the project folder. If the project's morphologies.py is importable, the three
REAL specimens are used; otherwise a stub maps every name to one .asc (MORPH_ASC, or the first
.asc found). UPLOAD_DIR overrides the folder that holds slicer.py / field.py / config.py.

Run:  python smoke_culture_videos.py      (writes into ./_smoke_culture/)
"""
import os
import sys
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# project folder = where slicer.py lives (normally this script's folder)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or next(
    (d for d in (HERE, os.getcwd(), "/mnt/user-data/uploads")
     if os.path.isfile(os.path.join(d, "slicer.py"))), HERE)
sys.path.insert(0, UPLOAD_DIR)
try:                                   # real project: the three real specimens
    import morphologies as _real_morph   # noqa: F401
    MORPH_SOURCE = "project morphologies.py"
except Exception:                      # sandbox: one .asc stands in for every specimen
    MORPH_ASC = os.environ.get("MORPH_ASC") or next(
        (os.path.join(UPLOAD_DIR, f) for f in sorted(os.listdir(UPLOAD_DIR)) if f.endswith(".asc")),
        "")
    stub = types.ModuleType("morphologies")
    stub.find_one_morphology = lambda name=None: MORPH_ASC
    stub.all_morphologies = lambda: [("stub", MORPH_ASC)]
    sys.modules["morphologies"] = stub
    MORPH_SOURCE = f"stub -> {MORPH_ASC or 'NO .asc FOUND'}"

import make_prob_videos as P            # noqa: E402
import make_culture_videos as C         # noqa: E402

OUT = "_smoke_culture"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(1)
SPECS = ["60308", "130303", "60303"]
TIMES = np.unique(np.round(np.r_[np.arange(-0.60, 0.40 + 1e-9, 0.025),
                                 np.arange(0.45, 3.0 + 1e-9, 0.25), [5, 10, 20, 50, 100, 200]], 4))


def synth(n_cult=4, n_neur=120, layers=(40, 80, 120), half=400.0, k=300.0, thr=12.0):
    h = 30.0
    ys = np.array([h, -h, -3 * h])
    ex = np.r_[np.full(3, -h), np.full(3, h)]; ey = np.r_[ys, ys]
    es = np.r_[np.ones(3), -np.ones(3)]
    pd.DataFrame({"x": ex, "y": ey, "sign": es.astype(int)}).to_csv(
        os.path.join(OUT, "electrodes.csv"), index=False)
    rows = []
    for c in range(n_cult):
        xy = RNG.uniform(-half, half, size=(n_neur, 2))
        th = RNG.uniform(0, 360, n_neur)
        mo = np.array(SPECS)[np.arange(n_neur) % 3]
        r = np.sqrt((xy[:, None, 0] - ex) ** 2 + (xy[:, None, 1] - ey) ** 2 + 100.0)
        g = (es / np.maximum(r, 12.5)).sum(axis=1)
        for L in layers:
            for i in range(n_neur):
                dv = -k * g[i] * P.pulse_current(TIMES)
                post = TIMES > 0
                dv[post] = 0.3 * k * g[i] * np.exp(-TIMES[post] / 15.0)
                dv = dv + RNG.normal(0, 0.01, TIMES.size)
                above = np.where(dv > thr)[0]
                tsp = TIMES[above[0]] if above.size else None
                for t, v in zip(TIMES, dv):
                    rows.append([c, mo[i], L, round(xy[i, 0], 2), round(xy[i, 1], 2), t, v,
                                 int(tsp is not None and t >= tsp), round(th[i], 2)])
    df = pd.DataFrame(rows, columns=["culture", "morph", "layer", "x", "y", "t_ms", "dVm_mV",
                                     "fired_by_t", "theta_deg"])
    path = os.path.join(OUT, "culture_frames_smoke.csv")
    df.to_csv(path, index=False)
    return path


def report(tag, cond, msg):
    print(f"[{tag}] {msg}: {'OK' if cond else 'FAIL'}")
    return bool(cond)


def main():
    ok = True
    q = C.rotate_translate([[10.0, 0.0]], 100.0, 50.0, 90.0)[0]
    ok &= report("1", np.allclose(q, [100.0, 60.0]), f"rotation -> {np.round(q, 6)}")

    g1 = C.field_mV_per_A(np.array([100.0]), np.array([0.0]), [[0.0, 0.0]], [1.0], 1.5, 12.5, 10.0)[0]
    ref = 1e3 / (2 * np.pi * 1.5 * np.sqrt(100.0 ** 2 + 10.0 ** 2) * 1e-6)
    ex = np.array([[-30, 30], [-30, -30], [-30, -90], [30, 30], [30, -30], [30, -90]], float)
    es = np.r_[np.ones(3), -np.ones(3)]
    g = C.field_mV_per_A(np.array([-30.0, 30.0]), np.array([-30.0, -30.0]), ex, es, 1.5, 12.5, 10.0)
    ph1 = g * P.pulse_current([-0.375])[0]; ph2 = g * P.pulse_current([-0.125])[0]
    ok &= report("2", abs(g1 - ref) / ref < 1e-12 and ph1[0] > 0 > ph1[1] and ph2[0] < 0 < ph2[1]
                 and P.pulse_current([0.5])[0] == 0,
                 f"single electrode {g1:.4g} vs {ref:.4g} mV/A; phase 1 anode {ph1[0]:+.3g}, "
                 f"cathode {ph1[1]:+.3g}")

    lo, hi = C.wilson(np.array([0, 3, 10]), np.array([10, 10, 10]))
    p = np.array([0, 0.3, 1.0])
    ok &= report("3", np.all(lo <= p + 1e-12) and np.all(hi >= p - 1e-12) and lo.min() >= 0
                 and hi.max() <= 1, f"Wilson lo {np.round(lo, 3)}, hi {np.round(hi, 3)}")

    path = synth()
    df, files = P.load_frames(path)
    df["morph"] = df["morph"].astype(str)
    d80 = df[df.layer == 80]
    times = np.sort(d80.t_ms.unique())
    elec, sign = P.load_electrodes(files)
    centre = C.dipole_centre(elec, sign)
    edges = np.arange(0, 600, 25.0)
    cnt = C.state_counts_vs_r(d80, times, centre, 10.0, edges, 0.1)
    k = int(np.argmin(np.abs(times + 0.375)))
    dk = d80[d80.t_ms == times[k]]
    r = np.sqrt((dk.x - centre[0]) ** 2 + (dk.y - centre[1]) ** 2 + 100.0)
    b = 3
    inb = (r >= edges[b]) & (r < edges[b + 1])
    a, dp, hp = P.states(dk.dVm_mV.to_numpy(), dk.fired_by_t.to_numpy(), 0.1)
    ok &= report("4", cnt["n"][k, b] == inb.sum() and cnt["dep"][k, b] == dp[inb.to_numpy()].sum()
                 and cnt["act"][k, b] == a[inb.to_numpy()].sum(),
                 f"bin {edges[b]:.0f}-{edges[b+1]:.0f} um at t={times[k]:+.3f}: "
                 f"n {int(cnt['n'][k, b])}, dep {int(cnt['dep'][k, b])}")

    sub = d80[d80.culture_id == 0]
    t_c, tab, dv, fb = C.neuron_matrices(sub)
    row = tab.iloc[5]
    direct = sub[(sub.x == row.x) & (sub.y == row.y) & (sub.t_ms == t_c[k])].dVm_mV.iloc[0]
    ok &= report("5", dv.shape == (len(times), 120) and np.isclose(dv[k, 5], direct),
                 f"matrix {dv.shape}, neuron 5 at t={t_c[k]:+.3f}: {dv[k,5]:.3f} == {direct:.3f}")

    polys = C.load_morph_polylines(SPECS, 80.0)
    n_poly = [len(polys.get(s, [])) for s in SPECS]
    ok &= report("6", all(n > 10 for n in n_poly),
                 f"polylines per specimen {dict(zip(SPECS, n_poly))} (source: {MORPH_SOURCE})")

    paths = C.main(["--frames", path, "--cultures", "0", "2", "--layer", "80", "--n-morph", "15",
                    "--fps", "12", "--hold-end", "0.5", "--outdir", OUT])
    snaps = [f for f in os.listdir(OUT) if f.startswith("culture_snapshots")]
    ok &= report("7", len(paths) == 2 and all(os.path.getsize(q) > 10_000 for q in paths)
                 and len(snaps) == 2, f"videos {[os.path.basename(q) for q in paths]}, {len(snaps)} snapshots")

    old = pd.read_csv(path).drop(columns=["theta_deg"])
    opath = os.path.join(OUT, "old", "culture_frames_old.csv")
    os.makedirs(os.path.dirname(opath), exist_ok=True)
    old.to_csv(opath, index=False)
    pd.read_csv(os.path.join(OUT, "electrodes.csv")).to_csv(os.path.join(OUT, "old", "electrodes.csv"), index=False)
    p8 = C.main(["--frames", opath, "--cultures", "1", "--layer", "80", "--hold-end", "0",
                 "--outdir", os.path.join(OUT, "old")])
    ok &= report("8", len(p8) == 1 and os.path.getsize(p8[0]) > 10_000, "no theta -> somata-only video")

    near = np.hypot(tab.x, tab.y + 30) < 120
    cath = (tab.x > 0) & near; anod = (tab.x < 0) & near
    k2 = int(np.argmin(np.abs(times + 0.125)))
    a1, d1, _ = P.states(dv[k], fb[k], 0.1); a2, d2, _ = P.states(dv[k2], fb[k2], 0.1)
    c9 = d1[cath.to_numpy()].mean() > d1[anod.to_numpy()].mean() and \
        d2[anod.to_numpy()].mean() > d2[cath.to_numpy()].mean()
    ok &= report("9", c9, f"depolarized near cathodes/anodes: phase 1 {d1[cath.to_numpy()].mean():.2f}/"
                 f"{d1[anod.to_numpy()].mean():.2f}, phase 2 {d2[cath.to_numpy()].mean():.2f}/"
                 f"{d2[anod.to_numpy()].mean():.2f}")

    th_a = C.dipole_theta_deg(np.array([-200.0]), np.array([-30.0]), elec, sign)[0]
    th_k = C.dipole_theta_deg(np.array([200.0]), np.array([-30.0]), elec, sign)[0]
    ed = np.arange(0, 181, 15.0)
    cth = C.state_counts_vs(C.dipole_theta_deg(d80.x, d80.y, elec, sign), d80, times, ed, 0.1)
    pdep = lambda kk, b: cth["dep"][kk, b].sum() / max(cth["n"][kk, b].sum(), 1)
    lo_b, hi_b = slice(0, 2), slice(10, 12)            # 0-30 deg (anode) / 150-180 deg (cathode)
    c10 = (abs(th_a) < 1e-6 and abs(th_k - 180) < 1e-6 and pdep(k, hi_b) > pdep(k, lo_b)
           and pdep(k2, lo_b) > pdep(k2, hi_b))
    ok &= report("10", c10, f"theta anode {th_a:.0f}, cathode {th_k:.0f}; P(dep) cathode/anode "
                 f"phase 1 {pdep(k, hi_b):.2f}/{pdep(k, lo_b):.2f}, phase 2 {pdep(k2, hi_b):.2f}/{pdep(k2, lo_b):.2f}")

    cp, cm, c0 = C.dvm_rgba([10.0, -10.0, 0.0], 20.0)
    ps = C.main(["--frames", path, "--cultures", "3", "--layer", "80", "--soma-color", "state",
                 "--hold-end", "0", "--outdir", os.path.join(OUT, "state")])
    c11 = (cp[0] > cp[2] + 0.3 and cm[2] > cm[0] + 0.3 and min(c0[:3]) > 0.85
           and len(ps) == 1 and os.path.getsize(ps[0]) > 10_000)
    ok &= report("11", c11, f"dVm colours +10 {np.round(cp[:3],2)}, -10 {np.round(cm[:3],2)}, "
                 f"0 {np.round(c0[:3],2)}; state-mode video written")

    print("\nSMOKE TEST:", "ALL PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)      # non-zero on failure, so a job gate can stop
