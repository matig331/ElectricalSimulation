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
 12  --n-somata: a seeded random subset is drawn; its counts (state_counts_matrix) equal a direct
     recount of those neurons; the plain style has grey branches and edge-less somata; the
     specimen style still renders; --morph-fraction f gives round(f n) random somata their arbor
     (the same ones on every call), all of them by default; more than MORPH_IMAGE_ABOVE arbors
     are drawn once as a grey image layer

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

    # 12 -- a subset of the culture's neurons, and the two styles
    keep = np.sort(np.random.default_rng(0).choice(len(tab), size=40, replace=False))
    tsub = tab.iloc[keep].reset_index(drop=True)
    ccoord = C.dipole_theta_deg(tsub.x, tsub.y, elec, sign)
    cm12 = C.state_counts_matrix(ccoord, dv[:, keep], fb[:, keep], ed, 0.1)
    xy_keep = set(zip(tsub.x.round(2), tsub.y.round(2)))
    dsub = sub[[xy in xy_keep for xy in zip(sub.x.round(2), sub.y.round(2))]]
    cdir = C.state_counts_vs(C.dipole_theta_deg(dsub.x, dsub.y, elec, sign), dsub, times, ed, 0.1)
    same = all(np.array_equal(cm12[q], cdir[q]) for q in ("n", "act", "dep", "hyp"))
    p12 = C.main(["--frames", path, "--cultures", "0", "--layer", "80", "--n-somata", "40",
                  "--hold-end", "0", "--outdir", os.path.join(OUT, "subset")])
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots()
    xg = np.arange(-100, 101, 50.0)
    gg = np.zeros((xg.size, xg.size))
    _, _, sc, nd = C._draw_scene(ax, 100.0, 50.0, gg, 1.0, elec, 25.0, tsub, polys, 5, centre, 10.0,
                                 20.0, "plain")
    a12, d12, h12 = P.states(dv[k][keep], fb[k][keep], 0.1)
    C._update_somata(sc, tsub, {"act": a12, "dep": d12, "hyp": h12, "neu": ~(a12 | d12 | h12)},
                     dv[k][keep], "dvm", "plain")
    lc = [c for c in ax.collections if isinstance(c, LineCollection)]
    grey = all(np.allclose(c.get_colors()[:, :3], C.matplotlib.colors.to_rgb(C.BRANCH_GREY))
               for c in lc) if lc else True
    noedge = all(len(sc[q].get_linewidths()) == 0 or float(np.max(sc[q].get_linewidths())) == 0.0
                 for q in ("act", "grad"))
    plt.close(fig)
    fig, ax = plt.subplots()                    # > MORPH_IMAGE_ABOVE morphologies: one image layer
    _, _, _, nd_all = C._draw_scene(ax, 400.0, 50.0, gg, 1.0, elec, 25.0, tab, polys, -1, centre,
                                    10.0, 20.0, "plain")
    ims = [im_ for im_ in ax.images if im_.get_zorder() == 3]
    arr = np.asarray(ims[0].get_array()) if ims else np.zeros((1, 1, 4))
    ink = arr[..., 3] > 0
    grey_img = bool(ink.any()) and np.allclose(arr[ink][:, :3].mean(axis=0) / 255.0,
                                               C.matplotlib.colors.to_rgb(C.BRANCH_GREY), atol=0.06)
    plt.close(fig)

    def arbors(frac, seed):                     # a random fraction of the drawn somata
        f_, a_ = plt.subplots()
        _, _, _, n_ = C._draw_scene(a_, 400.0, 50.0, gg, 1.0, elec, 25.0, tab, polys, -1, centre,
                                    10.0, 20.0, "plain", frac, seed)
        sg = [q for c in a_.collections if isinstance(c, LineCollection) for q in c.get_segments()]
        plt.close(f_)
        return n_, sg
    nq, sa = arbors(0.25, 7)
    nq2, sb = arbors(0.25, 7)
    n0, _ = arbors(0.0, 7)
    frac_ok = (nq == round(0.25 * len(tab)) and nq2 == nq and n0 == 0 and len(sa) == len(sb) > 0
               and all(np.array_equal(u, v) for u, v in zip(sa, sb)))
    pspec = C.main(["--frames", path, "--cultures", "1", "--layer", "80", "--style", "specimen",
                    "--hold-end", "0", "--outdir", os.path.join(OUT, "specimen")])
    ok &= report("12", same and len(p12) == 1 and os.path.getsize(p12[0]) > 10_000 and nd == min(5, 40)
                 and grey and noedge and len(pspec) == 1 and nd_all == len(tab) and grey_img
                 and frac_ok,
                 f"subset counts == direct recount: {same}; 40-neuron video written; branches grey: "
                 f"{grey} ({len(lc)} collections); soma edges off: {noedge}; specimen style renders; "
                 f"--n-morph -1 draws all {nd_all} arbors as one grey image layer: {grey_img}; "
                 f"--morph-fraction 0.25 -> {nq} of {len(tab)} arbors, same ones again, 0 -> {n0}")

    print("\nSMOKE TEST:", "ALL PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)      # non-zero on failure, so a job gate can stop
