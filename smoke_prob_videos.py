"""
smoke_prob_videos.py -- offline checks for make_prob_videos.py (no NEURON needed).

Synthetic culture_frames.csv with a toy but physically oriented response:
    Ve(x,y,t) ~ I(t) * sum_i s_i / r_i       (monopoles, anodes s=+1 at x<0, cathodes s=-1 at x>0)
    dVm(t)    = -k * Ve  during the pulse, then an exponential relaxation (tau 15 ms)
so in phase 1 (+I) depolarisation must sit on the CATHODE side (x>0) and in phase 2 (-I) on the
ANODE side (x<0). Neurons whose dVm exceeds a threshold "spike". Frame times follow the extended
post-pulse schedule recommended for culture_export (up to 200 ms).

Checks
   1  pulse_current: +1 mid phase 1, -1 mid phase 2, 0 outside, charge-balanced
   2  estimator: constant outcome 1 -> P == 1 wherever data exist
   3  estimator: Bernoulli(0.3) outcome -> mean P ~ 0.3
  3b  coverage inside the sampled square > 95%
   4  mutual exclusivity: P_act + P_dep + P_hyp <= 1 at every bin and time
   5  before the pulse: P_dep, P_hyp ~ 0 (neu = 0.1 mV)
   6  mirror: depolarisation centroid x > 0 in phase 1, x < 0 in phase 2
   7  per-culture filter: each culture video uses only its own neurons
   8  merged videos (pooled + 2 cultures) and snapshots are written
   9  frozen-pulse guard: pre_end_ms=0-like export is detected and rendering stops
  10  colour mixture: pure states give pure colours, all-zero -> white, NaN -> grey, 50/50 -> mean
  11  counts: per-frame counts equal a direct recount; act+dep+hyp <= total
  12  split layout still renders
  13  post-pulse relaxation: depol+hyp fraction at the last frame << at mid phase 1
  14  area metric: known half-covered map -> 50%, covered area = n bins x bin area, and the
      four expected areas sum to the covered area on the synthetic run

Run:  python smoke_prob_videos.py            (writes into ./_smoke_videos/)
"""
import os
import sys
import numpy as np
import pandas as pd

import make_prob_videos as V

OUT = "_smoke_videos"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)

FRAME_TIMES_MS = np.unique(np.round(np.concatenate([
    np.arange(-0.60, 0.40 + 1e-9, 0.025),
    np.arange(0.45, 3.0 + 1e-9, 0.05),
    np.arange(3.5, 20.0 + 1e-9, 0.5),
    np.arange(25.0, 200.0 + 1e-9, 5.0),
]), 4))


def synth(n_cult=2, n_neur=150, layers=(40, 80, 120), half=180.0, k=300.0, thr=12.0):
    h = 30.0
    ys = np.array([h, -h, -3 * h])
    ex = np.r_[np.full(3, -h), np.full(3, h)]
    ey = np.r_[ys, ys]
    es = np.r_[np.ones(3), -np.ones(3)]
    pd.DataFrame({"x": ex, "y": ey, "sign": es.astype(int)}).to_csv(
        os.path.join(OUT, "electrodes.csv"), index=False)
    times = FRAME_TIMES_MS
    rows = []
    for c in range(n_cult):
        xy = RNG.uniform(-half, half, size=(n_neur, 2))
        r = np.sqrt((xy[:, None, 0] - ex) ** 2 + (xy[:, None, 1] - ey) ** 2 + 10.0 ** 2)
        g = (es / np.maximum(r, 12.5)).sum(axis=1)
        for L in layers:
            for i in range(n_neur):
                dv_t = -k * g[i] * V.pulse_current(times)
                post = times > 0
                dv_t[post] = 0.3 * k * g[i] * np.exp(-times[post] / 15.0)   # small signed relaxation
                dv_t = dv_t + RNG.normal(0, 0.01, times.size)
                above = np.where(dv_t > thr)[0]
                t_sp = times[above[0]] if above.size else None
                for t, dv in zip(times, dv_t):
                    rows.append([c, "toy", L, xy[i, 0], xy[i, 1], t, dv,
                                 int(t_sp is not None and t >= t_sp)])
    df = pd.DataFrame(rows, columns=["culture", "morph", "layer", "x", "y",
                                     "t_ms", "dVm_mV", "fired_by_t"])
    path = os.path.join(OUT, "culture_frames.csv")
    df.to_csv(path, index=False)
    return path, df


def report(tag, cond, msg):
    print(f"[{tag}] {msg}: {'OK' if cond else 'FAIL'}")
    return bool(cond)


def main():
    ok = True

    I = V.pulse_current([-0.6, -0.375, -0.125, 0.1])
    tt = np.linspace(-0.7, 0.3, 20001)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    q = trap(V.pulse_current(tt), tt)
    ok &= report("1", np.allclose(I, [0, 1, -1, 0]) and abs(q) < 1e-6,
                 f"pulse {I}, charge {q:+.1e}")

    edges = np.arange(-400, 401, 20.0)
    x = RNG.uniform(-180, 180, 5000); y = RNG.uniform(-180, 180, 5000)
    p, _ = V.kernel_map(x, y, np.ones_like(x), edges, 20.0, 3.0)
    ok &= report("2", np.nanmax(np.abs(p - 1)) < 1e-9 and np.isfinite(p).any(),
                 "constant outcome -> P==1 where data")
    v = RNG.random(x.size) < 0.3
    p, _ = V.kernel_map(x, y, v, edges, 20.0, 3.0)
    ok &= report("3", abs(np.nanmean(p) - 0.3) < 0.03, f"Bernoulli(0.3) -> mean P {np.nanmean(p):.3f}")

    path, df = synth()
    times, maps, edges, _, counts = V.compute_maps(df, 400.0, 20.0, 25.0, 3.0, 0.1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    inner = (np.abs(xc)[None, :] < 160) & (np.abs(xc)[:, None] < 160)
    cov = np.isfinite(maps["act"][0][inner]).mean()
    ok &= report("3b", cov > 0.95, f"coverage inside sampled square {100*cov:.1f}%")

    s = maps["act"] + maps["dep"] + maps["hyp"]
    ok &= report("4", np.nanmax(s) <= 1 + 1e-9, f"max P_act+P_dep+P_hyp = {np.nanmax(s):.4f}")

    ok &= report("5", np.nanmax(maps["dep"][0]) < 0.05 and np.nanmax(maps["hyp"][0]) < 0.05,
                 f"before pulse t={times[0]:+.3f}: P_dep,P_hyp ~ 0")

    X = np.broadcast_to(xc[None, :], maps["dep"][0].shape)

    def centroid(k):
        w = np.nan_to_num(maps["dep"][k]); return float((w * X).sum() / max(w.sum(), 1e-12))
    k1 = int(np.argmin(np.abs(times + 0.375))); k2 = int(np.argmin(np.abs(times + 0.125)))
    ok &= report("6", centroid(k1) > 0 > centroid(k2),
                 f"depol centroid phase1 {centroid(k1):+.0f} um, phase2 {centroid(k2):+.0f} um")

    paths = V.main(["--frames", path, "--cultures", "0", "1", "--bin", "20", "--sigma", "25",
                    "--fps", "12", "--hold-end", "0.5", "--outdir", OUT])
    loaded, _ = V.load_frames(path)
    n0 = loaded[(loaded.culture_id == 0) & (loaded.t_ms == loaded.t_ms.min())].shape[0]
    ok &= report("7", n0 == 450, f"culture 0 observations/frame = {n0} (expected 450)")
    snaps = [f for f in os.listdir(OUT) if f.startswith("prob_snapshots")]
    ok &= report("8", len(paths) == 3 and all(os.path.getsize(q) > 10_000 for q in paths)
                 and len(snaps) == 3, f"merged videos {[os.path.basename(q) for q in paths]}, "
                 f"{len(snaps)} snapshots")

    fz_ok = V.frozen_pulse_fraction(loaded)
    frozen = loaded.copy(); key = ["culture_id", "layer", "x", "y"]
    t0 = frozen.loc[frozen.t_ms.abs().idxmin(), "t_ms"]
    v0 = frozen[frozen.t_ms == t0].set_index(key)["dVm_mV"]
    neg = frozen.t_ms < 0
    frozen.loc[neg, "dVm_mV"] = frozen.loc[neg].set_index(key).index.map(v0).to_numpy()
    fz_bad = V.frozen_pulse_fraction(frozen)
    fpath = os.path.join(OUT, "frozen_frames.csv")
    frozen.drop(columns=["_file", "culture_id"]).to_csv(fpath, index=False)
    stopped = False
    try:
        V.main(["--frames", fpath, "--outdir", os.path.join(OUT, "frozen")])
    except SystemExit:
        stopped = True
    ok &= report("9", fz_ok < 0.1 and fz_bad > 0.9 and stopped,
                 f"frozen guard: normal {fz_ok:.2f}, frozen {fz_bad:.2f}, stopped={stopped}")

    one, zero, nan = np.ones((1, 1)), np.zeros((1, 1)), np.full((1, 1), np.nan)
    c10 = (np.allclose(V.mix_rgb(zero, one, zero)[0, 0], V.COL["dep"])
           and np.allclose(V.mix_rgb(one, zero, zero)[0, 0], V.COL["act"])
           and np.allclose(V.mix_rgb(zero, zero, zero)[0, 0], [1, 1, 1])
           and np.allclose(V.mix_rgb(nan, nan, nan)[0, 0], V.NODATA)
           and np.allclose(V.mix_rgb(zero, 0.5 * one, 0.5 * one)[0, 0],
                           0.5 * (V.COL["dep"] + V.COL["hyp"])))
    ok &= report("10", c10, "colour mixture (pure, white, no-data, 50/50)")

    k = k1
    d = df[df.t_ms == times[k]]
    a, dp, hp = V.states(d.dVm_mV.to_numpy(), d.fired_by_t.to_numpy(), 0.1)
    c11 = (counts["act"][k] == a.sum() and counts["dep"][k] == dp.sum()
           and counts["hyp"][k] == hp.sum() and counts["tot"][k] == len(d)
           and np.all(counts["act"] + counts["dep"] + counts["hyp"] <= counts["tot"]))
    ok &= report("11", c11, f"counts at t={times[k]:+.3f}: act {a.sum()}, dep {dp.sum()}, "
                 f"hyp {hp.sum()}, tot {len(d)}")

    sp = V.main(["--frames", path, "--layout", "split", "--bin", "20", "--sigma", "25",
                 "--hold-end", "0", "--outdir", os.path.join(OUT, "split")])
    ok &= report("12", len(sp) == 1 and os.path.getsize(sp[0]) > 10_000, "split layout renders")

    frac = lambda kk: (counts["dep"][kk] + counts["hyp"][kk]) / counts["tot"][kk]
    ok &= report("13", frac(len(times) - 1) < 0.2 * frac(k1),
                 f"post-pulse relaxation: dep+hyp {100*frac(k1):.0f}% (phase 1) -> "
                 f"{100*frac(len(times)-1):.0f}% (t={times[-1]:.0f} ms)")

    e = np.arange(0.0, 101.0, 10.0)                     # 10x10 bins of 10 um -> 1e-4 mm2 each
    pa = np.full((1, 10, 10), np.nan); pa[0, :, :6] = 0.0; pa[0, :, :3] = 1.0   # 60 covered, 30 active
    zz = np.where(np.isfinite(pa), 0.0, np.nan)
    am = V.area_metrics({"act": pa, "dep": zz, "hyp": zz}, e)
    ar = V.area_metrics(maps, edges)
    tot = ar["act"] + ar["dep"] + ar["hyp"] + ar["neu"]
    c14 = (np.isclose(am["cov"][0], 60e-4) and np.isclose(100 * am["act"][0] / am["cov"][0], 50.0)
           and np.allclose(tot, ar["cov"]))
    ok &= report("14", c14, f"area: known map act {100*am['act'][0]/am['cov'][0]:.0f}% of "
                 f"{am['cov'][0]*1e4:.0f}e-4 mm2; synthetic run covered {ar['cov'][0]:.3f} mm2, "
                 f"states sum to it")

    print("\nSMOKE TEST:", "ALL PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)      # non-zero on failure, so a job gate can stop
