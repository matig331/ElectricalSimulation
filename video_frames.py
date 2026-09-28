"""video_frames.py -- the data of the stimulation videos, produced with the CURRENT HPC code.

For every neuron of every culture (at the chosen layers) it writes DeltaV(t) = V_stim(t) - V_sham(t)
of the soma on a grid of frame times, plus whether the soma has spiked by t. The two renderers
(make_prob_videos.py, make_culture_videos.py) read these files.

ALIGNED WITH THE CAMPAIGN (culture_export.py):
  * same cultures: culture_draws(seed, c, N, ...) -- same morphologies, positions, rotations;
  * same cells and states: a CellPool subclass (one live cell per process, rest and leak state
    exactly as the campaign's), same stimulus (config.i0_uA, rich_footprint.spikes_at);
  * same reference: the sham run (I = 0) of the same cell and protocol;
  * same label: phase2_end_outcome on DeltaV_end; every chunk checks that its DeltaV_end equals
    the campaign definition (v_at_end_of_phase2 - v_sham_end of the pool) and reports the max
    difference.
ONLY TWO THINGS DIFFER from a campaign simulation, both needed for a video:
  * the recorded window starts PRE_END_MS (1 ms) before the end of phase 2 (the campaign keeps
    it from t_end on), so the frames DURING the pulse are real, not copies of t_end;
  * the post-pulse window is VIDEO_TMAX (--tmax-ms, default 200 ms), on the campaign's long-window
    machinery (bump_ms, CVODE, bump_dt_ms grid).
Neither changes the simulated dynamics: they only change what is returned.

PARALLEL: the neurons are split into chunks (--chunk) and run on --processes worker processes
(spawn), each with its own one-live-cell pool. Chunks are merged per culture at the end.

OUTPUT (--out, default video_run/):
  culture_frames_S<seed>_C<c>.csv   seed, culture, neuron, morph, layer, x, y, theta_deg,
                                    t_ms, dVm_mV, fired_by_t        (one row per neuron x layer x frame)
  culture_video_neurons.csv         one row per neuron x layer: fired, dv_end_mV, label,
                                    dist_dipole3d_um, theta_pos_deg  (for checks and static maps)
  electrodes.csv                    x, y, sign
  video_manifest.json               every parameter of the run + consistency report

Run (from the pipeline folder; on HPC via jobs/video_frames.pbs):
    python video_frames.py --n-cultures 5 --neurons 1700 --span-um 571.75 --layers 80 --processes 32
Test (no NEURON): python smoke_video_frames.py
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PRE_END_MS = 1.0          # recorded window starts this long before the end of phase 2
SPIKE_MV = 0.0            # spike = somatic crossing of 0 mV (campaign definition of 'fired')
FRAME_COLS = ["seed", "culture", "neuron", "morph", "layer", "x", "y", "theta_deg",
              "t_ms", "dVm_mV", "fired_by_t"]
NEURON_COLS = ["seed", "culture", "neuron", "morph", "layer", "x", "y", "theta_deg",
               "dist_dipole3d_um", "theta_pos_deg", "fired", "dv_end_mV", "label"]


# ----------------------------------------------------------------------------- frame schedule
def frame_times(tmax_ms=200.0):
    """Dense during and just after the pulse, sparse in the tail (t = 0: end of phase 2)."""
    T = float(tmax_ms)
    return np.unique(np.round(np.concatenate([
        np.arange(-0.60, 0.40 + 1e-9, 0.025),              # pulse, 25 us
        np.arange(0.45, min(3.0, T) + 1e-9, 0.05),          # soma spikes (AIS -> soma delay)
        np.arange(3.5, min(20.0, T) + 1e-9, 0.5),           # fast relaxation
        np.arange(25.0, min(200.0, T) + 1e-9, 5.0),         # slow tail
        np.arange(220.0, T + 1e-9, 20.0),                   # beyond 200 ms
    ]), 4))


# ----------------------------------------------------------------------------- the cell pool
def _pool_class():
    """VideoPool, defined lazily so importing this module never imports NEURON."""
    import culture_export as CE

    class VideoPool(CE.CellPool):
        """CellPool that also returns the whole trace (window from PRE_END_MS before t_end to
        tmax_ms after it) of a stimulated run and of its sham, for the SAME live cell."""

        def __init__(self, cfg=None, cell_model=None, tmax_ms=200.0, rest_tstop_ms=1500.0):
            super().__init__(cfg, cell_model, rest_tstop_ms)
            self.tmax_ms = float(tmax_ms)
            self._vsham = {}

        def _run(self, key, pos_xy, theta_deg, i0_uA):
            from rich_footprint import spikes_at
            kin = self._info[key].get("kin") or {}
            return spikes_at(self._cell, (float(pos_xy[0]), float(pos_xy[1])), float(theta_deg),
                             i0_uA=float(i0_uA), detail=True, pre_end_ms=PRE_END_MS,
                             v_init_mV=self._info[key]["v_rest"], bump_ms=self.tmax_ms,
                             bump_dt_ms=float(kin.get("bump_dt_ms", 0.5)),
                             cvode_atol=float(kin.get("cvode_atol", 1e-6)),
                             play_margin_ms=float(kin.get("play_margin_ms", 1.0)))

        def trace(self, morph, layer, pos_xy, theta_deg, i0_uA):
            key = self._activate(morph, layer)
            if key not in self._vsham:
                sh = self._run(key, (0.0, 0.0), 0.0, 0.0)
                if int(sh[0]) > 0:
                    raise RuntimeError("%s L%g: spikes with zero stimulus" % key)
                self._vsham[key] = sh
            return self._run(key, pos_xy, theta_deg, i0_uA), self._vsham[key], self._info[key]

    return VideoPool


def _make_pool(cell_model, tmax_ms):
    if os.environ.get("VIDEO_FRAMES_FAKE"):                 # smoke test only (no NEURON)
        import smoke_video_frames as T
        return T.FakePool(tmax_ms)
    from config import CFG
    return _pool_class()(CFG, cell_model, tmax_ms)


# ----------------------------------------------------------------------------- one neuron
def neuron_frames(out, sham, v_sham_end, times):
    """Frames of ONE simulation.

    out / sham : spikes_at(detail=True, bump_ms>0) results: (n_spikes, v_max, tw, vw, tg, vg);
                 tw/vw fixed-dt window, tg/vg long-window grid; times relative to the end of
                 phase 2 (t = 0), as in culture_export.simulate_neuron.
    Returns dict(dv [n_frames], fired_by_t [n_frames], fired, dv_end, t_spike, dv_end_campaign).
    """
    import culture_export as CE
    from bump_kinetics import common_prefix_len
    nsp, tw, vw = int(out[0]), np.asarray(out[2], float), np.asarray(out[3], float)
    tws, vws = np.asarray(sham[2], float), np.asarray(sham[3], float)
    k = int(common_prefix_len(tw, tws))
    if k < 2:
        raise RuntimeError("stimulated and sham runs do not share their fixed-dt time base")
    t_all, dv_all, v_all = [tw[:k]], [vw[:k] - vws[:k]], [vw[:k]]
    if len(out) > 5 and out[4] is not None and len(out[4]):
        tg, vg = np.asarray(out[4], float), np.asarray(out[5], float)
        tgs, vgs = np.asarray(sham[4], float), np.asarray(sham[5], float)
        if tgs.shape == tg.shape and np.allclose(tgs, tg, rtol=0.0, atol=1e-9):
            dvg = vg - vgs
        else:
            dvg = vg - np.interp(tg, tgs, vgs)
        later = tg > tw[k - 1]
        t_all.append(tg[later]); dv_all.append(dvg[later]); v_all.append(vg[later])
    T = np.concatenate(t_all); DV = np.concatenate(dv_all); V = np.concatenate(v_all)
    if T[0] > times[0] + 1e-9 or T[-1] < times[-1] - 1e-9:
        raise RuntimeError("recorded window [%.3f, %.3f] ms does not cover the frames [%.3f, %.3f] "
                           "ms -- check PRE_END_MS and the long window" % (T[0], T[-1], times[0],
                                                                            times[-1]))
    dv = np.interp(times, T, DV)
    up = np.where((V[:-1] < SPIKE_MV) & (V[1:] >= SPIKE_MV))[0]
    t_sp = float(T[up[0] + 1]) if up.size else None
    fired = int(nsp > 0)
    if fired and t_sp is None:                     # a spike outside the returned window
        t_sp = float(T[int(np.argmax(V))])
    fired_by_t = (times >= t_sp).astype(int) if (fired and t_sp is not None) else np.zeros(times.size, int)
    dv_end = float(CE.v_at_end_of_phase2(tw, vw) - CE.v_at_end_of_phase2(tws, vws))
    dv_end_campaign = float(CE.v_at_end_of_phase2(tw, vw) - float(v_sham_end))
    return dict(dv=dv, fired_by_t=fired_by_t, fired=fired, dv_end=dv_end, t_spike=t_sp,
                dv_end_campaign=dv_end_campaign)


# ----------------------------------------------------------------------------- one chunk
_POOL = None


def _init_worker(cell_model, tmax_ms, tmpdir):
    global _POOL
    if tmpdir:
        os.environ["TMPDIR"] = tmpdir
    _POOL = _make_pool(cell_model, tmax_ms)


def _geometry(cfg):
    import culture_export as CE
    import field as F
    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=not cfg.bipolar)
    return dict(elec=elec, sign=sign, center=CE.electrode_center(elec),
                axis=CE.dipole_axis_deg(elec, sign), dip=CE.dipole_frame(elec, sign))


def run_chunk(task):
    """Simulate neurons [i0, i1) of culture c at every layer; write one chunk CSV."""
    import culture_export as CE
    from config import CFG
    g = _geometry(CFG)
    c, seed, N, span, layers, i_start, i_end = (task[k] for k in
                                                ("c", "seed", "N", "span", "layers", "i0", "i1"))
    morphs = [str(m) for m in CFG.morphologies]
    dc, dd = g["dip"]
    d = CE.culture_draws(seed, c, N, len(morphs), span, g["elec"], g["center"], dc, dd, g["axis"],
                         CFG.h_soma_um)
    times = np.asarray(task["times"], float)
    idx = list(range(i_start, i_end))
    order = sorted(((int(d["midx"][i]), L, i) for i in idx for L in layers))   # group by cell
    res = {}
    for mi, L, i in order:
        out, sham, info = _POOL.trace(morphs[mi], L, d["pos"][i], d["theta"][i], task["i0_uA"])
        res[(i, L)] = neuron_frames(out, sham, info["v_sham_end"], times)
    path = os.path.join(task["tmp_out"], "chunk_S%d_C%d_%06d.csv" % (seed, c, i_start))
    npath = path.replace("chunk_", "neurons_")
    max_diff, n_fired = 0.0, 0
    with open(path, "w", newline="") as fh, open(npath, "w", newline="") as fn:
        w, wn = csv.writer(fh), csv.writer(fn)
        for i in idx:
            m = morphs[int(d["midx"][i])]
            x, y, th = float(d["pos"][i, 0]), float(d["pos"][i, 1]), float(d["theta"][i])
            for L in layers:
                r = res[(i, L)]
                max_diff = max(max_diff, abs(r["dv_end"] - r["dv_end_campaign"]))
                n_fired += r["fired"]
                label = CE.phase2_end_outcome(r["fired"], r["dv_end_campaign"])[0]
                wn.writerow([seed, c, i, m, int(L), round(x, 2), round(y, 2), round(th, 2),
                             round(float(d["r_dip"][i]), 2), round(float(d["th_pos"][i]), 1),
                             r["fired"], round(r["dv_end_campaign"], 6), label])
                for t, dv, fb in zip(times, r["dv"], r["fired_by_t"]):
                    w.writerow([seed, c, i, m, int(L), round(x, 2), round(y, 2), round(th, 2),
                                float(t), round(float(dv), 5), int(fb)])
    return dict(c=c, i0=i_start, i1=i_end, path=path, npath=npath, max_dv_end_diff=max_diff,
                n_fired=n_fired, n_sims=len(order))


# ----------------------------------------------------------------------------- driver
def build_tasks(seed, cultures, N, span, layers, chunk, times, i0_uA, tmp_out):
    tasks = []
    for c in cultures:
        for a in range(0, N, int(chunk)):
            tasks.append(dict(c=int(c), seed=int(seed), N=int(N), span=float(span),
                              layers=[float(L) for L in layers], i0=a, i1=min(N, a + int(chunk)),
                              times=[float(t) for t in times], i0_uA=float(i0_uA), tmp_out=tmp_out))
    return tasks


def merge_chunks(results, out_dir, seed):
    """One frames file per culture (chunks in neuron order) + one neurons file."""
    paths = []
    for c in sorted(set(r["c"] for r in results)):
        parts = sorted((r for r in results if r["c"] == c), key=lambda r: r["i0"])
        dst = os.path.join(out_dir, "culture_frames_S%d_C%d.csv" % (seed, c))
        with open(dst, "w", newline="") as fh:
            fh.write(",".join(FRAME_COLS) + "\n")
            for r in parts:
                with open(r["path"]) as src:
                    shutil.copyfileobj(src, fh)
        paths.append(dst)
    npath = os.path.join(out_dir, "culture_video_neurons_S%d.csv" % seed)
    with open(npath, "w", newline="") as fh:
        fh.write(",".join(NEURON_COLS) + "\n")
        for r in sorted(results, key=lambda r: (r["c"], r["i0"])):
            with open(r["npath"]) as src:
                shutil.copyfileobj(src, fh)
    return paths, npath


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-cultures", type=int, default=5)
    ap.add_argument("--first-culture", type=int, default=0, help="cultures c0 .. c0+n-1")
    ap.add_argument("--neurons", type=int, default=None, help="per culture (default: config)")
    ap.add_argument("--span-um", type=float, default=None,
                    help="half-side of the soma square (default: config.span_half_um())")
    ap.add_argument("--layers", default="80", help="comma list, e.g. 80 or 40,80,120")
    ap.add_argument("--seed", type=int, default=None, help="default: config.seed")
    ap.add_argument("--cell-model", default=None, help="default: config.cell_model")
    ap.add_argument("--tmax-ms", type=float, default=200.0, help="post-pulse window")
    ap.add_argument("--chunk", type=int, default=50, help="neurons per task")
    ap.add_argument("--processes", type=int, default=os.cpu_count() or 1,
                    help="worker processes (0 = run in this process, for tests)")
    ap.add_argument("--out", default="video_run")
    a = ap.parse_args(argv)

    from config import CFG
    import culture_export as CE
    seed = int(CFG.seed if a.seed is None else a.seed)
    N = int(CFG.n_neurons_effective() if a.neurons is None else a.neurons)
    span = float(CFG.span_half_um() if a.span_um is None else a.span_um)
    layers = [float(v) for v in str(a.layers).split(",")]
    cultures = list(range(a.first_culture, a.first_culture + a.n_cultures))
    cell_model = a.cell_model or getattr(CFG, "cell_model", None)
    times = frame_times(a.tmax_ms)
    os.makedirs(a.out, exist_ok=True)
    tmp_out = os.path.join(a.out, "_chunks_S%d" % seed)
    tmpdir = os.path.join(a.out, "_scratch_S%d" % seed)
    for p in (tmp_out, tmpdir):
        os.makedirs(p, exist_ok=True)
    g = _geometry(CFG)
    with open(os.path.join(a.out, "electrodes.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["x", "y", "sign"])
        for (ex, ey), s in zip(np.asarray(g["elec"], float), np.asarray(g["sign"], float)):
            w.writerow([round(float(ex), 2), round(float(ey), 2), int(np.sign(s))])

    tasks = build_tasks(seed, cultures, N, span, layers, a.chunk, times, CFG.i0_uA, tmp_out)
    n_sims = len(cultures) * N * len(layers)
    print("[video_frames] model %s | seed %d | cultures %s | %d neurons x %d layer(s) = %d sims | "
          "span +/-%.2f um | %d frames (-0.6 .. %g ms) | %d tasks on %d process(es)"
          % (cell_model, seed, cultures, N, len(layers), n_sims, span, times.size, a.tmax_ms,
             len(tasks), a.processes), flush=True)
    t0 = time.time()
    results = []
    if a.processes <= 0:
        _init_worker(cell_model, a.tmax_ms, tmpdir)
        for k, t in enumerate(tasks, 1):
            results.append(run_chunk(t))
            print("  task %d/%d done (%.0f s)" % (k, len(tasks), time.time() - t0), flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")                     # NEURON must not be forked
        with ctx.Pool(a.processes, initializer=_init_worker,
                      initargs=(cell_model, a.tmax_ms, tmpdir)) as pool:
            for k, r in enumerate(pool.imap_unordered(run_chunk, tasks), 1):
                results.append(r)
                print("  task %d/%d done (culture %d, neurons %d-%d, %.0f s)"
                      % (k, len(tasks), r["c"], r["i0"], r["i1"] - 1, time.time() - t0), flush=True)
    paths, npath = merge_chunks(results, a.out, seed)
    max_diff = max(r["max_dv_end_diff"] for r in results)
    shutil.rmtree(tmp_out, ignore_errors=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    manifest = dict(seed=seed, cultures=cultures, neurons=N, span_um=span, layers=layers,
                    cell_model=cell_model, tmax_ms=a.tmax_ms, n_frames=int(times.size),
                    pre_end_ms=PRE_END_MS, i0_uA=float(CFG.i0_uA), processes=a.processes,
                    chunk=a.chunk, n_sims=n_sims, n_fired=int(sum(r["n_fired"] for r in results)),
                    max_abs_dv_end_diff_vs_campaign_mV=max_diff, seconds=round(time.time() - t0),
                    frames_files=paths, neurons_file=npath)
    with open(os.path.join(a.out, "video_manifest_S%d.json" % seed), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("[video_frames] done in %.0f s -> %d frames file(s) in %s" % (time.time() - t0, len(paths), a.out))
    print("  consistency: max |DeltaV_end(video sham) - DeltaV_end(campaign sham)| = %.2e mV%s"
          % (max_diff, "" if max_diff < 1e-6 else "   <-- NOT identical: check before using"))
    return 0 if max_diff < 1e-6 else 3


if __name__ == "__main__":
    sys.exit(main())
