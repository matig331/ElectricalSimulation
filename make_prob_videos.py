"""
make_prob_videos.py -- time-resolved probability maps from culture_frames.csv.

Input (written by culture_export.py):
    culture_frames.csv : culture, morph, layer, x, y, t_ms, dVm_mV, fired_by_t
    electrodes.csv     : x, y, sign            (optional; fallback = default 3+3 layout)
t_ms is relative to the END of phase 2 (pulse = [-0.5, 0] ms; phase 1 = [-0.5,-0.25] ms, +I).

States at time t (mutually exclusive, as in culture_export):
    activated      : spiked by t
    depolarized    : not spiked by t  and  dVm(t) >  neu
    hyperpolarized : not spiked by t  and  dVm(t) < -neu
    neutral        : the rest
For every map point g:   P_s(g,t) = sum_i w_i 1[state_i(t)=s] / sum_i w_i,
w_i = exp(-|x_i - g|^2 / (2 sigma^2))  (Gaussian-kernel / Nadaraya-Watson mean),
so P_act + P_dep + P_hyp + P_neu = 1 wherever data exist.

MERGED layout (default): one map whose colour is the probability-weighted mixture
    colour = P_act*green + P_dep*red + P_hyp*blue + P_neu*white     (grey = no data)
plus contours of P_act (activation is rare), the stimulus current, and the EXPECTED % OF THE
SIMULATED AREA in each state over time:  A_s(t) = integral of P_s(x,y,t) dA  (no threshold;
the four areas sum to the covered area). --metric neurons shows % of neuron observations instead.
SPLIT layout (--layout split): three separate panels (as before).

Area % depends on the simulated region: it is a fraction of the covered (non-grey) square.

Run
    python make_prob_videos.py --frames "culture_frames*.csv"                  # pooled
    python make_prob_videos.py --frames "culture_frames*.csv" --cultures 0 1   # + 2 cultures
    python make_prob_videos.py --list                                          # list cultures
Smoke test: python smoke_prob_videos.py
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

REQUIRED = ["culture", "layer", "x", "y", "t_ms", "dVm_mV", "fired_by_t"]
PHASE_MS = 0.25          # per phase (config.phase_dur_ms)
RAMP_MS = 0.10           # config.ramp_us
ELEC_UM = 25.0           # config.electrode_um

COL = {"act": np.array([0.12, 0.62, 0.24]),
       "dep": np.array([0.86, 0.15, 0.15]),
       "hyp": np.array([0.12, 0.32, 0.86]),
       "neu": np.array([1.00, 1.00, 1.00])}
NODATA = np.array([0.86, 0.86, 0.86])
LABEL = {"act": "activated", "dep": "depolarized", "hyp": "hyperpolarized", "neu": "neutral"}


# ----------------------------------------------------------------------------- data
def load_frames(pattern):
    """Read one or many frames CSVs (HPC shards). Adds culture_id = unique (file, culture)."""
    has_glob = any(c in pattern for c in "*?[")
    files = sorted(glob.glob(pattern)) if has_glob else [pattern]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"no frames CSV matches {pattern!r}")
    parts = []
    for k, f in enumerate(files):
        d = pd.read_csv(f)
        miss = [c for c in REQUIRED if c not in d.columns]
        if miss:
            raise ValueError(f"{f}: missing columns {miss}")
        d["_file"] = k
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    # a culture = (seed, culture) when the files carry the seed (video_frames.py), so a culture
    # split over several files is one culture; otherwise (file, culture) as before
    key = ["seed", "culture"] if "seed" in df.columns else ["_file", "culture"]
    df["culture_id"] = df.groupby(key, sort=True).ngroup()
    return df, files


def load_electrodes(frames_files):
    """electrodes.csv next to the first frames file, else the default 3 anodes / 3 cathodes."""
    cand = os.path.join(os.path.dirname(os.path.abspath(frames_files[0])), "electrodes.csv")
    if os.path.isfile(cand):
        e = pd.read_csv(cand)
        return e[["x", "y"]].to_numpy(float), np.sign(e["sign"].to_numpy(float))
    h = 30.0
    ys = np.array([h, -h, -3 * h])
    xy = np.vstack([np.column_stack([np.full(3, -h), ys]),
                    np.column_stack([np.full(3, +h), ys])])
    return xy, np.array([1, 1, 1, -1, -1, -1], float)


def frozen_pulse_fraction(df):
    """Fraction of neurons whose dVm at mid phase 1 equals dVm at t=0.

    culture_export with pre_end_ms=0 records only t >= 0, so every pulse frame is filled with the
    t=0 value -> a 'frozen' pulse. Returns None if the frames do not cover mid phase 1."""
    ts = np.sort(df["t_ms"].unique())
    t_mid = ts[np.argmin(np.abs(ts + 0.375))]
    t_end = ts[np.argmin(np.abs(ts))]
    if abs(t_mid + 0.375) > 0.05 or t_mid == t_end:
        return None
    key = ["culture_id", "layer", "x", "y"]
    a = df.loc[df["t_ms"] == t_mid, key + ["dVm_mV"]]
    b = df.loc[df["t_ms"] == t_end, key + ["dVm_mV"]]
    m = a.merge(b, on=key, suffixes=("_mid", "_end"))
    if m.empty:
        return None
    return float(np.isclose(m["dVm_mV_mid"], m["dVm_mV_end"], atol=1e-6).mean())


# ----------------------------------------------------------------------------- physics helpers
def pulse_current(t_ms, phase=PHASE_MS, ramp=RAMP_MS):
    """Normalised biphasic current (+1 phase 1, -1 phase 2), t=0 = end of phase 2."""
    t = np.asarray(t_ms, float)
    r = min(ramp, phase / 2.0)

    def trap(a, b, amp):
        y = np.zeros_like(t)
        if r <= 0:
            y[(t >= a) & (t < b)] = amp
            return y
        up = (t >= a) & (t < a + r); y[up] = amp * (t[up] - a) / r
        pl = (t >= a + r) & (t < b - r); y[pl] = amp
        dn = (t >= b - r) & (t < b); y[dn] = amp * (b - t[dn]) / r
        return y

    t0 = -2.0 * phase
    return trap(t0, t0 + phase, 1.0) + trap(t0 + phase, t0 + 2 * phase, -1.0)


def states(dv, fired, neu):
    """Mutually exclusive state indicators at one instant."""
    fired = np.asarray(fired) > 0
    dv = np.asarray(dv, float)
    act = fired
    dep = (~fired) & (dv > neu)
    hyp = (~fired) & (dv < -neu)
    return act, dep, hyp


# ----------------------------------------------------------------------------- estimator
def kernel_map(x, y, value, edges, sigma_um, min_weight):
    """Gaussian-kernel mean of `value` on the grid. Returns (P[y,x], n_eff[y,x])."""
    bin_um = float(edges[1] - edges[0])
    cnt, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    num, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=np.asarray(value, float))
    s = sigma_um / bin_um
    if s > 0:
        cnt = gaussian_filter(cnt, s, mode="constant")
        num = gaussian_filter(num, s, mode="constant")
        # gaussian_filter is normalised -> cnt is a smoothed count PER BIN; the effective number
        # of neurons inside the kernel footprint is that density times the kernel area 2*pi*s^2.
        n_eff = cnt * (2.0 * np.pi * s * s)
    else:
        n_eff = cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        p = num / cnt
    p[n_eff < min_weight] = np.nan
    return p.T, n_eff.T        # histogram2d is [x, y]; imshow wants [row=y, col=x]


def compute_maps(df, half, bin_um, sigma_um, min_weight, neu):
    """Maps and state counts for every frame time.

    Returns times, maps{act,dep,hyp: (T,ny,nx)}, edges, n_eff (ny,nx), counts{act,dep,hyp,tot: (T,)}.
    """
    edges = np.arange(-half, half + bin_um / 2, bin_um)
    times = np.sort(df["t_ms"].unique())
    maps = {k: [] for k in ("act", "dep", "hyp")}
    counts = {k: [] for k in ("act", "dep", "hyp", "tot")}
    n_eff = None
    for t in times:
        d = df[df["t_ms"] == t]
        act, dep, hyp = states(d["dVm_mV"].to_numpy(), d["fired_by_t"].to_numpy(), neu)
        x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
        for key, v in (("act", act), ("dep", dep), ("hyp", hyp)):
            p, w = kernel_map(x, y, v, edges, sigma_um, min_weight)
            maps[key].append(p)
            counts[key].append(int(v.sum()))
        counts["tot"].append(int(len(d)))
        if n_eff is None:
            n_eff = w
    maps = {k: np.stack(v) for k, v in maps.items()}
    counts = {k: np.asarray(v) for k, v in counts.items()}
    return times, maps, edges, n_eff, counts


def mix_rgb(pa, pdp, ph):
    """Probability-weighted colour mixture of the four exclusive states; grey where no data."""
    pa, pdp, ph = (np.asarray(a, float) for a in (pa, pdp, ph))
    pn = np.clip(1.0 - (pa + pdp + ph), 0.0, 1.0)
    rgb = (pa[..., None] * COL["act"] + pdp[..., None] * COL["dep"]
           + ph[..., None] * COL["hyp"] + pn[..., None] * COL["neu"])
    rgb[~np.isfinite(pa)] = NODATA
    return np.clip(rgb, 0.0, 1.0)


def area_metrics(maps, edges):
    """Expected area in each state over the covered (non-grey) region, per frame.

    A_s(t) = sum over covered bins of P_s(g,t) * bin_area   [mm^2]  (= integral of P_s dA)
    A_cov  = covered area [mm^2];  the four expected areas sum to A_cov.
    No threshold is involved: it is the expected fraction of the simulated region in each state.
    """
    bin_mm2 = ((edges[1] - edges[0]) / 1000.0) ** 2
    covered = np.isfinite(maps["act"])
    out = {"cov": covered.sum(axis=(1, 2)) * bin_mm2}
    for k in ("act", "dep", "hyp"):
        out[k] = np.nansum(maps[k], axis=(1, 2)) * bin_mm2
    out["neu"] = np.maximum(out["cov"] - out["act"] - out["dep"] - out["hyp"], 0.0)
    return out


def _series(metric, counts, area):
    """Per-state percentage over time for the side panel, and its axis label."""
    if metric == "area":
        cov = np.maximum(area["cov"], 1e-12)
        return {k: 100 * area[k] / cov for k in ("act", "dep", "hyp", "neu")}, \
            "expected % of simulated area"
    tot = np.maximum(counts["tot"], 1)
    neu = counts["tot"] - counts["act"] - counts["dep"] - counts["hyp"]
    return {"act": 100 * counts["act"] / tot, "dep": 100 * counts["dep"] / tot,
            "hyp": 100 * counts["hyp"] / tot, "neu": 100 * neu / tot}, "% of neuron observations"


def _metric_text(metric, counts, area, k, density):
    if metric == "neurons":
        return _count_text(counts, k)
    cov = max(float(area["cov"][k]), 1e-12)
    txt = "expected area:   " + "   ".join(
        f"{LABEL[s]} {100*area[s][k]/cov:.1f}% ({area[s][k]:.3f} mm^2)" for s in ("act", "dep", "hyp"))
    txt += f"   of {cov:.2f} mm^2 simulated"
    if density:
        txt += f"   |   expected directly activated ~= rho*A = {density*area['act'][k]:.0f} (rho = {density:.0f}/mm^2)"
    return txt


# ----------------------------------------------------------------------------- shared drawing
def _phase_label(t):
    if t < -2 * PHASE_MS:
        return "before pulse"
    if t < -PHASE_MS:
        return "phase 1 (+I)"
    if t < 0:
        return "phase 2 (-I)"
    return "after pulse"


def _elec_colors(sign, i_now):
    """Effective polarity = electrode sign x current sign. Source red, sink blue, off grey."""
    if abs(i_now) < 1e-9:
        return ["0.55"] * len(sign)
    eff = np.sign(sign) * np.sign(i_now)
    return ["#d62728" if e > 0 else "#1f4fd6" for e in eff]


def _add_electrodes(ax, elec_xy):
    patches = []
    for (ex, ey) in elec_xy:
        r = Rectangle((ex - ELEC_UM / 2, ey - ELEC_UM / 2), ELEC_UM, ELEC_UM,
                      facecolor="0.55", edgecolor="k", lw=0.9, zorder=6)
        ax.add_patch(r)
        patches.append(r)
    return patches


def _time_axis(ax, times):
    """symlog when the frames extend well beyond the pulse, so pulse and tail are both visible."""
    if times.max() > 2.0:
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.2)
    ax.set_xlim(times.min(), times.max())


def _legend_bars(ax):
    """Three white->colour bars: how to read the mixture."""
    ramp = np.linspace(0, 1, 256)
    rows = []
    for key in ("act", "dep", "hyp"):
        rows.append((1 - ramp)[:, None] * COL["neu"] + ramp[:, None] * COL[key])
    img = np.stack(rows)                      # (3, 256, 3)
    ax.imshow(img, aspect="auto", extent=[0, 1, 3, 0])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["P(activation)", "P(depolarization)", "P(hyperpolarization)"], fontsize=8)
    ax.set_xticks([0, 0.5, 1.0]); ax.tick_params(axis="x", labelsize=8)
    ax.set_title("colour = probability-weighted mix of states  (white = neutral, grey = no data)",
                 fontsize=8)


def _count_text(counts, k):
    tot = max(int(counts["tot"][k]), 1)
    na, nd, nh = (int(counts[s][k]) for s in ("act", "dep", "hyp"))
    nn = tot - na - nd - nh
    return (f"activated {na} ({100*na/tot:.1f}%)   depolarized {nd} ({100*nd/tot:.1f}%)   "
            f"hyperpolarized {nh} ({100*nh/tot:.1f}%)   neutral {nn} ({100*nn/tot:.1f}%)"
            f"   of {tot} neuron observations")


def _save(anim, out_path, fps):
    if animation.writers.is_available("ffmpeg"):
        path = out_path + ".mp4"
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=3000), dpi=110)
    else:
        path = out_path + ".gif"
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=90)
    return path


def _frame_sequence(n, fps, hold_end_s):
    return list(range(n)) + [n - 1] * int(round(hold_end_s * fps))


# ----------------------------------------------------------------------------- merged layout
def render_merged(times, maps, counts, edges, elec_xy, elec_sign, out_path, tag, fps,
                  act_levels, hold_end_s, metric="area", area=None, density=None):
    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.0], height_ratios=[0.5, 1.25, 0.42],
                          wspace=0.22, hspace=0.55)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_i = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[1, 1], sharex=ax_i)
    ax_leg = fig.add_subplot(gs[2, 1])

    ext = [edges[0], edges[-1], edges[0], edges[-1]]
    xc = 0.5 * (edges[:-1] + edges[1:])
    XC, YC = np.meshgrid(xc, xc)
    im = ax_map.imshow(mix_rgb(maps["act"][0], maps["dep"][0], maps["hyp"][0]),
                       origin="lower", extent=ext, interpolation="bilinear")
    patches = _add_electrodes(ax_map, elec_xy)
    ax_map.set_aspect("equal"); ax_map.set_xlabel("x (um)"); ax_map.set_ylabel("y (um)")
    handles = [Patch(fc=COL["act"], label="activated"), Patch(fc=COL["dep"], label="depolarized"),
               Patch(fc=COL["hyp"], label="hyperpolarized"),
               Patch(fc="white", ec="0.5", label="neutral"),
               Patch(fc=NODATA, label="no data")]
    if act_levels:
        handles.append(Line2D([0], [0], color="k", lw=1.4,
                              label="P(activation) = " + ", ".join(f"{v:g}" for v in act_levels)))
    ax_map.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9)

    # stimulus current (shared, possibly symlog, time axis)
    tt = np.unique(np.r_[np.linspace(times.min(), min(times.max(), 0.7), 800), times])
    ax_i.plot(tt, pulse_current(tt), "k", lw=1.3)
    ax_i.axvspan(-0.5, -0.25, color="#fde0dd", zorder=0)
    ax_i.axvspan(-0.25, 0.0, color="#deebf7", zorder=0)
    ax_i.axhline(0, color="0.7", lw=0.6)
    ax_i.set_ylim(-1.35, 1.35); ax_i.set_yticks([-1, 0, 1])
    ax_i.set_yticklabels(["-I0", "0", "+I0"], fontsize=8)
    ax_i.set_title("stimulus current", fontsize=9)
    plt.setp(ax_i.get_xticklabels(), visible=False)

    # states over time: expected % of the simulated AREA (default) or % of neuron observations
    ser, ylab = _series(metric, counts, area)
    for key in ("act", "dep", "hyp"):
        ax_n.plot(times, ser[key], color=COL[key], lw=2.0, label=LABEL[key])
    ax_n.plot(times, ser["neu"], color="0.5", lw=1.2, ls="--", label="neutral")
    ax_n.axvspan(-0.5, 0.0, color="0.93", zorder=0)
    ax_n.set_ylim(0, 100); ax_n.set_ylabel(ylab)
    ax_n.set_xlabel("time from end of phase 2 (ms)" +
                    ("  [linear |t|<1 ms, log beyond]" if times.max() > 2.0 else ""))
    ax_n.legend(fontsize=8, loc="center right"); ax_n.grid(alpha=0.2)
    _time_axis(ax_n, times)
    cursors = [ax_i.axvline(times[0], color="C1", lw=2.0), ax_n.axvline(times[0], color="C1", lw=2.0)]

    _legend_bars(ax_leg)
    sup = fig.suptitle("", fontsize=12)
    note = fig.text(0.5, 0.015, "", ha="center", fontsize=9)
    contour = {"cs": None}

    def update(k):
        t = float(times[k])
        pa = maps["act"][k]
        im.set_data(mix_rgb(pa, maps["dep"][k], maps["hyp"][k]))
        if contour["cs"] is not None:
            contour["cs"].remove(); contour["cs"] = None
        if act_levels and np.nanmax(np.nan_to_num(pa)) >= min(act_levels):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                contour["cs"] = ax_map.contour(XC, YC, np.nan_to_num(pa), levels=sorted(act_levels),
                                               colors="k", linewidths=1.3, zorder=5)
        for p, c in zip(patches, _elec_colors(elec_sign, float(pulse_current([t])[0]))):
            p.set_facecolor(c)
        for c in cursors:
            c.set_xdata([t, t])
        sup.set_text(f"{tag}   |   t = {t:+.3f} ms   |   {_phase_label(t)}")
        note.set_text(_metric_text(metric, counts, area, k, density))
        return [im, sup, note] + cursors

    anim = animation.FuncAnimation(fig, update, frames=_frame_sequence(len(times), fps, hold_end_s),
                                   blit=False)
    path = _save(anim, out_path, fps)
    plt.close(fig)
    return path


def snapshots_merged(times, maps, counts, edges, elec_xy, elec_sign, out_png, tag, act_levels,
                     metric="area", area=None):
    """Four key instants for slides: mid phase 1, mid phase 2, +2 ms, last frame."""
    wanted = [-0.375, -0.125, 2.0, float(times[-1])]
    idx = sorted(set(int(np.argmin(np.abs(times - w))) for w in wanted))
    ext = [edges[0], edges[-1], edges[0], edges[-1]]
    xc = 0.5 * (edges[:-1] + edges[1:]); XC, YC = np.meshgrid(xc, xc)
    fig, axes = plt.subplots(1, len(idx), figsize=(4.6 * len(idx), 5.4))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, idx):
        t = float(times[k]); pa = maps["act"][k]
        ax.imshow(mix_rgb(pa, maps["dep"][k], maps["hyp"][k]), origin="lower", extent=ext,
                  interpolation="bilinear")
        if act_levels and np.nanmax(np.nan_to_num(pa)) >= min(act_levels):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(XC, YC, np.nan_to_num(pa), levels=sorted(act_levels), colors="k",
                           linewidths=1.2)
        for (ex, ey), c in zip(elec_xy, _elec_colors(elec_sign, float(pulse_current([t])[0]))):
            ax.add_patch(Rectangle((ex - ELEC_UM / 2, ey - ELEC_UM / 2), ELEC_UM, ELEC_UM,
                                   facecolor=c, edgecolor="k", lw=0.8, zorder=6))
        ser, _ = _series(metric, counts, area)
        unit = "area" if metric == "area" else "neurons"
        ax.set_title(f"t = {t:+.3f} ms  ({_phase_label(t)})\n{unit}: act {ser['act'][k]:.1f}%  "
                     f"dep {ser['dep'][k]:.1f}%  hyp {ser['hyp'][k]:.1f}%", fontsize=9)
        ax.set_aspect("equal"); ax.set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    handles = [Patch(fc=COL["act"], label="activated"), Patch(fc=COL["dep"], label="depolarized"),
               Patch(fc=COL["hyp"], label="hyperpolarized"),
               Patch(fc="white", ec="0.5", label="neutral"), Patch(fc=NODATA, label="no data")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle(tag, fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- split layout
PANELS = (("act", "P(activation)", "Greens"),
          ("dep", "P(depolarization)", "Reds"),
          ("hyp", "P(hyperpolarization)", "Blues"))


def render_split(times, maps, counts, edges, elec_xy, elec_sign, out_path, tag, fps, hold_end_s,
                 metric="area", area=None, density=None):
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.22, 1.0], hspace=0.35)
    ax_i = fig.add_subplot(gs[0, :])
    tt = np.unique(np.r_[np.linspace(times.min(), min(times.max(), 0.7), 800), times])
    ax_i.plot(tt, pulse_current(tt), "k", lw=1.5)
    ax_i.axvspan(-0.5, -0.25, color="#fde0dd", zorder=0)
    ax_i.axvspan(-0.25, 0.0, color="#deebf7", zorder=0)
    ax_i.set_ylim(-1.35, 1.35); ax_i.set_yticks([-1, 0, 1])
    _time_axis(ax_i, times)
    cursor = ax_i.axvline(times[0], color="C1", lw=2.0)
    ext = [edges[0], edges[-1], edges[0], edges[-1]]
    ims, elec_patches = {}, []
    for j, (key, title, cmap) in enumerate(PANELS):
        ax = fig.add_subplot(gs[1, j])
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.86")
        ims[key] = ax.imshow(maps[key][0], origin="lower", extent=ext, cmap=cm, vmin=0, vmax=1,
                             interpolation="bilinear")
        elec_patches.append(_add_electrodes(ax, elec_xy))
        ax.set_title(title, fontsize=11); ax.set_aspect("equal"); ax.set_xlabel("x (um)")
        fig.colorbar(ims[key], ax=ax, fraction=0.046, pad=0.03)
    sup = fig.suptitle("", fontsize=12)
    note = fig.text(0.5, 0.01, "", ha="center", fontsize=9)

    def update(k):
        t = float(times[k])
        for key, _, _ in PANELS:
            ims[key].set_data(maps[key][k])
        cols = _elec_colors(elec_sign, float(pulse_current([t])[0]))
        for patches in elec_patches:
            for p, c in zip(patches, cols):
                p.set_facecolor(c)
        cursor.set_xdata([t, t])
        sup.set_text(f"{tag}   |   t = {t:+.3f} ms   |   {_phase_label(t)}")
        note.set_text(_metric_text(metric, counts, area, k, density))
        return list(ims.values()) + [cursor, sup, note]

    anim = animation.FuncAnimation(fig, update, frames=_frame_sequence(len(times), fps, hold_end_s),
                                   blit=False)
    path = _save(anim, out_path, fps)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- driver
def make_one(df, tag, out_stem, elec_xy, elec_sign, args):
    x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)
    data_half = float(max(np.abs(x).max(), np.abs(y).max()))
    if data_half < 0.8 * args.half:
        print(f"  NOTE: somata only span +/-{data_half:.0f} um; outside that the map is 'no data'.")
    times, maps, edges, _, counts = compute_maps(df, args.half, args.bin, args.sigma,
                                                 args.min_weight, args.neu)
    area = area_metrics(maps, edges)
    dens = args.density if args.expected_neurons else None
    paths = []
    if args.layout in ("merged", "both"):
        paths.append(render_merged(times, maps, counts, edges, elec_xy, elec_sign, out_stem, tag,
                                   args.fps, args.act_contours, args.hold_end,
                                   args.metric, area, dens))
        snapshots_merged(times, maps, counts, edges, elec_xy, elec_sign,
                         out_stem.replace("prob_video", "prob_snapshots") + ".png", tag,
                         args.act_contours, args.metric, area)
    if args.layout in ("split", "both"):
        paths.append(render_split(times, maps, counts, edges, elec_xy, elec_sign,
                                  out_stem + "_split", tag, args.fps, args.hold_end,
                                  args.metric, area, dens))
    print(f"  {tag}: simulated (covered) area {area['cov'][0]:.2f} mm^2")
    n_obs = int(counts["tot"][0])
    post = int((times > 0).sum())
    print(f"  {tag}: {len(times)} frames ({post} after the pulse, up to {times.max():+.1f} ms), "
          f"{n_obs} observations/frame -> {', '.join(paths)}")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="culture_frames.csv", help="file or glob of shards")
    ap.add_argument("--cultures", type=int, nargs="*", default=[],
                    help="culture_id(s) for single-culture videos (see --list)")
    ap.add_argument("--layer", type=float, default=None, help="restrict to one layer (default: pool)")
    ap.add_argument("--layout", choices=("merged", "split", "both"), default="merged")
    ap.add_argument("--half", type=float, default=400.0, help="map half-width (um)")
    ap.add_argument("--bin", type=float, default=10.0, help="grid bin (um)")
    ap.add_argument("--sigma", type=float, default=20.0, help="kernel width (um); raise for one culture")
    ap.add_argument("--min-weight", type=float, default=3.0,
                    help="min effective n of neurons in the kernel to show a value")
    ap.add_argument("--neu", type=float, default=1.0,
                    help="|dVm| below this (mV) = neutral; 1.0 = culture_statistics threshold")
    ap.add_argument("--act-contours", type=float, nargs="*", default=[0.1, 0.5],
                    help="P(activation) contour levels on the merged map (none: pass no values)")
    ap.add_argument("--hold-end", type=float, default=2.0, help="seconds to hold the last frame")
    ap.add_argument("--metric", choices=("area", "neurons"), default="area",
                    help="side panel/caption: expected %% of simulated area (default) or %% of neurons")
    ap.add_argument("--expected-neurons", action="store_true",
                    help="also print rho*A_act, the expected number of directly activated neurons")
    ap.add_argument("--density", type=float, default=None,
                    help="neurons/mm^2 for --expected-neurons (default: config.density_per_mm2)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--list", action="store_true", help="list cultures and exit")
    ap.add_argument("--allow-frozen", action="store_true",
                    help="render even if pulse frames look frozen (pre_end_ms=0 export)")
    args = ap.parse_args(argv)

    if args.expected_neurons and args.density is None:
        try:
            from config import CFG
            args.density = float(CFG.density_per_mm2)
        except Exception:
            raise SystemExit("[videos] --expected-neurons needs --density (config.py not found)")
    df, files = load_frames(args.frames)
    if args.layer is not None:
        df = df[np.isclose(df["layer"].astype(float), args.layer)]
    elec_xy, elec_sign = load_electrodes(files)
    print(f"[videos] {len(files)} file(s) | {df['culture_id'].nunique()} cultures | "
          f"{df['t_ms'].nunique()} frame times ({df['t_ms'].min():+.3f} .. {df['t_ms'].max():+.3f} ms)")
    if args.list:
        print(df.groupby("culture_id")[["_file", "culture"]].first().to_string())
        return []
    fz = frozen_pulse_fraction(df)
    if fz is not None and fz > 0.9:
        msg = (f"{100*fz:.0f}% of neurons have identical dVm at mid phase 1 and at t=0: the pulse "
               f"frames are FROZEN. Re-export with pre_end_ms=1.0 in culture_export.py "
               f"(spikes_at call). Use --allow-frozen to render anyway.")
        if not args.allow_frozen:
            raise SystemExit("[videos] STOP: " + msg)
        print("[videos] WARNING: " + msg)
    if df["t_ms"].max() < 1.0:
        print("[videos] NOTE: frames stop at "
              f"{df['t_ms'].max():+.2f} ms -> almost nothing after the pulse. Extend FRAME_TIMES_MS "
              "and post_ms in culture_export.py to see the post-pulse evolution.")
    os.makedirs(args.outdir, exist_ok=True)
    lay = "all layers" if args.layer is None else f"layer {args.layer:.0f} um"
    paths = make_one(df, f"All cultures ({lay})", os.path.join(args.outdir, "prob_video_all"),
                     elec_xy, elec_sign, args)
    for c in args.cultures:
        sub = df[df["culture_id"] == c]
        if sub.empty:
            print(f"  culture_id {c}: not found, skipped"); continue
        paths += make_one(sub, f"Culture {c} ({lay})",
                          os.path.join(args.outdir, f"prob_video_culture{c}"),
                          elec_xy, elec_sign, args)
    return paths


if __name__ == "__main__":
    main()
