"""
make_culture_videos.py -- one video per culture: the PHYSICAL scene plus its probabilities.

Purpose: show that the rule estimated on the whole ensemble of cultures also holds on single,
independent cultures. Each video shows ONE culture at ONE layer:

  map (left)
    background : extracellular potential Ve(x,y,t) of the electrode array (purple negative,
                 orange positive, symlog scale). It follows the biphasic current: one pattern in phase 1, the
                 reversed pattern in phase 2, zero after the pulse.
                 Ve = I0 * p(t) * sum_i s_i / (2 pi sigma r_i),  r_i = sqrt(dx^2 + dy^2 + h^2)
                 (same point-source half-space model as field.py, at the soma height h)
    electrodes : coloured by effective polarity (red source, blue sink, grey off)
    morphologies of the culture's neurons, coloured by specimen (the n closest to the array)
    somata     : one marker per neuron, edge = specimen. Default (--soma-color dvm): green star
                 = activated; every other soma filled by its dVm(t) on a continuous gradient
                 (red depolarized, blue hyperpolarized, white ~ 0). --soma-color state: discrete
                 states with the probability-video colours and shapes.
  right
    stimulus current with time cursor
    % of neurons in each state over time: THIS culture (solid) vs ALL cultures (dashed)
    P(state | angle to the dipole axis) at time t (default; --panel r for distance): THIS
      culture (points, Wilson 95% CI) vs ALL cultures (lines). The angle shows the field
      polarity: depolarisation on the cathode side in phase 1, on the anode side in phase 2.

Input: frames from culture_export_video.py (video_run/culture_frames*.csv, with theta_deg) and
video_run/electrodes.csv. Without theta_deg only somata are drawn. Morphologies are read with the
project's slicer (morphologies.py + slicer.py must be importable, i.e. run from the project folder).

--n-somata N draws only N neurons of the culture (uniform at random, --somata-seed): the map, the
"this culture" curves and points are those N neurons, the reference lines all the simulated ones.
Use it when a culture was simulated with many neurons for a smooth probability video.
--morph-fraction f: only a random fraction f of the drawn somata get their arbor (1 = all).
--style plain (default) / specimen: see BRANCH_GREY below.

Run
    python make_culture_videos.py --frames "video_run/culture_frames*.csv" --list
    python make_culture_videos.py --frames "video_run/culture_frames*.csv" --cultures 0 7 --layer 80
    python make_culture_videos.py --frames "video_run/culture_frames*.csv" --cultures 0 --n-somata 2000
Smoke test: python smoke_culture_videos.py
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import make_prob_videos as P          # shared: loading, pulse, states, colours, guards

SPEC_COLORS = {"60308": "#111111", "130303": "#1b9e77", "60303": "#e7298a"}
EXTRA_COLORS = ["#7f7f7f", "#a6761d", "#1f78b4", "#b2df8a", "#666666"]
FIELD_CMAP = "PuOr_r"     # purple = negative Ve, orange = positive Ve (red/blue/green are the states)
DVM_CMAP = "RdBu_r"       # soma dVm: red = depolarized, blue = hyperpolarized, white ~ 0


def dvm_norm(vmax):
    return SymLogNorm(linthresh=0.5, vmin=-vmax, vmax=vmax, base=10)


def dvm_rgba(values, vmax):
    """Colour of a non-spiking soma for a given dVm (mV)."""
    return plt.get_cmap(DVM_CMAP)(dvm_norm(vmax)(np.asarray(values, float)))
MARK = {"act": ("*", 150), "dep": ("^", 50), "hyp": ("v", 50), "neu": ("o", 20)}
# --style plain (default): the specimen does not matter -- branches light grey, every soma a small
# circle with no edge, filled by its dVm(t) (activated: green). --style specimen: the old look.
BRANCH_GREY = "#c4c4c4"
SOMA_S = 12               # plain style: soma marker area (pt^2)
NEUTRAL_FILL = "0.75"     # plain style, --soma-color state: neutral somata (white would vanish)


# ----------------------------------------------------------------------------- parameters
def physics_params():
    """Field parameters from config.py when available (same values used by the simulation)."""
    d = dict(sigma_Sm=1.5, rmin_um=12.5, h_soma_um=10.0, i0_uA=50.0, electrode_um=25.0,
             source="defaults")
    try:
        from config import CFG
        d.update(sigma_Sm=float(CFG.sigma_Sm), rmin_um=float(CFG.rmin_um),
                 h_soma_um=float(CFG.h_soma_um), i0_uA=float(CFG.i0_uA),
                 electrode_um=float(CFG.electrode_um), source="config.py")
    except Exception:
        pass
    return d


def spec_color(name, _cache={}):
    name = str(name)
    if name in SPEC_COLORS:
        return SPEC_COLORS[name]
    if name not in _cache:
        _cache[name] = EXTRA_COLORS[len(_cache) % len(EXTRA_COLORS)]
    return _cache[name]


# ----------------------------------------------------------------------------- physics
def field_mV_per_A(X, Y, elec_xy, sign, sigma_Sm, rmin_um, h_um):
    """Point-source half-space transfer (mV per A) at height h: sum_i s_i / (2 pi sigma r_i)."""
    g = np.zeros_like(np.asarray(X, float))
    for (ex, ey), s in zip(np.asarray(elec_xy, float), np.asarray(sign, float)):
        r = np.maximum(np.sqrt((X - ex) ** 2 + (Y - ey) ** 2 + h_um ** 2), rmin_um)
        g = g + s / (2.0 * np.pi * sigma_Sm * (r * 1e-6))
    return g * 1e3


def dipole_centre(elec_xy, sign):
    e = np.asarray(elec_xy, float); s = np.asarray(sign, float)
    if (s > 0).any() and (s < 0).any():
        return 0.5 * (e[s > 0].mean(axis=0) + e[s < 0].mean(axis=0))
    return e.mean(axis=0)


def dipole_theta_deg(x, y, elec_xy, sign):
    """Angle between soma position (from the dipole centre) and the dipole axis, in [0, 180] deg.
    Same convention as culture_export: d points cathode -> anode, 0 deg = anode side,
    180 deg = cathode side: theta = atan2(|v x d|, v . d)."""
    e = np.asarray(elec_xy, float); s = np.asarray(sign, float)
    c = dipole_centre(e, s)
    d = e[s > 0].mean(axis=0) - e[s < 0].mean(axis=0)
    d = d / np.linalg.norm(d)
    vx = np.asarray(x, float) - c[0]; vy = np.asarray(y, float) - c[1]
    return np.degrees(np.arctan2(np.abs(vx * d[1] - vy * d[0]), vx * d[0] + vy * d[1]))


def rotate_translate(pts, x, y, theta_deg):
    """Rotate soma-relative 2D points by theta (deg, counter-clockwise) and move to (x, y).
    Same convention as rich_footprint._placed."""
    th = np.radians(theta_deg); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(pts, float) @ R.T + np.array([x, y], float)


def wilson(k, n, z=1.959963984540054):
    k = np.asarray(k, float); n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


# ----------------------------------------------------------------------------- data
def load_morph_polylines(names, layer):
    """{specimen: [ (n,2) soma-relative polylines ]} from the project's slicer, or {}."""
    try:
        from morphologies import find_one_morphology
        from slicer import sliced_polylines
    except Exception as e:
        print(f"  NOTE: morphologies/slicer not importable ({e}); drawing somata only.")
        return {}
    out = {}
    for m in names:
        try:
            polys = sliced_polylines(find_one_morphology(str(m)), float(layer))
            out[str(m)] = [np.asarray(p, float) for p, _ in polys if len(p) >= 2]
        except Exception as e:
            print(f"  NOTE: morphology {m} not loaded ({e}).")
    return out


def neuron_matrices(sub):
    """One culture, one layer -> times, neuron table, dV[T, n], fired[T, n]."""
    sub = sub.copy()
    sub["morph"] = sub["morph"].astype(str)
    sub["nid"] = sub.groupby(["x", "y", "morph"], sort=True).ngroup()
    if sub.groupby(["nid", "t_ms"]).size().max() > 1:          # duplicated (x, y, morph): use file order
        nt = sub["t_ms"].nunique()
        sub["nid"] = np.arange(len(sub)) // nt
    cols = ["nid", "x", "y", "morph"] + (["theta_deg"] if "theta_deg" in sub.columns else [])
    tab = sub.drop_duplicates("nid")[cols].sort_values("nid").reset_index(drop=True)
    dv = sub.pivot(index="t_ms", columns="nid", values="dVm_mV").sort_index()
    fb = sub.pivot(index="t_ms", columns="nid", values="fired_by_t").sort_index()
    return dv.index.to_numpy(float), tab, dv.to_numpy(float), fb.to_numpy(float)


def state_counts_vs_r(df, times, centre, h_um, edges, neu):
    """Counts per (time, distance-from-dipole-centre bin) -> dict of arrays [T, nbins]."""
    r = np.sqrt((df["x"].to_numpy(float) - centre[0]) ** 2 +
                (df["y"].to_numpy(float) - centre[1]) ** 2 + h_um ** 2)
    return state_counts_vs(r, df, times, edges, neu)


def state_counts_vs(coord, df, times, edges, neu):
    """Counts per (time, coordinate bin): n, act, dep, hyp  -> dict of arrays [T, nbins]."""
    r = np.asarray(coord, float)
    act, dep, hyp = P.states(df["dVm_mV"].to_numpy(), df["fired_by_t"].to_numpy(), neu)
    ti = np.searchsorted(times, df["t_ms"].to_numpy(float))
    bi = np.digitize(r, edges) - 1
    ok = (bi >= 0) & (bi < len(edges) - 1) & (ti < len(times))
    out = {}
    for key, v in (("n", np.ones_like(act)), ("act", act), ("dep", dep), ("hyp", hyp)):
        a = np.zeros((len(times), len(edges) - 1))
        np.add.at(a, (ti[ok], bi[ok]), np.asarray(v, float)[ok])
        out[key] = a
    return out


MORPH_IMAGE_ABOVE = 100   # more morphologies than this: drawn once into an image (_arbor_image)


def _arbor_image(layers, half, px=1600):
    """RGBA image (transparent background) of the given polylines over [-half, half]^2:
    layers = [(segments, colour, linewidth_pt, alpha)]. The off-screen axes are 8 in wide, about
    the map's width in the video, so line widths look the same as when drawn directly."""
    dpi = px / 8.0
    fig = plt.figure(figsize=(8.0, 8.0), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.axis("off")
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    for segs, col, w, a in layers:
        ax.add_collection(LineCollection(segs, colors=col, linewidths=w, alpha=a))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def state_counts_matrix(coord, dv, fb, edges, neu):
    """state_counts_vs on the neuron x time matrices of neuron_matrices (dv, fb [T, n]) with one
    coordinate per neuron: the same counts, for any subset of the culture's neurons."""
    r = np.asarray(coord, float)
    bi = np.digitize(r, edges) - 1
    ok = (bi >= 0) & (bi < len(edges) - 1)
    T, nb = dv.shape[0], len(edges) - 1
    out = {k: np.zeros((T, nb)) for k in ("n", "act", "dep", "hyp")}
    for k in range(T):
        act, dep, hyp = P.states(dv[k], fb[k], neu)
        for key, v in (("n", np.ones(act.size)), ("act", act), ("dep", dep), ("hyp", hyp)):
            np.add.at(out[key][k], bi[ok], np.asarray(v, float)[ok])
    return out


# ----------------------------------------------------------------------------- rendering
def _draw_scene(ax, half, grid_um, g_grid, vmax, elec_xy, electrode_um, tab, polys, n_morph,
                centre, h_um, dvm_max=20.0, style="specimen", morph_fraction=1.0, morph_seed=0):
    """n_morph >= 0: the arbors of the n_morph somata closest to the array; n_morph < 0 (or
    None): the arbors of a random fraction morph_fraction of the drawn somata (seed morph_seed,
    so the animation and the snapshots show the same ones)."""
    plain = style == "plain"
    ext = [-half, half, -half, half]
    norm = SymLogNorm(linthresh=1.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(np.zeros_like(g_grid), origin="lower", extent=ext, cmap=FIELD_CMAP,
                   norm=norm, alpha=0.85, interpolation="bilinear", zorder=1)
    drawn = 0
    if polys and "theta_deg" in tab.columns:
        r = np.hypot(tab["x"] - centre[0], tab["y"] - centre[1]).to_numpy()
        if n_morph is not None and int(n_morph) >= 0:
            pick = np.argsort(r)[:int(n_morph)]
        else:
            k_m = int(min(max(round(float(morph_fraction) * len(tab)), 0), len(tab)))
            pick = (np.arange(len(tab)) if k_m == len(tab) else
                    np.sort(np.random.default_rng(morph_seed).choice(len(tab), size=k_m,
                                                                     replace=False)))
        segs = {}
        for i in pick:
            row = tab.iloc[i]
            for p in polys.get(str(row["morph"]), []):
                segs.setdefault("all" if plain else str(row["morph"]), []).append(
                    rotate_translate(p, row["x"], row["y"], row["theta_deg"]))
            drawn += 1
        # many morphologies (every drawn soma has its own): thinner, lighter lines, so the field
        # and the somata stay visible through the arbor
        lw, al = (0.6, 0.9) if drawn <= 100 else ((0.4, 0.7) if drawn <= 600 else (0.3, 0.55))
        layers = [(s, BRANCH_GREY if plain else spec_color(m), lw if plain else 0.6,
                   al if plain else 0.55) for m, s in segs.items()]
        if drawn > MORPH_IMAGE_ABOVE:
            # the arbor never changes: draw it ONCE into a transparent image instead of redrawing
            # ~10^5 polylines in every frame (2000 morphologies: ~8 s -> well under 1 s a frame)
            ax.imshow(_arbor_image(layers, half), extent=ext, origin="upper",
                      interpolation="antialiased", zorder=3)
        else:
            for s, col, w, a in layers:
                ax.add_collection(LineCollection(s, colors=col, linewidths=w, alpha=a, zorder=3))
    patches = []
    for (ex, ey) in elec_xy:
        rct = Rectangle((ex - electrode_um / 2, ey - electrode_um / 2), electrode_um, electrode_um,
                        facecolor="0.55", edgecolor="k", lw=0.9, zorder=6)
        ax.add_patch(rct); patches.append(rct)
    scat = {}
    for key, (mk, sz) in MARK.items():
        if plain:
            scat[key] = ax.scatter([], [], marker="o", s=SOMA_S, linewidths=0.0,
                                   zorder=6 if key == "act" else 5)
        else:
            scat[key] = ax.scatter([], [], marker=mk, s=sz, linewidths=0.6, zorder=5)
    scat["grad"] = ax.scatter([], [], c=[], cmap=DVM_CMAP, norm=dvm_norm(dvm_max), marker="o",
                              s=SOMA_S if plain else 40, linewidths=0.0 if plain else 1.3,
                              zorder=5)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    return im, patches, scat, drawn


def _update_somata(scat, tab, st, dvk=None, mode="dvm", style="specimen"):
    """style 'specimen': edge = specimen; style 'plain': no edge. mode 'dvm': activated = green
    (star, or a circle in the plain style), the others filled by their dVm on a continuous
    red-blue gradient. mode 'state': fill = discrete state (as the probability video)."""
    xy = tab[["x", "y"]].to_numpy(float)
    plain = style == "plain"
    edge = None if plain else np.array([spec_color(m) for m in tab["morph"]])
    empty = np.empty((0, 2))

    def edges(sc, m):
        if plain:
            sc.set_edgecolor("none"); sc.set_linewidth(0.0)
        else:
            sc.set_edgecolor(edge[m] if m.any() else []); sc.set_linewidth(1.3)
    if mode == "dvm":
        a = st["act"]
        scat["act"].set_offsets(xy[a] if a.any() else empty)
        scat["act"].set_facecolor([P.COL["act"]] * int(a.sum()) if a.any() else [])
        edges(scat["act"], a)
        g = ~a
        scat["grad"].set_offsets(xy[g] if g.any() else empty)
        scat["grad"].set_array(np.asarray(dvk, float)[g] if g.any() else np.array([]))
        edges(scat["grad"], g)
        for key in ("dep", "hyp", "neu"):
            scat[key].set_offsets(empty)
        return
    scat["grad"].set_offsets(empty); scat["grad"].set_array(np.array([]))
    fill = {"act": P.COL["act"], "dep": P.COL["dep"], "hyp": P.COL["hyp"],
            "neu": NEUTRAL_FILL if plain else "white"}
    for key in ("act", "dep", "hyp", "neu"):
        m = st[key]
        scat[key].set_offsets(xy[m] if m.any() else empty)
        scat[key].set_facecolor([fill[key]] * int(m.sum()) if m.any() else [])
        edges(scat[key], m)


def _scene_legend(ax, specimens, mode="dvm", style="specimen"):
    if style == "plain":
        def dot(c, lab):
            return Line2D([0], [0], marker="o", ls="", mfc=c, mec="none", ms=6, label=lab)
        h = [Line2D([0], [0], color=BRANCH_GREY, lw=2, label="dendrites / axon")]
        if mode == "dvm":
            h += [dot(P.COL["act"], "activated"), dot(P.COL["dep"], "soma: dVm > 0"),
                  dot(P.COL["hyp"], "soma: dVm < 0")]
        else:
            h += [dot(P.COL["act"], "activated"), dot(P.COL["dep"], "depolarized"),
                  dot(P.COL["hyp"], "hyperpolarized"), dot(NEUTRAL_FILL, "neutral")]
        ax.legend(handles=h, loc="upper left", fontsize=7, framealpha=0.9)
        return
    h = [Line2D([0], [0], color=spec_color(m), lw=2, label=f"specimen {m} (line / soma edge)")
         for m in specimens]
    if mode == "dvm":
        h += [Line2D([0], [0], marker="*", ls="", mfc=P.COL["act"], mec="k", ms=12, label="activated"),
              Line2D([0], [0], marker="o", ls="", mfc=P.COL["dep"], mec="k", ms=7, label="soma: dVm > 0"),
              Line2D([0], [0], marker="o", ls="", mfc=P.COL["hyp"], mec="k", ms=7, label="soma: dVm < 0")]
        ax.legend(handles=h, loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
        return
    h += [Line2D([0], [0], marker="*", ls="", mfc=P.COL["act"], mec="k", ms=12, label="activated"),
          Line2D([0], [0], marker="^", ls="", mfc=P.COL["dep"], mec="k", ms=8, label="depolarized"),
          Line2D([0], [0], marker="v", ls="", mfc=P.COL["hyp"], mec="k", ms=8, label="hyperpolarized"),
          Line2D([0], [0], marker="o", ls="", mfc="white", mec="k", ms=6, label="neutral")]
    ax.legend(handles=h, loc="upper left", fontsize=7, framealpha=0.9, ncol=2)


def render_culture(cid, times, tab, dv, fb, polys, ens, cul, edges_r, elec_xy, elec_sign, ph,
                   args, out_stem, xlabel="distance from dipole centre (um)", n_culture=None,
                   n_ensemble=None):
    """n_culture / n_ensemble: given when only some of the culture's neurons are drawn
    (--n-somata): the culture's size and the number of simulated neurons the lines use."""
    half = args.half
    style = getattr(args, "style", "specimen")
    subset = bool(n_culture) and n_culture > len(tab)
    cul_lab = ("the %d neurons shown" % len(tab)) if subset else "this culture"
    ens_lab = ("all %d simulated neurons" % n_ensemble) if (subset and n_ensemble) else "all cultures"
    xg = np.arange(-half, half + args.grid / 2, args.grid)
    Xg, Yg = np.meshgrid(xg, xg)
    g_grid = field_mV_per_A(Xg, Yg, elec_xy, elec_sign, ph["sigma_Sm"], ph["rmin_um"], ph["h_soma_um"])
    i0_A = ph["i0_uA"] * 1e-6
    vmax = float(np.percentile(np.abs(g_grid * i0_A), 99.5))
    centre = dipole_centre(elec_xy, elec_sign)
    rc = 0.5 * (edges_r[:-1] + edges_r[1:])

    fig = plt.figure(figsize=(16.5, 8.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.0], height_ratios=[0.45, 1.0, 1.15],
                          wspace=0.22, hspace=0.6)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_i = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[1, 1], sharex=ax_i)
    ax_r = fig.add_subplot(gs[2, 1])

    mfrac = float(getattr(args, "morph_fraction", 1.0))
    mseed = int(getattr(args, "somata_seed", 0)) + 1
    im, patches, scat, drawn = _draw_scene(ax_m, half, args.grid, g_grid, vmax, elec_xy,
                                           ph["electrode_um"], tab, polys, args.n_morph, centre,
                                           ph["h_soma_um"], args.dvm_max, style, mfrac, mseed)
    cb = fig.colorbar(im, ax=ax_m, fraction=0.035, pad=0.02)
    cb.set_label("extracellular potential Ve (mV, symlog)")
    if args.soma_color == "dvm":
        cb2 = fig.colorbar(scat["grad"], ax=ax_m, location="bottom", fraction=0.035, pad=0.08)
        v = args.dvm_max
        cb2.set_ticks([-v, -v / 4, -1, 0, 1, v / 4, v])
        cb2.set_ticklabels([f"{x:g}" for x in (-v, -v / 4, -1, 0, 1, v / 4, v)])
        cb2.set_label("soma DeltaVm (mV, symlog)   red = depolarized, blue = hyperpolarized")
    _scene_legend(ax_m, sorted(tab["morph"].astype(str).unique()), args.soma_color, style)

    tt = np.unique(np.r_[np.linspace(times.min(), min(times.max(), 0.7), 800), times])
    ax_i.plot(tt, P.pulse_current(tt), "k", lw=1.3)
    ax_i.axvspan(-0.5, -0.25, color="#fde0dd", zorder=0); ax_i.axvspan(-0.25, 0, color="#deebf7", zorder=0)
    ax_i.set_ylim(-1.35, 1.35); ax_i.set_yticks([-1, 0, 1]); ax_i.set_yticklabels(["-I0", "0", "+I0"], fontsize=8)
    ax_i.set_title("stimulus current", fontsize=9)
    plt.setp(ax_i.get_xticklabels(), visible=False)

    def pct(d, key):
        n = d["n"].sum(axis=1); return 100 * d[key].sum(axis=1) / np.maximum(n, 1)
    for key in ("act", "dep", "hyp"):
        ax_n.plot(times, pct(cul, key), color=P.COL[key], lw=2.0, label=f"{P.LABEL[key]}: {cul_lab}")
        ax_n.plot(times, pct(ens, key), color=P.COL[key], lw=1.3, ls="--")
    ax_n.plot([], [], color="0.3", ls="--", label=f"expected from {ens_lab}")
    ax_n.axvspan(-0.5, 0, color="0.93", zorder=0)
    ax_n.set_ylim(0, 100); ax_n.set_ylabel("% of neurons" if subset else "% of this culture's neurons")
    ax_n.set_xlabel("time from end of phase 2 (ms)" +
                    ("  [linear |t|<1 ms, log beyond]" if times.max() > 2 else ""))
    ax_n.legend(fontsize=7, loc="center right"); ax_n.grid(alpha=0.2)
    P._time_axis(ax_n, times)
    cursors = [ax_i.axvline(times[0], color="C1", lw=2), ax_n.axvline(times[0], color="C1", lw=2)]

    sup = fig.suptitle("", fontsize=12)
    note = fig.text(0.5, 0.012, "", ha="center", fontsize=9)
    n_neu = len(tab)
    seq = P._frame_sequence(len(times), args.fps, args.hold_end)

    def draw_r(k):
        ax_r.cla()
        for key in ("act", "dep", "hyp"):
            n_e, k_e = ens["n"][k], ens[key][k]
            with np.errstate(invalid="ignore", divide="ignore"):
                ax_r.plot(rc, np.where(n_e > 0, k_e / n_e, np.nan), color=P.COL[key], lw=1.8)
            n_c, k_c = cul["n"][k], cul[key][k]
            m = n_c > 0
            if m.any():
                p = k_c[m] / n_c[m]; lo, hi = wilson(k_c[m], n_c[m])
                yerr = [np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)]   # float residues ~1e-17
                ax_r.errorbar(rc[m], p, yerr=yerr, fmt="o", ms=4, color=P.COL[key],
                              mec="k", mew=0.4, elinewidth=0.8, capsize=2)
        ax_r.set_ylim(-0.02, 1.02); ax_r.set_xlim(edges_r[0], edges_r[-1])
        ax_r.set_xlabel(xlabel); ax_r.set_ylabel("P(state)")
        if args.panel == "theta":
            ax_r.set_xticks([0, 45, 90, 135, 180])
        ax_r.set_title(f"lines: {ens_lab}   |   points: {cul_lab} (Wilson 95% CI)", fontsize=8.5)
        ax_r.grid(alpha=0.2)

    def update(k):
        t = float(times[k])
        im.set_data(g_grid * i0_A * float(P.pulse_current([t])[0]))
        for p_, c in zip(patches, P._elec_colors(elec_sign, float(P.pulse_current([t])[0]))):
            p_.set_facecolor(c)
        a, d_, h_ = P.states(dv[k], fb[k], args.neu)
        _update_somata(scat, tab, {"act": a, "dep": d_, "hyp": h_, "neu": ~(a | d_ | h_)},
                       dv[k], args.soma_color, style)
        for c in cursors:
            c.set_xdata([t, t])
        draw_r(k)
        sup.set_text(f"Culture {cid}  --  layer {args.layer:.0f} um   |   t = {t:+.3f} ms   |   "
                     f"{P._phase_label(t)}")
        note.set_text(f"{cul_lab}: activated {a.sum()} ({100*a.mean():.1f}%)   depolarized "
                      f"{d_.sum()} ({100*d_.mean():.1f}%)   hyperpolarized {h_.sum()} "
                      f"({100*h_.mean():.1f}%)   of {n_neu} neurons"
                      + (f" (drawn at random from {n_culture})" if subset else "")
                      + f"   |   morphologies drawn: {drawn}")
        return [im, sup, note]

    anim = animation.FuncAnimation(fig, update, frames=seq, blit=False)
    path = P._save(anim, out_stem, args.fps)
    plt.close(fig)

    # snapshots for slides: mid phase 1, mid phase 2, +2 ms
    wanted = [-0.375, -0.125, 2.0]
    idx = sorted(set(int(np.argmin(np.abs(times - w))) for w in wanted))
    fig, axes = plt.subplots(1, len(idx), figsize=(6.2 * len(idx), 6.4))
    for ax, k in zip(np.atleast_1d(axes), idx):
        t = float(times[k])
        im2, pt2, sc2, _ = _draw_scene(ax, half, args.grid, g_grid, vmax, elec_xy, ph["electrode_um"],
                                       tab, polys, args.n_morph, centre, ph["h_soma_um"], args.dvm_max,
                                       style, mfrac, mseed)
        im2.set_data(g_grid * i0_A * float(P.pulse_current([t])[0]))
        for p_, c in zip(pt2, P._elec_colors(elec_sign, float(P.pulse_current([t])[0]))):
            p_.set_facecolor(c)
        a, d_, h_ = P.states(dv[k], fb[k], args.neu)
        _update_somata(sc2, tab, {"act": a, "dep": d_, "hyp": h_, "neu": ~(a | d_ | h_)},
                       dv[k], args.soma_color, style)
        ax.set_title(f"t = {t:+.3f} ms ({P._phase_label(t)})\nact {100*a.mean():.1f}%  "
                     f"dep {100*d_.mean():.1f}%  hyp {100*h_.mean():.1f}%", fontsize=10)
    _scene_legend(np.atleast_1d(axes)[0], sorted(tab["morph"].astype(str).unique()), args.soma_color,
                  style)
    fig.suptitle(f"Culture {cid} -- layer {args.layer:.0f} um", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_stem.replace("culture_video", "culture_snapshots") + ".png", dpi=140)
    plt.close(fig)
    return path, drawn


# ----------------------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="video_run/culture_frames*.csv")
    ap.add_argument("--cultures", type=int, nargs="*", default=[])
    ap.add_argument("--layer", type=float, default=None, help="default: middle layer present")
    ap.add_argument("--half", type=float, default=400.0)
    ap.add_argument("--grid", type=float, default=4.0, help="field image resolution (um)")
    ap.add_argument("--n-morph", type=int, default=-1,
                    help="-1 (default): arbors of a random --morph-fraction of the drawn somata; "
                         "k >= 0: only the arbors of the k somata closest to the array")
    ap.add_argument("--morph-fraction", type=float, default=1.0,
                    help="fraction of the drawn somata whose arbor is drawn, chosen at random "
                         "(seed --somata-seed + 1); 1 = all, 0 = none")
    ap.add_argument("--panel", choices=("theta", "r"), default="theta",
                    help="bottom panel: P(state | angle to dipole axis) or P(state | distance)")
    ap.add_argument("--thbin", type=float, default=15.0, help="angle bin (deg)")
    ap.add_argument("--rbin", type=float, default=25.0, help="distance bin for P(r) (um)")
    ap.add_argument("--neu", type=float, default=1.0, help="|dVm| < this = neutral (statistics: 1 mV)")
    ap.add_argument("--soma-color", choices=("dvm", "state"), default="dvm",
                    help="soma fill: continuous dVm gradient (default) or discrete state")
    ap.add_argument("--dvm-max", type=float, default=20.0, help="dVm colour-scale limit (mV)")
    ap.add_argument("--n-somata", type=int, default=0,
                    help="neurons drawn per culture video, uniform at random (0 = all)")
    ap.add_argument("--somata-seed", type=int, default=0, help="seed of that random choice")
    ap.add_argument("--style", choices=("plain", "specimen"), default="plain",
                    help="plain: grey branches, small edge-less circles; specimen: coloured by specimen")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hold-end", type=float, default=2.0)
    ap.add_argument("--outdir", default="video_run")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-frozen", action="store_true")
    args = ap.parse_args(argv)

    df, files = P.load_frames(args.frames)
    df["morph"] = df["morph"].astype(str)
    layers = np.sort(df["layer"].astype(float).unique())
    if args.layer is None:
        args.layer = float(layers[len(layers) // 2])
    df = df[np.isclose(df["layer"].astype(float), args.layer)]
    elec_xy, elec_sign = P.load_electrodes(files)
    ph = physics_params()
    print(f"[culture videos] {len(files)} file(s) | {df['culture_id'].nunique()} cultures | "
          f"layer {args.layer:.0f} um | field params from {ph['source']}")
    if args.list:
        print(df.groupby("culture_id").agg(file=("_file", "first"), culture=("culture", "first"),
                                           neurons=("x", lambda s: s.size)).assign(
            neurons=lambda d: d["neurons"] // df["t_ms"].nunique()).to_string())
        return []
    if not args.cultures:
        raise SystemExit("[culture videos] give --cultures (see --list)")
    fz = P.frozen_pulse_fraction(df)
    if fz is not None and fz > 0.9 and not args.allow_frozen:
        raise SystemExit("[culture videos] STOP: pulse frames are frozen (export with pre_end_ms=0). "
                         "Use culture_export_video.py.")
    if "theta_deg" not in df.columns:
        print("  NOTE: no theta_deg column (old export) -> somata only, no morphologies.")

    times = np.sort(df["t_ms"].unique())
    centre = dipole_centre(elec_xy, elec_sign)
    if args.panel == "theta":
        edges_r = np.arange(0, 180 + args.thbin / 2, args.thbin)
        coord = lambda d: dipole_theta_deg(d["x"], d["y"], elec_xy, elec_sign)
        xlabel = "angle to dipole axis theta (deg)   0 = anode side,  180 = cathode side"
    else:
        rmax = np.hypot(args.half, args.half)
        edges_r = np.arange(0, rmax + args.rbin, args.rbin)
        coord = lambda d: np.sqrt((d["x"].to_numpy(float) - centre[0]) ** 2 +
                                  (d["y"].to_numpy(float) - centre[1]) ** 2 + ph["h_soma_um"] ** 2)
        xlabel = "distance from dipole centre (um)"
    ens = state_counts_vs(coord(df), df, times, edges_r, args.neu)
    polys = load_morph_polylines(sorted(df["morph"].unique()), args.layer) \
        if "theta_deg" in df.columns else {}
    os.makedirs(args.outdir, exist_ok=True)
    paths = []
    for cid in args.cultures:
        sub = df[df["culture_id"] == cid]
        if sub.empty:
            print(f"  culture_id {cid}: not found, skipped"); continue
        t_c, tab, dv, fb = neuron_matrices(sub)
        if not np.allclose(t_c, times):
            print(f"  culture_id {cid}: frame times differ from the ensemble, skipped"); continue
        n_cul, n_ens = len(tab), None
        if 0 < args.n_somata < n_cul:
            keep = np.sort(np.random.default_rng(args.somata_seed).choice(n_cul, size=args.n_somata,
                                                                          replace=False))
            tab = tab.iloc[keep].reset_index(drop=True)
            dv, fb = dv[:, keep], fb[:, keep]
            cul = state_counts_matrix(coord(tab), dv, fb, edges_r, args.neu)
            n_ens = int(len(df) // len(times))
        else:
            cul = state_counts_vs(coord(sub), sub, times, edges_r, args.neu)
        stem = os.path.join(args.outdir, f"culture_video_{cid}_L{args.layer:.0f}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path, drawn = render_culture(cid, times, tab, dv, fb, polys, ens, cul, edges_r,
                                         elec_xy, elec_sign, ph, args, stem, xlabel,
                                         n_culture=n_cul, n_ensemble=n_ens)
        print(f"  culture {cid}: {len(tab)} of {n_cul} neurons drawn, {drawn} morphologies, "
              f"{len(times)} frames -> {path}")
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
