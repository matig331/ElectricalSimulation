"""
rich_footprint.py -- DEFINE the well area from the activating somata (Rich model).

Places the soma on a grid of positions, scans a few random orientations at each,
applies the bipolar +/-50 uA pulse, and marks the positions whose SOMA spikes.
The area to use = bounding box of the spiking somata + a margin (edge effects).
Dendrites branching beyond that box are fine: the box fixes where the somata sit.

Light run: only the soma Vm is recorded and a short window is used, so a whole
grid is affordable. Init at V_REST (settled rest), so no long pre-settling.

Output: rich_footprint.pdf (activating somata + suggested area) and a printed
suggested AREA_HALF. Loop `layers` to check how stable the area is across
complexities; overlay morphologies by calling for each and comparing.

Run:  python rich_footprint.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from neuron import h

from morphologies import find_one_morphology
from slicer import reduced_asc
from rich_cell import build_rich_cell
from active_cell import segment_coords, _soma_center
import field as F

V_REST = -73.9
H_SOMA = 10.0


def _placed(cell, pos_xy, theta_deg):
    coords0, refs = segment_coords(cell)
    sc = _soma_center(cell, coords0, refs)
    p = coords0 + (np.array([pos_xy[0], pos_xy[1], H_SOMA]) - sc)
    if theta_deg:
        th = np.radians(theta_deg); c, s = np.cos(th), np.sin(th)
        ox, oy = pos_xy
        x = ox + c * (p[:, 0] - ox) - s * (p[:, 1] - oy)
        y = oy + s * (p[:, 0] - ox) + c * (p[:, 1] - oy)
        p = np.column_stack([x, y, p[:, 2]])
    return p, refs


def spikes_at(cell, pos_xy, theta_deg, i0_uA=50.0, phase_dur_ms=0.25,
              baseline_ms=5.0, post_ms=6.0, dt_ms=0.025,
              sigma_Sm=1.5, rmin_um=12.5, ramp_us=100.0, interphase_us=0.0, phase="both",
              detail=False, pre_end_ms=0.0, v_init_mV=V_REST):
    """Light: soma-only recording, short window. Returns (n_spikes, Vsoma_max).
    detail=True -> (n_spikes, Vsoma_max, tw, vw): the soma trace from `pre_end_ms` before the
    END of phase 2 onward, with tw = t - t_end (tw = 0 exactly at the end of phase 2).
    v_init_mV: initial voltage of every compartment (finitialize)."""
    coords, refs = _placed(cell, pos_xy, theta_deg)
    elec, sign = F.default_array(monopolar=False)
    g = F.geom_factor(coords, elec, sign, sigma_Sm=sigma_Sm, rmin_um=rmin_um)
    t_on = baseline_ms
    post = post_ms      # detail=True must NOT lengthen the run (HPC cost)
    t = np.arange(0.0, baseline_ms + 2 * phase_dur_ms + post + dt_ms, dt_ms)
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
    h.celsius = 37; h.dt = dt_ms; h.tstop = t[-1]
    h.finitialize(float(v_init_mV)); h.continuerun(t[-1])
    v = np.asarray(vs)
    nsp = int(np.sum((v[:-1] < 0) & (v[1:] >= 0)))
    if detail:
        t_end = t_on + 2 * phase_dur_ms            # END of the biphasic pulse (after phase 2)
        t_start = t_end - max(0.0, float(pre_end_ms))
        m = t >= t_start
        tw = (t[m] - t_end).astype(float)          # t=0 is exactly END of phase 2
        vw = v[m].astype(float)
        return nsp, float(v.max()), tw, vw
    return nsp, float(v.max())


def footprint(cell, half=150.0, thetas=None, i0_uA=50.0, n=None, rng=None, step=None):
    """N random soma positions (config.n_samples) in the +/-half box; each position is
    tested over `thetas` orientations (config.orient_deg) and best-of decides whether it
    can be activated. Driven by the ONE global knob config.n_samples.
    `step` is accepted but IGNORED (kept so old grid-style callers don't break)."""
    from config import CFG
    thetas = CFG.orient_deg if thetas is None else thetas
    XY = CFG.random_positions(n=n, rng=rng, half=half)     # ONE knob: n_samples random positions
    X, Y, SP, VM = [], [], [], []
    for (x, y) in XY:
        sp, vb = 0, -1e9
        for th in thetas:
            nsp, vmax = spikes_at(cell, (float(x), float(y)), float(th), i0_uA=i0_uA)
            sp = max(sp, nsp); vb = max(vb, vmax)
        X.append(float(x)); Y.append(float(y)); SP.append(sp); VM.append(vb)
    return map(np.array, (X, Y, SP, VM))


def suggest_area(X, Y, SP, margin=40.0):
    fire = SP > 0
    if not fire.any():
        return None
    half = max(np.abs(X[fire]).max(), np.abs(Y[fire]).max()) + margin
    return float(half)


def main(pref="60308", layer=80.0, half=150.0, step=30.0, i0_uA=50.0, margin=40.0):
    here = os.path.dirname(os.path.abspath(__file__))
    asc = reduced_asc(find_one_morphology(pref), layer, out_path="_fp.asc")
    cell = build_rich_cell(asc)
    X, Y, SP, VM = footprint(cell, half=half, step=step, i0_uA=i0_uA)
    ah = suggest_area(X, Y, SP, margin=margin)
    elec, sign = F.default_array(monopolar=False)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sc = ax.scatter(X, Y, c=VM, cmap="inferno", s=180, marker="s", vmin=V_REST, vmax=40)
    fire = SP > 0
    ax.scatter(X[fire], Y[fire], s=60, facecolors="none", edgecolors="cyan", linewidths=2.0,
               label="soma spikes")
    for (ex, ey) in elec[sign > 0]:
        ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="red", edgecolor="k", zorder=5))
    for (ex, ey) in elec[sign < 0]:
        ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="blue", edgecolor="k", zorder=5))
    if ah:
        ax.add_patch(Rectangle((-ah, -ah), 2*ah, 2*ah, fill=False, ec="lime", lw=2.0,
                               label=f"suggested area 2x{ah:.0f} um"))
    ax.set_aspect("equal"); ax.set_xlabel("soma x (um)"); ax.set_ylabel("soma y (um)")
    ax.set_title(f"specimen_{pref} layer {int(layer)}um (Rich) - activating somata\n"
                 f"bipolar +/-{i0_uA:.0f} uA, best of {4} orientations", fontsize=10)
    fig.colorbar(sc, ax=ax, label="peak Vsoma (mV)"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(here, "rich_footprint.pdf")
    fig.savefig(out); plt.close(fig)
    os.remove(asc)
    n_fire = int((SP > 0).sum())
    print(f"spiking somata: {n_fire}/{len(SP)} | suggested AREA_HALF = {ah} um "
          f"(area {2*ah if ah else '-'} x {2*ah if ah else '-'} um)")
    print("done:", out)
    return ah


if __name__ == "__main__":
    main()


def multi(prefs=("60308", "130303", "60303"), layers=(40.0, 80.0, 120.0),
          half=150.0, step=30.0, i0_uA=50.0, margin=40.0, seed=0):
    """Area study over MANY morphologies x layers: collect every spiking soma
    position, take the ENVELOPE (max bounding box) + margin -> one robust area
    that contains all activating somata (none on the border).

    NOTE: heavy -- it is len(prefs)*len(layers) footprint grids. Run few points
    (bigger `step`) first to scope it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    here = os.path.dirname(os.path.abspath(__file__))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(8, 7))
    allx, ally = [], []
    per_morph = {}
    for mi, pref in enumerate(prefs):
        per_morph[pref] = []
        for layer in layers:
            asc = reduced_asc(find_one_morphology(pref), layer, out_path=f"_fpm.asc")
            cell = build_rich_cell(asc)
            X, Y, SP, VM = footprint(cell, half=half, step=step, i0_uA=i0_uA)
            os.remove(asc)
            fire = SP > 0
            allx += list(X[fire]); ally += list(Y[fire])
            per_morph[pref] += [max(np.abs(X[fire]).max(), np.abs(Y[fire]).max()) + margin] if fire.any() else []
            ax.scatter(X[fire], Y[fire], s=40, color=colors[mi % len(colors)],
                       alpha=0.6, label=f"{pref}" if layer == layers[0] else None)
            print(f"  {pref} layer {int(layer)}: {int(fire.sum())} spiking somata")
    elec, sign = F.default_array(monopolar=False)
    if allx:
        ah = max(max(np.abs(allx)), max(np.abs(ally))) + margin
        ax.add_patch(Rectangle((-ah, -ah), 2*ah, 2*ah, fill=False, ec="k", lw=2.0,
                               label=f"combined area 2x{ah:.0f} um"))
    else:
        ah = None
    for (ex, ey) in elec[sign > 0]:
        ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="red", edgecolor="k", zorder=5))
    for (ex, ey) in elec[sign < 0]:
        ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="blue", edgecolor="k", zorder=5))
    ax.set_aspect("equal"); ax.set_xlabel("soma x (um)"); ax.set_ylabel("soma y (um)")
    ax.set_title(f"Activating somata across {len(prefs)} morphologies x {len(layers)} layers\n"
                 f"envelope + {margin:.0f}um margin -> AREA_HALF = {ah}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(here, "rich_footprint_multi.pdf")
    fig.savefig(out); plt.close(fig)
    mh = [np.mean(v) for v in per_morph.values() if v]
    mean_ah = float(np.mean(mh)) if mh else None
    print("per-morphology AREA_HALF:", {k: round(np.mean(v),1) for k,v in per_morph.items() if v})
    print(f"ENVELOPE AREA_HALF = {ah} um | MEAN AREA_HALF = {mean_ah} um  (put one in config.py) | done: {out}")
    return ah
