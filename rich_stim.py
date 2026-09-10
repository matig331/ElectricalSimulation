"""
rich_stim.py -- before / during / after on the REAL Rich model (ACTIVE), batch.

For ONE morphology, sweeps 3 layers x 3 soma positions x 3 random rotations and
writes one page per condition to rich_stim_report.pdf. Each page:
  * row 1: Vm on the morphology at BEFORE (baseline) / POS peak / NEG peak / AFTER(+150ms)
  * row 2: Vm(t) and Ve(t) at a few marked points (points vary per condition)

The cell is settled at its true rest (V_REST, from the step-B baseline), a
BIPOLAR biphasic pulse (2 anodes +, 2 cathodes -) is applied after `baseline_ms`,
and the run continues `post_ms` after the pulse. Ve is the imposed extracellular
potential Ve = g*I(t); the standalone spatial field is in ve_field.py.

Point 4 (tolerance): the baseline is flagged "settled" only if the soma Vm slope
over the baseline window is below `settle_tol_mV_per_ms`.

REQUIREMENTS in the run folder:
  - compiled mechanisms: `nrnivmodl rich_mech`  (writes ./arm64 or ./x86_64 here)
  - eyal_active/  (only the .hoc template is needed)
  - the morphology bundle (so morphologies.find_one_morphology works)
  - field.py, morphologies.py, slicer.py, rich_cell.py, active_cell.py, gradient_map.py not required

Run:  python rich_stim.py          # -> rich_stim_report.pdf   (27 pages)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from neuron import h

from morphologies import find_one_morphology
from slicer import reduced_asc
from rich_cell import build_rich_cell
from active_cell import segment_coords, _soma_center
import field as F
from config import CFG

V_REST = -73.9            # settled rest (step B); correct v_init for the Rich model
AREA_HALF = CFG.area_half_um         # correct area = 200 x 200 um
H_SOMA = 10.0
COLORS = dict(soma="black", near="magenta", dend1="darkorange", dend2="royalblue")


# ----------------------------------------------------------------------
# placement (shift soma to pos_xy at H_SOMA, then rotate about pos in-plane)
# ----------------------------------------------------------------------
def _transform(pts3, soma_center, pos_xy, theta_deg, h_soma=H_SOMA):
    p = np.asarray(pts3, float) + (np.array([pos_xy[0], pos_xy[1], h_soma]) - soma_center)
    if theta_deg:
        th = np.radians(theta_deg); c, s = np.cos(th), np.sin(th)
        ox, oy = pos_xy
        x = ox + c * (p[:, 0] - ox) - s * (p[:, 1] - oy)
        y = oy + s * (p[:, 0] - ox) + c * (p[:, 1] - oy)
        p = np.column_stack([x, y, p[:, 2]])
    return p


def _place(cell, pos_xy, theta_deg):
    coords0, refs = segment_coords(cell)
    sc = _soma_center(cell, coords0, refs)
    coords = _transform(coords0, sc, pos_xy, theta_deg)
    return coords, refs, sc


def _placed_polylines(cell, seg_val, refs, sc, pos_xy, theta_deg):
    """Line segments (xy) coloured by seg_val, with the SAME placement+rotation
    as the field, so the drawing matches the coordinates used for Vm/Ve."""
    val_of = {(seg.sec.name(), round(seg.x, 5)): v for seg, v in zip(refs, seg_val)}
    lines, vals = [], []
    for sec in cell.all:
        if "axon" in sec.name():
            continue
        n = sec.n3d()
        if n < 2:
            continue
        arc = np.array([sec.arc3d(i) for i in range(n)])
        L = arc[-1] if arc[-1] > 0 else sec.L
        P = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(n)])
        Pt = _transform(P, sc, pos_xy, theta_deg)
        sa = np.array([seg.x * L for seg in sec])
        sv = np.array([val_of[(sec.name(), round(seg.x, 5))] for seg in sec])
        pv = np.interp(arc, sa, sv)
        for i in range(n - 1):
            lines.append([Pt[i, :2], Pt[i + 1, :2]])
            vals.append(0.5 * (pv[i] + pv[i + 1]))
    return np.array(lines), np.array(vals)


# ----------------------------------------------------------------------
# stimulation
# ----------------------------------------------------------------------
def stimulate_rich(cell, pos_xy=(0.0, 0.0), theta_deg=0.0, i0_uA=50.0,
                   phase_dur_ms=0.25, baseline_ms=100.0, post_ms=150.0,
                   dt_ms=0.025, sigma_Sm=1.5, rmin_um=12.5,
                   settle_tol_mV_per_ms=1e-2, ramp_us=100.0, interphase_us=0.0,
                   n_pulses=1, isi_ms=5000.0):
    coords, refs, sc = _place(cell, pos_xy, theta_deg)
    elec, sign = F.default_array(monopolar=False)                # bipolar
    g = F.geom_factor(coords, elec, sign, sigma_Sm=sigma_Sm, rmin_um=rmin_um)  # mV/A

    t_on = baseline_ms
    train_span = (n_pulses - 1) * isi_ms + 2 * phase_dur_ms
    t = np.arange(0.0, baseline_ms + train_span + post_ms + dt_ms, dt_ms)
    I = F.biphasic_train(t - t_on, i0_uA, phase_dur_ms, True, ramp_us, interphase_us,
                         n_pulses, isi_ms)  # ramped train, A
    ve_seg_t = np.outer(g, I)                                    # (n_seg, n_t) mV

    tvec = h.Vector(t); keep = [tvec]
    for sec in set(s.sec for s in refs):
        if not h.ismembrane("extracellular", sec=sec):
            sec.insert("extracellular")
    for k, seg in enumerate(refs):
        vv = h.Vector(ve_seg_t[k]); vv.play(seg._ref_e_extracellular, tvec, True)
        keep.append(vv)

    v_all = [h.Vector().record(seg._ref_v) for seg in refs]
    tr = h.Vector().record(h._ref_t)
    h.celsius = 37; h.dt = dt_ms; h.tstop = t[-1]
    h.finitialize(V_REST)
    h.continuerun(t[-1])
    t = np.asarray(tr); v_all = np.asarray([np.asarray(x) for x in v_all])

    idx = dict(before=int(np.argmin(np.abs(t - (t_on - dt_ms)))),
               pos=int(np.argmin(np.abs(t - (t_on + phase_dur_ms - dt_ms)))),
               neg=int(np.argmin(np.abs(t - (t_on + 2 * phase_dur_ms - dt_ms)))),
               after=int(np.argmin(np.abs(t - (t_on + 2 * phase_dur_ms + post_ms)))))
    si = _points(coords, refs, elec, np.random.default_rng(0))["soma"]
    base = v_all[si, :max(idx["before"], 6)]
    slope = float(np.abs(np.polyfit(t[:base.size], base, 1)[0]))
    return dict(t=t, v_all=v_all, ve_seg_t=ve_seg_t, coords=coords, refs=refs,
                sc=sc, pos_xy=pos_xy, theta=theta_deg, elec=elec, sign=sign, idx=idx,
                baseline_slope=slope, settled=slope < settle_tol_mV_per_ms, t_on=t_on)


def _points(coords, refs, elec, rng):
    soma_i = [i for i, s in enumerate(refs) if "soma" in s.sec.name()]
    si = soma_i[len(soma_i) // 2] if soma_i else 0
    d_el = np.min(np.linalg.norm(coords[:, None, :2] - elec[None, :, :], axis=2), axis=1)
    d_el[si] = np.inf
    near_i = int(np.argmin(d_el))
    dend = np.array([("axon" not in s.sec.name() and i != si) for i, s in enumerate(refs)])
    cand = np.where(dend)[0]
    pick = rng.choice(cand, size=min(2, len(cand)), replace=False) if cand.size else []
    pts = {"soma": si, "near": near_i}
    for j, p in enumerate(pick):
        pts[f"dend{j + 1}"] = int(p)
    return pts


# ----------------------------------------------------------------------
# figure (one condition)
# ----------------------------------------------------------------------
def _map(ax, lines, vals, vlim, title, elec, sign):
    lc = LineCollection(lines, cmap="RdBu_r", array=vals, lw=0.9); lc.set_clim(-vlim, vlim)
    ax.add_collection(lc)
    for (ex, ey) in elec[sign > 0]:
        ax.add_patch(Rectangle((ex - 12.5, ey - 12.5), 25, 25, facecolor="red", edgecolor="k", zorder=5))
    for (ex, ey) in elec[sign < 0]:
        ax.add_patch(Rectangle((ex - 12.5, ey - 12.5), 25, 25, facecolor="blue", edgecolor="k", zorder=5))
    ax.add_patch(Rectangle((-AREA_HALF, -AREA_HALF), 2 * AREA_HALF, 2 * AREA_HALF,
                           fill=False, ec="0.4", ls="--", lw=0.8))
    ax.set_aspect("equal"); ax.set_xlim(-AREA_HALF - 40, AREA_HALF + 40)
    ax.set_ylim(-AREA_HALF - 40, AREA_HALF + 40)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    return lc


def figure(res, cell, name, layer, rng):
    t, v_all, ve = res["t"], res["v_all"], res["ve_seg_t"]
    coords, refs, sc = res["coords"], res["refs"], res["sc"]
    pos_xy, theta, elec, sign, idx = res["pos_xy"], res["theta"], res["elec"], res["sign"], res["idx"]
    pts = _points(coords, refs, elec, rng)
    order = ["before", "pos", "neg", "after"]
    titles = {"before": "BEFORE (baseline)", "pos": "POS peak", "neg": "NEG peak", "after": "AFTER +150ms"}

    fig = plt.figure(figsize=(16, 8.5))
    fig.suptitle(f"{name} | layer {int(layer)}um | pos {pos_xy} | theta {theta:.0f}deg | "
                 f"Rich ACTIVE bipolar +/-50uA | settled={res['settled']} "
                 f"(slope {res['baseline_slope']:.1e})", fontweight="bold")
    dvm = v_all - v_all[:, :1]
    vlim = float(np.percentile(np.abs(dvm), 99)) or 1.0
    for j, key in enumerate(order):
        k = idx[key]
        lv, vv = _placed_polylines(cell, dvm[:, k], refs, sc, pos_xy, theta)
        ax = fig.add_subplot(2, 4, j + 1)
        lc = _map(ax, lv, vv, vlim, f"Vm - {titles[key]}", elec, sign)
        for kn, i in pts.items():
            ax.scatter(*coords[i, :2], s=45, marker="o", facecolors="none",
                       edgecolors=COLORS[kn], linewidths=1.6, zorder=6)
        if j == 3:
            fig.colorbar(lc, ax=ax, fraction=0.046, label="dVm (mV)")

    axv = fig.add_subplot(2, 2, 3)
    for kn, i in pts.items():
        axv.plot(t, v_all[i], color=COLORS[kn], lw=1.0, label=kn)
    for key in order:
        axv.axvline(t[idx[key]], color="0.7", ls=":", lw=0.8)
    axv.set_xlabel("t (ms)"); axv.set_ylabel("Vm (mV)"); axv.legend(fontsize=7)
    axv.set_title("Vm(t) -- window centred on the pulse", fontsize=9)
    axv.set_xlim(res["t_on"] - 5.0, res["t_on"] + 20.0)   # narrow: baseline approach + response
    # zoom inset on the pulse window (see the biphasic hyperpol/depol wiggle)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    t_on = res["t_on"]; w0, w1 = t_on - 0.2, t_on + 2*0.25 + 0.5
    axz = inset_axes(axv, width="45%", height="45%", loc="upper right")
    for kn, i in pts.items():
        axz.plot(t, v_all[i], color=COLORS[kn], lw=1.0)
    axz.axvspan(t_on, t_on + 2*0.25, color="0.85", zorder=0)
    axz.set_xlim(w0, w1); axz.set_title("zoom on pulse", fontsize=7)
    axz.tick_params(labelsize=6)

    axe = fig.add_subplot(2, 2, 4)
    for kn, i in pts.items():
        axe.plot(t, ve[i], color=COLORS[kn], lw=1.0, label=kn)
    for key in order:
        axe.axvline(t[idx[key]], color="0.7", ls=":", lw=0.8)
    axe.set_xlabel("t (ms)"); axe.set_ylabel("Ve (mV)")
    axe.set_title("Ve(t) -- window centred on the pulse", fontsize=9)
    axe.set_xlim(res["t_on"] - 1.0, res["t_on"] + 2 * 0.25 + 2.0)   # narrow: the 0.5 ms biphasic, not 250 ms
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _ia
    t_on = res["t_on"]; aze = _ia(axe, width="45%", height="45%", loc="upper right")
    for kn, i in pts.items():
        aze.plot(t, ve[i], color=COLORS[kn], lw=1.0)
    aze.axvspan(t_on, t_on + 2*0.25, color="0.85", zorder=0)
    aze.set_xlim(t_on - 0.2, t_on + 0.7); aze.set_title("zoom (trapezoid, not Dirac)", fontsize=6)
    aze.tick_params(labelsize=6)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ----------------------------------------------------------------------
# batch: 3 layers x 3 positions x 3 rotations -> one PDF
# ----------------------------------------------------------------------
def batch(pref="60308", layers=(40.0, 80.0, 120.0),
          positions=None, n_rot=None, seed=0, baseline_ms=100.0, post_ms=150.0,
          out_pdf="rich_stim_report.pdf"):
    from config import CFG
    here = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(seed)
    XY = CFG.random_positions(rng=rng)          # ONE knob: n_samples random positions
    out = os.path.join(here, out_pdf)
    with PdfPages(out) as pdf:
        for layer in layers:
            asc = reduced_asc(find_one_morphology(pref), layer,
                              out_path=os.path.join(here, f"_rs_{int(layer)}.asc"))
            cell = build_rich_cell(asc)
            for (px, py) in XY:
                pos = (float(px), float(py))
                theta = float(rng.uniform(0, 360))
                res = stimulate_rich(cell, pos_xy=pos, theta_deg=theta,
                                     baseline_ms=baseline_ms, post_ms=post_ms)
                fig = figure(res, cell, f"specimen_{pref}", layer, rng)
                pdf.savefig(fig); plt.close(fig)
                print(f"  layer {int(layer)} pos ({px:.0f},{py:.0f}) theta {theta:5.0f} "
                      f"settled={res['settled']}")
            del cell
            os.remove(asc)
    print("done:", out)
    return out


if __name__ == "__main__":
    batch()
