"""
channel_dynamics.py -- PER-CHANNEL opening/closing over time + WHERE each channel
sits on the neuron, BEFORE / DURING / AFTER stimulation, for ALL morphologies and
all layers.

One PAGE per (morphology, layer). Rows = channels (one per channel, NOT overlaid).
Each row:
  left  : morphology coloured by that channel's gbar  -> WHERE the channel is
  right : that channel's conductance g(t) at soma / near-electrode / distal dendrite
          -> WHEN it opens/closes, stim window shaded (before / during / after)

Run:  python channel_dynamics.py       # uses CFG (loops all morphologies x layers)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from neuron import h

from config import CFG
from morphologies import find_one_morphology
from slicer import reduced_asc
from rich_cell import build_rich_cell
from active_cell import segment_coords, _soma_center
import field as F

CH = ["NaTa_t", "Nap_Et2", "K_Pst", "K_Tst", "SKv3_1", "SK_E2", "Im", "Ca_LVAst", "Ca_HVA", "Ih"]
PT_COLORS = dict(soma="black", near="magenta", dend="green")


def _placed(cell, pos_xy, theta_deg, h_soma=10.0):
    coords, refs = segment_coords(cell)
    sc = _soma_center(cell, coords, refs)
    p = coords + (np.array([pos_xy[0], pos_xy[1], h_soma]) - sc)
    if theta_deg:
        th = np.radians(theta_deg); c, s = np.cos(th), np.sin(th); ox, oy = pos_xy
        x = ox + c*(p[:, 0]-ox) - s*(p[:, 1]-oy); y = oy + s*(p[:, 0]-ox) + c*(p[:, 1]-oy)
        p = np.column_stack([x, y, p[:, 2]])
    return p, refs


def _points(coords, refs, elec):
    soma = [i for i, s in enumerate(refs) if "soma" in s.sec.name()]
    si = soma[len(soma)//2] if soma else 0
    d = np.min(np.linalg.norm(coords[:, None, :2]-elec[None, :, :], axis=2), axis=1); d[si] = np.inf
    near = int(np.argmin(d))
    dend = np.array(["axon" not in s.sec.name() for s in refs])
    dd = np.where(dend, np.linalg.norm(coords[:, :2]-coords[si, :2], axis=1), -np.inf)
    return {"soma": si, "near": near, "dend": int(np.argmax(dd))}


def run(cfg, pref, layer, pos_xy, theta_deg):
    asc = reduced_asc(find_one_morphology(pref), layer, out_path=f"_cd_{int(layer)}.asc")
    cell = build_rich_cell(asc)
    coords, refs = _placed(cell, pos_xy, theta_deg)
    elec, sign = F.default_array(monopolar=not cfg.bipolar)
    g = F.geom_factor(coords, elec, sign, sigma_Sm=cfg.sigma_Sm, rmin_um=cfg.rmin_um)
    t_on = cfg.baseline_ms
    t = np.arange(0.0, cfg.baseline_ms + 2*cfg.phase_dur_ms + cfg.post_ms + cfg.dt_ms, cfg.dt_ms)
    I = F.biphasic_current(t-t_on, cfg.i0_uA, cfg.phase_dur_ms, cfg.anodic_first, cfg.ramp_us, cfg.interphase_us)
    tvec = h.Vector(t); keep = [tvec]
    for sec in set(s.sec for s in refs):
        if not h.ismembrane("extracellular", sec=sec):
            sec.insert("extracellular")
    for k, seg in enumerate(refs):
        vv = h.Vector(g[k]*I); vv.play(seg._ref_e_extracellular, tvec, True); keep.append(vv)
    pts = _points(coords, refs, elec)
    gbar = {c: np.array([float(getattr(s, f"g{c}bar_{c}")) if h.ismembrane(c, sec=s.sec) else 0.0
                         for s in refs]) for c in CH}
    gt = {c: {} for c in CH}
    for c in CH:
        for pk, pi in pts.items():
            if h.ismembrane(c, sec=refs[pi].sec):
                gt[c][pk] = h.Vector().record(getattr(refs[pi], f"_ref_g{c}_{c}"))
    tr = h.Vector().record(h._ref_t)
    h.celsius = 37; h.dt = cfg.dt_ms; h.finitialize(cfg.v_rest_mV); h.continuerun(t[-1])
    tr = np.asarray(tr)
    for c in CH:
        for pk in gt[c]:
            gt[c][pk] = np.asarray(gt[c][pk])
    del cell; os.remove(asc)
    return tr, coords, gbar, gt, pts, t_on


def page(pdf, cfg, pref, layer, pos_xy, theta_deg):
    tr, coords, gbar, gt, pts, t_on = run(cfg, pref, layer, pos_xy, theta_deg)
    fig, axes = plt.subplots(len(CH), 2, figsize=(11, 2.1*len(CH)),
                             gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle(f"specimen_{pref} - channels WHERE (left) & WHEN (right), layer {int(layer)}um, "
                 f"pos {pos_xy} theta {theta_deg:.0f}", fontweight="bold")
    for row, c in enumerate(CH):
        axm, axt = axes[row]
        present = gbar[c] > 0
        axm.scatter(coords[~present, 0], coords[~present, 1], s=2, color="0.85")
        if present.any():
            axm.scatter(coords[present, 0], coords[present, 1], s=6, c=gbar[c][present], cmap="viridis")
        for pk, pi in pts.items():
            axm.scatter(*coords[pi, :2], s=40, marker="o", facecolors="none",
                        edgecolors=PT_COLORS[pk], linewidths=1.5, zorder=6)
        axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([]); axm.set_ylabel(c, fontsize=9)
        for pk in ("soma", "near", "dend"):
            if pk in gt[c]:
                axt.plot(tr, gt[c][pk], color=PT_COLORS[pk], lw=1.0, label=pk)
        axt.axvspan(t_on, t_on+2*cfg.phase_dur_ms, color="0.8", zorder=0)
        axt.set_xlim(t_on-5, t_on+2*cfg.phase_dur_ms+cfg.post_ms); axt.set_ylabel("g", fontsize=7)
        if row == 0:
            axt.legend(fontsize=7, loc="upper right")
    axes[-1, 1].set_xlabel("t (ms)")
    fig.tight_layout(rect=[0, 0, 1, 0.985]); pdf.savefig(fig); plt.close(fig)


def main(prefs=None, layers=None, pos_xy=(0.0, 0.0), theta_deg=0.0):
    cfg = CFG
    prefs = prefs or cfg.morphologies
    layers = layers or cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "channel_dynamics.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                page(pdf, cfg, pref, layer, pos_xy, theta_deg)
                print(f"  {pref} layer {int(layer)}: done")
    print("done:", out)
    return out


if __name__ == "__main__":
    main()
