"""
channel_atlas.py -- confirm ALL channel placements at once, from the REAL model.

Builds the Rich cell and, for each of the 10 conductances, colours the
morphology by the actual per-segment gbar that was inserted (grey where the
channel is absent). This is the visual proof that every channel sits where
sec 4.1 says it should.

Run:  cd estim && python channel_atlas.py     # -> channel_atlas.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from neuron import h

from morphologies import find_one_morphology, all_morphologies
from slicer import reduced_asc
from rich_cell import build_rich_cell

CHANNELS = ["NaTa_t", "Nap_Et2", "K_Pst", "K_Tst", "SKv3_1", "SK_E2",
            "Im", "Ca_LVAst", "Ca_HVA", "Ih"]


def seg_coords_and_gbar(cell):
    xy, gbar = [], {c: [] for c in CHANNELS}
    for sec in cell.all:
        if "axon" in sec.name():           # skip the 6 mm stylized axon
            continue
        n = sec.n3d()
        if n >= 2:
            arc = np.array([sec.arc3d(i) for i in range(n)])
            L = arc[-1] if arc[-1] > 0 else sec.L
            X = np.array([sec.x3d(i) for i in range(n)])
            Y = np.array([sec.y3d(i) for i in range(n)])
        for seg in sec:
            if n >= 2:
                d = seg.x * L
                xy.append((np.interp(d, arc, X), np.interp(d, arc, Y)))
            else:
                xy.append((sec.x3d(0), sec.y3d(0)))
            for c in CHANNELS:
                if h.ismembrane(c, sec=sec):
                    gbar[c].append(float(getattr(seg, f"g{c}bar_{c}")))
                else:
                    gbar[c].append(0.0)
    xy = np.array(xy)
    for c in CHANNELS:
        gbar[c] = np.array(gbar[c])
    return xy, gbar


def main(pref="60308", layers=(40.0, 80.0, 120.0)):
    from matplotlib.backends.backend_pdf import PdfPages
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, f"channel_atlas_{pref}.pdf")
    with PdfPages(out) as pdf:
        for layer in layers:
            asc = reduced_asc(find_one_morphology(pref), layer, out_path="_atlas.asc")
            cell = build_rich_cell(asc)
            xy, gbar = seg_coords_and_gbar(cell)
            fig, axes = plt.subplots(3, 4, figsize=(16, 11))
            fig.suptitle(f"Rich 2021 channel atlas (specimen_{pref}, layer {int(layer)}um) "
                         f"- per-segment gbar (grey = absent)", fontweight="bold")
            axes = axes.ravel()
            for k, c in enumerate(CHANNELS):
                ax = axes[k]; g = gbar[c]; present = g > 0
                ax.scatter(xy[~present, 0], xy[~present, 1], s=3, color="0.85")
                if present.any():
                    vmin = g[present].min(); vmax = g[present].max()
                    norm = LogNorm(vmin=max(vmin, vmax * 1e-4), vmax=vmax) if vmax > vmin else None
                    sc = ax.scatter(xy[present, 0], xy[present, 1], s=6, c=g[present],
                                    cmap="viridis", norm=norm)
                    fig.colorbar(sc, ax=ax, fraction=0.046, label="S/cm2")
                ax.set_aspect("equal"); ax.set_title(c, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
            for k in range(len(CHANNELS), len(axes)):
                axes[k].axis("off")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig); os.remove(asc)
            print(f"  {pref} layer {int(layer)}: done")
    print("done:", out)
    return out


if __name__ == "__main__":
    main()
