"""
subthreshold.py -- SUBTHRESHOLD extracellular pulse: how the polarization spreads.

Applies a low-amplitude bipolar pulse (no spike) and shows:
  (1) map of dVm at the positive peak  -> the polarization pattern (dipole)
  (2) SIGNED dVm profile vs path distance from soma, dendrites vs STYLIZED axon
      -> how the polarization is distributed along the neuron (not a single-lambda fit,
         because the extracellular field drives all compartments at once)
  (3) dVm(t) at soma/near/dend           -> the biphasic subthreshold wiggle

Reuses stimulate_rich with a low i0. One page per layer.

Run:  python subthreshold.py            # uses CFG (i0 taken low, see i0_sub)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

from config import CFG
from morphologies import find_one_morphology
from slicer import reduced_asc
from rich_cell import build_rich_cell
from rich_stim import stimulate_rich, _placed_polylines, _points
import field as F


def main(prefs=None, layers=None, i0_sub=8.0, pos_xy=(40.0, 0.0), theta_deg=0.0):
    cfg = CFG
    prefs = prefs or cfg.morphologies
    layers = layers or cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "subthreshold.pdf")
    with PdfPages(out) as pdf:
      for pref in prefs:
        for layer in layers:
            asc = reduced_asc(find_one_morphology(pref), layer, out_path="_sub.asc")
            cell = build_rich_cell(asc)
            res = stimulate_rich(cell, pos_xy=pos_xy, theta_deg=theta_deg, i0_uA=i0_sub,
                                 phase_dur_ms=cfg.phase_dur_ms, baseline_ms=cfg.baseline_ms,
                                 post_ms=cfg.post_ms, dt_ms=cfg.dt_ms, sigma_Sm=cfg.sigma_Sm,
                                 rmin_um=cfg.rmin_um, ramp_us=cfg.ramp_us, interphase_us=cfg.interphase_us)
            t, v_all, coords, refs = res["t"], res["v_all"], res["coords"], res["refs"]
            elec, sign = res["elec"], res["sign"]
            dvm = v_all - v_all[:, :1]
            kpos = res["idx"]["pos"]
            dpk = dvm[:, kpos]
            nsp = int(np.sum((v_all[[j for j,s in enumerate(refs) if "soma" in s.sec.name()][0]][:-1] < 0) &
                             (v_all[[j for j,s in enumerate(refs) if "soma" in s.sec.name()][0]][1:] >= 0)))
            os.remove(asc)

            fig = plt.figure(figsize=(16, 4.6))
            fig.suptitle(f"specimen_{pref} layer {int(layer)}um - SUBTHRESHOLD (i0={i0_sub}uA) - "
                         f"spikes={nsp} (should be 0)", fontweight="bold")
            # (1) dVm map at pos peak
            ax1 = fig.add_subplot(1, 3, 1)
            lv, vv = _placed_polylines(cell, dpk, refs, res["sc"], pos_xy, theta_deg)
            vlim = float(np.percentile(np.abs(vv), 99)) or 1.0
            lc = LineCollection(lv, cmap="RdBu_r", array=vv, lw=0.9); lc.set_clim(-vlim, vlim); ax1.add_collection(lc)
            for (ex, ey) in elec[sign > 0]:
                ax1.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="red", edgecolor="k", zorder=5))
            for (ex, ey) in elec[sign < 0]:
                ax1.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="blue", edgecolor="k", zorder=5))
            ax1.set_aspect("equal"); ax1.autoscale(); ax1.set_title("dVm at POS peak (polarization)", fontsize=9)
            ax1.set_xticks([]); ax1.set_yticks([]); fig.colorbar(lc, ax=ax1, fraction=0.046, label="dVm (mV)")
            # (2) SIGNED dVm profile along the neuron: dendrites vs STYLIZED axon
            ax2 = fig.add_subplot(1, 3, 2)
            from neuron import h as _h
            _h.distance(0, cell.soma[0](0.5))
            pathd = np.array([_h.distance(seg) for seg in refs])
            def _dom(nm):
                if "axon" in nm: return "axon (stylized)"
                if "apic" in nm: return "apical"
                if "soma" in nm: return "soma"
                return "basal"
            dom = np.array([_dom(seg.sec.name()) for seg in refs])
            cols = {"basal": "green", "apical": "darkorange", "soma": "black",
                    "axon (stylized)": "purple"}
            for d in ["basal", "apical", "axon (stylized)"]:
                mm = dom == d
                if mm.any():
                    ax2.scatter(pathd[mm], dpk[mm], s=6, alpha=0.5, color=cols[d], label=d)
            ax2.axhline(0, color="0.7", lw=0.8)
            ax2.set_xlabel("path distance from soma (um)")
            ax2.set_ylabel("dVm (mV, signed: + depol / - hyperpol)")
            ax2.set_title("polarization profile along the neuron", fontsize=9)
            ax2.legend(fontsize=7)
            # (3) dVm(t) at points
            ax3 = fig.add_subplot(1, 3, 3)
            pts = _points(coords, refs, elec, np.random.default_rng(0))
            for kn, i in pts.items():
                ax3.plot(t, dvm[i], lw=1.0, label=kn)
            t_on = res["t_on"]; ax3.axvspan(t_on, t_on+2*cfg.phase_dur_ms, color="0.85", zorder=0)
            ax3.set_xlim(t_on-1, t_on+2*cfg.phase_dur_ms+2); ax3.set_xlabel("t (ms)"); ax3.set_ylabel("dVm (mV)")
            ax3.set_title("subthreshold biphasic wiggle (zoom)", fontsize=9); ax3.legend(fontsize=7)
            fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)
            print(f"  layer {int(layer)}: spikes={nsp}, max|dVm|={np.abs(dpk).max():.2f} mV")
    print("done:", out)
    return out


if __name__ == "__main__":
    main()
