"""
axon_sensitivity.py -- robustness of the stimulation results to the STYLIZED axon
length. Runs several axon lengths in one go and compares, so you can show the
footprint / thresholds / spike-initiation site do NOT depend on the axon.

For each length it (re)builds the Rich cell with that axon and computes:
  (1) P(spike) vs distance from the electrodes (best of 4 orientations)
  (2) P(spike) vs amplitude (M random pos/theta)
  (3) spike-initiation site (AIS vs axon terminal / sealed end) at a near position
Overlays (1)-(2) and tabulates (3). One PDF: axon_sensitivity.pdf.

Run:  python pipeline.py axon_sensitivity      # or: python axon_sensitivity.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from morphologies import find_one_morphology
from slicer import reduced_asc
from rich_cell import build_rich_cell
from rich_footprint import spikes_at


def _init_site(cell, pos=(40.0, 0.0), thetas=(0, 90, 180, 270)):
    from rich_stim import stimulate_rich
    from well import _spike_init
    for th in thetas:
        res = stimulate_rich(cell, pos_xy=pos, theta_deg=float(th),
                             baseline_ms=20.0, post_ms=10.0)
        v_all, refs, coords = res["v_all"], res["refs"], res["coords"]
        si = [j for j, s in enumerate(refs) if "soma" in s.sec.name()][0]
        vs = v_all[si]
        if np.sum((vs[:-1] < 0) & (vs[1:] >= 0)) > 0:
            ini = _spike_init(v_all, refs, coords=coords, soma_i=si)
            return ini[0] if ini else "?"
    return "no spike"


def _page(pdf, cfg, pref, layer, lengths, dists, amps, M, rng):
    orig = cfg.axon_len_um
    P = rng.uniform(-cfg.area_half_um, cfg.area_half_um, size=(M, 2))
    TH = rng.uniform(0, 360, size=M)
    res = {}
    for L in lengths:
        cfg.axon_len_um = L
        asc = reduced_asc(find_one_morphology(pref), layer, out_path="_axs.asc")
        cell = build_rich_cell(asc)
        pd = [np.mean([spikes_at(cell, (d, 0.0), float(th), i0_uA=cfg.i0_uA)[0] > 0
                       for th in (0, 90, 180, 270)]) for d in dists]
        pa = [np.mean([spikes_at(cell, tuple(P[k]), float(TH[k]), i0_uA=a)[0] > 0
                       for k in range(M)]) for a in amps]
        site = _init_site(cell)
        res[L] = dict(pd=np.array(pd), pa=np.array(pa), site=site)
        os.remove(asc); del cell
    cfg.axon_len_um = orig
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    fig.suptitle(f"specimen_{pref} L{int(layer)} - robustness vs stylized axon length",
                 fontweight="bold")
    for L in lengths:
        tag = "6mm (full)" if L == 0 else f"{int(L)}um"
        ax[0].plot(dists, res[L]["pd"], "s-", label=tag)
        ax[1].plot(amps, res[L]["pa"], "o-", label=tag)
    ax[0].set_xlabel("distance (um)"); ax[0].set_ylabel("P(spike)"); ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title("footprint vs distance", fontsize=9); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("amplitude (uA)"); ax[1].set_ylabel("P(spike)"); ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title("recruitment vs amplitude", fontsize=9); ax[1].legend(fontsize=8)
    ax[2].axis("off")
    rows = [["axon", "init site"]] + [["6mm" if L == 0 else f"{int(L)}um", res[L]["site"]] for L in lengths]
    tb = ax[2].table(cellText=rows, loc="center", cellLoc="center")
    tb.scale(1, 2); ax[2].set_title("spike-initiation site", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)
    print(f"  {pref} L{int(layer)}: " + ", ".join(
        ("6mm" if L == 0 else f"{int(L)}um") + "=" + res[L]["site"] for L in lengths))


def main(lengths=(0.0, 1000.0, 500.0), prefs=None, layers=None,
         dists=(20, 40, 60, 80, 100, 120), amps=(10, 20, 30, 50, 75, 100, 150), M=15):
    from matplotlib.backends.backend_pdf import PdfPages
    cfg = CFG
    prefs = prefs or cfg.morphologies
    layers = layers if layers is not None else cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "axon_sensitivity.pdf")
    rng = np.random.default_rng(cfg.seed)
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                _page(pdf, cfg, pref, layer, lengths, dists, amps, M, rng)
    print("done:", out)
    return out


if __name__ == "__main__":
    main()
