"""
recruitment.py -- recruitment / activation-probability curves (validation + the
probability distribution that feeds the point-neuron reduction).

Three curves, per morphology:
  (1) P(spike) vs stimulus amplitude   (M random pos/theta samples)
  (2) P(spike) vs distance from the electrode cluster (fixed amplitude, random theta)
  (3) P(spike) vs orientation theta    (fixed near-electrode position, a few amps)

Uses the light soma-only spike test (rich_footprint.spikes_at). Sizes are small by
default -> raise them on HPC. Heavy: sums of many short active runs.

Run:  python recruitment.py            # uses CFG
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import CFG
from morphologies import find_one_morphology
from slicer import reduced_asc
from rich_cell import build_rich_cell
from rich_footprint import spikes_at


def p_vs_amplitude(cell, cfg, amps, M=None, rng=None):
    rng = rng or np.random.default_rng(cfg.seed)
    M = cfg.n_samples if M is None else int(M)       # ONE knob: config.n_samples
    P = cfg.random_positions(n=M, rng=rng)           # shared random sampler
    TH = rng.uniform(0, 360, size=M)
    out = []
    for a in amps:
        fired = sum(spikes_at(cell, tuple(P[k]), float(TH[k]), i0_uA=a)[0] > 0 for k in range(M))
        out.append(fired / M)
    return np.array(out)


def p_vs_distance(cell, cfg, dists, i0_uA, thetas=(0, 90, 180, 270)):
    out = []
    for d in dists:
        fired = any_all = 0
        for th in thetas:
            fired += spikes_at(cell, (d, 0.0), float(th), i0_uA=i0_uA)[0] > 0
        out.append(fired / len(thetas))
    return np.array(out)


def p_vs_theta(cell, cfg, thetas, amps, pos=(58.0, 0.0)):
    grid = np.zeros((len(amps), len(thetas)))
    for i, a in enumerate(amps):
        for j, th in enumerate(thetas):
            grid[i, j] = spikes_at(cell, pos, float(th), i0_uA=a)[0] > 0
    return grid


def main(prefs=None, amps=(10, 20, 30, 50, 75, 100, 150), M=None,
         dists=(20, 40, 60, 80, 100, 120), thetas=(0, 45, 90, 135, 180, 225, 270, 315),
         theta_amps=(30, 45)):
    cfg = CFG
    prefs = prefs or cfg.morphologies
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "recruitment.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in cfg.layers_um:
                asc = reduced_asc(find_one_morphology(pref), layer, out_path="_rec.asc")
                cell = build_rich_cell(asc); rng = np.random.default_rng(cfg.seed)
                pa = p_vs_amplitude(cell, cfg, amps, M=M, rng=rng)
                pd = p_vs_distance(cell, cfg, dists, i0_uA=cfg.i0_uA)
                pt = p_vs_theta(cell, cfg, thetas, amps=theta_amps)  # near-threshold pos/amps
                os.remove(asc)
                fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
                fig.suptitle(f"specimen_{pref} - recruitment (layer {int(layer)}um)", fontweight="bold")
                ax[0].plot(amps, pa, "o-"); ax[0].set_xlabel("amplitude (uA)"); ax[0].set_ylabel("P(spike)")
                ax[0].set_title(f"vs amplitude (M={M} random pos/theta)", fontsize=9); ax[0].set_ylim(-0.05, 1.05)
                ax[1].plot(dists, pd, "s-", color="crimson"); ax[1].set_xlabel("distance from cluster (um)")
                ax[1].set_ylabel("P(spike)"); ax[1].set_title(f"vs distance (i0={cfg.i0_uA}uA)", fontsize=9); ax[1].set_ylim(-0.05, 1.05)
                for i, a in enumerate(theta_amps):
                    ax[2].plot(thetas, pt[i], "^-", label=f"{a:.0f}uA")
                ax[2].set_xlabel("theta (deg)"); ax[2].set_ylabel("spike (0/1)")
                ax[2].set_title("vs orientation (near-threshold, pos 58um)", fontsize=9)
                ax[2].legend(fontsize=8); ax[2].set_ylim(-0.05, 1.05)
                fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} layer {int(layer)}: P(amp)={np.round(pa,2)}")
    print("done:", out)
    return out


if __name__ == "__main__":
    main()
