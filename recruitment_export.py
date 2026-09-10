"""
recruitment_export.py -- the DELIVERABLE: the joint activation probability
P(spike | x, y, theta, morphology, layer) at the lab amplitude, saved as DATA (CSV) plus a
top-down P(x,y) map. Feeds the point-neuron / network reduction.

ONE KNOB drives the sampling everywhere: config.n_samples (number of RANDOM soma positions
per morphology x layer). Each position is tested over config.orient_deg orientations. This is
the SAME knob used by recruitment.py, rich_footprint.py (area) and rich_stim.py (bda) -- change
config.n_samples once and every sampling stage follows.

Outputs:
  recruitment_Pactivation.csv  : one row per (morph, layer, x, y, theta) -> fired 0/1 (+ distance)
  recruitment_Pmap.pdf         : top-down P(x,y) map (mean over theta), binned from the random
                                 samples, electrodes overlaid, one page per (morph, layer)

Sampling region: the footprint box +/-span_um, INDEPENDENT of config.area_half_um (which only
sizes the network). So the P table always covers the footprint no matter the network area.

Fixed amplitude = config.i0_uA (50 uA, lab). Single pulse (valid: block is sustained, coupling
is passive, train is invariant -- see the handoff).

Run:   python recruitment_export.py                 # pilot: first morphology, all layers
       python recruitment_export.py all             # all morphologies x layers (HEAVY on HPC)
Smoke: python smoke_test_recruitment_export.py       # offline, no NEURON

Cost = config.n_samples * len(config.orient_deg) * n_morph * n_layer light sims. The script
TIMES the first sim and prints the estimated total, so you can size the HPC job.
"""
import os
import csv
import time
import numpy as np

# ------------------------------------------------------------------ #
# PURE HELPERS (no NEURON -> unit-testable offline)                   #
# ------------------------------------------------------------------ #

def nearest_electrode_dist(x, y, elec_xy):
    """Min Euclidean distance (um) from (x,y) to any electrode."""
    e = np.asarray(elec_xy, float)
    return float(np.min(np.hypot(e[:, 0] - x, e[:, 1] - y)))


def p_over_theta(fired_list):
    """Activation probability = fraction of orientations that fired. [] -> nan."""
    f = np.asarray(fired_list, float)
    return float(np.mean(f)) if f.size else float("nan")


def bin_map(xs, ys, pvals, half, n_bins):
    """Bin scattered (x,y,P) samples into an n_bins x n_bins grid of MEAN P.
    Returns (Z, extent). Empty bins are nan. Used to turn random samples into a heatmap."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); pvals = np.asarray(pvals, float)
    edges = np.linspace(-half, half, int(n_bins) + 1)
    Z = np.full((int(n_bins), int(n_bins)), np.nan)
    ix = np.clip(np.digitize(xs, edges) - 1, 0, int(n_bins) - 1)
    iy = np.clip(np.digitize(ys, edges) - 1, 0, int(n_bins) - 1)
    for j in range(int(n_bins)):
        for i in range(int(n_bins)):
            m = (ix == i) & (iy == j)
            if m.any():
                Z[j, i] = float(np.mean(pvals[m]))
    return Z, [-half, half, -half, half]


# ------------------------------------------------------------------ #
# NEURON-BACKED EXPORT                                               #
# ------------------------------------------------------------------ #

def recruitment_export(prefs=None, layers=None, span_um=150.0, thetas=None,
                       n_samples=None, i0_uA=None, n_bins=13,
                       csv_path="recruitment_Pactivation.csv"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from config import CFG
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    from rich_cell import build_rich_cell
    from rich_footprint import spikes_at
    import field as F

    cfg = CFG
    prefs = prefs or [cfg.morphologies[0]]                 # pilot default: ONE morphology
    layers = layers or cfg.layers_um
    thetas = cfg.orient_deg if thetas is None else thetas
    N = cfg.n_samples if n_samples is None else int(n_samples)   # ONE knob
    i0 = cfg.i0_uA if i0_uA is None else float(i0_uA)
    elec, sign = F.default_array(monopolar=not cfg.bipolar)      # match spikes_at's array
    total = len(prefs) * len(layers) * N * len(thetas)
    print(f"[recruitment_export] {len(prefs)} morph x {len(layers)} layers x "
          f"N={N} random positions x {len(thetas)} thetas = {total} sims @ {i0:.0f} uA")

    here = os.path.dirname(os.path.abspath(__file__))
    csv_full = os.path.join(here, csv_path)
    pdf_full = os.path.join(here, "recruitment_Pmap.pdf")
    timed = False
    with open(csv_full, "w", newline="") as fh, PdfPages(pdf_full) as pdf:
        w = csv.writer(fh)
        w.writerow(["morphology", "layer_um", "x_um", "y_um", "theta_deg",
                    "dist_nearest_elec_um", "i0_uA", "fired"])
        for pref in prefs:
            for layer in layers:
                asc = reduced_asc(find_one_morphology(pref), layer, out_path="_rexp.asc")
                cell = build_rich_cell(asc)
                rng = np.random.default_rng(cfg.seed)
                XY = cfg.random_positions(n=N, rng=rng, half=span_um)   # same shared sampler
                xs, ys, ps = [], [], []
                for (x, y) in XY:
                    fired_list = []
                    for th in thetas:
                        if not timed:
                            t0 = time.time()
                        fired = int(spikes_at(cell, (float(x), float(y)), float(th), i0_uA=i0)[0] > 0)
                        if not timed:
                            dt1 = time.time() - t0
                            print(f"[recruitment_export] first sim {dt1*1000:.0f} ms "
                                  f"-> est. total ~{dt1*total/60:.1f} min single-core")
                            timed = True
                        fired_list.append(fired)
                        d = nearest_electrode_dist(x, y, elec)
                        w.writerow([pref, int(layer), round(float(x), 2), round(float(y), 2),
                                    int(th), round(d, 2), round(i0, 1), fired])
                    xs.append(float(x)); ys.append(float(y)); ps.append(p_over_theta(fired_list))
                if asc and os.path.exists(asc):
                    os.remove(asc)
                del cell

                Z, extent = bin_map(xs, ys, ps, span_um, n_bins)
                fig, a = plt.subplots(figsize=(6.2, 5.4))
                im = a.imshow(Z, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1,
                              aspect="equal", interpolation="nearest")
                for (ex, ey), s in zip(elec, sign):
                    a.scatter([ex], [ey], marker="+" if s > 0 else "_", s=130, c="cyan", linewidths=2)
                a.set_xlabel("x (um)"); a.set_ylabel("y (um)")
                a.set_title(f"{pref} L{int(layer)} - P(activation) @ {i0:.0f} uA "
                            f"(N={N} random pos, mean over {len(thetas)} theta)\n"
                            f"cyan + = anode, _ = cathode", fontsize=9)
                fig.colorbar(im, ax=a, label="P(spike)")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                pmax = np.nanmax(Z) if np.isfinite(Z).any() else float("nan")
                print(f"  {pref} L{int(layer)}: P max={pmax:.2f}, "
                      f"mean fired={np.mean(ps):.3f} over N={N}")
    print("done:", csv_full, "and", pdf_full)
    return csv_full, pdf_full


if __name__ == "__main__":
    import sys
    from config import CFG
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        recruitment_export(prefs=CFG.morphologies)      # all morphologies x layers (HEAVY)
    else:
        recruitment_export()                            # pilot: first morphology only
