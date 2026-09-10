"""
ve_field.py -- the extracellular potential Ve as a SPATIAL FIELD in the well.

Ve(x,y) = sum_e  sign_e * i0 / (2*pi*sigma*r_e)   [bipolar: 2 anodes +, 2 cathodes -]
evaluated on the soma plane (z = h_soma). This is the field itself, independent
of any neuron: positive lobe over the anodes, negative over the cathodes.

Run:  cd estim && python ve_field.py     # -> ve_field.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import field as F

AREA_HALF = 200.0                 # 200x200 um well
H_SOMA = 10.0


def ve_grid(i0_uA=50.0, half=AREA_HALF + 40, n=241, z=H_SOMA,
            sigma_Sm=1.5, rmin_um=12.5):
    elec, sign = F.default_array(monopolar=False)
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    g = F.geom_factor(pts, elec, sign, sigma_Sm=sigma_Sm, rmin_um=rmin_um)   # mV/A
    Ve = (g * i0_uA * 1e-6).reshape(X.shape)                                 # mV
    return X, Y, Ve, elec, sign


def main(i0_uA=50.0):
    here = os.path.dirname(os.path.abspath(__file__))
    X, Y, Ve, elec, sign = ve_grid(i0_uA)
    anod, cath = elec[sign > 0], elec[sign < 0]
    vlim = float(np.percentile(np.abs(Ve), 99))

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Ve -- spatial field in the well (bipolar +/-50 uA, soma plane z=10 um)",
                 fontweight="bold")
    for a, (peak, Vp) in zip(ax, [("POSITIVE peak (+50 uA)", Ve),
                                  ("NEGATIVE peak (-50 uA)", -Ve)]):
        im = a.pcolormesh(X, Y, Vp, cmap="PuOr_r", vmin=-vlim, vmax=vlim, shading="auto")
        a.contour(X, Y, Vp, levels=np.linspace(-vlim, vlim, 15), colors="k",
                  linewidths=0.4, alpha=0.5)
        for (ex, ey) in anod:
            a.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="red", edgecolor="k", zorder=5))
        for (ex, ey) in cath:
            a.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="blue", edgecolor="k", zorder=5))
        a.add_patch(Rectangle((-AREA_HALF, -AREA_HALF), 2*AREA_HALF, 2*AREA_HALF,
                              fill=False, ec="0.3", ls="--", lw=1.0))
        a.set_aspect("equal"); a.set_title(peak, fontsize=10)
        a.set_xlabel("x (um)"); a.set_ylabel("y (um)")
        fig.colorbar(im, ax=a, fraction=0.046, label="Ve (mV)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(here, "ve_field.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"Ve range: {Ve.min():.1f} .. {Ve.max():.1f} mV | done: {out}")
    return out


if __name__ == "__main__":
    main()
