"""
lambda_metrics.py -- d_lambda segmentation metrics per morphology (LFPy-free).

Computes the d_lambda rule directly from the morphology geometry (morphio) with
the analytic AC length constant at f=100 Hz,

    lambda_f = 1e5 * sqrt( d / (4*pi*f*Ra*cm) )   [um]   (d um, Ra Ohm-cm, cm uF/cm^2)
    nseg     = int((L / (0.1*lambda_f) + 0.9)/2)*2 + 1

using the Rich passive params (Ra, cm). No LFPy, no NEURON needed -> fast stage.
Per morphology: a metrics table (CSV) + one figure page (morphology coloured by
segment length + histogram of seg_len/lambda vs the 0.1 criterion).

Run:  python lambda_metrics.py       # -> lambda_report.pdf + lambda_metrics.csv
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages

import morphio
from morphio import SectionType
morphio.set_maximum_warnings(0)

from morphologies import all_morphologies

F_HZ = 100.0
RA = 495.73          # Rich passive (Ohm-cm)
CM = 1.5967          # Rich passive (uF/cm^2)  (soma uses 1.0; dendrites dominate the count)


def lambda_f(diam_um, Ra=RA, cm=CM, f=F_HZ):
    return 1e5 * np.sqrt(np.maximum(diam_um, 1e-3) / (4.0 * np.pi * f * Ra * cm))


def per_section(asc):
    """Per section: 2D polyline, mean diam, L, lambda, nseg, seg_len, seg/lambda."""
    m = morphio.Morphology(asc)
    sc = np.asarray(m.soma.points, float).mean(0)
    rows = []
    for s in m.iter():
        if s.type == SectionType.axon:
            continue
        P = np.asarray(s.points, float)
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        if L <= 0:
            continue
        d = float(np.asarray(s.diameters, float).mean())
        lam = float(lambda_f(d))
        nseg = int((L / (0.1 * lam) + 0.9) / 2) * 2 + 1
        rows.append(dict(xy=P[:, :2] - sc[:2], L=L, lam=lam, nseg=nseg,
                         seg=L / nseg, ratio=(L / nseg) / lam))
    return rows


def page(pdf, name, rows):
    seg = np.array([r["seg"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    nseg_tot = int(sum(r["nseg"] for r in rows))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(f"{name} - d_lambda (f={F_HZ:.0f}Hz, Ra={RA}, cm={CM}) : "
                 f"{nseg_tot} segments", fontweight="bold")
    lines, vals = [], []
    for r in rows:
        for i in range(len(r["xy"]) - 1):
            lines.append([r["xy"][i], r["xy"][i + 1]]); vals.append(r["seg"])
    lc = LineCollection(lines, cmap="viridis", array=np.array(vals), lw=1.0)
    ax[0].add_collection(lc); ax[0].set_aspect("equal"); ax[0].autoscale()
    ax[0].set_title("segment length (um)", fontsize=10)
    ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)")
    fig.colorbar(lc, ax=ax[0], fraction=0.046, label="um")
    ax[1].hist(ratio, bins=40, color="#4477aa")
    ax[1].axvline(0.1, color="r", ls="--", lw=1.2, label="d_lambda limit (0.1)")
    ax[1].set_xlabel("seg_len / lambda"); ax[1].set_ylabel("# sections")
    ax[1].set_title("compliance", fontsize=10); ax[1].legend(fontsize=8)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return dict(morphology=name, sections=len(rows), total_nseg=nseg_tot,
                median_seg_um=round(float(np.median(seg)), 2),
                max_seg_over_lambda=round(float(ratio.max()), 4),
                dlambda_ok=bool(ratio.max() < 0.11))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(here, "lambda_report.pdf")
    csv_path = os.path.join(here, "lambda_metrics.csv")
    out = []
    with PdfPages(pdf_path) as pdf:
        for name, asc in all_morphologies():
            rows = per_section(asc)
            out.append(page(pdf, name, rows))
            print(f"  {name}: nseg={out[-1]['total_nseg']}, "
                  f"median_seg={out[-1]['median_seg_um']}um, "
                  f"max seg/lambda={out[-1]['max_seg_over_lambda']}, ok={out[-1]['dlambda_ok']}")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("done:", pdf_path, "|", csv_path)


if __name__ == "__main__":
    main()
