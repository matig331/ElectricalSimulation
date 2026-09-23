"""plot_bump_statistics.py -- the post-pulse kinetics as a statistic over a campaign.

Reads any CSV carrying the bump_kinetics.STAGED_COLUMNS (a raw part, a per-outcome
culture_P*.csv, or a merged file) and writes ONE multi-page PDF plus a summary CSV.

    python plot_bump_statistics.py --input merged_full_tuned/culture_Pactivation.csv
    python plot_bump_statistics.py --input demo_full_tuned --bin-um 40

WHAT IS PLOTTED, AND WHY IT IS SPLIT IN TWO
A neuron either has a measurable bump or it does not, and the two questions need different
plots. Measured on the demo campaign: beyond ~350 um the bump is ~0.001 mV, three orders of
magnitude below the near-field 0.75 mV, and the fit is correctly REJECTED there (fit_ok 0)
with its taus pinned to the search bounds. Averaging those numbers in would be meaningless.

So the amplitude page answers "how often is there a bump, and how big" over ALL neurons
(rejected ones counted as no bump), and the kinetics page describes the SHAPE using only the
accepted fits. Every panel states its own n. Rows with fit_ok = 0 are never silently dropped:
their count is the point of the first panel.

NOTE on reading the CSV: a rejected fit still carries numbers in dexp_tau_rise_ms and friends
-- they are the parameters of a model that does not describe the trace. ALWAYS filter on
dexp_fit_ok = 1 before quoting a tau. This script does.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Okabe-Ito, fixed order, CVD-validated against a light surface. Identity is never colour
# alone: every series is also separated by marker or line style and carries a legend entry.
C_MAIN = "#0072B2"      # the accepted fits / the measured quantity
C_REJ = "#999999"       # rejected fits, shown but never averaged
C_BAND = "#56B4E9"      # binned median band
C_ALT = "#009E73"       # the second series on a panel
C_WARN = "#D55E00"      # thresholds and reference lines
C_INK, C_INK2, C_GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"

DIST_COL = "dist_dipole3d_um"


# ============================ 1. loading ============================
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    """Read one CSV, or every culture_P*.csv in a directory (the first is enough -- the
    kinetics columns are identical in all three, only the outcome column differs).

    Returns a dict of numpy arrays. Raises if the kinetics columns are absent, which is what
    a short-window (bump_ms = 0) campaign looks like.
    """
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*Pactivation*.csv"))) or \
            sorted(glob.glob(os.path.join(path, "*.csv")))
        if not cands:
            raise SystemExit("no CSV found in %s" % path)
        path = cands[0]
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("%s is empty" % path)
    need = ("dexp_peak_mV", "dexp_fit_ok", "dexp_tau_rise_ms", "dexp_tau_decay_ms")
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise SystemExit("%s has no kinetics columns (%s). It was written with bump_ms = 0, "
                         "or by an older version." % (path, ", ".join(missing)))
    get = lambda k: np.array([_f(r.get(k, "")) for r in rows])
    # measured (model-free) bump peak: campaign rows carry dexp_data_peak_mV, the example-
    # trace CSV of older versions carried bump_data_peak_mV. A CSV with neither can only show
    # rejected fits at their FITTED peak -- which is meaningless -- and the figure says so.
    if "dexp_data_peak_mV" in rows[0]:
        data_peak, measured = get("dexp_data_peak_mV"), True
    elif "bump_data_peak_mV" in rows[0]:
        data_peak, measured = get("bump_data_peak_mV"), True
    else:
        data_peak, measured = get("dexp_peak_mV"), False
        print("WARNING: %s has no measured-peak column; rejected fits are shown at their "
              "FITTED peak, which is not a measurement. Re-run with the current code." % path)
    return dict(path=path, n=len(rows),
                dist=get(DIST_COL), theta_pos=get("theta_pos_deg"),
                theta_or=get("theta_orient_deg"), layer=get("layer_um"),
                morph=np.array([r.get("morphology", "") for r in rows]),
                dv_end=get("deltaVm_end_phase2_mV"),
                peak=get("dexp_peak_mV"), t_peak=get("dexp_t_peak_ms"),
                tau_r=get("dexp_tau_rise_ms"), tau_d=get("dexp_tau_decay_ms"),
                r2=get("dexp_r2"), ok=get("dexp_fit_ok") > 0.5,
                data_peak=data_peak, measured_peak=measured,
                t_1e=get("early_t_1e_ms"), tau_m=get("early_tau_ms"),
                early_ok=get("early_fit_ok") > 0.5)


# ============================ 2. aggregation ============================
def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion -- the same estimator the outcome
    statistics use. Correct at the small n a per-bin count gives, where the normal
    approximation is not."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    out_c = np.full(k.shape, np.nan)
    out_lo = np.full(k.shape, np.nan)
    out_hi = np.full(k.shape, np.nan)
    m = n > 0
    if not np.any(m):
        return out_c, out_lo, out_hi
    ph = k[m] / n[m]
    den = 1.0 + z * z / n[m]
    cen = (ph + z * z / (2 * n[m])) / den
    half = z * np.sqrt(ph * (1 - ph) / n[m] + z * z / (4 * n[m] * n[m])) / den
    out_c[m], out_lo[m], out_hi[m] = cen, np.maximum(0, cen - half), np.minimum(1, cen + half)
    return out_c, out_lo, out_hi


def bin_by(x, y, edges, mask=None):
    """Per-bin n, median and inter-quartile range of y. Bins with no sample give nan, never 0
    -- an empty bin is a gap in the figure, not a measurement of zero."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        keep &= np.asarray(mask, bool)
    idx = np.digitize(x[keep], edges) - 1
    n = len(edges) - 1
    cnt = np.zeros(n, int)
    med = np.full(n, np.nan)
    q1 = np.full(n, np.nan)
    q3 = np.full(n, np.nan)
    for b in range(n):
        v = y[keep][idx == b]
        cnt[b] = v.size
        if v.size:
            med[b] = np.median(v)
            q1[b], q3[b] = np.percentile(v, 25), np.percentile(v, 75)
    return cnt, med, q1, q3


def fraction_by(x, flag, edges):
    """Per-bin OBSERVED proportion k/n of `flag`, with Wilson intervals around it.

    The plotted line is k/n, NOT the Wilson centre. The two differ sharply at small n, and in
    the wrong direction: for k = 0, n = 1 the Wilson centre is 0.40, so a bin where no neuron
    had a bump would be drawn at 40 %. The interval still carries the uncertainty -- that is
    its job -- but the point estimate has to be what was measured.

    Returns (n, proportion, lo, hi).
    """
    x = np.asarray(x, float)
    flag = np.asarray(flag, bool)
    keep = np.isfinite(x)
    idx = np.digitize(x[keep], edges) - 1
    n = len(edges) - 1
    k = np.zeros(n, float)
    tot = np.zeros(n, float)
    for b in range(n):
        sel = idx == b
        tot[b] = float(np.sum(sel))
        k[b] = float(np.sum(flag[keep][sel]))
    _cen, lo, hi = wilson(k, tot)
    prop = np.divide(k, tot, out=np.full(k.shape, np.nan), where=tot > 0)
    return tot, prop, lo, hi


# ============================ 3. plotting ============================
def _style(ax, xlabel, ylabel, title=None):
    ax.grid(True, color=C_GRID, lw=0.5, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=C_INK2, labelsize=7.5, length=3, width=0.8)
    ax.set_xlabel(xlabel, fontsize=8, color=C_INK)
    ax.set_ylabel(ylabel, fontsize=8, color=C_INK)
    if title:
        ax.set_title(title, fontsize=9, color=C_INK, loc="left", pad=6)


def _note(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=C_INK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRID, lw=0.7, alpha=0.92))


def _centers(edges):
    return 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))


def page_amplitude(d, edges, amp_thresh):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    x = _centers(edges)

    ax = axes[0]
    rej = ~d["ok"]
    ax.scatter(d["dist"][rej], d["data_peak"][rej], s=16, c=C_REJ, marker="x", lw=0.9,
               zorder=3, label="no measurable bump (fit rejected)")
    ax.scatter(d["dist"][d["ok"]], d["peak"][d["ok"]], s=22, c=C_MAIN, marker="o",
               edgecolors="white", linewidths=0.5, zorder=4, label="fitted bump peak")
    cnt, med, q1, q3 = bin_by(d["dist"], d["peak"], edges, mask=d["ok"])
    good = cnt > 0
    if good.any():
        ax.fill_between(x[good], q1[good], q3[good], color=C_BAND, alpha=0.30, lw=0, zorder=2,
                        label="binned IQR")
        ax.plot(x[good], med[good], color=C_MAIN, lw=2.0, zorder=5, label="binned median")
    ax.axhline(amp_thresh, color=C_WARN, ls="--", lw=1.0, zorder=3,
               label="amplitude gate %.2f mV" % amp_thresh)
    ax.set_yscale("symlog", linthresh=amp_thresh)
    _style(ax, "distance from the dipole centre (um)", "bump peak (mV)",
           "amplitude: how big, over every neuron")
    ax.text(0.98, 0.97, "n = %d neurons\nfit accepted %d (%.0f%%)"
            % (d["n"], int(d["ok"].sum()), 100.0 * d["ok"].mean()),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=C_INK,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRID, lw=0.7, alpha=0.92))
    ax.legend(fontsize=6.4, loc="lower left", frameon=True, framealpha=0.92, edgecolor=C_GRID)

    ax = axes[1]
    tot, c, lo, hi = fraction_by(d["dist"], d["ok"], edges)
    g = tot > 0
    ax.fill_between(x[g], lo[g], hi[g], color=C_BAND, alpha=0.35, lw=0, zorder=2,
                    label="Wilson 95% CI")
    ax.plot(x[g], c[g], color=C_MAIN, lw=2.0, marker="o", ms=5, zorder=4,
            label="observed fraction k/n")
    for xi, ni in zip(x[g], tot[g]):
        ax.annotate("%d" % ni, xy=(xi, 0.0), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=6, color=C_INK2)
    ax.set_ylim(-0.03, 1.03)
    _style(ax, "distance from the dipole centre (um)", "fraction of neurons",
           "prevalence: how often there is a bump at all")
    _note(ax, "n per bin printed on the axis\nbin = %.0f um\ntotal %d neurons"
          % (edges[1] - edges[0], d["n"]))
    ax.legend(fontsize=6.6, loc="upper right", frameon=True, framealpha=0.92, edgecolor=C_GRID)

    fig.suptitle("Post-pulse Ih bump -- amplitude and prevalence", fontsize=11, color=C_INK,
                 x=0.055, ha="left", y=0.975)
    if d.get("measured_peak", True):
        sub = ("A rejected fit is plotted at its MEASURED bump peak (grey x), never at its "
               "fitted peak: with no bump to fit, the model's parameters are meaningless and "
               "its taus sit on the search bounds.")
    else:
        sub = ("WARNING: this CSV has no measured-peak column, so rejected fits (grey x) are "
               "shown at their FITTED peak, which is NOT a measurement. Re-run the simulation "
               "with the current code.")
    fig.text(0.055, 0.915, sub, fontsize=7, color=C_INK2, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.885])
    return fig


def page_shape(d, edges):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    x = _centers(edges)
    ok = d["ok"]

    for ax, key, lab, col in ((axes[0, 0], "tau_r", "tau_rise (ms)", C_MAIN),
                              (axes[0, 1], "tau_d", "tau_decay (ms)", C_ALT)):
        ax.scatter(d["dist"][ok], d[key][ok], s=22, c=col, marker="o", edgecolors="white",
                   linewidths=0.5, zorder=4, label="accepted fits")
        cnt, med, q1, q3 = bin_by(d["dist"], d[key], edges, mask=ok)
        g = cnt > 0
        if g.any():
            ax.fill_between(x[g], q1[g], q3[g], color=col, alpha=0.20, lw=0, zorder=2,
                            label="binned IQR")
            ax.plot(x[g], med[g], color=col, lw=2.0, zorder=5, label="binned median")
        v = d[key][ok]
        v = v[np.isfinite(v)]
        if v.size:
            ax.axhline(np.median(v), color=C_WARN, ls="--", lw=1.0, zorder=3,
                       label="overall median %.0f ms" % np.median(v))
            _note(ax, "n = %d\nmedian %.1f ms\nIQR %.1f - %.1f ms"
                  % (v.size, np.median(v), np.percentile(v, 25), np.percentile(v, 75)))
        _style(ax, "distance from the dipole centre (um)", lab,
               "%s -- a property of Ih, not of placement" % lab.split(" ")[0])
        ax.legend(fontsize=6.6, loc="upper right", frameon=True, framealpha=0.92,
                  edgecolor=C_GRID)

    ax = axes[1, 0]
    v = d["t_peak"][ok]
    v = v[np.isfinite(v)]
    if v.size:
        ax.hist(v, bins=max(5, min(20, v.size)), color=C_MAIN, edgecolor="white", lw=0.6)
        ax.axvline(np.median(v), color=C_WARN, ls="--", lw=1.2,
                   label="median %.0f ms" % np.median(v))
        _note(ax, "n = %d\nmedian %.1f ms\nrange %.1f - %.1f ms"
              % (v.size, np.median(v), v.min(), v.max()))
        ax.legend(fontsize=6.6, loc="upper right", frameon=True, framealpha=0.92,
                  edgecolor=C_GRID)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # counts
    _style(ax, "time to the bump peak (ms)", "neurons", "when the bump peaks")

    ax = axes[1, 1]
    eok = d["early_ok"] & np.isfinite(d["t_1e"])
    ax.scatter(d["dist"][eok], d["t_1e"][eok], s=22, c=C_MAIN, marker="o",
               edgecolors="white", linewidths=0.5, zorder=4, label="t_1e (measured)")
    cnt, med, q1, q3 = bin_by(d["dist"], d["t_1e"], edges, mask=eok)
    g = cnt > 0
    if g.any():
        ax.plot(x[g], med[g], color=C_MAIN, lw=2.0, zorder=5, label="binned median")
    v = d["t_1e"][eok]
    if v.size:
        _note(ax, "n = %d\nmedian %.3f ms\nrange %.3f - %.3f ms"
              % (v.size, np.median(v), v.min(), v.max()))
    _style(ax, "distance from the dipole centre (um)",
           "1/e time of the direct relaxation (ms)",
           "direct relaxation: sub-millisecond, model-free")
    ax.legend(fontsize=6.6, loc="upper right", frameon=True, framealpha=0.92, edgecolor=C_GRID)

    fig.suptitle("Post-pulse Ih bump -- shape, from the accepted fits only", fontsize=11,
                 color=C_INK, x=0.055, ha="left", y=0.978)
    fig.text(0.055, 0.935, "t_1e is the measured time for |DeltaV| to fall to 1/e of its peak. "
             "It is quoted instead of the fitted tau_m because the direct relaxation is "
             "multi-exponential, so a fitted tau depends on the window it came from.",
             fontsize=7, color=C_INK2, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    return fig


def page_geometry(d, n_bins=12):
    """Does the bump depend on orientation and layer, as expected?"""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ok = d["ok"]

    ax = axes[0]
    edges = np.linspace(-180.0, 180.0, n_bins + 1)
    th = d["theta_or"]
    ax.scatter(th[ok], d["peak"][ok], s=22, c=C_MAIN, marker="o", edgecolors="white",
               linewidths=0.5, zorder=4, label="accepted fits")
    cnt, med, q1, q3 = bin_by(th, d["peak"], edges, mask=ok)
    x = _centers(edges)
    g = cnt > 0
    if g.any():
        ax.fill_between(x[g], q1[g], q3[g], color=C_BAND, alpha=0.30, lw=0, zorder=2,
                        label="binned IQR")
        ax.plot(x[g], med[g], color=C_MAIN, lw=2.0, zorder=5, label="binned median")
    _style(ax, "orientation relative to the dipole axis (deg)", "bump peak (mV)",
           "orientation")
    _note(ax, "n = %d accepted" % int(ok.sum()))
    ax.legend(fontsize=6.6, loc="upper right", frameon=True, framealpha=0.92, edgecolor=C_GRID)

    ax = axes[1]
    groups, labels = [], []
    for lay in sorted(set(d["layer"][np.isfinite(d["layer"])])):
        v = d["peak"][ok & (d["layer"] == lay)]
        v = v[np.isfinite(v)]
        if v.size:
            groups.append(v)
            labels.append("%d um\n(n=%d)" % (int(lay), v.size))
    for m in sorted(set(d["morph"])):
        v = d["peak"][ok & (d["morph"] == m)]
        v = v[np.isfinite(v)]
        if v.size:
            groups.append(v)
            labels.append("%s\n(n=%d)" % (m, v.size))
    if groups:
        try:                                   # matplotlib >= 3.9 renamed this argument
            bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.55)
        except TypeError:
            bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.55)
        for patch in bp["boxes"]:
            patch.set_facecolor(C_BAND)
            patch.set_alpha(0.45)
            patch.set_edgecolor(C_MAIN)
        for k in ("whiskers", "caps", "medians"):
            for ln in bp[k]:
                ln.set_color(C_MAIN)
        for ln in bp["medians"]:
            ln.set_linewidth(2.0)
    _style(ax, "slice thickness, then morphology", "bump peak (mV)", "layer and morphology")
    fig.suptitle("Post-pulse Ih bump -- does it depend on geometry?", fontsize=11, color=C_INK,
                 x=0.055, ha="left", y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ============================ 4. driver ============================
SUMMARY_COLUMNS = ("quantity", "n", "median", "q1", "q3", "min", "max")


def summarize(d):
    ok = d["ok"]
    rows = [["neurons", d["n"], "", "", "", "", ""],
            ["bump fit accepted", int(ok.sum()), "", "", "", "", ""],
            ["fraction accepted", d["n"], round(float(ok.mean()), 4), "", "", "", ""]]
    for lab, v in (("bump_peak_mV", d["peak"][ok]), ("bump_t_peak_ms", d["t_peak"][ok]),
                   ("tau_rise_ms", d["tau_r"][ok]), ("tau_decay_ms", d["tau_d"][ok]),
                   ("dexp_r2", d["r2"][ok]),
                   ("t_1e_ms", d["t_1e"][d["early_ok"] & np.isfinite(d["t_1e"])])):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size:
            rows.append([lab, int(v.size), round(float(np.median(v)), 5),
                         round(float(np.percentile(v, 25)), 5),
                         round(float(np.percentile(v, 75)), 5),
                         round(float(v.min()), 5), round(float(v.max()), 5)])
        else:
            rows.append([lab, 0, "", "", "", "", ""])
    return rows


def main(input_path, out_pdf=None, out_csv=None, bin_um=40.0, amp_thresh=0.05, verbose=True):
    d = load(input_path)
    base = input_path if os.path.isdir(input_path) else os.path.dirname(
        os.path.abspath(input_path))
    out_pdf = out_pdf or os.path.join(base, "bump_statistics.pdf")
    out_csv = out_csv or os.path.join(base, "bump_statistics.csv")

    dist = d["dist"][np.isfinite(d["dist"])]
    lo = float(np.floor(dist.min() / bin_um) * bin_um) if dist.size else 0.0
    hi = float(np.ceil(dist.max() / bin_um) * bin_um) if dist.size else bin_um
    edges = np.arange(lo, hi + bin_um, bin_um)

    with PdfPages(out_pdf) as pdf:
        for fig in (page_amplitude(d, edges, amp_thresh), page_shape(d, edges),
                    page_geometry(d)):
            pdf.savefig(fig)
            plt.close(fig)

    rows = summarize(d)
    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(SUMMARY_COLUMNS)
        wr.writerows(rows)
    if verbose:
        print("read %s: %d rows, %d with an accepted bump fit"
              % (d["path"], d["n"], int(d["ok"].sum())))
        for r in rows:
            print("  %-20s n=%-5s median=%-10s IQR %s - %s" % (r[0], r[1], r[2], r[3], r[4]))
        print("done PDF:", out_pdf)
        print("done CSV:", out_csv)
    return out_pdf, out_csv


def _cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="a CSV carrying the kinetics columns, or a directory holding one")
    ap.add_argument("--bin-um", type=float, default=40.0, help="distance bin (default 40)")
    ap.add_argument("--amp-threshold", type=float, default=0.05,
                    help="the amplitude gate drawn on the figure (default 0.05 mV, the same "
                         "min_amp_mV the fit uses)")
    ap.add_argument("--out-pdf", default=None)
    ap.add_argument("--out-csv", default=None)
    a = ap.parse_args(argv)
    return main(a.input, a.out_pdf, a.out_csv, a.bin_um, a.amp_threshold)


if __name__ == "__main__":
    _cli()
    sys.exit(0)
