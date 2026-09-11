"""
culture_statistics.py -- HPC-ready neuron-level post-processing for culture_Pactivation CSV files.

Purpose
-------
Merge one or many culture_Pactivation*.csv files produced by independent HPC jobs and
estimate P(fire) treating each neuron-level observation as the statistical unit, using the
distance from the central-dipole centre (DISTANCE_COL) for all distance-based plots, as requested.

IMPORTANT
---------
The main summaries DO NOT average probabilities by culture. For every 1D or 2D bin:

    P(fire) = n_fired / n_observations

with a 95% Wilson binomial confidence interval.

The script still creates globally unique source/culture/neuron identifiers for QC and tracing,
but culture is metadata only and is not the replicate level used by the main statistics.

SAFETY GUARDS (HPC)
-------------------
- One cell model per analysis. Each file's model is its 'cell_model' column; a zip-era
  soma-only serial export (soma_only_rest_mV column, no cell_model) is tagged
  'soma_only_v0_scalar_rest' because its depol/hyperpol labels used the scalar-rest reference
  (see culture_export); files without either are the preliminary 'full_active' schema.
  Mixing models raises -- analyse results_full_active/ and results_soma_only/ separately.
- No simulation counted twice. When a 'seed' column exists the identity is
  (model, seed, LOCAL culture, neuron, layer), stable across files, so reading the same
  simulation through two files (e.g. per-job results_<model>/<job>/ AND the all-jobs
  merged_<model>/) raises instead of silently inflating N and shrinking the CIs.

If the same soma is intentionally re-simulated at several layers, those rows share the same
`neuron_global` identifier. They are kept as separate neuron-layer observations in the pooled
layer analysis. This is useful for the requested descriptive P(layer) analysis, but it should not
be described as a fully independent repeated-measures inferential test across layers.

Outputs
-------
- merged_activation.csv
- summary_1d_distance.csv
- summary_1d_theta_pos.csv
- summary_1d_theta_orient.csv
- summary_1d_layer.csv
- summary_1d_x.csv
- summary_1d_y.csv
- summary_2d_distance_theta.csv
- summary_2d_layer_theta.csv
- summary_2d_distance_layer.csv
- summary_2d_thetaPos_thetaOrient.csv
- summary_2d_distance_orientation.csv
- summary_2d_xy.csv
- summary_1d_distance_by_morphology.csv
- culture_statistics.pdf

Usage
-----
Single file:
    python culture_statistics.py --input culture_Pactivation.csv --output culture_stats

Directory containing many HPC job folders (recursive):
    python culture_statistics.py --input /path/to/results --output culture_stats

Optional bin widths:
    python culture_statistics.py --input results --output culture_stats \
        --distance-bin 15 --angle-bin 15 --orientation-bin 15 --xy-bin 20
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_BASE_COLUMNS = {
    "culture", "neuron", "morphology", "layer_um",
    "x_um", "y_um", "dist_nearest_elec_um", "dist_center_um",
    "dist_dipole3d_um", "theta_orient_deg", "theta_pos_deg"
}

# Distance used by every distance-based summary and plot: 3D Euclidean distance from the
# central-dipole centre (the column culture_export.py writes as dist_dipole3d_um). Together with
# theta_pos_deg (position angle in the SAME dipole frame) it gives a polar (r, theta) description
# around the dipole centre. To switch metric, change these two lines only.
DISTANCE_COL = "dist_dipole3d_um"
DISTANCE_LABEL = "Distance from dipole centre (um)"

# Depolarization / hyperpolarization are labelled on the MAGNITUDE of DeltaV_end (the somatic
# change at the end of phase 2, referenced to the sham run). The CSVs carry sign-only labels
# (|DeltaV| > 1e-6 mV): far from the array every soma gets a micro-volt nudge whose sign is set
# only by which way it points along the field, so random orientations give ~50/50 there -- real
# but negligible responses. A threshold keeps only responses of physiological size.
# 0 = keep the CSV's sign-only labels.
POLARIZATION_OUTCOMES = ("depolarization", "hyperpolarization")
DEFAULT_DV_THRESHOLD_MV = 1.0
DV_COL = "deltaVm_end_phase2_mV"


def polarization_labels(outcome_name, dv_mV, activated, theta_mV):
    """0/1 label of a polarization outcome with a magnitude threshold theta_mV > 0:
         depolarization    : not activated and DeltaV_end >= +theta
         hyperpolarization : not activated and DeltaV_end <= -theta
    Activated somata are never counted (their DeltaV is the spike itself). NaN -> 0."""
    dv = np.asarray(dv_mV, dtype=float)
    act = np.asarray(activated, dtype=bool)
    theta = float(theta_mV)
    if theta <= 0:
        raise ValueError("theta_mV must be > 0 (use 0 upstream to keep sign-only labels)")
    if outcome_name == "depolarization":
        hit = dv >= theta
    elif outcome_name == "hyperpolarization":
        hit = dv <= -theta
    else:
        raise ValueError(f"not a polarization outcome: {outcome_name!r}")
    return (hit & ~act & np.isfinite(dv)).astype(int)

OUTCOMES = {
    "activation": ("culture_Pactivation*.csv", "fired"),
    "depolarization": ("culture_Pdepolarization*.csv", "depolarized"),
    "hyperpolarization": ("culture_Phyperpolarization*.csv", "hyperpolarized"),
}


# =============================================================================
# Loading / merging
# =============================================================================

def discover_csvs(input_path: str | Path,
                  pattern: str = "culture_Pactivation*.csv") -> list[Path]:
    """Return activation CSV files from one file or recursively from a directory."""
    p = Path(input_path).expanduser().resolve()
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    files = sorted(p.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found under {p}")
    return files


def infer_cell_model(df: pd.DataFrame, path) -> str:
    """Cell model of one file (see SAFETY GUARDS in the module docstring)."""
    if "cell_model" in df.columns:
        vals = sorted(set(df["cell_model"].dropna().astype(str)))
        if len(vals) != 1:
            raise ValueError(f"{path}: expected ONE cell_model per file, found {vals}")
        return vals[0]
    if "soma_only_rest_mV" in df.columns or "deltaVm_end_phase2_mV" in df.columns:
        return "soma_only_v0_scalar_rest"
    return "full_active"


def load_and_merge(files: Sequence[Path], outcome_col: str, dedupe_identical: bool = False) -> pd.DataFrame:
    """Merge job CSVs and create globally unique traceable IDs (one cell model only; every
    simulation at most once -- see SAFETY GUARDS)."""
    chunks: list[pd.DataFrame] = []
    models: dict[str, list[str]] = {}

    for file_index, path in enumerate(files):
        df = pd.read_csv(path)
        missing = REQUIRED_BASE_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if outcome_col not in df.columns:
            raise ValueError(f"{path} is missing outcome column {outcome_col!r}")
        if outcome_col != "fired":
            # a stray 'fired' column would become a DUPLICATE column after the rename
            df = df.drop(columns=[c for c in ("fired",) if c in df.columns])
            df = df.rename(columns={outcome_col: "fired"})

        model = infer_cell_model(df, path)
        models.setdefault(model, []).append(str(path))
        df = df.copy()
        df["cell_model"] = model
        df["source_file"] = str(path)
        df["source_index"] = int(file_index)

        local = df["local_culture"] if "local_culture" in df.columns else df["culture"]
        if "seed" in df.columns:
            # (model, seed, LOCAL culture) is stable across files -> cross-file duplicates caught
            df["culture_global"] = (model + "_S" + df["seed"].astype(str)
                                    + "_C" + local.astype(str))
        else:
            # no seed column (zip-era export): identity can only be file-scoped
            df["culture_global"] = f"F{file_index:05d}_C" + local.astype(str)
        df["neuron_global"] = (
            df["culture_global"] + "_N" + df["neuron"].astype(str)
        )
        # Unique row/simulation observation, useful because the same soma can appear at >1 layer.
        df["observation_global"] = (
            df["neuron_global"] + "_L" + df["layer_um"].astype(str)
        )
        chunks.append(df)

    if len(models) > 1:
        detail = "; ".join(f"{m}: {len(p)} file(s), e.g. {p[0]}" for m, p in sorted(models.items()))
        raise ValueError("refusing to pool different cell models -- " + detail
                         + ". Analyse each model's results folder separately.")

    out = pd.concat(chunks, ignore_index=True)
    n_dedup = 0
    dup = out["observation_global"].duplicated(keep=False)
    if dup.any():
        n_extra = int(out["observation_global"].duplicated(keep="first").sum())
        ex = out.loc[dup, ["observation_global", "source_file"]].head(6).to_string(index=False)
        if not dedupe_identical:
            raise ValueError(
                f"{n_extra} extra rows share an observation id (model, seed, culture, neuron, "
                f"layer): the same draw appears more than once. Usual causes: (a) --input holds "
                f"BOTH per-job outputs (results_<model>/<job>/) and an all-jobs merge "
                f"(merged_<model>/) -> point --input at ONE of them; (b) two batches ran with the "
                f"SAME seeds (e.g. run1 and runB1 both seed 1000) -> the same draws simulated "
                f"twice; if identical, re-run with --dedupe-identical to keep one copy.\n{ex}")
        same = [c for c in ("morphology", "x_um", "y_um", "theta_orient_deg", "fired")
                if c in out.columns]
        nun = out.loc[dup].groupby("observation_global", sort=False)[same].nunique(dropna=False)
        bad = nun[(nun > 1).any(axis=1)]
        if len(bad):
            raise ValueError(
                f"{len(bad)} duplicated observation id(s) DISAGREE on {same}: NOT the same "
                f"simulation (different code or morphology list under the same seed) -- they "
                f"cannot be de-duplicated; choose which batch to analyse. "
                f"Examples: {list(bad.index[:5])}")
        out = out.loc[~out["observation_global"].duplicated(keep="first")].copy()
        n_dedup = n_extra
        print(f"  --dedupe-identical: dropped {n_extra} exact duplicate observation(s); "
              f"one copy of each kept")

    numeric_cols = [
        "culture", "neuron", "layer_um", "x_um", "y_um",
        "dist_nearest_elec_um", "dist_center_um", "dist_dipole3d_um",
        "theta_orient_deg", "theta_pos_deg", "fired"
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["fired"]).copy()
    out["fired"] = out["fired"].astype(int)
    bad = ~out["fired"].isin([0, 1])
    if bad.any():
        raise ValueError("Column 'fired' must contain only 0/1 values")

    out = out.reset_index(drop=True)
    out.attrs["n_dedup"] = int(n_dedup)
    return out


# =============================================================================
# Binning
# =============================================================================

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def make_edges(values: pd.Series,
               width: float | None = None,
               n_bins: int | None = None,
               explicit_range: tuple[float, float] | None = None) -> np.ndarray:
    """Create stable left-closed bin edges."""
    x = _safe_numeric(values).dropna().to_numpy(float)
    if x.size == 0:
        raise ValueError("Cannot bin an empty variable")

    if explicit_range is None:
        lo, hi = float(np.min(x)), float(np.max(x))
    else:
        lo, hi = map(float, explicit_range)

    if width is not None:
        if width <= 0:
            raise ValueError("bin width must be > 0")
        start = math.floor(lo / width) * width
        stop = math.ceil(hi / width) * width
        if stop <= start:
            stop = start + width
        # last edge = first multiple of width >= max (NOT one extra bin beyond it: that bin
        # held only values exactly equal to the max, e.g. y = 500 or theta = 180 -> spurious
        # single-sample points); _cut folds values equal to the last edge into the last bin
        return np.arange(start, stop + width * 0.5, width, dtype=float)

    if n_bins is None:
        n_bins = 12
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, int(n_bins) + 1)


def _cut(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Left-closed bins [a, b), except that a value equal to the LAST edge belongs to the last
    bin (closed on the right), so the maximum is never dropped nor given its own bin."""
    v = _safe_numeric(values)
    top = float(edges[-1])
    v = v.where(v != top, np.nextafter(top, -np.inf))
    return pd.cut(v, edges, include_lowest=True, right=False)


def interval_midpoint(x) -> float:
    if isinstance(x, pd.Interval):
        return float((x.left + x.right) / 2.0)
    try:
        return float(x)
    except Exception:
        return np.nan


# =============================================================================
# Neuron-level binomial statistics
# =============================================================================

def wilson_interval(k: np.ndarray | pd.Series,
                    n: np.ndarray | pd.Series,
                    z: float = 1.959963984540054) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for a binomial proportion."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)

    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    den = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / den
    half = (
        z * np.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n)) / den
    )
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    return lo, hi


def _neuron_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Pooled neuron-level binomial summary for arbitrary grouping columns."""
    tmp = df.dropna(subset=group_cols + ["fired"]).copy()
    if tmp.empty:
        return pd.DataFrame()

    summary = (
        tmp.groupby(group_cols, observed=True, sort=True)
           .agg(
               n_neurons=("fired", "size"),
               n_fired=("fired", "sum"),
               n_cultures=("culture_global", "nunique"),
               n_unique_somata=("neuron_global", "nunique"),
           )
           .reset_index()
    )

    summary["P_fire"] = summary["n_fired"] / summary["n_neurons"]
    summary["SE_binomial"] = np.sqrt(
        summary["P_fire"] * (1.0 - summary["P_fire"]) / summary["n_neurons"]
    )
    lo, hi = wilson_interval(summary["n_fired"], summary["n_neurons"])
    summary["ci95_low"] = lo
    summary["ci95_high"] = hi
    return summary


def summarize_1d(df: pd.DataFrame,
                 variable: str,
                 *,
                 bin_width: float | None = None,
                 n_bins: int | None = None,
                 edges: np.ndarray | None = None,
                 categorical: bool = False) -> pd.DataFrame:
    """Neuron-level P(fire | variable)."""
    work = df.copy()

    if categorical:
        group_col = variable
    else:
        if edges is None:
            edges = make_edges(work[variable], width=bin_width, n_bins=n_bins)
        group_col = f"{variable}_bin"
        work[group_col] = _cut(work[variable], edges)

    summary = _neuron_summary(work, [group_col])
    if summary.empty:
        return summary

    if categorical:
        summary["x"] = pd.to_numeric(summary[group_col], errors="coerce")
    else:
        summary["x"] = summary[group_col].map(interval_midpoint)
        summary["bin_left"] = summary[group_col].map(lambda z: float(z.left))
        summary["bin_right"] = summary[group_col].map(lambda z: float(z.right))

    return summary


def summarize_abs_dv(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """|DeltaV_end| of NON-activated somata per distance bin: n, 5/25/50/75/95 % quantiles."""
    nf = df[~df["_activated"]]
    absdv = _safe_numeric(nf[DV_COL]).abs()
    bins = _cut(nf[DISTANCE_COL], edges)
    g = absdv.groupby(bins, observed=True)
    q = g.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    q.columns = ["q05", "q25", "q50", "q75", "q95"]
    out = q.join(g.size().rename("n")).reset_index().rename(columns={DISTANCE_COL: "bin"})
    out["x"] = out["bin"].map(interval_midpoint)
    return out


def summarize_2d(df: pd.DataFrame,
                 xvar: str,
                 yvar: str,
                 *,
                 x_edges: np.ndarray | None = None,
                 y_edges: np.ndarray | None = None,
                 x_width: float | None = None,
                 y_width: float | None = None,
                 x_bins: int | None = None,
                 y_bins: int | None = None,
                 x_categorical: bool = False,
                 y_categorical: bool = False) -> pd.DataFrame:
    """Neuron-level probability surface P(fire | x, y)."""
    work = df.copy()

    if x_categorical:
        xcol = xvar
    else:
        if x_edges is None:
            x_edges = make_edges(work[xvar], width=x_width, n_bins=x_bins)
        xcol = f"{xvar}_bin"
        work[xcol] = _cut(work[xvar], x_edges)

    if y_categorical:
        ycol = yvar
    else:
        if y_edges is None:
            y_edges = make_edges(work[yvar], width=y_width, n_bins=y_bins)
        ycol = f"{yvar}_bin"
        work[ycol] = _cut(work[yvar], y_edges)

    summary = _neuron_summary(work, [xcol, ycol])
    if summary.empty:
        return summary

    summary["x"] = (
        pd.to_numeric(summary[xcol], errors="coerce")
        if x_categorical else summary[xcol].map(interval_midpoint)
    )
    summary["y"] = (
        pd.to_numeric(summary[ycol], errors="coerce")
        if y_categorical else summary[ycol].map(interval_midpoint)
    )
    return summary


# =============================================================================
# Plotting
# =============================================================================

def _plot_1d(ax, summary: pd.DataFrame, xlabel: str, title: str, ylabel: str="P(event)"):
    if summary.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return

    s = summary.sort_values("x")
    x = s["x"].to_numpy(float)
    p = s["P_fire"].to_numpy(float)
    lo = s["ci95_low"].to_numpy(float)
    hi = s["ci95_high"].to_numpy(float)

    ax.plot(x, p, "o-", lw=1.5, ms=4)
    ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)


def _plot_heatmap(ax,
                  summary: pd.DataFrame,
                  xlabel: str,
                  ylabel: str,
                  title: str,
                  value_col: str = "P_fire"):
    if summary.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return None

    piv = (
        summary.pivot(index="y", columns="x", values=value_col)
               .sort_index()
               .sort_index(axis=1)
    )
    xs = piv.columns.to_numpy(float)
    ys = piv.index.to_numpy(float)
    z = piv.to_numpy(float)

    dx = 1.0 if len(xs) == 1 else float(np.median(np.diff(xs)))
    dy = 1.0 if len(ys) == 1 else float(np.median(np.diff(ys)))
    extent = [xs.min() - dx / 2, xs.max() + dx / 2,
              ys.min() - dy / 2, ys.max() + dy / 2]

    im = ax.imshow(
        z, origin="lower", aspect="auto", extent=extent,
        vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    return im


def render_pdf(output_pdf: Path,
               summaries_1d: dict[str, pd.DataFrame],
               summaries_2d: dict[str, pd.DataFrame],
               meta_text: str,
               outcome_label: str,
               dv_page: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_pdf) as pdf:
        # Page 1: metadata / methods
        fig, ax = plt.subplots(figsize=(8.3, 5.8))
        ax.axis("off")
        ax.text(0.02, 0.96, f"Neuron-level {outcome_label} statistics",
                fontsize=16, weight="bold", va="top")
        ax.text(0.02, 0.86, meta_text, fontsize=10, va="top", family="monospace")
        ax.text(
            0.02, 0.26,
            "Statistical unit: neuron-level observation.\n"
            f"For each 1D/2D bin: P({outcome_label}) = positive neurons / total neuron observations.\n"
            "Shaded bands: 95% Wilson binomial confidence intervals.\n"
            f"Heatmaps: pooled neuron-level P({outcome_label}) in each 2D bin.\n"
            "Culture IDs are retained only for traceability/QC and are not averaged first.",
            fontsize=9, va="top"
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: 1D effects
        keys = [
            ("distance", DISTANCE_LABEL, "Dipole-centre distance"),
            ("theta_pos", "Position angle theta_pos (deg)", "Position angle"),
            ("theta_orient", "Neuron orientation theta_orient (deg)", "Neuron orientation"),
            ("layer", "Layer thickness (um)", "Layer / complexity"),
            ("x", "x position (um)", "x position"),
            ("y", "y position (um)", "y position"),
        ]
        fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.5))
        for ax, (k, xl, ttl) in zip(axes.flat, keys):
            _plot_1d(ax, summaries_1d[k], xl, f"P({outcome_label}) vs {ttl}", ylabel=f"P({outcome_label})")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: requested pairwise effects
        pairs = [
            ("distance_theta", DISTANCE_LABEL, "Position angle theta_pos (deg)",
             f"P({outcome_label}): distance x position angle"),
            ("layer_theta", "Layer thickness (um)", "Position angle theta_pos (deg)",
             f"P({outcome_label}): layer x position angle"),
            ("distance_layer", DISTANCE_LABEL, "Layer thickness (um)",
             f"P({outcome_label}): distance x layer"),
            ("thetaPos_thetaOrient", "Position angle theta_pos (deg)", "Orientation theta_orient (deg)",
             f"P({outcome_label}): position angle x neuron orientation"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.5))
        for ax, (k, xl, yl, ttl) in zip(axes.flat, pairs):
            im = _plot_heatmap(ax, summaries_2d[k], xl, yl, ttl)
            if im is not None:
                cbar = fig.colorbar(
                    im, ax=ax, fraction=0.046, pad=0.04
                )
                cbar.set_label(f"P({outcome_label})")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: spatial + distance/orientation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        im0 = _plot_heatmap(
            axes[0], summaries_2d["xy"], "x (um)", "y (um)",
            f"Spatial P({outcome_label}): x x y"
        )
        im1 = _plot_heatmap(
            axes[1], summaries_2d["distance_orientation"],
            DISTANCE_LABEL, "Orientation theta_orient (deg)",
            f"P({outcome_label}): distance x neuron orientation"
        )
        if im0 is not None:
            cbar0 = fig.colorbar(
                im0, ax=axes[0], fraction=0.046, pad=0.04
            )
            cbar0.set_label(f"P({outcome_label})")
        if im1 is not None:
            cbar1 = fig.colorbar(
                im1, ax=axes[1], fraction=0.046, pad=0.04
            )
            cbar1.set_label(f"P({outcome_label})")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: sample count heatmaps, essential for sparse bins/QC
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        n0 = _plot_heatmap(
            axes[0], summaries_2d["distance_theta"],
            DISTANCE_LABEL, "Position angle theta_pos (deg)",
            "Neuron observations per distance x angle bin", value_col="n_neurons"
        )
        n1 = _plot_heatmap(
            axes[1], summaries_2d["xy"], "x (um)", "y (um)",
            "Neuron observations per spatial bin", value_col="n_neurons"
        )
        # Counts do not share a fixed 0-1 range, so redraw these two without inherited scale.
        for ax, summary, xl, yl, ttl in [
            (axes[0], summaries_2d["distance_theta"], DISTANCE_LABEL,
             "Position angle theta_pos (deg)", "Neuron observations per distance x angle bin"),
            (axes[1], summaries_2d["xy"], "x (um)", "y (um)", "Neuron observations per spatial bin"),
        ]:
            ax.clear()
            if summary.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_axis_off()
                continue
            piv = summary.pivot(index="y", columns="x", values="n_neurons").sort_index().sort_index(axis=1)
            xs = piv.columns.to_numpy(float); ys = piv.index.to_numpy(float); z = piv.to_numpy(float)
            dx = 1.0 if len(xs) == 1 else float(np.median(np.diff(xs)))
            dy = 1.0 if len(ys) == 1 else float(np.median(np.diff(ys)))
            extent = [xs.min()-dx/2, xs.max()+dx/2, ys.min()-dy/2, ys.max()+dy/2]
            im = ax.imshow(z, origin="lower", aspect="auto", extent=extent, interpolation="nearest")
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=9)
            fig.colorbar(im, ax=ax, label="n neuron observations", shrink=0.8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6 (polarization outcomes): response size vs distance, and the threshold's effect
        if dv_page is not None:
            th = float(dv_page["theta"])
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
            ax = axes[0]
            a = dv_page["abs"]
            if not a.empty:
                x = a["x"].to_numpy(float)
                lo = lambda c: np.maximum(a[c].to_numpy(float), 1e-6)
                ax.fill_between(x, lo("q05"), lo("q95"), alpha=0.15, label="5-95 %")
                ax.fill_between(x, lo("q25"), lo("q75"), alpha=0.35, label="25-75 %")
                ax.plot(x, lo("q50"), "-", lw=1.6, label="median")
            if th > 0:
                ax.axhline(th, ls="--", color="k", lw=1.0, label=f"threshold {th:g} mV")
            ax.set_yscale("log")
            ax.set_xlabel(DISTANCE_LABEL)
            ax.set_ylabel("|DeltaV_end| (mV), non-firing somata")
            ax.set_title("Size of the somatic response vs distance", fontsize=9)
            ax.grid(alpha=0.2, which="both")
            ax.legend(fontsize=8)
            ax = axes[1]
            for key, lab in (("p_sign", "sign only (|DeltaV| > 1e-6 mV)"),
                             ("p_thr", f"|DeltaV| >= {th:g} mV")):
                sm = dv_page.get(key)
                if sm is None or sm.empty:
                    continue
                sm = sm.sort_values("x")
                ax.plot(sm["x"], sm["P_fire"], "o-", ms=3, lw=1.3, label=lab)
                ax.fill_between(sm["x"], sm["ci95_low"], sm["ci95_high"], alpha=0.2)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel(DISTANCE_LABEL)
            ax.set_ylabel(f"P({outcome_label})")
            ax.set_title(f"P({outcome_label}) vs distance: sign-only vs thresholded", fontsize=9)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================

def analyze_one_outcome(input_path: str | Path,
                        outcome_name: str,
                        output_dir: str | Path,
                        distance_bin_um: float = 15.0,
                        angle_bin_deg: float = 15.0,
                        orientation_bin_deg: float = 15.0,
                        xy_bin_um: float = 20.0,
                        dedupe_identical: bool = False,
                        dv_threshold_mV: float = DEFAULT_DV_THRESHOLD_MV) -> dict[str, Path]:
    """Neuron-level statistics for one of activation/depolarization/hyperpolarization."""
    if outcome_name not in OUTCOMES:
        raise ValueError(f"Unknown outcome {outcome_name!r}; choose from {list(OUTCOMES)}")
    pattern, outcome_col = OUTCOMES[outcome_name]
    files = discover_csvs(input_path, pattern)
    outdir = Path(output_dir).expanduser().resolve() / outcome_name
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_and_merge(files, outcome_col=outcome_col, dedupe_identical=dedupe_identical)

    n_dedup_count = int(df.attrs.get("n_dedup", 0))     # captured before any column edits
    theta = float(dv_threshold_mV or 0.0)
    dv_page = None
    thr_text = "n/a (spike outcome)"
    if outcome_name in POLARIZATION_OUTCOMES:
        have = {DV_COL, "phase2_outcome"}.issubset(df.columns)
        if theta > 0 and not have:
            raise ValueError(
                f"--dv-threshold {theta:g} needs the {DV_COL} and phase2_outcome columns (soma-only "
                f"CSVs); these files lack them -- use --dv-threshold 0 for sign-only labels")
        thr_text = "sign only (|DeltaV_end| > 1e-6 mV, as written in the CSV)"
        if have:
            df["_activated"] = df["phase2_outcome"].astype(str).eq("activation").to_numpy()
            sign_lab = df["fired"].to_numpy()
            if theta > 0:
                new = polarization_labels(outcome_name, _safe_numeric(df[DV_COL]),
                                          df["_activated"], theta)
                gained = int(((new == 1) & (sign_lab == 0)).sum())
                if gained:
                    raise ValueError(f"{gained} rows would GAIN the {outcome_name} label under the "
                                     f"threshold: {DV_COL} and the CSV labels disagree")
                df["fired"] = new
                thr_text = f"|DeltaV_end| >= {theta:g} mV (non-firing somata)"
                print(f"  --dv-threshold {theta:g} mV: {int(new.sum()):,} of {int(sign_lab.sum()):,} "
                      f"sign-labelled {outcome_name} rows kept")
            dv_page = {"theta": theta, "sign_labels": sign_lab}

    dist_edges = make_edges(df[DISTANCE_COL], width=distance_bin_um)
    pos_angle_edges = make_edges(df["theta_pos_deg"], width=angle_bin_deg,
                                 explicit_range=(0.0, 180.0))
    orient_edges = make_edges(df["theta_orient_deg"], width=orientation_bin_deg,
                              explicit_range=(0.0, 90.0))
    x_edges = make_edges(df["x_um"], width=xy_bin_um)
    y_edges = make_edges(df["y_um"], width=xy_bin_um)

    s1 = {
        "distance": summarize_1d(df, DISTANCE_COL, edges=dist_edges),
        "theta_pos": summarize_1d(df, "theta_pos_deg", edges=pos_angle_edges),
        "theta_orient": summarize_1d(df, "theta_orient_deg", edges=orient_edges),
        "layer": summarize_1d(df, "layer_um", categorical=True),
        "x": summarize_1d(df, "x_um", edges=x_edges),
        "y": summarize_1d(df, "y_um", edges=y_edges),
    }
    s2 = {
        "distance_theta": summarize_2d(df, DISTANCE_COL, "theta_pos_deg",
                                       x_edges=dist_edges, y_edges=pos_angle_edges),
        "layer_theta": summarize_2d(df, "layer_um", "theta_pos_deg",
                                    x_categorical=True, y_edges=pos_angle_edges),
        "distance_layer": summarize_2d(df, DISTANCE_COL, "layer_um",
                                       x_edges=dist_edges, y_categorical=True),
        "thetaPos_thetaOrient": summarize_2d(df, "theta_pos_deg", "theta_orient_deg",
                                             x_edges=pos_angle_edges, y_edges=orient_edges),
        "xy": summarize_2d(df, "x_um", "y_um", x_edges=x_edges, y_edges=y_edges),
        "distance_orientation": summarize_2d(df, DISTANCE_COL, "theta_orient_deg",
                                             x_edges=dist_edges, y_edges=orient_edges),
    }

    if dv_page is not None:
        dv_page["abs"] = summarize_abs_dv(df, dist_edges)
        dv_page["p_thr"] = s1["distance"]
        alt = df[[DISTANCE_COL, "culture_global", "neuron_global"]].copy()
        alt["fired"] = dv_page.pop("sign_labels")
        dv_page["p_sign"] = summarize_1d(alt, DISTANCE_COL, edges=dist_edges)
        dv_page["abs"].drop(columns=["bin"]).to_csv(outdir / "summary_1d_abs_dv_distance.csv",
                                                    index=False)
        df = df.drop(columns=["_activated"])

    merged_path = outdir / f"merged_{outcome_name}.csv"
    df.to_csv(merged_path, index=False)
    for name, tab in s1.items():
        tab.to_csv(outdir / f"summary_1d_{name}.csv", index=False)
    for name, tab in s2.items():
        tab.to_csv(outdir / f"summary_2d_{name}.csv", index=False)

    morph_rows = []
    for morph, dm in df.groupby("morphology", observed=True):
        sm = summarize_1d(dm, DISTANCE_COL, edges=dist_edges)
        if not sm.empty:
            sm.insert(0, "morphology", morph)
            morph_rows.append(sm)
    morph_path = outdir / "summary_1d_distance_by_morphology.csv"
    (pd.concat(morph_rows, ignore_index=True) if morph_rows else pd.DataFrame()).to_csv(morph_path, index=False)

    n_total = int(len(df)); n_pos = int(df["fired"].sum())
    p_total = n_pos / n_total if n_total else float("nan")
    lo, hi = wilson_interval(np.array([n_pos]), np.array([n_total]))
    pdf_path = outdir / f"statistics_{outcome_name}.pdf"
    meta = (
        f"outcome               : {outcome_name}\n"
        f"cell model            : {df['cell_model'].iloc[0]}\n"
        f"distance metric       : {DISTANCE_COL} (dipole-centre 3D)\n"
        f"polarization label    : {thr_text}\n"
        f"files merged          : {len(files)}\n"
        f"duplicates dropped    : {n_dedup_count:,} (--dedupe-identical)\n"
        f"neuron observations   : {n_total:,}\n"
        f"unique soma IDs       : {df['neuron_global'].nunique():,}\n"
        f"source cultures       : {df['culture_global'].nunique():,}\n"
        f"morphologies          : {df['morphology'].nunique()}\n"
        f"layers                : {sorted(pd.unique(df['layer_um'].dropna()).tolist())}\n"
        f"overall positives     : {n_pos:,}/{n_total:,} = {p_total:.4f}\n"
        f"overall Wilson 95% CI : [{lo[0]:.4f}, {hi[0]:.4f}]"
    )
    render_pdf(pdf_path, s1, s2, meta, outcome_label=outcome_name, dv_page=dv_page)
    print(f"[{outcome_name}] N={n_total:,} positives={n_pos:,} P={p_total:.4f} -> {outdir}")
    return {"merged": merged_path, "pdf": pdf_path, "output_dir": outdir}


def analyze_cultures(input_path: str | Path,
                     output_dir: str | Path = "culture_stats",
                     outcome: str = "all",
                     distance_bin_um: float = 15.0,
                     angle_bin_deg: float = 15.0,
                     orientation_bin_deg: float = 15.0,
                     xy_bin_um: float = 20.0,
                     dedupe_identical: bool = False,
                     dv_threshold_mV: float = DEFAULT_DV_THRESHOLD_MV):
    """Analyze one outcome or all three. Main statistical unit = neuron observation."""
    names = list(OUTCOMES) if outcome == "all" else [outcome]
    return {
        name: analyze_one_outcome(
            input_path, name, output_dir,
            distance_bin_um=distance_bin_um,
            angle_bin_deg=angle_bin_deg,
            orientation_bin_deg=orientation_bin_deg,
            xy_bin_um=xy_bin_um,
            dedupe_identical=dedupe_identical,
            dv_threshold_mV=dv_threshold_mV,
        )
        for name in names
    }


# =============================================================================
# CLI
# =============================================================================

def _cli():
    ap = argparse.ArgumentParser(
        description="Merge HPC CSVs and compute neuron-level binomial statistics for activation/depolarization/hyperpolarization."
    )
    ap.add_argument("--input", required=True, help="One results directory or one compatible CSV")
    ap.add_argument("--output", default="culture_stats", help="Output root directory")
    ap.add_argument("--outcome", choices=["all", "activation", "depolarization", "hyperpolarization"],
                    default="all")
    ap.add_argument("--distance-bin", type=float, default=15.0)
    ap.add_argument("--angle-bin", type=float, default=15.0)
    ap.add_argument("--orientation-bin", type=float, default=15.0)
    ap.add_argument("--xy-bin", type=float, default=20.0)
    ap.add_argument("--dv-threshold", type=float, default=DEFAULT_DV_THRESHOLD_MV,
                    help="mV; a non-firing soma counts as depolarized (hyperpolarized) only if "
                         "DeltaV_end >= +T (<= -T). 0 = sign-only labels. Default %(default)s")
    ap.add_argument("--dedupe-identical", action="store_true",
                    help="keep ONE copy of observations present in several files with the same "
                         "seed/culture/neuron/layer AND identical placement, morphology and outcome "
                         "(two batches run with the same seeds); conflicting duplicates always fail")
    args = ap.parse_args()
    analyze_cultures(
        args.input, output_dir=args.output, outcome=args.outcome,
        distance_bin_um=args.distance_bin,
        angle_bin_deg=args.angle_bin,
        orientation_bin_deg=args.orientation_bin,
        xy_bin_um=args.xy_bin,
        dedupe_identical=args.dedupe_identical,
        dv_threshold_mV=args.dv_threshold,
    )


if __name__ == "__main__":
    _cli()
