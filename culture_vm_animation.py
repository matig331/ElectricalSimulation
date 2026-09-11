#!/usr/bin/env python3
"""
culture_vm_animation.py

Improved stimulation movie:
- TOP LEFT: one well/culture with ALL neurons shown as circles only.
    * activated neurons: red gradient
    * depolarized neurons: green gradient
    * hyperpolarized neurons: yellow gradient
    * no-response neurons: light grey
- TOP RIGHT: extracellular-potential field map Ve(x,y,t), updated during the
  biphasic pulse, with the same 6 electrodes.
- MIDDLE: true NEURON Vm(t)-Vrest traces for 3 representative neurons
  (one activated, one depolarized, one hyperpolarized), using the same
  red/green/yellow colors.
- BOTTOM: applied biphasic current I(t), synchronized with both panels.

IMPORTANT SCIENTIFIC NOTE
-------------------------
The CSVs contain the FINAL outcome of every neuron, not Vm(t) for every neuron.
Therefore:
  * the 3 representative Vm traces are TRUE re-simulated NEURON traces;
  * the field map is the TRUE time-varying extracellular potential;
  * the color intensity of ALL neurons is an outcome visualization:
      - depol/hyper intensity is scaled by |DeltaVm_end_phase2_mV|;
      - activation intensity is binary/endpoint-based because the CSV only says
        fired / not fired.
    Its time evolution is synchronized to the stimulus envelope; it is NOT a
    reconstructed Vm(t) for every neuron.

For an exact time-resolved Vm movie of every neuron, culture_export must save
Vm(t) for every neuron (very large output), or every displayed neuron must be
re-simulated.

Run from inside estim_pipeline:
    python culture_vm_animation.py --input . --output culture_vm_animation_v2.mp4

GIF fallback:
    python culture_vm_animation.py --input . --output culture_vm_animation_v2.gif
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

from config import CFG
import field as F
import culture_export as CE
from culture_statistics import DEFAULT_DV_THRESHOLD_MV, polarization_labels
from morphologies import find_one_morphology
from slicer import reduced_asc
from rich_cell import build_rich_cell, settled_resting_voltage
from rich_footprint import spikes_at


# ---------------------------------------------------------------------
# Colors requested by user
# ---------------------------------------------------------------------
ACT_COLOR = "#d62728"   # red
DEP_COLOR = "#2ca02c"   # green
HYP_COLOR = "#f1c40f"   # yellow
NEUTRAL_COLOR = "#d9d9d9"
ANODE_COLOR = "#d62728"  # red electrode
CATHODE_COLOR = "#1f77b4"  # blue electrode


# ---------------------------------------------------------------------
# CSV loading / merging
# ---------------------------------------------------------------------
def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' found under {root.resolve()}")
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} files match {pattern} under {root}: point --input at ONE "
                         f"merged folder (e.g. merged_soma_only/), not the whole results tree. "
                         f"First matches: {[str(m) for m in matches[:3]]}")
    return matches[0]


def load_outcomes(root: Path):
    fa = _find_one(root, "culture_Pactivation*.csv")
    fd = _find_one(root, "culture_Pdepolarization*.csv")
    fh = _find_one(root, "culture_Phyperpolarization*.csv")

    a = pd.read_csv(fa)
    d = pd.read_csv(fd)
    h = pd.read_csv(fh)

    base_required = {
        "culture", "neuron", "morphology", "layer_um",
        "x_um", "y_um", "theta_orient_deg"
    }
    for name, df in [("activation", a), ("depolarization", d), ("hyperpolarization", h)]:
        miss = base_required - set(df.columns)
        if miss:
            raise ValueError(f"{name} CSV missing columns: {sorted(miss)}")

    if "fired" not in a.columns:
        raise ValueError("Activation CSV must contain 'fired'.")
    if "depolarized" not in d.columns:
        raise ValueError("Depolarization CSV must contain 'depolarized'.")
    if "hyperpolarized" not in h.columns:
        raise ValueError("Hyperpolarization CSV must contain 'hyperpolarized'.")

    if "cell_model" in a.columns and set(a["cell_model"].astype(str)) - {"soma_only"}:
        raise ValueError("this movie re-simulates the soma_only model; the CSV holds "
                         f"{sorted(set(a['cell_model'].astype(str)))}")
    # One row per (culture, neuron, LAYER): every neuron is simulated at every layer, so joining
    # on (culture, neuron) alone would pair each layer with every other layer (3 x 3 rows).
    keys = ["culture", "neuron", "layer_um"] + (
        ["seed"] if all("seed" in df.columns for df in (a, d, h)) else [])
    # Keep geometry + rest from activation file.
    keep = [
        "culture", "neuron", "morphology", "layer_um",
        "x_um", "y_um", "theta_orient_deg", "fired"
    ] + (["seed"] if "seed" in keys else [])
    for extra in ["theta_abs_deg", "v_rest_mV", "soma_only_rest_mV", "deltaVm_end_phase2_mV"]:
        if extra in a.columns:
            keep.append(extra)

    out = a[keep].copy()

    # Merge class flags.
    out = out.merge(
        d[keys + ["depolarized"] + (
            ["deltaVm_end_phase2_mV"] if "deltaVm_end_phase2_mV" in d.columns else []
        )].rename(columns={"deltaVm_end_phase2_mV": "deltaVm_dep_mV"}),
        on=keys,
        how="inner"
    )
    out = out.merge(
        h[keys + ["hyperpolarized"] + (
            ["deltaVm_end_phase2_mV"] if "deltaVm_end_phase2_mV" in h.columns else []
        )].rename(columns={"deltaVm_end_phase2_mV": "deltaVm_hyp_mV"}),
        on=keys,
        how="inner"
    )

    # Consolidated final DeltaVm.
    out["deltaVm_end_mV"] = np.nan

    if "deltaVm_end_phase2_mV" in out.columns:
        # Activation CSV may also contain it; use it only as fallback.
        vals = pd.to_numeric(out["deltaVm_end_phase2_mV"], errors="coerce")
        out.loc[vals.notna(), "deltaVm_end_mV"] = vals[vals.notna()]

    if "deltaVm_dep_mV" in out.columns:
        vals = pd.to_numeric(out["deltaVm_dep_mV"], errors="coerce")
        m = (out["depolarized"] > 0) & vals.notna()
        out.loc[m, "deltaVm_end_mV"] = vals[m]

    if "deltaVm_hyp_mV" in out.columns:
        vals = pd.to_numeric(out["deltaVm_hyp_mV"], errors="coerce")
        m = (out["hyperpolarized"] > 0) & vals.notna()
        out.loc[m, "deltaVm_end_mV"] = vals[m]

    print("[input]")
    print(" activation       :", fa)
    print(" depolarization   :", fd)
    print(" hyperpolarization:", fh)
    return out


def choose_culture_layer(df, culture=None, layer=None):
    work = df.copy()
    if culture is not None:
        work = work[work["culture"] == int(culture)]
    if layer is not None:
        work = work[np.isclose(work["layer_um"].astype(float), float(layer))]

    candidates = []
    for (c, L), g in work.groupby(["culture", "layer_um"]):
        n_act = int((g["fired"] > 0).sum())
        n_dep = int(((g["fired"] == 0) & (g["depolarized"] > 0)).sum())
        n_hyp = int(((g["fired"] == 0) & (g["hyperpolarized"] > 0)).sum())
        if n_act and n_dep and n_hyp:
            candidates.append((len(g), int(c), float(L)))

    if not candidates:
        raise ValueError("No culture/layer contains activation + depolarization + hyperpolarization.")

    candidates.sort(reverse=True)
    _, c, L = candidates[0]
    g = df[(df["culture"] == c) & np.isclose(df["layer_um"].astype(float), L)].copy()
    return c, L, g


def choose_representatives(g):
    ga = g[g["fired"] > 0].copy()
    gd = g[(g["fired"] == 0) & (g["depolarized"] > 0)].copy()
    gh = g[(g["fired"] == 0) & (g["hyperpolarized"] > 0)].copy()

    if ga.empty or gd.empty or gh.empty:
        raise ValueError("Selected culture/layer does not contain all 3 representative outcomes.")

    # activated: pick one close to median position to avoid a visually extreme example
    ra = ga.iloc[len(ga) // 2]

    # depolarized: strongest positive final DeltaVm if available
    if gd["deltaVm_end_mV"].notna().any():
        rd = gd.loc[gd["deltaVm_end_mV"].idxmax()]
    else:
        rd = gd.iloc[0]

    # hyperpolarized: strongest negative final DeltaVm if available
    if gh["deltaVm_end_mV"].notna().any():
        rh = gh.loc[gh["deltaVm_end_mV"].idxmin()]
    else:
        rh = gh.iloc[0]

    return {
        "activation": ra,
        "depolarization": rd,
        "hyperpolarization": rh,
    }


# ---------------------------------------------------------------------
# Re-simulate only 3 true Vm traces
# ---------------------------------------------------------------------
def _morph_str(v):
    """Specimen ids are numeric strings ('60308'); the outcome join can turn the column into
    floats (60308.0). Normalise back so the identity check compares like with like."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(v)
    except (TypeError, ValueError):
        return str(v)


def absolute_theta(row):
    """(theta, x, y): the ABSOLUTE rotation and the EXACT soma position the export simulated.

    The CSV rounds x_um/y_um to 0.01 um; replaying from the rounded values shifts the soma by up
    to 0.005 um, enough to move a mid-spike Vm by ~1e-3 mV. The regenerated draw gives the exact
    position, so the replay is identical to the simulation, not just close.

    The CSV stores theta_orient_deg = orientation RELATIVE to the dipole axis, folded to
    [0, 90] -- a summary, not the rotation: replaying with it rotates the cell differently from
    the simulation (the replayed trace can then even have the opposite sign to the row's label).
    Placements are deterministic in (seed, culture), so the exact draw is regenerated with the
    same helpers the worker uses, and accepted only if it reproduces THIS row (x, y, morphology,
    folded angle); otherwise the config differs from the run and we refuse."""
    if "theta_abs_deg" in row.index and pd.notna(row["theta_abs_deg"]):
        return float(row["theta_abs_deg"]), float(row["x_um"]), float(row["y_um"])
    seed = int(row["seed"]) if "seed" in row.index and pd.notna(row["seed"]) else int(CFG.seed)
    c = (int(row["local_culture"]) if "local_culture" in row.index and pd.notna(row["local_culture"])
         else int(row["culture"]))
    i = int(row["neuron"])
    N = int(CFG.n_neurons_effective())
    elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
    dip_c, dip_d = CE.dipole_frame(elec, sign)
    d = CE.culture_draws(seed, c, N, len(CFG.morphologies), CFG.span_half_um(), elec,
                         CE.electrode_center(elec), dip_c, dip_d,
                         CE.dipole_axis_deg(elec, sign), CFG.h_soma_um)
    ok = (i < N
          and round(float(d["pos"][i, 0]), 2) == float(row["x_um"])
          and round(float(d["pos"][i, 1]), 2) == float(row["y_um"])
          and str(CFG.morphologies[int(d["midx"][i])]) == _morph_str(row["morphology"])
          and abs(round(float(d["th_or"][i]), 1) - float(row["theta_orient_deg"])) < 0.051)
    if not ok:
        raise ValueError(
            f"cannot recover the simulated orientation of seed={seed} culture={c} neuron={i}: "
            f"the draw regenerated with the CURRENT config.py does not match the CSV row. Run the "
            f"animation with the same config the simulations used (morphologies, neuron count "
            f"{N}, span {CFG.span_half_um()} um, electrodes).")
    return float(d["theta"][i]), float(d["pos"][i, 0]), float(d["pos"][i, 1])


def simulate_selected(row, post_ms=6.0):
    morph = str(row["morphology"])
    layer = float(row["layer_um"])
    x = float(row["x_um"])
    y = float(row["y_um"])

    theta, x, y = absolute_theta(row)          # exact rotation + position simulated (verified)

    # Same cell, rest and reference as culture_export: the sham (I = 0) trace from the same
    # initial state is subtracted, so the plotted DeltaVm is the stimulus-driven response and
    # its value at the end of phase 2 is exactly the CSV's deltaVm_end_phase2_mV.
    prep = CE.prepare_cell(morph, layer, CFG, "soma_only", tag="_anim_")
    vrest = prep["v_rest"]
    for col in ("v_rest_mV", "soma_only_rest_mV"):
        if col in row.index and pd.notna(row[col]) and abs(float(row[col]) - vrest) > 1e-3:
            print(f"[warning] CSV {col}={float(row[col]):.4f} differs from this build's rest "
                  f"{vrest:.4f} (different code or config?)")
    baseline_ms = 5.0
    phase_ms = float(CFG.phase_dur_ms)
    pre_end_ms = baseline_ms + 2.0 * phase_ms
    kw = dict(phase_dur_ms=phase_ms, baseline_ms=baseline_ms, post_ms=float(post_ms),
              dt_ms=float(CFG.dt_ms), sigma_Sm=float(CFG.sigma_Sm),
              rmin_um=float(CFG.electrode_um) / 2.0, detail=True, pre_end_ms=pre_end_ms,
              v_init_mV=vrest)
    nsp, vmax, tw, vw = spikes_at(prep["cell"], (x, y), theta, i0_uA=float(CFG.i0_uA), **kw)
    _n0, _v0, tw0, vw0 = spikes_at(prep["cell"], (x, y), theta, i0_uA=0.0, **kw)
    del prep                                   # one live cell at a time
    tw = np.asarray(tw, float)
    vw = np.asarray(vw, float)
    # The replay must BE the simulated neuron: same spike outcome, same DeltaV at the end of
    # phase 2 (t = 0 here) as the CSV row. Anything else means the movie would show a
    # different cell than the one the statistics count.
    k0 = int(np.argmin(np.abs(tw)))
    dv_end = float(vw[k0] - np.asarray(vw0, float)[k0])
    if "fired" in row.index and pd.notna(row["fired"]) and int(nsp > 0) != int(row["fired"]):
        raise ValueError(f"replay fired={int(nsp > 0)} but the CSV row has fired={int(row['fired'])}")
    if ("deltaVm_end_phase2_mV" in row.index and pd.notna(row["deltaVm_end_phase2_mV"])
            and abs(dv_end - float(row["deltaVm_end_phase2_mV"])) > 1e-5):
        raise ValueError(f"replay DeltaV_end={dv_end:+.6f} mV but the CSV row has "
                         f"{float(row['deltaVm_end_phase2_mV']):+.6f} mV")
    print(f"  [check] replay == simulation: fired={int(nsp > 0)}, DeltaV_end={dv_end:+.6f} mV "
          f"(theta_abs={theta:.1f} deg)")
    return {
        "row": row,
        "t": tw,
        "vm": vw,
        "dv": vw - np.asarray(vw0, float),
        "vrest": vrest,
        "nsp": int(nsp),
        "vmax": float(vmax),
        "x": x,
        "y": y,
    }


# ---------------------------------------------------------------------
# True biphasic current and extracellular field
# ---------------------------------------------------------------------
def current_relative(t_rel_ms):
    """t=0 is END of phase 2."""
    phase = float(CFG.phase_dur_ms)
    baseline = 5.0
    end = baseline + 2.0 * phase
    tabs = np.asarray(t_rel_ms) + end

    I_A = F.biphasic_current(
        tabs - baseline,
        i0_uA=float(CFG.i0_uA),
        phase_dur_ms=phase,
        anodic_first=True,
        ramp_us=float(getattr(CFG, "ramp_us", 100.0)),
        interphase_us=float(getattr(CFG, "interphase_us", 0.0)),
    )
    return np.asarray(I_A) * 1e6  # uA


def electrode_geometry():
    elec, sign = F.default_array(
        pitch_um=CFG.pitch_um,
        monopolar=not CFG.bipolar,
    )
    return np.asarray(elec, float), np.asarray(sign, float)


def field_map_unit_geometry(xgrid, ygrid, elec, sign):
    """
    Geometry-only Ve per unit electrode current.

    Uses the same half-space point-source law:
        Ve = sum s_e I / (2*pi*sigma*r)
    with z=0 display plane and rmin clamp.

    Returns V per ampere on the xy grid.
    """
    sigma = float(CFG.sigma_Sm)
    rmin_m = (float(CFG.electrode_um) / 2.0) * 1e-6

    X, Y = np.meshgrid(xgrid, ygrid)
    G = np.zeros_like(X, dtype=float)

    for (ex, ey), s in zip(elec, sign):
        dx = (X - ex) * 1e-6
        dy = (Y - ey) * 1e-6
        r = np.sqrt(dx * dx + dy * dy)
        r = np.maximum(r, rmin_m)
        G += s / (2.0 * np.pi * sigma * r)

    return X, Y, G


# ---------------------------------------------------------------------
# Outcome intensity for ALL neurons
# ---------------------------------------------------------------------
def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def temporal_response_gain(t_ms):
    """
    Visual timing only for the ALL-neuron classification map.

    Starts near zero before phase 1, rises through phase 1/2, reaches maximum
    at t=0 (end phase 2), then decays slowly after the pulse.

    This is NOT each neuron's reconstructed Vm(t).
    """
    phase = float(CFG.phase_dur_ms)
    start = -2.0 * phase

    if t_ms < start:
        return 0.0
    if t_ms <= 0.0:
        return smoothstep01((t_ms - start) / (0.0 - start))
    return float(np.exp(-t_ms / 2.0))


def endpoint_strength(g):
    """
    Returns endpoint strengths in [0,1] for display.

    Activation: fired => 1.
    Depol/hyper: scaled by |DeltaVm_end| using robust 95th percentile.
    """
    strength = np.zeros(len(g), dtype=float)

    fired = (g["fired"].to_numpy() > 0)
    dep = ((g["fired"].to_numpy() == 0) & (g["depolarized"].to_numpy() > 0))
    hyp = ((g["fired"].to_numpy() == 0) & (g["hyperpolarized"].to_numpy() > 0))

    strength[fired] = 1.0

    dv = pd.to_numeric(g["deltaVm_end_mV"], errors="coerce").to_numpy(float)
    finite = np.isfinite(dv)
    if finite.any():
        scale = np.nanpercentile(np.abs(dv[finite]), 95)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        strength[dep | hyp] = np.clip(np.abs(dv[dep | hyp]) / scale, 0.10, 1.0)
    else:
        strength[dep | hyp] = 0.75

    return fired, dep, hyp, strength


def blend_with_white(hex_color, intensity):
    """RGBA colors: 0 -> almost white, 1 -> requested base color."""
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(hex_color))
    intensity = np.asarray(intensity, float).reshape(-1, 1)
    white = np.ones((len(intensity), 3))
    out = white * (1.0 - intensity) + rgb.reshape(1, 3) * intensity
    alpha = np.ones((len(intensity), 1))
    return np.hstack([out, alpha])


# ---------------------------------------------------------------------
# Movie
# ---------------------------------------------------------------------
def make_movie(context, sims, output, fps=30, duration_s=16.0):
    labels = ["activation", "depolarization", "hyperpolarization"]
    trace_color = {
        "activation": ACT_COLOR,
        "depolarization": DEP_COLOR,
        "hyperpolarization": HYP_COLOR,
    }

    t = sims["activation"]["t"]
    for lab in labels[1:]:
        if len(sims[lab]["t"]) != len(t) or not np.allclose(sims[lab]["t"], t):
            raise ValueError("The three re-simulated traces do not share the same time axis.")

    I = current_relative(t)
    elec, sign = electrode_geometry()

    # Culture data
    x = context["x_um"].to_numpy(float)
    y = context["y_um"].to_numpy(float)
    fired, dep, hyp, strength = endpoint_strength(context)

    # Field grid
    xmin = min(x.min(), elec[:, 0].min()) - 35
    xmax = max(x.max(), elec[:, 0].max()) + 35
    ymin = min(y.min(), elec[:, 1].min()) - 35
    ymax = max(y.max(), elec[:, 1].max()) + 35

    gx = np.linspace(xmin, xmax, 180)
    gy = np.linspace(ymin, ymax, 180)
    X, Y, G = field_map_unit_geometry(gx, gy, elec, sign)

    Ipeak_A = max(abs(float(CFG.i0_uA)) * 1e-6, 1e-12)
    Ve_peak_mV = G * Ipeak_A * 1e3
    vmax_field = np.nanpercentile(np.abs(Ve_peak_mV), 98)
    if not np.isfinite(vmax_field) or vmax_field <= 0:
        vmax_field = 1.0

    # Much slower playback: same biological time stretched over duration_s.
    nframes = max(2, int(round(float(duration_s) * int(fps))))
    frame_idx = np.linspace(0, len(t) - 1, nframes).round().astype(int)

    fig = plt.figure(figsize=(13.5, 10.5))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[4.1, 2.6, 1.55],
        width_ratios=[1, 1],
        hspace=0.30,
        wspace=0.18
    )

    axwell = fig.add_subplot(gs[0, 0])
    axfield = fig.add_subplot(gs[0, 1])
    axvm = fig.add_subplot(gs[1, :])
    axI = fig.add_subplot(gs[2, :])

    # ---------------- TOP LEFT: all neurons ----------------
    base = axwell.scatter(
        x, y, s=34,
        facecolors=NEUTRAL_COLOR,
        edgecolors="none",
        alpha=0.75
    )

    # Overlay one artist per response class, still all circles.
    sc_act = axwell.scatter(
        x[fired], y[fired], s=42, marker="o",
        edgecolors="none"
    )
    sc_dep = axwell.scatter(
        x[dep], y[dep], s=42, marker="o",
        edgecolors="none"
    )
    sc_hyp = axwell.scatter(
        x[hyp], y[hyp], s=42, marker="o",
        edgecolors="none"
    )

    # Electrodes: explicitly red (+) and blue (-)
    for (ex, ey), s in zip(elec, sign):
        color = ANODE_COLOR if s > 0 else CATHODE_COLOR
        axwell.scatter(
            [ex], [ey], marker="s", s=210,
            facecolors=color, edgecolors="black",
            linewidths=1.0, zorder=10
        )
        axwell.text(
            ex, ey, "+" if s > 0 else "-",
            ha="center", va="center",
            color="white", fontweight="bold",
            fontsize=11, zorder=11
        )

    axwell.set_xlim(xmin, xmax)
    axwell.set_ylim(ymin, ymax)
    axwell.set_aspect("equal", adjustable="box")
    axwell.set_xlabel("x (um)")
    axwell.set_ylabel("y (um)")
    title_well = axwell.set_title("Neuronal response")

    # manual legend as colored circles
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ACT_COLOR,
               markeredgecolor="none", markersize=8, label="activation"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DEP_COLOR,
               markeredgecolor="none", markersize=8, label="depolarization"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HYP_COLOR,
               markeredgecolor="none", markersize=8, label="hyperpolarization"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=ANODE_COLOR,
               markeredgecolor="black", markersize=8, label="+ electrode"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CATHODE_COLOR,
               markeredgecolor="black", markersize=8, label="- electrode"),
    ]
    axwell.legend(handles=legend_items, loc="upper right", fontsize=8, frameon=True)

    # ---------------- TOP RIGHT: electric field / Ve ----------------
    # diverging map; midpoint fixed at 0
    field_cmap = "coolwarm"
    im = axfield.imshow(
        np.zeros_like(G),
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal",
        cmap=field_cmap,
        vmin=-vmax_field,
        vmax=vmax_field,
        interpolation="bilinear"
    )

    for (ex, ey), s in zip(elec, sign):
        color = ANODE_COLOR if s > 0 else CATHODE_COLOR
        axfield.scatter(
            [ex], [ey], marker="s", s=180,
            facecolors=color, edgecolors="black",
            linewidths=0.9, zorder=10
        )

    cbar = fig.colorbar(im, ax=axfield, fraction=0.046, pad=0.04)
    cbar.set_label("$V_e$ (mV)")
    axfield.set_xlim(xmin, xmax)
    axfield.set_ylim(ymin, ymax)
    axfield.set_xlabel("x (um)")
    axfield.set_ylabel("y (um)")
    title_field = axfield.set_title("Extracellular potential field")

    # ---------------- MIDDLE: 3 true Vm traces ----------------
    vm_progress = {}
    for lab in labels:
        s = sims[lab]
        axvm.plot(
            t, s["dv"],
            color=trace_color[lab],
            alpha=0.18,
            linewidth=1.2
        )
        lp, = axvm.plot(
            [], [],
            color=trace_color[lab],
            linewidth=2.5,
            label=lab
        )
        vm_progress[lab] = lp

    axvm.axhline(0, color="black", linewidth=0.8)
    cursor_vm = axvm.axvline(t[0], color="black", linewidth=1.0)

    phase = float(CFG.phase_dur_ms)
    for xx in (-2 * phase, -phase, 0.0):
        axvm.axvline(xx, color="0.55", linestyle="--", linewidth=0.8)

    axvm.set_xlim(t.min(), t.max())
    dv_all = np.concatenate([sims[k]["dv"] for k in labels])
    lo, hi = np.nanmin(dv_all), np.nanmax(dv_all)
    margin = max(2.0, 0.08 * max(1.0, hi - lo))
    axvm.set_ylim(lo - margin, hi + margin)
    axvm.set_ylabel(r"$\Delta V_m = V_m - V_m^{sham}$ (mV)")
    axvm.set_xlabel("Time relative to end of phase 2 (ms)")
    axvm.set_title("True somatic $V_m(t)$ -- three representative neurons")
    axvm.legend(loc="upper right")

    # ---------------- BOTTOM: current ----------------
    axI.plot(t, I, color="black", linewidth=2.0)
    cursor_I = axI.axvline(t[0], color="black", linewidth=1.0)
    axI.axhline(0, color="0.3", linewidth=0.8)
    for xx in (-2 * phase, -phase, 0.0):
        axI.axvline(xx, color="0.55", linestyle="--", linewidth=0.8)

    axI.set_xlim(t.min(), t.max())
    limI = max(1.0, np.max(np.abs(I)) * 1.25)
    axI.set_ylim(-limI, limI)
    axI.set_ylabel("Current (uA)")
    axI.set_xlabel("Time relative to end of phase 2 (ms)")
    axI.set_title("Applied biphasic current")

    fig.suptitle(
        "Electrical stimulation of the neuronal culture",
        fontsize=15
    )

    # class arrays for endpoint strengths
    s_act = strength[fired]
    s_dep = strength[dep]
    s_hyp = strength[hyp]

    def update(frame_number):
        j = int(frame_idx[frame_number])
        tj = float(t[j])
        gain = temporal_response_gain(tj)

        # all-neuron gradients
        if fired.any():
            sc_act.set_facecolors(
                blend_with_white(ACT_COLOR, np.clip(s_act * gain, 0.0, 1.0))
            )
        if dep.any():
            sc_dep.set_facecolors(
                blend_with_white(DEP_COLOR, np.clip(s_dep * gain, 0.0, 1.0))
            )
        if hyp.any():
            sc_hyp.set_facecolors(
                blend_with_white(HYP_COLOR, np.clip(s_hyp * gain, 0.0, 1.0))
            )

        # true electric field map
        I_A = float(I[j]) * 1e-6
        Ve_mV = G * I_A * 1e3
        im.set_data(Ve_mV)

        # true 3-neuron traces
        for lab in labels:
            vm_progress[lab].set_data(t[:j+1], sims[lab]["dv"][:j+1])

        cursor_vm.set_xdata([tj, tj])
        cursor_I.set_xdata([tj, tj])

        if tj < -phase:
            ph = "phase 1 (+I)"
        elif tj < 0:
            ph = "phase 2 (-I)"
        else:
            ph = "post-stimulation"

        title_well.set_text(
            f"Neuronal response -- {ph} | t={tj:+.3f} ms"
        )
        title_field.set_text(
            f"Extracellular potential $V_e(x,y,t)$ -- {ph}"
        )

        return (
            sc_act, sc_dep, sc_hyp, im,
            *vm_progress.values(),
            cursor_vm, cursor_I,
            title_well, title_field
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=1000.0 / fps,
        blit=False
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=3500)

    anim.save(output, writer=writer)
    plt.close(fig)
    print("[done]", output.resolve())


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=".")
    p.add_argument("--output", default="culture_vm_animation_v2.mp4")
    p.add_argument("--culture", type=int, default=None)
    p.add_argument("--layer", type=float, default=None)
    p.add_argument("--post-ms", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--duration-s", type=float, default=16.0,
        help="Playback duration. Higher = slower movie. Biological timing is unchanged."
    )
    p.add_argument(
        "--dv-threshold", type=float, default=DEFAULT_DV_THRESHOLD_MV,
        help="mV: a non-firing soma is depolarized/hyperpolarized only if |DeltaV_end| >= this "
             "(same rule as culture_statistics); 0 = sign-only labels. Default %(default)s"
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = load_outcomes(Path(args.input))
    if args.dv_threshold and args.dv_threshold > 0:
        if "deltaVm_end_phase2_mV" not in df.columns:
            raise ValueError("--dv-threshold needs deltaVm_end_phase2_mV in the CSVs "
                             "(use --dv-threshold 0 for sign-only labels)")
        dv = pd.to_numeric(df["deltaVm_end_phase2_mV"], errors="coerce").to_numpy()
        act = df["fired"].to_numpy() > 0
        df["depolarized"] = polarization_labels("depolarization", dv, act, args.dv_threshold)
        df["hyperpolarized"] = polarization_labels("hyperpolarization", dv, act, args.dv_threshold)
        print(f"[labels] |DeltaV_end| >= {args.dv_threshold:g} mV: "
              f"{int(df['depolarized'].sum())} depolarized, {int(df['hyperpolarized'].sum())} "
              f"hyperpolarized, {int(act.sum())} activated")

    culture, layer, context = choose_culture_layer(
        df, culture=args.culture, layer=args.layer
    )

    reps = choose_representatives(context)

    print(f"[selection] culture={culture}, layer={layer:g} um")
    for lab, row in reps.items():
        print(
            f"  {lab:16s} neuron={int(row['neuron'])} "
            f"morph={row['morphology']} "
            f"xy=({row['x_um']:.1f},{row['y_um']:.1f})"
        )

    if "theta_abs_deg" not in context.columns:
        print("[note] the CSV has no theta_abs_deg: each replayed neuron's absolute rotation is "
              "regenerated from (seed, culture) and verified against its row (see absolute_theta).")

    sims = {
        lab: simulate_selected(row, post_ms=args.post_ms)
        for lab, row in reps.items()
    }

    print("[re-simulated representative traces]")
    for lab, s in sims.items():
        iz = int(np.argmin(np.abs(s["t"])))
        print(
            f"  {lab:16s}: nsp={s['nsp']} "
            f"Vmax={s['vmax']:.2f} mV "
            f"DeltaVm(t=0)={s['dv'][iz]:+.3f} mV"
        )

    make_movie(
        context,
        sims,
        output=args.output,
        fps=args.fps,
        duration_s=args.duration_s
    )


if __name__ == "__main__":
    main()
