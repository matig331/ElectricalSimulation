"""plot_vm_examples.py -- soma Vm timecourse + the two post-pulse fits, one page per neuron.

Each page shows ONE cell instance (morphology x layer x soma position x orientation):

  top          the soma Vm(t) over the whole post-pulse window, with the stimulus window
               marked and TWO points highlighted -- the first hyperpolarisation /
               depolarisation peak just after the pulse, and the slow Ih bump.
  bottom left  a zoom on the stimulus and the direct post-pulse polarisation, with its
               single-exponential decay.
  bottom right the Ih bump with its DOUBLE exponential, pinned to zero at the end of the pulse.

WHY THE PINNING (the point of the whole figure)
The bump does not start from the resting potential: at the instant the stimulus ends the
membrane is still displaced by the direct polarisation, so DeltaV(t0) != 0. A double
exponential fitted to the raw trace has to climb out of that displacement and absorbs it into
its amplitude and both taus. Measured here on one real placement, a pinned fit WITHOUT the
residual term returns tau_rise = 400 ms and tau_decay = 5000 ms -- both pinned to the search
bounds -- at r2 = 0.60, against 52 ms / 196 ms at r2 = 0.9996 once the displacement is
accounted for. So the membrane potential at t0 is DEFINED to be the zero of the bump, and the
displacement is carried by a separate residual term:

    DeltaV(t) = DeltaV(t0)
              + Ab     * ( exp(-u/tau_decay) - exp(-u/tau_rise) )        u = t - t0 >= 0
              + offset * ( 1 - exp(-u/tau_m) )

Every term after DeltaV(t0) is exactly 0 at u = 0, for every parameter value, so the pinning
is structural rather than a penalty. offset should come out equal to -DeltaV(t0), because the
trace does return to the sham; that is printed on the panel as a check, not assumed.

WHY TWO STAGES AND TWO TIME GRIDS
The two components are separated by about two orders of magnitude in time -- the direct
relaxation is ~0.2 ms (the soma discharges by redistributing charge into the arbour, it does
not charge the whole cell) while the bump peaks near 95 ms. So:

  * tau_m is measured on the FINE trace (solver resolution, 0.025 ms). At the 0.5 ms grid the
    decay has two samples and no fit exists.
  * the bump is fitted on the uniform bump_dt_ms grid over the long window, taking tau_m as a
    FIXED input. A single least-squares over one uniform grid covering both cannot work: 1600
    samples of bump against 4 of polarisation means the polarisation is unweighted.

`--fit joint` instead runs bump_kinetics.fit_post_pulse, which fits all five parameters at
once on the coarse grid. Use it when the two timescales are NOT well separated; it is the
more general model but it cannot see a 0.2 ms relaxation on a 0.5 ms grid.

ONE CAVEAT, STATED ON EVERY PAGE
The direct relaxation is not actually a single exponential -- it is a cable relaxation with a
spectrum of time constants, and the fitted tau_m grows with the window it is fitted over
(measured on one placement: 0.22 ms stopping at 10 % of the peak, 0.31 at 30 %, 0.47 at 50 %,
with r2 near 0.91 throughout). The bump is unaffected -- that one really is a double
exponential, at r2 > 0.999. So the panel prints tau_m WITH its window, and beside it t_1e, the
measured time for |DeltaV| to fall to 1/e of the peak, which is model-free and does not move
with the window. Compare t_1e across placements, not tau_m.

DeltaV(t) = Vm_stim(t) - Vm_sham(t) with t = 0 at the END of phase 2 -- the same
sham-referenced quantity the export classifies, so these pages are directly comparable with
culture_Pdepolarization / culture_Phyperpolarization / culture_Pkinetics.

Run:
    python plot_vm_examples.py                              # full_tuned, 6 instances
    python plot_vm_examples.py --cell-model soma_only       # expect NO bump: Ih is dendritic
    python plot_vm_examples.py --placements "60308:80:90:-30:0,130303:80:-150:40:90"
    python plot_vm_examples.py --fit joint --help

Outputs (next to this file): plot_vm_examples.pdf, one page per instance, and
plot_vm_examples.csv, one row per instance with every fitted parameter.

Test:  python smoke_test_plot_vm_examples.py
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from bump_kinetics import (DEXP_COLUMNS, EXPDECAY_COLUMNS, dexp_row_values,
                           eval_bump_component, eval_exp_decay, eval_offset_component,
                           expdecay_row_values, fit_double_exp_from_zero, fit_exp_decay,
                           fit_post_pulse)

# --- palette -----------------------------------------------------------------------------
# Okabe-Ito, in fixed order, validated for colour-vision deficiency against a light surface
# (worst adjacent pair dE 11.0 deutan / 24.2 normal). Every series is ALSO separated by line
# style and carries a legend entry, so identity is never colour alone. Text stays in ink.
C_DATA = "#0072B2"          # the measured trace
C_BUMP = "#009E73"          # the bump component (double exponential)
C_POL = "#E69F00"           # the polarisation component (single exponential)
C_SHAM = "#7f7f7f"          # sham trajectory (a reference, not a result)
C_INK = "#1a1a1a"
C_INK2 = "#5a5a5a"
C_GRID = "#d9d9d9"
C_PHASE1 = "#4472c4"        # same shading convention as single_neuron_check
C_PHASE2 = "#c00000"

PLOT_CELL_MODELS = ("soma_only", "full_active", "full_tuned")
FIT_MODES = ("staged", "joint")


# ===========================================================================
# 1. model construction -- builds one cell; knows nothing about the stimulus
# ===========================================================================
def build_instance(morph, layer, cell_model, cfg=None, rest_tstop_ms=1500.0, tag="_pve_"):
    """Build one (morphology x layer) cell with the biophysics of `cell_model`.

      soma_only    active soma, passive dendrites (the campaign so far). Ih is dendritic, so
                   this model has essentially NO bump -- useful as a negative control.
      full_active  the full Rich channel set everywhere, legacy v_init
      full_tuned   the full set with Rich's PASSIVE axon, leak tuned so the measured settled
                   rest is imposed isopotentially across the whole arbour
                   (rich_cell.tune_leak_isopotential)

    Returns dict(cell, morph, layer, cell_model, v_rest, leak_report). v_rest is the v_init
    every simulation of this instance uses.
    """
    from config import CFG
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    from rich_cell import build_rich_cell, settled_resting_voltage, tune_leak_isopotential
    cfg = CFG if cfg is None else cfg
    if cell_model not in PLOT_CELL_MODELS:
        raise ValueError("cell_model must be one of %s, got %r" % (PLOT_CELL_MODELS, cell_model))

    out = "%s%d_%s_%d.asc" % (tag, os.getpid(), morph, int(layer))
    asc = reduced_asc(find_one_morphology(morph), float(layer), out_path=out)
    try:
        cell = build_rich_cell(asc, soma_only=(cell_model == "soma_only"),
                               axon_active=(cell_model != "full_tuned"))
    finally:
        if asc and os.path.exists(asc):
            os.remove(asc)

    report = None
    if cell_model == "full_active":
        v_rest = float(cfg.v_rest_mV)               # legacy init, matches the existing dataset
    else:
        v_rest = float(settled_resting_voltage(cell, tstop_ms=rest_tstop_ms, dt_ms=cfg.dt_ms))
        if cell_model == "full_tuned":
            # impose that measured rest everywhere by countering the other channels with e_pas
            report = tune_leak_isopotential(cell, v_rest, celsius=37.0)
    return dict(cell=cell, morph=str(morph), layer=float(layer), cell_model=str(cell_model),
                v_rest=v_rest, leak_report=report)


# ===========================================================================
# 2. simulation -- the protocol, independent of which cell it is applied to
# ===========================================================================
def default_protocol(cfg=None, i0_uA=None, bump_ms=800.0, bump_dt_ms=0.5, cvode_atol=1e-6,
                     play_margin_ms=3.0):
    """The stimulation protocol as one dict, so swapping it touches nothing else.

    bump_ms         window integrated AFTER the end of phase 2. The bump peaks near 95 ms and
                    decays over hundreds, so the 11.5 ms window of the campaign never held it.
    bump_dt_ms      the FIXED output grid the bump fit runs on -- fixed rather than solver
                    steps, so the time-to-peak is not quantised by wherever CVODE happens to
                    step on a broad maximum.
    play_margin_ms  how long the run stays on FIXED dt after the pulse before handing over to
                    CVODE. It sets how much of the trace the stimulated and the sham run share
                    sample for sample, and hence the window on which DeltaV can be formed at
                    full 0.025 ms resolution -- which is what the ~0.2 ms polarisation decay
                    needs. 3 ms costs ~120 extra fixed steps and nothing else.
    """
    from config import CFG
    cfg = CFG if cfg is None else cfg
    phi = float(cfg.phase_dur_ms)
    return dict(i0_uA=(float(cfg.i0_uA) if i0_uA is None else float(i0_uA)),
                phase_dur_ms=phi, dt_ms=float(cfg.dt_ms), pre_end_ms=2.0 * phi,
                bump_ms=float(bump_ms), bump_dt_ms=float(bump_dt_ms),
                cvode_atol=float(cvode_atol), play_margin_ms=float(play_margin_ms))


def _run(inst, proto, pos_xy, theta_deg, i0_uA):
    """One rich_footprint.spikes_at call -> (fired, v_max, t_fine, v_fine, t_grid, v_grid).

    t is rebased so t = 0 is the END of phase 2. `t_fine` is at solver resolution (fixed dt
    through the pulse and for play_margin_ms after it, CVODE steps thereafter); `t_grid` is
    the uniform bump_dt_ms grid.
    """
    from rich_footprint import spikes_at
    nsp, vmax, tw, vw, tb, vb = spikes_at(
        inst["cell"], (float(pos_xy[0]), float(pos_xy[1])), float(theta_deg),
        i0_uA=float(i0_uA), phase_dur_ms=proto["phase_dur_ms"], dt_ms=proto["dt_ms"],
        detail=True, pre_end_ms=proto["pre_end_ms"], v_init_mV=inst["v_rest"],
        bump_ms=proto["bump_ms"], bump_dt_ms=proto["bump_dt_ms"],
        cvode_atol=proto["cvode_atol"], play_margin_ms=proto["play_margin_ms"])
    return (int(nsp > 0), float(vmax), np.asarray(tw, float), np.asarray(vw, float),
            np.asarray(tb, float), np.asarray(vb, float))


def sham_reference(inst, proto):
    """The unstimulated trajectory of this instance: the SAME protocol with zero current.

    With I = 0 the extracellular drive is zero at every segment, so the sham is
    placement-independent -- compute it once per instance and reuse it for every placement.
    finitialize(v_rest) is not the exact spatial equilibrium, so Vm_sham drifts slowly;
    subtracting it is what leaves only the stimulus-evoked DeltaV.
    """
    fired, _vmax, tw, vw, tb, vb = _run(inst, proto, (0.0, 0.0), 0.0, 0.0)
    if fired:
        raise RuntimeError("%s L%d (%s): the cell spikes with ZERO stimulus -- the rest state "
                           "is not quiescent and every DeltaV would be meaningless"
                           % (inst["morph"], int(inst["layer"]), inst["cell_model"]))
    return dict(t_fine=tw, v_fine=vw, t_grid=tb, v_grid=vb)


def _common_prefix(t_a, t_b, atol=1e-9):
    """Length of the leading run over which two solver time bases agree exactly.

    Both runs integrate at FIXED dt up to t_end + play_margin_ms, so their step times are
    identical there and DeltaV can be formed by plain subtraction -- no interpolation, no
    interpolation error, on precisely the stretch where the trace moves fastest. After that
    CVODE adapts independently in the two runs and the time bases diverge.
    """
    n = int(min(t_a.size, t_b.size))
    if n == 0:
        return 0
    bad = np.nonzero(np.abs(t_a[:n] - t_b[:n]) > atol)[0]
    return int(bad[0]) if bad.size else n


def simulate_instance(inst, proto, sham, pos_xy, theta_deg):
    """Stimulate this instance at (pos_xy, theta_deg) and form DeltaV against the sham.

    Returns dict(fired, v_max, pos_xy, theta_deg, r_um,
                 t_fine, v_fine, t_fine_sham, v_fine_sham,       full traces, for the timecourse
                 t_dv_fine, dv_fine,                             exact DeltaV, 0.025 ms
                 t_grid, v_grid, v_grid_sham, dv)                uniform DeltaV, bump_dt_ms

    Two DeltaV are returned on purpose; see the module docstring under "two time grids".
    """
    fired, vmax, tw, vw, tb, vb = _run(inst, proto, pos_xy, theta_deg, proto["i0_uA"])
    tbs, vbs = sham["t_grid"], sham["v_grid"]
    if tbs.shape == tb.shape and np.allclose(tbs, tb, rtol=0.0, atol=1e-9):
        vb_sham = vbs
    else:
        vb_sham = np.interp(tb, tbs, vbs)           # should not happen; correct if it does

    k = _common_prefix(tw, sham["t_fine"])
    if k < 10:
        raise RuntimeError("stimulated and sham runs share only %d fixed-dt samples -- the "
                           "fast polarisation cannot be measured. Raise play_margin_ms." % k)
    return dict(fired=int(fired), v_max=float(vmax), t_fine=tw, v_fine=vw,
                t_fine_sham=sham["t_fine"], v_fine_sham=sham["v_fine"],
                t_dv_fine=tw[:k], dv_fine=vw[:k] - sham["v_fine"][:k],
                t_grid=tb, v_grid=vb, v_grid_sham=vb_sham, dv=vb - vb_sham,
                pos_xy=(float(pos_xy[0]), float(pos_xy[1])), theta_deg=float(theta_deg),
                r_um=float(np.hypot(pos_xy[0], pos_xy[1])))


# ===========================================================================
# 3. analysis -- numbers and model curves; draws nothing
# ===========================================================================
def analyse(rec, mode="staged", t0_ms=0.0, t_peak_search_ms=2.0, early_floor=0.25,
            min_amp_mV=0.05, min_r2=0.90):
    """Fit both post-pulse components of one record.

    t0_ms = 0 is the END of phase 2 -- the instant whose DeltaV is defined to be the zero of
    the bump. It is also where the physical bump starts, which is what makes the model exactly
    specified; moving t0 later costs that exactness (a bump sampled from t0 > 0 no longer
    carries equal and opposite weights on its two exponentials).

    early_floor is where the single-exponential fit stops, as a fraction of the peak. It is a
    REAL choice, not a detail: the measured relaxation is multi-exponential, so tau_ms moves
    with it (0.22 ms at 0.10, 0.31 at 0.30, 0.47 at 0.50 on one placement) while r2 stays near
    0.91 either way. 0.25 is the default here because it keeps the drawn curve close to the
    marked peak, which is what makes the panel readable -- it is not a claim that 0.25 is
    right. The comparable, window-free number is early["t_1e_ms"], and both are printed on the
    panel and written to the CSV.

    Returns dict(mode, early, bump, joint, early_fine, resid_fine, resid_grid, bump_fine,
                 bump_grid, model_grid) -- the fitted component curves already evaluated on
    rec's two time bases, so the plotting layer needs no knowledge of which mode made them.

    early_fine and resid_grid are deliberately NOT the same curve in staged mode:

      early_fine  amp * exp(-(t - t_peak)/tau_m), the single exponential fitted to the decay
                  FROM THE MEASURED PEAK. The peak is not always at t0 -- on some placements
                  DeltaV keeps growing for ~0.1 ms after the pulse ends before it relaxes.
      resid_grid  DeltaV(t0) + offset*(1 - exp(-u/tau_m)), the residual term the BUMP fit
                  used. It is anchored at t0 by construction, because that is what pins the
                  bump to zero there. Subtracting THIS (not early_fine) is what leaves the
                  quantity the drawn bump curve was fitted to.

    In joint mode both come from the single polarisation component and coincide.
    """
    if mode not in FIT_MODES:
        raise ValueError("mode must be one of %s, got %r" % (FIT_MODES, mode))
    t_f, dv_f = rec["t_dv_fine"], rec["dv_fine"]
    t_g, dv_g = rec["t_grid"], rec["dv"]
    joint = None

    if mode == "joint":
        joint = fit_post_pulse(t_g, dv_g, t0_ms=t0_ms, min_amp_mV=min_amp_mV, min_r2=min_r2)
        early, bump = joint["early"], joint["bump"]
        early_f = eval_exp_decay(t_f, early)
        resid_f, resid_g = early_f, eval_exp_decay(t_g, early)
    else:
        # stage 1: the fast relaxation, on the exact fine DeltaV only
        early = fit_exp_decay(t_f, dv_f, t0_ms=t0_ms, t_peak_search_ms=t_peak_search_ms,
                              t_max_ms=float(t_f[-1]), floor_fraction=float(early_floor))
        # stage 2: the bump on the long window, with tau_m held fixed from stage 1
        bump = fit_double_exp_from_zero(t_g, dv_g, t0_ms=t0_ms,
                                        tau_offset_ms=early["tau_ms"],
                                        min_amp_mV=min_amp_mV, min_r2=min_r2)
        early_f = eval_exp_decay(t_f, early)
        resid_f = eval_offset_component(t_f, bump)
        resid_g = eval_offset_component(t_g, bump)

    bump_g = eval_bump_component(t_g, bump)
    return dict(mode=mode, early=early, bump=bump, joint=joint,
                early_fine=early_f, resid_fine=resid_f, resid_grid=resid_g,
                bump_fine=eval_bump_component(t_f, bump), bump_grid=bump_g,
                model_grid=resid_g + bump_g)


# ===========================================================================
# 4. plotting -- takes finished arrays and a finished fit; runs nothing
# ===========================================================================
def _style(ax, xlabel, ylabel, title=None):
    ax.grid(True, color=C_GRID, lw=0.5, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=C_INK2, labelsize=8, length=3, width=0.8)
    ax.set_xlabel(xlabel, fontsize=8.5, color=C_INK)
    ax.set_ylabel(ylabel, fontsize=8.5, color=C_INK)
    if title:
        ax.set_title(title, fontsize=9.5, color=C_INK, loc="left", pad=6)


def _shade_pulse(ax, phi, label=False):
    """The stimulus window: phase 1 = [-2phi, -phi], phase 2 = [-phi, 0]."""
    ax.axvspan(-2.0 * phi, -phi, color=C_PHASE1, alpha=0.18, lw=0, zorder=1,
               label=("stimulus phase 1 (+I)" if label else None))
    ax.axvspan(-phi, 0.0, color=C_PHASE2, alpha=0.18, lw=0, zorder=1,
               label=("stimulus phase 2 (-I)" if label else None))
    ax.axvline(0.0, color=C_INK2, ls="--", lw=0.8, zorder=2)


def _box(ax, text, right=True):
    ax.text(0.975 if right else 0.025, 0.96, text, transform=ax.transAxes,
            ha="right" if right else "left", va="top", fontsize=7.0, color=C_INK,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_GRID, lw=0.8, alpha=0.92))


def _fmt(x, w=8, d=3):
    return ("%*.*f" % (w, d, x)) if np.isfinite(x) else ("%*s" % (w, "n/a"))


def plot_timecourse(ax, rec, fit, phi, bump_ms):
    """Top panel: the soma Vm, the stimulus, and the two marked points."""
    t, v = rec["t_fine"], rec["v_fine"]
    ax.plot(rec["t_fine_sham"], rec["v_fine_sham"], color=C_SHAM, ls="--", lw=1.0, zorder=3,
            label="sham (I = 0)")
    ax.plot(t, v, color=C_DATA, lw=1.4, zorder=4, label="soma Vm (stimulated)")
    _shade_pulse(ax, phi, label=True)

    # y-limits from the POST-pulse trace: the in-pulse excursion is an extracellular artefact
    # that would flatten everything else. Say so on the panel when it is clipped.
    post = t >= 0.0
    lo = min(float(np.min(v[post])), float(np.min(rec["v_fine_sham"])))
    hi = max(float(np.max(v[post])), float(np.max(rec["v_fine_sham"])))
    pad = 0.18 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    inp = (t >= -2.0 * phi) & (t < 0.0)
    clipped = bool(inp.any() and (float(np.max(v[inp])) > hi + pad
                                  or float(np.min(v[inp])) < lo - pad))

    # Labels are placed on the side of each mark that is free, and the stimulus callout goes
    # to whichever end of the panel the first peak is NOT at -- with a 0.5 ms pulse inside an
    # 800 ms axis the shaded band is a hairline, so it has to be pointed at, and a fixed
    # corner collides as soon as a placement peaks near that corner.
    ylo, yhi = ax.get_ylim()
    rng = yhi - ylo
    frac = lambda yv: (yv - ylo) / rng if rng > 0 else 0.5
    tp_e = fit["early"]["t_peak_ms"]
    peak_high = np.isfinite(tp_e) and frac(float(np.interp(tp_e, t, v))) > 0.5

    stim_y = (ylo + 0.03 * rng) if peak_high else (yhi - 0.03 * rng)
    ax.annotate("stimulus\n%.0f us biphasic" % (2000.0 * phi), xy=(0.0, stim_y),
                xytext=(0.035 * bump_ms, stim_y), fontsize=7, color=C_INK, ha="left",
                va="bottom" if peak_high else "top",
                arrowprops=dict(arrowstyle="->", color=C_PHASE2, lw=1.1))

    for (x, c, mk, lab) in ((tp_e, C_POL, "o",
                             "first %s peak" % (fit["early"]["sign"]
                                                if fit["early"]["sign"] != "none"
                                                else "post-pulse")),
                            (fit["bump"]["data_t_peak_ms"], C_BUMP, "D", "Ih bump")):
        if not np.isfinite(x):
            continue
        y = float(np.interp(x, t, v))
        ax.plot([x], [y], marker=mk, ms=8, mfc=c, mec="white", mew=1.4, ls="none", zorder=9,
                label=lab)
        above = frac(y) < 0.5                       # label on the roomy side of the mark
        ax.annotate("%s\n%.2f ms, %.3f mV" % (lab, x, y), xy=(x, y),
                    xytext=(14, 16 if above else -32), textcoords="offset points",
                    fontsize=7, color=C_INK, va="bottom" if above else "top",
                    arrowprops=dict(arrowstyle="-", color=C_INK2, lw=0.7, alpha=0.85))

    ax.set_xlim(-2.0 * phi - 0.02 * bump_ms, bump_ms)
    _style(ax, "time from end of phase 2 (ms)", "soma Vm (mV)")
    if clipped:
        ax.text(1.0, 1.02, "in-pulse excursion clipped -- see the lower-left panel",
                transform=ax.transAxes, fontsize=7, color=C_INK2, va="bottom", ha="right")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.92,
              edgecolor=C_GRID, ncol=2)


def plot_polarisation(ax, rec, fit, phi):
    """Bottom left: zoom on the stimulus and the direct polarisation, with its exponential.

    Returns a caveat string for the caller to place at figure level (empty when there is
    none), so the note can never be clipped by the axes box.
    """
    t, dv, pol = rec["t_dv_fine"], rec["dv_fine"], fit["early_fine"]
    e, b = fit["early"], fit["bump"]
    t_hi = min(float(t[-1]), max(3.0, 14.0 * e["tau_ms"]) if np.isfinite(e["tau_ms"]) else 3.0)
    m = (t >= -2.0 * phi) & (t <= t_hi)

    ax.axhline(0.0, color=C_INK2, lw=0.7, ls=":", zorder=2)
    if np.isfinite(e["t_fit_lo_ms"]) and np.isfinite(e["t_fit_hi_ms"]):
        ax.axvspan(e["t_fit_lo_ms"], e["t_fit_hi_ms"], color=C_POL, alpha=0.14, lw=0, zorder=1,
                   label="fitted window")
    ax.plot(t[m], dv[m], color=C_DATA, lw=1.6, zorder=5, label="DeltaV (0.025 ms)")
    # Solid inside the fitted window, dotted only FORWARD of it. The curve is never drawn
    # before t_fit_lo: the log-linear fit is weighted uniformly in log space, so when the
    # relaxation is not exactly single-exponential its intercept can overshoot the measured
    # peak (A vs peak in the box below), and extrapolating backwards would stretch the axis
    # with a value that was never fitted.
    if np.isfinite(e["t_fit_hi_ms"]):
        inw = m & (t >= e["t_fit_lo_ms"]) & (t <= e["t_fit_hi_ms"])
        fwd = m & (t > e["t_fit_hi_ms"])
    else:
        inw = fwd = np.zeros_like(m)
    ax.plot(t[fwd], pol[fwd], color=C_POL, lw=1.1, ls=":", zorder=6)
    ax.plot(t[inw], pol[inw], color=C_POL, lw=2.2, ls="--", zorder=7,
            label="fit  A exp(-(t-t_peak)/tau_m)")
    ax.plot(t[m], fit["bump_fine"][m], color=C_BUMP, lw=1.2, ls="-.", zorder=4,
            label="bump component (negligible here)")
    _shade_pulse(ax, phi)

    if np.isfinite(e["t_peak_ms"]):
        ax.plot([e["t_peak_ms"]], [e["peak_mV"]], marker="o", ms=8, mfc=C_POL, mec="white",
                mew=1.4, ls="none", zorder=9)
    if np.isfinite(b["dv_t0_mV"]):                 # the value the bump pinning sets to zero
        ax.plot([b["t_fit_lo_ms"]], [b["dv_t0_mV"]], marker="s", ms=7, mfc="white", mec=C_INK,
                mew=1.4, ls="none", zorder=9, label="DeltaV(t0), pinned to 0 in the bump fit")
    ax.set_xlim(-2.0 * phi - 0.02 * t_hi, t_hi)
    vis = [dv[m]] + ([pol[inw]] if inw.any() else [])     # the extrapolation must not set ylim
    lo = float(min(np.nanmin(a) for a in vis))
    hi = float(max(np.nanmax(a) for a in vis))
    pad = 0.12 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    _style(ax, "time from end of phase 2 (ms)", "DeltaV = Vm(stim) - Vm(sham)  (mV)",
           "direct polarisation: single exponential")
    _box(ax, "peak      = %s mV @ %s ms\nDeltaV(t0)= %s mV\n"
             "tau_m     = %s ms  (fit, %s-%s ms)\nt_1e      = %s ms  (measured)\n"
             "A/peak    = %s\nr2        = %s\nfit_ok    = %8d"
         % (_fmt(e["peak_mV"]), _fmt(e["t_peak_ms"], 6, 3), _fmt(b["dv_t0_mV"]),
            _fmt(e["tau_ms"], 8, 4), _fmt(e["t_fit_lo_ms"], 4, 2),
            _fmt(e["t_fit_hi_ms"], 4, 2), _fmt(e.get("t_1e_ms", float("nan")), 8, 4),
            _fmt(e.get("amp_over_peak", float("nan")), 8, 3), _fmt(e["r2"], 8, 4),
            int(e["fit_ok"])))
    ax.legend(fontsize=6.4, loc="lower right", frameon=True, framealpha=0.92,
              edgecolor=C_GRID)
    aop = e.get("amp_over_peak", float("nan"))
    if np.isfinite(aop) and abs(aop - 1.0) > 0.15:
        return ("A/peak = %.2f: over this window the relaxation is not a single exponential, "
                "so tau_m depends on where the window stops. t_1e (measured, model-free) is "
                "the number to compare across placements." % aop)
    return ""


def plot_bump(ax, rec, fit, bump_ms):
    """Bottom right: the Ih bump and its double exponential, pinned to zero at t0."""
    t, dv = rec["t_grid"], rec["dv"]
    b = fit["bump"]
    m = t >= (b["t_fit_lo_ms"] if np.isfinite(b["t_fit_lo_ms"]) else 0.0) - 1e-9

    ax.axhline(0.0, color=C_INK2, lw=0.7, ls=":", zorder=2)
    ax.plot(t[m], (dv - fit["resid_grid"])[m], color=C_DATA, lw=1.6, zorder=5,
            label="DeltaV - residual polarisation")
    ax.plot(t[m], fit["bump_grid"][m], color=C_BUMP, lw=2.0, ls="--", zorder=6,
            label="fit  Ab [exp(-t/tau_d) - exp(-t/tau_r)]")

    t0 = b["t_fit_lo_ms"] if np.isfinite(b["t_fit_lo_ms"]) else 0.0
    ax.plot([t0], [0.0], marker="s", ms=8, mfc="white", mec=C_INK, mew=1.5, ls="none", zorder=9)
    ax.annotate("t0 = end of pulse\nDeltaV(t0) := 0", xy=(t0, 0.0),
                xytext=(0.06 * bump_ms, 0.0), textcoords="data", fontsize=7, color=C_INK,
                va="center", arrowprops=dict(arrowstyle="-", color=C_INK2, lw=0.7, alpha=0.85))
    if np.isfinite(b["t_peak_ms"]):
        ax.plot([b["t_peak_ms"]], [b["peak_mV"]], marker="D", ms=8, mfc=C_BUMP, mec="white",
                mew=1.4, ls="none", zorder=9)
        ax.annotate("peak %.3f mV @ %.1f ms" % (b["peak_mV"], b["t_peak_ms"]),
                    xy=(b["t_peak_ms"], b["peak_mV"]), xytext=(16, -18),
                    textcoords="offset points", fontsize=7, color=C_INK,
                    arrowprops=dict(arrowstyle="-", color=C_INK2, lw=0.7, alpha=0.85))
    ax.set_xlim(-0.02 * bump_ms, bump_ms)
    _style(ax, "time from end of phase 2 (ms)", "bump component (mV)",
           "Ih bump: double exponential, pinned at t0")
    _box(ax, "Ab       = %s mV\ntau_rise = %s ms\ntau_decay= %s ms\n"
             "offset   = %s mV  (expect %s)\nr2       = %s\nfit_ok   = %8d"
         % (_fmt(b["amp_mV"]), _fmt(b["tau_rise_ms"], 8, 2), _fmt(b["tau_decay_ms"], 8, 2),
            _fmt(b["offset_mV"]), _fmt(b["offset_expected_mV"], 6, 3), _fmt(b["r2"], 8, 5),
            int(b["fit_ok"])))
    ax.legend(fontsize=6.6, loc="lower right", frameon=True, framealpha=0.92, edgecolor=C_GRID)


def plot_instance(rec, fit, inst, proto, figsize=(11.5, 8.0)):
    """One finished page. Runs no simulation and no fitting -- pass it finished arrays."""
    phi = float(proto["phase_dur_ms"])
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.46, wspace=0.22,
                          left=0.065, right=0.985, top=0.855, bottom=0.105)
    plot_timecourse(fig.add_subplot(gs[0, :]), rec, fit, phi, proto["bump_ms"])
    note = plot_polarisation(fig.add_subplot(gs[1, 0]), rec, fit, phi)
    plot_bump(fig.add_subplot(gs[1, 1]), rec, fit, proto["bump_ms"])
    if note:
        fig.text(0.065, 0.016, note, fontsize=6.8, color=C_INK2, ha="left", va="bottom")

    x, y = rec["pos_xy"]
    head = ("%s  |  morphology %s, layer %d um  |  soma (%+.0f, %+.0f) um, r = %.0f um, "
            "theta = %.0f deg  |  +/-%.0f uA%s"
            % (inst["cell_model"], inst["morph"], int(inst["layer"]), x, y, rec["r_um"],
               rec["theta_deg"], proto["i0_uA"], "   SPIKE" if rec["fired"] else ""))
    sub = ("DeltaV = Vm(stim) - Vm(sham);  t = 0 at the END of phase 2;  "
           "v_init = %.4f mV;  fit mode '%s'.\n"
           "The bump is pinned so that DeltaV(t0) counts as zero: the displacement the pulse "
           "leaves behind is carried by a separate\nresidual term, whose amplitude should "
           "come out equal to -DeltaV(t0) (both printed, lower right)."
           % (inst["v_rest"], fit["mode"]))
    fig.suptitle(head, fontsize=10.5, color=C_INK, x=0.065, ha="left", y=0.978)
    fig.text(0.065, 0.945, sub, fontsize=7.2, color=C_INK2, ha="left", va="top",
             linespacing=1.45)
    return fig


# ===========================================================================
# 5. driver
# ===========================================================================
CSV_HEADER = (["cell_model", "morph", "layer_um", "x_um", "y_um", "r_um", "theta_deg",
               "i0_uA", "bump_ms", "fit_mode", "fired", "v_rest_mV"]
              + list(EXPDECAY_COLUMNS) + list(DEXP_COLUMNS)
              + ["bump_data_peak_mV", "bump_data_t_peak_ms"])


def default_placements(cfg=None, n=6):
    """A spread of instances: different morphologies, layers, distances and orientations.

    Hand-picked rather than random so the same figure comes back every run, and chosen to span
    what the bump is expected to depend on -- radial distance from the array, orientation
    relative to the dipole, and slice thickness.
    """
    from config import CFG
    cfg = CFG if cfg is None else cfg
    m = list(cfg.morphologies)
    lay = list(cfg.layers_um)
    mid = lay[len(lay) // 2]
    base = [(m[0], mid, 90.0, -30.0, 0.0),
            (m[0], mid, -150.0, 40.0, 90.0),
            (m[0], lay[0], 0.0, -230.0, 0.0),
            (m[1 % len(m)], mid, 120.0, 60.0, 180.0),
            (m[1 % len(m)], lay[-1], -110.0, -110.0, 270.0),
            (m[2 % len(m)], mid, 170.0, -90.0, 45.0)]
    return base[:max(1, int(n))]


def parse_placements(text):
    """'morph:layer:x:y:theta,...' -> [(morph, layer, x, y, theta), ...]"""
    out = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 5:
            raise ValueError("placement %r is not morph:layer:x:y:theta" % chunk)
        out.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3]),
                    float(parts[4])))
    return out


def _row(inst, proto, rec, fit):
    b = fit["bump"]
    rnd = lambda v, n: (round(float(v), n) if np.isfinite(v) else "")
    return ([inst["cell_model"], inst["morph"], int(inst["layer"]), rec["pos_xy"][0],
             rec["pos_xy"][1], round(rec["r_um"], 2), rec["theta_deg"], proto["i0_uA"],
             proto["bump_ms"], fit["mode"], rec["fired"], round(inst["v_rest"], 4)]
            + expdecay_row_values(fit["early"]) + dexp_row_values(b)
            + [rnd(b["data_peak_mV"], 6), rnd(b["data_t_peak_ms"], 3)])


def main(cell_model="full_tuned", placements=None, i0_uA=None, bump_ms=800.0,
         bump_dt_ms=0.5, cvode_atol=1e-6, play_margin_ms=3.0, t0_ms=0.0, mode="staged",
         early_floor=0.25, n_instances=6, out_pdf="plot_vm_examples.pdf",
         out_csv="plot_vm_examples.csv", verbose=True):
    """Simulate every placement, fit it, and write one PDF page and one CSV row each.

    Cells are built per (morphology, layer) and reused across the placements that share them,
    because every live cell slows every simulation.
    """
    from config import CFG
    here = os.path.dirname(os.path.abspath(__file__))
    proto = default_protocol(CFG, i0_uA=i0_uA, bump_ms=bump_ms, bump_dt_ms=bump_dt_ms,
                             cvode_atol=cvode_atol, play_margin_ms=play_margin_ms)
    places = default_placements(CFG, n_instances) if placements is None else list(placements)

    rows, cache = [], {}
    pdf_path = os.path.join(here, out_pdf)
    with PdfPages(pdf_path) as pdf:
        for (morph, layer, x, y, theta) in places:
            key = (str(morph), float(layer))
            if key not in cache:
                if verbose:
                    print("[build] %s L%d (%s)" % (morph, int(layer), cell_model))
                inst = build_instance(morph, layer, cell_model, cfg=CFG)
                cache[key] = (inst, sham_reference(inst, proto))
            inst, sham = cache[key]

            rec = simulate_instance(inst, proto, sham, (x, y), theta)
            fit = analyse(rec, mode=mode, t0_ms=t0_ms, early_floor=early_floor)
            pdf.savefig(plot_instance(rec, fit, inst, proto))
            plt.close("all")
            rows.append(_row(inst, proto, rec, fit))
            if verbose:
                e, b = fit["early"], fit["bump"]
                print("  %-7s L%-3d (%+6.0f,%+6.0f) th=%3.0f | peak %+7.3f mV @%5.2f ms "
                      "tau_m %6.3f t_1e %6.3f (ok %d) | bump %+6.3f mV @%6.1f ms "
                      "tau_r %6.2f tau_d %7.2f r2 %.5f (ok %d)%s"
                      % (morph, int(layer), x, y, theta, e["peak_mV"], e["t_peak_ms"],
                         e["tau_ms"], e.get("t_1e_ms", float("nan")), e["fit_ok"],
                         b["peak_mV"], b["t_peak_ms"], b["tau_rise_ms"], b["tau_decay_ms"],
                         b["r2"], b["fit_ok"], "  SPIKE" if rec["fired"] else ""))

    csv_path = os.path.join(here, out_csv)
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(CSV_HEADER)
        wr.writerows(rows)
    if verbose:
        print("done PDF:", pdf_path, "(%d pages)" % len(rows))
        print("done CSV:", csv_path)
    return pdf_path, csv_path


def _cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-model", default="full_tuned", choices=list(PLOT_CELL_MODELS))
    ap.add_argument("--fit", dest="mode", default="staged", choices=list(FIT_MODES),
                    help="'staged' (default): tau_m from the fine trace, then the bump with "
                         "tau_m held fixed. 'joint': all five parameters at once on the "
                         "coarse grid (bump_kinetics.fit_post_pulse).")
    ap.add_argument("--placements", default=None,
                    help="morph:layer:x:y:theta, comma-separated. Default: a built-in spread.")
    ap.add_argument("--n-instances", type=int, default=6, help="how many default placements")
    ap.add_argument("--i0-uA", type=float, default=None, help="default: config.i0_uA")
    ap.add_argument("--bump-ms", type=float, default=800.0,
                    help="window integrated after the pulse (default 800)")
    ap.add_argument("--bump-dt-ms", type=float, default=0.5, help="fixed output grid for fits")
    ap.add_argument("--play-margin-ms", type=float, default=3.0,
                    help="fixed-dt tail after the pulse; sets the exact-DeltaV window")
    ap.add_argument("--cvode-atol", type=float, default=1e-6)
    ap.add_argument("--t0-ms", type=float, default=0.0,
                    help="anchor of the bump fit; 0 = end of phase 2 (the default, and the "
                         "only value for which the model is exactly specified)")
    ap.add_argument("--early-floor", type=float, default=0.25,
                    help="where the single-exponential fit stops, as a fraction of the peak. "
                         "The relaxation is multi-exponential, so tau_m moves with this; "
                         "early_t_1e_ms in the CSV is the window-free number.")
    ap.add_argument("--out-pdf", default="plot_vm_examples.pdf")
    ap.add_argument("--out-csv", default="plot_vm_examples.csv")
    a = ap.parse_args(argv)
    return main(cell_model=a.cell_model, mode=a.mode,
                placements=(parse_placements(a.placements) if a.placements else None),
                i0_uA=a.i0_uA, bump_ms=a.bump_ms, bump_dt_ms=a.bump_dt_ms,
                play_margin_ms=a.play_margin_ms, cvode_atol=a.cvode_atol, t0_ms=a.t0_ms,
                early_floor=a.early_floor, n_instances=a.n_instances, out_pdf=a.out_pdf,
                out_csv=a.out_csv)


if __name__ == "__main__":
    _cli()
    sys.exit(0)
