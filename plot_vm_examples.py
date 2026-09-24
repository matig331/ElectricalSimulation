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

NEURONS FROM A CAMPAIGN (--from-csv, section 6)
The default placements are hand-picked. --from-csv instead draws neurons a campaign actually
simulated: rows are chosen from its CSV (default: accepted fits spread over distance, plus the
rejected fits with the largest measured bump), each placement is REGENERATED from
(seed, culture) and checked against the row, then re-simulated with the campaign's protocol
and fitted with the same fit_staged. Output: a gallery page of all of them (DeltaV with the
reconstruction over it), then one page like the ones above per neuron, each saying whether the
re-simulation reproduces the row. Measured in the sandbox: DeltaV_end equal to the row's
1e-6 mV rounding, bump peak to ~1e-4 mV, tau_decay to the rounding. Parallel over
(morphology, layer) groups (spawned processes).

Run:
    python plot_vm_examples.py                              # full_tuned, 6 instances
    python plot_vm_examples.py --cell-model soma_only       # expect NO bump: Ih is dendritic
    python plot_vm_examples.py --placements "60308:80:90:-30:0,130303:80:-150:40:90"
    python plot_vm_examples.py --fit joint --help
    python plot_vm_examples.py --from-csv results_full_tuned/<run>/culture_Pactivation.csv \
        --bump-ms <the campaign's window> --n-examples 24 --processes 8

Outputs (next to this file unless a path is given): plot_vm_examples.pdf, one page per
instance, and plot_vm_examples.csv, one row per instance with every fitted parameter;
with --from-csv, campaign_examples.pdf / .csv (the CSV also holds the row's own numbers and
the re-simulation difference).

Test:  python smoke_test_plot_vm_examples.py        (NEURON, a few minutes)
       python smoke_test_campaign_examples.py       (no NEURON, seconds: row choice, placement
                                                     recovery, cross-check, gallery)
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

from bump_kinetics import (STAGED_COLUMNS, common_prefix_len, eval_bump_component,
                           eval_exp_decay, eval_offset_component, fit_post_pulse, fit_staged,
                           staged_row_values)

# --- palette -----------------------------------------------------------------------------
# Okabe-Ito, in fixed order, validated for colour-vision deficiency against a light surface
# (worst adjacent pair dE 11.0 deutan / 24.2 normal). Every series is ALSO separated by line
# style and carries a legend entry, so identity is never colour alone. Text stays in ink.
C_DATA = "#0072B2"          # the measured trace
C_BUMP = "#009E73"          # the bump component (double exponential)
C_POL = "#E69F00"           # the polarisation component (single exponential)
C_MODEL = "#D55E00"         # the reconstruction: both fitted components summed
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
    from rich_cell import build_rich_cell, settled_resting_voltage
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
        # full_tuned: impose that measured rest everywhere by countering the other channels
        # with e_pas. Through culture_export.apply_model_state -- the function the campaign
        # uses -- so an example is tuned exactly as a campaign neuron is (same celsius, same
        # config.leak_tuning switch), not by a look-alike copy here.
        from culture_export import apply_model_state
        report = apply_model_state(cell, cell_model, cfg, v_rest)
    return dict(cell=cell, morph=str(morph), layer=float(layer), cell_model=str(cell_model),
                v_rest=v_rest, leak_report=report)


# ===========================================================================
# 2. simulation -- the protocol, independent of which cell it is applied to
# ===========================================================================
def default_protocol(cfg=None, i0_uA=None, bump_ms=800.0, bump_dt_ms=0.5, cvode_atol=1e-6,
                     play_margin_ms=3.0, baseline_ms=100.0, pre_ms=None):
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
    baseline_ms     how long the cell is integrated BEFORE the stimulus. The campaign uses the
                    rich_footprint default of 5 ms, enough to classify an outcome but showing
                    nothing about calibration. 100 ms here puts 4000 samples of pre-stimulus
                    trace on the figure, so a tuned cell being genuinely flat at rest is
                    something you can SEE rather than take on trust.

                    DO NOT RAISE IT BLINDLY. The extracellular field is played into ~1250
                    segments on the full fixed-dt grid, so the play Vectors grow with the
                    baseline: measured 0.038 s per ms of simulation, and a SEGMENTATION FAULT
                    between 150 and 300 ms on a 7 GB machine. 100 ms costs ~4.7 s per run and
                    is verified; 50 ms costs 2.6 s. A larger node may take more -- raise it in
                    steps, and expect a hard crash rather than an exception if you go too far.
    pre_ms          how much of that baseline is returned and drawn. None = all of it.
    """
    from config import CFG
    cfg = CFG if cfg is None else cfg
    phi = float(cfg.phase_dur_ms)
    base = float(baseline_ms)
    return dict(i0_uA=(float(cfg.i0_uA) if i0_uA is None else float(i0_uA)),
                phase_dur_ms=phi, dt_ms=float(cfg.dt_ms), baseline_ms=base,
                pre_end_ms=(base if pre_ms is None else float(pre_ms)),
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
        baseline_ms=proto["baseline_ms"],
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

    k = common_prefix_len(tw, sham["t_fine"])
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
        st = fit_staged(t_f, dv_f, t_g, dv_g, t0_ms=t0_ms,
                        t_peak_search_ms=t_peak_search_ms, early_floor=early_floor,
                        min_amp_mV=min_amp_mV, min_r2=min_r2)
        early, bump = st["early"], st["bump"]
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


def reconstruction(rec, fit):
    """The fitted trace put back into absolute Vm, on the uniform grid.

        Vm_model(t) = Vm_sham(t) + DeltaV_model(t),
        DeltaV_model = the direct polarisation component + the bump component

    Both components are evaluated from the SAME fit the lower panels draw, so the top panel is
    not an independent claim -- it is those two curves summed and referred back to the sham.
    Returns (t, Vm_model, residual) on rec's uniform grid, all nan before t0. The residual is
    measured minus model, computed on that grid with no interpolation: the measured Vm is
    recorded on it directly.
    """
    t = rec["t_grid"]
    vm_model = rec["v_grid_sham"] + fit["model_grid"]
    return t, vm_model, rec["v_grid"] - vm_model


def baseline_flatness(rec, phi, guard_ms=1.0):
    """Peak-to-peak Vm over the pre-stimulus baseline, its mean, and the sample count.

    The calibration check, as a number. A cell whose leak was tuned to its own settled rest
    should sit flat: measured 1.75e-05 mV peak-to-peak over 50 ms, 5.71e-05 over 100 ms. A
    baseline that drifts means the cell was initialised away from its equilibrium, which is
    exactly what tune_leak_isopotential exists to remove.
    """
    t, v = rec["t_fine"], rec["v_fine"]
    m = t < -(2.0 * phi + float(guard_ms))
    if int(m.sum()) < 2:
        return float("nan"), float("nan"), 0
    return float(v[m].max() - v[m].min()), float(v[m].mean()), int(m.sum())


def plot_timecourse(ax, rec, fit, phi, bump_ms, pre_ms=0.0):
    """Top panel: the soma Vm, the stimulus, the two marked points, and the reconstruction."""
    t, v = rec["t_fine"], rec["v_fine"]
    ax.plot(rec["t_fine_sham"], rec["v_fine_sham"], color=C_SHAM, ls="--", lw=1.0, zorder=3,
            label="sham (I = 0)")
    ax.plot(t, v, color=C_DATA, lw=1.4, zorder=4, label="soma Vm (stimulated)")
    tm, vm_model, _res = reconstruction(rec, fit)
    fin = np.isfinite(vm_model)
    ax.plot(tm[fin], vm_model[fin], color=C_MODEL, ls="--", lw=1.8, zorder=6,
            label="reconstruction = sham + both fitted components")
    _shade_pulse(ax, phi, label=True)

    # y-limits from the POST-pulse trace: the in-pulse excursion is an extracellular artefact
    # that would flatten everything else. Say so on the panel when it is clipped.
    post = t >= 0.0
    lo = min(float(np.min(v[post])), float(np.min(rec["v_fine_sham"])))
    hi = max(float(np.max(v[post])), float(np.max(rec["v_fine_sham"])))
    if fin.any():                                  # the model must not be clipped out of view
        lo = min(lo, float(np.min(vm_model[fin])))
        hi = max(hi, float(np.max(vm_model[fin])))
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

    spread, mean_v, n_pre = baseline_flatness(rec, phi)
    if n_pre > 1:
        ax.axvspan(-float(pre_ms), -2.0 * phi, color=C_SHAM, alpha=0.07, lw=0, zorder=0)
        # ABOVE the axes: inside, it collided with the stimulus callout at the bottom-left
        # when the first peak sat high and at the top-left when it sat low.
        ax.text(0.0, 1.02, "baseline %.0f ms: Vm %.4f mV, peak-to-peak %.2e mV (%d samples)"
                % (pre_ms, mean_v, spread, n_pre), transform=ax.transAxes, ha="left",
                va="bottom", fontsize=6.8, color=C_INK, family="monospace",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_GRID, lw=0.7, alpha=0.92))
    ax.set_xlim(-float(pre_ms) - 0.01 * bump_ms, bump_ms)
    _style(ax, "", "soma Vm (mV)")
    ax.tick_params(labelbottom=False)
    if clipped:
        ax.text(1.0, 1.02, "in-pulse excursion clipped -- see the lower-left panel",
                transform=ax.transAxes, fontsize=7, color=C_INK2, va="bottom", ha="right")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.92,
              edgecolor=C_GRID, ncol=2)
    return tm, _res


def plot_residual(ax, t_res, res, phi, bump_ms, pre_ms=0.0):
    """The strip under the timecourse: measured Vm minus the reconstruction.

    This is the panel that makes the fit falsifiable. A structured residual -- a slow arc, a
    step, a ringing -- means the two-component model is missing something; scatter about zero
    means it is not.
    """
    fin = np.isfinite(res)
    ax.axhline(0.0, color=C_INK2, lw=0.8, ls=":", zorder=2)
    if fin.any():
        ax.plot(t_res[fin], res[fin], color=C_MODEL, lw=1.1, zorder=4)
        r = res[fin]
        rms = float(np.sqrt(np.mean(r ** 2)))
        mx = float(np.max(np.abs(r)))
        lim = max(1.2 * mx, 1e-4)
        ax.set_ylim(-lim, lim)
        ax.text(0.995, 0.90, "residual  RMS %.4f mV   max |.| %.4f mV" % (rms, mx),
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=C_INK,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_GRID, lw=0.7,
                          alpha=0.92))
    _style(ax, "time from end of phase 2 (ms)", "measured -\nmodel (mV)")


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
    vis = [a for a in vis if np.isfinite(a).any()]         # a rejected fit's curve is all nan
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
                    xy=(b["t_peak_ms"], b["peak_mV"]), xytext=(0, -26),
                    textcoords="offset points", fontsize=7, color=C_INK, ha="center",
                    va="top",
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


def plot_instance(rec, fit, inst, proto, figsize=(11.5, 9.0), note_line=""):
    """One finished page. Runs no simulation and no fitting -- pass it finished arrays."""
    phi = float(proto["phase_dur_ms"])
    fig = plt.figure(figsize=figsize)
    # Two blocks, each with its own spacing: the timecourse and its residual are glued
    # together (they share an x axis and must read as one panel), the fits below need room
    # for their own titles and x labels.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.95], hspace=0.30,
                             left=0.075, right=0.985, top=0.855, bottom=0.095)
    top = outer[0].subgridspec(2, 1, height_ratios=[1.05, 0.30], hspace=0.08)
    ax_top = fig.add_subplot(top[0])
    pre_ms = float(proto.get("pre_end_ms", 2.0 * phi))
    t_res, res = plot_timecourse(ax_top, rec, fit, phi, proto["bump_ms"], pre_ms)
    plot_residual(fig.add_subplot(top[1], sharex=ax_top), t_res, res, phi, proto["bump_ms"],
                  pre_ms)
    bot = outer[1].subgridspec(1, 2, wspace=0.22)
    note = plot_polarisation(fig.add_subplot(bot[0]), rec, fit, phi)
    plot_bump(fig.add_subplot(bot[1]), rec, fit, proto["bump_ms"])
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
    if note_line:
        sub += "\n" + note_line
    fig.suptitle(head, fontsize=10.5, color=C_INK, x=0.065, ha="left", y=0.978)
    fig.text(0.065, 0.945, sub, fontsize=7.2, color=C_INK2, ha="left", va="top",
             linespacing=1.45)
    return fig


# ===========================================================================
# 5. driver
# ===========================================================================
CSV_HEADER = (["cell_model", "morph", "layer_um", "x_um", "y_um", "r_um", "theta_deg",
               "i0_uA", "bump_ms", "fit_mode", "fired", "v_rest_mV"]
              + list(STAGED_COLUMNS))


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
    return ([inst["cell_model"], inst["morph"], int(inst["layer"]), rec["pos_xy"][0],
             rec["pos_xy"][1], round(rec["r_um"], 2), rec["theta_deg"], proto["i0_uA"],
             proto["bump_ms"], fit["mode"], rec["fired"], round(inst["v_rest"], 4)]
            + staged_row_values(fit))


def main(cell_model="full_tuned", placements=None, i0_uA=None, bump_ms=800.0,
         bump_dt_ms=0.5, cvode_atol=1e-6, play_margin_ms=3.0, baseline_ms=100.0,
         t0_ms=0.0, mode="staged",
         early_floor=0.25, n_instances=6, out_pdf="plot_vm_examples.pdf",
         out_csv="plot_vm_examples.csv", verbose=True):
    """Simulate every placement, fit it, and write one PDF page and one CSV row each.

    Cells are built per (morphology, layer) and reused across the placements that share them,
    because every live cell slows every simulation.
    """
    from config import CFG
    here = os.path.dirname(os.path.abspath(__file__))
    proto = default_protocol(CFG, i0_uA=i0_uA, bump_ms=bump_ms, bump_dt_ms=bump_dt_ms,
                             cvode_atol=cvode_atol, play_margin_ms=play_margin_ms,
                             baseline_ms=baseline_ms)
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


# ===========================================================================
# 6. examples FROM A CAMPAIGN: re-simulate rows of a results CSV and show their fits
#
# The campaign stores the fitted numbers of every neuron, not its trace (a trace per neuron
# would be ~2 MB each). To SEE the reconstruction for neurons the campaign actually ran, the
# rows are re-simulated here with the long window and fitted by the same fit_staged.
#
# A row does not store the neuron's raw rotation -- theta_orient_deg is folded to [0, 90] and
# cannot be inverted -- but it does not need to: placements are drawn deterministically from
# (seed, culture) by culture_export.culture_draws, so the exact placement is REGENERATED and
# then CHECKED against the row (morphology, x, y to 0.006 um, folded orientation to 0.06 deg).
# The regenerated position is the one simulated, not the row's rounded x/y.
# A row that fails the check is skipped and reported, never silently re-placed. Verified on
# the cluster dry run: 48 of 48 rows regenerated exactly.
#
# The re-simulation also checks the campaign: its DeltaV_end and fitted bump are compared with
# the numbers the row recorded, and the difference is printed and written to the CSV.
# ===========================================================================
SELECT_MODES = ("stratified", "accepted", "rejected", "nearest", "random")


def _ff(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def row_fit_state(row):
    """'accepted' | 'rejected' | 'unmeasured' for one campaign row.

    An unmeasured row (bump window off, or its culture outside config.bump_culture_fraction)
    is written with dexp_fit_ok = 0 like a rejected one; what tells them apart is
    dexp_dv_t0_mV, which every measured row has -- DeltaV at t0 exists whether or not the
    bump fit is accepted.
    """
    if str(row.get("dexp_fit_ok", "")).strip() in ("1", "1.0"):
        return "accepted"
    if str(row.get("dexp_dv_t0_mV", "")).strip() == "":
        return "unmeasured"
    return "rejected"


def load_campaign_rows(path):
    """(source, rows) of a campaign: a CSV file, a results directory holding
    culture_Pactivation.csv (any of the three per-outcome files carries the same placements
    and kinetics), or a PARTS directory of raw worker files part_*.csv -- usable while a run
    is still going, since every block is fsync'ed."""
    import glob
    files = [path]
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*Pactivation*.csv"))) or \
            sorted(glob.glob(os.path.join(path, "**", "*Pactivation*.csv"), recursive=True))
        files = cands[:1] or sorted(glob.glob(os.path.join(path, "part_*.csv")))
        if not files:
            raise SystemExit("no *Pactivation*.csv and no part_*.csv under %s" % path)
        path = files[0] if len(files) == 1 else "%s (%d part files)" % (path, len(files))
    rows = []
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    need = ("seed", "culture", "neuron", "morphology", "layer_um", "x_um", "y_um",
            "theta_orient_deg", "dist_dipole3d_um")
    missing = [c for c in need if c not in (rows[0] if rows else {})]
    if missing:
        raise SystemExit("%s lacks %s -- not a campaign row file" % (path, ", ".join(missing)))
    return path, rows


def draw_context(cfg):
    """Exactly the geometry the campaign drivers pass to culture_draws (culture_worker.py)."""
    import field as F
    from culture_export import electrode_center, dipole_axis_deg, dipole_frame
    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=not cfg.bipolar)
    dip_c, dip_d = dipole_frame(elec, sign)
    return dict(span=cfg.span_half_um(), elec=elec, center=electrode_center(elec),
                axis=dipole_axis_deg(elec, sign), dip_c=dip_c, dip_d=dip_d,
                morphs=list(cfg.morphologies), h_soma=cfg.h_soma_um)


def recover_placements(rows, cfg, n_per_culture=None):
    """(placements, skipped). Each placement is a dict(morph, layer, x, y, theta, row).

    Neurons per culture (N) is not stored in a row, and the draws depend on it (positions are
    drawn as one N x 2 block). Pass it when known -- full_tuned_run.pbs knows it as
    PER_CULTURE. Otherwise it is inferred as max(neuron) + 1 over each seed, which is right as
    long as ONE culture of that seed finished; if the walltime cut EVERY culture short, the
    inferred N is too small, the check below fails for every row, and the rows are skipped
    (never mis-placed) -- the message then says to pass --neurons-per-culture.
    """
    from collections import defaultdict
    from culture_export import culture_draws
    ctx = draw_context(cfg)
    n_of = defaultdict(int)
    for r in rows:
        n_of[r["seed"]] = max(n_of[r["seed"]], int(float(r["neuron"])) + 1)
    if n_per_culture:
        for k in n_of:
            n_of[k] = max(n_of[k], int(n_per_culture))
    cache, out, skipped = {}, [], []
    for r in rows:
        seed, i = int(float(r["seed"])), int(float(r["neuron"]))
        hit = None
        # 'culture' is the draw index in raw parts; culture_statistics' merged file keeps it as
        # 'local_culture' when it renumbers. Try both, keep whichever the row agrees with.
        for key in ("culture", "local_culture"):
            if key not in r or r[key] in ("", None):
                continue
            c = int(float(r[key]))
            k = (seed, c)
            if k not in cache:
                cache[k] = culture_draws(seed, c, n_of[r["seed"]], len(ctx["morphs"]),
                                         ctx["span"], ctx["elec"], ctx["center"],
                                         ctx["dip_c"], ctx["dip_d"], ctx["axis"],
                                         ctx["h_soma"])
            d = cache[k]
            if i >= len(d["midx"]):
                continue
            ok = (ctx["morphs"][int(d["midx"][i])] == r["morphology"]
                  and abs(float(d["pos"][i, 0]) - _ff(r["x_um"])) < 0.006
                  and abs(float(d["pos"][i, 1]) - _ff(r["y_um"])) < 0.006
                  and abs(float(d["th_or"][i]) - _ff(r["theta_orient_deg"])) < 0.06)
            if ok:
                # the EXACT draw, not the row's x/y: those are rounded to 0.01 um, and on a
                # soma ~100 um from the array that rounding alone moved DeltaV_end by 0.07 %
                hit = (float(d["pos"][i, 0]), float(d["pos"][i, 1]), float(d["theta"][i]))
                break
        if hit is None:
            skipped.append(r)
        else:
            out.append(dict(morph=r["morphology"], layer=_ff(r["layer_um"]),
                            x=hit[0], y=hit[1], theta=hit[2], row=r))
    return out, skipped


def select_examples(placements, n, mode="stratified", seed=0):
    """Choose `n` placements to plot, returned sorted by distance.

      stratified  ~60 % ACCEPTED fits spread evenly over the distance range (the middle row of
                  each of n equal-count distance chunks), ~40 % REJECTED fits with the largest
                  measured |bump| -- the failures worth looking at, where a bump may exist
                  but the fit said no. Rejected fits far from the array are flat and teach
                  nothing, which is why they are not stratified by distance.
      accepted    accepted fits only, stratified by distance
      rejected    rejected fits only, largest measured |bump| first
      nearest     the n placements closest to the array
      random      uniform, seeded

    Rows the campaign did not measure (row_fit_state 'unmeasured') are in neither the accepted
    nor the rejected pool; if NO row was measured, stratified/accepted fall back to spreading
    over every row by distance.
    """
    if mode not in SELECT_MODES:
        raise ValueError("select must be one of %s" % (SELECT_MODES,))
    n = int(n)
    dist = lambda p: _ff(p["row"]["dist_dipole3d_um"])

    def measured(p):
        r = p["row"]
        v = _ff(r.get("dexp_data_peak_mV", ""))
        return abs(v) if np.isfinite(v) else abs(_ff(r.get("deltaVm_end_phase2_mV", "")))

    ok = [p for p in placements if row_fit_state(p["row"]) == "accepted"]
    rej = [p for p in placements if row_fit_state(p["row"]) == "rejected"]
    if not ok and not rej and mode in ("stratified", "accepted"):
        ok = list(placements)                     # nothing measured: spread over everything

    def spread(pool, k):
        pool = sorted(pool, key=dist)
        if k <= 0 or not pool:
            return []
        if k >= len(pool):
            return pool
        return [ch[len(ch) // 2] for ch in np.array_split(np.array(pool, dtype=object), k)]

    def worst(pool, k):
        return sorted(pool, key=measured, reverse=True)[:max(0, k)]

    if mode == "accepted":
        pick = spread(ok, n)
    elif mode == "rejected":
        pick = worst(rej, n)
    elif mode == "nearest":
        pick = sorted(placements, key=dist)[:n]
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        idx = rng.permutation(len(placements))[:n]
        pick = [placements[i] for i in idx]
    else:
        n_rej = min(len(rej), int(round(0.4 * n)))
        n_ok = min(len(ok), n - n_rej)
        n_rej = min(len(rej), n - n_ok)              # top up with rejected if accepted ran out
        pick = spread(ok, n_ok) + worst(rej, n_rej)
    return sorted(list(pick), key=dist)


def _sim_group(task):
    """Worker (SPAWNED process): one (morphology, layer) cell, its sham, all its placements.

    Returns only picklable arrays and numbers -- never the cell -- so the parent can plot.
    """
    morph, layer, cell_model, proto, items = task
    from config import CFG
    inst = build_instance(morph, layer, cell_model, cfg=CFG, tag="_pvc_")
    sham = sham_reference(inst, proto)
    out = [(key, simulate_instance(inst, proto, sham, (x, y), th)) for key, x, y, th in items]
    meta = {k: inst[k] for k in ("morph", "layer", "cell_model", "v_rest")}
    return meta, out


def simulate_groups(places, cell_model, proto, processes=1, verbose=True):
    """Simulate placements grouped by (morphology, layer), in parallel over groups.

    `places` is a list of (key, morph, layer, x, y, theta). Returns {key: (meta, rec)}.
    Uses the SPAWN start method: a forked child would inherit the parent's NEURON state, and
    NEURON is not fork-safe. The parent never imports NEURON itself. A spawned child
    re-imports the calling script, so a script that calls this (or main_from_csv) must do so
    under `if __name__ == "__main__":` -- this module's own CLI does.
    """
    groups = {}
    for key, morph, layer, x, y, th in places:
        groups.setdefault((str(morph), float(layer)), []).append((key, x, y, th))
    tasks = [(m, l, cell_model, proto, items) for (m, l), items in sorted(groups.items())]
    processes = max(1, min(int(processes), len(tasks)))
    if verbose:
        print("[examples] %d placements in %d (morphology, layer) groups on %d process(es)"
              % (len(places), len(tasks), processes), flush=True)
    results = {}
    if processes == 1:
        it = map(_sim_group, tasks)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(processes)
        it = pool.imap_unordered(_sim_group, tasks, chunksize=1)
    try:
        for meta, out in it:
            for key, rec in out:
                results[key] = (meta, rec)
            if verbose:
                print("[examples] %s L%d done (%d placements)"
                      % (meta["morph"], int(meta["layer"]), len(out)), flush=True)
    except BaseException:
        if processes > 1:
            pool.terminate()        # don't wait for the other groups to finish a failed run
        raise
    if processes > 1:
        pool.close()
        pool.join()
    return results


def campaign_check(rec, fit, row):
    """Re-simulation vs the numbers the campaign row recorded.

    Measured in the sandbox (16-row full_tuned campaign, 8 re-simulated): DeltaV_end equal to
    the row's 1e-6 mV rounding on every example, and the 100 ms baseline here vs the
    campaign's 5 ms changes it by nothing at that precision. The bump numbers agree to ~1e-4
    mV / ~0.1 %; the tolerances below (1e-3 mV; 3 % peak, 5 % tau_decay) therefore catch a
    wrong neuron or a wrong window, not floating-point noise.
    """
    t, dv = rec["t_dv_fine"], rec["dv_fine"]
    dv_end = float(dv[int(np.argmin(np.abs(t)))]) if t.size else float("nan")
    row_dv = _ff(row.get("deltaVm_end_phase2_mV"))
    b = fit["bump"]
    state = row_fit_state(row)
    out = dict(dv_end_resim=dv_end, dv_end_row=row_dv, d_dv_end=abs(dv_end - row_dv),
               row_state=state, ok_resim=int(b["fit_ok"]), ok_row=int(state == "accepted"),
               peak_resim=b["peak_mV"], peak_row=_ff(row.get("dexp_peak_mV")),
               tau_d_resim=b["tau_decay_ms"], tau_d_row=_ff(row.get("dexp_tau_decay_ms")))
    agree = bool(out["d_dv_end"] < 1e-3)            # False for a nan difference as well
    if state != "unmeasured":
        # an unmeasured row has no fit to compare -- only DeltaV_end is checked for it
        agree = agree and out["ok_resim"] == out["ok_row"]
        if out["ok_resim"] and out["ok_row"]:
            agree = agree and abs(out["peak_resim"] - out["peak_row"]) < max(
                0.005, 0.03 * abs(out["peak_row"]))
            agree = agree and abs(out["tau_d_resim"] - out["tau_d_row"]) < \
                0.05 * out["tau_d_row"]
    out["agrees"] = int(agree)
    return out


def plot_gallery(items, proto, per_page=12, title=""):
    """Overview pages: one small DeltaV tile per example, with the reconstruction over it.

    DeltaV, not Vm, because each morphology rests at its own potential and DeltaV is the
    quantity every fit and statistic uses. The y range comes from t >= 3 ms, so the direct
    response -- up to ~7 mV and gone in a millisecond -- does not flatten the bump; the full
    transient is on each example's own detailed page.
    """
    figs = []
    ncol = 4
    for p0 in range(0, len(items), per_page):
        chunk = items[p0:p0 + per_page]
        nrow = int(np.ceil(len(chunk) / float(ncol)))
        fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 2.75 * nrow + 0.9), squeeze=False)
        for ax in axes.flat[len(chunk):]:
            ax.axis("off")
        for ax, it in zip(axes.flat, chunk):
            rec, fit, row, chk = it["rec"], it["fit"], it["row"], it["check"]
            b = fit["bump"]
            t, dv = rec["t_grid"], rec["dv"]
            model = fit["model_grid"]
            ok = bool(b["fit_ok"])
            ax.axhline(0.0, color=C_INK2, lw=0.6, ls=":", zorder=1)
            ax.plot(t, dv, color=C_DATA if ok else C_SHAM, lw=1.1, zorder=3)
            fin = np.isfinite(model)
            ax.plot(t[fin], model[fin], color=C_MODEL, lw=1.3 if ok else 0.9,
                    ls="--", zorder=4, alpha=1.0 if ok else 0.6)
            late = t >= 3.0
            vals = np.concatenate([dv[late], model[late & fin]]) if late.any() else dv
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo, hi = float(vals.min()), float(vals.max())
                pad = 0.12 * (hi - lo) + 1e-4
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.02 * proto["bump_ms"], proto["bump_ms"])
            ax.grid(True, color=C_GRID, lw=0.4)
            for sd in ("top", "right"):
                ax.spines[sd].set_visible(False)
            ax.tick_params(labelsize=6, colors=C_INK2, length=2)
            # the measured bump peak (model-free), as the diamond on the detailed pages
            tp, vp = b.get("data_t_peak_ms", np.nan), b.get("data_peak_mV", np.nan)
            if np.isfinite(tp) and np.isfinite(vp):
                ax.plot([tp], [vp], marker="D", ms=4.5, mfc=C_BUMP, mec="white", mew=0.8,
                        ls="none", zorder=6)
            # numbers in the title, not in a box: a box lands on the curve somewhere on every
            # layout, since accepted bumps peak mid-window and rejected traces fill it flat
            if ok:
                stats = ("peak %+.3f mV @ %.0f ms   r2 %.4f\ntau_rise %.1f ms   tau_decay %.0f ms"
                         % (b["peak_mV"], b["t_peak_ms"], b["r2"], b["tau_rise_ms"],
                            b["tau_decay_ms"]))
            else:
                stats = ("REJECTED   r2 %.3f\nmeasured peak %+.4f mV"
                         % (b["r2"], b.get("data_peak_mV", float("nan"))))
            ax.set_title("#%d  %s L%d  d = %.0f um%s\n%s"
                         % (it["idx"], it["morph"], int(it["layer"]),
                            _ff(row["dist_dipole3d_um"]), "" if chk["agrees"] else "  (!)",
                            stats),
                         fontsize=6.6, color=C_INK, loc="left", pad=3, linespacing=1.3)
        fig.suptitle("%s  --  DeltaV (solid) and the reconstruction (dashed); blue = fit "
                     "accepted, grey = rejected; diamond = measured bump peak;\n(!) = the "
                     "re-simulation differs from the campaign row. Page %d of %d; each example "
                     "has its own page after these."
                     % (title, p0 // per_page + 1, int(np.ceil(len(items) / float(per_page)))),
                     fontsize=8.0, color=C_INK, x=0.01, ha="left", y=0.995)
        fig.text(0.01, 0.005, "y range from t >= 3 ms: the direct response is clipped here and "
                 "shown in full on each example's own page. x: ms from the end of phase 2.",
                 fontsize=6.5, color=C_INK2, ha="left", va="bottom")
        fig.tight_layout(rect=[0, 0.02, 1, 1.0 - 0.40 / (2.75 * nrow + 0.9)], h_pad=1.2)
        figs.append(fig)
    return figs


EXAMPLE_CSV_HEADER = (["example", "seed", "culture", "neuron"] + list(CSV_HEADER)
                      + ["dist_dipole3d_um", "row_fit_state", "row_peak_mV", "row_tau_decay_ms",
                         "row_dv_end_mV", "resim_dv_end_mV", "abs_diff_dv_end_mV",
                         "agrees_with_row"])


def campaign_protocol(cfg, rows, bump_ms=None, baseline_ms=100.0):
    """The protocol the campaign ran, so a re-simulated row can be compared with the row.

    Everything comes from config.py (the same getattr defaults culture_export uses) except:
      i0_uA     from the rows themselves (a campaign run with --i0 records it in every row)
      bump_ms   NOT recorded in a row. It must be the window the campaign used -- config
                bump_ms, or ESTIM_BUMP_MS if the job overrode it (full_tuned_run.pbs passes it
                through). A different window fits a different stretch of trace, and the
                cross-check will then flag the taus as differing.
      baseline  the one deliberate difference: 100 ms of pre-stimulus trace for the figure
                instead of the campaign's 5 ms. A tuned cell is flat at rest, so DeltaV_end
                is unchanged (measured: equal to 1e-6 mV); checked per example, not assumed.
    """
    i0s = sorted(set(round(_ff(r.get("i0_uA")), 3) for r in rows if r.get("i0_uA", "") != ""))
    if len(i0s) > 1:
        raise SystemExit("rows mix amplitudes %s uA -- filter the CSV to one amplitude" % i0s)
    return default_protocol(
        cfg, i0_uA=(i0s[0] if i0s else None),
        bump_ms=float(getattr(cfg, "bump_ms", 800.0) if bump_ms is None else bump_ms),
        bump_dt_ms=float(getattr(cfg, "bump_dt_ms", 0.5)),
        cvode_atol=float(getattr(cfg, "cvode_atol", 1e-6)),
        play_margin_ms=float(getattr(cfg, "play_margin_ms", 1.0)),
        baseline_ms=float(baseline_ms))


def main_from_csv(csv_path, n=24, select="stratified", cell_model="full_tuned",
                  processes=1, bump_ms=None, baseline_ms=100.0, early_floor=None, t0_ms=None,
                  per_page=12, detail_pages=True, select_seed=0, n_per_culture=None,
                  out_pdf="campaign_examples.pdf", out_csv="campaign_examples.csv",
                  verbose=True):
    """Pick `n` neurons from a campaign CSV, re-simulate them, and write a gallery + one
    detailed page each. See the section header for how the placements are recovered.

    bump_ms / early_floor / t0_ms default to config (bump_ms, bump_early_floor, bump_t0_ms)
    -- what the campaign used. Only the window is worth passing explicitly, when the campaign
    ran with ESTIM_BUMP_MS.
    """
    from config import CFG
    if early_floor is None:
        early_floor = float(getattr(CFG, "bump_early_floor", 0.25))
    if t0_ms is None:
        t0_ms = float(getattr(CFG, "bump_t0_ms", 0.0))
    here = os.path.dirname(os.path.abspath(__file__))
    src, rows = load_campaign_rows(csv_path)
    models = sorted(set(r.get("cell_model", "") for r in rows))
    if models and models != [cell_model]:
        raise SystemExit("%s holds cell_model %s but --cell-model is %r; the examples must be "
                         "re-simulated with the model that produced the rows"
                         % (src, models, cell_model))
    places, skipped = recover_placements(rows, CFG, n_per_culture)
    if verbose:
        print("[examples] %s: %d rows, placements regenerated for %d, skipped %d"
              % (src, len(rows), len(places), len(skipped)))
    if skipped and verbose:
        print("[examples] skipped rows do not match what culture_draws regenerates. Either "
              "config.py (morphologies, span, electrodes) is not the one the campaign ran with, "
              "or every culture was cut short -- then pass --neurons-per-culture")
    chosen = select_examples(places, n, select, select_seed)
    if not chosen:
        raise SystemExit("nothing to plot (no rows could be recovered or selected)")
    proto = campaign_protocol(CFG, [p["row"] for p in chosen], bump_ms, baseline_ms)
    if verbose:
        print("[examples] protocol: +/-%.1f uA, window %.0f ms after the pulse (MUST be the "
              "campaign's), grid %.2f ms, play margin %.1f ms, early floor %.2f, baseline %.0f ms"
              % (proto["i0_uA"], proto["bump_ms"], proto["bump_dt_ms"], proto["play_margin_ms"],
                 early_floor, proto["baseline_ms"]))
    jobs = [(k, p["morph"], p["layer"], p["x"], p["y"], p["theta"]) for k, p in enumerate(chosen)]
    sims = simulate_groups(jobs, cell_model, proto, processes=processes, verbose=verbose)

    items, out_rows = [], []
    for k, p in enumerate(chosen):
        meta, rec = sims[k]
        fit = analyse(rec, t0_ms=t0_ms, early_floor=early_floor)
        chk = campaign_check(rec, fit, p["row"])
        items.append(dict(idx=k + 1, morph=p["morph"], layer=p["layer"], rec=rec, fit=fit,
                          row=p["row"], check=chk, meta=meta))
        r = p["row"]
        out_rows.append([k + 1, r["seed"], r["culture"], r["neuron"]]
                        + _row(meta, proto, rec, fit)
                        + [r["dist_dipole3d_um"], chk["row_state"], r.get("dexp_peak_mV", ""),
                           r.get("dexp_tau_decay_ms", ""), r.get("deltaVm_end_phase2_mV", ""),
                           round(chk["dv_end_resim"], 6), "%.2e" % chk["d_dv_end"],
                           chk["agrees"]])

    pdf_path = out_pdf if os.path.isabs(out_pdf) else os.path.join(here, out_pdf)
    title = "%d neurons from %s (%s)" % (len(items), os.path.basename(src), select)
    with PdfPages(pdf_path) as pdf:
        for fig in plot_gallery(items, proto, per_page=per_page, title=title):
            pdf.savefig(fig)
            plt.close(fig)
        if detail_pages:
            for it in items:
                r, c = it["row"], it["check"]
                note = ("#%d  campaign row: seed %s, culture %s, neuron %s  |  row: fit %s, "
                        "peak %s mV, tau_decay %s ms, DeltaV_end %s mV  |  re-simulated "
                        "DeltaV_end differs by %.1e mV  -> %s"
                        % (it["idx"], r["seed"], r["culture"], r["neuron"], c["row_state"],
                           r.get("dexp_peak_mV", "")[:7], r.get("dexp_tau_decay_ms", "")[:7],
                           r.get("deltaVm_end_phase2_mV", "")[:8], c["d_dv_end"],
                           "agrees" if c["agrees"] else "DIFFERS"))
                pdf.savefig(plot_instance(it["rec"], it["fit"], it["meta"], proto,
                                          note_line=note))
                plt.close("all")

    csv_out = out_csv if os.path.isabs(out_csv) else os.path.join(here, out_csv)
    with open(csv_out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(EXAMPLE_CSV_HEADER)
        wr.writerows(out_rows)
    if verbose:
        n_ag = sum(it["check"]["agrees"] for it in items)
        worst = float(np.nanmax([it["check"]["d_dv_end"] for it in items]))
        print("[examples] re-simulation agrees with the campaign row for %d / %d "
              "(worst |DeltaV_end| difference %.2e mV)" % (n_ag, len(items), worst))
        print("done PDF:", pdf_path, "(%d gallery page(s) + %d detail pages)"
              % (int(np.ceil(len(items) / float(per_page))),
                 len(items) if detail_pages else 0))
        print("done CSV:", csv_out)
    return pdf_path, csv_out


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
    ap.add_argument("--from-csv", default=None,
                    help="a campaign CSV (or its results directory): plot neurons the "
                         "campaign actually simulated, re-simulated with the long window")
    ap.add_argument("--n-examples", type=int, default=24, help="with --from-csv (default 24)")
    ap.add_argument("--select", default="stratified", choices=list(SELECT_MODES),
                    help="with --from-csv: which rows (default: accepted fits across the "
                         "distance range + the rejected fits with the largest measured bump)")
    ap.add_argument("--processes", type=int, default=0,
                    help="with --from-csv: parallel simulation processes (0 = all cores "
                         "the job was given)")
    ap.add_argument("--neurons-per-culture", type=int, default=0,
                    help="with --from-csv: N the cultures were drawn with (0 = infer from "
                         "the rows; needed only if the walltime cut every culture short)")
    ap.add_argument("--per-page", type=int, default=12, help="gallery tiles per page")
    ap.add_argument("--no-detail-pages", action="store_true",
                    help="with --from-csv: gallery only")
    ap.add_argument("--n-instances", type=int, default=6, help="how many default placements")
    ap.add_argument("--i0-uA", type=float, default=None, help="default: config.i0_uA")
    ap.add_argument("--bump-ms", type=float, default=None,
                    help="window integrated after the pulse (default 800; with --from-csv, "
                         "config.bump_ms -- pass the campaign's value if the job overrode it)")
    ap.add_argument("--bump-dt-ms", type=float, default=0.5, help="fixed output grid for fits")
    ap.add_argument("--baseline-ms", type=float, default=100.0,
                    help="pre-stimulus baseline drawn on the figure, to show the cell sits "
                         "flat at rest. ~0.038 s per ms per run; SEGFAULTS between 150 and "
                         "300 ms on a 7 GB machine -- raise in steps.")
    ap.add_argument("--play-margin-ms", type=float, default=3.0,
                    help="fixed-dt tail after the pulse; sets the exact-DeltaV window")
    ap.add_argument("--cvode-atol", type=float, default=1e-6)
    ap.add_argument("--t0-ms", type=float, default=0.0,
                    help="anchor of the bump fit; 0 = end of phase 2 (the default, and the "
                         "only value for which the model is exactly specified)")
    ap.add_argument("--early-floor", type=float, default=None,
                    help="where the single-exponential fit stops, as a fraction of the peak. "
                         "The relaxation is multi-exponential, so tau_m moves with this; "
                         "early_t_1e_ms in the CSV is the window-free number. Default "
                         "0.25; with --from-csv, config.bump_early_floor.")
    ap.add_argument("--out-pdf", default="plot_vm_examples.pdf")
    ap.add_argument("--out-csv", default="plot_vm_examples.csv")
    a = ap.parse_args(argv)
    if a.from_csv:
        procs = a.processes or int(os.environ.get("PBS_NUM_PPN", "0") or 0) or \
            len(os.sched_getaffinity(0))
        if a.mode != "staged":
            ap.error("--from-csv re-fits with the campaign's staged fit; --fit joint does not "
                     "apply")
        return main_from_csv(a.from_csv, n=a.n_examples, select=a.select,
                             cell_model=a.cell_model, processes=procs, bump_ms=a.bump_ms,
                             baseline_ms=a.baseline_ms, early_floor=a.early_floor,
                             per_page=a.per_page, detail_pages=not a.no_detail_pages,
                             n_per_culture=(a.neurons_per_culture or None),
                             out_pdf=(a.out_pdf if a.out_pdf != "plot_vm_examples.pdf"
                                      else "campaign_examples.pdf"),
                             out_csv=(a.out_csv if a.out_csv != "plot_vm_examples.csv"
                                      else "campaign_examples.csv"))
    return main(cell_model=a.cell_model, mode=a.mode,
                placements=(parse_placements(a.placements) if a.placements else None),
                i0_uA=a.i0_uA, bump_ms=(800.0 if a.bump_ms is None else a.bump_ms),
                bump_dt_ms=a.bump_dt_ms,
                play_margin_ms=a.play_margin_ms, baseline_ms=a.baseline_ms,
                cvode_atol=a.cvode_atol, t0_ms=a.t0_ms,
                early_floor=(0.25 if a.early_floor is None else a.early_floor),
                n_instances=a.n_instances, out_pdf=a.out_pdf,
                out_csv=a.out_csv)


if __name__ == "__main__":
    _cli()
    sys.exit(0)
