"""bump_kinetics.py -- shape of the slow post-stimulus Ih bump. Pure numpy, no NEURON.

The soma-only / full models carry Ih (ehcn = -49.846 mV, i.e. inward at rest). A stimulus that
hyperpolarises the cell activates it with a time constant of tens of ms, and it deactivates again
at rest over hundreds of ms, so the membrane rides a slow depolarising bump after the pulse.
Measured on the tuned full-channel model: peak about +1 mV at ~100 ms at 150 um, still +0.09 mV
at 250 um, and removing Ih abolishes it (+1.039 -> -0.054 mV) while removing SK_E2 or Im does
nothing. It is therefore an Ih signature and worth recording per neuron.

Two separate things live in the trace and MUST NOT be conflated:

  * the DIRECT post-pulse polarisation, decaying with the membrane time constant over a few ms.
    Measured: one placement peaks at 6.5 ms from this alone.
  * the Ih bump proper, peaking near 100 ms.

`t_skip_ms` exists solely to step over the first. Everything here is measured on
DeltaV(t) = V_stim(t) - V_sham(t), with t = 0 at the END of phase 2, which is the same reference
the depolarisation / hyperpolarisation outcomes already use.

Quantities returned (column names match the existing culture_Pkinetics.csv):

    peak_mV           signed extremum of DeltaV in [t_skip_ms, t_search_ms]
    time_to_peak_ms   when it occurs
    tau_rise_ms       approach to the peak,   DeltaV(t) = peak - A*exp(-(t - t_skip)/tau_rise)
    tau_decay_ms      return towards rest,    DeltaV(t) = peak  *exp(-(t - t_peak)/tau_decay)
    sign              'depol' / 'hyperpol' / 'none'
    r2_rise, r2_decay quality of the two log-linear fits
    fit_ok            every gate below passed

Both taus come from a log-linear fit, which is deterministic and needs no optimiser or scipy --
the same approach the repo already uses in _loglin_tau.
"""
import numpy as np

# a fit is only reported when it has at least this many samples and this much dynamic range
MIN_FIT_POINTS = 5
MIN_FIT_DECADES = 0.35          # ln-range of the fitted variable, about a factor 1.4
FLOOR_FRACTION = 0.02           # ignore samples within 2 % of the asymptote (ln -> -inf there)
T_TOL_MS = 1e-9                 # a recorded sample meant to be exactly t0 can land 1e-15 below
                                # it (NEURON rebases by subtracting t_end), and a bare
                                # t >= t0 test would then silently drop the anchor sample


def _loglin_ab(t, y):
    """Fit ln y = a + b t on strictly positive y.

    Returns (tau, y0, r2, n) with tau = -1/b (so tau > 0 exactly when y decays) and
    y0 = exp(a) = the fitted value of y AT t = 0 of the vector passed in. Callers rebase t to
    the start of the decay first, so y0 is the fitted amplitude at that start.
    tau / y0 / r2 come back nan when the fit is rejected; n is always the sample count.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if int(m.sum()) < MIN_FIT_POINTS:
        return float("nan"), float("nan"), float("nan"), int(m.sum())
    t, y = t[m], np.log(y[m])
    if (y.max() - y.min()) < MIN_FIT_DECADES:
        return float("nan"), float("nan"), float("nan"), int(t.size)
    b, a = np.polyfit(t, y, 1)
    if b >= 0:
        return float("nan"), float("nan"), float("nan"), int(t.size)
    resid = y - (a + b * t)
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else float("nan")
    return float(-1.0 / b), float(np.exp(a)), r2, int(t.size)


def _loglin(t, y):
    """Fit ln y = a + b t on strictly positive y. Returns (tau = -1/b, r2, n). tau > 0 when y decays."""
    tau, _y0, r2, n = _loglin_ab(t, y)
    return tau, r2, n


def fit_ih_bump(t_ms, dv_mV, t_skip_ms=7.0, t_search_ms=400.0, min_amp_mV=0.05,
                min_r2=0.90):
    """Shape of the slow bump in DeltaV. `t_ms` is relative to the end of phase 2.

    Returns a dict with the columns listed in the module docstring. Never raises: a trace that
    cannot be fitted comes back with nan taus and fit_ok False, so a worker cannot die on one
    awkward neuron.
    """
    out = {"peak_mV": float("nan"), "time_to_peak_ms": float("nan"),
           "tau_rise_ms": float("nan"), "tau_decay_ms": float("nan"),
           "sign": "none", "r2_rise": float("nan"), "r2_decay": float("nan"),
           "fit_ok": 0, "n_rise": 0, "n_decay": 0}
    t = np.asarray(t_ms, float)
    dv = np.asarray(dv_mV, float)
    if t.size != dv.size or t.size < MIN_FIT_POINTS:
        return out
    good = np.isfinite(t) & np.isfinite(dv)
    t, dv = t[good], dv[good]
    win = (t >= float(t_skip_ms)) & (t <= float(t_search_ms))
    if int(win.sum()) < MIN_FIT_POINTS:
        return out
    tw, vw = t[win], dv[win]

    k = int(np.argmax(np.abs(vw)))          # signed extremum: largest magnitude, sign kept
    peak, t_peak = float(vw[k]), float(tw[k])
    out["peak_mV"], out["time_to_peak_ms"] = peak, t_peak
    if abs(peak) < float(min_amp_mV):
        return out                           # negligible: sign stays 'none', taus stay nan
    out["sign"] = "depol" if peak > 0 else "hyperpol"
    s = 1.0 if peak > 0 else -1.0
    floor = FLOOR_FRACTION * abs(peak)

    # rise: s*(peak - DeltaV) decays from its value at t_skip to 0 at the peak
    r = tw < t_peak
    if r.any():
        y = s * (peak - vw[r])
        keep = y > floor
        tau_r, r2_r, n_r = _loglin(tw[r][keep], y[keep])
        out["tau_rise_ms"], out["r2_rise"], out["n_rise"] = tau_r, r2_r, n_r

    # decay: s*DeltaV falls from the peak back towards 0
    d = tw > t_peak
    if d.any():
        y = s * vw[d]
        keep = y > floor
        tau_d, r2_d, n_d = _loglin(tw[d][keep] - t_peak, y[keep])
        out["tau_decay_ms"], out["r2_decay"], out["n_decay"] = tau_d, r2_d, n_d

    edge = (t_peak <= tw[0] + 1e-9) or (t_peak >= tw[-1] - 1e-9)   # peak pinned to a window edge
    out["fit_ok"] = int(
        (not edge)
        and np.isfinite(out["tau_rise_ms"]) and np.isfinite(out["tau_decay_ms"])
        and (out["r2_rise"] >= min_r2) and (out["r2_decay"] >= min_r2)
        and (out["tau_decay_ms"] < 0.8 * (float(t_search_ms) - t_peak) * 3.0)
    )
    return out


KINETICS_COLUMNS = ("bump_peak_mV", "bump_t_peak_ms", "bump_tau_rise_ms", "bump_tau_decay_ms",
                    "bump_sign", "bump_r2_rise", "bump_r2_decay", "bump_fit_ok")


def kinetics_row_values(fit):
    """The KINETICS_COLUMNS values of one fit_ih_bump() result, rounded for the CSV."""
    rnd = lambda x, n: (round(float(x), n) if np.isfinite(x) else "")
    return [rnd(fit["peak_mV"], 6), rnd(fit["time_to_peak_ms"], 3),
            rnd(fit["tau_rise_ms"], 3), rnd(fit["tau_decay_ms"], 3),
            fit["sign"], rnd(fit["r2_rise"], 4), rnd(fit["r2_decay"], 4), int(fit["fit_ok"])]


# ===========================================================================
# Figure-grade fits: the two curves drawn UNDER a soma Vm timecourse.
#
# fit_ih_bump() above is the CHEAP per-neuron summary used in a campaign: two independent
# log-linear legs anchored at the MEASURED peak. The two functions below are what
# plot_vm_examples.py draws, and they are a different, stronger statement about the trace:
# one genuine double exponential for the bump, pinned to zero at the instant the stimulus
# ends, plus one single exponential for the direct post-pulse polarisation.
#
# Why the pinning matters (the reason these exist):
#   the bump does NOT start from the resting potential. At the instant the pulse ends the
#   membrane is already displaced by the direct polarisation, so DeltaV(t0) != 0, and a fit
#   that ignores that lets the displacement leak into the amplitude and into both taus. So
#   the bump is fitted to the REBASED trace
#         y(u) = DeltaV(t0 + u) - DeltaV(t0),        u = t - t0 >= 0,
#   which is exactly 0 at u = 0, and every basis function of the model is also exactly 0 at
#   u = 0, so yhat(0) = 0 for EVERY parameter value. The pinning is structural, not a penalty.
#
# The decomposition this implements, with t0 = the end of phase 2 (the default):
#
#   DeltaV(t) =   P * exp(-t/tau_m)                                    direct polarisation
#               + Ab * ( exp(-t/tau_decay) - exp(-t/tau_rise) )        Ih bump, 0 at t = 0
#
#   so    DeltaV(0) = P            (the bump contributes nothing at t = 0)
#   and   y(u)      = Ab * ( exp(-u/tau_decay) - exp(-u/tau_rise) )  -  P * ( 1 - exp(-u/tau_m) )
#
# which is EXACTLY the model fitted below, with offset_mV = -P = -DeltaV(t0). Both halves are
# therefore identified: fit_exp_decay() measures (P, tau_m) on the first few ms, and
# fit_double_exp_from_zero() takes tau_m as a fixed input (tau_offset_ms) and recovers
# (Ab, tau_rise, tau_decay). Agreement between the fitted offset_mV and -DeltaV(t0) is a
# free consistency check, reported as offset_expected_mV.
# ===========================================================================

# --- single exponential (direct post-pulse polarisation) -------------------
EXPDECAY_FLOOR_FRACTION = 0.10      # stop fitting once |DeltaV| falls below this fraction of |peak|
EXPDECAY_UPTURN_FRAC = 0.01         # an increase of this x |peak| between samples ends the decay

# --- double exponential pinned at zero (the Ih bump) -----------------------
DEXP_TAU_RISE_BOUNDS_MS = (0.5, 400.0)
DEXP_TAU_DECAY_BOUNDS_MS = (5.0, 5000.0)
DEXP_MIN_RATIO = 1.05               # tau_decay_ms >= DEXP_MIN_RATIO * tau_rise_ms (see below)
DEXP_N_GRID = 28                    # points per axis of the log-spaced tau grid
DEXP_N_REFINE = 4                   # zoom passes after the first grid
DEXP_ZOOM = 0.25                    # log-width of the search box, x this per pass
DEXP_EDGE_TOL_DEC = 0.02            # a tau this close (in decades) to a bound counts as pinned
DEXP_MAX_TAU_WINDOW_MULT = 5.0      # reject tau_decay_ms beyond this x the fitted window length


def fit_exp_decay(t_ms, dv_mV, t0_ms=0.0, t_peak_search_ms=5.0, t_max_ms=40.0,
                  floor_fraction=EXPDECAY_FLOOR_FRACTION,
                  upturn_frac=EXPDECAY_UPTURN_FRAC,
                  min_amp_mV=0.02, min_r2=0.90):
    """Single-exponential decay of the DIRECT post-stimulus polarisation.

    Model, for t >= t_peak_ms:

        DeltaV(t) = amp_mV * exp( -(t - t_peak_ms) / tau_ms )

    with amp_mV signed (negative for a hyperpolarisation) and tau_ms > 0. The asymptote is
    fixed at DeltaV = 0, i.e. at the sham trajectory, which is where the direct polarisation
    does return; the slow Ih bump that grows on top of it is EXCLUDED from the fitted window
    rather than absorbed into a free offset (see the two stopping rules below).

    Arguments
      t_ms              time, same clock as `dv_mV`; in this pipeline t = 0 is the END of
                        phase 2 of the biphasic pulse.
      dv_mV             DeltaV(t) = Vm_stim(t) - Vm_sham(t), in mV.
      t0_ms             earliest time considered (the instant after the stimulus).
      t_peak_search_ms  the first post-stimulus extremum is the signed extremum of DeltaV over
                        [t0_ms, t0_ms + t_peak_search_ms]. For a monotone relaxation it lands
                        on t0_ms itself, which is the correct answer: the phase-2 lobe peaks
                        at the pulse boundary.
      t_max_ms          absolute upper edge of the decay window (same clock as t_ms).
      floor_fraction    stop once s * DeltaV <= floor_fraction * |peak|, s = sign(peak).
                        Below that the trace is bump plus noise, not decay. NOTE: on the real
                        traces the post-pulse relaxation is NOT a single exponential -- it is
                        a cable relaxation with a spectrum of time constants, and tau_ms grows
                        monotonically with this window (measured: 0.22 ms at 0.10, 0.31 at
                        0.30, 0.47 at 0.50, with r2 stuck near 0.91 throughout). So tau_ms is
                        only meaningful together with the window it came from, and t_1e_ms
                        below is the number to compare across placements.
      upturn_frac       also stop at the first sample where s * DeltaV RISES by more than
                        upturn_frac * |peak| from the previous sample -- that is the slow bump
                        pulling the trace back, and it must not enter this fit.
      min_amp_mV,
      min_r2            gates on fit_ok only; the fitted numbers are always returned.

    Returns a dict. Never raises.

        peak_mV, t_peak_ms    the measured first post-stimulus extremum (the point marked on
                              the timecourse), signed
        sign                  'depol' / 'hyperpol' / 'none'
        tau_ms                fitted decay time constant (nan if the fit was rejected). This
                              is the tau_m that fit_double_exp_from_zero() wants as
                              tau_offset_ms.
        amp_mV                fitted DeltaV at t_peak_ms, signed. It is the FITTED amplitude,
                              not the measured peak_mV; comparing the two is a sanity check.
        amp_over_peak         amp_mV / peak_mV. Exactly 1 for a true single exponential;
                              the further from 1, the less the trace is one over this window.
        t_1e_ms               MODEL-FREE: the time from t_peak_ms for |DeltaV| to fall to
                              |peak|/e, read off the data by linear interpolation between the
                              bracketing samples. For a true single exponential it equals
                              tau_ms; unlike tau_ms it does not depend on floor_fraction, so
                              it is the comparable number when the relaxation is not one.
        r2, n                 quality and sample count of the log-linear fit
        t_fit_lo_ms,
        t_fit_hi_ms           the window actually fitted
        fit_ok                1 when every gate above passed
    """
    nan = float("nan")
    out = {"peak_mV": nan, "t_peak_ms": nan, "sign": "none", "tau_ms": nan, "amp_mV": nan,
           "amp_over_peak": nan, "t_1e_ms": nan, "r2": nan, "n": 0, "t_fit_lo_ms": nan,
           "t_fit_hi_ms": nan, "fit_ok": 0}
    t = np.asarray(t_ms, float)
    dv = np.asarray(dv_mV, float)
    if t.size != dv.size or t.size < 2:
        return out
    good = np.isfinite(t) & np.isfinite(dv)
    t, dv = t[good], dv[good]
    if t.size < 2:
        return out

    win = (t >= float(t0_ms) - T_TOL_MS) & (t <= float(t0_ms) + float(t_peak_search_ms))
    if not win.any():
        return out
    tw, vw = t[win], dv[win]
    k = int(np.argmax(np.abs(vw)))
    peak, t_peak = float(vw[k]), float(tw[k])
    out["peak_mV"], out["t_peak_ms"] = peak, t_peak
    if abs(peak) < float(min_amp_mV):
        return out
    s = 1.0 if peak > 0 else -1.0
    out["sign"] = "depol" if peak > 0 else "hyperpol"

    d = (t >= t_peak - T_TOL_MS) & (t <= float(t_max_ms))
    if int(d.sum()) < MIN_FIT_POINTS:
        return out
    td, y = t[d], s * dv[d]

    # model-free 1/e time, read off the data before any window is chosen
    thr = abs(peak) / np.e
    xs = np.nonzero(y <= thr)[0]
    if xs.size and int(xs[0]) > 0:
        i = int(xs[0])
        y0, y1 = float(y[i - 1]), float(y[i])
        f = 0.0 if y0 == y1 else (y0 - thr) / (y0 - y1)
        out["t_1e_ms"] = float(td[i - 1]) + f * (float(td[i]) - float(td[i - 1])) - t_peak

    # 1) stop at the first real upturn -- that is the bump, not the decay
    rise = np.nonzero(np.diff(y) > float(upturn_frac) * abs(peak))[0]
    if rise.size:
        td, y = td[:int(rise[0]) + 1], y[:int(rise[0]) + 1]
    # 2) stop at the floor (y decays, so the first crossing ends the usable run)
    below = np.nonzero(y <= float(floor_fraction) * abs(peak))[0]
    if below.size:
        td, y = td[:int(below[0])], y[:int(below[0])]
    if td.size < MIN_FIT_POINTS:
        out["t_fit_lo_ms"] = t_peak
        out["t_fit_hi_ms"] = float(td[-1]) if td.size else nan
        return out

    tau, y0, r2, n = _loglin_ab(td - t_peak, y)
    out["tau_ms"], out["r2"], out["n"] = tau, r2, int(n)
    out["amp_mV"] = s * y0 if np.isfinite(y0) else nan
    out["amp_over_peak"] = (out["amp_mV"] / peak) if np.isfinite(out["amp_mV"]) else nan
    out["t_fit_lo_ms"], out["t_fit_hi_ms"] = t_peak, float(td[-1])
    out["fit_ok"] = int(np.isfinite(tau) and np.isfinite(r2) and (r2 >= float(min_r2))
                        and (n >= MIN_FIT_POINTS))
    return out


def eval_exp_decay(t_ms, fit):
    """The fitted curve of fit_exp_decay(), in DeltaV (mV). nan before t_peak_ms."""
    t = np.asarray(t_ms, float)
    out = np.full(t.shape, float("nan"))
    tau, amp, t_peak = fit.get("tau_ms"), fit.get("amp_mV"), fit.get("t_peak_ms")
    if not (np.isfinite(tau) and np.isfinite(amp) and np.isfinite(t_peak)):
        return out
    m = t >= float(t_peak) - T_TOL_MS
    out[m] = float(amp) * np.exp(-(t[m] - float(t_peak)) / float(tau))
    return out


def _dexp_design(u, tau_rise_ms, tau_decay_ms, free_offset, tau_offset_ms=None):
    """Design matrix of the pinned double exponential at FIXED taus.

    Columns -- both are exactly 0 at u = 0, which is what pins yhat(0) = 0:

        b1(u) = exp(-u/tau_decay_ms) - exp(-u/tau_rise_ms)      scaled by amp_mV
        b2(u) = 1 - exp(-u/tau_offset_ms)                       scaled by offset_mV

    tau_offset_ms is HELD FIXED (it is the membrane time constant of the direct polarisation,
    measured independently by fit_exp_decay), so only the two bump taus are ever searched.
    Passing None falls back to tau_rise_ms, which keeps the column well defined but makes the
    model only approximately right whenever tau_m and tau_rise differ.
    """
    cols = [np.exp(-u / float(tau_decay_ms)) - np.exp(-u / float(tau_rise_ms))]
    if free_offset:
        to = float(tau_rise_ms) if tau_offset_ms is None else float(tau_offset_ms)
        cols.append(1.0 - np.exp(-u / to))
    return np.column_stack(cols)


def _dexp_solve(u, y, tau_rise_ms, tau_decay_ms, free_offset, tau_offset_ms=None):
    """Least-squares (amp_mV[, offset_mV]) at fixed taus. Returns (sse, coef)."""
    X = _dexp_design(u, tau_rise_ms, tau_decay_ms, free_offset, tau_offset_ms)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X.dot(coef)
    return float(r.dot(r)), coef


def _dexp_search(u, y, free_offset, bounds_rise, bounds_decay, n_grid, n_refine, zoom,
                 tau_offset_ms=None):
    """Grid search over (tau_rise_ms, tau_decay_ms), refined by zooming on the winner.

    Only the two bump taus are searched: at fixed taus the model is LINEAR in
    (amp_mV, offset_mV), so those come from one least-squares solve per candidate. The grid
    is log-spaced because the taus span decades, and each pass re-centres a box `zoom` times
    narrower (in log space) on the current winner. Deterministic and numpy-only -- no
    optimiser, no scipy, no random restarts.

    Returns (sse, coef, tau_rise_ms, tau_decay_ms); coef is None if no candidate was valid.
    """
    lo_r, hi_r = np.log10(float(bounds_rise[0])), np.log10(float(bounds_rise[1]))
    lo_d, hi_d = np.log10(float(bounds_decay[0])), np.log10(float(bounds_decay[1]))
    box = [lo_r, hi_r, lo_d, hi_d]
    best = (float("inf"), None, float("nan"), float("nan"))
    for _ in range(int(n_refine) + 1):
        TR = np.logspace(box[0], box[1], int(n_grid))
        TD = np.logspace(box[2], box[3], int(n_grid))
        for tr in TR:
            for td in TD:
                if td < DEXP_MIN_RATIO * tr:
                    continue                      # ordering convention, see fit docstring
                sse, coef = _dexp_solve(u, y, tr, td, free_offset, tau_offset_ms)
                if sse < best[0]:
                    best = (sse, coef, float(tr), float(td))
        if best[1] is None:
            return best
        hw_r = 0.5 * (box[1] - box[0]) * float(zoom)
        hw_d = 0.5 * (box[3] - box[2]) * float(zoom)
        cr, cd = np.log10(best[2]), np.log10(best[3])
        box = [max(lo_r, cr - hw_r), min(hi_r, cr + hw_r),
               max(lo_d, cd - hw_d), min(hi_d, cd + hw_d)]
    return best


def fit_double_exp_from_zero(t_ms, dv_mV, t0_ms=0.0, t_end_ms=None, free_offset=True,
                             tau_offset_ms=None,
                             bounds_rise_ms=DEXP_TAU_RISE_BOUNDS_MS,
                             bounds_decay_ms=DEXP_TAU_DECAY_BOUNDS_MS,
                             n_grid=DEXP_N_GRID, n_refine=DEXP_N_REFINE, zoom=DEXP_ZOOM,
                             min_amp_mV=0.05, min_r2=0.90):
    """Double exponential for the slow Ih bump, PINNED TO ZERO at t0_ms.

    Everything is fitted to the REBASED trace

        y(u) = DeltaV(t0_ms + u) - DeltaV(t0_ms),        u = t - t0_ms >= 0,

    which is exactly 0 at u = 0, against the model

        yhat(u) = amp_mV    * ( exp(-u/tau_decay_ms) - exp(-u/tau_rise_ms) )
                + offset_mV * ( 1 - exp(-u/tau_offset_ms) )

    Every basis function vanishes at u = 0, so yhat(0) = 0 for EVERY value of the parameters.

      amp_mV          signed bump amplitude scale (negative for a hyperpolarising bump).
      offset_mV       weight of the residual direct polarisation. yhat(u -> infinity) =
                      offset_mV, because the rebased trace does not return to 0: it returns
                      to -DeltaV(t0_ms), since DeltaV itself decays back to the sham. With the
                      module-header decomposition, offset_mV should come out equal to
                      -DeltaV(t0_ms); that value is reported as offset_expected_mV and the two
                      agreeing is a free check on the whole fit. free_offset=False forces
                      offset_mV = 0 and gives the pure two-exponential -- correct only when
                      DeltaV(t0_ms) is already negligible, otherwise the residual is absorbed
                      by tau_decay_ms and the decay tau comes out too long.
      tau_offset_ms   the membrane time constant of that residual. HELD FIXED, not searched:
                      pass fit_exp_decay(...)["tau_ms"] measured on the same trace. None falls
                      back to tau_rise_ms (see _dexp_design).
      tau_rise_ms,
      tau_decay_ms    by convention tau_decay_ms >= DEXP_MIN_RATIO * tau_rise_ms. This costs
                      no generality: swapping the two taus only flips the sign of the first
                      basis function, which amp_mV absorbs, so every model is still reachable
                      -- the restriction just fixes which of the two is called the rise.

    In DeltaV units the fitted curve is DeltaV_hat(t) = DeltaV(t0_ms) + yhat(t - t0_ms);
    eval_double_exp_from_zero() returns exactly that, ready to overlay on the trace.

    Arguments
      t_ms, dv_mV   the trace; t = 0 is the END of phase 2 in this pipeline.
      t0_ms         the instant after the stimulus at which the bump fit starts, and the point
                    whose DeltaV is DEFINED to be 0. The default 0.0 is the end of the pulse,
                    which is also where the physical bump starts -- that is what makes the
                    model above exactly specified rather than approximate. Raising t0_ms
                    (e.g. to 7 ms, the t_skip_ms fit_ih_bump uses) steps over the first few ms
                    at the cost of that exactness, because a bump sampled from t0_ms > 0 no
                    longer has equal and opposite weights on its two exponentials.
      t_end_ms      upper edge of the fitted window; None = the end of the trace.
      min_amp_mV,
      min_r2        gates on fit_ok only.

    Returns a dict. Never raises.

        t0_ms, dv_t0_mV        the anchor, and the DeltaV subtracted there
        amp_mV, offset_mV      fitted linear parameters (offset_mV is 0.0 when free_offset
                               is False)
        offset_expected_mV     -dv_t0_mV, what offset_mV should equal (consistency check)
        tau_rise_ms,
        tau_decay_ms           fitted bump time constants
        tau_offset_ms          the fixed residual time constant actually used
        peak_mV, t_peak_ms     extremum of the BUMP COMPONENT alone, i.e. of
                               amp_mV*(exp(-u/tau_decay) - exp(-u/tau_rise)), excluding the
                               offset term. This is the bump amplitude. It is NOT the extremum
                               of the whole rebased model: that one is dominated by the
                               asymptote offset_mV = -DeltaV(t0), which is the residual
                               polarisation leaving, not the bump.
        peak_dv_mV             the same point read off the DeltaV axis (= dv_t0_mV + the whole
                               model there), i.e. where the marker sits on a DeltaV plot
        data_peak_mV,
        data_t_peak_ms         the MEASURED extremum of DeltaV over [t0 + 5*tau_offset, t_end]
                               -- once the residual polarisation is gone DeltaV IS the bump,
                               so this is the observed bump peak and compares directly with
                               peak_mV
        sign                   'depol' / 'hyperpol' / 'none', from peak_mV
        r2, rmse_mV, n         quality and sample count
        t_fit_lo_ms,
        t_fit_hi_ms            the window actually fitted
        free_offset            echo of the argument (0/1)
        fit_ok                 1 when every gate passed: r2, amplitude, neither bump tau
                               pinned to a search bound, the model peak strictly inside the
                               window, and tau_decay_ms within DEXP_MAX_TAU_WINDOW_MULT
                               window lengths
    """
    nan = float("nan")
    out = {"t0_ms": float(t0_ms), "dv_t0_mV": nan, "amp_mV": nan, "offset_mV": nan,
           "offset_expected_mV": nan, "tau_rise_ms": nan, "tau_decay_ms": nan,
           "tau_offset_ms": nan, "peak_mV": nan, "t_peak_ms": nan, "peak_dv_mV": nan,
           "data_peak_mV": nan, "data_t_peak_ms": nan, "sign": "none", "r2": nan,
           "rmse_mV": nan, "n": 0, "t_fit_lo_ms": nan, "t_fit_hi_ms": nan,
           "free_offset": int(bool(free_offset)), "fit_ok": 0}
    t = np.asarray(t_ms, float)
    dv = np.asarray(dv_mV, float)
    if t.size != dv.size or t.size < MIN_FIT_POINTS:
        return out
    good = np.isfinite(t) & np.isfinite(dv)
    t, dv = t[good], dv[good]
    if t.size < MIN_FIT_POINTS:
        return out
    hi = float(t[-1]) if t_end_ms is None else float(t_end_ms)
    win = (t >= float(t0_ms) - T_TOL_MS) & (t <= hi)
    if int(win.sum()) < MIN_FIT_POINTS:
        return out
    tw, vw = t[win], dv[win]

    # the anchor: DeltaV at the first sample at or after t0_ms, defined to be 0 from here on
    dv_t0 = float(vw[0])
    out["dv_t0_mV"], out["offset_expected_mV"] = dv_t0, -dv_t0
    out["t_fit_lo_ms"], out["t_fit_hi_ms"] = float(tw[0]), float(tw[-1])
    u = tw - float(tw[0])
    y = vw - dv_t0                                   # y(0) = 0 exactly

    tau_off = None if tau_offset_ms is None or not np.isfinite(tau_offset_ms) \
        else float(tau_offset_ms)
    sse, coef, tau_r, tau_d = _dexp_search(u, y, bool(free_offset), bounds_rise_ms,
                                           bounds_decay_ms, n_grid, n_refine, zoom, tau_off)
    if coef is None:
        return out
    amp = float(coef[0])
    off = float(coef[1]) if bool(free_offset) else 0.0
    out["amp_mV"], out["offset_mV"] = amp, off
    out["tau_rise_ms"], out["tau_decay_ms"] = float(tau_r), float(tau_d)
    out["tau_offset_ms"] = float(tau_r) if tau_off is None else tau_off
    out["n"] = int(u.size)
    sst = float(np.sum((y - y.mean()) ** 2))
    out["r2"] = (1.0 - sse / sst) if sst > 0 else nan
    out["rmse_mV"] = float(np.sqrt(sse / u.size))

    # measured bump peak: once the residual polarisation is gone, DeltaV IS the bump
    lag = 5.0 * (tau_off if tau_off is not None else tau_r)
    late = u >= min(lag, 0.5 * float(u[-1]))
    if late.any():
        k = int(np.argmax(np.abs(vw[late])))
        out["data_peak_mV"] = float(vw[late][k])
        out["data_t_peak_ms"] = float(tw[late][k])

    # extremum of the BUMP COMPONENT alone, on a dense grid (see the docstring: the extremum
    # of the whole rebased model would just find the offset asymptote)
    ud = np.linspace(0.0, float(u[-1]), 4001)
    bd = amp * (np.exp(-ud / tau_d) - np.exp(-ud / tau_r))
    j = int(np.argmax(np.abs(bd)))
    yd_j = float(_dexp_design(ud[j:j + 1], tau_r, tau_d, bool(free_offset), tau_off).dot(coef)[0])
    out["peak_mV"] = float(bd[j])
    out["t_peak_ms"] = float(tw[0]) + float(ud[j])
    out["peak_dv_mV"] = dv_t0 + yd_j
    if abs(out["peak_mV"]) >= float(min_amp_mV):
        out["sign"] = "depol" if out["peak_mV"] > 0 else "hyperpol"

    lo_r, hi_r = np.log10(float(bounds_rise_ms[0])), np.log10(float(bounds_rise_ms[1]))
    lo_d, hi_d = np.log10(float(bounds_decay_ms[0])), np.log10(float(bounds_decay_ms[1]))
    pinned = (min(abs(np.log10(tau_r) - lo_r), abs(np.log10(tau_r) - hi_r)) < DEXP_EDGE_TOL_DEC
              or min(abs(np.log10(tau_d) - lo_d), abs(np.log10(tau_d) - hi_d)) < DEXP_EDGE_TOL_DEC)
    span = float(tw[-1]) - float(tw[0])
    inside = (ud[j] > 1e-9) and (ud[j] < float(u[-1]) - 1e-9)
    out["fit_ok"] = int((not pinned) and inside
                        and np.isfinite(out["r2"]) and (out["r2"] >= float(min_r2))
                        and (abs(out["peak_mV"]) >= float(min_amp_mV))
                        and (tau_d <= DEXP_MAX_TAU_WINDOW_MULT * span))
    return out


def eval_bump_component(t_ms, fit):
    """Just the bump term of fit_double_exp_from_zero() / fit_post_pulse(): 
    amp_mV*(exp(-u/tau_decay) - exp(-u/tau_rise)), with the offset term left out. nan before
    t0. This is what the bump panel of the figure draws."""
    t = np.asarray(t_ms, float)
    out = np.full(t.shape, float("nan"))
    tau_r, tau_d, amp = fit.get("tau_rise_ms"), fit.get("tau_decay_ms"), fit.get("amp_mV")
    t_lo = fit.get("t_fit_lo_ms")
    vals = (tau_r, tau_d, amp, t_lo)
    if not all(v is not None and np.isfinite(v) for v in vals):
        return out
    m = t >= float(t_lo) - T_TOL_MS
    u = t[m] - float(t_lo)
    out[m] = float(amp) * (np.exp(-u / float(tau_d)) - np.exp(-u / float(tau_r)))
    return out


def eval_offset_component(t_ms, fit):
    """Just the residual-polarisation term of fit_double_exp_from_zero(), in DeltaV units:
    DeltaV(t0) + offset_mV*(1 - exp(-u/tau_offset)). nan before t0."""
    t = np.asarray(t_ms, float)
    out = np.full(t.shape, float("nan"))
    off, tau_o = fit.get("offset_mV"), fit.get("tau_offset_ms")
    t_lo, dv_t0 = fit.get("t_fit_lo_ms"), fit.get("dv_t0_mV")
    vals = (off, tau_o, t_lo, dv_t0)
    if not all(v is not None and np.isfinite(v) for v in vals):
        return out
    m = t >= float(t_lo) - T_TOL_MS
    u = t[m] - float(t_lo)
    out[m] = float(dv_t0) + float(off) * (1.0 - np.exp(-u / float(tau_o)))
    return out


def eval_double_exp_from_zero(t_ms, fit):
    """The fitted curve of fit_double_exp_from_zero(), in DeltaV (mV) -- i.e. WITH the
    dv_t0_mV offset added back, so it overlays directly on the trace. nan before t0."""
    t = np.asarray(t_ms, float)
    out = np.full(t.shape, float("nan"))
    tau_r, tau_d = fit.get("tau_rise_ms"), fit.get("tau_decay_ms")
    tau_o = fit.get("tau_offset_ms")
    amp, off = fit.get("amp_mV"), fit.get("offset_mV")
    t_lo, dv_t0 = fit.get("t_fit_lo_ms"), fit.get("dv_t0_mV")
    vals = (tau_r, tau_d, tau_o, amp, off, t_lo, dv_t0)
    if not all(v is not None and np.isfinite(v) for v in vals):
        return out
    m = t >= float(t_lo) - T_TOL_MS
    u = t[m] - float(t_lo)
    coef = np.array([float(amp), float(off)])
    out[m] = float(dv_t0) + _dexp_design(u, tau_r, tau_d, True, tau_o).dot(coef)
    return out


EXPDECAY_COLUMNS = ("early_peak_mV", "early_t_peak_ms", "early_sign", "early_tau_ms",
                    "early_t_1e_ms", "early_amp_mV", "early_amp_over_peak",
                    "early_t_fit_hi_ms", "early_r2", "early_fit_ok")

DEXP_COLUMNS = ("dexp_t0_ms", "dexp_dv_t0_mV", "dexp_peak_mV", "dexp_t_peak_ms",
                "dexp_tau_rise_ms", "dexp_tau_decay_ms", "dexp_tau_offset_ms",
                "dexp_amp_mV", "dexp_offset_mV", "dexp_sign", "dexp_r2", "dexp_fit_ok")


def _rnd(x, n):
    return round(float(x), n) if np.isfinite(x) else ""


def expdecay_row_values(fit):
    """The EXPDECAY_COLUMNS values of one fit_exp_decay() result, rounded for a CSV."""
    return [_rnd(fit["peak_mV"], 6), _rnd(fit["t_peak_ms"], 4), fit["sign"],
            _rnd(fit["tau_ms"], 4), _rnd(fit.get("t_1e_ms", float("nan")), 4),
            _rnd(fit["amp_mV"], 6), _rnd(fit.get("amp_over_peak", float("nan")), 4),
            _rnd(fit["t_fit_hi_ms"], 4), _rnd(fit["r2"], 4), int(fit["fit_ok"])]


def dexp_row_values(fit):
    """The DEXP_COLUMNS values of one fit_double_exp_from_zero() result, rounded for a CSV."""
    return [_rnd(fit["t0_ms"], 3), _rnd(fit["dv_t0_mV"], 6), _rnd(fit["peak_mV"], 6),
            _rnd(fit["t_peak_ms"], 3), _rnd(fit["tau_rise_ms"], 3),
            _rnd(fit["tau_decay_ms"], 3), _rnd(fit["tau_offset_ms"], 3),
            _rnd(fit["amp_mV"], 6), _rnd(fit["offset_mV"], 6), fit["sign"],
            _rnd(fit["r2"], 4), int(fit["fit_ok"])]


# ===========================================================================
# The JOINT fit -- what the figure actually draws.
#
# fit_exp_decay() and fit_double_exp_from_zero() above each look at one half of the trace and
# are honest about it, but they cannot be trusted together: the two components overlap in
# time. The direct polarisation is still decaying while the bump is rising, so the early
# single-exponential window contains part of the bump and its tau comes out too FAST, and the
# bump window starts on a trace the polarisation has not finished leaving. Measured on an
# exactly-known synthetic trace (tau_m = 12 ms, tau_rise = 25 ms, tau_decay = 180 ms), fitting
# the halves separately returns tau_m = 6.1 ms and tau_rise = 41.8 ms -- a factor of two out
# on both, with r2 = 0.97 in each panel. A good r2 on half a trace is not evidence.
#
# fit_post_pulse() fits ONE model to the whole post-pulse window instead:
#
#   DeltaV_hat(t) = P  * exp(-u/tau_m)                                    u = t - t0 >= 0
#                 + Ab * ( exp(-u/tau_decay) - exp(-u/tau_rise) )
#
# The bump term is exactly 0 at u = 0 (this is the pinning: the bump contributes nothing at
# the instant the stimulus ends, so relative to that instant it starts from zero), hence
# DeltaV_hat(t0) = P: the whole displacement at the pulse end belongs to the direct
# polarisation, which is the statement "take the membrane potential at t0 as the zero of the
# double exponential". Both panels of the figure are then views of this one fit:
#
#   early panel : DeltaV(t) - Ab*(exp(-u/tau_decay) - exp(-u/tau_rise))  vs  P*exp(-u/tau_m)
#   bump  panel : DeltaV(t) - P*exp(-u/tau_m)                            vs  the bump term
#
# Linear in (P, Ab) at fixed taus, so only (tau_m, tau_rise, tau_decay) are searched -- a
# log-spaced 3-D grid, zoomed on the winner. numpy only, deterministic, no scipy.
# ===========================================================================

# The direct post-pulse relaxation measured on the tuned full model is ~0.2 ms -- the soma
# discharges by redistributing charge into the arbour, not by charging the whole cell, so this
# is far faster than a whole-cell membrane time constant. The lower bound must sit well below
# it or the search pins there and fit_ok is (correctly) 0.
POSTPULSE_TAU_M_BOUNDS_MS = (0.02, 200.0)
POSTPULSE_N_GRID_M = 20             # tau_m points per pass
POSTPULSE_N_GRID = 24               # tau_rise / tau_decay points per pass
POSTPULSE_N_REFINE = 5
POSTPULSE_ZOOM = 0.30


def _solve2(b1, b2, y, yy):
    """Least squares for y ~ c1*b1 + c2*b2 by 2x2 normal equations. Returns (sse, c1, c2).

    Exact for two columns and about 4x faster than np.linalg.lstsq, which matters inside a
    3-D grid search. Falls back to lstsq when the Gram matrix is near-singular (which happens
    when the two columns become nearly parallel, e.g. tau_decay -> tau_rise makes b2 -> 0).
    """
    a11 = float(b1.dot(b1)); a12 = float(b1.dot(b2)); a22 = float(b2.dot(b2))
    c1v = float(b1.dot(y)); c2v = float(b2.dot(y))
    det = a11 * a22 - a12 * a12
    if not np.isfinite(det) or det <= 1e-12 * a11 * a22:
        X = np.column_stack([b1, b2])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X.dot(coef)
        return float(r.dot(r)), float(coef[0]), float(coef[1])
    c1 = (a22 * c1v - a12 * c2v) / det
    c2 = (a11 * c2v - a12 * c1v) / det
    return max(0.0, yy - (c1 * c1v + c2 * c2v)), c1, c2


def _postpulse_search(u, y, bounds_m, bounds_rise, bounds_decay,
                      n_grid_m, n_grid, n_refine, zoom, enforce_tau_order=True):
    """3-D log grid over (tau_m, tau_rise, tau_decay), refined by zooming on the winner.

    Two candidates are tracked at once: the best with tau_m <= tau_rise and the best with
    tau_m > tau_rise. The model

        P*exp(-u/tau_m) + Ab*exp(-u/tau_decay) - Ab*exp(-u/tau_rise)

    is a sum of three exponentials in which the tau_m and tau_rise terms BOTH decay and have
    free (P) and tied (-Ab) amplitudes, so whenever |P| and |Ab| are comparable and tau_m and
    tau_rise are within a factor of a few, exchanging their roles produces an almost identical
    curve. This is the classic ill-conditioning of multi-exponential fitting, not a bug: on a
    synthetic trace with tau_m = 12, tau_rise = 25 the unconstrained search returns
    tau_m = 21.7, tau_rise = 10.3 at r2 = 0.99999.

    enforce_tau_order=True therefore restricts the reported fit to tau_m <= tau_rise, which is
    a PHYSICAL constraint for this preparation -- the soma charges with the passive membrane
    time constant (~10-20 ms here) while the bump is driven by Ih gating, whose mTau is 43 ms
    at -90 mV and 343 ms at -73.5 mV, i.e. always slower. It is an identifiability constraint
    imposed by hand, not something the data decided, so the loser is returned as well and
    fit_post_pulse reports swap_dsse_frac = (sse_swapped - sse)/sse. Near 0 means the data do
    NOT separate the two components and neither labelling should be believed; large means the
    ordering is supported by the trace itself.

    Returns (sse, P, Ab, tau_m, tau_rise, tau_decay, sse_other).
    """
    yy = float(y.dot(y))
    lb = [np.log10(float(b[0])) for b in (bounds_m, bounds_rise, bounds_decay)]
    ub = [np.log10(float(b[1])) for b in (bounds_m, bounds_rise, bounds_decay)]
    box = list(lb) + list(ub)                       # [lo_m, lo_r, lo_d, hi_m, hi_r, hi_d]
    nan = float("nan")
    keep = (float("inf"), nan, nan, nan, nan, nan)  # tau_m <= tau_rise
    other = float("inf")                            # best tau_m > tau_rise, for the diagnostic
    for _ in range(int(n_refine) + 1):
        TM = np.logspace(box[0], box[3], int(n_grid_m))
        TR = np.logspace(box[1], box[4], int(n_grid))
        TD = np.logspace(box[2], box[5], int(n_grid))
        EM = {tm: np.exp(-u / tm) for tm in TM}      # each exp evaluated once per pass
        ER = {tr: np.exp(-u / tr) for tr in TR}
        ED = {td: np.exp(-u / td) for td in TD}
        for tr in TR:
            for td in TD:
                if td < DEXP_MIN_RATIO * tr:
                    continue                         # ordering convention on the bump pair
                b2 = ED[td] - ER[tr]
                for tm in TM:
                    sse, P, Ab = _solve2(EM[tm], b2, y, yy)
                    if bool(enforce_tau_order) and tm > tr:
                        if sse < other:
                            other = sse
                        continue
                    if sse < keep[0]:
                        keep = (sse, P, Ab, float(tm), float(tr), float(td))
        if not np.isfinite(keep[1]):
            return keep + (other,)
        hw = [0.5 * (box[3 + i] - box[i]) * float(zoom) for i in range(3)]
        ctr = [np.log10(keep[3]), np.log10(keep[4]), np.log10(keep[5])]
        box = [max(lb[i], ctr[i] - hw[i]) for i in range(3)] + \
              [min(ub[i], ctr[i] + hw[i]) for i in range(3)]
    return keep + (other,)


def fit_post_pulse(t_ms, dv_mV, t0_ms=0.0, t_end_ms=None,
                   bounds_m_ms=POSTPULSE_TAU_M_BOUNDS_MS,
                   bounds_rise_ms=DEXP_TAU_RISE_BOUNDS_MS,
                   bounds_decay_ms=DEXP_TAU_DECAY_BOUNDS_MS,
                   n_grid_m=POSTPULSE_N_GRID_M, n_grid=POSTPULSE_N_GRID,
                   n_refine=POSTPULSE_N_REFINE, zoom=POSTPULSE_ZOOM,
                   enforce_tau_order=True, min_amp_mV=0.05, min_r2=0.90):
    """Joint fit of the two post-pulse components. See the section header for the model.

        DeltaV_hat(t) = P  * exp(-u/tau_m)
                      + Ab * ( exp(-u/tau_decay) - exp(-u/tau_rise) ),     u = t - t0_ms >= 0

    The second term is exactly 0 at u = 0 for every parameter value, so the bump is pinned to
    zero at t0_ms and DeltaV_hat(t0_ms) = P. By convention tau_decay >= DEXP_MIN_RATIO *
    tau_rise; swapping the two only flips the sign of the bracket, which Ab absorbs, so no
    model is lost -- the restriction just fixes which tau is called the rise.

    Arguments
      t_ms, dv_mV   the trace; t = 0 is the END of phase 2 in this pipeline.
      t0_ms         origin of both components, and the instant whose DeltaV is assigned
                    entirely to the direct polarisation. Default 0.0 = the end of the pulse.
      t_end_ms      upper edge of the fitted window; None = the end of the trace.
      bounds_*_ms   log-search bounds; a tau landing on one of them sets fit_ok = 0.
      enforce_tau_order
                    restrict the fit to tau_m <= tau_rise. On by default and strongly
                    recommended -- see _postpulse_search for why the unconstrained problem is
                    ill-conditioned and what swap_dsse_frac reports about it.
      min_amp_mV    the bump component must reach this |peak| for fit_ok.
      min_r2        gate on fit_ok.

    Returns a dict. Never raises.

        early   a fit_exp_decay()-shaped dict for the polarisation component, so
                eval_exp_decay(t, res["early"]) draws P*exp(-(t-t0)/tau_m) directly.
        bump    a fit_double_exp_from_zero()-shaped dict for the bump component, with
                dv_t0_mV = 0 and offset_mV = 0, so eval_double_exp_from_zero(t, res["bump"])
                draws Ab*(exp(-u/tau_decay) - exp(-u/tau_rise)) directly.
        P_mV, tau_m_ms, Ab_mV, tau_rise_ms, tau_decay_ms      the five parameters
        dv_t0_mV                DeltaV of the DATA at t0_ms. P_mV is the model's value there;
                                the two agreeing is a check on the fit, not an assumption.
        bump_peak_mV,
        bump_t_peak_ms          extremum of the bump COMPONENT (this is the bump amplitude)
        data_peak_mV,
        data_t_peak_ms          extremum of the DATA over [t0 + 3*tau_m, t_end], i.e. the
                                measured bump peak once the polarisation has essentially gone
                                -- the point marked on the timecourse
        r2, rmse_mV, n          over the whole fitted window
        swap_dsse_frac          (sse with tau_m > tau_rise - sse) / sse. How much WORSE the
                                exchanged labelling fits. Near 0 = the trace does not separate
                                the polarisation from the bump rise and neither set of taus
                                should be quoted; the bump peak and tau_decay stay reliable
                                either way. It can come out slightly NEGATIVE in that case --
                                the exchanged labelling then fits marginally better and the
                                constraint is doing all the work. nan when enforce_tau_order
                                is False.
        r2_early                the same model scored on [t0, t0 + 5*tau_m] alone
        sign                    'depol' / 'hyperpol' / 'none', from bump_peak_mV
        t_fit_lo_ms, t_fit_hi_ms
        fit_ok                  1 when every gate passed: r2, bump amplitude, no tau pinned to
                                a bound, the bump peak strictly inside the window, and
                                tau_decay within DEXP_MAX_TAU_WINDOW_MULT window lengths
    """
    nan = float("nan")
    empty_early = {"peak_mV": nan, "t_peak_ms": nan, "sign": "none", "tau_ms": nan,
                   "amp_mV": nan, "amp_over_peak": nan, "t_1e_ms": nan, "r2": nan, "n": 0,
                   "t_fit_lo_ms": nan, "t_fit_hi_ms": nan, "fit_ok": 0}
    empty_bump = {"t0_ms": float(t0_ms), "dv_t0_mV": nan, "amp_mV": nan, "offset_mV": nan,
                  "offset_expected_mV": nan, "tau_rise_ms": nan, "tau_decay_ms": nan,
                  "tau_offset_ms": nan, "peak_mV": nan, "t_peak_ms": nan, "peak_dv_mV": nan,
                  "data_peak_mV": nan, "data_t_peak_ms": nan, "sign": "none", "r2": nan,
                  "rmse_mV": nan, "n": 0, "t_fit_lo_ms": nan, "t_fit_hi_ms": nan,
                  "free_offset": 0, "fit_ok": 0}
    out = {"early": empty_early, "bump": empty_bump, "P_mV": nan, "tau_m_ms": nan,
           "Ab_mV": nan, "tau_rise_ms": nan, "tau_decay_ms": nan, "dv_t0_mV": nan,
           "bump_peak_mV": nan, "bump_t_peak_ms": nan, "data_peak_mV": nan,
           "data_t_peak_ms": nan, "r2": nan, "r2_early": nan, "rmse_mV": nan, "n": 0,
           "sign": "none", "t_fit_lo_ms": nan, "t_fit_hi_ms": nan, "swap_dsse_frac": nan,
           "fit_ok": 0}

    t = np.asarray(t_ms, float)
    dv = np.asarray(dv_mV, float)
    if t.size != dv.size or t.size < MIN_FIT_POINTS:
        return out
    good = np.isfinite(t) & np.isfinite(dv)
    t, dv = t[good], dv[good]
    if t.size < MIN_FIT_POINTS:
        return out
    hi = float(t[-1]) if t_end_ms is None else float(t_end_ms)
    win = (t >= float(t0_ms) - T_TOL_MS) & (t <= hi)
    if int(win.sum()) < MIN_FIT_POINTS:
        return out
    tw, vw = t[win], dv[win]
    t_lo = float(tw[0])
    u, y = tw - t_lo, vw
    out["dv_t0_mV"] = float(vw[0])
    out["t_fit_lo_ms"], out["t_fit_hi_ms"] = t_lo, float(tw[-1])
    out["n"] = int(u.size)

    sse, P, Ab, tau_m, tau_r, tau_d, sse_other = _postpulse_search(
        u, y, bounds_m_ms, bounds_rise_ms, bounds_decay_ms, n_grid_m, n_grid, n_refine, zoom,
        enforce_tau_order)
    if not np.isfinite(P):
        return out
    out["swap_dsse_frac"] = ((sse_other - sse) / sse) if (sse > 0 and np.isfinite(sse_other)) \
        else nan
    out["P_mV"], out["tau_m_ms"] = float(P), float(tau_m)
    out["Ab_mV"], out["tau_rise_ms"], out["tau_decay_ms"] = float(Ab), float(tau_r), float(tau_d)
    sst = float(np.sum((y - y.mean()) ** 2))
    out["r2"] = (1.0 - sse / sst) if sst > 0 else nan
    out["rmse_mV"] = float(np.sqrt(sse / u.size))

    pol = P * np.exp(-u / tau_m)
    bmp = Ab * (np.exp(-u / tau_d) - np.exp(-u / tau_r))
    e = u <= 5.0 * tau_m                              # the same model scored on the early part
    if int(e.sum()) >= MIN_FIT_POINTS:
        se = float(np.sum((y[e] - pol[e] - bmp[e]) ** 2))
        sse_t = float(np.sum((y[e] - y[e].mean()) ** 2))
        out["r2_early"] = (1.0 - se / sse_t) if sse_t > 0 else nan

    # extremum of the bump COMPONENT, on a dense grid (no closed form worth the risk)
    ud = np.linspace(0.0, float(u[-1]), 4001)
    bd = Ab * (np.exp(-ud / tau_d) - np.exp(-ud / tau_r))
    j = int(np.argmax(np.abs(bd)))
    out["bump_peak_mV"], out["bump_t_peak_ms"] = float(bd[j]), t_lo + float(ud[j])
    if abs(out["bump_peak_mV"]) >= float(min_amp_mV):
        out["sign"] = "depol" if out["bump_peak_mV"] > 0 else "hyperpol"

    # measured bump peak: the DATA extremum once the polarisation is down to ~5 %
    late = u >= 3.0 * tau_m
    if late.any():
        k = int(np.argmax(np.abs(y[late])))
        out["data_peak_mV"] = float(y[late][k])
        out["data_t_peak_ms"] = float(tw[late][k])

    lb = [np.log10(float(b[0])) for b in (bounds_m_ms, bounds_rise_ms, bounds_decay_ms)]
    ub = [np.log10(float(b[1])) for b in (bounds_m_ms, bounds_rise_ms, bounds_decay_ms)]
    pinned = any(min(abs(np.log10(v) - lb[i]), abs(np.log10(v) - ub[i])) < DEXP_EDGE_TOL_DEC
                 for i, v in enumerate((tau_m, tau_r, tau_d)))
    span = float(tw[-1]) - t_lo
    inside = (ud[j] > 1e-9) and (ud[j] < float(u[-1]) - 1e-9)
    out["fit_ok"] = int((not pinned) and inside
                        and np.isfinite(out["r2"]) and (out["r2"] >= float(min_r2))
                        and (abs(out["bump_peak_mV"]) >= float(min_amp_mV))
                        and (tau_d <= DEXP_MAX_TAU_WINDOW_MULT * span))

    # views of the SAME fit, shaped so the existing eval_* functions draw them unchanged
    out["early"] = {"peak_mV": out["dv_t0_mV"], "t_peak_ms": t_lo,
                    "sign": ("depol" if P > 0 else "hyperpol" if P < 0 else "none"),
                    "tau_ms": float(tau_m), "amp_mV": float(P),
                    "amp_over_peak": (float(P) / out["dv_t0_mV"]
                                      if out["dv_t0_mV"] else nan),
                    "t_1e_ms": float(tau_m),      # exact for this model by construction
                    "r2": out["r2_early"], "n": int(u.size), "t_fit_lo_ms": t_lo,
                    "t_fit_hi_ms": float(tw[-1]), "fit_ok": out["fit_ok"]}
    out["bump"] = {"t0_ms": t_lo, "dv_t0_mV": 0.0, "amp_mV": float(Ab), "offset_mV": 0.0,
                   "offset_expected_mV": 0.0, "tau_rise_ms": float(tau_r),
                   "tau_decay_ms": float(tau_d), "tau_offset_ms": float(tau_m),
                   "peak_mV": out["bump_peak_mV"], "t_peak_ms": out["bump_t_peak_ms"],
                   "peak_dv_mV": out["bump_peak_mV"], "data_peak_mV": out["data_peak_mV"],
                   "data_t_peak_ms": out["data_t_peak_ms"], "sign": out["sign"],
                   "r2": out["r2"], "rmse_mV": out["rmse_mV"], "n": int(u.size),
                   "t_fit_lo_ms": t_lo, "t_fit_hi_ms": float(tw[-1]), "free_offset": 0,
                   "fit_ok": out["fit_ok"]}
    return out


POSTPULSE_COLUMNS = ("pp_P_mV", "pp_tau_m_ms", "pp_Ab_mV", "pp_tau_rise_ms",
                     "pp_tau_decay_ms", "pp_bump_peak_mV", "pp_bump_t_peak_ms",
                     "pp_dv_t0_mV", "pp_sign", "pp_r2", "pp_rmse_mV", "pp_swap_dsse_frac",
                     "pp_fit_ok")


def postpulse_row_values(fit):
    """The POSTPULSE_COLUMNS values of one fit_post_pulse() result, rounded for a CSV."""
    return [_rnd(fit["P_mV"], 6), _rnd(fit["tau_m_ms"], 4), _rnd(fit["Ab_mV"], 6),
            _rnd(fit["tau_rise_ms"], 3), _rnd(fit["tau_decay_ms"], 3),
            _rnd(fit["bump_peak_mV"], 6), _rnd(fit["bump_t_peak_ms"], 3),
            _rnd(fit["dv_t0_mV"], 6), fit["sign"], _rnd(fit["r2"], 5),
            _rnd(fit["rmse_mV"], 6), _rnd(fit["swap_dsse_frac"], 5), int(fit["fit_ok"])]
