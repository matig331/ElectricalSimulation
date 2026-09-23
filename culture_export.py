"""
culture_export.py -- CULTURE-based activation / depolarization / hyperpolarization export.

Unit = a CULTURE: one random realisation in which ALL morphologies are present, each neuron a
random soma POSITION and random ORIENTATION. The SAME placement (position, orientation,
morphology) is then re-simulated at each LAYER thickness -> 3 simulations per soma. For every
simulation we record whether the soma spikes and, if it does not, the sign of its polarization
at the END of phase 2 of the biphasic pulse.

CELL MODEL (config.cell_model -- the single switch, recorded in every CSV row):
  "soma_only"   : soma = full Rich active set; dendrites + stylized axon/AIS PASSIVE.
                  v_init = the settled no-stimulus rest of THAT morphology x layer cell.
  "full_active" : the original model (Rich + active Eyal AIS), v_init = config.v_rest_mV.
                  Reproduces the preliminary full-active dataset bit for bit ('fired').

OUTCOMES (mutually exclusive, one per simulation):
  activation        fired = 1 (soma crossed 0 mV at least once)
  depolarization    not fired and DeltaV_end >  +PHASE2_EPS_MV
  hyperpolarization not fired and DeltaV_end <  -PHASE2_EPS_MV
  neutral           otherwise (counted as 0 in all three files)

  DeltaV_end = Vm_soma(t_end) - Vm_sham(t_end), t_end = END of phase 2.
  Vm_sham is the SAME simulation with zero stimulus current (same init, same dt, same
  window) -- one sham per morphology x layer, computed once per process. It equals the
  settled rest V_rest exactly when the cell starts at equilibrium. It does NOT here:
  finitialize(V_rest) sets EVERY compartment to the somatic rest, but the equilibrium is
  spatially non-uniform (passive axon/dendrites sit closer to e_pas), so the unstimulated soma
  drifts by ~+0.09 mV by t_end. Referencing the scalar V_rest would let that drift, not the
  stimulus, decide the sign of every weak far-field response. The drift is written to the CSV
  (ctrl_drift_mV), so the scalar-referenced value is recoverable:
      Vm(t_end) - V_rest = deltaVm_end_phase2_mV + ctrl_drift_mV.

Statistic: pulses at 0.2 Hz are dynamically independent (train_invariance), so one pulse gives
the per-neuron outcome of the whole >= 1 min epoch (config.stim_duration_s -> n_pulses column).

ONE shared core, TWO drivers:
  CellPool + iter_culture_blocks() + build_row() are used by BOTH the serial path
  (culture_export() below) and the parallel HPC path (culture_worker.py), so the two can
  never diverge again. CSV_HEADER is the single source of truth for the raw schema.
  CellPool keeps ONE live NEURON cell per process: NEURON integrates every section that
  exists, and keeping all morphology x layer cells alive made each simulation ~6x slower.

Outputs of the serial path (the parallel path writes the same via culture_merge.py):
  culture_Pactivation.csv, culture_Pdepolarization.csv, culture_Phyperpolarization.csv
      one row per (culture, neuron, layer); all identity/provenance columns + ONE outcome
      column (fired / depolarized / hyperpolarized) -- the input of culture_statistics.py.
  culture_Pmap_<outcome>.pdf : quick-look maps (pooled neuron-level P, Wilson 95% CI).

Run:   python culture_export.py            # serial, one core -- dry runs only
HPC:   bash jobs/submit_seeded.sh ...       # parallel (culture_worker.py + culture_merge.py)
Smoke: python smoke_test_culture_export.py  # offline, no NEURON
       python smoke_test_soma_only.py       # NEURON-backed, end to end
"""
import os, csv, gc, shutil, tempfile
import numpy as np

# ----------------------------- SCHEMA (single source of truth) ----------------------------- #
# The 15 columns of the preliminary full-active campaign, in their original order. Parts
# written with exactly this header are still accepted by culture_merge.py (read-only legacy).
LEGACY_CSV_HEADER = ["culture", "neuron", "morphology", "layer_um", "x_um", "y_um",
                     "dist_nearest_elec_um", "dist_center_um", "dist_dipole3d_um",
                     "theta_orient_deg", "theta_pos_deg", "n_pulses", "i0_uA", "fired", "seed"]
# Current raw schema = legacy columns (unchanged order) + soma-only provenance and outcomes.
# 'seed' is the RNG base actually used for that row's culture: culture_merge.py uses the
# (seed, culture) pair to tell a true duplicate from an independent replicate.
# ... + the post-pulse kinetics (bump_kinetics.STAGED_COLUMNS). Every row carries them; a
# row whose kinetics were not measured (bump_ms = 0, or a culture outside the long-window
# subsample) has them BLANK, never absent -- the schema is constant across a campaign.
from bump_kinetics import (STAGED_COLUMNS, common_prefix_len, empty_staged, fit_staged,
                           staged_row_values)
# The 22 columns of the soma-only campaign, BEFORE the post-pulse kinetics were added. Parts
# written with exactly this header are still read (read-only), so the existing
# results_soma_only/ tree keeps merging and analysing unchanged. culture_merge.py refuses to
# mix schemas, so an old part and a new one can never end up in one merged file.
OUTCOME_CSV_HEADER = LEGACY_CSV_HEADER + ["cell_model", "v_rest_mV", "ctrl_drift_mV",
                                          "deltaVm_end_phase2_mV", "phase2_outcome",
                                          "depolarized", "hyperpolarized"]
CSV_HEADER = OUTCOME_CSV_HEADER + list(STAGED_COLUMNS)
# The first kinetics schema (44 columns): the same, without dexp_data_peak_mV and
# dexp_data_t_peak_ms. Written only by the first full_tuned dry runs. Recognised, read-only,
# and -- like every schema -- never merged together with another one.
KINETICS_V1_CSV_HEADER = [c for c in CSV_HEADER
                          if c not in ("dexp_data_peak_mV", "dexp_data_t_peak_ms")]
OUTCOME_COLS = ("fired", "depolarized", "hyperpolarized")
# outcome name -> (file-name stem, 0/1 column). MUST match culture_statistics.OUTCOMES
# (checked by smoke_test_culture_statistics.py).
OUTCOME_FILES = {
    "activation": ("Pactivation", "fired"),
    "depolarization": ("Pdepolarization", "depolarized"),
    "hyperpolarization": ("Phyperpolarization", "hyperpolarized"),
}
CELL_MODELS = ("soma_only", "full_active", "full_tuned")
PHASE2_EPS_MV = 1e-6      # |DeltaV_end| <= eps -> 'neutral' (numerically zero)


# ----------------------- PURE HELPERS (offline-testable) ----------------------- #

def electrode_center(elec_xy):
    return np.asarray(elec_xy, float).mean(axis=0)

def dipole_axis_deg(elec_xy, sign):
    """Angle (deg) of the anode-centroid -> cathode-centroid axis (the field dipole).
    If not bipolar (all one sign) returns 0.0."""
    e = np.asarray(elec_xy, float); s = np.asarray(sign, float)
    if (s > 0).any() and (s < 0).any():
        v = e[s < 0].mean(axis=0) - e[s > 0].mean(axis=0)
        return float(np.degrees(np.arctan2(v[1], v[0])))
    return 0.0

def dist_from_center(pos_xy, center):
    p = np.atleast_2d(np.asarray(pos_xy, float)); c = np.asarray(center, float)
    return np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])

def dist_from_nearest_electrode(pos_xy, elec_xy):
    """Distance (um) from each position to the NEAREST electrode. This is the physically
    meaningful predictor when the electrodes are spread out (activation happens near each
    electrode, not around the array centroid)."""
    p = np.atleast_2d(np.asarray(pos_xy, float)); e = np.asarray(elec_xy, float)
    # (n_pos, n_elec) distances -> min over electrodes
    dx = p[:, None, 0] - e[None, :, 0]; dy = p[:, None, 1] - e[None, :, 1]
    return np.min(np.hypot(dx, dy), axis=1)

def rel_orientation_deg(theta_deg, axis_deg):
    """Neuron orientation relative to the dipole axis, folded to [0, 90] (a neurite and its
    180-flip couple to the field the same way)."""
    d = np.abs((np.asarray(theta_deg, float) - axis_deg) % 180.0)
    return np.minimum(d, 180.0 - d)

def dipole_frame(elec_xy, sign):
    """Frame of the field dipole: centre c = midpoint of the anode/cathode centroids, and the
    unit direction d pointing from the CATHODE side (-) to the ANODE side (+). Returns (c[2], d[2])."""
    e = np.asarray(elec_xy, float); s = np.asarray(sign, float)
    if (s > 0).any() and (s < 0).any():
        a = e[s > 0].mean(axis=0)          # anode centroid (+)
        k = e[s < 0].mean(axis=0)          # cathode centroid (-)
        c = 0.5 * (a + k)
        d = a - k                          # from - to +
        n = np.hypot(d[0], d[1]) or 1.0
        return c, d / n
    return e.mean(axis=0), np.array([1.0, 0.0])

def directional_rt(pos_xy, center, direction, z_um=10.0):
    """Express each soma in the DIPOLE frame: r = 3D distance from the dipole centre (including
    the soma height z_um above the electrode plane), and theta = angle (deg) between the in-plane
    offset (pos - centre) and the dipole direction d (which points - -> +), UNFOLDED to [0,180]:
    theta = 0deg -> toward the anode (+) side; theta = 180deg -> toward the cathode (-) side; theta = 90deg -> across.
    Keeping the full [0,180] range is what makes the cathodic preference visible. Returns (r_3d, theta_deg)."""
    p = np.atleast_2d(np.asarray(pos_xy, float)); c = np.asarray(center, float)
    d = np.asarray(direction, float)
    v = p - c                                        # in-plane offset (n,2)
    r_in = np.hypot(v[:, 0], v[:, 1])
    r3d = np.sqrt(r_in ** 2 + float(z_um) ** 2)
    dot = v[:, 0] * d[0] + v[:, 1] * d[1]            # signed: >0 toward +, <0 toward -
    cross = np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])  # |v| sin (>=0)
    theta = np.degrees(np.arctan2(cross, dot))       # [0,180]: 0 -> anode side, 180 -> cathode side
    theta = np.where(r_in > 1e-9, theta, 0.0)
    return r3d, theta

def smooth_P_field(X, Y, fired, gx, gy, bw):
    """Kernel-smoothed activation probability field around the somata (Nadaraya-Watson):
    P(gx,gy) = sum_i fired_i * K_i / sum_i K_i, K_i = exp(-d^2 / 2 bw^2). Gives a readable
    'activation area' AROUND THE SOMATA even from sparse samples. Returns Z[len(gy),len(gx)]."""
    X = np.asarray(X, float); Y = np.asarray(Y, float); fired = np.asarray(fired, float)
    GX, GY = np.meshgrid(gx, gy)
    Z = np.full(GX.shape, np.nan)
    inv2b2 = 1.0 / (2.0 * bw * bw)
    for j in range(GX.shape[0]):
        for i in range(GX.shape[1]):
            d2 = (X - GX[j, i]) ** 2 + (Y - GY[j, i]) ** 2
            k = np.exp(-d2 * inv2b2)
            sk = k.sum()
            if sk > 1e-6:
                Z[j, i] = float((k * fired).sum() / sk)
    return Z

def radial_bin_P(dist, fired, edges):
    """Sholl-like: fraction of somata that fired, per radial bin. Returns (centers, P, counts)."""
    dist = np.asarray(dist, float); fired = np.asarray(fired, float)
    edges = np.asarray(edges, float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    P = np.full(len(centers), np.nan); counts = np.zeros(len(centers), int)
    for k in range(len(centers)):
        m = (dist >= edges[k]) & (dist < edges[k + 1])
        counts[k] = int(m.sum())
        if counts[k] > 0:
            P[k] = float(fired[m].mean())
    return centers, P, counts

def per_culture_P(dist, fired, culture, edges):
    """P(distance) computed SEPARATELY for each culture (the replicate unit), then the mean
    and SD ACROSS cultures per bin. Avoids pseudoreplication (pooling all neurons as if
    independent). Returns (centers, mean, sd, n_cultures_per_bin, stack[n_cult, n_bins])."""
    dist = np.asarray(dist, float); fired = np.asarray(fired, float)
    culture = np.asarray(culture)
    centers = 0.5 * (np.asarray(edges)[:-1] + np.asarray(edges)[1:])
    cults = np.unique(culture)
    stack = np.full((len(cults), len(centers)), np.nan)
    for ci, c in enumerate(cults):
        m = culture == c
        _, P, cnt = radial_bin_P(dist[m], fired[m], edges)
        stack[ci] = P
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0)
        ncb = np.sum(np.isfinite(stack), axis=0)
    return centers, mean, sd, ncb, stack

def logistic_decreasing(r, r50, w):
    """Decreasing recruitment sigmoid: 1 near the electrodes, -> 0 far. r50 = distance at P=0.5."""
    return 1.0 / (1.0 + np.exp((np.asarray(r, float) - r50) / w))

def _cross(r, p, level):
    """Distance at which the (decreasing) curve p(r) crosses `level`, by linear interpolation.
    r must be sorted ascending. Returns nan-safe within-range values."""
    r = np.asarray(r, float); p = np.asarray(p, float)
    idx = np.where(p < level)[0]
    if idx.size == 0:
        return float(r[-1])            # never drops below the level -> at/after the last point
    i = idx[0]
    if i == 0:
        return float(r[0])             # already below at the first point
    p0, p1, r0, r1 = p[i - 1], p[i], r[i - 1], r[i]
    if p0 == p1:
        return float(r0)
    return float(r0 + (r1 - r0) * (p0 - level) / (p0 - p1))


def fit_logistic_decreasing(r, p, r50_0=None, w0=None, **_):
    """Robust, dependency-free summary of a decreasing recruitment curve (no scipy, no
    gradient descent). r50 = the 0.5-crossing (interpolated); w from the 0.75->0.25 width
    (a logistic spans that over ~2.197.w). Deterministic and stable even for sharp drops."""
    r = np.asarray(r, float); p = np.asarray(p, float)
    m = np.isfinite(r) & np.isfinite(p)
    r, p = r[m], p[m]
    if r.size < 2 or np.ptp(r) == 0:
        return (float(np.median(r)) if r.size else 0.0), float(w0 or 10.0)
    order = np.argsort(r); r, p = r[order], p[order]
    r50 = _cross(r, p, 0.5)            # from the data (robust); r50_0 kept only for API compat
    r75, r25 = _cross(r, p, 0.75), _cross(r, p, 0.25)
    if np.isfinite(r75) and np.isfinite(r25) and r25 > r75:
        w = max((r25 - r75) / 2.197, 0.5)
    else:
        w = float(w0) if w0 else max(0.05 * np.ptp(r), 1.0)
    return float(r50), float(w)

def assign_morphologies(n, n_morph, rng):
    """Round-robin then shuffle: guarantees every morphology appears when n >= n_morph."""
    idx = np.array([i % n_morph for i in range(n)], int)
    rng.shuffle(idx)
    return idx

def grid_P_2d(xs, ys, fired, half, n_bins):
    """Bin scattered (x,y,fired) into an n_bins x n_bins grid of MEAN fired = P(x,y).
    Returns (Z, extent). Empty bins are nan. Used for the iso-probability area map."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); fired = np.asarray(fired, float)
    edges = np.linspace(-half, half, int(n_bins) + 1)
    ix = np.clip(np.digitize(xs, edges) - 1, 0, int(n_bins) - 1)
    iy = np.clip(np.digitize(ys, edges) - 1, 0, int(n_bins) - 1)
    Z = np.full((int(n_bins), int(n_bins)), np.nan)
    for j in range(int(n_bins)):
        for i in range(int(n_bins)):
            m = (ix == i) & (iy == j)
            if m.any():
                Z[j, i] = float(fired[m].mean())
    return Z, [-half, half, -half, half]

def activation_area_per_culture(culture, fired, span_um):
    """ACTIVATION AREA per culture (um^2), the reviewer's per-culture quantity.
    Somata are placed uniformly at density N/(sampled area), so the number that fire is
    density x (activated area) => activated area = (fraction of the culture's somata that
    fired) x (sampled area). One value per culture; then take mean +/- SD across cultures.
    Also returns the equivalent radius r_eq = sqrt(area/pi) (relative to the array centroid).
    Returns (cultures, area_um2[per culture], r_eq_um[per culture], mean_area, sd_area)."""
    culture = np.asarray(culture); fired = np.asarray(fired, float)
    sampled = (2.0 * float(span_um)) ** 2
    cults = np.unique(culture)
    area = np.array([fired[culture == c].mean() * sampled for c in cults]) if len(cults) else np.array([])
    r_eq = np.sqrt(area / np.pi)
    mean = float(np.mean(area)) if area.size else float("nan")
    sd = float(np.std(area)) if area.size else float("nan")
    return cults, area, r_eq, mean, sd


def phase_windows(baseline_ms=5.0, phase_dur_ms=0.25, interphase_us=0.0):
    """Time windows (ms) of the two phases of the anodic-first biphasic pulse and the transition
    instant. Phase 1 (positive): [t_on, t_on+phi]; transition at t_on+phi (+gap); Phase 2 (negative):
    [t_on+phi+gap, t_on+2phi+gap]. Returns (t_on, p1, t_transition, p2)."""
    t_on = float(baseline_ms); phi = float(phase_dur_ms); gap = float(interphase_us) * 1e-3
    p1 = (t_on, t_on + phi)
    t_tr = t_on + phi
    p2 = (t_on + phi + gap, t_on + 2 * phi + gap)
    return t_on, p1, t_tr, p2


def phase_diagram(i0_uA=None, phase_dur_ms=None, out_path="phase_diagram.pdf"):
    """ILLUSTRATIVE (no NEURON): the biphasic waveform with its two phases and the exact transition
    instant, plus two spatial panels showing that the dipole polarity -- and hence the excitatory
    cathode -- flips between phase 1 (- -> +, cathode = sign- electrodes) and phase 2 (+ -> -, cathode
    = sign+ electrodes). Charge-balanced, so the two sides are excited symmetrically in principle."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import CFG
    import field as F
    i0 = CFG.i0_uA if i0_uA is None else float(i0_uA)
    phi = CFG.phase_dur_ms if phase_dur_ms is None else float(phase_dur_ms)
    elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
    an = elec[sign > 0]; ca = elec[sign < 0]
    t_on, p1, t_tr, p2 = phase_windows(5.0, phi, CFG.interphase_us)
    t = np.arange(0.0, 2 * phi + 10.0, CFG.dt_ms / 4)
    I = F.biphasic_current(t - t_on, i0, phi, True, CFG.ramp_us, CFG.interphase_us) * 1e6  # uA

    fig = plt.figure(figsize=(8.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.0], hspace=0.32, wspace=0.18)

    # top: the waveform with the two phases and the transition
    axw = fig.add_subplot(gs[0, :])
    axw.plot(t, I, lw=1.8, color="k")
    axw.axvspan(*p1, color="royalblue", alpha=0.18); axw.axvspan(*p2, color="crimson", alpha=0.18)
    axw.axvline(t_tr, color="0.3", ls="--", lw=1.5)
    axw.text(np.mean(p1), i0 * 0.6, "phase 1 (+)\ncathode = -", ha="center", fontsize=8, color="royalblue")
    axw.text(np.mean(p2), -i0 * 0.6, "phase 2 (-)\ncathode = +", ha="center", fontsize=8, color="crimson")
    axw.text(t_tr, i0 * 1.02, "transition", ha="center", fontsize=8, color="0.3")
    axw.set_xlim(t_on - 0.2, p2[1] + 0.4); axw.set_xlabel("t (ms)"); axw.set_ylabel("I (uA)")
    axw.set_title("Biphasic pulse: the dipole polarity reverses at the transition", fontsize=10)

    def panel(ax, arrow_from, arrow_to, cath, title, col):
        ax.annotate("", xy=arrow_to, xytext=arrow_from,
                    arrowprops=dict(arrowstyle="-|>", color="cyan", lw=3), zorder=4)
        for (ex, ey) in cath:                       # the excitatory (cathodic) electrodes this phase
            ax.add_patch(plt.Circle((ex, ey), 22, color=col, alpha=0.30, zorder=1))
        for (ex, ey), s in zip(elec, sign):
            ax.scatter([ex], [ey], marker="+" if s > 0 else "_", s=150, c="k", linewidths=2, zorder=5)
        h = CFG.pitch_um
        ax.set_xlim(-2.2 * h, 2.2 * h); ax.set_ylim(-3.0 * h, 2.2 * h); ax.set_aspect("equal")
        ax.set_xlabel("x (um)"); ax.set_title(title, fontsize=9)

    # phase 1: dipole - -> + (cathode centroid -> anode centroid); excitatory cathode = sign-
    ax1 = fig.add_subplot(gs[1, 0])
    panel(ax1, ca.mean(0), an.mean(0), ca, "Phase 1 (+I):  - -> +\nexcitation near the - electrodes", "royalblue")
    ax1.set_ylabel("y (um)")
    # phase 2: dipole flips + -> - ; excitatory cathode = sign+
    ax2 = fig.add_subplot(gs[1, 1])
    panel(ax2, an.mean(0), ca.mean(0), an, "Phase 2 (-I):  + -> -\nexcitation near the + electrodes", "crimson")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, out_path)
    fig.savefig(out); plt.close(fig)
    print("done:", out)
    return out


def per_soma_P(X, Y, fired):
    """Per-soma activation probability = mean 'fired' over that soma's rows (its layers).
    Groups the pooled rows by unique (x, y). Returns (xs, ys, P) for the scatter."""
    X = np.asarray(X, float); Y = np.asarray(Y, float); fired = np.asarray(fired, float)
    key = np.round(np.stack([X, Y], 1), 2)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)                  # numpy 1.x / 2.x return shapes differ
    # vectorised group mean (was a per-soma boolean scan: O(N x U), hours at HPC row counts)
    s = np.bincount(inv, weights=fired, minlength=len(uniq))
    n = np.bincount(inv, minlength=len(uniq))
    P = s / np.maximum(n, 1)
    return uniq[:, 0], uniq[:, 1], P


def iso_radii(r50, w, levels=(0.75, 0.5, 0.25)):
    """Radii at which the fitted decreasing logistic crosses each probability level.
    P(r)=level  =>  r = r50 + w*ln((1-level)/level). Returns list of (level, radius)."""
    out = []
    for L in levels:
        L = float(np.clip(L, 1e-6, 1 - 1e-6))
        out.append((L, float(r50 + w * np.log((1.0 - L) / L))))
    return out

# ------------- LEGACY extremum-based classifiers -- NOT USED BY THE EXPORT ------------- #
# _loglin_tau / fit_peak_kinetics / fit_biexp / classify_response / _post_phase2_peak belong
# to an earlier criterion (sign of the dominant post-pulse extremum). The export now uses ONLY
# the sign of DeltaV at the END of phase 2 (phase2_end_outcome). Kept importable because
# smoke_test_culture_export.py tests them and older notebooks may call them.

def _loglin_tau(t, y):
    """tau from a log-linear fit of a positive, decaying y(t): ln y = a + b t -> tau = -1/b.
    Returns (tau_ms, amplitude_at_t0). nan if it cannot be fit."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = y > 1e-4
    if int(m.sum()) < 3:
        return float("nan"), float("nan")
    b, a = np.polyfit(t[m], np.log(y[m]), 1)
    tau = -1.0 / b if b < 0 else float("nan")
    return float(tau), float(np.exp(a))


def fit_peak_kinetics(t, y, t_skip_ms=7.0, t_bump_ms=200.0, t_max_ms=1000.0, min_amp_mV=0.3):
    """Bump peak + membrane relaxation, on the window from the END of the biphasic pulse (t from 0).
    The first `t_skip` ms (fast capacitive transient) are DROPPED. In [t_skip, t_bump] the peak is the
    largest SIGNED deflection from rest -- taken on the RAW trace, NOT |y|: the most-positive and the
    most-negative excursions are compared, and the one larger IN MAGNITUDE wins, keeping its sign. So
    a trace that goes mostly negative is classified hyperpol (not forced positive). tau_decay is a
    log-linear fit of the return to rest after that peak. Returns (peak_signed_mV, t_peak_ms,
    tau_decay_ms). numpy-only."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    bump = (t >= t_skip_ms) & (t <= t_bump_ms)
    if int(bump.sum()) < 3:
        return float("nan"), float("nan"), float("nan")
    tb = t[bump]; yb = y[bump]
    ipos = int(np.argmax(yb)); ymax = float(yb[ipos])      # most positive (depol candidate)
    ineg = int(np.argmin(yb)); ymin = float(yb[ineg])      # most negative (hyperpol candidate)
    if abs(ymax) >= abs(ymin):                             # larger MAGNITUDE wins, keep its raw sign
        peak, tpk = ymax, float(tb[ipos])
    else:
        peak, tpk = ymin, float(tb[ineg])
    if abs(peak) < min_amp_mV:
        return peak, tpk, float("nan")
    s = 1.0 if peak >= 0 else -1.0
    dec = (t > tpk) & (t <= t_max_ms)                      # relaxation of the bump back to rest
    tau = float("nan")
    if int(dec.sum()) > 5:
        td, _ = _loglin_tau(t[dec] - t[dec][0], s * y[dec])
        tau = td if (np.isfinite(td) and 0 < td <= t_max_ms) else float("nan")
    return peak, tpk, tau


# kept for reference / backward-compat
def fit_biexp(t, y):
    a, tpk, td = fit_peak_kinetics(t, y)
    return a, tpk, float("nan"), td


def classify_response(fired, dep_mV, hyp_mV, thr_mV=0.5):
    """LEGACY -- NOT used by the export (see phase2_end_outcome).
    Mutually-exclusive per-soma outcome from the Vm response: 'activation' if it spiked;
    otherwise 'depol'/'hyperpol' by the SIGN of the dominant sub-threshold Vm swing (the larger of
    the max depolarizing vs max hyperpolarizing deflection); 'none' if the deflection is negligible."""
    if fired:
        return "activation"
    if dep_mV >= hyp_mV and dep_mV > thr_mV:
        return "depol"
    if hyp_mV > dep_mV and hyp_mV > thr_mV:
        return "hyperpol"
    return "none"


MAX_DISKS = 3000   # page-3 display cap: disks drawn for at most this many somata (see below)


def render_outcome_maps(pdf_path, tag, outcome, *, elec, sign, center, dip_c, dip_d, span,
                        bin_um, edges, i0, n_pulses, cfg, X, Y, CULT, DN, RD, TP,
                        max_disks=MAX_DISKS):
    """Render the 5-page map set for one outcome (activation/depolarization/hyperpolarization).
    NO NEURON: works from rows alone, so the serial export and culture_merge.py share it.
    `outcome` is the per-row 0/1 array. Statistic = pooled neuron-level P with Wilson 95% CI
    (the same estimator as culture_statistics.py).
    HPC scale: page 3 draws one disk per soma for at most `max_disks` somata (a fixed-seed
    random subset -- display only; P, the smooth field and every other page use ALL rows), and
    the dense layers are rasterised so the PDF stays small at millions of rows.
    (Was _render_maps; renamed because culture_merge.py imports it.)"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        # (1) RECRUITMENT curve: pooled NEURON-level P vs 3D dipole distance.
        fig, ax = plt.subplots(figsize=(8, 5))
        ctr, P1, cnt1 = radial_bin_P(RD, outcome, edges)
        k1 = np.rint(P1 * cnt1).astype(float)
        good = np.isfinite(P1) & (cnt1 > 0)
        z = 1.959963984540054
        n = cnt1.astype(float)
        ph = np.divide(k1, n, out=np.full_like(k1, np.nan), where=n > 0)
        den = 1.0 + z*z/n
        cen = (ph + z*z/(2*n))/den
        half = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n))/den
        lo, hi = np.clip(cen-half,0,1), np.clip(cen+half,0,1)
        ax.fill_between(ctr[good], lo[good], hi[good], alpha=0.22, label="95% Wilson CI")
        ax.plot(ctr[good], P1[good], "o-", ms=4, label="pooled neuron-level P")
        gm = P1[good]
        monotone = gm.size >= 3 and int(np.nanargmax(gm)) <= 1
        if monotone:
            try:
                r50, w = fit_logistic_decreasing(ctr[good], P1[good])
                rr = np.linspace(0, span, 200)
                ax.plot(rr, logistic_decreasing(rr, r50, w), "k--", lw=2,
                        label=f"logistic fit (r50={r50:.0f}um, w={w:.0f})")
            except Exception as e:
                print("  fit failed:", e)
        ax.set_xlabel("3D distance from dipole centre (um)")
        ax.set_ylabel(f"P -- {tag}"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{tag}: neuron-level P(r) @ {i0:.0f} uA\n"
                     f"each neuron observation contributes independently; Wilson 95% CI", fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # (1b) pooled neuron-level P vs distance from the NEAREST electrode.
        fig, ax = plt.subplots(figsize=(8, 5))
        ctr2, P2, cnt2 = radial_bin_P(DN, outcome, edges)
        k2 = np.rint(P2 * cnt2).astype(float)
        good2 = np.isfinite(P2) & (cnt2 > 0)
        n2 = cnt2.astype(float)
        ph2 = np.divide(k2, n2, out=np.full_like(k2, np.nan), where=n2 > 0)
        den2 = 1.0 + z*z/n2
        cen2 = (ph2 + z*z/(2*n2))/den2
        half2 = z*np.sqrt(ph2*(1-ph2)/n2 + z*z/(4*n2*n2))/den2
        lo2, hi2 = np.clip(cen2-half2,0,1), np.clip(cen2+half2,0,1)
        ax.fill_between(ctr2[good2], lo2[good2], hi2[good2], alpha=0.22, label="95% Wilson CI")
        ax.plot(ctr2[good2], P2[good2], "o-", ms=4, label="pooled neuron-level P")
        try:
            r50n, wn = fit_logistic_decreasing(ctr2[good2], P2[good2])
            rr = np.linspace(0, span, 200)
            ax.plot(rr, logistic_decreasing(rr, r50n, wn), "k--", lw=2,
                    label=f"logistic fit (r50={r50n:.0f}um, w={wn:.0f})")
        except Exception as e:
            print("  nearest fit failed:", e)
        ax.set_xlabel("distance from NEAREST electrode (um)")
        ax.set_ylabel(f"P -- {tag}"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{tag}: neuron-level P vs nearest-electrode distance @ {i0:.0f} uA\n"
                     f"Wilson 95% CI", fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # (2) directional anisotropy: P vs 3D distance, split by POSITION angle to the dipole axis,
        #     UNFOLDED so the cathode side (theta->180deg) is distinguishable from the anode side (theta->0deg)
        fig, ax = plt.subplots(figsize=(8, 5))
        obins = [(0, 60, "toward sign+ electrodes"), (60, 120, "~= perpendicular"), (120, 180, "toward sign- electrodes")]
        for lo, hi, lab in obins:
            mo = (TP >= lo) & (TP < hi)
            ctr, P, cnt = radial_bin_P(RD[mo], outcome[mo], edges)
            ax.plot(ctr, P, "o-", ms=4, alpha=0.85, label=f"{lo}-{hi}deg . {lab}")
        ax.set_xlabel("3D distance from dipole centre (um)")
        ax.set_ylabel(f"P -- {tag}"); ax.set_ylim(-0.02, 1.02)
        ax.set_title("Directional dependence: P vs distance, by the soma's angle to the dipole axis\n"
                     "(theta=0deg toward the sign+ electrodes, theta=180deg toward sign-; the excitatory cathode "
                     "flips with the biphasic phase)", fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # (3) ACTIVATION AREA -- an AREA PER SOMA (disk coloured by its P) over a faint smooth field,
        #     with the dipole DIRECTIONS drawn as arrows (- -> +)
        sx, sy, sp = per_soma_P(X, Y, outcome)
        cmap = plt.get_cmap("plasma")
        margin = 0.12 * span; lim = span + margin
        gx = np.linspace(-lim, lim, 90); gy = np.linspace(-lim, lim, 90)
        bw = max(1.5 * bin_um, 20.0)
        Zf = smooth_P_field(X, Y, outcome, gx, gy, bw)
        fig, ax = plt.subplots(figsize=(7.0, 6.3))
        ax.pcolormesh(gx, gy, np.nan_to_num(Zf, nan=0.0), cmap=cmap, vmin=0, vmax=1,
                      shading="auto", zorder=0, alpha=0.35, rasterized=True)  # faint continuous area
        soma_r = max(0.6 * bw, 12.0)                                  # radius of each soma's area
        n_soma = len(sx)
        if max_disks is not None and n_soma > int(max_disks):       # display cap (see docstring)
            show = np.random.default_rng(0).choice(n_soma, int(max_disks), replace=False)
        else:
            show = np.arange(n_soma)
        for xi, yi, pi in zip(sx[show], sy[show], sp[show]):         # ONE AREA PER SOMA
            ax.add_patch(plt.Circle((xi, yi), soma_r, color=cmap(pi), alpha=0.55, ec="none", zorder=3))
        sc = ax.scatter(sx[show], sy[show], c=sp[show], cmap=cmap, vmin=0, vmax=1, s=14,
                        edgecolors="w", linewidths=0.4, zorder=5, rasterized=True)  # soma centres
        # dipole directions: an arrow from each cathode to its paired anode (- -> +)
        an = elec[sign > 0]; ca = elec[sign < 0]
        for (kx, ky) in ca:                                          # nearest anode to each cathode
            j = int(np.argmin(np.hypot(an[:, 0] - kx, an[:, 1] - ky)))
            ax.annotate("", xy=(an[j, 0], an[j, 1]), xytext=(kx, ky),
                        arrowprops=dict(arrowstyle="-|>", color="cyan", lw=2.2, shrinkA=0, shrinkB=0), zorder=7)
        for (ex, ey), s in zip(elec, sign):
            ax.scatter([ex], [ey], marker="+" if s > 0 else "_", s=120, c="k", linewidths=1.8, zorder=8)
        ax.plot([], [], color="cyan", lw=2.2, label="dipole direction (- -> +)")
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        shown = ("" if len(show) == n_soma
                 else f" (disks: random {len(show)} of {n_soma} somata; field uses all)")
        ax.set_title(f"{tag} area @ {i0:.0f} uA -- one area per soma, coloured by P{shown}\n"
                     f"cyan arrows = the three parallel dipole directions (- -> +)", fontsize=9)
        fig.colorbar(sc, ax=ax, label=f"P -- {tag}"); ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # (4) EXPLAINER: the CENTRAL dipole drawn - -> + between its two electrodes, the position
        #     vectors and their angle theta, and the example somata coloured by ACTIVATED vs silent.
        an = elec[sign > 0]; ca = elec[sign < 0]
        if len(an) and len(ca):
            ca_c = an.mean(0); ck_c = ca.mean(0)
            c_an = an[int(np.argmin(np.hypot(an[:, 0] - ca_c[0], an[:, 1] - ca_c[1])))]   # central anode (+)
            c_ka = ca[int(np.argmin(np.hypot(ca[:, 0] - ck_c[0], ca[:, 1] - ck_c[1])))]   # central cathode (-)
        else:
            c_an, c_ka = dip_c + dip_d * 30, dip_c - dip_d * 30
        sx2, sy2, sp2 = per_soma_P(X, Y, outcome)
        vlen = np.hypot(sx2 - dip_c[0], sy2 - dip_c[1]); keep = vlen > 1e-6
        sx2, sy2, sp2 = sx2[keep], sy2[keep], sp2[keep]
        ang = np.degrees(np.arctan2(sy2 - dip_c[1], sx2 - dip_c[0]))
        order = np.argsort(ang); n_ex = min(20, len(order))                               # <= 20 examples
        pick = order[np.linspace(0, len(order) - 1, n_ex).astype(int)] if len(order) else []
        outcome_ex = sp2 >= 0.5                                                             # "activated"
        fig, ax = plt.subplots(figsize=(6.9, 6.5))
        # central dipole: a line with an arrow from the - electrode to the + electrode
        ax.annotate("", xy=(c_an[0], c_an[1]), xytext=(c_ka[0], c_ka[1]),
                    arrowprops=dict(arrowstyle="-|>", color="cyan", lw=3.5), zorder=4)
        ax.text(*(0.5 * (c_an + c_ka) + np.array([0, 6])), "central dipole  - -> +",
                color="c", fontsize=9, ha="center", zorder=5)
        for k in pick:
            vx, vy = sx2[k] - dip_c[0], sy2[k] - dip_c[1]
            col = "crimson" if outcome_ex[k] else "0.55"
            ax.annotate("", xy=(sx2[k], sy2[k]), xytext=(dip_c[0], dip_c[1]),
                        arrowprops=dict(arrowstyle="-|>", color="0.6", lw=1.1), zorder=3)
            ax.scatter([sx2[k]], [sy2[k]], s=95, c=col, edgecolors="k", linewidths=0.6, zorder=6)
            th = np.degrees(np.arctan2(abs(vx * dip_d[1] - vy * dip_d[0]), vx * dip_d[0] + vy * dip_d[1]))
            ax.text(sx2[k], sy2[k], f"  theta={th:.0f}deg", fontsize=7.5, zorder=7)
        for (ex, ey), s in zip(elec, sign):
            ax.scatter([ex], [ey], marker="+" if s > 0 else "_", s=120, c="k", linewidths=1.8, zorder=2)
        ax.scatter([dip_c[0]], [dip_c[1]], marker="o", s=40, c="k", zorder=6)
        ax.scatter([], [], c="crimson", edgecolors="k", label=f"{tag}")
        ax.scatter([], [], c="0.55", edgecolors="k", label="other")
        ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        ax.set_title(f"theta = angle between the central dipole (- -> +) and each soma's position vector\n"
                     f"(grey arrows = position vectors from the dipole centre; {n_ex} example somata)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return pdf_path


# ------------------------- PHASE-2 OUTCOME (pure, offline-testable) ------------------------- #

def v_at_end_of_phase2(tw, vw):
    """Somatic Vm at the END of phase 2. `tw` = time relative to the end of the biphasic pulse
    (t = 0 = end of phase 2), as returned by rich_footprint.spikes_at(detail=True); the sample
    nearest t = 0 is used. nan if the window is empty or tw/vw lengths differ."""
    tw = np.asarray(tw, float)
    vw = np.asarray(vw, float)
    if tw.size == 0 or vw.size == 0 or tw.size != vw.size:
        return float("nan")
    return float(vw[int(np.argmin(np.abs(tw)))])


def phase2_end_outcome(fired, dv_end_mV, eps_mV=PHASE2_EPS_MV):
    """Mutually exclusive outcome of ONE simulation -> (label, activation, depolarized,
    hyperpolarized). A spike wins; otherwise ONLY the sign of DeltaV_end decides (no extremum,
    no delayed rebound). |DeltaV_end| <= eps, or nan -> 'neutral' (0 in all three files)."""
    if int(fired):
        return "activation", 1, 0, 0
    dv = float(dv_end_mV)
    if not np.isfinite(dv):
        return "neutral", 0, 0, 0
    if dv > eps_mV:
        return "depol", 0, 1, 0
    if dv < -eps_mV:
        return "hyperpol", 0, 0, 1
    return "neutral", 0, 0, 0


def _phase2_end_sign(tw, vw, v_ref_mV, eps_mV=PHASE2_EPS_MV):
    """Compatibility wrapper for older call sites -> (Vm(t_end) - v_ref, sign label).
    Pass the SHAM value at t_end as v_ref, not the scalar rest (see the module docstring)."""
    dv = v_at_end_of_phase2(tw, vw) - float(v_ref_mV)
    return dv, phase2_end_outcome(0, dv, eps_mV)[0]


# --------------------------- SCHEMA HELPERS (pure, offline-testable) --------------------------- #

def build_row(culture, neuron, morph, layer, x_um, y_um, d_near, d_cen, r_dip, th_or, th_pos,
              n_pulses, i0_uA, seed, prep, res):
    """ONE raw CSV row in CSV_HEADER order. `prep` = prepare_cell() dict (model provenance),
    `res` = simulate_neuron() dict (outcomes). Used by the serial export AND the worker, so
    the two can never write different schemas. DeltaV/drift are kept to 1e-6 mV so the stored
    value never contradicts the label (the sign threshold is PHASE2_EPS_MV = 1e-6 mV)."""
    row = [int(culture), int(neuron), str(morph), int(layer),
           round(float(x_um), 2), round(float(y_um), 2),
           round(float(d_near), 2), round(float(d_cen), 2), round(float(r_dip), 2),
           round(float(th_or), 1), round(float(th_pos), 1),
           int(n_pulses), round(float(i0_uA), 1), int(res["fired"]), int(seed),
           prep["cell_model"], round(float(prep["v_rest"]), 4),
           round(float(prep["ctrl_drift"]), 6), round(float(res["dv_end"]), 6),
           res["outcome"], int(res["depolarized"]), int(res["hyperpolarized"])]
    row += staged_row_values(res.get("kinetics") or empty_staged())
    if len(row) != len(CSV_HEADER):
        raise AssertionError("build_row: %d values for %d columns" % (len(row), len(CSV_HEADER)))
    return row


def outcome_paths(act_csv_path):
    """{outcome: path} of the three deliverable files, derived from the ACTIVATION path by
    replacing 'Pactivation' in its FILE name (culture_Pactivation.csv -> culture_Pdepolarization.csv
    and culture_Phyperpolarization.csv, in the same directory)."""
    d, b = os.path.split(str(act_csv_path))
    if "Pactivation" not in b:
        raise ValueError("activation CSV name must contain 'Pactivation' (got %r)" % b)
    return {name: os.path.join(d, b.replace("Pactivation", stem))
            for name, (stem, _col) in OUTCOME_FILES.items()}


def outcome_columns(header, outcome_col):
    """Columns of ONE deliverable file: every non-outcome column of `header` (identity and
    provenance), then the single 0/1 `outcome_col`. Exactly one outcome column per file, so
    culture_statistics.py's rename of it to 'fired' can never create a duplicate column."""
    return [c for c in header if c not in OUTCOME_COLS] + [outcome_col]


class OutcomeWriter(object):
    """Write the per-outcome deliverable CSVs in ONE streaming pass (constant memory):
        with OutcomeWriter(header, "dir/culture_Pactivation.csv") as ow:
            for row in rows:            # lists in `header` order
                ow.write(row)
        ow.paths                        # {outcome: path} actually written
    Outcomes whose 0/1 column is absent from `header` are skipped (legacy full-active parts
    carry only 'fired' -> activation file only)."""

    def __init__(self, header, act_csv_path):
        header = list(header)
        targets = outcome_paths(act_csv_path)
        out_dir = os.path.dirname(os.path.abspath(str(act_csv_path)))
        os.makedirs(out_dir, exist_ok=True)
        self.paths, self._fh, self._wr, self._idx = {}, [], [], []
        for name, (_stem, col) in OUTCOME_FILES.items():
            if col not in header:
                continue
            cols = outcome_columns(header, col)
            # written as <name>.partial and renamed on a clean close(): a crashed run can never
            # leave a truncated culture_P*.csv that culture_statistics.py would silently read
            fh = open(targets[name] + ".partial", "w", newline="")
            wr = csv.writer(fh)
            wr.writerow(cols)
            self.paths[name] = targets[name]
            self._fh.append(fh)
            self._wr.append(wr)
            self._idx.append([header.index(c) for c in cols])
        if not self.paths:
            raise ValueError("header has none of the outcome columns %s" % (OUTCOME_COLS,))

    def write(self, row):
        for wr, idx in zip(self._wr, self._idx):
            wr.writerow([row[i] for i in idx])

    def close(self, commit=True):
        """Close all files; commit=True renames each *.partial to its final name, commit=False
        deletes the partials (used when the with-block raised). Idempotent."""
        fhs, self._fh = self._fh, []
        for fh in fhs:
            fh.close()
        if not fhs:
            return
        for p in self.paths.values():
            if commit:
                os.replace(p + ".partial", p)
            elif os.path.exists(p + ".partial"):
                os.remove(p + ".partial")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        self.close(commit=exc_type is None)
        return False


def culture_draws(seed, c, N, n_morph, span, elec, center, dip_c, dip_d, axis_deg, h_soma_um):
    """Placement of culture c. rng = default_rng(seed + c): the draws depend ONLY on (seed, c),
    never on which cultures ran before -- this is what makes the parallel split exact. The draw
    ORDER (morphology shuffle, positions, orientations) is the one every earlier version used,
    so a given (seed, c) places the somata exactly where the full-active campaign placed them.
    Returns a dict of per-neuron arrays."""
    rng = np.random.default_rng(int(seed) + int(c))
    midx = assign_morphologies(int(N), int(n_morph), rng)
    pos = rng.uniform(-span, span, size=(int(N), 2))
    theta = rng.uniform(0, 360, size=int(N))
    r_dip, th_pos = directional_rt(pos, dip_c, dip_d, z_um=h_soma_um)
    return dict(midx=midx, pos=pos, theta=theta,
                d_near=dist_from_nearest_electrode(pos, elec),
                d_cen=dist_from_center(pos, center),
                r_dip=r_dip, th_pos=th_pos, th_or=rel_orientation_deg(theta, axis_deg))


# --------------------------------- NEURON-BACKED SHARED CORE --------------------------------- #

def resolve_cell_model(cfg_or_name):
    """Validated cell-model name from a config object (its .cell_model) or a plain string.
    Fails loudly (no silent default) if missing or unknown: the model is scientifically
    meaningful and is recorded in every row."""
    name = cfg_or_name if isinstance(cfg_or_name, str) else getattr(cfg_or_name, "cell_model", None)
    if name not in CELL_MODELS:
        raise ValueError("cell_model must be one of %s, got %r (set it in config.py)"
                         % (CELL_MODELS, name))
    return name


def build_cell_for_model(asc_path, cell_model):
    """Build the Rich cell with the biophysics of `cell_model` (see the module docstring).

    full_tuned uses Rich's OWN axon, which is passive: axon_active=False drops the Eyal Na/Kv
    from the stylized AIS. That is what makes the leak tuning safe for excitability -- moving
    a passive axon's potential touches no sodium channel (measured: identical spike counts,
    peak Vm shifted by <= 0.0475 mV). The tuning itself is applied in _rest_and_sham(), after
    every channel and every config multiplier is in place.
    """
    from rich_cell import build_rich_cell
    name = resolve_cell_model(cell_model)
    return build_rich_cell(asc_path, soma_only=(name == "soma_only"),
                           axon_active=(name != "full_tuned"))


def apply_model_state(cell, cell_model, cfg, v_target_mV):
    """Per-build biophysical state that is NOT part of the morphology and must therefore be
    re-applied every time the cell is rebuilt.

    Today that is the full_tuned leak tuning: e_pas is set per segment so the net standing
    current is zero at `v_target_mV`, imposing that potential isopotentially across the whole
    arbour. It is temperature-dependent (eca is Nernst-computed), hence the explicit celsius.

    WHY THIS IS SEPARATE FROM _rest_and_sham: CellPool keeps ONE live cell and rebuilds it
    whenever the morphology or layer changes, but it computes rest and the sham only ONCE per
    (morphology, layer) and caches them. If the tuning lived only in that cached path, every
    rebuild after the first would hand back an UNTUNED cell while the stored sham still came
    from a tuned one -- and the mismatch shows up as a large, placement-independent DeltaV.
    Measured before this was split out: a spurious 0.409 mV 'bump' with near-degenerate taus,
    identical at 370 and 500 um. smoke_test_full_tuned.py now asserts that a rebuild
    reproduces the same DeltaV.

    Returns the tuning report (dict) or None when the model needs no per-build state.
    """
    if resolve_cell_model(cell_model) != "full_tuned":
        return None
    if not bool(getattr(cfg, "leak_tuning", True)):
        return None
    from rich_cell import tune_leak_isopotential
    return tune_leak_isopotential(cell, float(v_target_mV),
                                  celsius=float(getattr(cfg, "leak_tune_celsius", 37.0)))


def _rest_and_sham(cell, cell_model, cfg, morph, layer, rest_tstop_ms=1500.0):
    """Initial condition and sham reference of a freshly built cell -> dict(v_rest, v_sham_end,
    ctrl_drift).
      v_rest     : v_init of every simulation. soma_only -> the settled no-stimulus rest of THIS
                   cell (rich_cell.settled_resting_voltage, from -85 mV); full_active ->
                   cfg.v_rest_mV (the legacy init, so 'fired' reproduces the preliminary dataset)
      v_sham_end : somatic Vm at the end of phase 2 of the SAME protocol with ZERO current. It is
                   placement-independent (with I = 0 the extracellular drive is 0 everywhere).
      ctrl_drift : v_sham_end - v_rest, the init drift that DeltaV_end removes."""
    from rich_cell import settled_resting_voltage
    from rich_footprint import spikes_at
    if cell_model == "full_active":
        v_rest = float(cfg.v_rest_mV)
    else:
        v_rest = float(settled_resting_voltage(cell, tstop_ms=rest_tstop_ms, dt_ms=cfg.dt_ms))
    leak = apply_model_state(cell, cell_model, cfg, v_rest)

    bump_ms = float(getattr(cfg, "bump_ms", 0.0) or 0.0)
    kin = dict(bump_ms=bump_ms, bump_dt_ms=float(getattr(cfg, "bump_dt_ms", 0.5)),
               cvode_atol=float(getattr(cfg, "cvode_atol", 1e-6)),
               play_margin_ms=float(getattr(cfg, "play_margin_ms", 1.0)),
               t0_ms=float(getattr(cfg, "bump_t0_ms", 0.0)),
               early_floor=float(getattr(cfg, "bump_early_floor", 0.25)))
    out = spikes_at(cell, (0.0, 0.0), 0.0, i0_uA=0.0, detail=True, pre_end_ms=0.0,
                    v_init_mV=v_rest, bump_ms=bump_ms, bump_dt_ms=kin["bump_dt_ms"],
                    cvode_atol=kin["cvode_atol"], play_margin_ms=kin["play_margin_ms"])
    n0, _vmax0, tw0, vw0 = out[0], out[1], out[2], out[3]
    if n0 > 0:
        raise RuntimeError("%s L%d (%s): the cell spikes with ZERO stimulus -- the rest state "
                           "is not quiescent, every outcome would be meaningless"
                           % (morph, int(layer), cell_model))
    v_sham_end = v_at_end_of_phase2(tw0, vw0)
    if not np.isfinite(v_sham_end):
        raise RuntimeError("%s L%d: sham run returned no sample at the end of phase 2"
                           % (morph, int(layer)))
    # The sham traces are kept so DeltaV(t) can be formed over the WHOLE window, not just at
    # t_end. With I = 0 the extracellular drive is zero everywhere, so one sham serves every
    # placement of this cell. Two grids: the solver-resolution one (the fast relaxation lives
    # there) and the uniform bump_dt_ms one (the bump).
    info = dict(v_rest=v_rest, v_sham_end=float(v_sham_end),
                ctrl_drift=float(v_sham_end - v_rest), leak_report=leak, kin=kin,
                t_fine_sham=np.asarray(tw0, float), v_fine_sham=np.asarray(vw0, float))
    if bump_ms > 0:
        info["t_grid_sham"] = np.asarray(out[4], float)
        info["v_grid_sham"] = np.asarray(out[5], float)
    return info


def prepare_cell(morph, layer, cfg=None, cell_model=None, tag="_ce_", rest_tstop_ms=1500.0):
    """ONE-OFF helper for diagnostics and tests: build one (morphology x layer) cell and return
    dict(cell, morph, layer, cell_model, v_rest, v_sham_end, ctrl_drift) (see _rest_and_sham).
    Do NOT keep many of these alive in a loop -- every live cell slows every simulation; the
    drivers use CellPool instead. The temporary sliced .asc is pid-tagged and removed."""
    from config import CFG
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    cfg = CFG if cfg is None else cfg
    cell_model = resolve_cell_model(cfg if cell_model is None else cell_model)
    out = "%s%d_%s_%d.asc" % (tag, os.getpid(), morph, int(layer))
    asc = reduced_asc(find_one_morphology(morph), layer, out_path=out)
    try:
        cell = build_cell_for_model(asc, cell_model)
    finally:
        if asc and os.path.exists(asc):
            os.remove(asc)
    info = _rest_and_sham(cell, cell_model, cfg, morph, layer, rest_tstop_ms)
    return dict(cell=cell, morph=str(morph), layer=float(layer), cell_model=cell_model, **info)


def simulate_neuron(prep, pos_xy, theta_deg, i0_uA, with_kinetics=True):
    """ONE stimulation of a prepared cell (prepare_cell) placed at `pos_xy` (um), rotated by
    `theta_deg`. Same protocol as the sham (rich_footprint.spikes_at defaults, v_init = v_rest).

    Returns dict(fired, dv_end, outcome, depolarized, hyperpolarized, kinetics),
    dv_end = Vm(t_end) - Vm_sham(t_end).

    `kinetics` is a bump_kinetics.fit_staged() result: the direct post-pulse relaxation and
    the Ih bump, both measured on DeltaV(t) = Vm_stim(t) - Vm_sham(t). It is empty_staged()
    (all blank, fit_ok 0) when config.bump_ms is 0 or `with_kinetics` is False, so the row
    schema never changes. The outcome columns are NOT affected by the long window: the pulse
    is integrated at fixed dt either way and t_end precedes the switch to CVODE (verified
    identical to 0.00e+00 mV).
    """
    from rich_footprint import spikes_at
    kin = prep.get("kin") or {}
    bump_ms = float(kin.get("bump_ms", 0.0)) if with_kinetics else 0.0
    out = spikes_at(prep["cell"], (float(pos_xy[0]), float(pos_xy[1])), float(theta_deg),
                    i0_uA=float(i0_uA), detail=True, pre_end_ms=0.0,
                    v_init_mV=prep["v_rest"], bump_ms=bump_ms,
                    bump_dt_ms=float(kin.get("bump_dt_ms", 0.5)),
                    cvode_atol=float(kin.get("cvode_atol", 1e-6)),
                    play_margin_ms=float(kin.get("play_margin_ms", 1.0)))
    nsp, tw, vw = out[0], out[2], out[3]
    dv_end = v_at_end_of_phase2(tw, vw) - prep["v_sham_end"]
    label, act, depo, hypo = phase2_end_outcome(int(nsp > 0), dv_end)

    fit = empty_staged()
    if bump_ms > 0 and "t_grid_sham" in prep:
        # fine DeltaV: the leading stretch where the stimulated and the sham run share their
        # fixed-dt time base exactly, so this is a plain subtraction with no interpolation.
        k = common_prefix_len(tw, prep["t_fine_sham"])
        tf, vf = tw[:k], (vw[:k] - prep["v_fine_sham"][:k])
        tg = np.asarray(out[4], float)
        vg = tg * 0.0
        tgs, vgs = prep["t_grid_sham"], prep["v_grid_sham"]
        if tgs.shape == tg.shape and np.allclose(tgs, tg, rtol=0.0, atol=1e-9):
            vg = np.asarray(out[5], float) - vgs
        else:
            vg = np.asarray(out[5], float) - np.interp(tg, tgs, vgs)
        fit = fit_staged(tf, vf, tg, vg, t0_ms=float(kin.get("t0_ms", 0.0)),
                         early_floor=float(kin.get("early_floor", 0.25)))
    return dict(fired=act, dv_end=float(dv_end), outcome=label,
                depolarized=depo, hyperpolarized=hypo, kinetics=fit)


class CellPool(object):
    """At most ONE live NEURON cell per process -- the drivers' only way to get cells.

    WHY: NEURON integrates EVERY section that exists, so each simulation also pays for every
    other live cell. Measured (soma_only, one morphology): 1 live cell 0.52 s/sim; 6 live
    cells, each placed once (the previous worker design, which kept ALL morphology x layer
    cells) 2.96 s/sim, with bit-identical voltages. Removing 'extracellular' from idle cells
    does not help (2.71 s); destroying them does (0.40 s). So the pool destroys the live cell
    before building the next.

    Rebuilds are cheap and exact: the sliced .asc of each (morphology, layer) is cached in a
    private temp dir (scratch created and removed by THIS process), a rebuild takes ~0.1-0.2 s,
    and it reproduces v_rest, the sham and every Vm sample bit for bit -- so v_rest/v_sham_end
    are computed on the FIRST build only and reused (see smoke_test_soma_only.py).

    Never keep a reference to a pool cell (it would stay alive): use simulate(), n_spikes(),
    info(). with-statement or close() removes the scratch dir."""

    def __init__(self, cfg=None, cell_model=None, rest_tstop_ms=1500.0):
        from config import CFG
        self.cfg = CFG if cfg is None else cfg
        self.cell_model = resolve_cell_model(self.cfg if cell_model is None else cell_model)
        self.rest_tstop_ms = float(rest_tstop_ms)
        self._tmp = tempfile.mkdtemp(prefix="estim_cells_%d_" % os.getpid())
        self._asc, self._info = {}, {}
        self._key, self._cell = None, None
        self.n_builds = 0

    def _drop(self):
        self._cell, self._key = None, None
        gc.collect()                       # make sure the old cell's sections are really gone

    def _activate(self, morph, layer):
        key = (str(morph), float(layer))
        if key == self._key:
            return key
        self._drop()                       # destroy BEFORE building: never two live cells
        asc = self._asc.get(key)
        if asc is None:
            from morphologies import find_one_morphology
            from slicer import reduced_asc
            asc = reduced_asc(find_one_morphology(key[0]), key[1],
                              out_path=os.path.join(self._tmp, "%s_L%d.asc" % (key[0], int(key[1]))))
            self._asc[key] = asc
        self._cell = build_cell_for_model(asc, self.cell_model)
        self._key = key
        self.n_builds += 1
        if key not in self._info:
            # first build of this (morphology, layer): settle, tune, and run the sham
            self._info[key] = _rest_and_sham(self._cell, self.cell_model, self.cfg,
                                             key[0], key[1], self.rest_tstop_ms)
        else:
            # a REBUILD. rest and the sham are cached, but the per-build biophysical state is
            # not part of the .asc and has to be re-applied, or this cell is not the cell the
            # cached sham came from. Cheap: one finitialize + fcurrent, no 1500 ms settle.
            apply_model_state(self._cell, self.cell_model, self.cfg,
                              self._info[key]["v_rest"])
        return key

    def info(self, morph, layer):
        """Provenance of (morph, layer): dict(morph, layer, cell_model, v_rest, v_sham_end,
        ctrl_drift). Builds the cell on first use (to compute rest and sham)."""
        key = (str(morph), float(layer))
        if key not in self._info:
            self._activate(*key)
        return dict(morph=key[0], layer=key[1], cell_model=self.cell_model, **self._info[key])

    def simulate(self, morph, layer, pos_xy, theta_deg, i0_uA, with_kinetics=True):
        """Biphasic stimulation + classification (simulate_neuron) of (morph, layer) placed at
        pos_xy / theta_deg. with_kinetics=False skips the long post-pulse window for this one
        simulation (the outcome columns are unaffected)."""
        key = self._activate(morph, layer)
        return simulate_neuron(dict(cell=self._cell, **self._info[key]), pos_xy, theta_deg,
                               i0_uA, with_kinetics=with_kinetics)

    def n_spikes(self, morph, layer, pos_xy, theta_deg, i0_uA, phase="both"):
        """Somatic spike count only (phase='p1'/'p2' = monophasic delivery, for phase_split)."""
        from rich_footprint import spikes_at
        key = self._activate(morph, layer)
        return int(spikes_at(self._cell, (float(pos_xy[0]), float(pos_xy[1])), float(theta_deg),
                             i0_uA=float(i0_uA), phase=phase,
                             v_init_mV=self._info[key]["v_rest"])[0])

    def close(self):
        self._drop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def culture_has_kinetics(seed, c, fraction=1.0):
    """Whether culture `c` gets the long post-pulse window.

    Deterministic in (seed, c) and drawn from a stream INDEPENDENT of culture_draws(), so
    changing the fraction -- or turning the subsample off entirely -- cannot move a single
    soma. Subsampling by CULTURE rather than by neuron keeps the kinetics sample unbiased in
    distance and orientation, which restricting the window by position would not.
    """
    f = float(fraction)
    if f >= 1.0:
        return True
    if f <= 0.0:
        return False
    return bool(np.random.default_rng([int(seed) + int(c), 0x6B696E]).random() < f)


def iter_culture_blocks(pool, c, d, morphs, layers, i0, n_pulses, seed, block=25,
                        with_kinetics=True):
    """Simulate culture `c` (draws `d` from culture_draws) in BLOCKS of `block` neurons; yield
    (rows, n_neurons_done) after each block. Shared by the serial driver and the worker.

    Inside a block the simulations are GROUPED by (morphology, layer), so the one-live-cell
    pool rebuilds each cell at most once per block (<= n_morph x n_layers cheap rebuilds). The
    rows are returned NEURON-major (neuron i with all its layers), exactly the order every
    earlier version wrote, so a culture cut short after any block is an unbiased random subset
    of COMPLETE neurons (the flush happens between blocks, see culture_worker.py)."""
    N = len(d["midx"])
    block = max(1, int(block))
    with_kinetics = bool(with_kinetics)
    for b0 in range(0, N, block):
        idx = list(range(b0, min(N, b0 + block)))
        res = {}
        for mi, morph in enumerate(morphs):
            members = [i for i in idx if int(d["midx"][i]) == mi]
            if not members:
                continue
            for layer in layers:
                for i in members:
                    res[(i, layer)] = pool.simulate(morph, layer, d["pos"][i], d["theta"][i],
                                                    i0, with_kinetics=with_kinetics)
        rows = []
        for i in idx:
            morph = morphs[int(d["midx"][i])]
            for layer in layers:
                rows.append(build_row(c, i, morph, layer, d["pos"][i, 0], d["pos"][i, 1],
                                      d["d_near"][i], d["d_cen"][i], d["r_dip"][i],
                                      d["th_or"][i], d["th_pos"][i], n_pulses, i0, seed,
                                      pool.info(morph, layer), res[(i, layer)]))
        yield rows, idx[-1] + 1


# columns render_outcome_set() needs, in this order (depolarized/hyperpolarized nan = absent)
PLOT_COLUMNS = ("culture", "x_um", "y_um", "dist_nearest_elec_um", "dist_dipole3d_um",
                "theta_pos_deg", "layer_um", "fired", "depolarized", "hyperpolarized")


def render_outcome_set(arr, out_dir, prefix, span, bin_um, i0, n_pulses, cfg, elec, sign):
    """Quick-look maps for every outcome present in `arr` (float array, columns PLOT_COLUMNS;
    an outcome column that is all-nan -- legacy data -- is skipped). Writes
    <out_dir>/<prefix>Pmap_<outcome>.pdf; returns {outcome: pdf}. NO NEURON."""
    arr = np.asarray(arr, dtype=float)
    center = electrode_center(elec)
    dip_c, dip_d = dipole_frame(elec, sign)
    ctx = dict(elec=elec, sign=sign, center=center, dip_c=dip_c, dip_d=dip_d, span=span,
               bin_um=bin_um, edges=np.arange(0, span + bin_um, bin_um), i0=i0,
               n_pulses=n_pulses, cfg=cfg, X=arr[:, 1], Y=arr[:, 2], CULT=arr[:, 0],
               DN=arr[:, 3], RD=arr[:, 4], TP=arr[:, 5])
    out = {}
    for k, name in enumerate(("activation", "depolarization", "hyperpolarization")):
        col = arr[:, 7 + k]
        if not np.all(np.isfinite(col)):
            continue
        out[name] = render_outcome_maps(os.path.join(out_dir, "%sPmap_%s.pdf" % (prefix, name)),
                                        name, col, **ctx)
    return out


# ------------------------------------- SERIAL DRIVER ------------------------------------- #

def culture_export(n_cultures=None, neurons_per_culture=None, layers=None,
                   span_um=None, i0_uA=None, bin_um=8.0, distance_mode=None,
                   csv_path="culture_Pactivation.csv", seed=None, make_figures=True,
                   block=25):
    """SERIAL driver (ONE core): dry runs and smoke tests only. Production HPC runs use the
    parallel path (culture_worker.py + culture_merge.py), which calls the same prepare_cell /
    simulate_neuron / build_row, in the same neuron-major order, so for the same (seed, culture)
    both paths write the same rows.

    Writes the three per-outcome CSVs (names derived from `csv_path`, see outcome_paths) and,
    unless make_figures=False, three quick-look PDFs next to them. `distance_mode` is accepted
    for API compatibility only: every distance metric is written to the CSV.
    Returns (csv_activation, csv_depolarization, csv_hyperpolarization)."""
    import matplotlib; matplotlib.use("Agg")
    from config import CFG
    import field as F

    cfg = CFG
    n_cultures = cfg.n_cultures if n_cultures is None else int(n_cultures)
    seed_used = cfg.seed if seed is None else int(seed)
    N = cfg.n_neurons_effective() if neurons_per_culture is None else int(neurons_per_culture)
    layers = layers or cfg.layers_um
    span = cfg.span_half_um() if span_um is None else float(span_um)
    i0 = cfg.i0_uA if i0_uA is None else float(i0_uA)
    cell_model = resolve_cell_model(cfg)
    n_pulses = cfg.n_pulses_for_duration()
    morphs = cfg.morphologies
    M = len(morphs)
    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=not cfg.bipolar)
    center = electrode_center(elec)
    axis = dipole_axis_deg(elec, sign)
    dip_c, dip_d = dipole_frame(elec, sign)

    print(f"[culture_export SERIAL | {cell_model}] {n_cultures} cultures x {N} neurons x "
          f"{len(layers)} layers = {n_cultures * N * len(layers)} sims @ {i0:.0f} uA | "
          f"{M} morphologies | seed={seed_used} | stat over {n_pulses} pulses")
    print("  outcome: sign of DeltaV = Vm(end of phase 2) - Vm_sham(end of phase 2)")

    here = os.path.dirname(os.path.abspath(__file__))
    act_path = csv_path if os.path.isabs(csv_path) else os.path.join(here, csv_path)
    pcols = [CSV_HEADER.index(k) for k in PLOT_COLUMNS]
    plot = []
    with CellPool(cfg, cell_model) as pool:
        for layer in layers:                      # rest + sham of every cell, logged up front
            for m in morphs:
                p = pool.info(m, layer)
                print(f"  {m} L{int(layer)} um: v_rest = {p['v_rest']:.4f} mV | "
                      f"sham drift at end of phase 2 = {p['ctrl_drift']:+.6f} mV")
        with OutcomeWriter(CSV_HEADER, act_path) as ow:
            for c in range(n_cultures):
                d = culture_draws(seed_used, c, N, M, span, elec, center, dip_c, dip_d, axis,
                                  cfg.h_soma_um)
                wk = culture_has_kinetics(seed_used, c,
                                          getattr(cfg, "bump_culture_fraction", 1.0))
                for rows, _n_done in iter_culture_blocks(pool, c, d, morphs, layers, i0,
                                                         n_pulses, seed_used, block,
                                                         with_kinetics=wk):
                    for row in rows:
                        ow.write(row)
                        plot.append([float(row[j]) for j in pcols])
                print(f"  culture {c}: done{'' if wk else '  (kinetics skipped)'}")
        paths = ow.paths
        print("  cell builds: %d (one live cell at a time)" % pool.n_builds)
    print("done CSVs:", ", ".join(paths[k] for k in OUTCOME_FILES))

    if make_figures and plot:
        prefix = os.path.basename(act_path).split("Pactivation")[0]
        pdfs = render_outcome_set(np.asarray(plot, dtype=float), os.path.dirname(act_path),
                                  prefix, span, bin_um, i0, n_pulses, cfg, elec, sign)
        for pdf in pdfs.values():
            print("done PDF:", pdf)
    return paths["activation"], paths["depolarization"], paths["hyperpolarization"]


def _post_phase2_peak(t, dv, search_ms=5.0):
    """LEGACY -- NOT used by the export or by single_neuron_check (see phase2_end_outcome).
    Choose the signed response attributable to the END of phase 2.

    t is relative to the END of the biphasic pulse (t=0 = end of phase 2).
    dv is Vm - passive resting baseline.

    Criterion:
      1) inspect only 0..search_ms after pulse end;
      2) find the FIRST true local extremum after t=0 (max or min);
      3) compare it with the boundary value at 0+; if the boundary deflection is
         already larger in magnitude, keep 0+ (the phase-2 lobe peaked at pulse end);
      4) sign of the selected value: + = depol, - = hyperpol.

    Returns peak, t_peak, sign, dv_at_end.
    """
    t = np.asarray(t, float)
    dv = np.asarray(dv, float)
    m = (t >= 0.0) & (t <= float(search_ms))
    if int(m.sum()) < 2:
        return float("nan"), float("nan"), "none", float("nan")

    tt = t[m]
    yy = dv[m]
    dv0 = float(yy[0])

    # 3-point smoothing only for extremum detection; reported amplitude is RAW.
    ys = yy.copy()
    if yy.size >= 3:
        ys[1:-1] = (yy[:-2] + yy[1:-1] + yy[2:]) / 3.0

    # First local max OR min after t=0.
    cand = []
    for i in range(1, len(ys) - 1):
        is_max = ys[i] > ys[i - 1] and ys[i] >= ys[i + 1]
        is_min = ys[i] < ys[i - 1] and ys[i] <= ys[i + 1]
        if is_max or is_min:
            cand.append(i)
            break

    if cand:
        i = cand[0]
        local_peak = float(yy[i])
        # If the largest deflection is already present at pulse end, call t=0 the peak.
        if abs(dv0) >= abs(local_peak):
            peak, tpk = dv0, float(tt[0])
        else:
            peak, tpk = local_peak, float(tt[i])
    else:
        # Monotonic relaxation: the phase-2 response peaks at the pulse boundary.
        peak, tpk = dv0, float(tt[0])

    if peak > 0:
        sign = "depol"
    elif peak < 0:
        sign = "hyperpol"
    else:
        sign = "neutral"
    return float(peak), float(tpk), sign, dv0


def single_neuron_check(morphology=None, layer=None, positions=None, thetas=None, i0_uA=None,
                        t_max_ms=20.0, out_pdf="single_neuron_check.pdf",
                        out_csv="single_neuron_check.csv", soma_only=True):
    """One morphology/layer tested over several soma positions and orientations.

    Visual check of the phase-2 sign criterion used by the export. Each trace covers the
    whole 0.5-ms biphasic pulse:
        phase 1: -0.50 .. -0.25 ms
        phase 2: -0.25 ..  0.00 ms
        t = 0: END of phase 2 / END of the biphasic pulse.

    Plotted and classified: DeltaVm(t) = Vm_stim(t) - Vm_sham(t), where Vm_sham is the same
    protocol with zero current (the unstimulated trajectory from the same init) -- exactly the
    reference the export uses. Sign criterion: sign of DeltaVm(t=0) ONLY
    (+ -> depolarization, - -> hyperpolarization); no extremum search, no delayed rebound.
    The CSV also records ctrl_drift_mV = Vm_sham(0) - v_rest (the init drift removed).
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import ceil
    from config import CFG
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    from rich_cell import build_rich_cell
    from rich_footprint import spikes_at

    cfg = CFG
    morph = morphology or cfg.morphologies[0]
    layer = float(layer if layer is not None else cfg.layers_um[0])
    i0 = cfg.i0_uA if i0_uA is None else float(i0_uA)
    phi = float(cfg.phase_dur_ms)

    if positions is None:
        positions = [
            (90, -30), (-90, -30), (120, -30), (-120, -30),
            (150, -30), (-150, -30), (0, 90), (0, -150),
            (100, 60), (-100, 60), (100, -120), (-100, -120),
            (140, 40), (-140, 40), (160, -90), (-160, -90),
            (110, -110), (-110, -110), (130, 10), (-130, 10)
        ]
    if thetas is None:
        # One neuron, same morphology, sampled at clearly different orientations.
        thetas = [0.0, 45.0, 90.0, 135.0]

    asc = reduced_asc(find_one_morphology(morph), layer,
                      out_path=f"_snc_{morph}_{int(layer)}.asc")
    cell = build_rich_cell(asc, soma_only=soma_only)
    if asc and os.path.exists(asc):
        os.remove(asc)

    # IMPORTANT: a pure passive cell has only 'pas', hence its exact equilibrium is e_pas.
    # Do not initialise it at the active-model rest (~ -73.9 mV), otherwise the trace contains
    # an artificial relaxation toward -84.395 mV that can masquerade as post-pulse polarity.
    from rich_cell import settled_resting_voltage
    v_baseline = settled_resting_voltage(cell, tstop_ms=1500.0, dt_ms=cfg.dt_ms)
    # SHAM: identical protocol and window, zero current -> the unstimulated trajectory.
    # finitialize(v_baseline) is not the (spatially non-uniform) equilibrium, so Vm_sham drifts;
    # subtracting it leaves only the stimulus-evoked change (see the module docstring).
    _n0, _vm0, tw_sham, vw_sham = spikes_at(cell, (0.0, 0.0), 0.0, i0_uA=0.0, detail=True,
                                            pre_end_ms=2.0 * phi, v_init_mV=v_baseline)
    v_sham_end = v_at_end_of_phase2(tw_sham, vw_sham)
    drift = v_sham_end - v_baseline
    print(f"[single_neuron_check] rest {v_baseline:.4f} mV | sham drift at end of phase 2 "
          f"{drift:+.6f} mV")

    panels, rows = [], []
    for (x, y) in positions:
        for th in thetas:
            nsp, vmax, tw, vw = spikes_at(
                cell, (float(x), float(y)), float(th), i0_uA=i0, detail=True,
                pre_end_ms=2.0 * phi, v_init_mV=v_baseline
            )
            fired = int(nsp > 0)
            if len(vw) != len(vw_sham):
                raise RuntimeError("stim and sham traces have different lengths")
            dv = vw - vw_sham                          # stimulus-evoked DeltaVm(t)

            dv0 = v_at_end_of_phase2(tw, vw) - v_sham_end
            sgn2 = phase2_end_outcome(0, dv0)[0]       # sign only (spike shown separately)
            peak2, tpk2 = dv0, 0.0

            rows.append([
                x, y, th, fired,
                round(float(v_baseline), 4),
                round(float(dv0), 6),
                round(float(peak2), 6),
                round(float(tpk2), 4),
                sgn2,
                round(float(drift), 6)
            ])
            panels.append((x, y, th, tw, dv, peak2, tpk2, dv0, sgn2, fired))

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, out_csv), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "x_um", "y_um", "theta_deg", "fired", "baseline_mV",
            "deltaVm_at_end_mV", "phase2_peak_mV", "phase2_peak_time_ms", "sign",
            "ctrl_drift_mV"
        ])
        wr.writerows(rows)

    n = len(panels)
    ncol = 4
    nrow = int(ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.75 * nrow), squeeze=False)

    for ax, (x, y, th, tw, dv, peak2, tpk2, dv0, sgn2, fired) in zip(axes.flat, panels):
        m = (tw >= -2.0 * phi) & (tw <= t_max_ms)
        tt, yy = tw[m], dv[m]
        col = "crimson" if sgn2 == "depol" else ("steelblue" if sgn2 == "hyperpol" else "0.4")

        ax.plot(tt, yy, color=col, lw=1.25, label="DeltaVm = Vm(stim) - Vm(sham)")

        # Show exactly where the two stimulus phases occur.
        ax.axvspan(-2.0 * phi, -phi, color="royalblue", alpha=0.12, zorder=0,
                   label="phase 1 (+I)")
        ax.axvspan(-phi, 0.0, color="crimson", alpha=0.12, zorder=0,
                   label="phase 2 (-I)")
        ax.axvline(-phi, color="0.45", ls=":", lw=0.9)
        ax.axvline(0.0, color="k", ls="--", lw=1.0)
        ax.axhline(0.0, color="0.65", ls=":", lw=0.8)

        # Orange square: actual membrane deflection exactly at the end of phase 2.
        ax.scatter([0.0], [dv0], marker="s", s=28, c="darkorange", zorder=7,
                   label=f"DeltaVm(0+)={dv0:+.3f} mV")

        # Black dot: the point whose sign is actually used.
        ax.scatter([tpk2], [peak2], c="k", s=32, zorder=8,
                   label=f"SELECTED={peak2:+.3f} mV @{tpk2:.3f} ms -> {sgn2}")

        

        # Tight zoom around the full biphasic response + early post-pulse period.
        early = (tt >= -2.0 * phi) & (tt <= min(t_max_ms, 5.0))
        if early.any():
            lo, hi = float(np.min(yy[early])), float(np.max(yy[early]))
            pad = 0.18 * (hi - lo + 1e-6)
            ax.set_ylim(lo - pad, hi + pad)

        spike_txt = " . SPIKE" if fired else ""
        ax.set_title(f"({x:.0f},{y:.0f}) um . theta={th:.0f}deg . {sgn2}{spike_txt}", fontsize=7.2)
        ax.tick_params(labelsize=6)
        ax.set_xlim(-2.0 * phi - 0.05, t_max_ms)
        ax.legend(fontsize=4.8, loc="best")

    for ax in axes.flat[n:]:
        ax.axis("off")

    model_txt = "SOMA-ONLY ACTIVE" if soma_only else "FULL ACTIVE"
    fig.suptitle(
        f"Single-neuron phase-2 sign check -- {model_txt} -- morphology {morph}, "
        f"layer {int(layer)} um, +/-{i0:.0f} uA\n"
        f"phase 1 = [-{2*phi:.2f}, -{phi:.2f}] ms; phase 2 = [-{phi:.2f}, 0] ms; "
        f"t=0 = END pulse. Sign criterion: sign[Vm(t=0) - Vm_sham(t=0)]. "
        f"Black dot/square at t=0 = value that decides the sign.",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    p = os.path.join(here, out_pdf)
    fig.savefig(p)
    plt.close(fig)
    print("done:", p, "and", os.path.join(here, out_csv))
    return p


def phase_split(n_cultures=None, neurons_per_culture=None, layers=None, span_um=None, i0_uA=None):
    """MECHANISM check (separate from the deliverable): deliver each phase of the biphasic pulse
    ALONE and map which somata it activates. Phase 1 (+): cathode = sign- electrodes, dipole - -> +.
    Phase 2 (-): cathode = sign+ electrodes, dipole flips + -> -. Produces culture_phase_maps.pdf
    with the two per-soma P maps and the correct (flipped) dipole arrow for each phase.

    Caveat: phase1-only + phase2-only is NOT the full biphasic pulse (phase 1 preconditions the
    membrane for phase 2). Read these as 'what each phase excites on its own' -- the causal
    attribution of activation to a phase -- not an exact decomposition. The deliverable P (both
    phases, culture_export) remains the one the network consumes.
    Uses config.cell_model and the same v_init as the export (prepare_cell), so the phase
    attribution describes the SAME model as the deliverable."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from config import CFG
    from rich_footprint import spikes_at
    import field as F

    cfg = CFG
    n_cultures = cfg.n_cultures if n_cultures is None else int(n_cultures)
    N = (cfg.n_neurons_effective() if neurons_per_culture is None else int(neurons_per_culture))
    layers = layers or cfg.layers_um
    span = cfg.span_half_um() if span_um is None else float(span_um)
    i0 = cfg.i0_uA if i0_uA is None else float(i0_uA)
    morphs = cfg.morphologies; M = len(morphs)
    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=not cfg.bipolar)
    an = elec[sign > 0]; ca = elec[sign < 0]
    print(f"[phase_split] {n_cultures} cultures x {N} x {len(layers)} layers x 2 phases "
          f"= {n_cultures*N*len(layers)*2} sims @ {i0:.0f} uA")

    X, Y, F1, F2 = [], [], [], []
    with CellPool(cfg) as pool:                   # one live cell; grouped by (layer, morphology)
        for c in range(n_cultures):
            rng = np.random.default_rng(cfg.seed + c)
            midx = assign_morphologies(N, M, rng)
            pos = rng.uniform(-span, span, size=(N, 2)); theta = rng.uniform(0, 360, size=N)
            for layer in layers:
                for mi, m in enumerate(morphs):
                    for i in np.flatnonzero(midx == mi):
                        p = (float(pos[i, 0]), float(pos[i, 1])); th = float(theta[i])
                        f1 = int(pool.n_spikes(m, layer, p, th, i0, phase="p1") > 0)
                        f2 = int(pool.n_spikes(m, layer, p, th, i0, phase="p2") > 0)
                        X.append(p[0]); Y.append(p[1]); F1.append(f1); F2.append(f2)
            print(f"  culture {c}: done")
    X, Y, F1, F2 = map(np.asarray, (X, Y, F1, F2))

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "culture_phase_maps.pdf")
    cmap = plt.get_cmap("plasma"); margin = 0.12 * span; lim = span + margin

    def panel(ax, Fp, arrow_from, arrow_to, title):
        sx, sy, sp = per_soma_P(X, Y, Fp)
        gx = np.linspace(-lim, lim, 80); gy = np.linspace(-lim, lim, 80)
        Zf = smooth_P_field(X, Y, Fp, gx, gy, max(2 * cfg.pitch_um, 20.0))
        ax.pcolormesh(gx, gy, np.nan_to_num(Zf, nan=0.0), cmap=cmap, vmin=0, vmax=1, shading="auto", alpha=0.35, zorder=0)
        for xi, yi, pi in zip(sx, sy, sp):
            ax.add_patch(plt.Circle((xi, yi), max(0.6 * cfg.pitch_um, 12.0), color=cmap(pi), alpha=0.5, ec="none", zorder=3))
        ax.annotate("", xy=arrow_to, xytext=arrow_from, arrowprops=dict(arrowstyle="-|>", color="cyan", lw=3), zorder=6)
        for (ex, ey), s in zip(elec, sign):
            ax.scatter([ex], [ey], marker="+" if s > 0 else "_", s=110, c="k", linewidths=1.8, zorder=7)
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("x (um)"); ax.set_title(title, fontsize=9)

    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        panel(axes[0], F1, ca.mean(0), an.mean(0), "Phase 1 only (+):  dipole - -> +\ncathode = - electrodes")
        panel(axes[1], F2, an.mean(0), ca.mean(0), "Phase 2 only (-):  dipole + -> -\ncathode = + electrodes")
        axes[0].set_ylabel("y (um)")
        fig.suptitle(f"Per-phase activation (monophasic) @ {i0:.0f} uA -- somata coloured by P(spike)\n"
                     f"caveat: phase1+phase2 alone != full biphasic (phase-1 preconditions phase-2)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print("done:", out)
    return out


# ---------------- legacy per-culture renderer (mean +/- SD ACROSS cultures) ---------------- #
# Kept with its original API (smoke_test_culture_parallel.py; external callers). The HPC
# merge now renders render_outcome_maps (neuron-level, same estimator as culture_statistics).

def render_culture_figures(rows, span, bin_um, i0, n_pulses, cfg, elec, sign,
                           dip_c, dip_d, out_pdf):
    """Render the 4-page culture_Pmap.pdf from simulation rows. NO NEURON, NO simulation.

    Split out of culture_export() so the serial path and the parallel path
    (culture_worker.py + culture_merge.py) share ONE renderer instead of duplicating it.

    rows : array, columns = (culture, x, y, d_nearest, r_dipole, theta_pos, layer, fired)
    span : float, half-width (um) of the sampling box (radial bin range and plot limits)
    bin_um : float, radial bin width (um)
    i0 : float, stimulus amplitude (uA), titles only
    n_pulses : int, pulses the statistic corresponds to, titles only
    cfg : the CFG config object (reads cfg.stim_duration_s for titles)
    elec : (E,2) electrode positions; sign : (E,) signed weights (+1 anode, -1 cathode)
    dip_c : (2,) dipole centre; dip_d : (2,) unit direction pointing - -> +
    out_pdf : absolute path of the PDF to write. Returns out_pdf.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = np.array(rows)  # cols: culture, x, y, d_nearest, r_dipole, theta_pos, layer, fired
    CULT, X, Y, DN, RD, TP, lay, fired = (rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3],
                                          rows[:, 4], rows[:, 5], rows[:, 6], rows[:, 7])
    edges = np.arange(0, span + bin_um, bin_um)

    pdf_full = out_pdf
    with PdfPages(pdf_full) as pdf:
        # (1) RECRUITMENT curve: P vs 3D distance from the DIPOLE centre, mean+/-SD across cultures
        fig, ax = plt.subplots(figsize=(8, 5))
        ctr, mean, sd, ncb, stack = per_culture_P(RD, fired, CULT, edges)
        for ci in range(stack.shape[0]):
            ax.plot(ctr, stack[ci], color="0.85", lw=0.7, zorder=1)
        good = np.isfinite(mean) & (ncb >= 1)
        ax.fill_between(ctr[good], np.clip(mean - sd, 0, 1)[good], np.clip(mean + sd, 0, 1)[good],
                        color="steelblue", alpha=0.25, zorder=2, label="+/-SD across cultures")
        ax.plot(ctr[good], mean[good], "o-", color="steelblue", ms=4, zorder=3,
                label=f"mean of {stack.shape[0]} cultures")
        # fit only if the curve is (roughly) monotonic-decreasing, i.e. its peak is in the first bins
        gm = mean[good]
        monotone = gm.size >= 3 and int(np.nanargmax(gm)) <= 1
        r50, w = float(np.median(ctr[good])) if good.any() else span / 2, 15.0
        if monotone:
            try:
                r50, w = fit_logistic_decreasing(ctr[good], mean[good])
                rr = np.linspace(0, span, 200)
                ax.plot(rr, logistic_decreasing(rr, r50, w), "k--", lw=2,
                        label=f"logistic fit (r50={r50:.0f}um, w={w:.0f})")
            except Exception as e:
                print("  fit failed:", e)
        else:
            ax.text(0.5, 0.9, "non-monotonic -- no sigmoid fit", transform=ax.transAxes,
                    ha="center", fontsize=8, color="0.4")
        ax.set_xlabel("3D distance from dipole centre (um)")
        ax.set_ylabel("P(soma spike)"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Recruitment curve @ {i0:.0f} uA (over {n_pulses} pulses / {cfg.stim_duration_s:.0f}s)\n"
                     f"per-culture (grey), mean +/- SD across {stack.shape[0]} cultures "
                     f"(nearest-electrode distance is in the CSV)", fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # (1b) RECRUITMENT SIGMOID: P vs distance from the NEAREST electrode -- monotonic, fittable
        fig, ax = plt.subplots(figsize=(8, 5))
        ctr2, mean2, sd2, ncb2, stack2 = per_culture_P(DN, fired, CULT, edges)
        for ci in range(stack2.shape[0]):
            ax.plot(ctr2, stack2[ci], color="0.85", lw=0.7, zorder=1)
        good2 = np.isfinite(mean2) & (ncb2 >= 1)
        ax.fill_between(ctr2[good2], np.clip(mean2 - sd2, 0, 1)[good2], np.clip(mean2 + sd2, 0, 1)[good2],
                        color="seagreen", alpha=0.22, zorder=2, label="+/-SD across cultures")
        ax.plot(ctr2[good2], mean2[good2], "o-", color="seagreen", ms=4, zorder=3,
                label=f"mean of {stack2.shape[0]} cultures")
        try:
            r50n, wn = fit_logistic_decreasing(ctr2[good2], mean2[good2])
            rr = np.linspace(0, span, 200)
            ax.plot(rr, logistic_decreasing(rr, r50n, wn), "k--", lw=2,
                    label=f"logistic fit (r50={r50n:.0f}um, w={wn:.0f})")
        except Exception as e:
            print("  nearest fit failed:", e)
        ax.set_xlabel("distance from NEAREST electrode (um)")
        ax.set_ylabel("P(soma spike)"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Recruitment sigmoid @ {i0:.0f} uA -- nearest-electrode distance (monotonic)\n"
                     f"mean +/- SD across {stack2.shape[0]} cultures", fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # (2) directional anisotropy: P vs 3D distance, split by POSITION angle to the dipole axis,
        #     UNFOLDED so the cathode side (theta->180deg) is distinguishable from the anode side (theta->0deg)
        fig, ax = plt.subplots(figsize=(8, 5))
        obins = [(0, 60, "toward sign+ electrodes"), (60, 120, "~= perpendicular"), (120, 180, "toward sign- electrodes")]
        for lo, hi, lab in obins:
            mo = (TP >= lo) & (TP < hi)
            ctr, P, cnt = radial_bin_P(RD[mo], fired[mo], edges)
            ax.plot(ctr, P, "o-", ms=4, alpha=0.85, label=f"{lo}-{hi}deg . {lab}")
        ax.set_xlabel("3D distance from dipole centre (um)")
        ax.set_ylabel("P(soma spike)"); ax.set_ylim(-0.02, 1.02)
        ax.set_title("Directional dependence: P vs distance, by the soma's angle to the dipole axis\n"
                     "(theta=0deg toward the sign+ electrodes, theta=180deg toward sign-; the excitatory cathode "
                     "flips with the biphasic phase)", fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # (3) ACTIVATION AREA -- an AREA PER SOMA (disk coloured by its P) over a faint smooth field,
        #     with the dipole DIRECTIONS drawn as arrows (- -> +)
        sx, sy, sp = per_soma_P(X, Y, fired)
        cmap = plt.get_cmap("plasma")
        margin = 0.12 * span; lim = span + margin
        gx = np.linspace(-lim, lim, 90); gy = np.linspace(-lim, lim, 90)
        bw = max(1.5 * bin_um, 20.0)
        Zf = smooth_P_field(X, Y, fired, gx, gy, bw)
        fig, ax = plt.subplots(figsize=(7.0, 6.3))
        ax.pcolormesh(gx, gy, np.nan_to_num(Zf, nan=0.0), cmap=cmap, vmin=0, vmax=1,
                      shading="auto", zorder=0, alpha=0.35)          # faint continuous area
        soma_r = max(0.6 * bw, 12.0)                                  # radius of each soma's area
        for xi, yi, pi in zip(sx, sy, sp):                           # ONE AREA PER SOMA
            ax.add_patch(plt.Circle((xi, yi), soma_r, color=cmap(pi), alpha=0.55, ec="none", zorder=3))
        sc = ax.scatter(sx, sy, c=sp, cmap=cmap, vmin=0, vmax=1, s=14, edgecolors="w",
                        linewidths=0.4, zorder=5)                    # soma centres
        # dipole directions: an arrow from each cathode to its paired anode (- -> +)
        an = elec[sign > 0]; ca = elec[sign < 0]
        for (kx, ky) in ca:                                          # nearest anode to each cathode
            j = int(np.argmin(np.hypot(an[:, 0] - kx, an[:, 1] - ky)))
            ax.annotate("", xy=(an[j, 0], an[j, 1]), xytext=(kx, ky),
                        arrowprops=dict(arrowstyle="-|>", color="cyan", lw=2.2, shrinkA=0, shrinkB=0), zorder=7)
        for (ex, ey), s in zip(elec, sign):
            ax.scatter([ex], [ey], marker="+" if s > 0 else "_", s=120, c="k", linewidths=1.8, zorder=8)
        ax.plot([], [], color="cyan", lw=2.2, label="dipole direction (- -> +)")
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        ax.set_title(f"Activation area @ {i0:.0f} uA -- one area per soma, coloured by P(spike)\n"
                     f"cyan arrows = the three parallel dipole directions (- -> +)", fontsize=9)
        fig.colorbar(sc, ax=ax, label="P(spike)"); ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # (4) EXPLAINER: the CENTRAL dipole drawn - -> + between its two electrodes, the position
        #     vectors and their angle theta, and the example somata coloured by ACTIVATED vs silent.
        an = elec[sign > 0]; ca = elec[sign < 0]
        if len(an) and len(ca):
            ca_c = an.mean(0); ck_c = ca.mean(0)
            c_an = an[int(np.argmin(np.hypot(an[:, 0] - ca_c[0], an[:, 1] - ca_c[1])))]   # central anode (+)
            c_ka = ca[int(np.argmin(np.hypot(ca[:, 0] - ck_c[0], ca[:, 1] - ck_c[1])))]   # central cathode (-)
        else:
            c_an, c_ka = dip_c + dip_d * 30, dip_c - dip_d * 30
        sx2, sy2, sp2 = per_soma_P(X, Y, fired)
        vlen = np.hypot(sx2 - dip_c[0], sy2 - dip_c[1]); keep = vlen > 1e-6
        sx2, sy2, sp2 = sx2[keep], sy2[keep], sp2[keep]
        ang = np.degrees(np.arctan2(sy2 - dip_c[1], sx2 - dip_c[0]))
        order = np.argsort(ang); n_ex = min(20, len(order))                               # <= 20 examples
        pick = order[np.linspace(0, len(order) - 1, n_ex).astype(int)] if len(order) else []
        fired_ex = sp2 >= 0.5                                                             # "activated"
        fig, ax = plt.subplots(figsize=(6.9, 6.5))
        # central dipole: a line with an arrow from the - electrode to the + electrode
        ax.annotate("", xy=(c_an[0], c_an[1]), xytext=(c_ka[0], c_ka[1]),
                    arrowprops=dict(arrowstyle="-|>", color="cyan", lw=3.5), zorder=4)
        ax.text(*(0.5 * (c_an + c_ka) + np.array([0, 6])), "central dipole  - -> +",
                color="c", fontsize=9, ha="center", zorder=5)
        for k in pick:
            vx, vy = sx2[k] - dip_c[0], sy2[k] - dip_c[1]
            col = "crimson" if fired_ex[k] else "0.55"
            ax.annotate("", xy=(sx2[k], sy2[k]), xytext=(dip_c[0], dip_c[1]),
                        arrowprops=dict(arrowstyle="-|>", color="0.6", lw=1.1), zorder=3)
            ax.scatter([sx2[k]], [sy2[k]], s=95, c=col, edgecolors="k", linewidths=0.6, zorder=6)
            th = np.degrees(np.arctan2(abs(vx * dip_d[1] - vy * dip_d[0]), vx * dip_d[0] + vy * dip_d[1]))
            ax.text(sx2[k], sy2[k], f"  theta={th:.0f}deg", fontsize=7.5, zorder=7)
        for (ex, ey), s in zip(elec, sign):
            ax.scatter([ex], [ey], marker="+" if s > 0 else "_", s=120, c="k", linewidths=1.8, zorder=2)
        ax.scatter([dip_c[0]], [dip_c[1]], marker="o", s=40, c="k", zorder=6)
        ax.scatter([], [], c="crimson", edgecolors="k", label="activated soma")
        ax.scatter([], [], c="0.55", edgecolors="k", label="silent soma")
        ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        ax.set_title(f"theta = angle between the central dipole (- -> +) and each soma's position vector\n"
                     f"(grey arrows = position vectors from the dipole centre; {n_ex} example somata)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return pdf_full


if __name__ == "__main__":
    culture_export()
