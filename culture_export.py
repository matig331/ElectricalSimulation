"""
culture_export.py -- CULTURE-based activation-probability export (reviewer's spec).

Unit = a CULTURE: one random realisation in which ALL morphologies are present, each neuron a
random soma POSITION and random ORIENTATION. The SAME placement (position, orientation,
morphology) is then re-simulated at each LAYER thickness -> 3 simulations per culture. For every
soma we record whether it spikes when the well is stimulated at the lab amplitude, and we express
the result as a probability of spiking vs the RELATIVE DISTANCE from the centre of the electrode
array (not x,y), accounting for the neuron's ORIENTATION relative to the electrode dipole.

Statistic: the lab delivers >= 1 min of pulses (config.stim_duration_s). Because the pulses at
0.2 Hz are dynamically independent (train_invariance), each pulse gives the same per-neuron
outcome, so the whole-epoch statistic is computed from one pulse and annotated with the pulse
count that the >= 1 min epoch corresponds to (recorded in the CSV as n_pulses).

Outputs:
  culture_Pactivation.csv   : one row per (culture, neuron, layer): morphology, x, y,
                              dist_from_centre, theta, rel_theta_to_dipole, n_pulses, fired
  culture_Pmap.pdf          : P(spike) vs distance-from-centre (per layer + logistic fit),
                              P vs distance split by dipole-orientation, and one example well.

Run:   python culture_export.py            # uses config.n_cultures x n_neurons x 3 layers
Smoke: python smoke_test_culture_export.py  # offline, no NEURON
"""
import os, csv
import numpy as np

# Single source of truth for the CSV schema, shared by the serial path
# (culture_export()) and the parallel path (culture_worker.run_worker()).
# 'seed' records the RNG base actually used for that row's culture -- this is
# what lets culture_merge.py tell a genuine duplicate (same seed, same culture)
# apart from a legitimate independent replicate (different seed, same LOCAL
# culture index) when merging output from multiple seed-differentiated jobs.
CSV_HEADER = ["culture", "neuron", "morphology", "layer_um", "x_um", "y_um",
              "dist_nearest_elec_um", "dist_center_um", "dist_dipole3d_um",
              "theta_orient_deg", "theta_pos_deg", "n_pulses", "i0_uA", "fired", "seed"]

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
    P = np.array([fired[inv == k].mean() for k in range(len(uniq))])
    return uniq[:, 0], uniq[:, 1], P


def iso_radii(r50, w, levels=(0.75, 0.5, 0.25)):
    """Radii at which the fitted decreasing logistic crosses each probability level.
    P(r)=level  =>  r = r50 + w*ln((1-level)/level). Returns list of (level, radius)."""
    out = []
    for L in levels:
        L = float(np.clip(L, 1e-6, 1 - 1e-6))
        out.append((L, float(r50 + w * np.log((1.0 - L) / L))))
    return out

# ----------------------- NEURON-BACKED EXPORT ----------------------- #

def culture_export(n_cultures=None, neurons_per_culture=None, layers=None,
                   span_um=None, i0_uA=None, bin_um=8.0, distance_mode=None,
                   csv_path="culture_Pactivation.csv", seed=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from config import CFG
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    from rich_cell import build_rich_cell
    from rich_footprint import spikes_at
    import field as F

    cfg = CFG
    n_cultures = cfg.n_cultures if n_cultures is None else int(n_cultures)
    seed_used = cfg.seed if seed is None else int(seed)   # RNG base; recorded per row below
    # neurons in a culture = the biological well count (cfg.n_neurons); NOT a separate dial
    N = (cfg.n_neurons_effective() if neurons_per_culture is None else int(neurons_per_culture))
    layers = layers or cfg.layers_um
    span = cfg.span_half_um() if span_um is None else float(span_um)
    i0 = cfg.i0_uA if i0_uA is None else float(i0_uA)
    distance_mode = getattr(cfg, "culture_distance_mode", "centroid") if distance_mode is None else distance_mode
    n_pulses = cfg.n_pulses_for_duration()
    morphs = cfg.morphologies; M = len(morphs)
    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=not cfg.bipolar)
    center = electrode_center(elec); axis = dipole_axis_deg(elec, sign)
    dip_c, dip_d = dipole_frame(elec, sign)        # dipole centre + unit direction (- -> +)

    print(f"[culture_export] {n_cultures} cultures x {N} neurons x {len(layers)} layers "
          f"= {n_cultures*N*len(layers)} sims @ {i0:.0f} uA | stat over {n_pulses} pulses "
          f"({cfg.stim_duration_s:.0f} s); seed={seed_used} | all {M} morphologies per culture "
          f"| distance='{distance_mode}'")

    # cache one built cell per (morphology, layer): spikes_at re-places it per call
    cells = {}
    for layer in layers:
        for m in morphs:
            asc = reduced_asc(find_one_morphology(m), layer, out_path=f"_ce_{m}_{int(layer)}.asc")
            cells[(m, layer)] = build_rich_cell(asc)
            if asc and os.path.exists(asc):
                os.remove(asc)

    here = os.path.dirname(os.path.abspath(__file__))
    csv_full = os.path.join(here, csv_path)
    rows = []
    with open(csv_full, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for c in range(n_cultures):
            rng = np.random.default_rng(seed_used + c)         # unique culture
            midx = assign_morphologies(N, M, rng)              # all morphologies present
            pos = rng.uniform(-span, span, size=(N, 2))        # random soma positions
            theta = rng.uniform(0, 360, size=N)                # random orientations
            d_near = dist_from_nearest_electrode(pos, elec)
            d_cen = dist_from_center(pos, center)
            r_dip, th_pos = directional_rt(pos, dip_c, dip_d, z_um=cfg.h_soma_um)  # 3D dist + angle-to-dipole
            th_or = rel_orientation_deg(theta, axis)           # neuron orientation vs field
            d = {"centroid": d_cen, "nearest": d_near, "dipole": r_dip}.get(distance_mode, d_cen)
            for layer in layers:                               # SAME placement, vary layer
                for i in range(N):
                    fired = int(spikes_at(cells[(morphs[midx[i]], layer)],
                                          (float(pos[i, 0]), float(pos[i, 1])),
                                          float(theta[i]), i0_uA=i0)[0] > 0)
                    w.writerow([c, i, morphs[midx[i]], int(layer), round(float(pos[i, 0]), 2),
                                round(float(pos[i, 1]), 2), round(float(d_near[i]), 2),
                                round(float(d_cen[i]), 2), round(float(r_dip[i]), 2),
                                round(float(th_or[i]), 1), round(float(th_pos[i]), 1),
                                n_pulses, round(i0, 1), fired, seed_used])
                    rows.append((c, float(pos[i, 0]), float(pos[i, 1]), float(d_near[i]),
                                 float(r_dip[i]), float(th_pos[i]), int(layer), fired))
            print(f"  culture {c}: done")

    rows = np.array(rows)  # cols: culture, x, y, d_nearest, r_dipole, theta_pos, layer, fired
    pdf_full = render_culture_figures(rows, span, bin_um, i0, n_pulses, cfg, elec, sign,
                                      dip_c, dip_d, os.path.join(here, "culture_Pmap.pdf"))

    print("done:", csv_full, "and", pdf_full)
    return csv_full, pdf_full


def phase_split(n_cultures=None, neurons_per_culture=None, layers=None, span_um=None, i0_uA=None):
    """MECHANISM check (separate from the deliverable): deliver each phase of the biphasic pulse
    ALONE and map which somata it activates. Phase 1 (+): cathode = sign- electrodes, dipole - -> +.
    Phase 2 (-): cathode = sign+ electrodes, dipole flips + -> -. Produces culture_phase_maps.pdf
    with the two per-soma P maps and the correct (flipped) dipole arrow for each phase.

    Caveat: phase1-only + phase2-only is NOT the full biphasic pulse (phase 1 preconditions the
    membrane for phase 2). Read these as 'what each phase excites on its own' -- the causal
    attribution of activation to a phase -- not an exact decomposition. The deliverable P (both
    phases, culture_export) remains the one the network consumes."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from config import CFG
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    from rich_cell import build_rich_cell
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

    cells = {}
    for layer in layers:
        for m in morphs:
            asc = reduced_asc(find_one_morphology(m), layer, out_path=f"_ps_{m}_{int(layer)}.asc")
            cells[(m, layer)] = build_rich_cell(asc)
            if asc and os.path.exists(asc):
                os.remove(asc)

    X, Y, F1, F2 = [], [], [], []
    for c in range(n_cultures):
        rng = np.random.default_rng(cfg.seed + c)
        midx = assign_morphologies(N, M, rng)
        pos = rng.uniform(-span, span, size=(N, 2)); theta = rng.uniform(0, 360, size=N)
        for layer in layers:
            for i in range(N):
                cell = cells[(morphs[midx[i]], layer)]
                p = (float(pos[i, 0]), float(pos[i, 1])); th = float(theta[i])
                f1 = int(spikes_at(cell, p, th, i0_uA=i0, phase="p1")[0] > 0)
                f2 = int(spikes_at(cell, p, th, i0_uA=i0, phase="p2")[0] > 0)
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


if __name__ == "__main__":
    culture_export()


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
