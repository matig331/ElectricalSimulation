"""
field.py -- Extracellular field of a planar point-source electrode array.

Pure numpy, no NEURON/LFPy, no morphology: this module knows nothing about
cables. It provides three decoupled pieces (separation of concerns):

  1. geom_factor(seg_xyz, ...)   -> g(seg)      [mV per Ampere]   (geometry only)
  2. biphasic_current(t, ...)    -> I(t)        [Ampere]          (waveform only)
  3. v_ext_timeseries(...)       -> v_ext(seg,t)[mV]  = g(seg) * I(t)

Physics
-------
Point source in a HALF-space (insulating MEA substrate below, saline above):

    Ve(r) = I / (2*pi*sigma*r)            [half-space -> factor 2*pi, not 4*pi]

The neuron sits ON the array, so distances are taken in full 3D:
r = sqrt((x-xe)^2 + (y-ye)^2 + z^2), with electrodes on the z=0 plane and the
neuron translated to a chosen soma height above it. The point-source
singularity is regularized by clamping r at rmin (electrode radius).

Units: positions in micrometres [um], currents in microamperes [uA] at the API
boundary (converted to A internally), sigma in S/m, potentials returned in mV.
"""
import numpy as np

# ----------------------------------------------------------------------------- waveform
def biphasic_current(t_ms, i0_uA=50.0, phase_dur_ms=0.25, anodic_first=True,
                     ramp_us=0.0, interphase_us=0.0):
    """Symmetric, charge-balanced biphasic current, returned in AMPERES.

    positive-then-negative when anodic_first=True (matches the protocol).
    ramp_us > 0 makes each phase a TRAPEZOID with linear rise/fall of ramp_us
    (bounds dI/dt -> no Dirac-delta dVe/dt at the transitions). interphase_us
    inserts a zero-current gap between the two phases. Both phases share the same
    (opposite-sign) trapezoid, so the pulse stays charge-balanced.
    """
    t = np.asarray(t_ms, float)
    i0 = i0_uA * 1e-6
    s = 1.0 if anodic_first else -1.0
    r = min(ramp_us * 1e-3, phase_dur_ms / 2.0)          # ms, clamp to half-phase
    gap = interphase_us * 1e-3

    def trap(tt, t0, t1, amp):
        y = np.zeros_like(tt)
        if r <= 0:
            y[(tt >= t0) & (tt < t1)] = amp
            return y
        up = (tt >= t0) & (tt < t0 + r); y[up] = amp * (tt[up] - t0) / r
        pl = (tt >= t0 + r) & (tt < t1 - r); y[pl] = amp
        dn = (tt >= t1 - r) & (tt < t1); y[dn] = amp * (t1 - tt[dn]) / r
        return y

    p2 = phase_dur_ms + gap
    return trap(t, 0.0, phase_dur_ms, s * i0) + trap(t, p2, p2 + phase_dur_ms, -s * i0)

def biphasic_train(t_ms, i0_uA=50.0, phase_dur_ms=0.25, anodic_first=True,
                   ramp_us=100.0, interphase_us=0.0, n_pulses=1, isi_ms=5000.0):
    """Train of n_pulses identical biphasic pulses spaced isi_ms apart (first at t=0).
    isi_ms = 1000/freq_hz (0.2 Hz -> 5000 ms). Returns AMPERES."""
    t = np.asarray(t_ms, float)
    out = np.zeros_like(t)
    for k in range(int(n_pulses)):
        out = out + biphasic_current(t - k * isi_ms, i0_uA, phase_dur_ms,
                                      anodic_first, ramp_us, interphase_us)
    return out


# ----------------------------------------------------------------------------- geometry
def default_array(pitch_um=None, monopolar=None):
    """Electrode layout. Default = SIX electrodes, arranged like the original 4 with one extra
    per column: an ANODE column (x = -pitch/2) and a CATHODE column (x = +pitch/2), each with
    3 electrodes stacked at pitch spacing (rows y = +pitch/2, -pitch/2, -3*pitch/2). Compact,
    NOT spread. Electrode SIZE stays config.electrode_um.

    Override with config.electrodes = [(x, y, sign), ...] for the REAL device positions.

    monopolar=True: all electrodes are SOURCES (+1); the +/- comes from the biphasic I(t).
    monopolar=False (bipolar): 3 anodes (left) / 3 cathodes (right).

    Returns (elec_xy [(6,2) um], sign [(6,)]).
    """
    from config import CFG
    explicit = getattr(CFG, "electrodes", None)
    if explicit:                                   # real device geometry from config
        arr = np.asarray(explicit, float)
        return arr[:, :2].copy(), arr[:, 2].copy()
    p = (CFG.pitch_um if pitch_um is None else float(pitch_um))
    h = p / 2.0
    ys = np.array([h, -h, -3.0 * h])                     # original pair + one below
    anodes = np.column_stack([np.full(3, -h), ys])       # left column  (x = -pitch/2)
    cathodes = np.column_stack([np.full(3, +h), ys])     # right column (x = +pitch/2)
    elec = np.vstack([anodes, cathodes])
    mono = (not CFG.bipolar) if monopolar is None else bool(monopolar)
    sign = np.ones(6) if mono else np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    return elec, sign


def phase_mask(t_ms, t_on, phase_dur_ms, phase="both", interphase_us=0.0):
    """Boolean mask over time samples selecting one phase of the biphasic pulse (for delivering a
    MONOPHASIC pulse to attribute activation to a single phase). t relative to t_on: phase 1 is
    [0, phi) (positive), phase 2 is [phi+gap, 2phi+gap) (negative). "both" -> all True.
    NOTE: phase1-only + phase2-only is NOT identical to the full biphasic pulse (phase 1
    preconditions the membrane for phase 2); it is the cleanest per-phase attribution, not an
    exact decomposition."""
    t = np.asarray(t_ms, float); tr = t - float(t_on)
    phi = float(phase_dur_ms); gap = float(interphase_us) * 1e-3
    if phase == "p1":
        return (tr >= 0.0) & (tr < phi)
    if phase == "p2":
        return (tr >= phi + gap) & (tr < 2 * phi + gap)
    return np.ones_like(t, dtype=bool)


def _i_scale(n_elec, current_mode):
    """Per-electrode current -> 1; total current split equally -> 1/n_elec."""
    if current_mode == "per_electrode":
        return 1.0
    if current_mode == "total":
        return 1.0 / float(n_elec)
    raise ValueError("current_mode must be 'per_electrode' or 'total'")


def geom_factor(seg_xyz, elec_xy, sign, sigma_Sm=1.5, rmin_um=12.5):
    """Geometric transfer factor g(seg) in [mV per Ampere].

    v_ext_mV(seg, t) = g(seg) * I_A(t). Electrodes lie on the z=0 plane.

    seg_xyz : (n,3) segment midpoints [um]
    elec_xy : (E,2) electrode centres [um]
    sign    : (E,) +1 anode / -1 cathode  (unit spatial pattern)
    """
    seg = np.atleast_2d(np.asarray(seg_xyz, float))
    g = np.zeros(len(seg))
    for (ex, ey), s in zip(np.asarray(elec_xy, float), np.asarray(sign, float)):
        dx = seg[:, 0] - ex
        dy = seg[:, 1] - ey
        dz = seg[:, 2] - 0.0
        r_um = np.sqrt(dx * dx + dy * dy + dz * dz)
        r_um = np.maximum(r_um, rmin_um)
        g += s / (2.0 * np.pi * sigma_Sm * (r_um * 1e-6))   # V per A
    return g * 1e3                                           # -> mV per A


def v_ext_timeseries(seg_xyz, t_ms, elec_xy, sign, i0_uA=50.0,
                     phase_dur_ms=0.25, anodic_first=True,
                     sigma_Sm=1.5, rmin_um=12.5, current_mode="per_electrode"):
    """Full extracellular potential v_ext(seg, t) [mV], shape (n_seg, n_t).

    current_mode: 'per_electrode' (each site carries i0) or 'total' (i0 split
    equally over the sites). Configurable per the protocol ambiguity.
    """
    g = geom_factor(seg_xyz, elec_xy, sign, sigma_Sm, rmin_um)          # (n_seg,)
    I = biphasic_current(t_ms, i0_uA, phase_dur_ms, anodic_first)        # (n_t,)
    return np.outer(g, I) * _i_scale(len(np.asarray(elec_xy)), current_mode)


def static_ve_mV(xyz, elec_xy, sign, i0_uA=50.0, sigma_Sm=1.5, rmin_um=12.5,
                 current_mode="per_electrode"):
    """Closed-form Ve [mV] at points xyz for a *constant* current i0 (smoke test)."""
    g = geom_factor(xyz, elec_xy, sign, sigma_Sm, rmin_um)
    return g * (i0_uA * 1e-6) * _i_scale(len(np.asarray(elec_xy)), current_mode)
