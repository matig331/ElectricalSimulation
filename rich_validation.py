"""
rich_validation.py -- validate the built model against Rich et al. 2021 (the SOURCE
paper) and isolate the depolarization-block culprit that `intrinsic.model_diagnostic`
structurally could NOT see: it only toggled slicing x shifts, but every variant it ran
still contained the Eyal Na/Kv AIS and the stylized axon -- and Rich's axon is PASSIVE.

Stages:
  fig4             -> rich_fig4_check.pdf       Q5 : reproduce Rich Fig 4 (50/100/150 pA)
  ais_isolation    -> rich_ais_isolation.pdf    Q5-C: AIS na/kv x{1,0.5,0}; 0 == Rich's passive axon
  channel_transfer -> rich_channel_transfer.pdf Q5-D: gbar-vs-distance (Ca hotspot 360-600um, Ih Eq.4)
  af_by_domain     -> rich_af_by_domain.pdf     Q2 : AF-dVm scatter coloured by domain; corr w/ & w/o axon
  subthr_passive   -> rich_subthr_passive.pdf   Q3 : subthreshold dVm active vs passive (should be ~identical)

Run:  python rich_validation.py <stage>     (or wire the stage into pipeline.py)
Smoke test (no NEURON):  python smoke_test_rich_validation.py

Reuses: intrinsic.py, coupling_checks.py, active_cell.py, rich_stim.py, field.py, config.py.
Nothing here edits config.py or mutates the model permanently: every scan multiplies the
FRESHLY built cell in place, exactly like the existing intrinsic scans.

NOTE ON CURRENT RANGE: Rich characterises repetitive firing at 50-150 pA (= 0.05-0.15 nA)
only (Fig 4). Depolarisation block above ~300-400 pA is expected (Bianchi 2012) and is NOT
evidence of a broken model. The question these stages answer is whether the RS regime
(50-150 pA) sustains firing the way Rich Fig 4 does.
"""
import os
import numpy as np

# ------------------------------------------------------------------ #
# PURE-PYTHON HELPERS (no NEURON import here -> unit-testable offline) #
# ------------------------------------------------------------------ #

# Approximate values read off Rich et al. 2021, Fig 4E (initial instantaneous
# frequency = frequency from the cell's first two spikes). The paper reports the
# model marker "falls within the range, often near the average". These are
# visual reads of the butterfly plots (mean, low, high), in Hz, keyed by pA.
RICH_FIG4E_HZ = {
    50:  (20.0,  5.0,  55.0),
    100: (30.0, 10.0,  95.0),
    150: (40.0, 15.0, 105.0),
}


def first_isi_freq_hz(spike_times_ms):
    """Rich Fig 4E metric: instantaneous frequency from the FIRST TWO spikes (Hz).
    Returns 0.0 if fewer than two spikes."""
    st = np.asarray(spike_times_ms, dtype=float)
    if st.size < 2:
        return 0.0
    isi = st[1] - st[0]
    return 1000.0 / isi if isi > 0 else 0.0


def sustained(spike_times_ms, delay_ms, dur_ms, tail_ms=100.0):
    """True if at least one spike occurs in the last `tail_ms` of the step
    (i.e. the neuron did NOT go into depolarisation block during the step)."""
    st = np.asarray(spike_times_ms, dtype=float)
    lo, hi = delay_ms + dur_ms - tail_ms, delay_ms + dur_ms
    return bool(np.any((st >= lo) & (st < hi)))


def split_corr(x, y, is_axon):
    """Pearson corr on (x,y) for: all, no-axon, dendrites-only-mask combos.
    `is_axon` is a boolean array (True = axon segment). Returns a dict; entries
    with < 3 finite points are np.nan. Used by af_by_domain (Q2)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    is_axon = np.asarray(is_axon, bool)

    def _c(mask):
        m = mask & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return float("nan")
        xs, ys = x[m], y[m]
        if np.std(xs) == 0 or np.std(ys) == 0:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])

    all_mask = np.ones_like(is_axon)
    return {"all": _c(all_mask), "no_axon": _c(~is_axon), "axon_only": _c(is_axon)}


def hotspot_ratio(dist_um, gbar, lo=360.0, hi=600.0):
    """Ratio of median gbar INSIDE the [lo,hi] um window to median gbar OUTSIDE it.
    For Rich's Ca_LVA hot spot this should be ~100 (Ca_LVA x100 at 360-600 um apical).
    Returns np.nan if either side has no samples. Used by channel_transfer (Q5-D)."""
    dist_um = np.asarray(dist_um, float); gbar = np.asarray(gbar, float)
    inside = (dist_um >= lo) & (dist_um <= hi)
    outside = ~inside
    if inside.sum() == 0 or outside.sum() == 0:
        return float("nan")
    g_in = np.median(gbar[inside]); g_out = np.median(gbar[outside])
    if g_out == 0:
        return float("inf") if g_in > 0 else float("nan")
    return float(g_in / g_out)


def pulse_onsets(t_on_ms, n_pulses, isi_ms):
    """Onset time (ms) of each pulse in a train: t_on + k*ISI, k=0..n_pulses-1."""
    return float(t_on_ms) + np.arange(int(n_pulses)) * float(isi_ms)


def aligned_windows(t, v, onsets, pre_ms=5.0, win_ms=25.0):
    """Cut v into one equal-length slice per onset: [onset-pre, onset-pre+(pre+win)].
    Equal length (fixed #samples on a uniform grid) so pulses compare sample-by-sample."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    dt = np.median(np.diff(t)) if t.size > 1 else 1.0
    n = int(round((pre_ms + win_ms) / dt))
    out = []
    for on in np.atleast_1d(onsets):
        i0 = int(np.argmin(np.abs(t - (on - pre_ms))))
        seg = v[i0:i0 + n]
        if seg.size == n:
            out.append(seg)
    return out


def max_pulse_delta(windows):
    """Max |window[k] - window[0]| over all samples/pulses (k>=1). 0.0 if <2 windows.
    Pulse-to-pulse response drift: ~0 means the pulses are dynamically independent."""
    if len(windows) < 2:
        return 0.0
    w0 = windows[0]
    return float(max(np.max(np.abs(w - w0)) for w in windows[1:]))


# ------------------------------------------------------------------ #
# NEURON-BACKED STAGES (lazy imports so the helpers above stay offline) #
# ------------------------------------------------------------------ #

def _prep(prefs, layers):
    from config import CFG
    cfg = CFG
    return cfg, (prefs or cfg.morphologies), (layers or cfg.layers_um)


def _build(pref, layer):
    """Build the pipeline cell for (morphology, layer). Returns (cell, asc_path_to_remove_or_None)."""
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    from rich_cell import build_rich_cell
    asc = reduced_asc(find_one_morphology(pref), layer, out_path="_rv.asc")
    return build_rich_cell(asc), asc


# ---------- Q5: Fig 4 reproduction (focus: Fig 4B = 100 pA trace) ----------
def fig4(prefs=None, layers=None, pA=(50, 100, 150),
         delay=200.0, dur=700.0, post=100.0):
    """Reproduce Rich Fig 4A-C/E: repetitive firing at 50/100/150 pA.
    Left panel = Vm(t) at 100 pA (the Fig 4B analogue). Right panel = initial
    instantaneous frequency vs current, with Rich Fig 4E mean/range overlaid, and a
    'sustained?' flag (did it stay out of depol block for the whole step)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from intrinsic import _run_iclamp, _spike_times

    cfg, prefs, layers = _prep(prefs, layers)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rich_fig4_check.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                cell, asc = _build(pref, layer)
                v100 = t100 = None
                f_init, sust = [], []
                for pa in pA:
                    t, v, _ic = _run_iclamp(cell, pa / 1000.0, delay=delay, dur=dur,
                                            post=post, v_init=cfg.v_rest_mV)
                    st = _spike_times(t, v)
                    f_init.append(first_isi_freq_hz(st))
                    sust.append(sustained(st, delay, dur))
                    if pa == 100:
                        t100, v100 = t, v
                if asc and os.path.exists(asc):
                    os.remove(asc)

                fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
                if v100 is not None:
                    ax[0].plot(t100, v100, lw=0.6, color="crimson")
                ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("Vm (mV)")
                ax[0].set_title("100 pA repetitive firing (cf. Rich Fig 4B)", fontsize=9)

                xs = np.array(pA, float)
                ax[1].plot(xs, f_init, "o-", color="k", label="model (1/first-ISI)")
                for pa in pA:
                    if pa in RICH_FIG4E_HZ:
                        mean, lo, hi = RICH_FIG4E_HZ[pa]
                        ax[1].errorbar(pa, mean, yerr=[[mean - lo], [hi - mean]],
                                       fmt="s", color="steelblue", capsize=4, alpha=0.7)
                ax[1].plot([], [], "s", color="steelblue", label="Rich Fig 4E (mean/range)")
                for pa, s in zip(pA, sust):
                    if not s:
                        ax[1].annotate("BLOCK", (pa, 0), color="red", fontsize=7, ha="center")
                ax[1].set_xlabel("injected current (pA)")
                ax[1].set_ylabel("initial inst. frequency (Hz)")
                ax[1].set_title("initial f vs I  (BLOCK = silent by end of step)", fontsize=9)
                ax[1].legend(fontsize=7)
                fig.suptitle(f"{pref} L{int(layer)} - Rich Fig 4 reproduction", fontweight="bold")
                fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: f_init(50/100/150pA)="
                      f"{[round(f,1) for f in f_init]} Hz, sustained={sust}")
    print("done:", out); return out


# ---------- Q5-C: isolate the AIS (na/kv). factor 0.0 == Rich's passive axon ----------
def ais_isolation(prefs=None, layers=None, factors=(1.0, 0.5, 0.0),
                  pA=(50, 100, 150, 200, 300), delay=200.0, dur=700.0, post=100.0):
    """Steady-state f-I for the AIS Na/Kv scaled by each factor. factor=0.0 makes the
    axon passive == Rich's ACTUAL configuration (the accessible proxy for a native-Rich
    control when only the stylized axon is available). If AIS=0.0 sustains RS firing at
    50-150 pA while AIS=1.0 over-fires or blocks, the added AIS is the culprit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from intrinsic import _run_iclamp, _spike_times, _rate_windows, scale_ais

    cfg, prefs, layers = _prep(prefs, layers)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rich_ais_isolation.pdf")
    xs = np.array(pA, float)
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                fig, ax = plt.subplots(figsize=(8, 5))
                for fct in factors:
                    cell, asc = _build(pref, layer)
                    scale_ais(cell, fct)
                    ss = []
                    for pa in pA:
                        t, v, _ic = _run_iclamp(cell, pa / 1000.0, delay=delay, dur=dur,
                                                post=post, v_init=cfg.v_rest_mV)
                        _n, _fi, f_ss = _rate_windows(t, v, delay, dur)
                        ss.append(f_ss)
                    if asc and os.path.exists(asc):
                        os.remove(asc)
                    lbl = f"AIS x{fct}" + ("  (== Rich passive axon)" if fct == 0.0 else "")
                    ax.plot(xs, ss, "o-", label=lbl)
                    del cell
                ax.axhspan(10, 40, color="0.9", zorder=0, label="RS band ~10-40 Hz (Rich Fig 4)")
                ax.axvspan(50, 150, color="#eef", zorder=0, label="Rich characterised range")
                ax.set_xlabel("injected current (pA)")
                ax.set_ylabel("steady-state rate (Hz, last 100 ms)")
                ax.set_title(f"{pref} L{int(layer)} - AIS na/kv isolation "
                             f"(0.0 = Rich's passive axon)", fontsize=9)
                ax.legend(fontsize=7); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: ais_isolation done")
    print("done:", out); return out


# ---------- Q5-D: are Rich's distance-based distributions transferred correctly? ----------
def channel_transfer(prefs=None, layers=None):
    """Check whether Rich's morphology-specific channel distributions survive the port to
    each Eyal morphology: (a) the Ca_LVA 'hot spot' (x100 at 360-600 um apical) and
    (b) the Ih exponential rise along the apical tree (Rich Eq 4, whose '1000 um'
    denominator was the distance to the most distal dendrite of *Rich's* cell)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from neuron import h
    from active_cell import segment_coords

    cfg, prefs, layers = _prep(prefs, layers)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rich_channel_transfer.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                cell, asc = _build(pref, layer)
                _coords, refs = segment_coords(cell)
                h.distance(0, cell.soma[0](0.5))          # set origin at soma
                dist, gca, gih, is_apic = [], [], [], []
                for seg in refs:
                    nm = seg.sec.name()
                    d = h.distance(1, seg)
                    dist.append(d)
                    is_apic.append("apic" in nm)
                    gca.append(getattr(seg, "gCa_LVAstbar_Ca_LVAst", 0.0)
                               if h.ismembrane("Ca_LVAst", sec=seg.sec) else 0.0)
                    gih.append(getattr(seg, "gIhbar_Ih", 0.0)
                               if h.ismembrane("Ih", sec=seg.sec) else 0.0)
                if asc and os.path.exists(asc):
                    os.remove(asc)
                dist = np.array(dist); gca = np.array(gca); gih = np.array(gih)
                is_apic = np.array(is_apic, bool)
                dmax = dist[is_apic].max() if is_apic.any() else dist.max()
                ratio = hotspot_ratio(dist[is_apic], gca[is_apic]) if is_apic.any() else float("nan")

                fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
                ax[0].scatter(dist[is_apic], gca[is_apic], s=8, alpha=0.5)
                ax[0].axvspan(360, 600, color="orange", alpha=0.15, label="Rich hotspot window")
                ax[0].set_xlabel("path distance from soma (um)")
                ax[0].set_ylabel("gCa_LVAst (S/cm2)")
                ax[0].set_title(f"Ca_LVA hotspot: in/out ratio ~{ratio:.0f} (Rich target ~100)",
                                fontsize=9)
                ax[0].legend(fontsize=7)
                ax[1].scatter(dist[is_apic], gih[is_apic], s=8, alpha=0.5)
                ax[1].set_xlabel("path distance from soma (um)")
                ax[1].set_ylabel("gIh (S/cm2)")
                ax[1].set_title(f"Ih apical gradient (Rich Eq4 norm = 1000 um; "
                                f"this morph max apic = {dmax:.0f} um)", fontsize=9)
                fig.suptitle(f"{pref} L{int(layer)} - channel distribution transfer check",
                             fontweight="bold")
                fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
                flag = "" if 700 <= dmax <= 1300 else "  <-- apical extent far from Rich's ~1000 um"
                print(f"  {pref} L{int(layer)}: Ca hotspot ratio~{ratio:.0f}, "
                      f"apic_dmax={dmax:.0f}um{flag}")
    print("done:", out); return out


# ---------- Q2: AF-dVm scatter, coloured by compartment domain ----------
def af_by_domain(prefs=None, layers=None, i0_sub=8.0):
    """Same AF vs subthreshold-dVm as coupling_checks.check_activating_function, but each
    segment is labelled by domain (soma/apic/dend/axon). Prints corr(all) vs corr(no axon)
    vs corr(axon only) so you can SEE whether the AF~0 vertical stripe is the axon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from active_cell import segment_coords, _soma_center
    from coupling_checks import _dvm_subthreshold
    import field as F

    cfg, prefs, layers = _prep(prefs, layers)
    elec, sign = F.default_array(monopolar=not cfg.bipolar)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rich_af_by_domain.pdf")
    DOMAINS = ("soma", "apic", "dend", "axon")
    COLORS = {"soma": "k", "apic": "tab:red", "dend": "tab:blue", "axon": "tab:green"}
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                cell, asc = _build(pref, layer)
                coords, refs = segment_coords(cell)
                sc = _soma_center(cell, coords, refs)
                placed = coords + (np.array([40.0, 0.0, 10.0]) - sc)
                Ve = F.geom_factor(placed, elec, sign, sigma_Sm=cfg.sigma_Sm,
                                   rmin_um=cfg.rmin_um) * i0_sub
                # second difference of Ve along arc length, within each section
                AF = np.zeros(len(refs)); bysec = {}
                for i, s in enumerate(refs):
                    bysec.setdefault(s.sec, []).append(i)
                for sec, idxs in bysec.items():
                    idxs = sorted(idxs, key=lambda j: refs[j].x)
                    if len(idxs) < 3:
                        continue
                    arc = np.array([refs[j].x for j in idxs]) * sec.L
                    ve = Ve[idxs]
                    for m in range(1, len(idxs) - 1):
                        ds1 = arc[m] - arc[m - 1]; ds2 = arc[m + 1] - arc[m]
                        if ds1 > 0 and ds2 > 0:
                            AF[idxs[m]] = 2 * (ve[m - 1] / (ds1 * (ds1 + ds2))
                                               - ve[m] / (ds1 * ds2)
                                               + ve[m + 1] / (ds2 * (ds1 + ds2)))
                dvm_peak, _res = _dvm_subthreshold(cell, coords, refs, i0_sub, cfg)
                if asc and os.path.exists(asc):
                    os.remove(asc)

                dom = np.array(["soma" if "soma" in r.sec.name()
                                else "apic" if "apic" in r.sec.name()
                                else "axon" if "axon" in r.sec.name()
                                else "dend" for r in refs])
                keep = AF != 0
                cc = split_corr(AF[keep], dvm_peak[keep], dom[keep] == "axon")

                fig, ax = plt.subplots(figsize=(7.5, 5.2))
                for d in DOMAINS:
                    m = keep & (dom == d)
                    if m.any():
                        ax.scatter(AF[m], dvm_peak[m], s=8, alpha=0.5,
                                   color=COLORS[d], label=f"{d} (n={int(m.sum())})")
                ax.axvline(0, color="0.8", lw=0.8)
                ax.set_xlabel("activating function d2Ve/ds2 (a.u.)")
                ax.set_ylabel("subthreshold dVm (mV)")
                ax.set_title(f"{pref} L{int(layer)}  corr all={cc['all']:.2f}  "
                             f"no-axon={cc['no_axon']:.2f}  axon-only={cc['axon_only']:.2f}",
                             fontsize=9)
                ax.legend(fontsize=7); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: corr all={cc['all']:.2f}, "
                      f"no_axon={cc['no_axon']:.2f}, axon_only={cc['axon_only']:.2f}")
    print("done:", out); return out


# ---------- Q3: is the subthreshold dVm essentially passive? ----------
def subthr_passive(prefs=None, layers=None, i0_sub=8.0):
    """Compute the subthreshold dVm at the pulse peak with active channels, then again
    on a passivised copy (active gbar zeroed), and correlate. r~1 & slope~1 confirms the
    subthreshold response is passive electrotonics (Ih frozen at 0.25 ms, per Rich)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from active_cell import segment_coords
    from coupling_checks import _dvm_subthreshold, _make_passive

    cfg, prefs, layers = _prep(prefs, layers)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rich_subthr_passive.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                cell, asc = _build(pref, layer)
                coords, refs = segment_coords(cell)
                dvm_active, _ = _dvm_subthreshold(cell, coords, refs, i0_sub, cfg)
                _make_passive(cell)
                dvm_passive, _ = _dvm_subthreshold(cell, coords, refs, i0_sub, cfg)
                if asc and os.path.exists(asc):
                    os.remove(asc)
                m = np.isfinite(dvm_active) & np.isfinite(dvm_passive)
                r = float(np.corrcoef(dvm_active[m], dvm_passive[m])[0, 1]) if m.sum() > 3 else float("nan")
                slope = float(np.polyfit(dvm_passive[m], dvm_active[m], 1)[0]) if m.sum() > 3 else float("nan")

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(dvm_passive[m], dvm_active[m], s=8, alpha=0.5)
                lim = np.array([np.nanmin(dvm_passive[m]), np.nanmax(dvm_passive[m])])
                ax.plot(lim, lim, "r--", lw=0.8, label="identity")
                ax.set_xlabel("dVm passive (mV)"); ax.set_ylabel("dVm active (mV)")
                ax.set_title(f"{pref} L{int(layer)} - subthreshold is passive?  "
                             f"r={r:.3f}, slope={slope:.3f}", fontsize=9)
                ax.legend(fontsize=7); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: r={r:.3f}, slope={slope:.3f} "
                      f"({'passive' if r > 0.98 else 'active channels matter'})")
    print("done:", out); return out


# ---------- train: are the 36 lab pulses (0.2 Hz) dynamically independent? ----------
def train_invariance(prefs=None, layers=None, n_pulses=3, isi_ms=None,
                     spacing_cap_ms=800.0, pos_xy=(40.0, 0.0), theta_deg=0.0,
                     i0_uA=None, pre_ms=5.0, win_ms=25.0, reset_eps_mV=0.5):
    """Does the lab train (0.2 Hz, 36 pulses, 3 min) behave like a single pulse repeated?
    Runs n_pulses and checks (a) every pulse gives the SAME soma/AIS response and (b) Vm
    returns to rest before the next pulse.

    ISI: defaults to the config's OWN value cfg.isi_ms (= 1000/stim_freq_hz = 5000 ms at
    0.2 Hz) -- NOT a hard-coded number. But 5 s of dead time per pulse is heavy in RAM/time
    AND >>14x the slowest recovery tau in the model (Ih ~350 ms; CaDynamics decay 80 ms),
    so the SIMULATED spacing is capped at spacing_cap_ms (default 800 ms, still >2x that tau):
    invariance at 800 ms a fortiori implies invariance at 5000 ms (more recovery, easier).
    Pass spacing_cap_ms=None to simulate the literal lab ISI (heavy).

    MEMORY: unlike stimulate_rich (which records v_all = every segment x every timestep, and
    would blow up over a train), this records ONLY the soma and AIS Vm.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from neuron import h
    import field as F
    from rich_stim import _place

    cfg, prefs, layers = _prep(prefs, layers)
    isi_lab = cfg.isi_ms if isi_ms is None else float(isi_ms)
    sim_isi = isi_lab if spacing_cap_ms is None else min(isi_lab, float(spacing_cap_ms))
    i0_uA = cfg.i0_uA if i0_uA is None else float(i0_uA)
    phase, dt = cfg.phase_dur_ms, cfg.dt_ms
    ramp_us, interph = cfg.ramp_us, cfg.interphase_us
    baseline_ms, post_ms = cfg.baseline_ms, cfg.post_ms
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rich_train_invariance.pdf")

    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                cell, asc = _build(pref, layer)
                coords, refs, _sc = _place(cell, pos_xy, theta_deg)
                elec, sign = F.default_array(monopolar=not cfg.bipolar)
                g = F.geom_factor(coords, elec, sign, sigma_Sm=cfg.sigma_Sm,
                                  rmin_um=cfg.rmin_um)                    # mV/A

                t_on = baseline_ms
                train_span = (n_pulses - 1) * sim_isi + 2 * phase
                t = np.arange(0.0, baseline_ms + train_span + post_ms + dt, dt)
                I = F.biphasic_train(t - t_on, i0_uA, phase, True, ramp_us, interph,
                                     n_pulses, sim_isi)                   # A

                tvec = h.Vector(t); keep = [tvec]
                for sec in set(s.sec for s in refs):
                    if not h.ismembrane("extracellular", sec=sec):
                        sec.insert("extracellular")
                for k, seg in enumerate(refs):
                    vv = h.Vector(g[k] * I)
                    vv.play(seg._ref_e_extracellular, tvec, True); keep.append(vv)

                soma_i = [i for i, s in enumerate(refs) if "soma" in s.sec.name()]
                ais_i = [i for i, s in enumerate(refs) if "axon" in s.sec.name()]
                si = soma_i[len(soma_i) // 2] if soma_i else 0
                ai = ais_i[0] if ais_i else si
                v_soma = h.Vector().record(refs[si]._ref_v)
                v_ais = h.Vector().record(refs[ai]._ref_v)
                tr = h.Vector().record(h._ref_t)
                h.celsius = 37; h.dt = dt; h.tstop = t[-1]
                h.finitialize(cfg.v_rest_mV); h.continuerun(t[-1])
                tt = np.asarray(tr); vs = np.asarray(v_soma); va = np.asarray(v_ais)
                if asc and os.path.exists(asc):
                    os.remove(asc)
                del cell

                onsets = pulse_onsets(t_on, n_pulses, sim_isi)
                w_soma = aligned_windows(tt, vs, onsets, pre_ms, win_ms)
                d_soma = max_pulse_delta(w_soma)
                pre_idx = [int(np.argmin(np.abs(tt - (on - dt)))) for on in onsets[1:]]
                reset_ok = (all(abs(vs[i] - cfg.v_rest_mV) < reset_eps_mV for i in pre_idx)
                            if pre_idx else True)
                lat = []
                for w in w_soma:
                    cr = np.where((w[:-1] < 0.0) & (w[1:] >= 0.0))[0]
                    lat.append(cr[0] * dt - pre_ms if cr.size else np.nan)

                fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
                taxis = np.arange(len(w_soma[0])) * dt - pre_ms if w_soma else []
                for k, w in enumerate(w_soma):
                    ax[0].plot(taxis, w, lw=0.9, alpha=0.8, label=f"pulse {k + 1}")
                ax[0].set_xlabel("time from pulse onset (ms)"); ax[0].set_ylabel("soma Vm (mV)")
                ax[0].set_title(f"per-pulse soma Vm overlaid  (max p2p dVm = {d_soma:.3f} mV)",
                                fontsize=9); ax[0].legend(fontsize=7)
                ax[1].plot(tt, vs, lw=0.5, color="crimson", label="soma")
                ax[1].plot(tt, va, lw=0.5, color="steelblue", alpha=0.7, label="AIS")
                for on in onsets:
                    ax[1].axvline(on, color="0.85", lw=0.6, zorder=0)
                ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("Vm (mV)")
                ax[1].set_title(f"full train (sim ISI={sim_isi:.0f} ms; lab ISI={isi_lab:.0f} ms)",
                                fontsize=9); ax[1].legend(fontsize=7)
                verdict = "INDEPENDENT" if (d_soma < 0.5 and reset_ok) else "CHECK drift/reset"
                fig.suptitle(f"{pref} L{int(layer)} - pulse-to-pulse invariance: {verdict}",
                             fontweight="bold")
                fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: max_p2p_dVm={d_soma:.3f} mV, "
                      f"reset_before_next={reset_ok}, "
                      f"latencies={[round(x, 2) if x == x else None for x in lat]} ms "
                      f"(sim {sim_isi:.0f}/lab {isi_lab:.0f} ms) -> {verdict}")
    print("done:", out); return out


_STAGES = {"fig4": fig4, "ais_isolation": ais_isolation, "channel_transfer": channel_transfer,
           "af_by_domain": af_by_domain, "subthr_passive": subthr_passive,
           "train_invariance": train_invariance}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in _STAGES:
        print("stages:", ", ".join(_STAGES)); sys.exit(1)
    _STAGES[sys.argv[1]]()
