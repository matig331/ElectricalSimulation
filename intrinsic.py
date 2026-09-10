"""
intrinsic.py -- intrinsic excitability: f-I curve + AP features (validation of the
active model AND the protocol to fit AdEx/CAdEx, a la Brette & Gerstner 2005).

Injects a somatic current step (IClamp), sweeps amplitude, measures firing rate
(f-I curve, rheobase), and extracts the first-AP features (threshold, peak,
amplitude, half-width, AHP). One page per morphology.

Run:  python intrinsic.py              # uses CFG
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from neuron import h

from config import CFG
from morphologies import find_one_morphology
from slicer import reduced_asc
from rich_cell import build_rich_cell


def _run_iclamp(cell, amp_nA, delay=100.0, dur=600.0, post=100.0, dt=0.025, v_init=-73.9):
    ic = h.IClamp(cell.soma[0](0.5)); ic.delay = delay; ic.dur = dur; ic.amp = amp_nA
    v = h.Vector().record(cell.soma[0](0.5)._ref_v); t = h.Vector().record(h._ref_t)
    h.celsius = 37; h.dt = dt; h.finitialize(v_init); h.continuerun(delay + dur + post)
    return np.asarray(t), np.asarray(v), ic


def _spike_times(t, v, thr=0.0):
    cr = np.where((v[:-1] < thr) & (v[1:] >= thr))[0]
    return t[cr]


def _rate_windows(t, v, delay, dur):
    """firing rate in the first vs last 100 ms of the step (init vs steady-state)."""
    st = _spike_times(t, v)
    n_init = np.sum((st >= delay) & (st < delay + 100))
    n_ss = np.sum((st >= delay + dur - 100) & (st < delay + dur))
    return len(st), n_init / 0.1, n_ss / 0.1


def scale_soma_k(cell, factor):
    """Scale somatic+apical SKv3_1 (fast repolarizing K) by `factor` in place."""
    from neuron import h
    for sec in cell.all:
        nm = sec.name()
        if ("soma" in nm or "apic" in nm) and h.ismembrane("SKv3_1", sec=sec):
            for seg in sec:
                seg.gSKv3_1bar_SKv3_1 *= factor


def scale_soma_na(cell, factor):
    """Scale somatic+apical NaTa_t by `factor` in place (lower to reduce firing rate)."""
    from neuron import h
    for sec in cell.all:
        nm = sec.name()
        if ("soma" in nm or "apic" in nm) and h.ismembrane("NaTa_t", sec=sec):
            for seg in sec:
                seg.gNaTa_tbar_NaTa_t *= factor


def scale_ais(cell, factor):
    """Scale the stylized axon (AIS) Na/Kv densities by `factor` in place."""
    from neuron import h
    for sec in cell.all:
        if "axon" not in sec.name():
            continue
        for seg in sec:
            for mech, var in (("na", "gbar_na"), ("kv", "gbar_kv")):
                if h.ismembrane(mech, sec=sec):
                    try:
                        setattr(seg, var, getattr(seg, var) * factor)
                    except Exception:
                        pass


def _ap_features(t, v, dt):
    dvdt = np.gradient(v, t)
    up = np.where(dvdt > 20.0)[0]           # threshold: dV/dt > 20 mV/ms
    if up.size == 0:
        return None
    thr_i = up[0]; thr = v[thr_i]
    pk_i = thr_i + int(np.argmax(v[thr_i:thr_i + int(5/dt)]))
    peak = v[pk_i]; amp = peak - thr
    half = thr + amp/2
    seg = v[thr_i:pk_i + int(5/dt)]
    above = np.where(seg >= half)[0]
    width = (above[-1]-above[0])*dt if above.size else np.nan
    ahp = v[pk_i:pk_i + int(50/dt)].min() - thr
    return dict(threshold=thr, peak=peak, amplitude=amp, half_width_ms=width, ahp=ahp)


def main(prefs=None, amps_nA=(0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2)):
    cfg = CFG
    prefs = prefs or cfg.morphologies
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "intrinsic.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in cfg.layers_um:
                asc = reduced_asc(find_one_morphology(pref), layer, out_path="_intr.asc")
                cell = build_rich_cell(asc)
                dur = 600.0; delay = 100.0
                f_init = []; f_ss = []; trace_supra = None
                for a in amps_nA:
                    t, v, _ = _run_iclamp(cell, a, delay=delay, dur=dur, v_init=cfg.v_rest_mV)
                    ntot, fi, fs = _rate_windows(t, v, delay, dur)
                    f_init.append(fi); f_ss.append(fs)
                    if trace_supra is None and ntot >= 2:
                        trace_supra = (t, v, a)
                os.remove(asc)
                f_init = np.array(f_init); f_ss = np.array(f_ss)
                rheobase = next((amps_nA[k] for k in range(len(amps_nA)) if f_init[k] > 0), None)
                # depolarization block: fires initially but silent at steady state
                db = [amps_nA[k] for k in range(len(amps_nA)) if f_init[k] > 0 and f_ss[k] == 0]
                db_onset = min(db) if db else None
                fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
                fig.suptitle(f"specimen_{pref} - intrinsic (layer {int(layer)}um) - "
                             f"rheobase ~ {rheobase} nA - depol-block onset: {db_onset} nA",
                             fontweight="bold")
                ax[0].plot(amps_nA, f_init, "o-", label="initial (first 100ms)")
                ax[0].plot(amps_nA, f_ss, "s--", color="crimson", label="steady-state (last 100ms)")
                if db_onset is not None:
                    ax[0].axvspan(db_onset, amps_nA[-1], color="0.9", zorder=0, label="depol block")
                ax[0].set_xlabel("I (nA)"); ax[0].set_ylabel("firing rate (Hz)")
                ax[0].set_title("f-I: initial vs steady-state", fontsize=9); ax[0].legend(fontsize=7)
                if trace_supra:
                    t, v, a = trace_supra; m = (t > 95) & (t < 160)
                    ax[1].plot(t[m], v[m], lw=0.8)
                    feat = _ap_features(t[m], v[m], cfg.dt_ms)
                    ax[1].set_title(f"AP @ {a} nA: " + (", ".join(f"{k}={np.round(val,1)}"
                                    for k, val in feat.items()) if feat else "-"), fontsize=7)
                ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("Vm (mV)")
                fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: rheobase~{rheobase}nA, depol-block@{db_onset}nA, "
                      f"f_ss_max={f_ss.max():.0f}Hz")
    print("done:", out)
    return out


def _ss_fI(cell, cfg, amps_nA):
    out = []
    for a in amps_nA:
        t, v, _ = _run_iclamp(cell, a, dur=600.0, v_init=cfg.v_rest_mV)
        _, _, fs = _rate_windows(t, v, 100.0, 600.0); out.append(fs)
    return out


def ais_scan(prefs=None, layers=None, factors=(1.0, 0.5, 0.25),
             amps_nA=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2)):
    """Steady-state f-I for several AIS Na/Kv scale factors, ALL morphologies x layers."""
    cfg = CFG; prefs = prefs or cfg.morphologies; layers = layers or cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "intrinsic_ais_scan.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                fig, ax = plt.subplots(figsize=(7, 4.6))
                for fct in factors:
                    asc = reduced_asc(find_one_morphology(pref), layer, out_path="_ais.asc")
                    cell = build_rich_cell(asc); scale_ais(cell, fct)
                    ax.plot(amps_nA, _ss_fI(cell, cfg, amps_nA), "o-", label=f"AIS x{fct}")
                    os.remove(asc)
                ax.set_xlabel("I (nA)"); ax.set_ylabel("steady-state rate (Hz)")
                ax.set_title(f"specimen_{pref} L{int(layer)} - AIS density vs depol block", fontsize=10)
                ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: ais_scan done")
    print("done:", out); return out


def somatic_k_scan(prefs=None, layers=None, mults=(1.0, 1.5, 2.0),
                   amps_nA=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2)):
    """Steady-state f-I for several somatic SKv3_1 multipliers, ALL morphologies x layers."""
    cfg = CFG; prefs = prefs or cfg.morphologies; layers = layers or cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "intrinsic_somatic_k_scan.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                fig, ax = plt.subplots(figsize=(7, 4.6))
                for m in mults:
                    asc = reduced_asc(find_one_morphology(pref), layer, out_path="_ksc.asc")
                    cell = build_rich_cell(asc); scale_soma_k(cell, m)
                    ax.plot(amps_nA, _ss_fI(cell, cfg, amps_nA), "o-", label=f"K_soma x{m}")
                    os.remove(asc)
                ax.set_xlabel("I (nA)"); ax.set_ylabel("steady-state rate (Hz)")
                ax.set_title(f"specimen_{pref} L{int(layer)} - somatic K vs depol block", fontsize=10)
                ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: somatic_k_scan done")
    print("done:", out); return out


def excitability_scan(prefs=None, layers=None, k_mults=(1.5, 2.0),
                      na_mults=(1.0, 0.7, 0.5),
                      amps_nA=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2)):
    """2D calibration: steady-state f-I for combinations of somatic K up x Na down.
    Goal: a regular-spiking curve (monotone, plateau ~20-30 Hz, block only at high I).
    ALL morphologies x layers."""
    cfg = CFG; prefs = prefs or cfg.morphologies; layers = layers or cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "intrinsic_excitability_scan.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                fig, ax = plt.subplots(figsize=(8, 5))
                for km in k_mults:
                    for nm in na_mults:
                        asc = reduced_asc(find_one_morphology(pref), layer, out_path="_exc.asc")
                        cell = build_rich_cell(asc)
                        scale_soma_k(cell, km); scale_soma_na(cell, nm)
                        ax.plot(amps_nA, _ss_fI(cell, cfg, amps_nA), "o-",
                                label=f"K x{km} / Na x{nm}")
                        os.remove(asc)
                ax.axhspan(10, 30, color="0.9", zorder=0, label="RS target ~10-30 Hz")
                ax.set_xlabel("I (nA)"); ax.set_ylabel("steady-state rate (Hz)")
                ax.set_title(f"specimen_{pref} L{int(layer)} - excitability calibration (K up x Na down)",
                             fontsize=9)
                ax.legend(fontsize=7, ncol=2); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: excitability_scan done")
    print("done:", out); return out


def model_diagnostic(prefs=None, layers=None,
                     amps_nA=(0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2)):
    """WHERE does the depolarization block come from? Isolate our pipeline additions
    one at a time and compare steady-state f-I:
      (1) full pipeline (sliced + shifts)   (2) shifts OFF
      (3) no slicing (full 3D)              (4) no slicing + shifts OFF
    If (2)/(3)/(4) remove/soften the block -> it is OUR modification (justified fix).
    If all block -> it is the base Rich channel set (retuning must be justified).
    One page per morphology x layer.  NOTE: does NOT change config; restores shifts.
    """
    from neuron import h
    cfg = CFG; prefs = prefs or cfg.morphologies; layers = layers or cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "intrinsic_model_diagnostic.pdf")
    variants = [("full pipeline (sliced, shifts on)", "sliced", True),
                ("shifts OFF (sliced)", "sliced", False),
                ("no slicing (full 3D, shifts on)", "full", True),
                ("no slicing + shifts OFF", "full", False)]
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                fig, ax = plt.subplots(figsize=(8, 5))
                for label, morph_mode, shifts_on in variants:
                    if morph_mode == "sliced":
                        asc = reduced_asc(find_one_morphology(pref), layer, out_path="_diag.asc"); rm = True
                    else:
                        asc = find_one_morphology(pref); rm = False
                    cell = build_rich_cell(asc)
                    h.shift_NaTa_t = 5.0 if shifts_on else 0.0
                    h.shift_SKv3_1 = 10.0 if shifts_on else 0.0
                    ax.plot(amps_nA, _ss_fI(cell, cfg, amps_nA), "o-", label=label)
                    del cell
                    if rm:
                        os.remove(asc)
                h.shift_NaTa_t = 5.0; h.shift_SKv3_1 = 10.0   # restore pipeline default
                ax.axhspan(10, 30, color="0.9", zorder=0, label="RS target ~10-30 Hz")
                ax.set_xlabel("I (nA)"); ax.set_ylabel("steady-state rate (Hz)")
                ax.set_title(f"specimen_{pref} L{int(layer)} - what causes the depol block?", fontsize=9)
                ax.legend(fontsize=7); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: model_diagnostic done")
    print("done:", out); return out


if __name__ == "__main__":
    main()
