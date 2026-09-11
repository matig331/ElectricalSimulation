"""
rich_cell.py -- the REAL human L5 active model (Rich et al. 2021).

Placement + reference gbar transcribed from human_L5_ion_channels.md sec 4.1
(HumanCell/L5_params_exponential.hoc), applied on the Eyal morphology loader
(Import3d + stylized axon). Rich leaves the axon passive; per the user's
request we KEEP Eyal's Na/Kv on the stylized AIS so spikes can initiate there.

Domains (by section name): soma / apic (apical) / dend (basal) / axon (AIS).
  R0 all   : pas (g=1.75e-5, e=-84.395), Ra=495.73, cm=1.5967 (soma cm=1.0)
  R1 soma  : NaTa_t(2.1) Nap_Et2 K_Pst K_Tst SKv3_1 SK_E2 Ca_LVAst Ca_HVA
             Ih CaDynamics_E2   (NOT Im)
  R2 apic  : NaTa_t(0.001) SKv3_1 SK_E2 Ca_LVAst Ca_HVA Im Ih CaDynamics_E2
             (NOT K_Pst/K_Tst); Ih exp gradient; Ca hot spot 360-600 um
  R3 dend  : Ih only, uniform = somatic value
  R4 axon  : Eyal na(200)/kv(100) on the AIS  (Rich = passive; kept per request)

Modes of build_rich_cell(asc, passive_only=False, soma_only=False):
  default      : R0-R4 above (the full-active model of the preliminary campaign).
  soma_only    : R0 everywhere + R1 on the soma ONLY; apical, basal and the stylized
                 axon/AIS stay passive (pas only). This is config.cell_model = "soma_only".
  passive_only : R0 only (control).

Segmentation: d_lambda (lambda_f=100 Hz).  ENa=+50, EK=-85 (Rich K), ek=-90 (Eyal kv);
shift_NaTa_t=5, shift_SKv3_1=10.

Needs the compiled mechanisms in ../rich_mech (11 Rich + na + kv) and the Eyal
template hoc in eyal_active/.

Run:  cd estim && python rich_cell.py     # builds 60308, prints rest + segment count
"""
import os
import glob
import numpy as np
from neuron import h

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = "model_0603_cell08_cm045"
_LOADED = False

# reference maximal conductances (S/cm^2), sec 4.1
GBAR = dict(Ca_LVAst=9.9839e-4, Ca_HVA=5.6938e-9, Ih=5.135e-5, Im=2e-4,
            K_Pst=0.065, K_Tst=2e-5, SK_E2=2.4536e-9, SKv3_1=0.04,
            Nap_Et2=1e-6, NaTa_t_soma=2.1, NaTa_t_apic=0.001)
ENA, EK = 50.0, -85.0
XMAX = 1000.0


def _load():
    global _LOADED
    if _LOADED:
        return
    pats = ["arm64/libnrnmech.dylib", "x86_64/libnrnmech.dylib", "x86_64/libnrnmech.so",
            "arm64/libnrnmech.so",
            "rich_mech/*/libnrnmech.so", "rich_mech/*/libnrnmech.dylib",
            "../rich_mech/*/libnrnmech.so", "../rich_mech/*/libnrnmech.dylib",
            "*/libnrnmech.so", "*/libnrnmech.dylib"]
    so = []
    for p in pats:
        so += glob.glob(os.path.join(_HERE, p))
    if not so:
        raise RuntimeError(
            "compiled mechanisms not found. In this folder run:  nrnivmodl rich_mech\n"
            "(it writes ./x86_64/ or ./arm64/ HERE, not inside rich_mech/).")
    try:
        h.nrn_load_dll(so[0])
    except Exception:
        pass                                 # already auto-loaded from CWD ./arch
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file(os.path.join(_HERE, "eyal_active", f"{TEMPLATE}.hoc"))
    _LOADED = True


def _dlambda(sec):
    return int((sec.L / (0.1 * h.lambda_f(100.0, sec=sec)) + 0.9) / 2) * 2 + 1


def _gbar(seg, suffix, value):
    setattr(seg, f"g{suffix}bar_{suffix}", value)


def build_rich_cell(asc_path, passive_only=False, soma_only=False):
    _load()
    if passive_only and soma_only:
        raise ValueError("Choose either passive_only=True or soma_only=True, not both")
    cell = getattr(h, TEMPLATE)()
    nl = h.Import3d_Neurolucida3(); nl.input(asc_path)
    h.Import3d_GUI(nl, 0).instantiate(cell)

    # passive + d_lambda (Rich passive so lambda_f is right)
    for sec in cell.all:
        sec.Ra = 495.73
        sec.cm = 1.5967
        sec.nseg = _dlambda(sec)
    cell.create_axon()
    for sec in cell.axon:
        sec.Ra = 495.73; sec.cm = 1.5967; sec.nseg = _dlambda(sec)
    try:
        from config import CFG as _CFG
        _al = float(getattr(_CFG, "axon_len_um", 0.0) or 0.0)
    except Exception:
        _al = 0.0
    if _al > 0:
        _tot = sum(s.L for s in cell.axon)
        if _tot > 0:
            _f = _al / _tot
            for s in cell.axon:
                s.L = s.L * _f; s.nseg = _dlambda(s)
        print(f"[rich_cell] axon shortened to ~{_al:.0f} um (was {_tot:.0f})")
    h.define_shape()

    h.shift_NaTa_t = 5.0
    h.shift_SKv3_1 = 10.0
    cell.soma[0].push(); h.distance(); h.pop_section()   # origin at soma(0)

    for sec in cell.all:
        name = sec.name()
        sec.insert("pas")
        for seg in sec:
            seg.g_pas = 1.75e-5
            seg.e_pas = -84.395

        if "soma" in name:
            sec.cm = 1.0

        # Passive-only control: keep only pas + membrane capacitance.
        if passive_only:
            continue

        # Soma-only active model: retain the full Rich active channel set ONLY in soma.
        # Apical/basal dendrites and the stylized axon remain passive.
        if soma_only and "soma" not in name:
            continue

        if "soma" in name:
            for m in ("NaTa_t", "Nap_Et2", "K_Pst", "K_Tst", "SKv3_1", "SK_E2",
                      "Ca_LVAst", "Ca_HVA", "Ih", "CaDynamics_E2"):
                sec.insert(m)
            sec.ena = ENA; sec.ek = EK
            for seg in sec:
                _gbar(seg, "NaTa_t", GBAR["NaTa_t_soma"]); _gbar(seg, "Nap_Et2", GBAR["Nap_Et2"])
                _gbar(seg, "K_Pst", GBAR["K_Pst"]); _gbar(seg, "K_Tst", GBAR["K_Tst"])
                _gbar(seg, "SKv3_1", GBAR["SKv3_1"]); _gbar(seg, "SK_E2", GBAR["SK_E2"])
                _gbar(seg, "Ca_LVAst", GBAR["Ca_LVAst"]); _gbar(seg, "Ca_HVA", GBAR["Ca_HVA"])
                _gbar(seg, "Ih", GBAR["Ih"])
                seg.decay_CaDynamics_E2 = 460.0
                seg.gamma_CaDynamics_E2 = 0.000501

        elif "apic" in name:
            for m in ("NaTa_t", "SKv3_1", "SK_E2", "Ca_LVAst", "Ca_HVA",
                      "Im", "Ih", "CaDynamics_E2"):
                sec.insert(m)
            sec.ena = ENA; sec.ek = EK
            for seg in sec:
                x = h.distance(seg.x, sec=sec)
                _gbar(seg, "NaTa_t", GBAR["NaTa_t_apic"])
                _gbar(seg, "SKv3_1", GBAR["SKv3_1"]); _gbar(seg, "SK_E2", GBAR["SK_E2"])
                _gbar(seg, "Im", GBAR["Im"])
                _gbar(seg, "Ih", GBAR["Ih"] * (-0.8696 + 2.0870 * np.exp(3.6161 * x / XMAX)))
                if 360.0 < x < 600.0:
                    _gbar(seg, "Ca_LVAst", 100.0 * GBAR["Ca_LVAst"])
                    _gbar(seg, "Ca_HVA", 10.0 * GBAR["Ca_HVA"])
                else:
                    _gbar(seg, "Ca_LVAst", GBAR["Ca_LVAst"])
                    _gbar(seg, "Ca_HVA", GBAR["Ca_HVA"])
                seg.decay_CaDynamics_E2 = 460.0
                seg.gamma_CaDynamics_E2 = 0.000501

        elif "dend" in name:                              # basal: Ih only, uniform
            sec.insert("Ih")
            for seg in sec:
                _gbar(seg, "Ih", GBAR["Ih"])

        elif "axon" in name:                              # AIS: keep Eyal na/kv
            sec.insert("na"); sec.insert("kv")
            sec.ena = ENA; sec.ek = -90.0
            for seg in sec:
                seg.gbar_na = 200.0
                seg.gbar_kv = 100.0
    # Apply channel multipliers to whichever active compartments actually exist.
    if not passive_only:
        try:
            from config import CFG as _CFG
            _km = float(getattr(_CFG, "k_soma_mult", 1.0) or 1.0)
            _nm = float(getattr(_CFG, "ais_na_mult", 1.0) or 1.0)
            _vm = float(getattr(_CFG, "ais_kv_mult", 1.0) or 1.0)
            _sn = float(getattr(_CFG, "na_soma_mult", 1.0) or 1.0)
        except Exception:
            _km = _nm = _vm = _sn = 1.0
        if any(x != 1.0 for x in (_km, _nm, _vm, _sn)):
            for sec in cell.all:
                _n = sec.name()
                for seg in sec:
                    if ("soma" in _n or "apic" in _n):
                        if _km != 1.0 and h.ismembrane("SKv3_1", sec=sec):
                            seg.gSKv3_1bar_SKv3_1 *= _km
                        if _sn != 1.0 and h.ismembrane("NaTa_t", sec=sec):
                            seg.gNaTa_tbar_NaTa_t *= _sn
                    if "axon" in _n:
                        if _nm != 1.0 and h.ismembrane("na", sec=sec):
                            seg.gbar_na *= _nm
                        if _vm != 1.0 and h.ismembrane("kv", sec=sec):
                            seg.gbar_kv *= _vm

    return cell


def settled_resting_voltage(cell, tstop_ms=1500.0, dt_ms=0.025, v_init_mV=-85.0):
    """Return the no-stimulus settled somatic Vm for this exact cell model.

    Useful for reduced biophysics (e.g. soma-only active), whose equilibrium need not
    equal the full-model cfg.v_rest_mV or the pure-passive e_pas. This is intended to
    be called ONCE per cached morphology x layer, not once per neuron placement.
    """
    h.celsius = 37
    h.dt = float(dt_ms)
    h.tstop = float(tstop_ms)
    vs = h.Vector().record(cell.soma[0](0.5)._ref_v)
    # NOTE: no h.cvode.use_fast_imem(1) here. Nothing reads i_membrane_, and the flag is
    # GLOBAL: left on, it made every later stimulation ~5% slower (measured).
    h.finitialize(float(v_init_mV))
    h.continuerun(float(tstop_ms))
    v = np.asarray(vs, dtype=float)
    if v.size == 0:
        raise RuntimeError("Could not determine settled resting voltage")
    return float(v[-1])


def resting_check(cell, tstop_ms=100.0, dt_ms=0.025):
    """Run tstop with no stimulus; report soma Vm settling and net membrane current."""
    h.celsius = 37
    h.dt = dt_ms
    vs = h.Vector().record(cell.soma[0](0.5)._ref_v)
    im = h.Vector().record(cell.soma[0](0.5)._ref_i_membrane_) \
        if hasattr(cell.soma[0](0.5), "_ref_i_membrane_") else None
    h.finitialize(-85.0)
    try:
        h.cvode.use_fast_imem(1)
    except Exception:
        pass
    h.continuerun(tstop_ms)
    v = np.asarray(vs)
    return v


if __name__ == "__main__":
    import sys
    sys.path.insert(0, _HERE)
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    asc = reduced_asc(find_one_morphology("60308"), 120.0, out_path="_rich.asc")
    cell = build_rich_cell(asc, soma_only=True)
    nseg = sum(s.nseg for s in cell.all)
    v = resting_check(cell)
    print(f"Rich cell | {nseg} segments | Vm(0)={v[0]:.2f} -> Vm(100ms)={v[-1]:.2f} mV | "
          f"drift={v[-1]-v[0]:+.2f} mV | min/max over 100ms = {v.min():.2f}/{v.max():.2f}")
    os.remove(asc)
