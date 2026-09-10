"""
active_cell.py -- Eyal ACTIVE model under extracellular stimulation.

Builds the Eyal et al. 2018 active neuron (passive dendrites with spine
correction; Na+Kv at soma and axon; stylized AIS/axon from create_axon) and
drives it with the SAME extracellular field as the passive pipeline
(field.py) -- only the cable/solver changes (native NEURON active integration
instead of LFPy passive). Activation here is a real SPIKE, not a Delta Vm proxy.

Needs the compiled channels: run once, in this folder,
    nrnivmodl eyal_active            (produces eyal_active/x86_64/...)
or keep the provided eyal_active/x86_64. NEURON auto-loads them from CWD.

Run:
    cd estim
    python active_cell.py            # one cell, reports spike count under +/-50 uA
    python active_cell.py 60308 40   # specimen substring, amplitude uA
"""
import os
import sys
import glob
import numpy as np
from neuron import h

import field as F
from morphologies import find_one_morphology

_HOC_LOADED = False
TEMPLATE = "model_0603_cell08_cm045"
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_hoc():
    global _HOC_LOADED
    if _HOC_LOADED:
        return
    # load compiled mechanisms from eyal_active/ if present
    for sub in ("eyal_active", "."):
        so = glob.glob(os.path.join(_HERE, sub, "*", "libnrnmech.so")) \
            + glob.glob(os.path.join(_HERE, sub, "*", ".libs", "libnrnmech.so"))
        if so:
            try:
                h.nrn_load_dll(so[0])
            except Exception:
                pass
            break
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file(os.path.join(_HERE, "eyal_active", f"{TEMPLATE}.hoc"))
    _HOC_LOADED = True


def build_active_cell(asc_path):
    """Instantiate the Eyal active cell from a Neurolucida .asc.
    Order matches Eyal Fig3a: geom_nseg -> create_axon -> biophys -> active_biophys.
    define_shape() then gives the stylized axon 3D coordinates (schematic layout)."""
    _load_hoc()
    cell = getattr(h, TEMPLATE)()
    nl = h.Import3d_Neurolucida3()
    nl.input(asc_path)
    h.Import3d_GUI(nl, 0).instantiate(cell)
    # d_lambda segmentation (lambda_f = 100 Hz), consistent with the passive
    # pipeline, replacing Eyal's geom_nseg (L/40). Uses the Eyal passive Ra/cm
    # so lambda_f is correct before biophys() re-applies them.
    for sec in cell.all:
        sec.Ra = 203.23
        sec.cm = 0.45234
        sec.nseg = int((sec.L / (0.1 * h.lambda_f(100.0, sec=sec)) + 0.9) / 2) * 2 + 1
    cell.create_axon()
    cell.biophys()
    cell.active_biophys()
    h.define_shape()            # 3D coords for the (unplaced) stylized axon
    return cell


def segment_coords(cell):
    """(coords (n,3) [um], segref list) for every segment with 3D geometry."""
    coords, refs = [], []
    for sec in cell.all:
        n = sec.n3d()
        if n < 2:
            continue
        arc = np.array([sec.arc3d(i) for i in range(n)])
        xs = np.array([sec.x3d(i) for i in range(n)])
        ys = np.array([sec.y3d(i) for i in range(n)])
        zs = np.array([sec.z3d(i) for i in range(n)])
        Ltot = arc[-1] if arc[-1] > 0 else sec.L
        for seg in sec:
            a = seg.x * Ltot
            coords.append((np.interp(a, arc, xs), np.interp(a, arc, ys), np.interp(a, arc, zs)))
            refs.append(seg)
    return np.asarray(coords, float), refs


def _soma_center(cell, coords, refs):
    si = [i for i, s in enumerate(refs) if "soma" in s.sec.name()]
    return coords[si].mean(axis=0)


def stimulate(cell, pos_xy=(80.0, 60.0), h_soma_um=10.0, i0_uA=50.0,
              phase_dur_ms=0.25, anodic_first=True, current_mode="per_electrode",
              sigma_Sm=1.5, rmin_um=12.5, tstop_ms=10.0, dt_ms=0.025, theta_deg=0.0,
              monopolar=False):
    """Apply one biphasic monopolar pulse via e_extracellular; return
    (t [ms], v_soma [mV], n_spikes, v_all (n_seg,n_t), refs). No injected current.

    theta_deg rotates the whole cell (dendrites + stylized axon) about the soma
    in the array plane -- orientation matters for extracellular activation.
    n_spikes counts SOMATIC action potentials (upward crossings of 0 mV at the
    soma), which are real APs, not the passive near-field blow-up on dendrites.
    """
    coords, refs = segment_coords(cell)
    shift = np.array([pos_xy[0], pos_xy[1], h_soma_um]) - _soma_center(cell, coords, refs)
    coords = coords + shift                                   # place soma at target
    if theta_deg:                                            # rotate about the soma (z axis)
        th = np.radians(theta_deg)
        c, s = np.cos(th), np.sin(th)
        o = np.array([pos_xy[0], pos_xy[1], 0.0])
        d = coords - o
        coords = np.column_stack([o[0] + c * d[:, 0] - s * d[:, 1],
                                  o[1] + s * d[:, 0] + c * d[:, 1],
                                  coords[:, 2]])

    elec, sign = F.default_array(monopolar=monopolar)
    g = F.geom_factor(coords, elec, sign, sigma_Sm=sigma_Sm, rmin_um=rmin_um)  # mV/A
    t = np.arange(0.0, tstop_ms + dt_ms, dt_ms)
    I = F.biphasic_current(t, i0_uA, phase_dur_ms, anodic_first)               # A
    scale = 1.0 if current_mode == "per_electrode" else 1.0 / len(elec)

    tvec = h.Vector(t)
    keep = [tvec]                                             # keep refs alive!
    secs = set(s.sec for s in refs)
    for sec in secs:
        if not h.ismembrane("extracellular", sec=sec):
            sec.insert("extracellular")
    for k, seg in enumerate(refs):
        ve = h.Vector(g[k] * scale * I)                      # mV over time
        ve.play(seg._ref_e_extracellular, tvec, True)
        keep.append(ve)

    v_all = [h.Vector().record(seg._ref_v) for seg in refs]
    vs = h.Vector().record(cell.soma[0](0.5)._ref_v)
    tr = h.Vector().record(h._ref_t)
    h.celsius = 37
    h.v_init = -86
    h.tstop = tstop_ms
    h.dt = dt_ms
    h.finitialize(-86)
    h.run()

    v = np.asarray(vs)
    n_spikes = int(np.sum((v[:-1] < 0) & (v[1:] >= 0)))
    return np.asarray(tr), v, n_spikes, np.asarray(v_all), refs


if __name__ == "__main__":
    pref = sys.argv[1] if len(sys.argv) > 1 else "60308"
    i0 = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    asc = find_one_morphology(pref)
    cell = build_active_cell(asc)
    ntot = sum(s.nseg for s in cell.all)
    t, v, nsp, _, _ = stimulate(cell, i0_uA=i0)
    print(f"{os.path.basename(os.path.dirname(asc))} | {ntot} segments | "
          f"+/-{i0:.0f} uA -> spikes={nsp}, Vmax={v.max():.1f} mV, Vrest={v[0]:.1f} mV")
