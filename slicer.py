"""
slicer.py -- 3D -> 2D morphology reduction by z-slab (point 2).

Replicates the slicing of the original scripts (stim_3d.py / stimolazione.py):

  * a section is kept iff it is NOT an axon, its parent is already kept
    (connectivity from the soma is preserved), and at least `thr` of its
    arc-length falls inside a z-slab of thickness `layer` (um) centred on the
    soma (the fraction is `_frac_in_slab`, the old `fin`).
  * `layer`  = slab thickness -> controls complexity: thin slab keeps only the
    near-planar sections (simpler / more 2D), thick slab keeps almost the full
    tree.  `thr` = minimum in-slab fraction to keep a section (old THRESH=0.80).

Unlike the old code (which only projected polylines for plotting), this builds
an actual reduced NEURON/LFPy morphology via morphio surgery, so the dynamic
Vm = Vi - Ve can be run on each reduced cell. `flatten=True` additionally
collapses z onto the soma plane (the true 2D model used downstream).

Run:
    cd estim
    python slicer.py                 # smoke test + slice_report.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.backends.backend_pdf import PdfPages

import morphio
from morphio import SectionType
from morphio.mut import Morphology as MutMorphology
morphio.set_maximum_warnings(0)

from morphologies import all_morphologies, find_one_morphology
from field import default_array

DEFAULT_LAYERS = (40.0, 80.0, 120.0)   # slab thicknesses (um), old LAYERS
DEFAULT_THR = 0.80                      # old THRESH


# ----------------------------------------------------------------------
# core slicing  (replicates tree / fin / keep from the originals)
# ----------------------------------------------------------------------
def _read_tree(asc):
    """soma_center (3,), and dict id -> {type, pts(3D), parent_id, children_ids}."""
    m = morphio.Morphology(asc)
    sc = np.asarray(m.soma.points, float).mean(0)
    S = {}
    for s in m.iter():
        S[s.id] = dict(type=s.type,
                       pts=np.asarray(s.points, float),
                       parent=(None if s.is_root else s.parent.id),
                       children=[c.id for c in s.children])
    return sc, S


def _frac_in_slab(pts, zc, T):
    """Old `fin`: fraction of a section's arc-length inside [zc-T/2, zc+T/2]."""
    z = pts[:, 2]
    if len(z) < 2:
        return 1.0 if abs(z[0] - zc) <= T / 2 else 0.0
    lo, hi = zc - T / 2, zc + T / 2
    L = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = 0.0
    for i in range(len(z) - 1):
        a, b = sorted((z[i], z[i + 1]))
        ov = max(0.0, min(b, hi) - max(a, lo))
        f = ov / (b - a) if b > a else (1.0 if lo <= a <= hi else 0.0)
        s += f * L[i]
    return s / L.sum() if L.sum() > 0 else 0.0


def kept_ids(S, zc, layer, thr):
    """Old `keep`: connected subtree of non-axon sections with frac >= thr."""
    K = set()
    stack = [i for i, s in S.items() if s["parent"] is None]
    while stack:
        i = stack.pop()
        s = S[i]
        if s["type"] == SectionType.axon:
            continue
        parent_ok = (s["parent"] is None) or (s["parent"] in K)
        if parent_ok and _frac_in_slab(s["pts"], zc, layer) >= thr:
            K.add(i)
            stack.extend(s["children"])
    return K


def sliced_polylines(asc, layer, thr=DEFAULT_THR):
    """Old `proj`: [(pts2D relative to soma-xy, type)] for the kept sections."""
    sc, S = _read_tree(asc)
    K = kept_ids(S, sc[2], layer, thr)
    return [(S[i]["pts"][:, :2] - sc[:2], S[i]["type"]) for i in K]


# ----------------------------------------------------------------------
# reduced morphology (morphio surgery)  -> loadable by LFPy
# ----------------------------------------------------------------------
def reduced_asc(asc, layer, thr=DEFAULT_THR, flatten=False, out_path=None):
    """Write a reduced .asc keeping only the in-slab, non-axon subtree.

    flatten=True collapses every point's z onto the soma plane (2D model).
    Returns out_path.
    """
    sc, S = _read_tree(asc)
    K = kept_ids(S, sc[2], layer, thr)

    mut = MutMorphology(asc)
    # topmost dropped sections: id not in K but parent kept/root -> delete subtree
    to_delete = []
    for s in mut.iter():
        if s.id in K:
            continue
        parent_kept = s.is_root or (s.parent.id in K)
        if parent_kept:
            to_delete.append(s)
    for s in to_delete:
        mut.delete_section(s, recursive=True)

    if flatten:
        zc = float(sc[2])
        for s in mut.iter():
            p = np.asarray(s.points, float)
            p[:, 2] = zc
            s.points = p
        sp = np.asarray(mut.soma.points, float)
        sp[:, 2] = zc
        mut.soma.points = sp

    mut.remove_unifurcations()
    if out_path is None:
        tag = f"L{int(layer)}_{'2d' if flatten else '3d'}"
        base = os.path.splitext(os.path.basename(asc))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(asc)) or ".",
                                f"{base}__{tag}.asc")
    mut.write(out_path)
    return out_path


# ----------------------------------------------------------------------
# Vm on a reduced cell (reuses field / cell / stimulate)
# ----------------------------------------------------------------------
def vm_on_reduced(asc_reduced, i0_uA=50.0, current_mode="per_electrode"):
    """Load the reduced morphology, run one monopolar biphasic pulse.

    Returns (cell, absdvm[(n_seg,)]).  absdvm = peak |Vm - v_rest| per segment.
    """
    elec, sign = default_array(monopolar=True)
    from cell import load_cell            # lazy: only the passive path needs LFPy
    from stimulate import simulate_stim, peak_response
    cell = load_cell(asc_reduced)
    _, vmem, v_rest = simulate_stim(cell, elec, sign, i0_uA=i0_uA,
                                    current_mode=current_mode)
    pk = peak_response(vmem, v_rest)
    return cell, pk["absdvm"]


# ----------------------------------------------------------------------
# report: 3D + 2D per morphology across layers, coloured by peak |dVm|
# ----------------------------------------------------------------------
def _segments_2d(cell):
    x, y = np.asarray(cell.x), np.asarray(cell.y)
    return np.stack([np.stack([x[:, 0], y[:, 0]], 1),
                     np.stack([x[:, 1], y[:, 1]], 1)], 1)


def _segments_3d(cell):
    x, y, z = np.asarray(cell.x), np.asarray(cell.y), np.asarray(cell.z)
    return np.stack([np.stack([x[:, 0], y[:, 0], z[:, 0]], 1),
                     np.stack([x[:, 1], y[:, 1], z[:, 1]], 1)], 1)


def report(morphs=None, layers=DEFAULT_LAYERS, thr=DEFAULT_THR,
           flatten=False, out_pdf=None):
    here = os.path.dirname(os.path.abspath(__file__))
    if morphs is None:
        morphs = all_morphologies()
    if out_pdf is None:
        out_pdf = os.path.join(here, "slice_report.pdf")
    with PdfPages(out_pdf) as pdf:
        for name, asc in morphs:
            n = len(layers)
            fig = plt.figure(figsize=(4.2 * n, 8))
            fig.suptitle(f"{name} -- 3D reduced (top) / 2D projection (bottom), "
                         f"coloured by peak |DeltaVm|", fontweight="bold")
            for j, T in enumerate(layers):
                red = reduced_asc(asc, T, thr, flatten=flatten,
                                  out_path=os.path.join(here, f"_tmp_{name}_L{int(T)}.asc"))
                cell, absdvm = vm_on_reduced(red)
                seg3 = _segments_3d(cell); seg2 = _segments_2d(cell)
                vmax = float(np.percentile(absdvm, 98)) if absdvm.size else 1.0
                ax3 = fig.add_subplot(2, n, j + 1, projection="3d")
                lc3 = Line3DCollection(seg3, cmap="inferno", lw=0.8)
                lc3.set_array(absdvm); lc3.set_clim(0, vmax); ax3.add_collection(lc3)
                _autoscale3d(ax3, seg3)
                ax3.set_title(f"layer {int(T)}um -- {cell.totnsegs} seg\npeak {absdvm.max():.0f} mV",
                              fontsize=9)
                ax2 = fig.add_subplot(2, n, n + j + 1)
                lc2 = LineCollection(seg2, cmap="inferno", array=absdvm, lw=1.0)
                lc2.set_clim(0, vmax); ax2.add_collection(lc2)
                ax2.set_aspect("equal"); ax2.autoscale()
                ax2.set_xlabel("x (um)"); ax2.set_ylabel("y (um)")
                fig.colorbar(lc2, ax=ax2, fraction=0.046, label="|DeltaVm| (mV)")
                del cell
                os.remove(red)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig); plt.close(fig)
            print(f"  {name}: done ({', '.join(str(int(t)) for t in layers)} um)")
    print("done:", out_pdf)
    return out_pdf


def _autoscale3d(ax, segs):
    p = segs.reshape(-1, 3)
    for setlim, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        lo, hi = p[:, i].min(), p[:, i].max()
        m = (hi - lo) * 0.05 + 1e-6
        setlim(lo - m, hi + m)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


# ----------------------------------------------------------------------
# smoke test
# ----------------------------------------------------------------------
def smoke_test(asc=None, layers=(40.0, 80.0, 120.0), thr=DEFAULT_THR):
    if asc is None:
        asc = find_one_morphology()
    print(f"[slicer smoke] {os.path.basename(asc)}")
    sc, S = _read_tree(asc)

    # (1) NESTING: thinner slab kept-set subset of thicker slab
    Ks = [kept_ids(S, sc[2], T, thr) for T in layers]
    for a, b, Ta, Tb in zip(Ks[:-1], Ks[1:], layers[:-1], layers[1:]):
        assert a <= b, f"nesting violated: {Ta} not subset of {Tb}"
    print(f"  NESTING ok: |K| = {[len(k) for k in Ks]} for layers {list(map(int, layers))}")

    # (2) CONNECTIVITY: every kept section's parent kept or root
    for K in Ks:
        for i in K:
            p = S[i]["parent"]
            assert p is None or p in K, "connectivity broken"
    # (3) NO AXON kept
    for K in Ks:
        assert all(S[i]["type"] != SectionType.axon for i in K), "axon leaked"
    print("  CONNECTIVITY + NO-AXON ok")

    # (4) FLATTEN: reduced 2D morphology is coplanar (z range ~ 0 before set_pos)
    red2d = reduced_asc(asc, layers[-1], thr, flatten=True, out_path="_smoke_2d.asc")
    zr = np.ptp(np.asarray(morphio.Morphology(red2d).points, float)[:, 2])
    assert zr < 1e-3, f"flatten failed, z range = {zr}"
    print(f"  FLATTEN ok: z range = {zr:.2e} um")
    os.remove(red2d)

    # (5) REDUCED LOADS + SIMULATES, and zero current -> rest
    red = reduced_asc(asc, layers[1], thr, flatten=False, out_path="_smoke_3d.asc")
    cell, absdvm = vm_on_reduced(red)
    assert np.isfinite(absdvm).all(), "non-finite Vm"
    assert absdvm.max() > 0, "no response under stim"
    del cell
    cell0, absdvm0 = vm_on_reduced(red, i0_uA=0.0)
    assert absdvm0.max() < 1e-6, f"zero-current drive leaked: {absdvm0.max()}"
    del cell0
    os.remove(red)
    print(f"  VM ok: peak |DeltaVm| = {absdvm.max():.1f} mV (stim) / "
          f"{absdvm0.max():.2e} mV (I0=0)")
    print("[slicer smoke] all checks passed")


if __name__ == "__main__":
    smoke_test()
    report()


def report_shapes(morphs=None, layers=DEFAULT_LAYERS, thr=DEFAULT_THR, out_pdf=None):
    """LFPy-free slicing report: 3D reduced (top) + 2D projection (bottom) per
    layer, coloured by neurite type. Pure morphio -- no NEURON, no LFPy."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    TYPE_COLOR = {SectionType.soma: "black", SectionType.basal_dendrite: "green",
                  SectionType.apical_dendrite: "darkorange"}
    here = os.path.dirname(os.path.abspath(__file__))
    if morphs is None:
        morphs = all_morphologies()
    if out_pdf is None:
        out_pdf = os.path.join(here, "slice_report.pdf")
    with PdfPages(out_pdf) as pdf:
        for name, asc in morphs:
            n = len(layers)
            fig = plt.figure(figsize=(4.4 * n, 8))
            fig.suptitle(f"{name} - slicing 3D (top) / 2D (bottom) by layer", fontweight="bold")
            for j, T in enumerate(layers):
                sc, S = _read_tree(asc)
                K = kept_ids(S, sc[2], T, thr)
                ax3 = fig.add_subplot(2, n, j + 1, projection="3d")
                segs3, cols = [], []
                for i in K:
                    P = S[i]["pts"] - sc
                    for k in range(len(P) - 1):
                        segs3.append([P[k], P[k + 1]]); cols.append(TYPE_COLOR.get(S[i]["type"], "0.5"))
                if segs3:
                    ax3.add_collection3d(Line3DCollection(segs3, colors=cols, lw=0.6))
                    p = np.array(segs3).reshape(-1, 3)
                    for setlim, c in ((ax3.set_xlim, 0), (ax3.set_ylim, 1), (ax3.set_zlim, 2)):
                        setlim(p[:, c].min(), p[:, c].max())
                ax3.set_title(f"layer {int(T)}um - {len(K)} sec", fontsize=9)
                ax2 = fig.add_subplot(2, n, n + j + 1)
                for pts2, typ in sliced_polylines(asc, T, thr):
                    ax2.plot(pts2[:, 0], pts2[:, 1], color=TYPE_COLOR.get(typ, "0.5"), lw=0.6)
                ax2.set_aspect("equal"); ax2.set_xlabel("x (um)"); ax2.set_ylabel("y (um)")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig); plt.close(fig)
            print(f"  {name}: done")
    print("done:", out_pdf)
    return out_pdf
