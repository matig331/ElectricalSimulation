"""
well.py -- the multi-morphology well with a VARIABLE number of neurons.

Places CFG.n_neurons somata at random positions/orientations inside the correct
area, each a random pick of the 3 configured morphologies, sliced to a chosen
layer (2D), and draws them coloured by morphology type on the real electrodes.
No simulation here (fast) -- this is the spatial layout you tune before running
the (few) active sims. CFG.n_neurons_hpc() prints the biological count.

Run:  python well.py            # -> well_layout.pdf  (uses CFG)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from config import CFG
from morphologies import find_one_morphology
from slicer import sliced_polylines
import field as F

MORPH_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def populate(cfg, rng):
    N = cfg.n_neurons
    idx = rng.integers(0, len(cfg.morphologies), size=N)
    pos = rng.uniform(-cfg.area_half_um, cfg.area_half_um, size=(N, 2))
    theta = rng.uniform(0, 360, size=N)
    return idx, pos, theta


def _rot_trans(pts2, theta_deg, pos):
    th = np.radians(theta_deg); c, s = np.cos(th), np.sin(th)
    x = c * pts2[:, 0] - s * pts2[:, 1] + pos[0]
    y = s * pts2[:, 0] + c * pts2[:, 1] + pos[1]
    return np.column_stack([x, y])


def draw(cfg, layer=None):
    layer = layer if layer is not None else cfg.layers_um[-1]
    rng = np.random.default_rng(cfg.seed)
    idx, pos, theta = populate(cfg, rng)
    # cache sliced polylines per morphology
    cache = {}
    for m in set(idx):
        asc = find_one_morphology(cfg.morphologies[m])
        cache[m] = sliced_polylines(asc, layer, cfg.slice_thresh)

    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=not cfg.bipolar)
    fig, ax = plt.subplots(figsize=(8, 8))
    for m in range(len(cfg.morphologies)):
        segs = []
        for i in np.where(idx == m)[0]:
            for pts2, _typ in cache[m]:
                P = _rot_trans(pts2, theta[i], pos[i])
                for k in range(len(P) - 1):
                    segs.append([P[k], P[k + 1]])
        if segs:
            ax.add_collection(LineCollection(segs, colors=MORPH_COLORS[m], lw=0.5, alpha=0.8))
        ax.plot([], [], color=MORPH_COLORS[m], label=cfg.morphologies[m])

    ah = cfg.area_half_um
    for (ex, ey) in elec[sign > 0]:
        ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="red", edgecolor="k", zorder=6))
    for (ex, ey) in (elec[sign < 0] if cfg.bipolar else []):
        ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="blue", edgecolor="k", zorder=6))
    ax.add_patch(Rectangle((-ah, -ah), 2*ah, 2*ah, fill=False, ec="k", ls="--", lw=1.2))
    ax.scatter(pos[:, 0], pos[:, 1], s=12, c="k", zorder=5)          # somata
    ax.set_aspect("equal"); ax.set_xlim(-ah-350, ah+350); ax.set_ylim(-ah-350, ah+350)
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title(f"Well -- {cfg.n_neurons} neurons (layer {int(layer)}um), 3 morphologies\n"
                 f"area {2*ah:.0f}x{2*ah:.0f} um . HPC count at this area = "
                 f"{cfg.n_neurons_hpc()}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", title="morphology")
    fig.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "well_layout.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"drew {cfg.n_neurons} neurons | HPC count for area = {cfg.n_neurons_hpc()}")
    print("done:", out)
    return out


if __name__ == "__main__":
    draw(CFG)


def check_area(cfg, layers=None):
    """ACTIVE area check over ALL layers (one page each). Flags any ACTIVE soma
    within edge_margin of the border (active-at-edge => area too small)."""
    from rich_cell import build_rich_cell
    from rich_footprint import spikes_at
    from slicer import reduced_asc
    from matplotlib.patches import Rectangle
    from matplotlib.backends.backend_pdf import PdfPages
    import os as _os
    layers = layers if layers is not None else cfg.layers_um
    ah = cfg.area_half_um; edge_margin = 20.0
    out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "well_area_check.pdf")
    ok_all = True
    with PdfPages(out) as pdf:
        for layer in layers:
            rng = np.random.default_rng(cfg.seed)
            idx, pos, theta = populate(cfg, rng)
            active = np.zeros(len(idx), bool)
            for m in sorted(set(idx)):
                asc = reduced_asc(find_one_morphology(cfg.morphologies[m]), layer, out_path=f"_ca_{m}.asc")
                cell = build_rich_cell(asc)
                for i in np.where(idx == m)[0]:
                    active[i] = spikes_at(cell, tuple(pos[i]), float(theta[i]), i0_uA=cfg.i0_uA)[0] > 0
                del cell; _os.remove(asc)
            at_edge = np.abs(pos).max(axis=1) > (ah - edge_margin)
            active_edge = active & at_edge; ok_all = ok_all and not active_edge.any()
            elec, sign = F.default_array(monopolar=not cfg.bipolar)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(pos[~active, 0], pos[~active, 1], s=40, facecolors="none", edgecolors="0.6", label="inactive soma")
            ax.scatter(pos[active, 0], pos[active, 1], s=60, c="crimson", label="ACTIVE soma", zorder=6)
            if active_edge.any():
                ax.scatter(pos[active_edge, 0], pos[active_edge, 1], s=180, facecolors="none",
                           edgecolors="lime", linewidths=2.5, label="ACTIVE at edge (BAD)", zorder=7)
            for (ex, ey) in elec[sign > 0]:
                ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="red", edgecolor="k", zorder=5))
            for (ex, ey) in (elec[sign < 0] if cfg.bipolar else []):
                ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="blue", edgecolor="k", zorder=5))
            ax.add_patch(Rectangle((-ah, -ah), 2*ah, 2*ah, fill=False, ec="k", ls="--", lw=1.2))
            ax.set_aspect("equal"); ax.set_xlim(-ah-40, ah+40); ax.set_ylim(-ah-40, ah+40)
            verdict = "AREA OK" if not active_edge.any() else "AREA TOO SMALL (active soma at edge)"
            ax.set_title(f"Area check (layer {int(layer)}um) - {int(active.sum())}/{len(idx)} active - {verdict}", fontsize=10)
            ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
            print(f"  layer {int(layer)}: active {int(active.sum())}/{len(idx)}, active-at-edge {int(active_edge.sum())} -> {verdict}")
    print("done:", out)
    return ok_all


def _classify(name):
    if "axon" in name: return "axon"
    if "soma" in name: return "soma"
    if "apic" in name: return "apical"
    return "basal"


def _classify_init(refs, seg_i, coords, soma_i, axon_term_um=200.0):
    """Classify the initiation segment; distinguish AIS (proximal axon) from the
    distal AXON TERMINAL (sealed-end artifact) by distance-from-soma."""
    name = refs[seg_i].sec.name()
    if "axon" in name:
        d = float(np.linalg.norm(coords[seg_i, :2] - coords[soma_i, :2]))
        return "axon terminal (sealed end)" if d > axon_term_um else "AIS"
    return _classify(name)


def _neuron_points(coords, refs, elec, si):
    d = np.min(np.linalg.norm(coords[:, None, :2]-elec[None, :, :], axis=2), axis=1); d[si] = np.inf
    near = int(np.argmin(d))
    dend = np.array(["axon" not in s.sec.name() for s in refs])
    dd = np.where(dend, np.linalg.norm(coords[:, :2]-coords[si, :2], axis=1), -np.inf)
    return {"soma": si, "near": near, "dend": int(np.argmax(dd))}


def _spike_init(v_all, refs, coords=None, soma_i=None, thr=0.0):
    """Return (compartment label, seg index) of the FIRST segment to cross thr.
    If coords+soma_i given, distinguishes AIS from axon-terminal (sealed end)."""
    n_t = v_all.shape[1]
    first = np.full(v_all.shape[0], n_t, dtype=int)
    for k in range(v_all.shape[0]):
        cr = np.where((v_all[k, :-1] < thr) & (v_all[k, 1:] >= thr))[0]
        if cr.size:
            first[k] = cr[0]
    if first.min() >= n_t:
        return None
    seg = int(np.argmin(first))
    if coords is not None and soma_i is not None:
        return _classify_init(refs, seg, coords, soma_i), seg
    return _classify(refs[seg].sec.name()), seg


PT_STYLE = dict(soma="black", near="magenta", dend="green")


PT_STYLE = dict(soma="black", near="magenta", dend="green")


def well_active(cfg, layers=None, zoom_margin=80.0):
    """3 morphologies in ONE well, ACTIVE, all layers.
    Per layer -> TWO pages:
      page 1: morphologies by type + Vm maps BEFORE/POS/NEG/AFTER (spike-init ringed)
      page 2: ONE PANEL PER NEURON with its Vm(t) at soma/near/dend, labelled with
              neuron id, morphology, position, theta, #spikes and initiation site.
    All cfg.n_neurons neurons are plotted (set n_neurons >= 12). Random pos+theta.
    """
    from rich_stim import stimulate_rich, _placed_polylines
    from rich_cell import build_rich_cell
    from slicer import reduced_asc
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    from matplotlib.backends.backend_pdf import PdfPages
    import os as _os
    from collections import Counter
    layers = layers if layers is not None else cfg.layers_um
    rng = np.random.default_rng(cfg.seed)
    idx, pos, theta = populate(cfg, rng)
    ah = cfg.area_half_um; zl = ah + zoom_margin
    elec, sign = F.default_array(monopolar=not cfg.bipolar)
    anod, cath = elec[sign > 0], elec[sign < 0]
    order = ["before", "pos", "neg", "after"]
    titles = {"before": "BEFORE (rest)", "pos": "POS peak", "neg": "NEG peak", "after": "AFTER"}
    here = _os.path.dirname(_os.path.abspath(__file__))
    out = _os.path.join(here, "well_active.pdf")

    with PdfPages(out) as pdf:
        for layer in layers:
            morph_lines = {m: [] for m in range(len(cfg.morphologies))}
            vm = {k: {"lines": [], "vals": []} for k in order}
            per_neuron = []            # one dict per neuron (all of them)
            init_marks = []; init_counter = Counter()
            tt = None; idxmap = None; nspk = 0
            for m in sorted(set(idx)):
                asc = reduced_asc(find_one_morphology(cfg.morphologies[m]), layer, out_path=f"_wa_{m}.asc")
                cell = build_rich_cell(asc)
                for i in np.where(idx == m)[0]:
                    res = stimulate_rich(cell, pos_xy=tuple(pos[i]), theta_deg=float(theta[i]),
                                         i0_uA=cfg.i0_uA, phase_dur_ms=cfg.phase_dur_ms,
                                         baseline_ms=cfg.baseline_ms, post_ms=cfg.post_ms,
                                         dt_ms=cfg.dt_ms, sigma_Sm=cfg.sigma_Sm, rmin_um=cfg.rmin_um,
                                         ramp_us=cfg.ramp_us, interphase_us=cfg.interphase_us)
                    v_all, refs, coords = res["v_all"], res["refs"], res["coords"]
                    dvm = v_all - v_all[:, :1]
                    si = [j for j, s in enumerate(refs) if "soma" in s.sec.name()][0]
                    vs = v_all[si]; sp = int(np.sum((vs[:-1] < 0) & (vs[1:] >= 0))); nspk += sp
                    if tt is None:
                        tt = res["t"]; idxmap = res["idx"]
                    init_type = "-"
                    if sp > 0:
                        ini = _spike_init(v_all, refs, coords=coords, soma_i=si)
                        if ini:
                            init_type = ini[0]; init_counter[ini[0]] += 1; init_marks.append(coords[ini[1], :2])
                    pts = _neuron_points(coords, refs, elec, si)
                    per_neuron.append(dict(id=int(i), morph=cfg.morphologies[m], pos=tuple(pos[i]),
                                           theta=float(theta[i]), sp=sp, init=init_type,
                                           soma=v_all[pts["soma"]], near=v_all[pts["near"]], dend=v_all[pts["dend"]]))
                    l0, _ = _placed_polylines(cell, np.zeros(dvm.shape[0]), refs, res["sc"],
                                              tuple(pos[i]), float(theta[i]))
                    morph_lines[m].append(l0)
                    for k in order:
                        lv, vv = _placed_polylines(cell, dvm[:, res["idx"][k]], refs, res["sc"],
                                                   tuple(pos[i]), float(theta[i]))
                        vm[k]["lines"].append(lv); vm[k]["vals"].append(vv)
                del cell; _os.remove(asc)

            allv = np.concatenate([np.concatenate(vm[k]["vals"]) for k in order])
            vlim = float(np.percentile(np.abs(allv), 99)) or 1.0
            init_str = ", ".join(f"{v} {k}" for k, v in init_counter.items()) or "-"

            # ---- PAGE 1: maps ----
            fig = plt.figure(figsize=(22, 5.5))
            gs = fig.add_gridspec(1, 5)
            fig.suptitle(f"Active well - layer {int(layer)}um - {cfg.n_neurons} neurons, "
                         f"area {2*ah:.0f}um - {nspk} spikes - initiation: {init_str}", fontweight="bold")
            ax0 = fig.add_subplot(gs[0, 0])
            for m in range(len(cfg.morphologies)):
                segs = [ln for arr in morph_lines[m] for ln in arr]
                if segs:
                    ax0.add_collection(LineCollection(segs, colors=MORPH_COLORS[m], lw=0.5, alpha=0.8))
                ax0.plot([], [], color=MORPH_COLORS[m], label=cfg.morphologies[m])
            ax0.legend(fontsize=7, title="morphology"); ax0.set_title("morphologies", fontsize=9)
            map_axes = [ax0]; lc = None
            for col, k in enumerate(order):
                ax = fig.add_subplot(gs[0, col+1])
                for lv, vv in zip(vm[k]["lines"], vm[k]["vals"]):
                    lc = LineCollection(lv, cmap="RdBu_r", array=vv, lw=0.6); lc.set_clim(-vlim, vlim)
                    ax.add_collection(lc)
                if k == "after" and init_marks:
                    im = np.array(init_marks)
                    ax.scatter(im[:, 0], im[:, 1], s=60, facecolors="none", edgecolors="lime",
                               linewidths=1.5, zorder=7, label="spike start")
                    ax.legend(fontsize=6, loc="upper right")
                ax.set_title(f"Vm - {titles[k]}", fontsize=9); map_axes.append(ax)
            for ax in map_axes:
                for (ex, ey) in anod:
                    ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="red", edgecolor="k", zorder=6))
                for (ex, ey) in cath:
                    ax.add_patch(Rectangle((ex-12.5, ey-12.5), 25, 25, facecolor="blue", edgecolor="k", zorder=6))
                ax.add_patch(Rectangle((-ah, -ah), 2*ah, 2*ah, fill=False, ec="k", ls="--", lw=1.0))
                ax.set_aspect("equal"); ax.set_xlim(-zl, zl); ax.set_ylim(-zl, zl)
                ax.set_xticks([]); ax.set_yticks([])
            if lc is not None:
                fig.colorbar(lc, ax=map_axes, fraction=0.012, label="dVm (mV)")
            pdf.savefig(fig); plt.close(fig)

            # ---- PAGE 2: one panel per neuron ----
            N = len(per_neuron); ncol = 4; nrow = int(np.ceil(N/ncol))
            fig2, axes2 = plt.subplots(nrow, ncol, figsize=(4.6*ncol, 2.6*nrow), squeeze=False)
            fig2.suptitle(f"Active well - layer {int(layer)}um - Vm(t) per neuron "
                          f"(black=soma, magenta=near-electrode, green=dendrite)", fontweight="bold")
            for j, nd in enumerate(per_neuron):
                ax = axes2[j//ncol][j % ncol]
                for pk in ("soma", "near", "dend"):
                    ax.plot(tt, nd[pk], color=PT_STYLE[pk], lw=0.8)
                ax.axvspan(tt[idxmap["pos"]], tt[idxmap["neg"]], color="0.85", zorder=0)
                ax.set_title(f"N{nd['id']} {nd['morph']} pos({nd['pos'][0]:.0f},{nd['pos'][1]:.0f}) "
                             f"th{nd['theta']:.0f} sp={nd['sp']} init={nd['init']}", fontsize=6.5)
                ax.tick_params(labelsize=6)
            for j in range(N, nrow*ncol):
                axes2[j//ncol][j % ncol].axis("off")
            fig2.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig2); plt.close(fig2)
            print(f"  layer {int(layer)}: {nspk} spikes, initiation {init_str}")
    print("done:", out)
    return out
