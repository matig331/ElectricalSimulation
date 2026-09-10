"""
coupling_checks.py -- GROUP 1: quantitative validation of the field->Vm coupling and
of the current-distance law. These underpin the activation-probability-vs-distance,
so they must be right.

(1) check_polarization       -- somatic polarization per unit uniform field (mV per V/m)
                                vs Radman 2009 / Bikson 2004 (~0.1-0.3 mV per V/m soma)
(2) check_current_distance_k -- absolute Stoney constant k in I_th = k * r^n
                                vs Stoney 1968 / Tehovnik 2006 (k ~ 1000-2000 uA/mm2)
(3) check_activating_function-- Rattay's activating function (d2Ve/ds2) predicts where
                                the membrane depolarises (AF vs dVm correlation)

All loop ALL morphologies x layers. Heavy (thresholds need bisection).

Run:  python pipeline.py check_polarization   (etc.)
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
from active_cell import segment_coords, _soma_center
from rich_footprint import spikes_at
import field as F


# ---------- (1) polarization per unit uniform field ----------
_ACTIVE = ["NaTa_t", "Nap_Et2", "K_Pst", "K_Tst", "SKv3_1", "SK_E2", "Im",
           "Ca_LVAst", "Ca_HVA", "Ih", "na", "kv"]


def _make_passive(cell):
    """Zero all active conductances (keep pas/leak) -> pure electrotonic coupling."""
    for sec in cell.all:
        for mech in _ACTIVE:
            if not h.ismembrane(mech, sec=sec):
                continue
            var = f"gbar_{mech}" if mech in ("na", "kv") else f"g{mech}bar_{mech}"
            for seg in sec:
                try:
                    setattr(seg, var, 0.0)
                except Exception:
                    pass
def _uniform_dVm(cell, coords, refs, axis, E_Vperm, baseline_ms=50.0, field_ms=120.0,
                 dt=0.025, v_init=-73.9):
    proj = coords @ axis                      # um along the field axis
    for sec in set(s.sec for s in refs):
        if not h.ismembrane("extracellular", sec=sec):
            sec.insert("extracellular")
    t = np.arange(0.0, baseline_ms + field_ms + dt, dt)
    tvec = h.Vector(t); keep = [tvec]
    for k, seg in enumerate(refs):
        ve_mV = -E_Vperm * proj[k] / 1000.0   # (V/m)*(um)/1000 = mV
        wave = np.where(t >= baseline_ms, ve_mV, 0.0)
        vv = h.Vector(wave); vv.play(seg._ref_e_extracellular, tvec, True); keep.append(vv)
    vsoma = h.Vector().record(cell.soma[0](0.5)._ref_v)
    vall = h.Vector()  # not used; keep simple
    h.celsius = 37; h.dt = dt; h.finitialize(v_init); h.continuerun(t[-1])
    vsoma = np.asarray(vsoma)
    ipre = int(baseline_ms / dt) - 2
    return vsoma[-1] - vsoma[ipre]


def check_polarization(prefs=None, layers=None, fields_Vperm=(-40, -20, -10, 10, 20, 40),
                       passive=True):
    """Somatic polarization per unit uniform field. passive=True zeroes active channels
    -> pure electrotonic coupling (linear, comparable to Radman/Bikson & cable theory).
    Also reports the dendritic-tip coupling (dendrites polarize far more than the soma)."""
    cfg = CFG; prefs = prefs or cfg.morphologies; layers = layers or cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(here, "check_polarization.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                asc = reduced_asc(find_one_morphology(pref), layer, out_path="_pol.asc")
                cell = build_rich_cell(asc)
                if passive:
                    _make_passive(cell)
                coords, refs = segment_coords(cell)
                X = coords - coords.mean(0)
                axis = np.linalg.svd(X, full_matrices=False)[2][0]     # principal axis
                dv = np.array([_uniform_dVm(cell, coords, refs, axis, E, v_init=cfg.v_rest_mV)
                               for E in fields_Vperm])
                os.remove(asc)
                E = np.array(fields_Vperm, float)
                p = np.polyfit(E, dv, 1); slope = p[0]                 # mV per V/m
                resid = dv - np.polyval(p, E)
                r2 = 1 - np.sum(resid**2) / max(np.sum((dv - dv.mean())**2), 1e-12)
                fig, ax = plt.subplots(figsize=(6.5, 4.4))
                ax.plot(E, dv, "o", label="data")
                ax.plot(E, np.polyval(p, E), "r-", label=f"fit {slope:.3f} mV/(V/m), R2={r2:.3f}")
                ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
                ax.set_xlabel("uniform field E (V/m, along principal axis)")
                ax.set_ylabel("somatic dVm (mV)"); ax.legend(fontsize=7)
                inband = 0.1 <= abs(slope) <= 0.3
                mode = "passive" if passive else "ACTIVE"
                ax.set_title(f"{pref} L{int(layer)} [{mode}] - coupling |{slope:.3f}| mV/(V/m)  "
                             f"[Radman/Bikson 0.1-0.3 -> {'OK' if inband else 'CHECK'}]", fontsize=8)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)} [{mode}]: coupling={slope:.3f} mV/(V/m), "
                      f"linearity R2={r2:.3f} ({'OK' if inband else 'out of band'})")
    print("done:", out); return out


# ---------- (2) absolute current-distance constant k ----------
def _threshold_uA(cell, r, thetas=(0, 90, 180, 270), lo=1.0, hi=400.0, steps=8):
    def fires(a):
        return any(spikes_at(cell, (r, 0.0), float(th), i0_uA=a)[0] > 0 for th in thetas)
    if not fires(hi):
        return np.nan
    if fires(lo):
        return lo
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if fires(mid): hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


def check_current_distance_k(prefs=None, layers=None, dists=(20, 30, 40, 50, 60, 70, 80)):
    cfg = CFG; prefs = prefs or cfg.morphologies; layers = layers or cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(here, "check_current_distance_k.pdf")
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                asc = reduced_asc(find_one_morphology(pref), layer, out_path="_cdk.asc")
                cell = build_rich_cell(asc)
                Ith = np.array([_threshold_uA(cell, d) for d in dists])
                os.remove(asc)
                r = np.array(dists, float); ok = np.isfinite(Ith)
                fig, ax = plt.subplots(figsize=(6.5, 4.4))
                k = n = np.nan
                if ok.sum() >= 3:
                    b = np.polyfit(np.log(r[ok]), np.log(Ith[ok]), 1)   # log I = n log r + log k
                    n = b[0]; k_umum = np.exp(b[1])                     # uA / um^n
                    k = k_umum * (1000.0 ** n)                          # convert to uA/mm^2 if n~2
                    rr = np.linspace(r[ok].min(), r[ok].max(), 50)
                    ax.plot(rr, np.exp(b[1]) * rr ** n, "r-",
                            label=f"fit n={n:.2f}, k~{k:.0f} uA/mm^2")
                ax.plot(r[ok], Ith[ok], "o"); ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel("distance r (um)"); ax.set_ylabel("threshold I (uA)")
                ax.set_title(f"{pref} L{int(layer)} - current-distance (Stoney n~2; "
                             f"Tehovnik k~1000-2000 uA/mm2)", fontsize=8)
                ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: n={n:.2f}, k~{k:.0f} uA/mm2")
    print("done:", out); return out


# ---------- (3) activating function vs dVm (Rattay) ----------
def _dvm_subthreshold(cell, coords, refs, i0_uA, cfg):
    from rich_stim import stimulate_rich
    res = stimulate_rich(cell, pos_xy=(40.0, 0.0), theta_deg=0.0, i0_uA=i0_uA,
                         baseline_ms=20.0, post_ms=5.0)
    dvm = res["v_all"] - res["v_all"][:, :1]
    return dvm[:, res["idx"]["pos"]], res


def check_activating_function(prefs=None, layers=None, i0_sub=8.0):
    """AF_seg ~ d2Ve/ds2 along each section; correlate with the subthreshold dVm.
    High correlation => the activating function predicts where the membrane depolarises."""
    cfg = CFG; prefs = prefs or cfg.morphologies; layers = layers or cfg.layers_um
    here = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(here, "check_activating_function.pdf")
    elec, sign = F.default_array(monopolar=not cfg.bipolar)
    with PdfPages(out) as pdf:
        for pref in prefs:
            for layer in layers:
                asc = reduced_asc(find_one_morphology(pref), layer, out_path="_af.asc")
                cell = build_rich_cell(asc); coords, refs = segment_coords(cell)
                sc = _soma_center(cell, coords, refs)
                placed = coords + (np.array([40.0, 0.0, 10.0]) - sc)
                Ve = F.geom_factor(placed, elec, sign, sigma_Sm=cfg.sigma_Sm, rmin_um=cfg.rmin_um) * i0_sub
                # AF per section: second difference of Ve along arc length
                AF = np.zeros(len(refs))
                bysec = {}
                for i, s in enumerate(refs):
                    bysec.setdefault(s.sec, []).append(i)
                for sec, idxs in bysec.items():
                    idxs = sorted(idxs, key=lambda j: refs[j].x)
                    if len(idxs) < 3:
                        continue
                    arc = np.array([refs[j].x for j in idxs]) * sec.L    # um
                    ve = Ve[idxs]
                    for m in range(1, len(idxs) - 1):
                        ds1 = arc[m] - arc[m - 1]; ds2 = arc[m + 1] - arc[m]
                        if ds1 > 0 and ds2 > 0:
                            AF[idxs[m]] = 2 * (ve[m - 1] / (ds1 * (ds1 + ds2))
                                               - ve[m] / (ds1 * ds2)
                                               + ve[m + 1] / (ds2 * (ds1 + ds2)))
                dvm_peak, res = _dvm_subthreshold(cell, coords, refs, i0_sub, cfg)
                os.remove(asc)
                m = AF != 0
                cc = np.corrcoef(AF[m], dvm_peak[m])[0, 1] if m.sum() > 5 else np.nan
                fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
                ax[0].scatter(AF[m], dvm_peak[m], s=6, alpha=0.5)
                ax[0].set_xlabel("activating function  d2Ve/ds2  (a.u.)")
                ax[0].set_ylabel("subthreshold dVm (mV)")
                ax[0].set_title(f"AF predicts dVm  (corr = {cc:.2f})", fontsize=9)
                sc2 = ax[1].scatter(placed[:, 0], placed[:, 1], c=AF, cmap="RdBu_r", s=8)
                imax = int(np.argmax(np.abs(AF)))
                ax[1].scatter(placed[imax, 0], placed[imax, 1], s=120, facecolors="none",
                              edgecolors="lime", linewidths=2, label="max |AF|")
                ax[1].set_aspect("equal"); ax[1].set_xticks([]); ax[1].set_yticks([])
                ax[1].legend(fontsize=7); ax[1].set_title("activating function map", fontsize=9)
                fig.colorbar(sc2, ax=ax[1], fraction=0.046)
                fig.suptitle(f"{pref} L{int(layer)} - Rattay activating function (i0={i0_sub}uA, subthreshold)",
                             fontweight="bold")
                fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
                print(f"  {pref} L{int(layer)}: AF-dVm corr={cc:.2f}")
    print("done:", out); return out


if __name__ == "__main__":
    check_polarization(); check_current_distance_k(); check_activating_function()
