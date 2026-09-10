"""
checks.py -- numerical sanity checks with explicit tolerances. Prints PASS/FAIL.

Cheap (no NEURON):  charge balance, field 1/r + superposition.
Active (Rich):      zero-current -> rest, dt convergence.

Run:  python checks.py         # runs all (needs rich_mech compiled for the active ones)
"""
import numpy as np
import field as F
from config import CFG

_trapz = getattr(np, "trapezoid", getattr(np, "trapz"))   # numpy >=2 vs <2


def _ok(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return cond


def check_charge_balance(cfg=CFG, tol=1e-9):
    t = np.arange(0, 2 * cfg.phase_dur_ms + 0.05, cfg.dt_ms)
    I = F.biphasic_current(t, cfg.i0_uA, cfg.phase_dur_ms, cfg.anodic_first,
                           cfg.ramp_us, cfg.interphase_us)
    q = float(_trapz(I, t))
    return _ok("charge balance", abs(q) < tol, f"integralI dt = {q:.2e} (tol {tol})")


def check_field(cfg=CFG, tol=1e-3):
    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=True)  # single-sign for 1/r
    e0 = elec[:1]; s0 = np.array([1.0])
    ex, ey = elec[0]
    p1 = np.array([[200.0, 0.0, cfg.h_soma_um]]); p2 = np.array([[400.0, 0.0, cfg.h_soma_um]])
    r1 = np.linalg.norm(p1[0] - [ex, ey, 0.0]); r2 = np.linalg.norm(p2[0] - [ex, ey, 0.0])
    g1 = F.geom_factor(p1, e0, s0, sigma_Sm=cfg.sigma_Sm, rmin_um=cfg.rmin_um)[0]
    g2 = F.geom_factor(p2, e0, s0, sigma_Sm=cfg.sigma_Sm, rmin_um=cfg.rmin_um)[0]
    # Ve = I/(2 pi sigma r) -> g*r must be constant regardless of geometry
    r_ok = _ok("1/r decay", abs(g1 * r1 - g2 * r2) < abs(g1 * r1) * tol,
               f"g*r: {g1*r1:.4f} vs {g2*r2:.4f} (must match)")
    # superposition: 2 electrodes summed == sum of singles
    two = F.geom_factor(p1, elec[:2], np.array([1.0, 1.0]), sigma_Sm=cfg.sigma_Sm, rmin_um=cfg.rmin_um)[0]
    singles = sum(F.geom_factor(p1, elec[k:k+1], np.array([1.0]), sigma_Sm=cfg.sigma_Sm,
                                rmin_um=cfg.rmin_um)[0] for k in range(2))
    s_ok = _ok("superposition", abs(two - singles) < abs(singles) * 1e-6,
               f"sum={two:.4f} vs {singles:.4f}")
    return r_ok and s_ok


def _build(cfg):
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    from rich_cell import build_rich_cell
    import os
    asc = reduced_asc(find_one_morphology(cfg.morphologies[0]), cfg.layers_um[1], out_path="_chk.asc")
    cell = build_rich_cell(asc); os.remove(asc)
    return cell


def check_zero_current(cfg=CFG, cell=None, tol_mV=0.5):
    from rich_footprint import spikes_at
    if cell is None:
        cell = _build(cfg)
    nsp, vmax = spikes_at(cell, (0.0, 0.0), 0.0, i0_uA=0.0)
    return _ok("zero-current -> rest", nsp == 0 and abs(vmax - cfg.v_rest_mV) < tol_mV,
               f"spikes={nsp}, Vmax={vmax:.2f} (rest {cfg.v_rest_mV})"), cell


def check_dt_convergence(cfg=CFG, cell=None, tol_mV=5.0):
    from rich_footprint import spikes_at
    if cell is None:
        cell = _build(cfg)
    _, v_coarse = spikes_at(cell, (30.0, 30.0), 0.0, i0_uA=cfg.i0_uA, dt_ms=cfg.dt_ms)
    _, v_fine = spikes_at(cell, (30.0, 30.0), 0.0, i0_uA=cfg.i0_uA, dt_ms=cfg.dt_ms / 2)
    return _ok("dt convergence", abs(v_coarse - v_fine) < tol_mV,
               f"Vpeak dt={v_coarse:.2f}  dt/2={v_fine:.2f}  |Delta|={abs(v_coarse-v_fine):.2f} (tol {tol_mV})"), cell


def run_all(cfg=CFG, active=True):
    print("CHEAP checks:")
    a = check_charge_balance(cfg); b = check_field(cfg)
    if not active:
        return
    print("ACTIVE checks (Rich):")
    cell = _build(cfg)
    c, cell = check_zero_current(cfg, cell)
    d, cell = check_dt_convergence(cfg, cell)
    print(f"summary: {sum([a,b,c,d])}/4 passed")


if __name__ == "__main__":
    run_all()


# --------------------------------------------------------------------- literature
def _threshold_uA(cell, pos_xy, thetas=(0, 90, 180, 270), i_lo=2.0, i_hi=400.0,
                  tol=3.0, **stim):
    """Smallest i0 (uA) that fires a somatic spike at the best orientation (bisection)."""
    from rich_footprint import spikes_at
    def fires(i0):
        return any(spikes_at(cell, pos_xy, float(th), i0_uA=i0, **stim)[0] > 0 for th in thetas)
    if not fires(i_hi):
        return None
    while i_hi - i_lo > tol:
        mid = 0.5 * (i_lo + i_hi)
        if fires(mid): i_hi = mid
        else: i_lo = mid
    return i_hi


def check_stoney(cfg=CFG, cell=None, distances=(20.0, 40.0, 60.0, 80.0, 100.0)):
    """Stoney 1968: threshold current vs distance -> I_th ~ k * r^n, expect n ~ 2.
    HEAVY: bisection at each distance. (Soma is placed at distance r from the
    electrode-cluster centre; with the 4-electrode array this is the cluster
    distance, not a single point source -- expect n in ~1.5-2.5.)"""
    import numpy as np
    if cell is None:
        cell = _build(cfg)
    rs, ith = [], []
    for r in distances:
        it = _threshold_uA(cell, (r, 0.0))
        if it is not None:
            rs.append(r); ith.append(it)
        print(f"    d={r:.0f}um -> I_th={it}")
    if len(rs) < 3:
        return _ok("Stoney I~r^n", False, "too few thresholds found"), cell
    n = float(np.polyfit(np.log(rs), np.log(ith), 1)[0])
    return _ok("Stoney I~r^n", 1.3 < n < 2.7, f"exponent n = {n:.2f} (expect ~2)"), cell


def check_wagenaar(cfg=CFG, cell=None, phase_durs_ms=(0.1, 0.2, 0.4, 0.8)):
    """Wagenaar 2004: strength-duration -> chronaxie (dur at 2x rheobase) in a
    physiological range (~0.1-0.7 ms). HEAVY: threshold at each phase duration."""
    import numpy as np
    if cell is None:
        cell = _build(cfg)
    pos = (30.0, 0.0)
    th = []
    for pd in phase_durs_ms:
        it = _threshold_uA(cell, pos, phase_dur_ms=pd,
                           baseline_ms=5.0, post_ms=6.0)
        th.append(it); print(f"    phase={pd:.2f}ms -> I_th={it}")
    th = [x for x in th if x is not None]
    if len(th) < 3:
        return _ok("Wagenaar strength-duration", False, "too few thresholds"), cell
    rheobase = min(th)
    # chronaxie ~ duration where threshold = 2*rheobase (interpolate)
    dd = np.array(phase_durs_ms[:len(th)]); tt = np.array(th)
    chron = float(np.interp(2 * rheobase, tt[::-1], dd[::-1])) if tt.min() < 2*rheobase < tt.max() else np.nan
    return _ok("Wagenaar chronaxie", 0.05 < chron < 0.8,
               f"rheobase~{rheobase:.0f}uA, chronaxie~{chron:.2f}ms"), cell
