"""
baseline.py -- step B: homeostatic calibration / static-current balance.

Runs the Rich cell with NO stimulus for 1500 ms from v_init=-85 mV and shows:
  * fast phase (~100 ms): Vm depolarises -85 -> -67 mV (fast channels charge);
  * slow phase (~1 s): Vm relaxes to the true rest ~-74 mV, driven by the slow
    Ih kinetics; the net soma membrane current decays toward 0.

Conclusion: the static currents balance at ~-74 mV (NOT -85). So the correct
v_init for stimulation is ~-74 mV, and a clean baseline needs ~1 s of settling.

Run:  cd estim && python baseline.py     # -> baseline_report.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neuron import h

from morphologies import find_one_morphology
from slicer import reduced_asc
from rich_cell import build_rich_cell


def main(pref="60308", layer=120.0, tstop_ms=1500.0, dt_ms=0.025):
    here = os.path.dirname(os.path.abspath(__file__))
    asc = reduced_asc(find_one_morphology(pref), layer, out_path="_base.asc")
    cell = build_rich_cell(asc)
    h.celsius = 37; h.dt = dt_ms
    try:
        h.cvode.use_fast_imem(1)
    except Exception:
        pass
    v = h.Vector().record(cell.soma[0](0.5)._ref_v)
    im = h.Vector().record(cell.soma[0](0.5)._ref_i_membrane_)
    t = h.Vector().record(h._ref_t)
    h.finitialize(-85.0)
    h.continuerun(tstop_ms)
    t, v, im = np.asarray(t), np.asarray(v), np.asarray(im)
    v_rest = float(v[-1])

    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(f"Step B - homeostatic baseline (specimen_{pref}), no stimulus",
                 fontweight="bold")
    ax[0].plot(t, v, color="navy")
    ax[0].axhline(v_rest, color="0.6", ls=":", lw=0.9)
    ax[0].annotate(f"settled rest ~ {v_rest:.1f} mV", xy=(t[-1], v_rest),
                   xytext=(0.55, 0.5), textcoords="axes fraction", fontsize=9)
    ax[0].set_ylabel("soma Vm (mV)"); ax[0].set_title("membrane potential", fontsize=10)

    ax[1].plot(t, im, color="crimson")
    ax[1].axhline(0, color="0.6", ls=":", lw=0.9)
    ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("net soma current (nA)")
    ax[1].set_title("net membrane current -> 0 as the cell settles", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(here, "baseline_report.pdf")
    fig.savefig(out); plt.close(fig)
    os.remove(asc)
    print(f"settled rest = {v_rest:.2f} mV | net current at end = {im[-1]:+.2e} nA")
    print("done:", out)
    return out


if __name__ == "__main__":
    main()
