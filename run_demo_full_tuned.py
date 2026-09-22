"""run_demo_full_tuned.py -- a SMALL full-channel campaign, end to end, in about a minute.

Runs the real pipeline (culture_export.culture_export -> the same CellPool, simulate_neuron
and build_row the HPC campaign uses), at a size you can watch, into its own folder. Then
plot_bump_statistics.py turns the rows into the kinetics figures.

    python run_demo_full_tuned.py                      # 3 cultures x 6 neurons x 1 layer
    python run_demo_full_tuned.py --cultures 6 --neurons 10 --layers 40,80
    python run_demo_full_tuned.py --cell-model soma_only   # the negative control: no bump

Measured: 3 x 6 x 1 = 18 simulations at an 800 ms window took 53 s on one core, of which most
is the two cell builds (settle + leak tuning). Cost scales with cultures x neurons x layers
plus one build per (morphology, layer).

It does NOT touch results_<model>/ -- everything goes to --out (default demo_<cell_model>/),
so this can be run while a campaign's parts are on disk without any chance of mixing.

Outputs in --out:
    culture_Pactivation.csv / _Pdepolarization / _Phyperpolarization   raw rows + kinetics
    culture_Pmap_*.pdf                                                 the outcome maps
    bump_statistics.pdf / bump_statistics.csv                          the kinetics figures
"""
import argparse
import os
import sys
import time


def main(cell_model="full_tuned", cultures=3, neurons=6, layers=(80.0,), morphologies=None,
         seed=1234, bump_ms=800.0, out=None, i0_uA=None, bin_um=60.0, figures=True):
    from config import CFG
    import culture_export as ce

    CFG.cell_model = str(cell_model)
    CFG.use_hpc_count = False              # honour --neurons instead of the biological count
    CFG.n_neurons = int(neurons)
    CFG.layers_um = tuple(float(x) for x in layers)
    CFG.bump_ms = float(bump_ms)
    CFG.bump_culture_fraction = 1.0        # a demo measures every neuron
    if morphologies:
        CFG.morphologies = list(morphologies)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, out or ("demo_%s" % cell_model))
    os.makedirs(out_dir, exist_ok=True)

    print("[demo] %s | %d cultures x %d neurons x %d layer(s) = %d sims | bump window %.0f ms"
          % (cell_model, cultures, neurons, len(CFG.layers_um),
             cultures * neurons * len(CFG.layers_um), CFG.bump_ms))
    print("[demo] morphologies: %s | seed %d | out %s"
          % (", ".join(CFG.morphologies), seed, out_dir))

    t0 = time.time()
    paths = ce.culture_export(n_cultures=int(cultures), neurons_per_culture=int(neurons),
                              layers=CFG.layers_um, seed=int(seed), i0_uA=i0_uA,
                              csv_path=os.path.join(out_dir, "culture_Pactivation.csv"),
                              make_figures=bool(figures), block=max(1, int(neurons)))
    print("[demo] simulation done in %.1f s" % (time.time() - t0))

    if CFG.bump_ms > 0:
        import plot_bump_statistics as pbs
        pbs.main(paths[0], bin_um=float(bin_um))
    else:
        print("[demo] bump_ms = 0, so there are no kinetics to plot")
    return out_dir


def _cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-model", default="full_tuned",
                    choices=["full_tuned", "full_active", "soma_only"])
    ap.add_argument("--cultures", type=int, default=3)
    ap.add_argument("--neurons", type=int, default=6)
    ap.add_argument("--layers", default="80", help="comma-separated, um (default 80)")
    ap.add_argument("--morphologies", default=None,
                    help="comma-separated (default: the first two in config)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--bump-ms", type=float, default=800.0, help="0 disables the kinetics")
    ap.add_argument("--i0-uA", type=float, default=None)
    ap.add_argument("--bin-um", type=float, default=60.0, help="distance bin of the figures")
    ap.add_argument("--out", default=None, help="default demo_<cell_model>/")
    ap.add_argument("--no-figures", action="store_true", help="skip the outcome maps")
    a = ap.parse_args(argv)
    morphs = [m.strip() for m in a.morphologies.split(",")] if a.morphologies else None
    if morphs is None:
        from config import CFG
        morphs = list(CFG.morphologies)[:2]
    return main(a.cell_model, a.cultures, a.neurons,
                [float(x) for x in a.layers.split(",")], morphs, a.seed, a.bump_ms,
                a.out, a.i0_uA, a.bin_um, not a.no_figures)


if __name__ == "__main__":
    _cli()
    sys.exit(0)
