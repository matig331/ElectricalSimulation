"""culture_worker.py -- SIMULATION ONLY. Runs an assigned subset of cultures.

One worker = one OS process = one core. It runs every culture it was assigned and appends
to its own partial CSV (schema: culture_export.CSV_HEADER). No plotting, no merging:
culture_merge.py does that.

WHAT IT SIMULATES
    Exactly what the serial culture_export() simulates, through the SAME shared core
    (culture_export.CellPool + iter_culture_blocks + build_row): the cell model is
    config.cell_model ("soma_only" for the final dataset), every row records it, and each
    simulation is classified activation / depolarization / hyperpolarization / neutral from
    DeltaV_end = Vm(end of phase 2) - Vm_sham(end of phase 2). See culture_export's docstring.

WHY THE SPLIT IS EXACT, NOT JUST STATISTICALLY EQUIVALENT
    Every culture is seeded independently:  rng = default_rng(seed + c)
    (culture_export.culture_draws). The draws for culture c depend ONLY on (seed, c), never on
    how many cultures ran before it or in what order. Running culture 7 alone reproduces
    culture 7 of a serial run bit for bit, for a given seed.

ONE LIVE CELL, BLOCKS OF NEURONS
    NEURON integrates every section that exists; keeping all morphology x layer cells alive
    (the previous design) made every simulation ~6x slower for identical results. CellPool
    keeps one live cell and rebuilds from a cached slice (~0.1-0.2 s, bit-identical). To keep
    rebuilds rare, neurons are processed in blocks of --flush-every neurons, grouped by
    (morphology, layer) inside the block; rows are still written NEURON-major and the file is
    flushed + fsync'ed after every block, so a culture cut short by the walltime is an
    unbiased random subset of COMPLETE neurons (all layers each).

TWO WAYS TO GET MORE CULTURES WITHOUT OVERLAP -- pick ONE per set of jobs, don't mix:
  (a) SAME seed, DIFFERENT culture-index ranges (--ids 0-19 vs --ids 20-39). This is
      what jobs/submit_batches.sh does via CULTURE_OFFSET. No --seed needed.
  (b) DIFFERENT --seed, SAME local culture-index range (0..N-1) per job -- independent
      replicates. Each row records which seed produced it (the CSV 'seed' column), so
      culture_merge.py can tell "same seed, same culture" (a true duplicate, rejected)
      apart from "different seed, same LOCAL culture index" (a legitimate independent
      replicate, allowed and remapped to its own global id). See jobs/submit_seeded.sh.
  Reusing the seeds of the full-active campaign places the somata exactly where that
  campaign placed them (paired comparison); the outputs live in separate results_<model>/
  folders, so nothing is overwritten or mixed.

USAGE
    python culture_worker.py --ids 0,3,6,9 --out parts/part_00.csv
    python culture_worker.py --ids 0-9     --out parts/part_00.csv   # ranges allowed
    python culture_worker.py --ids 0-19 --out p.csv --seed 1000       # replicate run
    python culture_worker.py --ids 4 --out p.csv --neurons 20        # override N (testing)
"""
import argparse
import csv
import os
import time

from culture_export import CSV_HEADER   # single source of truth for the schema


def parse_ids(spec):
    """Parse '0,3,6' or '0-9' or '0-3,7,10-12' into a sorted list of unique ints.

    Kept dependency-free and importable so the smoke test can exercise it with no NEURON.
    """
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-") and not part.startswith("-"):
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                raise ValueError("bad range (hi < lo): %r" % part)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("no culture ids parsed from %r" % (spec,))
    if min(out) < 0:
        raise ValueError("culture ids must be >= 0, got %d" % min(out))
    return sorted(out)


def split_ids(n_cultures, n_workers):
    """Round-robin culture ids over workers -> list (len n_workers) of id-lists.

    Round-robin rather than contiguous blocks so that if cultures ever differ in cost,
    the load stays balanced. Every id appears exactly once across all workers.
    """
    n_cultures, n_workers = int(n_cultures), int(n_workers)
    if n_cultures < 1 or n_workers < 1:
        raise ValueError("n_cultures and n_workers must be >= 1")
    n_workers = min(n_workers, n_cultures)          # never spawn idle workers
    return [list(range(w, n_cultures, n_workers)) for w in range(n_workers)]


def run_worker(culture_ids, out_csv, neurons_per_culture=None, layers=None,
               span_um=None, i0_uA=None, distance_mode=None, seed=None,
               flush_every=25, quiet=False):
    """Simulate the given cultures and write one partial CSV. Returns the CSV path.

    `seed`: None -> config.seed (offset-split jobs); an explicit int -> an independent
    replicate run (jobs/submit_seeded.sh). The seed actually used is in every row.
    `flush_every`: neurons per block (= flush + fsync interval). `distance_mode` is accepted
    for API compatibility only (every distance metric is written to the CSV).
    """
    from config import CFG
    from culture_export import (CellPool, culture_draws, culture_has_kinetics,
                                iter_culture_blocks, resolve_cell_model, electrode_center,
                                dipole_axis_deg, dipole_frame)
    import field as F

    cfg = CFG
    flush_every = int(flush_every)
    if flush_every < 1:
        raise ValueError("flush_every must be >= 1, got %d" % flush_every)
    seed_used = cfg.seed if seed is None else int(seed)
    cell_model = resolve_cell_model(cfg)
    N = cfg.n_neurons_effective() if neurons_per_culture is None else int(neurons_per_culture)
    layers = list(layers or cfg.layers_um)
    span = cfg.span_half_um() if span_um is None else float(span_um)
    i0 = cfg.i0_uA if i0_uA is None else float(i0_uA)
    n_pulses = cfg.n_pulses_for_duration()
    morphs = cfg.morphologies
    M = len(morphs)

    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=not cfg.bipolar)
    center = electrode_center(elec)
    axis = dipole_axis_deg(elec, sign)
    dip_c, dip_d = dipole_frame(elec, sign)
    pid = os.getpid()

    if not quiet:
        n_sim = len(culture_ids) * N * len(layers)
        print("[worker pid=%d] model=%s seed=%d cultures %s | %d neurons x %d layers = %d sims "
              "@ %.0f uA | block/flush every %d neurons"
              % (pid, cell_model, seed_used, culture_ids, N, len(layers), n_sim, i0,
                 flush_every), flush=True)

    d_out = os.path.dirname(os.path.abspath(out_csv))
    if d_out and not os.path.isdir(d_out):
        os.makedirs(d_out, exist_ok=True)

    with CellPool(cfg, cell_model) as pool:
        # rest + sham of every (morphology, layer) once, up front: visible at the top of the log
        for layer in layers:
            for m in morphs:
                p = pool.info(m, layer)
                if not quiet:
                    print("[worker pid=%d] %s L%d um: v_rest %.4f mV | sham drift at end of "
                          "phase 2 %+.6f mV" % (pid, m, int(layer), p["v_rest"], p["ctrl_drift"]),
                          flush=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_HEADER)
            fh.flush()
            os.fsync(fh.fileno())
            for c in culture_ids:
                d = culture_draws(seed_used, c, N, M, span, elec, center, dip_c, dip_d, axis,
                                  cfg.h_soma_um)
                t0 = time.time()
                # the long post-pulse window on a (seed, culture)-determined subsample of
                # cultures (config.bump_culture_fraction). This line was missing: the serial
                # driver honoured the fraction and this, the PRODUCTION path, silently did not.
                wk = culture_has_kinetics(seed_used, c,
                                          getattr(cfg, "bump_culture_fraction", 1.0))
                for rows, n_done in iter_culture_blocks(pool, c, d, morphs, layers, i0,
                                                        n_pulses, seed_used, flush_every,
                                                        with_kinetics=wk):
                    w.writerows(rows)
                    # flush() alone only reaches the LOCAL node's page cache; on a shared
                    # filesystem the login node can still see size 0. fsync() forces it out
                    # so progress is visible (and durable) from anywhere, immediately.
                    fh.flush()
                    os.fsync(fh.fileno())
                    if not quiet:
                        dt = time.time() - t0
                        print("[worker pid=%d] culture %d: %d/%d neurons | %.2f s/sim | %d builds"
                              % (pid, c, n_done, N, dt / max(1, n_done * len(layers)),
                                 pool.n_builds), flush=True)
                if not quiet:
                    print("[worker pid=%d] culture %d done" % (pid, c), flush=True)
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Simulate a subset of cultures (one worker).")
    ap.add_argument("--ids", required=True,
                    help="culture ids: '0,3,6' or '0-9' or '0-3,7'")
    ap.add_argument("--out", required=True, help="partial CSV to write")
    ap.add_argument("--neurons", type=int, default=None,
                    help="override neurons per culture (default: config n_neurons_effective())")
    ap.add_argument("--i0", type=float, default=None, help="override amplitude (uA)")
    ap.add_argument("--span", type=float, default=None, help="override sampling half-width (um)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the RNG base (default: config.seed). Use a distinct value "
                         "per job for an independent replicate run -- see jobs/submit_seeded.sh.")
    ap.add_argument("--job-name", default=None,
                    help="label only, for log messages -- not written to the CSV (the 'seed' "
                         "column is what makes a row's provenance unambiguous)")
    ap.add_argument("--flush-every", type=int, default=25,
                    help="neurons per block = flush+fsync interval (default 25). Larger = fewer "
                         "cell rebuilds, progress visible less often.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.flush_every < 1:
        ap.error("--flush-every must be >= 1")
    if a.job_name and not a.quiet:
        print("[worker pid=%d] job-name=%s" % (os.getpid(), a.job_name), flush=True)
    run_worker(parse_ids(a.ids), a.out, neurons_per_culture=a.neurons,
               span_um=a.span, i0_uA=a.i0, seed=a.seed,
               flush_every=a.flush_every, quiet=a.quiet)


if __name__ == "__main__":
    main()
