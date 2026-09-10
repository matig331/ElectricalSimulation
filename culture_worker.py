"""culture_worker.py -- SIMULATION ONLY. Runs an assigned subset of cultures.

One worker = one OS process = one core. It builds the 9 (morphology x layer) cells
ONCE, then runs every culture it was assigned, appending to its own partial CSV.
No plotting, no merging: culture_merge.py does that.

WHY THE SPLIT IS EXACT, NOT JUST STATISTICALLY EQUIVALENT
    culture_export() seeds each culture independently:  rng = default_rng(seed + c).
    The draws for culture c therefore depend ONLY on (seed, c), never on how many
    cultures ran before it or in what order. Running culture 7 alone reproduces
    culture 7 of a serial run bit for bit, for a given seed.

WHY WHOLE CULTURES AND NOT INDIVIDUAL NEURONS
    Building the cells (slicing + Import3d + d_lambda on large .asc trees) is a fixed
    cost B paid once per PROCESS, not per culture. Splitting at the neuron level would
    force every core to pay B while giving each fewer simulations to amortise it over.

TWO WAYS TO GET MORE CULTURES WITHOUT OVERLAP -- pick ONE per set of jobs, don't mix:
  (a) SAME seed, DIFFERENT culture-index ranges (--ids 0-19 vs --ids 20-39). This is
      what jobs/submit_batches.sh does via CULTURE_OFFSET. No --seed needed.
  (b) DIFFERENT --seed, SAME local culture-index range (0..N-1) per job -- independent
      replicates. Each row records which seed produced it (the CSV 'seed' column), so
      culture_merge.py can tell "same seed, same culture" (a true duplicate, rejected)
      apart from "different seed, same LOCAL culture index" (a legitimate independent
      replicate, allowed and remapped to its own global id for correct per-culture
      statistics). See jobs/submit_seeded.sh.

USAGE
    python culture_worker.py --ids 0,3,6,9 --out parts/part_00.csv
    python culture_worker.py --ids 0-9     --out parts/part_00.csv   # ranges allowed
    python culture_worker.py --ids 0-19 --out p.csv --seed 1000       # replicate run
    python culture_worker.py --ids 4 --out p.csv --neurons 20        # override N (testing)
"""
import argparse
import csv
import os

import numpy as np

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

    Mirrors culture_export()'s inner loop exactly (same seeding, same column order),
    minus the plotting. `seed`: None -> use config.seed (offset-split jobs, the usual
    case); an explicit int -> an independent replicate run (see jobs/submit_seeded.sh).
    Either way the seed actually used is recorded in every row's 'seed' column.
    """
    from config import CFG
    from morphologies import find_one_morphology
    from slicer import reduced_asc
    from rich_cell import build_rich_cell
    from rich_footprint import spikes_at
    from culture_export import (assign_morphologies, dist_from_nearest_electrode,
                                dist_from_center, directional_rt, rel_orientation_deg,
                                electrode_center, dipole_axis_deg, dipole_frame)
    import field as F

    cfg = CFG
    seed_used = cfg.seed if seed is None else int(seed)
    N = cfg.n_neurons_effective() if neurons_per_culture is None else int(neurons_per_culture)
    layers = layers or cfg.layers_um
    span = cfg.span_half_um() if span_um is None else float(span_um)
    i0 = cfg.i0_uA if i0_uA is None else float(i0_uA)
    distance_mode = (getattr(cfg, "culture_distance_mode", "centroid")
                     if distance_mode is None else distance_mode)
    n_pulses = cfg.n_pulses_for_duration()
    morphs = cfg.morphologies
    M = len(morphs)

    elec, sign = F.default_array(pitch_um=cfg.pitch_um, monopolar=not cfg.bipolar)
    center = electrode_center(elec)
    axis = dipole_axis_deg(elec, sign)
    dip_c, dip_d = dipole_frame(elec, sign)

    if not quiet:
        n_sim = len(culture_ids) * N * len(layers)
        print("[worker pid=%d] seed=%d cultures %s | %d neurons x %d layers = %d sims @ %.0f uA"
              % (os.getpid(), seed_used, culture_ids, N, len(layers), n_sim, i0), flush=True)

    # Build once per (morphology, layer) -- the fixed cost this whole design amortises.
    cells = {}
    for layer in layers:
        for m in morphs:
            tag = "_cw%d_%s_%d.asc" % (os.getpid(), m, int(layer))   # pid-tagged: no clash
            asc = reduced_asc(find_one_morphology(m), layer, out_path=tag)
            cells[(m, layer)] = build_rich_cell(asc)
            if asc and os.path.exists(asc):
                os.remove(asc)
    if not quiet:
        print("[worker pid=%d] %d cells built" % (os.getpid(), len(cells)), flush=True)

    d_out = os.path.dirname(os.path.abspath(out_csv))
    if d_out and not os.path.isdir(d_out):
        os.makedirs(d_out, exist_ok=True)

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for c in culture_ids:
            rng = np.random.default_rng(seed_used + c)   # IDENTICAL rule to culture_export()
            midx = assign_morphologies(N, M, rng)
            pos = rng.uniform(-span, span, size=(N, 2))
            theta = rng.uniform(0, 360, size=N)
            d_near = dist_from_nearest_electrode(pos, elec)
            d_cen = dist_from_center(pos, center)
            r_dip, th_pos = directional_rt(pos, dip_c, dip_d, z_um=cfg.h_soma_um)
            th_or = rel_orientation_deg(theta, axis)
            # NEURON-MAJOR (not layer-major) on purpose: each neuron is written with
            # all its layers together, so a culture cut short by the walltime is an
            # unbiased random SUBSET OF NEURONS rather than a layer-40-only sample.
            # The set of rows is identical either way -- only the order changes.
            for i in range(N):
                for layer in layers:                     # SAME placement, vary layer
                    fired = int(spikes_at(cells[(morphs[midx[i]], layer)],
                                          (float(pos[i, 0]), float(pos[i, 1])),
                                          float(theta[i]), i0_uA=i0)[0] > 0)
                    w.writerow([c, i, morphs[midx[i]], int(layer),
                                round(float(pos[i, 0]), 2), round(float(pos[i, 1]), 2),
                                round(float(d_near[i]), 2), round(float(d_cen[i]), 2),
                                round(float(r_dip[i]), 2), round(float(th_or[i]), 1),
                                round(float(th_pos[i]), 1), n_pulses, round(i0, 1), fired,
                                seed_used])
                # flush() alone only reaches the LOCAL node's page cache; on a shared
                # filesystem the login node can still see size 0. fsync() forces it out
                # so progress is visible (and durable) from anywhere, immediately.
                if (i + 1) % flush_every == 0 or i == N - 1:
                    fh.flush()
                    os.fsync(fh.fileno())
                    if not quiet:
                        print("[worker pid=%d] culture %d: %d/%d neurons"
                              % (os.getpid(), c, i + 1, N), flush=True)
            fh.flush()
            os.fsync(fh.fileno())
            if not quiet:
                print("[worker pid=%d] culture %d done" % (os.getpid(), c), flush=True)
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
                    help="flush+fsync every N neurons (default 25). Lower = more frequent\n"
                         "partial saves and progress visible sooner; each fsync costs a little I/O.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.job_name and not a.quiet:
        print("[worker pid=%d] job-name=%s" % (os.getpid(), a.job_name), flush=True)
    run_worker(parse_ids(a.ids), a.out, neurons_per_culture=a.neurons,
               span_um=a.span, i0_uA=a.i0, seed=a.seed,
               flush_every=a.flush_every, quiet=a.quiet)


if __name__ == "__main__":
    main()
