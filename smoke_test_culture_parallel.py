"""smoke_test_culture_parallel.py -- OFFLINE test of the parallel split/merge.

Runs with NO NEURON and NO simulation: it exercises the id parsing, the worker/culture
split, the RNG-equivalence argument the whole design rests on, the merge validation
(overlap + truncation detection), and the shared renderer.

    python smoke_test_culture_parallel.py        # expect: 'All smoke tests passed.'

Every check prints PASS/FAIL, so a partial failure is visible rather than swallowed.
"""
import csv
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from culture_worker import CSV_HEADER, parse_ids, split_ids

FAILURES = []


def check(name, cond, detail=""):
    print("  %-62s %s" % (name, "PASS" if cond else "FAIL " + detail))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- id parsing
print("\n[1] culture-id parsing")
check("'0,3,6' -> [0,3,6]", parse_ids("0,3,6") == [0, 3, 6])
check("'0-4' -> [0,1,2,3,4]", parse_ids("0-4") == [0, 1, 2, 3, 4])
check("'0-2,7,9-10' -> [0,1,2,7,9,10]", parse_ids("0-2,7,9-10") == [0, 1, 2, 7, 9, 10])
check("'5' -> [5]", parse_ids("5") == [5])
check("duplicates collapse: '1,1,2' -> [1,2]", parse_ids("1,1,2") == [1, 2])
check("whitespace tolerated: ' 1 , 2 '", parse_ids(" 1 , 2 ") == [1, 2])
try:
    parse_ids("")
    check("empty spec rejected", False, "(no exception)")
except Exception:
    check("empty spec rejected", True)
try:
    parse_ids("5-2")
    check("reversed range rejected", False, "(no exception)")
except Exception:
    check("reversed range rejected", True)

# ------------------------------------------------------------ worker split
print("\n[2] culture -> worker split")
for n_cult, n_work in [(20, 4), (50, 48), (7, 3), (1, 1), (96, 96), (3, 10)]:
    parts = split_ids(n_cult, n_work)
    flat = [c for p in parts for c in p]
    ok_cover = sorted(flat) == list(range(n_cult))
    ok_unique = len(flat) == len(set(flat))
    ok_nonempty = all(len(p) > 0 for p in parts)
    spread = max(len(p) for p in parts) - min(len(p) for p in parts)
    check("n_cultures=%-3d n_workers=%-3d -> %d workers, every id exactly once, "
          "imbalance<=1" % (n_cult, n_work, len(parts)),
          ok_cover and ok_unique and ok_nonempty and spread <= 1,
          "(cover=%s unique=%s nonempty=%s spread=%d)"
          % (ok_cover, ok_unique, ok_nonempty, spread))
check("more workers than cultures does not create idle workers",
      len(split_ids(3, 10)) == 3)

# ------------------------------------------------------ multi-job offsets
print("\n[2b] multiple jobs via CULTURE_OFFSET (the property just asked about)")
job0_ids = {i + 0 for p in split_ids(20, 8) for i in p}     # job 0: offset=0,  n=20
job1_ids = {i + 20 for p in split_ids(20, 8) for i in p}    # job 1: offset=20, n=20
job2_ids = {i + 40 for p in split_ids(15, 4) for i in p}    # job 2: offset=40, n=15 (diff conc)
check("job0/job1/job2 culture-id sets are pairwise disjoint",
      not (job0_ids & job1_ids) and not (job0_ids & job2_ids) and not (job1_ids & job2_ids))
check("their union is exactly 0..54 with no gaps",
      job0_ids | job1_ids | job2_ids == set(range(55)))
# same offset run twice (deliberate re-run) must reproduce the SAME id set, not drift
check("re-deriving job0's ids (e.g. a resubmit) gives the identical set",
      {i + 0 for p in split_ids(20, 8) for i in p} == job0_ids)

# ------------------------------------------- the RNG-equivalence argument
print("\n[3] parallel == serial (the correctness argument)")
# culture_export and culture_worker both do: default_rng(seed + c). Reproduce that
# and confirm the draws for culture c do not depend on which cultures ran before it.
SEED, N, SPAN = 0, 25, 180.0


def draws_for(c):
    rng = np.random.default_rng(SEED + c)
    pos = rng.uniform(-SPAN, SPAN, size=(N, 2))
    theta = rng.uniform(0, 360, size=N)
    return pos, theta


serial = {c: draws_for(c) for c in range(6)}            # as in a serial 0..5 loop
shuffled = {c: draws_for(c) for c in [4, 1, 5, 0, 3, 2]}  # as in workers, any order
same = all(np.array_equal(serial[c][0], shuffled[c][0]) and
           np.array_equal(serial[c][1], shuffled[c][1]) for c in range(6))
check("culture draws depend only on the culture index, not on run order", same)
distinct = all(not np.array_equal(serial[a][0], serial[b][0])
               for a in range(6) for b in range(6) if a < b)
check("different cultures draw different placements (no accidental reuse)", distinct)

# --------------------------------------------------- merge: CSV round trip
print("\n[4] merge: concatenation, overlap and truncation detection")
tmp = tempfile.mkdtemp()
try:
    parts_dir = os.path.join(tmp, "parts")
    os.makedirs(parts_dir)

    def write_part(path, cultures, rows_per_culture=6, header=None, seed=0):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header if header is not None else CSV_HEADER)
            for c in cultures:
                for i in range(rows_per_culture):
                    w.writerow([c, i, "60303", 40, 1.0 * i, 2.0 * i, 10.0, 20.0, 21.0,
                                30.0, 40.0, 36, 50.0, i % 2, seed])

    write_part(os.path.join(parts_dir, "part_00.csv"), [0, 2])
    write_part(os.path.join(parts_dir, "part_01.csv"), [1, 3])

    sys.path.insert(0, tmp)
    import culture_merge
    rows, counts = culture_merge.read_parts(sorted(
        [os.path.join(parts_dir, f) for f in os.listdir(parts_dir)]))
    check("2 parts x 2 cultures x 6 rows -> 24 rows", len(rows) == 24, "(got %d)" % len(rows))
    check("all 4 (seed,culture) pairs present after merge",
          sorted(counts) == [(0, 0), (0, 1), (0, 2), (0, 3)])
    check("each pair has 6 rows", set(counts.values()) == {6})

    # overlapping split (SAME seed, same culture, two files) must be REJECTED
    dup_dir = os.path.join(tmp, "dup")
    os.makedirs(dup_dir)
    write_part(os.path.join(dup_dir, "part_00.csv"), [0, 1])
    write_part(os.path.join(dup_dir, "part_01.csv"), [1, 2])   # culture 1 twice, same seed
    try:
        culture_merge.read_parts(sorted(
            [os.path.join(dup_dir, f) for f in os.listdir(dup_dir)]))
        check("same-seed overlapping cultures rejected", False, "(no exception raised)")
    except SystemExit:
        check("same-seed overlapping cultures rejected", True)

    # a wrong header must be caught
    bad_dir = os.path.join(tmp, "bad")
    os.makedirs(bad_dir)
    write_part(os.path.join(bad_dir, "part_00.csv"), [0], header=["wrong", "header"])
    try:
        culture_merge.read_parts([os.path.join(bad_dir, "part_00.csv")])
        check("malformed header rejected", False, "(no exception raised)")
    except SystemExit:
        check("malformed header rejected", True)

    # truncated worker -> warning, not silent corruption
    trunc_dir = os.path.join(tmp, "trunc")
    os.makedirs(trunc_dir)
    write_part(os.path.join(trunc_dir, "part_00.csv"), [0], rows_per_culture=6)
    write_part(os.path.join(trunc_dir, "part_01.csv"), [1], rows_per_culture=2)
    _, tc = culture_merge.read_parts(sorted(
        [os.path.join(trunc_dir, f) for f in os.listdir(trunc_dir)]))
    check("unequal culture sizes detected (truncated worker)", set(tc.values()) == {6, 2})

    # --- multi-JOB merge: two DIFFERENT offset directories, disjoint cultures ---
    multi = os.path.join(tmp, "multijob")
    os.makedirs(multi)
    os.makedirs(os.path.join(multi, "parts_off0"))
    os.makedirs(os.path.join(multi, "parts_off20"))
    write_part(os.path.join(multi, "parts_off0", "part_000.csv"), [0, 1, 2])
    write_part(os.path.join(multi, "parts_off20", "part_000.csv"), [20, 21])
    import glob as _glob
    pat = os.path.join(multi, "parts_off*")
    files = sorted(_glob.glob(os.path.join(pat, "part_*.csv")))
    check("glob 'parts_off*' finds both jobs' parts", len(files) == 2)
    rows_mj, counts_mj = culture_merge.read_parts(files)
    check("two disjoint-offset jobs (same seed) merge to their union with no overlap error",
          sorted(counts_mj) == [(0, 0), (0, 1), (0, 2), (0, 20), (0, 21)])

    # a genuine cross-JOB collision (SAME seed, same local culture, two dirs) must
    # still be rejected -- this is the offset-mistake case (e.g. overlapping ranges)
    collide = os.path.join(tmp, "collide")
    os.makedirs(os.path.join(collide, "parts_off0"))
    os.makedirs(os.path.join(collide, "parts_offBAD"))
    write_part(os.path.join(collide, "parts_off0", "part_000.csv"), [0, 1])
    write_part(os.path.join(collide, "parts_offBAD", "part_000.csv"), [1, 2])  # 1 reused, same seed
    try:
        files2 = sorted(_glob.glob(os.path.join(collide, "parts_off*", "part_*.csv")))
        culture_merge.read_parts(files2)
        check("cross-job offset collision (same seed) rejected", False, "(no exception raised)")
    except SystemExit:
        check("cross-job offset collision (same seed) rejected", True)

    # --- SEED-DIFFERENTIATED replicates: SAME local culture index, DIFFERENT seed --
    # this is jobs/submit_seeded.sh's use case -- must be ALLOWED, not rejected, and
    # each (seed, culture) pair must land in its own global replicate group.
    seeded = os.path.join(tmp, "seeded")
    os.makedirs(os.path.join(seeded, "parts_run1"))
    os.makedirs(os.path.join(seeded, "parts_run2"))
    os.makedirs(os.path.join(seeded, "parts_run3"))
    write_part(os.path.join(seeded, "parts_run1", "part_000.csv"), [0, 1], seed=1000)
    write_part(os.path.join(seeded, "parts_run2", "part_000.csv"), [0, 1], seed=2000)  # same
    write_part(os.path.join(seeded, "parts_run3", "part_000.csv"), [0, 1], seed=3000)  # local ids!
    files3 = sorted(_glob.glob(os.path.join(seeded, "parts_run*", "part_*.csv")))
    rows3, counts3 = culture_merge.read_parts(files3)   # must NOT raise
    check("different-seed jobs sharing local culture ids are NOT rejected",
          sorted(counts3) == [(1000, 0), (1000, 1), (2000, 0), (2000, 1), (3000, 0), (3000, 1)])

    out_seeded = os.path.join(tmp, "seeded_out.csv")
    culture_merge.merge(os.path.join(seeded, "parts_run*"), out_seeded, make_figures=False)
    with open(out_seeded, newline="") as fh:
        merged_rows = list(csv.DictReader(fh))
    check("merged output has 'local_culture' and 'seed' columns for traceability",
          {"local_culture", "seed"} <= set(merged_rows[0].keys()))
    global_ids = {r["culture"] for r in merged_rows}
    check("6 (seed,culture) replicates -> 6 distinct GLOBAL culture ids, sequential 0..5",
          sorted(int(g) for g in global_ids) == [0, 1, 2, 3, 4, 5])
    # the point of the whole exercise: seed 1000's "culture 0" and seed 2000's
    # "culture 0" must NOT collapse into the same global replicate
    g_1000_0 = {r["culture"] for r in merged_rows
               if r["seed"] == "1000" and r["local_culture"] == "0"}
    g_2000_0 = {r["culture"] for r in merged_rows
               if r["seed"] == "2000" and r["local_culture"] == "0"}
    check("seed=1000 culture=0 and seed=2000 culture=0 map to DIFFERENT global ids "
          "(no pseudoreplication)", g_1000_0 and g_2000_0 and g_1000_0 != g_2000_0)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------- shared renderer
print("\n[5] shared renderer (serial and parallel produce the same figures)")
try:
    from culture_export import render_culture_figures, dipole_frame
    import field as F
    from config import CFG

    elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
    dip_c, dip_d = dipole_frame(elec, sign)
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(-180, 180, n)
    y = rng.uniform(-180, 180, n)
    r = np.hypot(x, y)
    fired = (rng.uniform(size=n) < np.clip(1.0 - r / 150.0, 0, 1)).astype(float)
    arr = np.column_stack([np.repeat(np.arange(4), n // 4), x, y, r, r,
                           rng.uniform(0, 180, n), np.full(n, 40.0), fired])
    out = os.path.join(tempfile.mkdtemp(), "test_map.pdf")
    render_culture_figures(arr, 180.0, 8.0, 50.0, 36, CFG, elec, sign, dip_c, dip_d, out)
    check("renderer produces a non-empty PDF from rows alone (no NEURON)",
          os.path.exists(out) and os.path.getsize(out) > 1000,
          "(size=%d)" % (os.path.getsize(out) if os.path.exists(out) else -1))
except Exception as e:
    check("renderer runs standalone", False, "(%s: %s)" % (type(e).__name__, e))

# ---------------------------------------------------------------- verdict
print()
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("All smoke tests passed.")
