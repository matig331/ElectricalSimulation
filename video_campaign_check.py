"""video_campaign_check.py -- the neurons of a video run ARE the campaign's neurons.

video_frames.py re-simulates cultures (seed, c) with the campaign's code and writes, for every
neuron x layer, culture_video_neurons_S<seed>.csv: specimen, soma position, distance to the
dipole centre, whether it fired and DeltaV_end(p) = V_stim(t_end | p) - V_sham(t_end | m, L),
where p = (m, L, x, y, theta) is the placement (morphology m, slice thickness L, soma position
(x, y), orientation theta) and t_end the end of phase 2. This script compares those rows with
the campaign's raw rows (results_<model>/parts_<job>/part_*.csv) of the same
(seed, culture, neuron, layer):

    morphology                       identical
    x, y, distance to dipole centre  within 0.011 um (both stored to 0.01 um)
    fired                            identical
    DeltaV_end                       within 2e-6 mV (both stored to 1e-6 mV)

Every video row must find its campaign row, unless --allow-missing (a culture of a job that is
still running). Exit 0 = the video shows the campaign's neurons; 1 = it does not (the video job
then stops before rendering).

    python video_campaign_check.py --video-dir video_run_X --parts results_full_tuned/parts_ftd01
    python video_campaign_check.py --seed-of results_full_tuned/parts_ftd01   # prints the seed
Smoke test: python smoke_video_campaign.py
"""
import argparse
import csv
import glob
import os
import sys

TOL_XY_UM = 0.011
TOL_TH_DEG = 0.051        # theta_orient_deg is stored to 0.1 deg
TOL_DV_MV = 2e-6
CAMPAIGN_COLS = ("seed", "culture", "neuron", "layer_um", "morphology", "x_um", "y_um",
                 "dist_dipole3d_um", "fired", "deltaVm_end_phase2_mV")


def _part_files(parts_dir):
    files = sorted(glob.glob(os.path.join(parts_dir, "part_*.csv")))
    if not files:
        raise SystemExit("no part_*.csv in %s" % parts_dir)
    return files


def seed_of(parts_dir):
    """The seed of a campaign job: the seed column of the first complete data row."""
    for f in _part_files(parts_dir):
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd, None)
            if not head or "seed" not in head:
                continue
            for row in rd:
                if len(row) == len(head) and row[head.index("seed")].strip():
                    return int(float(row[head.index("seed")]))
    raise SystemExit("%s: no data row with a seed yet" % parts_dir)


def load_video(video_dir):
    """{(seed, culture, neuron, layer): row dict} from culture_video_neurons_S*.csv."""
    files = sorted(glob.glob(os.path.join(video_dir, "culture_video_neurons_S*.csv")))
    if not files:
        raise SystemExit("no culture_video_neurons_S*.csv in %s -- did video_frames.py run?"
                         % video_dir)
    rows = {}
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (int(r["seed"]), int(r["culture"]), int(r["neuron"]),
                       int(round(float(r["layer"]))))
                rows[key] = r
    return rows


def load_campaign(parts_dir, keys):
    """Campaign rows of the given keys (other rows are not kept). A row with the wrong number
    of fields (the last row of a part still being written) is ignored."""
    want = set(keys)
    found = {}
    for f in _part_files(parts_dir):
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd, None)
            if not head:
                continue
            miss = [c for c in CAMPAIGN_COLS if c not in head]
            if miss:
                raise SystemExit("%s: not a campaign part (missing %s)" % (f, miss))
            ix = {c: head.index(c) for c in CAMPAIGN_COLS}
            for row in rd:
                if len(row) != len(head):
                    continue
                try:
                    key = (int(row[ix["seed"]]), int(row[ix["culture"]]), int(row[ix["neuron"]]),
                           int(round(float(row[ix["layer_um"]]))))
                except ValueError:
                    continue
                if key in want:
                    found[key] = {c: row[ix[c]] for c in CAMPAIGN_COLS}
    return found


def precheck(parts_dir, seed, cultures, n_neurons=None, span=None, allow_missing=False):
    """BEFORE any simulation: regenerate the placements of the cultures a video run will
    simulate -- culture_draws with video_frames.py's own geometry, numpy only, no NEURON -- and
    compare specimen, soma position and orientation with the campaign rows of those neurons. Catches a
    culture size, square or seed that differs from the campaign's, which would otherwise show
    only after the whole simulation. Returns (n_checked, problems)."""
    import numpy as np
    from config import CFG
    import culture_export as CE
    import video_frames as VF
    g = VF._geometry(CFG)
    N = int(CFG.n_neurons_effective() if n_neurons is None else n_neurons)
    sp = float(CFG.span_half_um() if span is None else span)
    morphs = [str(m) for m in CFG.morphologies]
    dc, dd = g["dip"]
    want = {(int(seed), int(c)) for c in cultures}
    rows = {}
    for f in _part_files(parts_dir):
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd, None)
            if not head:
                continue
            ix = {c: head.index(c) for c in ("seed", "culture", "neuron", "morphology",
                                             "x_um", "y_um", "theta_orient_deg")}
            for row in rd:
                if len(row) != len(head):
                    continue
                try:
                    k = (int(row[ix["seed"]]), int(row[ix["culture"]]), int(row[ix["neuron"]]))
                    if k[:2] in want and k not in rows:
                        rows[k] = (row[ix["morphology"]], float(row[ix["x_um"]]),
                                   float(row[ix["y_um"]]), float(row[ix["theta_orient_deg"]]))
                except ValueError:
                    continue
    problems, n_checked = [], 0
    for c in cultures:
        d = CE.culture_draws(int(seed), int(c), N, len(morphs), sp, g["elec"], g["center"], dc,
                             dd, g["axis"], CFG.h_soma_um, place_center=g["place_center"])
        have = sorted(k[2] for k in rows if k[:2] == (int(seed), int(c)))
        if not have and not allow_missing:
            problems.append("culture %d of seed %d has no row in %s" % (c, seed, parts_dir))
        if have and have[-1] >= N:
            problems.append("culture %d: the campaign has neuron %d, but a video culture has %d "
                            "neurons (config.neurons_per_culture changed?)" % (c, have[-1], N))
            continue
        for i in have:
            m, x, y, th = rows[(int(seed), int(c), i)]
            px, py = float(np.round(d["pos"][i, 0], 2)), float(np.round(d["pos"][i, 1], 2))
            pth = float(np.round(d["th_or"][i], 1))
            n_checked += 1
            if (m != morphs[int(d["midx"][i])] or abs(px - x) > TOL_XY_UM
                    or abs(py - y) > TOL_XY_UM or abs(pth - th) > TOL_TH_DEG):
                problems.append("seed %d culture %d neuron %d: campaign %s at (%.2f, %.2f) %.1f deg, "
                                "video draw %s at (%.2f, %.2f) %.1f deg"
                                % (seed, c, i, m, x, y, th, morphs[int(d["midx"][i])], px, py, pth))
    return n_checked, problems


def map_half(span=None, margin_um=20.0):
    """Half-width of the video maps (centred on (0, 0)): the soma square plus a margin."""
    import numpy as np
    from config import CFG
    import video_frames as VF
    g = VF._geometry(CFG)
    sp = float(CFG.span_half_um() if span is None else span)
    return int(np.ceil(sp + float(np.abs(np.asarray(g["place_center"], float)).max())
                       + float(margin_um)))


def compare(video, campaign):
    """(problems, stats): one message per disagreeing or missing row."""
    problems, missing = [], []
    worst_xy, worst_dv, n_ok, n_fired = 0.0, 0.0, 0, 0
    for key in sorted(video):
        v = video[key]
        c = campaign.get(key)
        if c is None:
            missing.append(key)
            continue
        bad = []
        if str(v["morph"]) != str(c["morphology"]):
            bad.append("morphology %s vs %s" % (v["morph"], c["morphology"]))
        dxy = max(abs(float(v["x"]) - float(c["x_um"])), abs(float(v["y"]) - float(c["y_um"])),
                  abs(float(v["dist_dipole3d_um"]) - float(c["dist_dipole3d_um"])))
        worst_xy = max(worst_xy, dxy)
        if dxy > TOL_XY_UM:
            bad.append("position differs by %.3f um" % dxy)
        if int(v["fired"]) != int(c["fired"]):
            bad.append("fired %s vs %s" % (v["fired"], c["fired"]))
        dv = abs(float(v["dv_end_mV"]) - float(c["deltaVm_end_phase2_mV"]))
        worst_dv = max(worst_dv, dv)
        if dv > TOL_DV_MV:
            bad.append("DeltaV_end %s vs %s mV" % (v["dv_end_mV"], c["deltaVm_end_phase2_mV"]))
        if bad:
            problems.append("seed %d culture %d neuron %d layer %d: %s" % (key + ("; ".join(bad),)))
        else:
            n_ok += 1
            n_fired += int(v["fired"])
    return problems, missing, dict(n_ok=n_ok, n_fired=n_fired, worst_xy=worst_xy,
                                   worst_dv=worst_dv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video-dir", help="output folder of video_frames.py")
    ap.add_argument("--parts", help="the campaign job's parts folder, e.g. "
                                    "results_full_tuned/parts_ftd01")
    ap.add_argument("--allow-missing", action="store_true",
                    help="accept video neurons the campaign has not written yet")
    ap.add_argument("--seed-of", metavar="PARTS_DIR",
                    help="print the seed of a campaign job's parts folder and exit")
    ap.add_argument("--precheck", action="store_true",
                    help="before simulating: the placements of --cultures (seed --seed) against "
                         "--parts, numpy only")
    ap.add_argument("--seed", type=int, help="with --precheck")
    ap.add_argument("--cultures", type=int, nargs="*", default=[], help="with --precheck")
    ap.add_argument("--neurons", type=int, default=None, help="with --precheck (default: config)")
    ap.add_argument("--span-um", type=float, default=None,
                    help="with --precheck / --map-half (default: config)")
    ap.add_argument("--map-half", action="store_true",
                    help="print the half-width of the video maps (soma square + 20 um) and exit")
    a = ap.parse_args(argv)
    if a.seed_of:
        print(seed_of(a.seed_of))
        return 0
    if a.map_half:
        print(map_half(a.span_um))
        return 0
    if a.precheck:
        if not (a.parts and a.seed is not None and a.cultures):
            ap.error("--precheck needs --parts, --seed and --cultures")
        n, problems = precheck(a.parts, a.seed, a.cultures, a.neurons, a.span_um, a.allow_missing)
        print("[campaign precheck] seed %d cultures %s: %d campaign neurons regenerated, %d differ"
              % (a.seed, a.cultures, n, len(problems)))
        for p in problems[:10]:
            print("  " + p)
        if problems or n == 0:
            print("[campaign precheck] FAILED -- these would not be the campaign's cultures "
                  "(culture size, soma square or seed differ from the campaign's)")
            return 1
        print("[campaign precheck] OK -- same specimens and soma positions as the campaign")
        return 0
    if not (a.video_dir and a.parts):
        ap.error("--video-dir and --parts are required (or --seed-of)")
    video = load_video(a.video_dir)
    campaign = load_campaign(a.parts, video.keys())
    problems, missing, st = compare(video, campaign)
    seeds = sorted({k[0] for k in video})
    cultures = sorted({k[1] for k in video})
    layers = sorted({k[3] for k in video})
    print("[campaign check] video rows %d (seed %s, cultures %s, layers %s) vs %s"
          % (len(video), seeds, cultures, layers, a.parts))
    print("[campaign check] %d agree (%d fired) | max |dx|,|dy|,|dr| %.3f um | max |dDeltaV_end| "
          "%.1e mV | %d missing from the campaign | %d disagree"
          % (st["n_ok"], st["n_fired"], st["worst_xy"], st["worst_dv"], len(missing),
             len(problems)))
    for p in problems[:10]:
        print("  DISAGREE " + p)
    for k in missing[:5]:
        print("  MISSING  seed %d culture %d neuron %d layer %d" % k)
    if problems or (missing and not a.allow_missing) or st["n_ok"] == 0:
        print("[campaign check] FAILED -- the video neurons are not the campaign's"
              + (" (missing rows: pass --allow-missing if that job is still running)"
                 if missing and not problems else ""))
        return 1
    print("[campaign check] OK -- the video shows the campaign's own neurons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
