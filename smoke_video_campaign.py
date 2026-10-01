"""smoke_video_campaign.py -- checks of the campaign videos (videos of a campaign job's own
cultures), no simulation.

    python smoke_video_campaign.py        # ~1 min, ends with ALL PASSED

  1  make_prob_videos.gaussian_filter (numpy only) equals a direct 2-D Gaussian sum with zero
     padding -- and scipy.ndimage.gaussian_filter(mode="constant") where scipy exists
  2  make_prob_videos and make_culture_videos import without scipy (the cluster has none)
  3  video_campaign_check: identical rows pass; a DeltaV_end 1e-5 mV off, a flipped spike, a
     soma 0.05 um off and a row the campaign lacks fail (the last passes with --allow-missing);
     --seed-of reads a job's seed
  4  the precheck regenerates placements with culture_draws: the campaign's culture size,
     square and seed pass; another culture size, the legacy square, another seed fail
  5  jobs/video_frames.pbs with JOB=<job> (video_frames.py on its NEURON-free toy pool, as in
     smoke_video_frames.py): runs precheck -> simulation -> check and finishes; FRAMES_DIR renders
     an earlier run again without simulating (VIDEOS=culture, PLOT_NEURONS drawn); refuses
     FRAMES_DIR with JOB, NEURONS or SPAN with JOB, and a SEED that is not the job's; stops before
     simulating when the placements differ, and before rendering when a DeltaV_end differs (needs
     NEURON installed only because the job imports rich_footprint -- skipped otherwise)
  6  pure ASCII, LF only
Exit status 1 on any failure.
"""
import csv
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


# ----------------------------------------------------------------------------- 1, 2
def direct_gauss(a, s, truncate=4.0):
    """Reference: sum over the (2r+1)^2 window of a[i+di, j+dj] w[di] w[dj], zero outside."""
    r = int(truncate * s + 0.5)
    x = np.arange(-r, r + 1, dtype=float)
    w = np.exp(-0.5 * (x / s) ** 2)
    w /= w.sum()
    out = np.zeros_like(a, dtype=float)
    ny, nx = a.shape
    for i in range(ny):
        for j in range(nx):
            acc = 0.0
            for di in range(-r, r + 1):
                ii = i + di
                if ii < 0 or ii >= ny:
                    continue
                for dj in range(-r, r + 1):
                    jj = j + dj
                    if 0 <= jj < nx:
                        acc += a[ii, jj] * w[di + r] * w[dj + r]
            out[i, j] = acc
    return out


def test_gaussian():
    print("1  numpy Gaussian smoothing of the probability maps")
    import make_prob_videos as P
    rng = np.random.default_rng(3)
    worst, worst_sp = 0.0, None
    for shape, s in (((15, 17), 1.3), ((9, 11), 5.0), ((20, 20), 0.3), ((12, 7), 2.0)):
        a = rng.poisson(0.8, size=shape).astype(float)
        worst = max(worst, float(np.max(np.abs(P.gaussian_filter(a, s) - direct_gauss(a, s)))))
    check(worst < 1e-12, "equals the direct 2-D sum (max difference %.1e)" % worst)
    try:
        from scipy.ndimage import gaussian_filter as sp_gf
        worst_sp = 0.0
        for shape, s in (((70, 70), 2.0), ((71, 53), 0.7), ((9, 11), 5.0)):
            a = rng.poisson(0.8, size=shape).astype(float)
            worst_sp = max(worst_sp, float(np.max(np.abs(P.gaussian_filter(a, s)
                                                         - sp_gf(a, s, mode="constant")))))
        check(worst_sp < 1e-12, "equals scipy.ndimage.gaussian_filter(mode='constant') "
                                "(max difference %.1e)" % worst_sp)
    except ImportError:
        print("  --    scipy not installed here: compared with the direct sum only")


def test_no_scipy():
    print("2  the renderers import without scipy")
    code = ("import sys; sys.modules['scipy'] = None; sys.modules['scipy.ndimage'] = None; "
            "import make_prob_videos, make_culture_videos; print('imported')")
    r = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True)
    check(r.returncode == 0 and "imported" in r.stdout,
          "make_prob_videos + make_culture_videos with scipy blocked%s"
          % ("" if r.returncode == 0 else " -- " + r.stderr.strip().splitlines()[-1]))


# ----------------------------------------------------------------------------- helpers
def campaign_parts(parts_dir, seed, cultures, N, layers=(80,), legacy=False, fake_dv=None):
    """Campaign-like parts: CSV_HEADER rows whose placements come from culture_draws with the
    current (or legacy) square; fired / DeltaV_end from fake_dv(c, i, L) -> (fired, dv)."""
    from config import CFG
    import culture_export as CE
    import video_frames as VF
    g = VF._geometry(CFG)
    span, centre = CE.placement_frame(CFG, g["elec"], g["sign"], legacy=legacy)
    morphs = [str(m) for m in CFG.morphologies]
    dc, dd = g["dip"]
    H = CE.CSV_HEADER
    os.makedirs(parts_dir, exist_ok=True)
    with open(os.path.join(parts_dir, "part_000.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(H)
        for c in cultures:
            d = CE.culture_draws(seed, c, N, len(morphs), span, g["elec"], g["center"], dc, dd,
                                 g["axis"], CFG.h_soma_um, place_center=centre)
            for i in range(N):
                for L in layers:
                    fired, dv = fake_dv(c, i, L) if fake_dv else (0, 0.0)
                    row = [""] * len(H)
                    vals = dict(culture=c, neuron=i, morphology=morphs[int(d["midx"][i])],
                                layer_um=int(L), x_um=round(float(d["pos"][i, 0]), 2),
                                y_um=round(float(d["pos"][i, 1]), 2),
                                dist_dipole3d_um=round(float(d["r_dip"][i]), 2), fired=int(fired),
                                theta_orient_deg=round(float(d["th_or"][i]), 1),
                                seed=seed, cell_model="full_tuned",
                                deltaVm_end_phase2_mV=round(float(dv), 6))
                    for k, v in vals.items():
                        row[H.index(k)] = v
                    w.writerow(row)


def video_from_parts(parts_dir, video_dir, seed):
    """A culture_video_neurons file holding exactly the campaign's values."""
    import video_frames as VF
    os.makedirs(video_dir, exist_ok=True)
    with open(os.path.join(parts_dir, "part_000.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(os.path.join(video_dir, "culture_video_neurons_S%d.csv" % seed), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(VF.NEURON_COLS)
        for r in rows:
            w.writerow([r["seed"], r["culture"], r["neuron"], r["morphology"], r["layer_um"],
                        r["x_um"], r["y_um"], 0.0, r["dist_dipole3d_um"], 0.0, r["fired"],
                        r["deltaVm_end_phase2_mV"], "x"])
    return os.path.join(video_dir, "culture_video_neurons_S%d.csv" % seed)


def edit_video(path, fn):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fn(rows)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_check(*argv):
    import video_campaign_check as VC
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = VC.main(list(argv))
    return rc, buf.getvalue()


# ----------------------------------------------------------------------------- 3, 4
def test_check(tmp):
    print("3  video_campaign_check: the video rows against the campaign rows")
    fake = lambda c, i, L: (int(i % 5 == 0), 0.001 * (i - 3) * (L / 40.0) + 0.1234567 * c)
    parts = os.path.join(tmp, "parts_A")
    campaign_parts(parts, 50000, [0, 1], 6, layers=(40, 80), fake_dv=fake)
    vdir = os.path.join(tmp, "video_A")
    vpath = video_from_parts(parts, vdir, 50000)
    rc, out = run_check("--video-dir", vdir, "--parts", parts)
    check(rc == 0 and "24 agree" in out, "identical rows pass (24 = 2 cultures x 6 x 2 layers)")
    for tag, fn in (("DeltaV_end 1e-5 mV off",
                     lambda rows: rows[5].update(dv_end_mV=repr(float(rows[5]["dv_end_mV"]) + 1e-5))),
                    ("a flipped spike", lambda rows: rows[2].update(fired=str(1 - int(rows[2]["fired"])))),
                    ("a soma 0.05 um off", lambda rows: rows[7].update(x=repr(float(rows[7]["x"]) + 0.05)))):
        vpath = video_from_parts(parts, vdir, 50000)
        edit_video(vpath, fn)
        rc, out = run_check("--video-dir", vdir, "--parts", parts)
        check(rc == 1 and "1 disagree" in out, "%s fails" % tag)
    vpath = video_from_parts(parts, vdir, 50000)
    edit_video(vpath, lambda rows: rows.append(dict(rows[-1], neuron="6")))
    rc, out = run_check("--video-dir", vdir, "--parts", parts)
    check(rc == 1 and "1 missing" in out, "a video row the campaign lacks fails")
    rc, out = run_check("--video-dir", vdir, "--parts", parts, "--allow-missing")
    check(rc == 0, "... and passes with --allow-missing (a job still running)")
    rc, out = run_check("--seed-of", parts)
    check(rc == 0 and out.strip() == "50000", "--seed-of reads the job's seed (%s)" % out.strip())


def test_precheck(tmp):
    print("4  precheck: placements regenerated before any simulation")
    import video_campaign_check as VC
    good = os.path.join(tmp, "parts_P")
    campaign_parts(good, 51000, [2, 3], 9)
    n, probs = VC.precheck(good, 51000, [2, 3], n_neurons=9)
    check(n == 18 and not probs, "the campaign's culture size, square and seed: %d neurons agree" % n)
    n, probs = VC.precheck(good, 51000, [2, 3], n_neurons=10)
    check(len(probs) > 0, "another culture size (10 instead of 9) fails (%d differ)" % len(probs))
    n, probs = VC.precheck(good, 51000, [2, 3], n_neurons=8)
    check(len(probs) > 0, "a smaller culture size (8) fails (%d differ)" % len(probs))
    legacy = os.path.join(tmp, "parts_L")
    campaign_parts(legacy, 51000, [2], 9, legacy=True)
    n, probs = VC.precheck(legacy, 51000, [2], n_neurons=9)
    check(len(probs) == 9, "parts of the legacy square (+/-500 um around (0, 0)) fail (%d of 9)"
          % len(probs))
    other = os.path.join(tmp, "parts_S")
    campaign_parts(other, 52000, [2], 9)
    n, probs = VC.precheck(other, 51000, [2], n_neurons=9)
    check(len(probs) > 0, "a culture absent from the parts (other seed) fails")
    n, probs = VC.precheck(good, 51000, [4], n_neurons=9, allow_missing=True)
    check(not probs, "... unless --allow-missing")


# ----------------------------------------------------------------------------- 5
def make_job_repo(tmp, n_neurons):
    repo = os.path.join(tmp, "repo")
    if os.path.isdir(repo):
        shutil.rmtree(repo)
    os.makedirs(os.path.join(repo, "jobs"))
    for f in os.listdir(HERE):
        if f.endswith(".py"):
            shutil.copy(os.path.join(HERE, f), repo)
    for f in ("video_frames.pbs", "results_layout.sh"):
        shutil.copy(os.path.join(HERE, "jobs", f), os.path.join(repo, "jobs"))
    with open(os.path.join(repo, "jobs", "env_setup.sh"), "w") as fh:
        fh.write("export MPLBACKEND=Agg\n")               # the real one activates conda + NEURON
    cfg = os.path.join(repo, "config.py")
    src = open(cfg).read()
    src = src.replace('    cell_model: str = "soma_only"', '    cell_model: str = "full_tuned"', 1)
    src = src.replace("neurons_per_culture: Optional[int] = 2000",
                      "neurons_per_culture: Optional[int] = %d" % n_neurons, 1)
    open(cfg, "w").write(src)
    shim = os.path.join(tmp, "bin")
    os.makedirs(shim, exist_ok=True)
    if not os.path.exists(os.path.join(shim, "python")):
        os.symlink(sys.executable, os.path.join(shim, "python"))
    env = dict(os.environ, PATH=shim + os.pathsep + os.environ["PATH"], VIDEO_FRAMES_FAKE="1",
               PYTHONDONTWRITEBYTECODE="1", PBS_NUM_PPN="2", SMOKE="0", RENDER="0",
               N_CULTURES="2", LAYERS="80", TMAX="20")
    for k in ("JOB", "SEED", "NEURONS", "SPAN", "MODEL", "OUT", "FIRST", "HALF",
              "ALLOW_MISSING", "PBS_O_WORKDIR"):
        env.pop(k, None)
    return repo, env


def job(repo, env, **extra):
    e = dict(env, **{k: str(v) for k, v in extra.items()})
    return subprocess.run(["bash", "jobs/video_frames.pbs"], cwd=repo, env=e,
                          capture_output=True, text=True)


def test_job(tmp):
    print("5  jobs/video_frames.pbs with JOB=<campaign job> (toy pool, no simulation)")
    if importlib.util.find_spec("neuron") is None:
        print("  --    skipped: NEURON is not installed here (the job imports rich_footprint)")
        return
    N = 6
    repo, env = make_job_repo(tmp, N)
    # the "campaign": the toy pool's own values, written as a job's parts (seed 50000)
    r = subprocess.run([sys.executable, "video_frames.py", "--n-cultures", "2", "--seed", "50000",
                        "--layers", "80", "--tmax-ms", "20", "--processes", "0", "--out", "ref"],
                       cwd=repo, env=env, capture_output=True, text=True)
    ref = os.path.join(repo, "ref", "culture_video_neurons_S50000.csv")
    check(r.returncode == 0 and os.path.exists(ref), "reference run of the toy pool")
    if not os.path.exists(ref):
        print(r.stdout[-800:], r.stderr[-800:])
        return
    parts = os.path.join(repo, "results_full_tuned", "parts_ftdT")
    os.makedirs(parts)
    import culture_export as CE
    import video_frames as VF
    from config import CFG
    H = CE.CSV_HEADER
    axis = VF._geometry(CFG)["axis"]
    with open(ref, newline="") as fh:
        vrows = list(csv.DictReader(fh))

    def write_parts(mutate=None):
        with open(os.path.join(parts, "part_000.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(H)
            for k, v in enumerate(vrows):
                row = [""] * len(H)
                vals = dict(seed=v["seed"], culture=v["culture"], neuron=v["neuron"],
                            morphology=v["morph"], layer_um=v["layer"], x_um=v["x"], y_um=v["y"],
                            dist_dipole3d_um=v["dist_dipole3d_um"], fired=v["fired"],
                            theta_orient_deg=round(float(np.asarray(CE.rel_orientation_deg(
                                np.asarray([float(v["theta_deg"])]), axis)).ravel()[0]), 1),
                            deltaVm_end_phase2_mV=v["dv_end_mV"], cell_model="full_tuned")
                if mutate:
                    mutate(k, vals)
                for c, x in vals.items():
                    row[H.index(c)] = x
                w.writerow(row)
    write_parts()
    r = job(repo, env, JOB="ftdT", OUT="v_ok")
    out = r.stdout + r.stderr
    check(r.returncode == 0 and "[campaign precheck] OK" in out and "[campaign check] OK" in out
          and "done -> v_ok" in out, "JOB=ftdT: precheck OK, simulation, check OK, done")
    if r.returncode != 0:
        print(out[-1500:])
    r = job(repo, env, FRAMES_DIR="v_ok", VIDEOS="culture", CULTURE_IDS="0", PLOT_NEURONS="3")
    out = r.stdout + r.stderr
    vids = [f for f in os.listdir(os.path.join(repo, "v_ok")) if f.startswith("culture_video_0_L80")]
    check(r.returncode == 0 and "render only" in out and "[video_frames]" not in out
          and "video 1" not in out and "3 of 6 neurons drawn" in out and vids,
          "FRAMES_DIR=v_ok VIDEOS=culture PLOT_NEURONS=3: renders the culture video again, 3 of "
          "6 neurons, no simulation, no probability video")
    if r.returncode != 0:
        print(out[-1500:])
    r = job(repo, env, FRAMES_DIR="v_ok", JOB="ftdT")
    check(r.returncode != 0 and "give one of the two" in r.stderr, "FRAMES_DIR with JOB is refused")
    r = job(repo, env, JOB="ftdT", NEURONS=N, OUT="v_n")
    check(r.returncode != 0 and "leave NEURONS and SPAN unset" in r.stderr,
          "NEURONS with JOB is refused")
    r = job(repo, env, JOB="ftdT", SEED=1, OUT="v_s")
    check(r.returncode != 0 and "ran with seed 50000" in r.stderr, "a SEED that is not the job's "
          "is refused")
    write_parts(lambda k, vals: vals.update(deltaVm_end_phase2_mV=repr(
        float(vals["deltaVm_end_phase2_mV"]) + 1e-4)) if k == 3 else None)
    r = job(repo, env, JOB="ftdT", OUT="v_dv", RENDER="1")
    out = r.stdout + r.stderr
    check(r.returncode != 0 and "[campaign check] FAILED" in out and "video 1" not in out
          and "done ->" not in out, "a DeltaV_end that differs stops the job before rendering")
    write_parts(lambda k, vals: vals.update(x_um=repr(float(vals["x_um"]) + 1.0)))
    r = job(repo, env, JOB="ftdT", OUT="v_pos")
    out = r.stdout + r.stderr
    frames = [f for f in os.listdir(os.path.join(repo, "v_pos"))
              if f.startswith("culture_frames")] if os.path.isdir(os.path.join(repo, "v_pos")) else []
    check(r.returncode != 0 and "[campaign precheck] FAILED" in out and not frames,
          "placements that differ stop the job before any simulation")


# ----------------------------------------------------------------------------- 6
def test_ascii():
    print("6  pure ASCII, LF only")
    files = ["make_prob_videos.py", "make_culture_videos.py", "smoke_culture_videos.py",
             "video_campaign_check.py", "smoke_video_campaign.py",
             "culture_merge.py", "smoke_test_campaign_analysis.py", "video_frames.py",
             "jobs/video_frames.pbs", "jobs/analysis.pbs", "jobs/merge_all.sh"]
    bad = [f for f in files
           if (lambda b: b"\r" in b or any(x > 127 for x in b))(open(os.path.join(HERE, f), "rb").read())]
    check(not bad, "%d files%s" % (len(files), (" -- not clean: %s" % bad) if bad else ""))


def main():
    tmp = tempfile.mkdtemp(prefix="smoke_video_campaign_")
    try:
        test_gaussian()
        test_no_scipy()
        test_check(tmp)
        test_precheck(tmp)
        test_job(tmp)
        test_ascii()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("-" * 70)
    if FAILS:
        print("FAILED: %d check(s)" % len(FAILS))
        for m in FAILS:
            print("   - " + m)
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
