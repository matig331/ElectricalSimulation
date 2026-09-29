"""smoke_test_campaign_setup.py -- OFFLINE checks (no NEURON, no scheduler) of what a campaign
launch depends on: the protocol and soma square in config.py, the placement helpers, and
jobs/launch_campaign.sh against a mock qsub / qstat.

    python smoke_test_campaign_setup.py        # ~10 s, ends with ALL PASSED

  1  config: one pulse per neuron (n_pulses 1), the rest before it split into its variable-step
     and fixed-dt parts on the bump grid, the window after it, the square and culture size
  2  placement: the square is centred on the dipole centre; the legacy frame is the one of the
     earlier campaigns; culture_draws with a centre is the origin-centred draw shifted, with the
     same morphologies and rotations (the random stream is untouched)
  3  launcher, mock scheduler: dry run submits nothing; a real run submits every job with the
     right walltime, queue, cores, seed and flush size and logs it; a second run while the jobs
     are queued is refused; FIRST resumes; seeds that would repeat an earlier run's cultures,
     a wrong model and a walltime too short for the predicted worst case are refused
  4  pure ASCII, LF only
Exit status 1 on any failure.
"""
import copy
import os
import shutil
import stat
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import CFG                                # noqa: E402
import culture_export as CE                           # noqa: E402
import field as F                                     # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def test_config():
    print("1  config: one pulse, the rest before it, the window after it, square and culture")
    base, settle = CE.pre_stim_split(CFG)
    check(CFG.n_pulses_for_duration() == 1 and int(CFG.n_pulses) == 1,
          "one pulse per neuron: n_pulses written = %d (stim_duration_s %g s x %g Hz)"
          % (CFG.n_pulses_for_duration(), CFG.stim_duration_s, CFG.stim_freq_hz))
    check(abs(base + settle - CFG.pre_stim_ms) < 1e-12 and base == CE.FIXED_PRE_MS,
          "rest %g ms = %g variable-step + %g fixed-dt" % (CFG.pre_stim_ms, settle, base))
    t_end = CFG.pre_stim_ms + 2 * CFG.phase_dur_ms
    check(abs(t_end / CFG.bump_dt_ms - round(t_end / CFG.bump_dt_ms)) < 1e-9,
          "end of phase 2 (%g ms) on the %g ms bump grid" % (t_end, CFG.bump_dt_ms))
    bad = copy.copy(CFG)
    bad.pre_stim_ms = 47.3
    try:
        CE.pre_stim_split(bad)
        check(False, "a rest putting t_end off the grid must be refused")
    except ValueError:
        check(True, "a rest putting t_end off the grid (47.3 ms) is refused")
    short = copy.copy(CFG)
    short.pre_stim_ms = 5.0
    check(CE.pre_stim_split(short) == (5.0, 0.0),
          "pre_stim_ms 5 -> no variable-step part: the earlier campaigns' protocol")
    check(CFG.n_neurons_effective() == (CFG.neurons_per_culture if CFG.use_hpc_count
                                        else CFG.n_neurons),
          "culture size %d (neurons_per_culture)" % CFG.n_neurons_effective())
    check(CFG.span_half_um() == CFG.placement_half_um and CFG.legacy_span_half_um() == max(
        CFG.network_half_um, CFG.area_half_um),
        "square half-side %g um (legacy %g um)" % (CFG.span_half_um(), CFG.legacy_span_half_um()))


def test_placement():
    print("2  placement square")
    elec, sign = F.default_array(pitch_um=CFG.pitch_um, monopolar=not CFG.bipolar)
    dip_c, dip_d = CE.dipole_frame(elec, sign)
    span, cen = CE.placement_frame(CFG, elec, sign)
    check(np.allclose(cen, dip_c) and span == CFG.placement_half_um,
          "centre = dipole centre (%g, %g) um, half-side %g um" % (cen[0], cen[1], span))
    frames = CE.placement_frames(CFG, elec, sign)
    check(len(frames) == 2 and frames[1]["legacy"] and np.allclose(frames[1]["center"], 0.0)
          and frames[1]["span"] == CFG.legacy_span_half_um(),
          "regenerating tools also try the legacy square (+/-%g um around (0, 0))"
          % frames[1]["span"])
    args = (6, CFG.span_half_um(), elec, CE.electrode_center(elec), dip_c, dip_d,
            CE.dipole_axis_deg(elec, sign), CFG.h_soma_um)
    d0 = CE.culture_draws(50000, 3, 500, *args)
    d1 = CE.culture_draws(50000, 3, 500, *args, place_center=cen)
    check(np.array_equal(d1["pos"], d0["pos"] + cen.reshape(1, 2))
          and np.array_equal(d1["midx"], d0["midx"]) and np.array_equal(d1["theta"], d0["theta"]),
          "centred draw = origin draw + centre; same morphologies and rotations")
    check(bool(np.all(np.abs(d1["pos"] - cen) <= span)),
          "all 500 somata inside +/-%g um around the centre" % span)
    rmax = float(np.sqrt(2 * span ** 2 + CFG.h_soma_um ** 2))
    check(float(d1["r_dip"].max()) <= rmax + 1e-9 and float(d1["r_dip"].min()) >= CFG.h_soma_um,
          "distances from the dipole centre in [%g, %.1f] um" % (CFG.h_soma_um, rmax))


QSTAT = """#!/bin/bash
echo "Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time"
echo "--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----"
[ -n "$MOCK_QUEUED" ] && echo "1900001.davinci user     cpu      $MOCK_QUEUED          --   1  48    --  24:00 Q   -- "
exit 0
"""
QSUB = """#!/bin/bash
echo "$*" >> "$MOCK_QSUB_LOG"
echo "$((1800000 + $(wc -l < "$MOCK_QSUB_LOG"))).mock"
"""


def _exe(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_launcher(tmp):
    print("3  jobs/launch_campaign.sh against a mock qsub / qstat")
    repo = os.path.join(tmp, "repo")
    os.makedirs(os.path.join(repo, "jobs"))
    for f in ("config.py", "culture_export.py", "field.py", "bump_kinetics.py"):
        shutil.copy(os.path.join(HERE, f), repo)
    for f in ("launch_campaign.sh", "submit_seeded.sh", "results_layout.sh", "parallel.pbs"):
        shutil.copy(os.path.join(HERE, "jobs", f), os.path.join(repo, "jobs"))
    cfg = os.path.join(repo, "config.py")
    src = open(cfg).read().replace('    cell_model: str = "soma_only"',
                                   '    cell_model: str = "full_tuned"', 1)
    open(cfg, "w").write(src)
    mock = os.path.join(tmp, "bin")
    os.makedirs(mock)
    _exe(os.path.join(mock, "qstat"), QSTAT)
    _exe(os.path.join(mock, "qsub"), QSUB)
    os.symlink(sys.executable, os.path.join(mock, "python"))
    qlog = os.path.join(tmp, "qsub.txt")
    # WALLTIME 72 h here so that the test does not depend on the culture size chosen in config
    env = dict(os.environ, PATH=mock + os.pathsep + os.environ["PATH"], MOCK_QSUB_LOG=qlog,
               PYTHONDONTWRITEBYTECODE="1", USER="user", WALLTIME="72:00:00")
    for k in ("PREFIX", "FIRST", "DRY", "FORCE", "SEED0", "MOCK_QUEUED", "EXPECT_MODEL",
              "TARGET_NEURONS", "CULT_PER_JOB", "CORES", "SEED_STEP", "FLUSH_EVERY", "QUEUE"):
        env.pop(k, None)
    N = CFG.neurons_per_culture
    J = int(np.ceil(2000000 / (48.0 * N)))

    def run(**extra):
        if os.path.exists(qlog):
            os.remove(qlog)
        r = subprocess.run(["bash", "jobs/launch_campaign.sh"], cwd=repo, capture_output=True,
                           text=True, env=dict(env, **{k: str(v) for k, v in extra.items()}))
        calls = open(qlog).read().splitlines() if os.path.exists(qlog) else []
        return r, calls

    r, calls = run(DRY=1)
    check(r.returncode == 0 and not calls and ("%d jobs x 48 cultures x %d neurons" % (J, N))
          in r.stdout, "dry run: plan of %d jobs x 48 x %d, nothing submitted" % (J, N))
    r, calls = run()
    ok = (r.returncode == 0 and len(calls) == J and "walltime=72:00:00" in calls[0]
          and "-q cpu" in calls[0] and "nodes=1:ppn=48" in calls[0]
          and "JOB_SEED=50000,JOB_NAME=ftd01,NCULT=48,NPROC=48" in calls[0]
          and "FLUSH_EVERY=10" in calls[0]
          and ("JOB_SEED=%d,JOB_NAME=ftd%02d" % (50000 + (J - 1) * 1000, J)) in calls[-1])
    check(ok, "submits ftd01..ftd%02d, seeds 50000..%d, the walltime given, queue cpu, 48 "
          "cores, flush 10" % (J, 50000 + (J - 1) * 1000))
    logf = os.path.join(repo, "results_full_tuned", "campaign_ftd.log")
    check(os.path.exists(logf) and open(logf).read().count("(seed=") == J,
          "plan and the %d submissions appended to results_full_tuned/campaign_ftd.log" % J)
    r, calls = run(MOCK_QUEUED="ftd01")
    check(r.returncode != 0 and not calls and "not submitting twice" in (r.stdout + r.stderr),
          "a second run while ftd01 is queued is refused")
    r, calls = run(MOCK_QUEUED="ftd01", FIRST=J)
    check(r.returncode == 0 and len(calls) == 1 and ("JOB_NAME=ftd%02d" % J) in calls[0],
          "FIRST=%d resumes with the last job only" % J)
    old = os.path.join(repo, "results_full_tuned", "parts_old")
    os.makedirs(old)
    from culture_export import CSV_HEADER
    row = [""] * len(CSV_HEADER)
    row[CSV_HEADER.index("seed")], row[CSV_HEADER.index("culture")] = "50007", "0"
    with open(os.path.join(old, "part_000.csv"), "w") as fh:
        fh.write(",".join(CSV_HEADER) + "\n" + ",".join(row) + "\n")
    with open(os.path.join(old, "tasks.txt"), "w") as fh:
        fh.write("0,1,2\t%s/part_000.csv\n" % old)
    r, calls = run(DRY=1)
    check(r.returncode != 0 and "repeat earlier draws" in r.stdout,
          "seeds whose cultures an earlier run already drew are refused")
    shutil.rmtree(old)
    r, calls = run(DRY=1, WALLTIME="00:30:00")
    check(r.returncode != 0 and "exceeds WALLTIME" in r.stdout,
          "a walltime shorter than the predicted worst case is refused")
    r, calls = run(DRY=1, WALLTIME="00:30:00", FORCE=1)
    check(r.returncode == 0, "... unless FORCE=1")
    r, calls = run(DRY=1, EXPECT_MODEL="soma_only")
    check(r.returncode != 0 and not calls, "a config of another model is refused")


def test_ascii():
    print("4  pure ASCII, LF only")
    files = ["config.py", "rich_footprint.py", "culture_export.py", "culture_worker.py",
             "plot_vm_examples.py", "culture_vm_animation.py", "video_frames.py",
             "smoke_test_protocol.py", "smoke_test_campaign_setup.py",
             "smoke_test_campaign_examples.py", "smoke_video_frames.py",
             "jobs/launch_campaign.sh", "jobs/full_tuned_run.pbs", "jobs/paper_figure.pbs",
             "jobs/vm_examples.pbs"]
    bad = []
    for f in files:
        b = open(os.path.join(HERE, f), "rb").read()
        if b"\r" in b or any(x > 127 for x in b):
            bad.append(f)
    check(not bad, "%d files%s" % (len(files), (" -- not clean: %s" % bad) if bad else ""))


def main():
    tmp = tempfile.mkdtemp(prefix="smoke_campaign_setup_")
    try:
        test_config()
        test_placement()
        test_launcher(tmp)
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
