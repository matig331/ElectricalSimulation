# HPC run guide -- preliminary full-active analysis and final soma-only campaign

This replaces the earlier HPC_RUN_COMPLETE.md. Three instructions in that version were wrong for
the cluster and are corrected here:

| earlier guide | problem | correct |
|---|---|---|
| `nrnivmodl` (bare) | compiles ZERO channels, still prints "Successfully created" | `nrnivmodl rich_mech` |
| `source ~/nrnenv/bin/activate` | laptop venv; the cluster env is conda | `conda activate neuron_env` (jobs do it via `jobs/env_setup.sh`) |
| `python culture_export.py` as the HPC run | SERIAL, one core: ~700+ core-hours in one process | `bash jobs/submit_seeded.sh ...` (parallel workers) |

## What changed in the code (summary)

- **One model switch**: `config.cell_model = "soma_only"` (default) or `"full_active"`. Written
  in every CSV row (`cell_model`). Soma = full Rich set; dendrites and stylized axon/AIS passive.
- **The HPC path now runs the new model.** `culture_worker.py` and the serial `culture_export()`
  share ONE core (`CellPool`, `iter_culture_blocks`, `build_row`), so they cannot diverge again
  (the zip changed only the serial path; the workers would have kept simulating full-active).
- **One live NEURON cell per process** (`CellPool`). NEURON integrates every section in memory;
  the old worker kept all 6 morphologies x 3 layers alive, making every simulation several times
  slower for identical results. Neurons are processed in blocks of 25, rows written neuron by
  neuron and fsync'ed after each block: a walltime kill loses at most one block per worker.
- **DeltaV_end is referenced to a sham run**, not to the scalar rest:
  `DeltaV_end = Vm(end of phase 2) - Vm_sham(end of phase 2)`, where the sham is the same
  simulation with zero current. Reason: `finitialize(V_rest)` flattens the non-uniform
  equilibrium (passive axon/dendrites sit near e_pas), so the unstimulated soma drifts by about
  +0.09 mV before the end of phase 2. Referenced to the scalar rest, that drift -- not the
  stimulus -- decided the sign of every weak far-field response (measured: at 450-600 um the old
  rule called 4/4 orientations "depol", 2/4 were depolarized). With the sham: I = 0 gives exactly
  0, the response is odd in I and linear in |I|. The drift is stored (`ctrl_drift_mV`), so the old
  quantity is recoverable: `Vm - V_rest = deltaVm_end_phase2_mV + ctrl_drift_mV`.
- **Three deliverables per merge**: `culture_Pactivation.csv`, `culture_Pdepolarization.csv`,
  `culture_Phyperpolarization.csv` (each: identity/provenance columns + ONE 0/1 outcome column).
- **Data safety**: each model writes to its own folder `results_<model>/`; a job refuses to reuse
  a name whose parts already hold data (unless `OVERWRITE=1`), refuses to run if
  `config.cell_model` changed while it was queued, and logs to `logs/<model>_<name>.log`.
- **Statistics guards**: `culture_statistics.py` refuses to pool cell models and refuses rows that
  appear twice (same model, seed, culture, neuron, layer); `--dedupe-identical` keeps one copy
  when the copies are identical (batch A and batch B used the same seeds 1000-4000).
- `pipeline.py` compiles again (the zip had a stray character); all sources are ASCII + LF.

---

## 0. Deploy

Check nothing from the old code is still running in that directory (a running `run_parallel.sh`
re-reads its own file; replacing it mid-job breaks the job's final merge step):

```bash
qstat -u $USER            # no R/Q jobs of batch A/B left
```

Either route (the tarball contains only changed/new files, paths relative to the repo root):

```bash
cd "/davinci-1/home/ldellamea/MATILDE THESIS/estim_pipeline"
tar -xzf ~/estim_soma_only_hpc.tar.gz        # route 1: overlay the changed files
# or, if this directory is a git clone:
git apply --check ~/estim_soma_only_hpc.patch && git apply ~/estim_soma_only_hpc.patch
```

## 1. Environment and mechanisms (once per checkout)

```bash
conda activate neuron_env
nrnivmodl rich_mech                 # NOT bare nrnivmodl
ls x86_64/NaTa_t.o                  # must exist
python -m pip install pandas        # culture_statistics.py / culture_vm_animation.py
```

## 2. Verify the delivery on the cluster (before any qsub)

```bash
grep -lr $'\r' --include=*.py --include=*.sh --include=*.pbs --include=*.slurm . ; echo "CRLF: ^ must be empty"
python - <<'PY'
import glob
bad = [f for f in glob.glob("*.py") + glob.glob("jobs/*") if any(b > 127 for b in open(f, "rb").read())]
print("non-ASCII:", bad or "none")
PY
python -m py_compile *.py && echo "compile OK"
python smoke_test_culture_export.py     | tail -1    # All smoke tests passed.
python smoke_test_culture_parallel.py   | tail -1    # All smoke tests passed.
python smoke_test_culture_statistics.py | tail -1    # All smoke tests passed.
```

NEURON end to end (interactive session on a compute node, a few minutes):

```bash
python smoke_test_soma_only.py | tail -25
```

Pass: last line `All smoke tests passed.` It also prints `measured cost: X s per simulation
(single process)` -- use it for the walltime (section 4). Checks include: soma-only has channels
only on the soma; I = 0 gives DeltaV = 0 exactly; far-field response odd and linear; a soma on an
electrode fires; one live cell; rebuilt cell bit-identical; worker == serial path; merge ->
statistics counts exact.

## 3. Part A -- preliminary statistics on the existing full-active data

Never pool these with soma-only data (the statistics script refuses anyway). Batch A (`run1-4`)
and batch B (`runB1-4`) used the SAME seeds, so their overlapping cultures are the same draws
simulated twice.

**If the old per-job merged CSVs exist** (`culture_Pactivation_run1.csv`, ...):

```bash
for t in run1 run2 run3 run4 runB1 runB2 runB3 runB4; do
  [ -f culture_Pactivation_$t.csv ] && mkdir -p results_full_active/$t && \
    cp culture_Pactivation_$t.csv results_full_active/$t/culture_Pactivation.csv
done
```

**If only the raw parts exist** (a job hit the walltime before its merge), merge each batch
separately (merging both at once is refused: same seed + same culture = duplicate):

```bash
python culture_merge.py --parts "parts_run[0-9]*" --out-dir results_full_active/batchA --no-figures --expect-model full_active
python culture_merge.py --parts "parts_runB*"     --out-dir results_full_active/batchB --no-figures --expect-model full_active
```

Then (use only ONE of the two layouts above in `results_full_active/`):

```bash
python culture_statistics.py --input results_full_active --output stats_full_active \
    --outcome activation --distance-bin 5 --angle-bin 5 --orientation-bin 5 --xy-bin 10 \
    --dedupe-identical
```

Pass: `--dedupe-identical: dropped N exact duplicate observation(s)` and the PDF under
`stats_full_active/activation/`. If it says the duplicates DISAGREE, the two batches are not the
same simulation (different code/morphology list): analyse one batch folder at a time.

## 4. Part B -- the soma-only campaign

Config check (expected: `soma_only 1700 500.0 180.0 (40.0, 80.0, 120.0)`):

```bash
python -c "from config import CFG; print(CFG.cell_model, CFG.n_neurons_effective(), CFG.span_half_um(), CFG.stim_duration_s, tuple(CFG.layers_um))"
```

Walltime -- two numbers, not one. `jobs/dryrun.pbs`/`smoke_test_soma_only.py` measure the SOLO
per-simulation cost (one process, otherwise-idle node). They do NOT measure what happens when
`cores_per_job` processes run at once on the same node, which is a separate number:

  known history: the earlier full-active campaign measured 13.19 s/sim solo, assumed it would
  hold at 48 workers/node, and got real cost >= 29.6 s/sim once actual rows were counted after
  42 h -- a >=2.25x slowdown, cause never confirmed (candidate: 48 logical vs 24 physical cores
  on that node, i.e. hyperthreading; `lscpu` to tell them apart was never run). Soma-only is a
  different, lighter workload (no active dendrites/axon) -- do not assume its concurrent scaling
  matches full-active's either way; measure it.

Before sizing a multi-day submission:
```bash
lscpu | grep -E "^CPU\(s\)|Thread|Core|Socket"      # physical cores = CPU(s) / Threads per core
```
If `cores_per_job` exceeds the physical core count, halve it or expect the concurrent number below
to be markedly worse than solo.
```bash
# concurrent per-sim cost at the REAL cores_per_job, on one node, a few minutes (NEURONS_PER_CULTURE
# is a TIMING-ONLY override -- 20 neurons/culture like dryrun.pbs, NOT the biological 1700 -- so
# this exercises real multi-process contention without taking hours):
NEURONS_PER_CULTURE=20 JOB_SEED=999999 JOB_NAME=concurrency_check \
    bash jobs/run_parallel.sh <cores_per_job> <cores_per_job>
# -> "wall time: E s on <cores_per_job> core(s)" for <cores_per_job> x 20 x 3 sims total;
#    concurrent s/sim = E / (cores_per_job x 20 x 3) -- compare it to the solo s/sim from section 2
#    (from dryrun.pbs / smoke_test_soma_only.py) before trusting the walltime formula below
rm -rf results_soma_only/parts_concurrency_check results_soma_only/concurrency_check   # throwaway
```
Walltime: rows per core = cultures_per_job x 1700 x 3 / cores_per_job; walltime ~= rows per core
x (CONCURRENT s/sim just measured, not the solo number) + 20% margin. The block size bounds the
loss if it is too short.

Submit with NEW names (old parts stay untouched in the repo root / results_full_active). Reusing
seeds 1000-4000 gives the SAME placements as the full-active runs -> a paired comparison:

```bash
WALLTIME=72:00:00 bash jobs/submit_seeded.sh <cultures_per_job> <cores_per_job> pbs \
    runS1:1000 runS2:2000 runS3:3000 runS4:4000
```

Monitor:

```bash
qstat -u $USER
tail -f logs/soma_only_runS1.log                       # progress + s/sim per block
wc -l results_soma_only/parts_runS1/part_*.csv          # grows every block (flushed + fsync'ed)
```

Guards you may see (all intended, nothing is deleted):

| message | meaning / fix |
|---|---|
| `FATAL: results_<model>/parts_<name>/ already holds data` (submit) / `already holds simulated data` (job) | name reused: pick a new name, or `OVERWRITE=1` to really redo it |
| `submitted for cell_model=X but config.py now says Y` | config edited while the job queued: fix config or resubmit |
| `FATAL: seed 'S' given twice` / `name 'N' given twice` | two jobs with one seed would duplicate draws: give each a distinct name and seed |

## 5. Part C -- merge and final statistics

```bash
bash jobs/merge_all.sh                     # -> merged_soma_only/culture_P*.csv + quick-look PDFs
python culture_statistics.py --input merged_soma_only --output stats_soma_only \
    --outcome all --distance-bin 5 --angle-bin 5 --orientation-bin 5 --xy-bin 10
```

Point `--input` at ONE layer of outputs: `merged_soma_only` (all jobs) OR `results_soma_only`
(per-job files) -- both at once is refused as duplicates. Check the sample-count heatmaps before
reading the fine-binned probability maps.

Optional movie (interactive/laptop; needs ffmpeg, else use a .gif name):

```bash
python culture_vm_animation.py --input merged_soma_only --output culture_vm_animation.mp4
```

### All figures as ONE batch job (merge once, then everything in parallel)

```bash
qsub jobs/analysis.pbs                                   # merge + all figures
qsub -v MERGED=analysis/<previous_run>/merged jobs/analysis.pbs   # reuse a merge (faster)
```

Options (comma-separated after `-v`): `PARTS`, `OUT`, `MERGED`, `DEDUPE=1`, `MOVIE=0`,
`DV_THRESHOLDS=0.5:1:2` (mV, colon-separated), `DV_MAIN=1` (movie labels), `DIST_BIN`,
`ANGLE_BIN`, `ORIENT_BIN`, `XY_BIN`.

**Depolarization / hyperpolarization are thresholded on |DeltaV_end|.** A non-firing soma counts
as depolarized only if DeltaV_end >= +T and hyperpolarized only if DeltaV_end <= -T. The CSVs keep
the raw sign-only labels and the continuous DeltaV, so T is applied at analysis time, with no new
simulation. Sign-only labels give P ~= 0.5 everywhere beyond ~150-200 um: there every soma gets a
micro-volt response whose sign is set only by which way the cell points along the field, and
orientations are random. Those responses are real but negligible; the threshold keeps only
responses of physiological size. The job produces T = 0.5, 1 and 2 mV side by side (sensitivity
analysis); page 6 of each polarization PDF shows |DeltaV_end| vs distance with T marked, and the
sign-only vs thresholded P(distance). `--dv-threshold 0` reproduces the sign-only labels.

Output: `stats/activation/`, `stats/dv_<T>mV/{depolarization,hyperpolarization}/`, `maps/`,
`maps/dv_<T>mV/`, `vm_movie_dv<T>mV.*`, one log per task in `logs/`. 15 tasks -> ppn=16 (more
cores would sit idle). Memory: ~7 statistics processes on 3.7 M rows at once, several GB each;
disk: each statistics folder writes a merged CSV of ~1 GB.

## Git: push (laptop) and pull (cluster)

Laptop, in your clone of github.com/matig331/ElectricalSimulation (the tarball carries whole
files, so this works whatever state `main` is in):

```bash
cd ~/path/to/ElectricalSimulation
git pull
tar -xzf ~/Downloads/estim_soma_only_hpc.tar.gz
git status                                    # only the shipped files may appear
git add $(tar -tzf ~/Downloads/estim_soma_only_hpc.tar.gz)
git commit -m "soma-only HPC pipeline; |DeltaV| threshold analysis; analysis job"
git push
```

Cluster, in the pipeline folder (data folders are git-ignored and untouched):

```bash
cd "/davinci-1/home/ldellamea/MATILDE THESIS/estim_pipeline"
qstat -u $USER                                # no job may be running from this folder
git rev-parse --is-inside-work-tree 2>/dev/null && echo "git clone" || echo "not a git clone"
# git clone:
git status --short                            # modified files = ones you overlaid from tarballs
git checkout -- $(git diff --name-only)       # drop those overlays (the commit carries them)
git pull
# not a git clone: overlay the same tarball instead
tar -xzf ~/estim_soma_only_hpc.tar.gz
```

## Output layout

```text
results_soma_only/parts_<name>/part_NNN.csv   raw worker rows (all outcomes per row)
results_soma_only/<name>/culture_P*.csv       per-job merge (if the job finished)
merged_soma_only/culture_P*.csv + PDFs        all jobs (jobs/merge_all.sh)
stats_soma_only/{activation,depolarization,hyperpolarization}/
logs/soma_only_<name>.log                     one log per job
```

## Interpretation rules

- Never pool models. Files are tagged `full_active`, `soma_only`, or -- for CSVs produced by the
  earlier zip's serial export (scalar-rest reference) -- `soma_only_v0_scalar_rest`. All three are
  analysed separately; mixing is refused.
- The statistical unit in `culture_statistics.py` is the neuron observation (Wilson 95% CI);
  cultures are the replicate units of the sampling design.
- Declare in the methods: soma-only active model (passive dendrites and passive stylized
  axon/AIS -- this reverses the earlier choice of an active AIS); DeltaV_end referenced to the
  sham run; one pulse representative of the 0.2 Hz train.
