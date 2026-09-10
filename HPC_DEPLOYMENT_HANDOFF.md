# HPC deployment handoff -- estim_pipeline parallelization
*For picking this up in a new chat. Covers deployment/ops only -- the biophysics,
morphology, and stimulation-protocol decisions are unchanged and already documented in
`HANDOFF.md`, `MODULES.md`, `ESTIM_HPC_HANDOFF.md`, `note_post_handoff.md`,
`REPORT_estim_EN.md` in this project. Read those for the science; read this for "why
is the cluster doing X" and "what do I run next."*

| date | change |
|---|---|
| 2026-09-04 | Initial handoff, written mid-campaign: two batches of jobs exist (one running with a known bug, one about to be submitted with the fix). See "Where things stand right now." |

---

## Where things stand right now

**Two job batches, do not confuse them:**

| batch | names | seeds | cultures/job | cores/job | walltime | code | status |
|---|---|---|---|---|---|---|---|
| A (first real submission) | `run1`..`run4` | 1000,2000,3000,4000 | 192 (=4/core x 48) | 48 | 120h | **old**, layer-major, no fsync | submitted, was still `R` at ~42h with zero flushed data (see "The two bugs" below); user was told it's expected to bank ~2 complete cultures/core (~384 total) by the 120h wall, and was given the choice to `qdel` or let it ride -- **last known: user chose to keep it running and rename the new batch instead, so batch A should still be in flight** |
| B (fix verified, not yet confirmed submitted) | `runB1`..`runB4` | 1000,2000,3000,4000 (same seeds as A, deliberately or not -- see open item) | 96 (=4/core x 24) | 24 (halved, see open item below) | 120h | **new**, neuron-major + fsync + `FLUSH_EVERY=25` | command was given to the user; **no confirmation yet that it was run** |

Exact command for batch B, given but unconfirmed:
```bash
WALLTIME=120:00:00 QUEUE=cpu FLUSH_EVERY=25 \
  bash jobs/submit_seeded.sh 96 24 pbs runB1:1000 runB2:2000 runB3:3000 runB4:4000
```

**First thing to do in the new chat:** ask the user for `qstat -u $USER` to see which of A/B are actually running, and `ls -la parts_*/part_*.csv | awk '{print $5, $NF}' | sort | uniq -c` (or similar) to see whether anything has flushed non-zero yet.

---

## Cluster environment (verified in this conversation)

| item | value | how verified |
|---|---|---|
| repo path | `/davinci-1/home/ldellamea/MATILDE THESIS/estim_pipeline/` | used throughout, space in path handled everywhere with quoting |
| conda env | `neuron_env` | user ran `culture_worker.py` interactively in it successfully; `(base)` and `(hdmea_env)` are OTHER envs seen in the user's prompt at various points, unrelated to this pipeline, do not use them |
| scheduler | PBS/Torque | `qsub`/`qstat`/`qdel` all used successfully |
| queue | `cpu` | user confirmed directly |
| compute node(s) seen | `dvnode008` (batch A), `dvnode003` (interactive testing) | from `qstat -f` / prompt |
| morphology archive | `eyal_archive/` with `specimen_60303, 60308, 60311, 130303, 130305, 130306` + `manifest.json` + `comparison_targets.csv` | `ls` output, confirmed 6 distinct specimens |
| `python culture_worker.py ...` outside the env | fails `ModuleNotFoundError: No module named 'morphio'` | this is what killed the FIRST real 4-job submission before batch A (192 workers died in ~2s each, leaving 0-byte files) -- root cause was never fully nailed down (see open items) but activating `neuron_env` fixed it for every subsequent interactive test |

---

## config.py -- current confirmed values

Verified against a full paste of the file (not from memory). If it's changed since, trust the cluster file, not this table.

| field | value | note |
|---|---|---|
| `morphologies` | `["60308","130303","60303","60311","130305","130306"]` | **6 distinct** -- was `["60308","130303","60303","130303","130305","130306"]` (130303 twice) until this conversation caught it; a duplicate silently double-weights that specimen in the pooled P(r) |
| `area_half_um` | `180` | measured footprint, **never change** (file's own comment agrees) |
| `density_per_mm2` | `1700.0` | |
| `network_half_um` | `500` | -> `n_neurons_hpc()` = 1700 x (2x500/1000)^2 = **1700 neurons/culture** |
| `use_hpc_count` | `True` | |
| `n_cultures` | `384` | **stale / cosmetic mismatch**: every actual job submission has passed an explicit culture count on the command line (192 for batch A, 96 for batch B), which always overrides this field (`run_parallel.sh`: `if [ "${2:-}" != "" ]; then NCULT="$2"; else ...config.n_cultures...`). A `sed` to set this to 192 was suggested but **never confirmed applied**. Harmless as long as jobs are always launched via `submit_seeded.sh`/`submit_batches.sh` with an explicit count; would silently do 384 if anyone ever runs `run_parallel.sh` or `culture_export.py` with no override. |
| `seed` | `0` | base seed; every job overrides it per-job via `JOB_SEED` -> `--seed`, so this base value is not actually used by any current job |
| `culture_distance_mode` | `"dipole"` | |
| `stim_duration_s` | `80.0` | -> `n_pulses_for_duration()` = round(80*0.2) = **16 pulses**. Above the report's stated >=60s/12-pulse minimum, below the lab's 180s/36-pulse protocol. Confirm this is deliberate, not a leftover. |
| `i0_uA` | `50.0` | |
| `layers_um` | `(40.0, 80.0, 120.0)` | |

---

## Files added/changed for parallelization (all in `estim_pipeline/`)

None of this exists in a persistent sandbox -- the cluster copy is the only authoritative copy. Everything below was delivered as incremental tar.gz patches over several rounds; assume the cluster has all of it unless a specific check below says otherwise.

| file | role | status |
|---|---|---|
| `culture_worker.py` | Simulates an assigned subset of cultures (`--ids`), one process = one core. Builds its 9-per-morphology-set cells once, then loops. **Just patched**: neuron-major loop + `os.fsync()` + `--flush-every N` (default 25) -- see "The two bugs," below. | patch given this session, **not yet confirmed deployed/run** |
| `culture_export.py` | Serial single-process path (also where `CSV_HEADER`, `render_culture_figures()`, and the geometry helpers live -- `culture_worker.py` imports from here). Used for the very first successful dry run (542s/60 sims, 3 morphologies) before the parallel path existed. | stable, not touched by the flush fix |
| `culture_merge.py` | Concatenates every job's `parts_*/part_*.csv`, using `(seed, local_culture)` as the true identity (not `culture` alone -- see design note below), remaps to a fresh global `culture` id, writes `local_culture`+`seed` for traceability, renders figures via `culture_export.render_culture_figures`. **Just patched**: the truncated-culture warning is now a NOTE (partial cultures are expected and, since the flush fix, verified layer-balanced) rather than an alarm. | patch given this session |
| `jobs/run_parallel.sh` | Fans out `culture_worker.py` across N cores via `xargs -P`, then merges. Reads `CULTURE_OFFSET` (disjoint-range splitting), `JOB_SEED`/`JOB_NAME` (replicate splitting), `FLUSH_EVERY`. Names its `parts_<tag>/` dir and output CSV after `JOB_NAME` if given, else `off<N>[_seed<M>]`. | patch given this session (added `FLUSH_EVERY` passthrough) |
| `jobs/submit_batches.sh` | Submits N jobs with disjoint `CULTURE_OFFSET` blocks (one shared seed) -- for scaling up ONE run's statistics. | stable, **not used in this campaign** (campaign uses the seeded path instead) |
| `jobs/submit_seeded.sh` | Submits named, seed-differentiated replicate jobs. Validates no seed or name is reused across the call. Supports `WALLTIME=`/`QUEUE=` env overrides -> `qsub -l walltime=... -q ...`, and now `-l nodes=1:ppn=${CORES}` so the cores-per-job argument actually shapes the allocation. | this is the path both batch A and batch B used/use |
| `jobs/merge_all.sh` | Globs `parts_*` (covers both offset- and seed-named dirs) and calls `culture_merge.py`. | stable |
| `jobs/env_setup.sh` | Sourced by every job script: activates `neuron_env` (with `set +u`/`set -u` bracketing the conda calls -- conda's own `geotiff-deactivate.sh` hook references an unset variable and dies under this script's `set -u` otherwise), checks NEURON imports, checks `x86_64/NaTa_t.o` exists (catches the bare-`nrnivmodl` silent-failure trap), checks `eyal_archive/` exists. | stable, fixed earlier this campaign |
| `jobs/parallel.pbs` / `.slurm` | The actual PBS/SLURM job script `submit_seeded.sh` calls. Self-locates the repo across 6 candidate paths (works whether `qsub` was run from the repo root or from `jobs/`). **Previously had inline `# comment` text after `#PBS -l ...` directives, which PBS parses as part of the resource value and rejects** -- fixed; every directive line is now clean. | stable since that fix |
| `jobs/dryrun.pbs` / `.slurm`, `jobs/culture_export.pbs` / `.slurm` | Earlier, single-job / serial-path templates from before the parallel infrastructure existed. `dryrun.*` is still useful for a quick end-to-end sanity check; the `culture_export.*` pair is effectively **superseded** by the seeded/parallel path for real campaigns. | legacy, not used by batches A/B |
| `smoke_test_culture_parallel.py` | Offline (no NEURON) test of id-parsing, worker/culture splitting, RNG-equivalence, merge validation, and the (seed,culture) identity logic including the pseudoreplication guard. 28 checks, all passing as of the last patch. | run this after applying any patch, before trusting a real submission |

---

## Design note: why `(seed, culture)` is the identity, not `culture` alone

Two ways to get more total cultures without recomputing the same ones:

- **Offset split** (`submit_batches.sh`): one shared seed, each job owns a disjoint block of the global culture index (`CULTURE_OFFSET`). Non-overlap is automatic because the index ranges never intersect.
- **Seed split** (`submit_seeded.sh`, what this campaign uses): every job runs the SAME local range `0..N-1`, but its own seed. "Culture 0" of `run1` and "culture 0" of `run2` are genuinely independent random draws, not the same computation twice.

Both can produce rows where the raw `culture` column repeats across files. `culture_merge.py` keys uniqueness (and the "is this a real duplicate" check) on the **`(seed, culture)` pair**, then remaps every unique pair to a fresh sequential global id before handing the array to `render_culture_figures()`. This was verified with an actual 4-seed, shared-local-range merge test: correct rejection of a genuine same-seed-same-culture duplicate, correct acceptance and correct-into-distinct-global-ids remapping of same-culture-different-seed rows.

---

## The two bugs found in batch A (both fixed for batch B, neither retroactively fixes batch A's already-running process)

**Bug 1 -- layer-major flush order (fixed, verified).** The simulation loop was `for layer: for neuron:`, and `fh.flush()` fired once per fully-completed culture. Two consequences: (a) nothing at all is visible until an entire culture (5,100 sims at N=1700) finishes, and (b) -- the one that actually matters -- if a job is killed mid-culture, that in-progress culture contributes *zero* rows (the old code never partial-flushes), so it's not a bias risk, just a "how much did we lose" question. Fixed to neuron-major (`for neuron: for layer:`) with a flush+`os.fsync()` every `--flush-every` neurons (default 25); verified by construction that the row *set* is identical either way, and that a truncated file under the new order is layer-balanced (tested 10/10/10 split) where the old order would have been drastically skewed (tested 5/1/0).

**Bug 2 -- `flush()` without `fsync()` on a networked filesystem.** `flush()` only pushes Python's buffer into the writing node's OS page cache; without `fsync()`, another node (e.g. the login node running `ls`) can see stale (zero) size even after data is technically written. Confirmed directly: `ssh dvnode008` and checking the file *from the compute node itself* also showed 0 bytes at the 42h mark -- ruling out "it's just a caching artifact" and confirming the stall was real. `os.fsync(fh.fileno())` added alongside every flush.

**What was hypothesized but NOT confirmed: why batch A was ~2.25x slower than the solo measurement.** `qstat -f` showed `cpupercent=4798%` (near the 4800% theoretical max for 48 logical CPUs) -- all cores busy -- yet 42h produced zero completed cultures, implying real per-sim cost >=~29.6s versus the ~13.19s/sim measured on an otherwise-idle node. The working hypothesis offered to the user was CPU hyperthreading (48 logical / 24 physical cores, so `cpupercent` looks perfect while each physical core is actually shared by two memory-bound NEURON processes) -- **the `lscpu` command to confirm this was given to the user but the output was never returned**. Batch B was sized at 24 workers/job on that hypothesis. If `lscpu` turns out to show 48 *physical* cores, the real cause is something else (most likely memory-bandwidth contention among 48 simultaneous NEURON processes, which would look similar) and is still worth checking directly.

---

## Timing numbers actually measured (for sizing future walltime requests)

All from real runs in this conversation, not estimates:

| run | morphologies | neurons | sims | wall time | context |
|---|---|---|---|---|---|
| very first dry run | 3 | 20 | 60 | 542 s | `culture_export.py` serial path, pre-dates the duplicate-morphology fix |
| `--neurons 2` test | 6 (post-fix) | 2 | 6 | 54.9 s | `culture_worker.py`, single process, idle node |
| `--neurons 20` test | 6 | 20 | 60 | 767.1 s | same conditions as above |

From the last two: **p (per-sim, solo/idle) ~= 13.19 s**, with cell-build cost small and noisy relative to it (solving the two-point system gives an unphysical negative build cost, meaning build time is genuinely small compared to measurement noise -- trust p, don't trust the build-cost split). This number is **not** validated under 48-way (or 24-way) concurrency -- batch A's evidence suggests real contention pushes it well above 13.19s; batch B (24 workers) is the test of whether halving concurrency recovers something close to the solo number.

At p=13.19s solo, 4 cultures/core x 1700 neurons x 3 layers = 20,400 sims/core = ~74.7h -- comfortably inside 120h *if* the solo number holds under 24-way concurrency. If it doesn't fully recover, first flush for batch B should appear noticeably before batch A's 42h+ silence; that comparison is itself the next data point.

---

## Immediate next steps for the new chat

1. Get current `qstat -u $USER` and file listings for both batches -- don't assume anything above is still true; treat "last known" as exactly that.
2. If batch B was never submitted, confirm the user still wants it, then submit with the command in "Where things stand right now."
3. Once batch B has been running an hour or so, check `parts_runB1/part_000.csv` size -- if it's already non-zero (it will be, with `FLUSH_EVERY=25` on a 1700-neuron culture, roughly every 25/1700 ~= 1.5% of a culture's worth of wall time), that alone confirms the fsync fix works and the job is alive, without waiting anywhere near 12+ hours to know.
4. Get the `lscpu` output that was never returned -- it resolves the one still-open causal question from batch A.
5. When both batches eventually finish (or are stopped): `bash jobs/merge_all.sh`, which will pick up every `parts_*` directory (both `run*` and `runB*`) via its `parts_*` glob into one `culture_Pactivation.csv`. Watch its printed NOTE about partial cultures -- it will explicitly say whether any are layer-unbalanced (would only be possible from a batch-B job killed mid-flush-interval, not from batch A per Bug 1's analysis above) and should report "every partial culture is layer-balanced. Safe to include." if all is well.
