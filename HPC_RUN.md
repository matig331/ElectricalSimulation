> **Culture export (soma-only, three outcomes): see HPC_RUN_COMPLETE.md** -- it supersedes the
> culture_export / merge commands below (outputs now go to results_<model>/, merge takes
> --out-dir). The nrnivmodl and environment notes below still apply.

# HPC run guide -- in-silico extracellular stimulation pipeline

This guide explains, in order, **what to configure** and **which commands to run** on the cluster.
The main deliverable is the **activation-probability export** (`culture_export.py`), which produces the
CSV the point-neuron network consumes. Everything else is validation.

---

## 0. Environment (once)

```bash
# activate the NEURON virtual environment (numpy + matplotlib only; NO scipy is required)
source ~/nrnenv/bin/activate          # or your venv path

# compile the .mod mechanisms once, from the estim_pipeline/ folder
cd /path/to/estim_pipeline
nrnivmodl rich_mech                    # builds ./x86_64/ (or ./arm64/) HERE, from rich_mech/*.mod
```

Notes
- **The argument matters.** A bare `nrnivmodl` (no argument) compiles whatever `.mod` files sit
  directly in the current directory. Since the mod sources live in the `rich_mech/` subfolder, a
  bare `nrnivmodl` finds none, still prints `Successfully created x86_64/special`, and exits 0 --
  but the resulting library has zero of the 13 channels in it, so every later `insert("NaTa_t")`
  etc. fails. Always run `nrnivmodl rich_mech`, and verify with `ls x86_64/NaTa_t.o` (or
  `arm64/NaTa_t.o`) -- if that file is missing, the mechanisms were not actually compiled even
  though `nrnivmodl` reported success.
- The code uses **only numpy and matplotlib** -- do not install scipy.
- If you are on a login node with conda, make sure the `nrnenv` venv is the active Python
  (a stray conda base can shadow NEURON). `which python` should point inside `nrnenv`.

---

## 1. Configure `config.py`

You normally touch only a handful of fields. The two that matter most for the deliverable are the
**network size** and **how many cultures**.

### The neuron count is now automatic

The number of somata per culture is computed from **density x area** -- you no longer copy it by hand.
Set these:

```python
use_hpc_count  = True          # <-- turn ON the biological count (default False = local test)
density_per_mm2 = 1300          # your culture's cell density (change if yours differs)
network_half_um = 700.0         # HALF-width (um) of the well/network you actually simulate
                               #   None -> falls back to the measured footprint (area_half_um)
n_cultures     = 50            # number of independent random cultures = the statistics dial
```

With this, `n_neurons_effective()` = `density_per_mm2 x (2.network_half_um/1000)^2`.
Examples at 1300/mm^2: `network_half_um=500 -> ~1300`, `700 -> ~2548`, `1000 -> ~5200`.

### area_half_um is the MINIMUM (measured footprint), not the network

- `area_half_um` is the **measured activation footprint** (a lower bound), obtained from the `area`
  stage in Step 2. It is NOT the network size.
- `network_half_um` is the **well you simulate** (of order mm) and is what sets the neuron count.
- If `network_half_um` is left `None`, the count falls back to the footprint (`area_half_um`).

### Other fields (usually leave as-is)

```python
i0_uA           = 50.0         # lab amplitude (+/-50 uA)
phase_dur_ms    = 0.25         # 500 us total biphasic
stim_freq_hz    = 0.2          # 0.2 Hz
stim_duration_s = 180.0        # 3 min epoch (>= 60 s required for the statistic)
pitch_um        = 60.0         # electrode pitch
electrode_um    = 25.0         # electrode size -> rmin clamp
bipolar         = True         # 3 anodes / 3 cathodes (False = monopolar)
sigma_Sm        = 1.5          # medium/saline conductivity (S/m)
layers_um       = (40.0, 80.0, 120.0)   # the 3 slab thicknesses (3 sims per culture)
culture_distance_mode = "dipole"        # "dipole" (r,theta) | "nearest" | "centroid"
electrodes      = None          # put the 6 real (x, y, sign) positions here if you have them
```

- `culture_distance_mode`: use **`"nearest"`** if you want the clean monotonic recruitment sigmoid;
  **`"dipole"`** for the directional (r, theta) analysis; **`"centroid"`** only if electrodes are clustered.
  (All three distances are written to the CSV regardless, so you can re-analyse offline.)

---

## 2. Set the activation footprint (`area_half_um`)

`area_half_um` is the **measured activation footprint** (the MINIMUM area -- a lower bound, not the
network). It is a **number you set in `config.py`**.

**If you already measured it** (e.g. 180 um, measured with the 6-electrode array) -- just write it and
skip the measurement:

```python
# in config.py
area_half_um = 180.0
```

**If you have NOT measured it yet** -- measure it once, then copy the printed value in:

```bash
python pipeline.py area                 # samples out to area_scan_half_um (200 um) so it isn't clipped
# read the printed MEAN AREA_HALF and put it into config.area_half_um
python pipeline.py area_check           # optional: placement-box sanity check
```

Re-measure only if the electrode geometry changes (e.g. you switch `electrodes`/`pitch_um`); a footprint
measured with the current 6-electrode array stays valid.

---

## 3. Run the deliverable: `culture_export.py`

This is the main output -- the activation probability per soma, saved as data.

```bash
python culture_export.py
```

What it does (all read from `config.py`):
- draws `n_cultures` random **cultures**; each culture has `n_neurons_effective()` somata, all
  morphologies present, each soma a random position + random orientation;
- re-simulates the **same** placement at the 3 layer thicknesses (3 sims per culture);
- delivers the full biphasic pulse at `i0_uA`; records fire / no-fire per soma;
- total sims = `n_cultures x n_neurons_effective() x 3`.

Outputs (written next to the script):
- **`culture_Pactivation.csv`** -- the deliverable the network consumes. One row per
  (culture, neuron, layer) with: morphology, x, y, `dist_nearest_elec_um`, `dist_center_um`,
  `dist_dipole3d_um`, `theta_orient_deg`, `theta_pos_deg`, `n_pulses`, `i0_uA`, `fired`.
- **`culture_Pmap.pdf`** -- figures: (1) P vs distance from dipole centre (per-culture, mean+/-SD),
  (1b) recruitment sigmoid vs nearest-electrode distance (with logistic fit), (2) directional
  dependence by angle to the dipole axis, (3) activation area (one coloured disk per soma),
  (4) the angle explainer with the central dipole - -> +.

### Quick smoke run before the full job

Always test the wiring with a tiny run first (a couple of minutes):

```bash
python -c "import culture_export as ce; ce.culture_export(n_cultures=2, neurons_per_culture=30, span_um=200)"
```

You can also override any parameter at call time:

```bash
python -c "import culture_export as ce; ce.culture_export(n_cultures=50, span_um=700)"
```

### Offline sanity test (no NEURON)

```bash
python smoke_test_culture_export.py         # expect: 'All smoke tests passed.'
```

---

## 4. (Optional) Per-phase mechanism check

To see which phase of the biphasic pulse excites which side (cathodic alternation), deliver each phase
alone. This is a **mechanism** analysis, not the deliverable, and costs 2x the sims.

```bash
# instant, no NEURON: the illustrative waveform + flipping dipole
python -c "import culture_export as ce; ce.phase_diagram()"          # -> phase_diagram.pdf

# data version (runs NEURON): phase-1-only and phase-2-only maps
python -c "import culture_export as ce; ce.phase_split(n_cultures=8, neurons_per_culture=60, span_um=200)"
# -> culture_phase_maps.pdf  (phase 1: dipole - -> +, cathode = - ; phase 2: + -> -, cathode = +)
```

Caveat: phase-1-only + phase-2-only is **not** identical to the full biphasic pulse (phase 1
preconditions the membrane for phase 2). Use it for attribution, not to replace the full-pulse P.

---

## 5. (Optional) Full validation -- the field-dependent stages

Only needed if you want the complete validation set. These use the extracellular field, so with the
6-electrode array they must be (re)run. Increase `n_samples` in `config.py` for higher resolution.

```bash
python pipeline.py ve_field           # ve_field.pdf         (the 6-electrode field)
python pipeline.py checks             # field checks (1/r, superposition, charge)
python pipeline.py waveform           # waveform_check.pdf   (pulse centred + ramp zoom)
python pipeline.py bda                # rich_stim_<morph>.pdf (Ve/Vm windows on the pulse)
python pipeline.py well_active        # well_active.pdf      (Vm maps + spike-initiation site)
python pipeline.py subthreshold       # subthreshold.pdf
python pipeline.py channel_dynamics   # channel_dynamics.pdf
python pipeline.py axon_sensitivity   # axon_sensitivity.pdf
python pipeline.py recruitment        # recruitment.pdf      (validation marginals)
python pipeline.py check_polarization
python pipeline.py check_current_distance_k
python pipeline.py check_activating_function
```

Typing an unknown stage name prints the full list of available stages:

```bash
python pipeline.py                    # or:  python pipeline.py help
```

### What you do NOT need to re-run for the 6 electrodes

These use current injection (IClamp) or channel distributions only -- they do **not** depend on the
electrode array and stay valid:

```
intrinsic , baseline , channel_atlas ,
fig4 / ais_scan / somatic_k_scan / channel_transfer   (the depolarization-block investigation)
```

---

## 6. Minimal recipe (copy/paste)

Everything you choose lives in **`config.py`**. Edit these lines, then run the two commands.

```python
# ---- config.py : set YOUR values here ----
area_half_um    = 180.0     # your measured footprint (6-electrode) -- already known, no need to re-measure
use_hpc_count   = True      # turn on the biological neuron count
density_per_mm2 = 1300      # your culture's density
network_half_um = 700.0     # the well/network you simulate (um half-width); None -> use the footprint (180)
n_cultures      = 50        # statistics dial
culture_distance_mode = "dipole"   # or "nearest" for the clean sigmoid
```

```bash
source ~/nrnenv/bin/activate
cd /path/to/estim_pipeline
nrnivmodl rich_mech                         # once (compile the .mod channels -- argument required, see 0.)

python smoke_test_culture_export.py         # offline check -> 'All smoke tests passed.'
python -c "import culture_export as ce; ce.culture_export(n_cultures=2)"   # tiny test run first
python culture_export.py                    # the deliverable
```

Outputs: **`culture_Pactivation.csv`** (the file the network consumes) + `culture_Pmap.pdf`.

You did NOT need `python pipeline.py area` here because you already have the footprint (180). Run it only
if you ever need to re-measure it (new electrode geometry).

### Where each choice goes (summary)

| you want to change... | edit in `config.py` | effect |
|---|---|---|
| the well/network size | `network_half_um` | sets the neuron count (density x area) |
| cell density | `density_per_mm2` | sets the neuron count |
| the footprint (min area) | `area_half_um` | lower bound for sampling; = 180 for you |
| how much statistics | `n_cultures` | more cultures = tighter +/-SD (linear cost) |
| distance metric | `culture_distance_mode` | `"dipole"` / `"nearest"` / `"centroid"` |
| stimulus amplitude / timing | `i0_uA`, `phase_dur_ms`, `stim_freq_hz`, `stim_duration_s` | the pulse |
| real electrode positions | `electrodes = [(x,y,sign),...]` | overrides the generated 3+3 |

The count is then automatic: `n_neurons_effective()` = `density_per_mm2 x (2.network_half_um/1000)^2`
(or the footprint if `network_half_um` is None). You never copy the neuron number by hand.


---

## 7. Submitting as a batch job

Four templates live in `jobs/`, plus `jobs/env_setup.sh` which they all source
(it activates the conda env, checks NEURON imports, checks the mechanisms were
really compiled, and checks `eyal_archive/` exists). Edit the env name in
`env_setup.sh` once and every job picks it up.

**Always run the dry run first** -- it proves the pipeline works end to end and
prints the measured per-simulation cost, which is how the full job's walltime
gets sized from data instead of guessed:

```bash
qsub   jobs/dryrun.pbs        # PBS/Torque      \
sbatch jobs/dryrun.slurm      # SLURM            > 1 culture x 20 neurons = 60 sims
```

Read `dryrun.log`, put the printed walltime estimate into the full job's
`#PBS -l walltime=` (or `#SBATCH --time=`) line, then:

```bash
qsub   jobs/culture_export.pbs
sbatch jobs/culture_export.slurm
```

All four scripts self-locate the repo rather than trusting the working directory
(a scheduler runs them from a spool dir), and quote every path because the real
one contains a space.

### If the estimate exceeds the queue limit

Don't just shrink the science to fit -- **parallelize instead.** `jobs/run_parallel.sh`
splits cultures across cores (one culture per core minimum; each core builds its cells
once, then runs every culture assigned to it -- see the module docstrings in
`culture_worker.py` and `culture_merge.py` for why this granularity is the right one).

**Scaling up ONE run's statistics (more total cultures, same experiment):**
```bash
bash jobs/submit_batches.sh <n_jobs> <cultures_per_job> <cores_per_job> pbs
# e.g. 5 jobs x 20 cultures x 48 cores = 100 cultures total, no overlap
bash jobs/merge_all.sh          # once every job has finished
```
Each job gets a disjoint `CULTURE_OFFSET` automatically, so together they add up to
one bigger total instead of recomputing the same cultures.

**Several NAMED, independent replicate runs (e.g. "4 jobs, different name and seed"):**
```bash
bash jobs/submit_seeded.sh <cultures_per_job> <cores_per_job> pbs \
     run1:1000 run2:2000 run3:3000 run4:4000
bash jobs/merge_all.sh
```
Each job runs the SAME local culture range (0..N-1) but its OWN seed -- so "culture 0"
of `run1` and "culture 0" of `run2` are genuinely different random draws, not the same
one recomputed. Every row records which seed produced it; `culture_merge.py` uses the
(seed, culture) pair as the true identity, so it correctly tells a real duplicate
(same seed AND same culture, rejected) apart from a legitimate independent replicate
(different seed, same local index, allowed) -- and remaps every unique pair to its own
sequential id in the merged CSV, so the per-culture mean +/- SD never accidentally
pools two different replicates as if they were one.

Still too slow even parallelized? Then reduce the science:
1. **Lower `n_cultures`** (or `cultures_per_job x n_jobs`). Costs statistical precision
   directly (the +/-SD band across cultures widens), but 10-20 still gives a meaningful band.
2. **Lower `network_half_um`.** Fewer neurons per culture; most far-out ones sit outside
   the ~180 um footprint and contribute P ~= 0 anyway, so this trims mostly-uninformative
   samples -- though note dendrites can reach up to ~1275 um (the longest apical extent
   across the morphology set), so don't shrink this below what's needed to resolve where
   P(r) truly reaches zero.

### Cluster-side check before submitting

```bash
python -m py_compile *.py && echo "OK: everything compiles"
python smoke_test_culture_export.py     # expect: 'All smoke tests passed.'
```
