#!/bin/bash
# ---------------------------------------------------------------------------
# Shared environment setup, sourced by every job script in this folder.
# Edit CONDA_ENV here ONCE and every job picks it up.
# ---------------------------------------------------------------------------

CONDA_ENV="neuron_env"          # <-- EDIT here if the env is ever renamed

# 'conda activate' is a shell FUNCTION, not a binary. In a batch job the shell is
# non-interactive, so that function does not exist and 'conda activate' fails with
# "CommandNotFoundError: Your shell has not been properly configured".
# Sourcing the hook below defines it. This is the single most common reason an
# HPC job dies in its first second.
if ! command -v conda >/dev/null 2>&1; then
    echo "FATAL: conda not on PATH in the batch environment." >&2
    exit 1
fi
# Conda's own activate/deactivate hook scripts (e.g. geotiff-deactivate.sh from a
# geospatial package in 'base') reference internal variables without a default,
# e.g. _CONDA_SET_GEOTIFF_CSV. The calling job script runs 'set -euo pipefail'
# before sourcing this file, so nounset (-u) is already active here, and it kills
# the script the instant one of those hooks fires -- before any of our own checks
# even run. Relax -u for exactly the conda calls, nothing else.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
CONDA_OK=$?
set -u
if [ "$CONDA_OK" -ne 0 ]; then
    echo "FATAL: could not activate '$CONDA_ENV' (or locate conda.sh)." >&2
    exit 1
fi

# Headless plotting. The modules already call matplotlib.use("Agg"); this is a
# belt-and-braces guard so no import path can pick an interactive backend.
export MPLBACKEND=Agg

# Report the environment BEFORE doing any work, so a wrong env is visible on
# line 1 of the log rather than inferred from a crash 20 minutes in.
echo "host   : $(hostname)"
echo "date   : $(date)"
echo "repo   : $PWD"
echo "python : $(which python)"
python -c "import neuron; print('NEURON :', neuron.__version__)" || {
    echo "FATAL: NEURON not importable in '$CONDA_ENV'." >&2; exit 1; }

# The mechanisms must be compiled with 'nrnivmodl rich_mech'. A bare 'nrnivmodl'
# exits 0 and prints the same success message while compiling ZERO channels, so
# check for a real compiled object rather than just the output directory.
if [ ! -f x86_64/NaTa_t.o ] && [ ! -f arm64/NaTa_t.o ]; then
    echo "FATAL: rich_mech channels not compiled." >&2
    echo "       Run 'nrnivmodl rich_mech' in $PWD first." >&2
    exit 1
fi

# The morphology data is not part of the code bundle; fail loudly if it is absent.
if [ ! -d eyal_archive ]; then
    echo "FATAL: eyal_archive/ not found in $PWD." >&2
    exit 1
fi

echo "environment OK"
echo "---------------------------------------------------------------"
