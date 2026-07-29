# Source this in an srun/compute-node shell:  source scripts/srun_env.sh
#
# Why this exists:
#   On srun-allocated nodes, /home is a root symlink to the NFS-mounted
#   /localstorage/home, and it does not resolve until NSS maps your user
#   (the "I have no name!" / "cannot find name for group ID" window).
#   During that window any /home/<user>/... path fails, including the conda
#   binary that `conda init` hardcoded into ~/.bashrc. This script uses the
#   always-real /localstorage path and waits for the node to become ready,
#   so it works without touching ~/.bashrc.

# Absolute, NFS-real prefix (never goes through the /home symlink).
LEAD_LS_HOME=/localstorage/home/f20221129
LEAD_CONDA_ROOT="${LEAD_LS_HOME}/miniconda3"

# 1) Wait for NSS user resolution (fixes "I have no name!").
for _i in $(seq 1 30); do
    id -un >/dev/null 2>&1 && break
    sleep 1
done

# 2) Wait for the conda install on NFS to be reachable.
for _i in $(seq 1 30); do
    [ -x "${LEAD_CONDA_ROOT}/bin/conda" ] && break
    sleep 1
done

# 3) Activate conda. conda.sh has been patched to export /localstorage-rooted
#    CONDA_EXE/_CONDA_EXE/CONDA_PYTHON_EXE, so re-sourcing it here cleanly
#    overrides any stale /home-based values inherited from ~/.bashrc.
# shellcheck source=/dev/null
source "${LEAD_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate lead

# 4) Export the paths LEAD needs (mirrors ~/.bashrc but /localstorage-rooted).
export LEAD_PROJECT_ROOT="${LEAD_LS_HOME}/lead-adapt"
# shellcheck source=/dev/null
source "${LEAD_PROJECT_ROOT}/scripts/main.sh"

# 5) CARLA PythonAPI provides the `agents` module imported across lead/.
#    train.sh adds this too, but set it here so plain `python` works as well.
export PYTHONPATH="${LEAD_PROJECT_ROOT}/3rd_party/CARLA_0915/PythonAPI/carla:${PYTHONPATH}"

# 6) Point HOME and every cache var at writable /localstorage. On srun nodes
#    /home/<user> can be non-writable (the /home symlink + NSS race), so any
#    library that defaults to $HOME (matplotlib, huggingface, fontconfig, ...)
#    fails with EPERM. Setting HOME to the real /localstorage path makes all of
#    them derive writable defaults; the explicit vars below are belt-and-braces.
export HOME="${LEAD_LS_HOME}"
export XDG_CACHE_HOME="${HOME}/.cache"
export XDG_CONFIG_HOME="${HOME}/.config"
export MPLCONFIGDIR="${HOME}/.cache/matplotlib"
export HF_HOME="${HOME}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export FONTCONFIG_PATH="/etc/fonts"
export XDG_RUNTIME_DIR="${HOME}/.cache/runtime"
mkdir -p "${MPLCONFIGDIR}" "${HUGGINGFACE_HUB_CACHE}" \
         "${XDG_CACHE_HOME}/fontconfig" "${XDG_RUNTIME_DIR}" 2>/dev/null

unset _i LEAD_LS_HOME LEAD_CONDA_ROOT

echo "lead env ready: $(which python) (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV})"
