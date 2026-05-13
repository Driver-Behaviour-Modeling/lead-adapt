# Source from a bash shell with `source scripts/activate_lead.sh` after `cd`-ing
# into the repo root, or `source /home/divyanshu/carla/lead-adapt/scripts/activate_lead.sh`
# from anywhere. Activates the `lead` conda env and exports the paths LEAD needs.

conda activate lead
export LEAD_PROJECT_ROOT=/home/divyanshu/carla/lead-adapt
source "${LEAD_PROJECT_ROOT}/scripts/main.sh"
export PYTHONPATH="${LEAD_PROJECT_ROOT}/3rd_party/CARLA_0915/PythonAPI/carla:${PYTHONPATH}"
cd "${LEAD_PROJECT_ROOT}"
