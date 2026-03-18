#!/bin/bash -l

#SBATCH --job-name="wetland-batch0"
# Optional default partition; override with: sbatch -p compute|dask|spot ...
#SBATCH -p compute
#SBATCH -o /mnt/exacloud/ext_bmaglio_alaska_edu/testing_io/batch_0/wetland-%x-%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --exclusive

set -euo pipefail

ulimit -s unlimited
ulimit -l unlimited

source /etc/profile.d/z00_lmod.sh
module purge
module use /mnt/exacloud/lustre/modulefiles
module load openmpi/v4.1.x
module load dvmdostem-deps/2026-02

export PMIX_MCA_pcompress_base_silence_warning=1
export HDF5_USE_FILE_LOCKING=FALSE

# One MPI rank per available CPU on this node.
CPUS_AVAILABLE="${SLURM_CPUS_ON_NODE:-}"
if [[ "$CPUS_AVAILABLE" =~ ^([0-9]+)\(x[0-9]+\)$ ]]; then
  CPUS_AVAILABLE="${BASH_REMATCH[1]}"
fi
if [[ -z "$CPUS_AVAILABLE" ]]; then
  CPUS_AVAILABLE="$(nproc)"
fi

MPI_RANKS="${MPI_RANKS:-$CPUS_AVAILABLE}"

# Keep ranks single-threaded by default.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMPI_MCA_rmaps_base_oversubscribe=1

BASE_CONFIG="${BASE_CONFIG:-/mnt/exacloud/ext_bmaglio_alaska_edu/testing_io/batch_0/config/config.js}"
DVM_BIN="${DVM_BIN:-/home/ext_bmaglio_alaska_edu/dvm-dos-tem/dvmdostem}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-/mnt/exacloud/ext_bmaglio_alaska_edu/testing_io/batch_0/output_wetland_${SLURM_JOB_PARTITION:-local}_${SLURM_JOB_ID:-manual}}"
RUNTIME_CONFIG="/tmp/config_wetland_${SLURM_JOB_ID:-manual}_$$.js"

mkdir -p "${RUN_OUTPUT_DIR}"

command -v python3 >/dev/null 2>&1
python3 - "$BASE_CONFIG" "$RUNTIME_CONFIG" "$RUN_OUTPUT_DIR" <<'PY'
import json
import sys

base_cfg, runtime_cfg, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
if not out_dir.endswith("/"):
    out_dir = out_dir + "/"

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["IO"]["output_dir"] = out_dir

with open(runtime_cfg, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=4)
    f.write("\n")
PY
test -s "${RUNTIME_CONFIG}"
trap 'rm -f "${RUNTIME_CONFIG}"' EXIT

echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Node: $(hostname)"
echo "MPI_RANKS=${MPI_RANKS}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "DVM_BIN=${DVM_BIN}"
echo "BASE_CONFIG=${BASE_CONFIG}"
echo "RUN_OUTPUT_DIR=${RUN_OUTPUT_DIR}"

mpirun -np "${MPI_RANKS}" \
  --oversubscribe --use-hwthread-cpus \
  --bind-to core --map-by core \
  --mca io ^ompio \
  -x HDF5_USE_FILE_LOCKING \
  -x PMIX_MCA_pcompress_base_silence_warning \
  -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND \
  -x OMPI_MCA_rmaps_base_oversubscribe \
  "${DVM_BIN}" \
    -f "${RUNTIME_CONFIG}" \
    -l disabled --max-output-volume=-1 \
    -p 100 -e 2000 -s 200 -t 124 -n 76
