#!/bin/bash -l

#SBATCH --job-name="$job_name"

#SBATCH -p $partition

#SBATCH -o $log_file_path

#SBATCH -N 1

ulimit -s unlimited
ulimit -l unlimited

source /etc/profile.d/z00_lmod.sh
module purge
module use /mnt/exacloud/lustre/modulefiles
module avail

module load openmpi/v4.1.x
module load dvmdostem-deps/2026-02

# Suppress PMIx compression library warning (optional, cosmetic)
export PMIX_MCA_pcompress_base_silence_warning=1

# Lustre: disable HDF5 file locking (incompatible with Lustre without flock)
export HDF5_USE_FILE_LOCKING=FALSE

# One MPI rank per available CPU on this node.
CPUS_AVAILABLE="$${SLURM_CPUS_ON_NODE:-}"
if [[ "$$CPUS_AVAILABLE" =~ ^([0-9]+)\(x[0-9]+\)$$ ]]; then
  CPUS_AVAILABLE="$${BASH_REMATCH[1]}"
fi
if [[ -z "$$CPUS_AVAILABLE" ]]; then
  CPUS_AVAILABLE="$$(nproc)"
fi

MPI_RANKS="$${MPI_RANKS:-$mpi_ranks}"
MPI_RANKS="$${MPI_RANKS:-$$CPUS_AVAILABLE}"

# Keep ranks single-threaded by default.
export OMP_NUM_THREADS="$${OMP_NUM_THREADS:-1}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMPI_MCA_rmaps_base_oversubscribe=1

BASE_CONFIG="$${BASE_CONFIG:-$config_path}"
DVM_BIN="$${DVM_BIN:-$dvmdostem_binary}"
RUN_OUTPUT_DIR="$${RUN_OUTPUT_DIR:-$$(dirname $$(dirname $config_path))/output}"
RUNTIME_CONFIG="/tmp/config_$${SLURM_JOB_ID:-manual}_$$$$.js"

mkdir -p "$${RUN_OUTPUT_DIR}"

command -v python3 >/dev/null 2>&1
python3 - "$$BASE_CONFIG" "$$RUNTIME_CONFIG" "$$RUN_OUTPUT_DIR" <<'PY'
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
test -s "$${RUNTIME_CONFIG}"
trap 'rm -f "$${RUNTIME_CONFIG}"' EXIT

echo "Partition: $${SLURM_JOB_PARTITION:-unknown}"
echo "Node: $$(hostname)"
echo "MPI_RANKS=$$MPI_RANKS"
echo "OMP_NUM_THREADS=$$OMP_NUM_THREADS"
echo "DVM_BIN=$$DVM_BIN"
echo "BASE_CONFIG=$$BASE_CONFIG"
echo "RUN_OUTPUT_DIR=$$RUN_OUTPUT_DIR"

# OpenMPI 4.1.x: use ROMIO instead of buggy OMPIO for NetCDF/HDF5 parallel I/O
mpirun -np $$MPI_RANKS -x HDF5_USE_FILE_LOCKING -x PMIX_MCA_pcompress_base_silence_warning -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND --use-hwthread-cpus --mca io ^ompio $$DVM_BIN -f $$RUNTIME_CONFIG -l $log_level $flags_before_max_output --max-output-volume=-1 $additional_flags -p $p -e $e -s $s -t $t -n $n
