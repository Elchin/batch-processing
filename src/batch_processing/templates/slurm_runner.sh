#!/bin/bash -l

#SBATCH --job-name="$job_name"

#SBATCH -p $partition

#SBATCH -o $log_file_path.out

#SBATCH -e $log_file_path.err

#SBATCH -N 1

set -euo pipefail

BATCH_DIR="$batch_dir"
CONFIG="$batch_dir/config/config.js"
BINARY="$dvmdostem_binary"

{
  echo "=== $$(date -Is) job=$${SLURM_JOB_ID:-?} node=$${SLURM_NODELIST:-$$(hostname)} ==="
  echo "batch_dir=$${BATCH_DIR}"
  echo "config=$${CONFIG}"
  echo "binary=$${BINARY}"
} >&2

for f in "$$CONFIG" "$$BINARY"; do
  [[ -e "$$f" ]] || { echo "ERROR: missing required file: $$f" >&2; exit 1; }
done

ulimit -s unlimited
ulimit -l unlimited

source /etc/profile.d/z00_lmod.sh
module purge
module use /mnt/exacloud/lustre/modulefiles

module load openmpi/v4.1.x
module load dvmdostem-deps/2026-02

# Suppress PMIx compression library warning (optional, cosmetic)
export PMIX_MCA_pcompress_base_silence_warning=1

# Lustre: disable HDF5 file locking (incompatible with Lustre without flock)
export HDF5_USE_FILE_LOCKING=FALSE

# Pin the parallel (MPI-enabled) NetCDF the dvmdostem binary was linked against.
# The binary uses RUNPATH, which is searched AFTER LD_LIBRARY_PATH; the deps
# module puts a non-parallel NetCDF on LD_LIBRARY_PATH. Prepending the parallel
# build guarantees nc_create_par/nc_open_par work (avoids the
# "Parallel operation on file opened for non-parallel access" crash).
export LD_LIBRARY_PATH=/mnt/exacloud/cchang_woodwellclimate_org/software/netcdf-c/4.9.2/lib:$${LD_LIBRARY_PATH:-}

# OpenMPI 4.1.x: use ROMIO instead of buggy OMPIO for NetCDF/HDF5 parallel I/O.
# Default: --use-hwthread-cpus. Pass --mpi-ranks N to wiemip_split/split for mpirun -n N.
mpirun $mpirun_rank_flags \
  -x HDF5_USE_FILE_LOCKING -x PMIX_MCA_pcompress_base_silence_warning \
  --mca io ^ompio \
  "$$BINARY" -f "$$CONFIG" -l $log_level $flags_before_max_output \
  --max-output-volume=-1 $additional_flags -p $p -e $e -s $s -t $t -n $n \
  || { echo "ERROR: mpirun failed (exit $$?)" >&2; exit 1; }
