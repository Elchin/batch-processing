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
MPI_RANK_FLAGS="$mpirun_rank_flags"

# Minimum free space (GB) required on node-local scratch for this batch.
# One batch's transient output for the full run is ~8 GB worst case; 15 GB
# leaves margin. The run aborts early if scratch is missing/tmpfs/too small.
MIN_SCRATCH_GB=15

# How often (seconds) to copy the small run_status.nc back to NFS so users
# can monitor progress during the run (Option B live-progress sync).
PROGRESS_INTERVAL=300

# --- Node-local scratch (boot disk), unique per job ------------------------
SCRATCH_BASE="$${SLURM_TMPDIR:-/tmp}"
LOCAL_RUN="$${SCRATCH_BASE}/wiemip_$${SLURM_JOB_ID:-$$(date +%s)}_$$(basename "$$BATCH_DIR")"
LOCAL_OUT="$${LOCAL_RUN}/output"
LOCAL_CONFIG="$${LOCAL_RUN}/config.js"
NFS_OUT="$${BATCH_DIR}/output"

mkdir -p "$$LOCAL_OUT" "$$NFS_OUT"

PROGRESS_PID=""

# Stage results back to NFS on ANY exit (success or failure), then clean up.
# This guarantees run_status.nc and partial output return so that
# wiemip_re-run / completion checks work even after a failed job.
stage_back() {
  local rc=$$?
  if [[ -n "$${PROGRESS_PID}" ]]; then
    kill "$${PROGRESS_PID}" 2>/dev/null || true
  fi
  echo "=== [stage-back] $$(date -Is) rc=$${rc}: $${LOCAL_OUT} -> $${NFS_OUT} ===" >&2
  if [[ -d "$$LOCAL_OUT" ]]; then
    shopt -s nullglob
    cp -f "$$LOCAL_OUT"/*.nc          "$$NFS_OUT"/ 2>/dev/null || true
    cp -f "$$LOCAL_OUT"/*.js          "$$NFS_OUT"/ 2>/dev/null || true
    cp -f "$$LOCAL_OUT"/fail_log.txt  "$$NFS_OUT"/ 2>/dev/null || true
    shopt -u nullglob
  fi
  rm -rf "$$LOCAL_RUN" || true
  echo "=== [stage-back] done ===" >&2
}
trap stage_back EXIT

# --- Guard: local scratch must be real disk with enough room ---------------
avail_gb=$$(df -BG --output=avail "$$SCRATCH_BASE" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -z "$$avail_gb" || "$$avail_gb" -lt "$$MIN_SCRATCH_GB" ]]; then
  echo "ERROR: local scratch $$SCRATCH_BASE has $${avail_gb:-?}GB free (< $${MIN_SCRATCH_GB}GB)" >&2
  exit 1
fi

{
  echo "=== $$(date -Is) job=$${SLURM_JOB_ID:-?} node=$${SLURM_NODELIST:-$$(hostname)} ==="
  echo "batch_dir=$${BATCH_DIR}"
  echo "local_scratch=$${LOCAL_RUN} (avail $${avail_gb}GB)"
  echo "config=$${CONFIG}"
  echo "binary=$${BINARY}"
  echo "mpi_rank_flags=$${MPI_RANK_FLAGS}"
} >&2

for f in "$$CONFIG" "$$BINARY"; do
  [[ -e "$$f" ]] || { echo "ERROR: missing required file: $$f" >&2; exit 1; }
done

# --- Build a run-local config: output_dir -> local scratch -----------------
# Inputs, parameters and output_spec stay on NFS (read-only, cached).
python3 - "$$CONFIG" "$$LOCAL_OUT" "$$LOCAL_CONFIG" <<'PY'
import json, sys
src, local_out, dst = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as fh:
    cfg = json.load(fh)
cfg["IO"]["output_dir"] = local_out.rstrip("/") + "/"
with open(dst, "w") as fh:
    json.dump(cfg, fh, indent=4)
PY

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

# Prepend the dvmdostem binary's own RUNPATH so its linked (MPI-parallel)
# NetCDF/HDF5 take precedence over any non-parallel copies the deps module puts
# on LD_LIBRARY_PATH. Portable: follows whatever the binary was built against
# (no per-user hardcoded paths), and avoids the
# "Parallel operation on file opened for non-parallel access" crash.
BIN_RUNPATH=$$(readelf -d "$$BINARY" 2>/dev/null | sed -n 's/.*(RUNPATH).*\[\(.*\)\]/\1/p; s/.*(RPATH).*\[\(.*\)\]/\1/p' | head -1)
if [[ -n "$$BIN_RUNPATH" ]]; then
  export LD_LIBRARY_PATH="$$BIN_RUNPATH:$${LD_LIBRARY_PATH:-}"
fi

# --- Option B: live-progress sync of the tiny run_status.nc to NFS ---------
# run_status.nc is ~10 KB; copying it every few minutes is negligible I/O and
# lets users monitor per-cell completion on NFS during the run.
( while true; do
    sleep "$$PROGRESS_INTERVAL"
    cp -f "$$LOCAL_OUT/run_status.nc" "$$NFS_OUT/run_status.nc" 2>/dev/null || true
  done ) &
PROGRESS_PID=$$!

# OpenMPI 4.1.x: use ROMIO instead of buggy OMPIO for NetCDF/HDF5 parallel I/O.
# The model writes to node-local scratch ($$LOCAL_CONFIG has output_dir on /tmp);
# the EXIT trap stages results back to NFS.
mpirun $mpirun_rank_flags \
  -x HDF5_USE_FILE_LOCKING -x PMIX_MCA_pcompress_base_silence_warning \
  --mca io ^ompio \
  "$$BINARY" -f "$$LOCAL_CONFIG" -l $log_level $flags_before_max_output \
  --max-output-volume=-1 $additional_flags -p $p -e $e -s $s -t $t -n $n \
  || { echo "ERROR: mpirun failed (exit $$?)" >&2; exit 1; }
