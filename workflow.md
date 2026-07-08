## WIEMIP workflow for processing climate input data and running TEM simulations

This document describes the workflow for preparing WIEMIP climate inputs and running end-to-end TEM simulations. The WIEMIP input data are stored in:

```bash
gs://wiemip/1pctCO2/input
```

The available climate models are:

- `GFDL`
- `IPSL`
- `UKESM`

Start with `UKESM`, since it is the current priority.

---

## 1. Process the WIEMIP climate input data

### Step 1. Clone the input-conversion repository

```bash
git clone https://github.com/whrc/wiemip_tem_input_conversion.git
cd wiemip_tem_input_conversion
```

### Step 2. Follow the example in the repository README

Use the example provided in the [README](https://github.com/whrc/wiemip_tem_input_conversion) as the template for processing each dataset.  
Each WIEMIP climate model should be processed using the same workflow.

You will repeat this for each model, for example:

- `UKESM`
- `GFDL`
- `IPSL`

### Step 3. Reduce file size if needed

After processing, the historic-climate file is about 7.1Gb. Reduce file size by masking unused grid cells. For example, for the UKESM model:
```bash
python masking/apply_run_mask_to_climate.py path-to/setup_05deg_updated/run-mask.nc tem_UKES_output/historic-climate-UKESM1-0-LL.nc tem_UKES_output/historic-climate-UKESM1-0-LL-masked.nc
```

### Step 4. Give the processed file a meaningful name

Use filenames that clearly identify the climate model. For example:

```bash
historic-climate-UKESM.nc
historic-climate-GFDL-ESM4.nc
historic-climate-IPSL-CM6A-LR.nc
```

### Step 5. Save the processed climate file to the setup bucket

Upload the final file to:

```bash
gs://wiemip/setup_05deg_updated
```

Example:

```bash
gsutil cp historic-climate-UKESM.nc gs://wiemip/setup_05deg_updated/
```

---

## 2. Prepare the WIEMIP simulation environment

### Step 1. Clone the batch-processing repository

```bash
git clone https://github.com/Elchin/batch-processing.git
cd batch-processing
```

### Step 2. Switch to the `wiemip` branch

```bash
git checkout wiemip
```

### Step 3. Move to your WIEMIP working directory

```bash
cd /mnt/<yourname>_woodwellclimate_org/wiemip
```

Example:

```bash
cd /mnt/yourname_woodwellclimate_org/wiemip
```

### Step 4. Copy the setup files from the bucket

```bash
gsutil -m cp -r gs://wiemip/setup_05deg_updated .
```

This copies the updated setup directory into your local WIEMIP workspace.

### Step 5. Select the climate forcing file

The workflow expects the chosen climate file to be named:

```bash
historic-climate.nc
```

For example, if you want to run the model using `historic-climate-GFDL-ESM4.nc`, rename or copy it as follows:

```bash
cd /mnt/<yourname>_woodwellclimate_org/wiemip/setup_05deg_updated
cp historic-climate-GFDL-ESM4.nc historic-climate.nc
```

Using `cp` is safer because it keeps the original file.  
If you prefer to rename it directly:

```bash
mv historic-climate-GFDL-ESM4.nc historic-climate.nc
```

---

## 3. Run the end-to-end WIEMIP simulation workflow

### Step 1. Read the README in `batch-processing`

Before running the workflow, make sure you are on the correct branch and read the `README`.

```bash
cd ~/batch-processing
git branch
```

You should see:

```bash
* wiemip
```

### Step 2. Run a short test first on Dask nodes

Before launching a full run, do a quick end-to-end test.

Example:

```bash
python ~/batch-processing/src/batch_processing/extra/wiemip_end_to_end.py \
  --input /mnt/exacloud/yourname_woodwellclimate_org/wiemip/setup_GFDL-ESM4 \
  --split /mnt/exacloud/yourname_woodwellclimate_org/wiemip/test_gfdl_split_3 \
  -sp dask \
  -p 10 -e 10 -s 10 -t 10
```

### Step 3. Meaning of the key arguments

- `--input`  
  Path to the WIEMIP setup directory containing the selected climate forcing and required inputs.

- `--split`  
  Path for the split workspace or test split output.

- `-sp dask`  
  Run the test using Dask workers.

- `-p 10`  
  Number of years for the prerun phase.

- `-e 10`  
  Number of years for equilibrium.

- `-s 10`  
  Number of years for spinup.

- `-t 10`  
  Number of years for the transient or test phase.

For a quick test, these small values are sufficient. For production runs, use the values recommended in the workflow documentation.

### Restart from prior split

Use this when you want a **new split directory** and **fresh run-masks** from a WIEMIP setup (with CMT filters), while **reusing restart NetCDFs** from a completed split.

**Requirements**

- `--input` must be a full WIEMIP setup directory (top-level `run-mask.nc`, `vegetation.nc`, climate files). Do not use a prior split’s `_wiemip_filtered_input` folder.
- `--restart_from` must be the root of a prior split (contains `batch_0`, `batch_1`, …).
- Batch count in `--restart_from` must match the active-row count implied by `--input` `run-mask.nc` (the script fails if they differ).

**What comes from where**

| Artifact | Source |
|----------|--------|
| `batch_N/input/run-mask.nc`, vegetation, climate | `wiemip_split` from `--input` (+ `--max-cmt`, `--cmt0-filter`, climate prefilter) |
| `batch_N/output/restart-sp.nc` (or `--restart_file`) | Copied from `--restart_from/batch_N/output/` |
| `config.js` (except `IO.restart_from`) | New batch workdir from dvm-dos-tem + `update_config` |
| `IO.restart_from` in `config.js` | Patched to new `batch_N/output/<restart_file>` path |

**Example**

```bash
python ~/batch-processing/src/batch_processing/extra/wiemip_end_to_end.py \
  --input /mnt/exacloud/$USER/wiemip/setup_stable \
  --split /mnt/exacloud/$USER/wiemip/stable_split_veg1_restart \
  --restart_from /mnt/exacloud/$USER/wiemip/stable_split_veg1 \
  --restart_file restart-sp.nc \
  --max-cmt 1 \
  --cmt0-filter \
  -sp dask \
  -p 0 -e 0 -s 0 -t 20
```

**Anti-patterns**

- `--input .../stable_split_veg1/_wiemip_filtered_input` — reuses old staging masks; CMT flags may appear to do nothing.
- `--restart_from` set but omitting `--max-cmt` when you expect strict CMT — default is `--max-cmt 74`, which disables few cells.
- Expecting `run-mask.nc` to be copied from the source split — Option 2 does not do that; only restart files are copied.

Dry-run validation (paths and batch-count check, no Slurm):

```bash
python ~/batch-processing/src/batch_processing/extra/wiemip_end_to_end.py \
  --dry-run \
  --input /mnt/exacloud/$USER/wiemip/setup_stable \
  --split /mnt/exacloud/$USER/wiemip/stable_split_veg1_restart \
  --restart_from /mnt/exacloud/$USER/wiemip/stable_split_veg1 \
  --max-cmt 1 --cmt0-filter
```

---

## 4. Standard production run (rect split + local scratch)

This is the recommended way to run a full WIEMIP simulation. It uses:

- **rect split** (`--split-mode rect`) — 2D active-cell blocks balanced by cell count, so every batch has a similar amount of work (better than legacy latitude stripes).
- **local scratch** (`--localscratch`) — each batch writes model output to the compute node's local disk during the run and stages it back to shared storage when the job exits. This avoids the shared-filesystem (NFS) I/O contention that slows every job when many batches run at once.
- **CMT filters** (`--cmt0-filter --max-cmt 74`) — skip cells that should not be simulated so they don't waste compute or fail.

You can run this either **manually** (Option A, one command per stage, full control) or **end-to-end** (Option B, a single command that chains every stage automatically). Both use the same flags below.

### 4.1 Flags used in this workflow

**Core flags (used in every command below):**

| Flag | Meaning |
|------|---------|
| `-i/--input` | WIEMIP setup directory (top-level `run-mask.nc`, `vegetation.nc`, `historic-climate.nc`, …). |
| `-b/--split` | Output directory for the split batches (also where results are merged). |
| `--split-mode rect` | Use 2D active-cell blocks (recommended). |
| `--cells-per-batch N` | Target active cells per batch. Controls how many batches you get. Use `bp batch suggest-split` to pick a good value. |
| `--min-cells-per-batch M` | Rect only: merge trailing tiny blocks so no batch is smaller than `M`. Use roughly `2 × --mpi-ranks` to avoid idle MPI ranks. |
| `--localscratch` | Write output to node-local disk, stage back on exit. |
| `--cmt0-filter` | Disable cells where `veg_class == 0`. |
| `--max-cmt 74` | Disable cells where `veg_class > 74` (permissive default; lower it for stricter CMT filtering). |
| `-sp/--slurm-partition compute` | Partition to submit to. Use `compute` for production (use `dask` only for short tests). |
| `--mpi-ranks 8` | MPI ranks per batch job. |
| `-p -e -s -t -n` | Pre-run / equilibrium / spin-up / transient / scenario years. |

**Optional flags you may add:**

| Flag | When to use |
|------|-------------|
| `--no-max-cmt` | Turn off the max-CMT filter entirely (keep all high `veg_class` cells). |
| `--no-runmask-prefilter` | Skip the climate-validity prefilter (normally leave it on). |
| `--nbatches N` | Fixed batch count instead of `--cells-per-batch` (split/manual only). |
| `--throttle` (+ `--max-concurrent`, `--max-queue-depth`) | Submit jobs gradually instead of all at once, to be nice to the scheduler on very large runs. |
| `--restart_from PATH` / `--restart_file NAME` | Continue from a prior split's restart files. See [Restart from prior split](#restart-from-prior-split). Requires `-p 0 -e 0` (and `-s 0` for a spin-up restart). |
| `--job-name-prefix NAME` | Prefix Slurm job names to keep runs distinguishable in `squeue`. |

### 4.2 Option A — Manual workflow (split → run → merge)

**Step 1 (optional): size your batches.** Get a suggested `--cells-per-batch` from the active-cell count in your setup:

```bash
bp batch suggest-split \
  -i /mnt/exacloud/$USER/wiemip/setup_UKESM \
  --target-batches 100 --mpi-ranks 8 \
  --cmt0-filter --max-cmt 74
```

**Step 2: split.** Create the batches with the recommended flags:

```bash
bp batch wiemip_split \
  -i /mnt/exacloud/$USER/wiemip/setup_UKESM \
  -b /mnt/exacloud/$USER/wiemip/UKESM_run \
  --split-mode rect --cells-per-batch 200 --min-cells-per-batch 16 \
  --localscratch \
  --cmt0-filter --max-cmt 74 \
  -sp compute --mpi-ranks 8 \
  -p 100 -e 2000 -s 200 -t 150 -n 0
```

This writes batches to `/mnt/exacloud/$USER/wiemip/UKESM_run/batch_0`, `batch_1`, … and generates a node-local-scratch `slurm_runner.sh` in each.

**Step 3: submit the jobs.**

```bash
bp batch run -b /mnt/exacloud/$USER/wiemip/UKESM_run
```

By default every job is submitted immediately and Slurm queues them. Add `--throttle` for very large runs (see 4.1).

**Step 4: monitor** until all jobs finish (see [4.4](#44-monitoring-jobs-under-local-scratch)).

**Step 5 (optional): recover incomplete batches.** If some cells did not finish (spot preemption, transient failures), re-run just the incomplete cells of a batch and merge the retry back in:

```bash
# Re-run only the incomplete cells of one batch (use compute for mpi-ranks > 4)
bp batch wiemip_re-run /mnt/exacloud/$USER/wiemip/UKESM_run/batch_17 --partition compute

# After the retry finishes, merge the retry output back into that batch
bp batch wiemip_rerun_merge /mnt/exacloud/$USER/wiemip/UKESM_run/batch_17
```

**Step 6: merge all batches** into full-grid outputs:

```bash
bp batch wiemip_merge -b /mnt/exacloud/$USER/wiemip/UKESM_run
```

This produces two folders under the split directory:

- `wiemip_merged/merged_filtered` — batches concatenated in filtered (cropped) space.
- `wiemip_merged/merged_restored` — restored to the full original grid (for example `360×720`). This is the final product.

### 4.3 Option B — End-to-end workflow (one command)

`wiemip_end_to_end.py` runs the whole pipeline for you: **split → submit → wait → up to two automatic rerun passes for incomplete batches → merge → plot**. Same flags as the manual workflow:

```bash
python ~/batch-processing/src/batch_processing/extra/wiemip_end_to_end.py \
  --input /mnt/exacloud/$USER/wiemip/setup_UKESM \
  --split /mnt/exacloud/$USER/wiemip/UKESM_run \
  --split-mode rect --cells-per-batch 200 --min-cells-per-batch 16 \
  --localscratch \
  --cmt0-filter --max-cmt 74 \
  -sp compute --mpi-ranks 8 --rerun-partition compute \
  -p 100 -e 2000 -s 200 -t 150
```

Useful extras for the end-to-end script:

- `--dry-run` — print every command and the batch-count check without submitting anything. Always do this first for a new setup.
- `--skip-split` — reuse an existing split directory (recovery / re-submit only).
- `--throttle --max-concurrent 16 --max-queue-depth 32` — gradual submission on large runs.
- `--rerun-partition compute` — partition for the automatic recovery passes (avoid `dask` when `--mpi-ranks > 4`).

The script blocks until everything finishes, then writes the merged outputs to `<split>/wiemip_merged/merged_restored` and runs the plot script.

### 4.4 Monitoring jobs under local scratch

With `--localscratch` the model output does **not** appear in `batch_N/output/` on shared storage while the job is running — it lives on the compute node's local disk and is copied back only when the job exits (on success **or** failure, via an exit trap). Keep this in mind while monitoring:

- **Job state:** `squeue -u $USER` (jobs are named `<split-dir>-batch-<index>`). Add `--partition compute` to filter.
- **Live progress:** the small `run_status.nc` is synced back to `batch_N/output/run_status.nc` on shared storage about **every 5 minutes**, so you can track per-cell completion during the run even though the large output files are not there yet.
- **Logs:** stdout/stderr for each batch are written on shared storage at `<split>/logs/batch-<index>.out` and `<split>/logs/batch-<index>.err`. The local-scratch runner logs a `=== stage-back ... ===` line when it copies results back, and staging errors show up here.
- **Completion:** a batch is done when `run_status.nc` shows `100` for all active cells (`run-mask.nc == 1`). If a job disappears from `squeue` but the batch is not complete, re-run it (Step 5, or the end-to-end script does this automatically).

### 4.5 Things to be aware of

- **Free space on the node:** the local-scratch runner requires at least **15 GB** free on the node's scratch (`/tmp` / `$SLURM_TMPDIR`). If less is available, the job aborts immediately with a clear error — check the `.err` log.
- **Restarts need zero PR/EQ years:** dvmdostem asserts that pre-run and equilibrium years are `0` when a restart file is set. If you use `--restart_from`, set `-p 0 -e 0` (and `-s 0` when restarting from a spin-up file into the transient stage). `wiemip_split` prints a loud warning if you forget.
- **Restart layout must match:** rect and y-stripe splits (or different `--cells-per-batch`) are **not** batch-compatible for restart seeding. Only reuse restart files across splits produced with the same geometry and sizing.
- **`--max-cmt 74` is permissive:** it disables very few cells. Lower `N` (for example `--max-cmt 5`) for stricter CMT filtering, or `--no-max-cmt` to keep everything.
- **Partition choice:** use `compute` for production and for reruns with `--mpi-ranks > 4`; `dask` is fine only for short tests.
- **Merge assumes WIEMIP batches:** use `bp batch wiemip_merge` for batches created by `bp batch wiemip_split`. (`bp batch merge` is for the plain `bp batch split` layout and is not interchangeable.)

---

## 5. Recommended workflow order

1. Process `UKESM` first.
2. Save the processed file with a clear name, for example:

   ```bash
   historic-climate-UKESM.nc
   ```

3. Upload it to:

   ```bash
   gs://wiemip/setup_05deg_updated
   ```

4. Copy the updated setup files to your local WIEMIP workspace.
5. Copy the selected climate forcing file to `historic-climate.nc`.
6. Run a short Dask test.
7. If the test succeeds, proceed with the full end-to-end simulation.

---

## 6. Example command sequence

```bash
# Clone repositories
git clone https://github.com/whrc/wiemip_tem_input_conversion.git
git clone https://github.com/Elchin/batch-processing.git

# Switch to WIEMIP branch
cd batch-processing
git checkout wiemip

# Move to WIEMIP workspace
cd /mnt/yourname_woodwellclimate_org/wiemip

# Copy updated setup files
gsutil -m cp -r gs://wiemip/setup_05deg_updated .

# Choose the climate forcing file
cd setup_05deg_updated
cp historic-climate-UKESM.nc historic-climate.nc

# Run a short test on Dask
python ~/batch-processing/src/batch_processing/extra/wiemip_end_to_end.py \
  --input /mnt/exacloud/yourname_woodwellclimate_org/wiemip/setup_05deg_updated \
  --split /mnt/exacloud/yourname_woodwellclimate_org/wiemip/test_ukesm_split_3 \
  -sp dask \
  -p 10 -e 10 -s 10 -t 10
```

---

## 7. Notes

- Use `UKESM` first unless there is a reason to prioritise another climate model.
- Keep original processed files with descriptive names.
- Use `cp` rather than `mv` when setting `historic-climate.nc`, so the original named file remains available.
- If file size becomes a problem, apply a mask and/or compression before uploading to the bucket.
- Always run a small test before starting a full simulation.

---

## 8. Suggested folder naming convention

To keep runs organised, it helps to use a model-specific setup directory name, for example:

```bash
setup_UKESM
setup_GFDL-ESM4
setup_IPSL-CM6A-LR
```

and corresponding split directories such as:

```bash
test_ukesm_split_3
test_gfdl_split_3
test_ipsl_split_3
```

That makes it easier to track which forcing dataset was used for each run.
