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

Use the example provided in the `README` as the template for processing each dataset.  
Each WIEMIP climate model should be processed using the same workflow.

You will repeat this for each model, for example:

- `UKESM`
- `GFDL`
- `IPSL`

### Step 3. Reduce file size if needed

After processing, the output file may be larger than necessary. If needed, reduce file size by masking unused grid cells and/or saving with compression. [I will add this section later]

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

---

## 4. Recommended workflow order

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

## 5. Example command sequence

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

## 6. Notes

- Use `UKESM` first unless there is a reason to prioritise another climate model.
- Keep original processed files with descriptive names.
- Use `cp` rather than `mv` when setting `historic-climate.nc`, so the original named file remains available.
- If file size becomes a problem, apply a mask and/or compression before uploading to the bucket.
- Always run a small test before starting a full simulation.

---

## 7. Suggested folder naming convention

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
