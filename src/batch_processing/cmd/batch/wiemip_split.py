from __future__ import annotations

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr

from batch_processing.cmd.batch.split import BatchSplitCommand
from batch_processing.utils.utils import create_chunks, interpret_path
from batch_processing.utils.wiemip_processing import (
    RUN_ENABLED_VALUE,
    RUN_MASK_VARIABLE,
    SPLIT_METADATA_FILENAME,
    ActiveBBox,
    WiemipSplitMetadata,
    compute_active_bbox,
    detect_spatial_dims,
    extract_run_mask_2d,
    filter_dataset_to_cropped_mask,
    open_dataset_for_read,
    write_split_metadata,
)

WIEMIP_NAME_ALIASES = {
    "historic_climate_GFDL-ESM4.nc": "historic-climate.nc",
    "historic-climate_GFDL-ESM4.nc": "historic-climate.nc",
    "historic_climate.nc": "historic-climate.nc",
}
MASKED_SUFFIX = "_masked.nc"
RUN_MASK_DESTINATION = "run-mask.nc"
OUTPUT_ROW_DIM = "Y"
OUTPUT_COL_DIM = "X"
OUTPUT_NETCDF_FORMAT = "NETCDF4_CLASSIC"


class WiemipSplitCommand(BatchSplitCommand):
    """Split WIEMIP inputs into internal-filtered cropped 2D Y-stripe batches."""

    def _sanitize_grid_mapping_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Remove misleading grid_mapping_name attrs from non-scalar data variables.

        dvmdostem scans for any variable containing 'grid_mapping_name' and then tries
        to copy that variable into output files using destination dim names (y/x).
        WIEMIP inputs often put this attr on regular spatial variables (e.g. veg_class
        with Y/X dims), which triggers a dim-name mismatch and NC_EBADDIM.
        """
        for var_name in ds.data_vars:
            da = ds[var_name]
            if da.ndim > 0 and "grid_mapping_name" in da.attrs:
                da.attrs.pop("grid_mapping_name", None)
        return ds

    def _resolve_destination_name(self, source_name: str) -> str:
        mapped_name = WIEMIP_NAME_ALIASES.get(source_name, source_name)
        if mapped_name.endswith(MASKED_SUFFIX):
            mapped_name = f"{mapped_name[: -len(MASKED_SUFFIX)]}.nc"
        return mapped_name

    def _discover_inputs(
        self, input_path: Path
    ) -> Tuple[List[Tuple[Path, str, str, str, int, int]], List[Tuple[Path, str]], Path]:
        netcdf_files = sorted(p for p in input_path.glob("*.nc") if p.is_file())
        if not netcdf_files:
            raise FileNotFoundError(f"No .nc files found under {input_path}")

        row_spatial_files: List[Tuple[Path, str, str, str, int, int]] = []
        copy_files: List[Tuple[Path, str]] = []
        destination_names = {}
        run_mask_source: Path | None = None

        for src in netcdf_files:
            dst_name = self._resolve_destination_name(src.name)
            if dst_name in destination_names:
                conflicting = destination_names[dst_name]
                raise ValueError(
                    f"Multiple source files map to {dst_name}: "
                    f"{conflicting.name} and {src.name}"
                )
            destination_names[dst_name] = src

            with open_dataset_for_read(src) as ds:
                spatial_dims = detect_spatial_dims(ds.dims)
                if spatial_dims is None:
                    copy_files.append((src, dst_name))
                    continue
                row_dim, col_dim = spatial_dims
                row_spatial_files.append(
                    (
                        src,
                        dst_name,
                        row_dim,
                        col_dim,
                        int(ds.sizes[row_dim]),
                        int(ds.sizes[col_dim]),
                    )
                )

            if dst_name == RUN_MASK_DESTINATION:
                run_mask_source = src

        if run_mask_source is None:
            raise FileNotFoundError(
                "Could not locate run-mask input. Expected run-mask.nc "
                "or an alias that resolves to run-mask.nc."
            )
        if not row_spatial_files:
            raise ValueError(
                "No spatial split candidates found. Expected files with recognized "
                "Y/X or latitude/longitude dimensions."
            )
        return row_spatial_files, copy_files, run_mask_source

    def _prepare_filtered_staging(
        self,
        original_input_path: Path,
        staging_path: Path,
        run_mask_da: xr.DataArray,
        run_row_dim: str,
        run_col_dim: str,
        bbox: ActiveBBox,
    ) -> dict[str, str]:
        staging_path.mkdir(parents=True, exist_ok=True)
        file_mappings: dict[str, str] = {}
        seen_destinations: set[str] = set()
        source_files = sorted(p for p in original_input_path.glob("*.nc") if p.is_file())
        print(
            f"[staging] Preparing {len(source_files)} NetCDF files from "
            f"{original_input_path} into {staging_path}"
        )

        for file_index, src in enumerate(source_files, start=1):
            dst_name = self._resolve_destination_name(src.name)
            if dst_name in seen_destinations:
                raise ValueError(
                    f"Multiple source files resolve to {dst_name} during staging."
                )
            seen_destinations.add(dst_name)
            file_mappings[src.name] = dst_name
            dst_path = staging_path / dst_name
            print(f"[staging {file_index}/{len(source_files)}] {src.name} -> {dst_name}")

            if dst_name == RUN_MASK_DESTINATION:
                print("  [copy] run-mask passthrough")
                shutil.copy2(src, dst_path)
                continue

            with open_dataset_for_read(src) as ds:
                spatial_dims = detect_spatial_dims(ds.dims)
                if spatial_dims is None:
                    print("  [copy] non-spatial file")
                    shutil.copy2(src, dst_path)
                    continue
                ds_row_dim, ds_col_dim = spatial_dims
                print(
                    f"  [filter] spatial dims {ds_row_dim}/{ds_col_dim}; "
                    "applying cropped active-mask filter"
                )
                filtered_ds = filter_dataset_to_cropped_mask(
                    in_ds=ds,
                    run_mask_da=run_mask_da,
                    run_row_dim=run_row_dim,
                    run_col_dim=run_col_dim,
                    ds_row_dim=ds_row_dim,
                    ds_col_dim=ds_col_dim,
                    bbox=bbox,
                    active_value=RUN_ENABLED_VALUE,
                )
                filtered_ds = self._sanitize_grid_mapping_attrs(filtered_ds)
                filtered_ds.to_netcdf(
                    dst_path.as_posix(),
                    engine="netcdf4",
                    format=OUTPUT_NETCDF_FORMAT,
                )
                filtered_ds.close()
                print(f"  [write] {dst_path}")
        return file_mappings

    def _split_spatial_file_to_y_stripes(
        self,
        src_file: Path,
        destination_name: str,
        row_dim: str,
        col_dim: str,
        is_full_grid: bool,
        bbox: Tuple[int, int, int, int],
        y_ranges: List[Tuple[int, int]],
        batch_input_dirs: List[Path],
    ) -> None:
        row_min, row_max, col_min, col_max = bbox
        print(f"Splitting {src_file.name} -> {destination_name} (Y stripes)")
        with open_dataset_for_read(src_file) as ds:
            if is_full_grid:
                base_ds = ds.isel(
                    {
                        row_dim: slice(row_min, row_max + 1),
                        col_dim: slice(col_min, col_max + 1),
                    }
                )
            else:
                base_ds = ds

            total_batches = len(batch_input_dirs)
            progress_every = max(1, total_batches // 10)
            for batch_index, ((start, end), batch_input_dir) in enumerate(
                zip(y_ranges, batch_input_dirs), start=1
            ):
                subset_ds = base_ds.isel({row_dim: slice(start, end)}).load()
                if destination_name == RUN_MASK_DESTINATION and RUN_MASK_VARIABLE in subset_ds:
                    run_values = np.asarray(subset_ds[RUN_MASK_VARIABLE].values)
                    normalized_values = np.where(
                        np.isfinite(run_values) & np.isclose(run_values, RUN_ENABLED_VALUE),
                        RUN_ENABLED_VALUE,
                        0,
                    ).astype(subset_ds[RUN_MASK_VARIABLE].dtype, copy=False)
                    subset_ds[RUN_MASK_VARIABLE] = xr.DataArray(
                        normalized_values,
                        dims=subset_ds[RUN_MASK_VARIABLE].dims,
                        coords=subset_ds[RUN_MASK_VARIABLE].coords,
                        attrs=subset_ds[RUN_MASK_VARIABLE].attrs,
                    )

                output_file = batch_input_dir / destination_name
                if row_dim != OUTPUT_ROW_DIM or col_dim != OUTPUT_COL_DIM:
                    rename_map = {}
                    if row_dim != OUTPUT_ROW_DIM:
                        rename_map[row_dim] = OUTPUT_ROW_DIM
                    if col_dim != OUTPUT_COL_DIM:
                        rename_map[col_dim] = OUTPUT_COL_DIM
                    subset_ds = subset_ds.rename(rename_map)
                subset_ds = self._sanitize_grid_mapping_attrs(subset_ds)
                subset_ds.to_netcdf(
                    output_file.as_posix(),
                    engine="netcdf4",
                    format=OUTPUT_NETCDF_FORMAT,
                )
                subset_ds.close()
                if (
                    batch_index == 1
                    or batch_index == total_batches
                    or batch_index % progress_every == 0
                ):
                    print(
                        f"  [split-progress] {destination_name}: "
                        f"batch {batch_index}/{total_batches} rows {start}:{end - 1}"
                    )

    def _validate_split_run_mask(
        self,
        run_mask_file: Path,
        expected_row_dim: str,
        expected_col_dim: str,
        expected_rows: int,
        expected_cols: int,
    ) -> None:
        with open_dataset_for_read(run_mask_file) as ds:
            if RUN_MASK_VARIABLE not in ds:
                raise KeyError(f"{run_mask_file} missing '{RUN_MASK_VARIABLE}' variable.")
            if expected_row_dim not in ds.dims or expected_col_dim not in ds.dims:
                raise ValueError(
                    f"{run_mask_file} must include '{expected_row_dim}' and "
                    f"'{expected_col_dim}' dimensions."
                )
            if int(ds.sizes[expected_row_dim]) != expected_rows:
                raise ValueError(
                    f"{run_mask_file} has {ds.sizes[expected_row_dim]} {expected_row_dim} rows, "
                    f"expected {expected_rows}."
                )
            if int(ds.sizes[expected_col_dim]) != expected_cols:
                raise ValueError(
                    f"{run_mask_file} has {ds.sizes[expected_col_dim]} {expected_col_dim} cols, "
                    f"expected {expected_cols}."
                )

    def execute(self) -> None:
        print("[wiemip_split] Starting integrated WIEMIP split workflow")
        if str(self.input_path).startswith("gcs://"):
            raise NotImplementedError(
                "bp batch wiemip_split currently supports local input paths only."
            )

        input_path = Path(interpret_path(self.input_path))
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        print(f"[wiemip_split] Input path: {input_path}")

        nbatches = int(getattr(self._args, "nbatches", 0))
        if nbatches < 1:
            raise ValueError("nbatches must be >= 1")
        print(f"[wiemip_split] Requested batch count: {nbatches}")

        print("[wiemip_split:1/8] Discovering run-mask source")
        _, _, run_mask_source = self._discover_inputs(input_path)
        print(f"[wiemip_split] Run-mask source: {run_mask_source}")
        print("[wiemip_split:2/8] Loading run-mask and computing active bbox")
        with open_dataset_for_read(run_mask_source) as run_mask_ds:
            run_mask_da, run_mask_row_dim, run_mask_col_dim = extract_run_mask_2d(
                run_mask_ds, run_mask_source.name, run_var=RUN_MASK_VARIABLE
            )
            full_rows = int(run_mask_da.sizes[run_mask_row_dim])
            full_cols = int(run_mask_da.sizes[run_mask_col_dim])

        bbox = compute_active_bbox(run_mask_da, active_value=RUN_ENABLED_VALUE)
        row_min, row_max = bbox.row_start, bbox.row_end
        col_min, col_max = bbox.col_start, bbox.col_end
        bbox_rows, bbox_cols = bbox.n_rows, bbox.n_cols
        if nbatches > bbox_rows:
            raise ValueError(
                f"nbatches ({nbatches}) cannot exceed cropped Y size ({bbox_rows})."
            )
        print(
            "[wiemip_split] Active bbox: "
            f"{run_mask_row_dim}[{row_min}:{row_max}], "
            f"{run_mask_col_dim}[{col_min}:{col_max}] -> {bbox_rows}x{bbox_cols}"
        )

        self.base_batch_dir.mkdir(exist_ok=True, parents=True)
        self.log_path.mkdir(exist_ok=True, parents=True)
        staging_path = self.base_batch_dir / "_wiemip_filtered_input"
        if staging_path.exists():
            shutil.rmtree(staging_path)
        print("[wiemip_split:3/8] Building internal filtered staging inputs")
        file_mappings = self._prepare_filtered_staging(
            original_input_path=input_path,
            staging_path=staging_path,
            run_mask_da=run_mask_da,
            run_row_dim=run_mask_row_dim,
            run_col_dim=run_mask_col_dim,
            bbox=bbox,
        )
        print(
            f"[wiemip_split] Filtered staging ready at {staging_path} "
            f"with {len(file_mappings)} mapped files"
        )
        print("[wiemip_split:4/8] Writing split metadata")
        metadata = WiemipSplitMetadata(
            schema_version=1,
            original_input_path=input_path.resolve().as_posix(),
            filtered_staging_path=staging_path.resolve().as_posix(),
            run_mask_filename=RUN_MASK_DESTINATION,
            row_dim=run_mask_row_dim,
            col_dim=run_mask_col_dim,
            active_value=RUN_ENABLED_VALUE,
            full_rows=full_rows,
            full_cols=full_cols,
            active_bbox={
                "row_start": row_min,
                "row_end": row_max,
                "col_start": col_min,
                "col_end": col_max,
            },
            file_mappings=file_mappings,
        )
        write_split_metadata(self.base_batch_dir / SPLIT_METADATA_FILENAME, metadata)
        print(f"Wrote split metadata: {self.base_batch_dir / SPLIT_METADATA_FILENAME}")

        print("[wiemip_split:5/8] Planning Y-stripe split from staged inputs")
        row_spatial_files, copy_files, _ = self._discover_inputs(staging_path)
        chunks = create_chunks(bbox_rows, nbatches)
        y_ranges = [(int(chunk.start), int(chunk.end)) for chunk in chunks]

        split_specs = []
        for src_file, destination_name, row_dim, col_dim, row_size, col_size in row_spatial_files:
            if row_size == full_rows and col_size == full_cols:
                is_full_grid = True
            elif row_size == bbox_rows and col_size == bbox_cols:
                is_full_grid = False
            else:
                raise ValueError(
                    f"{src_file.name} has unexpected shape {row_dim}/{col_dim}="
                    f"{row_size}/{col_size}. Expected full {full_rows}/{full_cols} "
                    f"or cropped {bbox_rows}/{bbox_cols}."
                )
            split_specs.append(
                (src_file, destination_name, row_dim, col_dim, is_full_grid)
            )

        print(f"Found {len(split_specs)} spatial files to split into Y stripes.")
        print(f"Found {len(copy_files)} non-spatial files to copy.")
        print(
            "Active-cell bbox on run-mask: "
            f"{run_mask_row_dim}[{row_min}:{row_max}], "
            f"{run_mask_col_dim}[{col_min}:{col_max}] "
            f"-> cropped grid {bbox_rows}x{bbox_cols}"
        )

        print("[wiemip_split:6/8] Cleaning old batches and creating batch directories")
        print("Cleaning up existing batch_* directories")
        if self.base_batch_dir.exists():
            pattern = re.compile(r"^batch_\d+$")
            to_remove = [
                d
                for d in self.base_batch_dir.iterdir()
                if d.is_dir() and pattern.match(d.name)
            ]
            with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                executor.map(shutil.rmtree, to_remove)

        batch_dirs: List[Path] = []
        batch_input_dirs: List[Path] = []
        for index in range(nbatches):
            batch_dir = self.base_batch_dir / f"batch_{index}"
            batch_dirs.append(batch_dir)
            batch_input_dirs.append(batch_dir / "input")

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            executor.map(lambda p: p.mkdir(exist_ok=True, parents=True), batch_input_dirs)

        print("[wiemip_split:7/8] Building batch input datasets")
        print("Copying non-spatial input files to every batch")
        for src_file, destination_name in copy_files:
            for batch_input_dir in batch_input_dirs:
                shutil.copy2(src_file, batch_input_dir / destination_name)

        print("Splitting spatial input files into Y-stripe batches")
        for src_file, destination_name, row_dim, col_dim, is_full_grid in split_specs:
            self._split_spatial_file_to_y_stripes(
                src_file=src_file,
                destination_name=destination_name,
                row_dim=row_dim,
                col_dim=col_dim,
                is_full_grid=is_full_grid,
                bbox=(row_min, row_max, col_min, col_max),
                y_ranges=y_ranges,
                batch_input_dirs=batch_input_dirs,
            )

        print("Validating split run-mask files")
        for (start, end), batch_input_dir in zip(y_ranges, batch_input_dirs):
            run_mask_file = batch_input_dir / RUN_MASK_DESTINATION
            if not run_mask_file.exists():
                raise FileNotFoundError(
                    f"Expected split run-mask file missing: {run_mask_file}"
                )
            self._validate_split_run_mask(
                run_mask_file=run_mask_file,
                expected_row_dim=OUTPUT_ROW_DIM,
                expected_col_dim=OUTPUT_COL_DIM,
                expected_rows=end - start,
                expected_cols=bbox_cols,
            )

        print("[wiemip_split:8/8] Creating runnable batch workdirs and configs")
        print("Setting up batch simulation folders")
        for batch_dir, batch_input_dir in zip(batch_dirs, batch_input_dirs):
            self._run_utils(batch_dir, batch_input_dir)

        print("Configuring each batch")
        for index, batch_dir in enumerate(batch_dirs):
            self._configure(index, batch_dir)

        print("Deleting duplicated inputs/ directories created by setup script")
        duplicated_inputs = self.base_batch_dir.glob("*/inputs")
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            executor.map(shutil.rmtree, duplicated_inputs)
        print("[wiemip_split] Completed successfully")
