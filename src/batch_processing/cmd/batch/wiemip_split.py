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
from batch_processing.utils.split_planning import (
    apply_run_mask_filters,
    count_active_cells,
    plan_rect_blocks_by_active_cells,
    plan_y_ranges_by_active_cells,
    plan_y_stripe_blocks_by_active_cells,
    summarize_active_cells_per_block,
)
from batch_processing.utils.utils import create_chunks, interpret_path
from batch_processing.utils.wiemip_processing import (
    RUN_ENABLED_VALUE,
    RUN_MASK_VARIABLE,
    SPLIT_METADATA_FILENAME,
    SPLIT_MODE_RECT,
    SPLIT_MODE_Y_STRIPE,
    BATCH_LAYOUT_FILENAME,
    ActiveBBox,
    WiemipSplitMetadata,
    compute_active_bbox,
    detect_spatial_dims,
    extract_run_mask_2d,
    filter_dataset_to_cropped_mask,
    open_dataset_for_read,
    write_batch_layout,
    write_split_metadata,
)

WIEMIP_NAME_ALIASES = {
    "historic_climate_GFDL-ESM4.nc": "historic-climate.nc",
    "historic-climate_GFDL-ESM4.nc": "historic-climate.nc",
    "historic_climate.nc": "historic-climate.nc",
}
MASKED_SUFFIX = "_masked.nc"
RUN_MASK_DESTINATION = "run-mask.nc"
VEGETATION_DESTINATION = "vegetation.nc"
VEG_CLASS_VARIABLE = "veg_class"
OUTPUT_ROW_DIM = "Y"
OUTPUT_COL_DIM = "X"
OUTPUT_NETCDF_FORMAT = "NETCDF4_CLASSIC"
CLIMATE_INPUT_FILENAMES = ("historic-climate.nc", "projected-climate.nc")
REQUIRED_CLIMATE_VARS = ("tair", "vapor_press", "precip", "nirr")
NONNEGATIVE_CLIMATE_VARS = ("vapor_press", "precip", "nirr")
MISSING_SENTINEL_THRESHOLD = -900.0


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
        
        n_years = getattr(self._args, "n", 0)

        for src in netcdf_files:
            dst_name = self._resolve_destination_name(src.name)
            
            if n_years == 0 and dst_name.startswith("projected-"):
                print(f"Skipping {src.name} because --n is not specified (n=0)")
                continue
                
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
                print("  [filter] cropping CMT-filtered run-mask to active bbox")
                cropped_da = run_mask_da.isel(
                    {
                        run_row_dim: slice(bbox.row_start, bbox.row_end + 1),
                        run_col_dim: slice(bbox.col_start, bbox.col_end + 1),
                    }
                )
                out_ds = xr.Dataset({RUN_MASK_VARIABLE: cropped_da})
                out_ds.to_netcdf(
                    dst_path.as_posix(),
                    engine="netcdf4",
                    format=OUTPUT_NETCDF_FORMAT,
                )
                out_ds.close()
                print(f"  [write] {dst_path}")
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

    def _split_spatial_file_to_blocks(
        self,
        src_file: Path,
        destination_name: str,
        row_dim: str,
        col_dim: str,
        is_full_grid: bool,
        bbox: Tuple[int, int, int, int],
        blocks: List[Tuple[int, int, int, int]],
        batch_input_dirs: List[Path],
    ) -> None:
        row_min, row_max, col_min, col_max = bbox
        print(f"Splitting {src_file.name} -> {destination_name} ({len(blocks)} blocks)")
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
            for batch_index, (block, batch_input_dir) in enumerate(
                zip(blocks, batch_input_dirs), start=1
            ):
                start_row, end_row, start_col, end_col = block
                subset_ds = base_ds.isel(
                    {
                        row_dim: slice(start_row, end_row),
                        col_dim: slice(start_col, end_col),
                    }
                ).load()
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
                        f"batch {batch_index}/{total_batches} "
                        f"rows {start_row}:{end_row - 1} cols {start_col}:{end_col - 1}"
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

    def _compute_invalid_spatial_mask(
        self, da: xr.DataArray, row_dim: str, col_dim: str, var_name: str
    ) -> xr.DataArray:
        if row_dim not in da.dims or col_dim not in da.dims:
            raise ValueError(
                f"Climate variable '{var_name}' must include '{row_dim}' and "
                f"'{col_dim}' dims. Found {tuple(da.dims)}."
            )

        invalid = ~np.isfinite(da)
        invalid = invalid | (da <= MISSING_SENTINEL_THRESHOLD)
        if var_name in NONNEGATIVE_CLIMATE_VARS:
            invalid = invalid | (da < 0)

        reduce_dims = [d for d in da.dims if d not in (row_dim, col_dim)]
        if reduce_dims:
            invalid = invalid.any(dim=reduce_dims)
        return invalid.transpose(row_dim, col_dim)

    def _compute_veg_class_zero_mask(
        self, da: xr.DataArray, row_dim: str, col_dim: str, var_name: str
    ) -> xr.DataArray:
        if row_dim not in da.dims or col_dim not in da.dims:
            raise ValueError(
                f"Vegetation variable '{var_name}' must include '{row_dim}' and "
                f"'{col_dim}' dims. Found {tuple(da.dims)}."
            )

        da_work = da
        for dim_name in list(da_work.dims):
            if dim_name in (row_dim, col_dim):
                continue
            sz = int(da_work.sizes[dim_name])
            if sz == 1:
                da_work = da_work.isel({dim_name: 0}, drop=True)
            elif sz < 1:
                raise ValueError(
                    f"{var_name}: dimension '{dim_name}' has invalid size {sz}."
                )

        zero = da_work == 0
        reduce_dims = [d for d in zero.dims if d not in (row_dim, col_dim)]
        if reduce_dims:
            zero = zero.any(dim=reduce_dims)
        return zero.transpose(row_dim, col_dim)

    def _compute_veg_class_gt_mask(
        self,
        da: xr.DataArray,
        row_dim: str,
        col_dim: str,
        var_name: str,
        max_cmt: int,
    ) -> xr.DataArray:
        if row_dim not in da.dims or col_dim not in da.dims:
            raise ValueError(
                f"Vegetation variable '{var_name}' must include '{row_dim}' and "
                f"'{col_dim}' dims. Found {tuple(da.dims)}."
            )

        da_work = da
        for dim_name in list(da_work.dims):
            if dim_name in (row_dim, col_dim):
                continue
            sz = int(da_work.sizes[dim_name])
            if sz == 1:
                da_work = da_work.isel({dim_name: 0}, drop=True)
            elif sz < 1:
                raise ValueError(
                    f"{var_name}: dimension '{dim_name}' has invalid size {sz}."
                )

        # NaN > N is False: those cells are not disabled by this rule.
        high = da_work > max_cmt
        reduce_dims = [d for d in high.dims if d not in (row_dim, col_dim)]
        if reduce_dims:
            high = high.any(dim=reduce_dims)
        return high.transpose(row_dim, col_dim)

    def _prefilter_batch_run_mask(
        self, batch_input_dir: Path, required_vars: tuple[str, ...]
    ) -> dict[str, int]:
        run_mask_file = batch_input_dir / RUN_MASK_DESTINATION
        if not run_mask_file.exists():
            raise FileNotFoundError(f"Missing run-mask for prefilter: {run_mask_file}")

        n_years = getattr(self._args, "n", 0)
        expected_climate_files = [f for f in CLIMATE_INPUT_FILENAMES if n_years > 0 or not f.startswith("projected-")]
        
        climate_files = [
            batch_input_dir / fname
            for fname in expected_climate_files
            if (batch_input_dir / fname).exists()
        ]
        if not climate_files:
            raise FileNotFoundError(
                f"No climate files found in {batch_input_dir}. "
                f"Expected one of {expected_climate_files}."
            )

        with open_dataset_for_read(run_mask_file) as run_mask_ds:
            run_mask_da, row_dim, col_dim = extract_run_mask_2d(
                run_mask_ds, run_mask_file.name, run_var=RUN_MASK_VARIABLE
            )
            active_before = np.isfinite(run_mask_da) & np.isclose(
                run_mask_da, RUN_ENABLED_VALUE
            )
            active_before_count = int(active_before.sum().item())

            invalid_any = xr.zeros_like(run_mask_da, dtype=bool)
            for climate_file in climate_files:
                with open_dataset_for_read(climate_file) as climate_ds:
                    missing_vars = [name for name in required_vars if name not in climate_ds]
                    if missing_vars:
                        raise KeyError(
                            f"{climate_file} missing required climate vars: {missing_vars}"
                        )
                    for var_name in required_vars:
                        invalid_mask = self._compute_invalid_spatial_mask(
                            da=climate_ds[var_name],
                            row_dim=row_dim,
                            col_dim=col_dim,
                            var_name=var_name,
                        )
                        invalid_any = invalid_any | invalid_mask

            disable_mask = active_before & invalid_any
            disabled_count = int(disable_mask.sum().item())
            active_after_count = active_before_count - disabled_count

            if disabled_count == 0:
                return {
                    "active_before": active_before_count,
                    "active_after": active_after_count,
                    "disabled": disabled_count,
                    "checked_files": len(climate_files),
                }

            updated_values = np.where(
                active_before.values & ~disable_mask.values, RUN_ENABLED_VALUE, 0
            ).astype(run_mask_da.dtype, copy=False)
            run_mask_out = run_mask_ds.copy(deep=True)
            run_mask_out[RUN_MASK_VARIABLE] = xr.DataArray(
                updated_values,
                dims=run_mask_da.dims,
                coords=run_mask_da.coords,
                attrs=run_mask_da.attrs.copy(),
            )
            tmp_path = run_mask_file.with_suffix(".tmp.nc")
            run_mask_out.to_netcdf(
                tmp_path.as_posix(),
                engine="netcdf4",
                format=OUTPUT_NETCDF_FORMAT,
            )
            run_mask_out.close()

        tmp_path.replace(run_mask_file)
        return {
            "active_before": active_before_count,
            "active_after": active_after_count,
            "disabled": disabled_count,
            "checked_files": len(climate_files),
        }

    def _prefilter_split_run_masks(
        self, batch_input_dirs: List[Path], required_vars: tuple[str, ...]
    ) -> None:
        total_disabled = 0
        batches_changed = 0
        for batch_index, batch_input_dir in enumerate(batch_input_dirs, start=1):
            result = self._prefilter_batch_run_mask(
                batch_input_dir=batch_input_dir, required_vars=required_vars
            )
            total_disabled += result["disabled"]
            if result["disabled"] > 0:
                batches_changed += 1
            print(
                "  [runmask-prefilter] "
                f"{batch_input_dir.parent.name} ({batch_index}/{len(batch_input_dirs)}): "
                f"active {result['active_before']} -> {result['active_after']} "
                f"(disabled {result['disabled']}, climate files {result['checked_files']})"
            )
        print(
            "[runmask-prefilter] Done: "
            f"disabled {total_disabled} active cells across {batches_changed} batches."
        )

    def _prefilter_batch_run_mask_cmt0(self, batch_input_dir: Path) -> dict[str, int]:
        run_mask_file = batch_input_dir / RUN_MASK_DESTINATION
        vegetation_file = batch_input_dir / VEGETATION_DESTINATION
        if not run_mask_file.exists():
            raise FileNotFoundError(f"Missing run-mask for cmt0-filter: {run_mask_file}")
        if not vegetation_file.exists():
            raise FileNotFoundError(
                f"Missing vegetation for cmt0-filter: {vegetation_file}"
            )

        with open_dataset_for_read(run_mask_file) as run_mask_ds:
            run_mask_da, row_dim, col_dim = extract_run_mask_2d(
                run_mask_ds, run_mask_file.name, run_var=RUN_MASK_VARIABLE
            )
            active_before = np.isfinite(run_mask_da) & np.isclose(
                run_mask_da, RUN_ENABLED_VALUE
            )
            active_before_count = int(active_before.sum().item())

            with open_dataset_for_read(vegetation_file) as veg_ds:
                if VEG_CLASS_VARIABLE not in veg_ds:
                    raise KeyError(
                        f"{vegetation_file} missing '{VEG_CLASS_VARIABLE}' variable."
                    )
                veg_zero = self._compute_veg_class_zero_mask(
                    veg_ds[VEG_CLASS_VARIABLE],
                    row_dim=row_dim,
                    col_dim=col_dim,
                    var_name=VEG_CLASS_VARIABLE,
                )

            if veg_zero.sizes != run_mask_da.sizes:
                raise ValueError(
                    f"veg_class grid {dict(veg_zero.sizes)} does not match "
                    f"run grid {dict(run_mask_da.sizes)} in {batch_input_dir}."
                )

            disable_mask = active_before & veg_zero
            disabled_count = int(disable_mask.sum().item())
            active_after_count = active_before_count - disabled_count

            if disabled_count == 0:
                return {
                    "active_before": active_before_count,
                    "active_after": active_after_count,
                    "disabled": disabled_count,
                }

            updated_values = np.where(
                active_before.values & ~disable_mask.values, RUN_ENABLED_VALUE, 0
            ).astype(run_mask_da.dtype, copy=False)
            run_mask_out = run_mask_ds.copy(deep=True)
            run_mask_out[RUN_MASK_VARIABLE] = xr.DataArray(
                updated_values,
                dims=run_mask_da.dims,
                coords=run_mask_da.coords,
                attrs=run_mask_da.attrs.copy(),
            )
            tmp_path = run_mask_file.with_suffix(".tmp.nc")
            run_mask_out.to_netcdf(
                tmp_path.as_posix(),
                engine="netcdf4",
                format=OUTPUT_NETCDF_FORMAT,
            )
            run_mask_out.close()

        tmp_path.replace(run_mask_file)
        return {
            "active_before": active_before_count,
            "active_after": active_after_count,
            "disabled": disabled_count,
        }

    def _prefilter_split_run_masks_cmt0(self, batch_input_dirs: List[Path]) -> None:
        total_disabled = 0
        batches_changed = 0
        for batch_index, batch_input_dir in enumerate(batch_input_dirs, start=1):
            result = self._prefilter_batch_run_mask_cmt0(batch_input_dir=batch_input_dir)
            total_disabled += result["disabled"]
            if result["disabled"] > 0:
                batches_changed += 1
            print(
                "  [cmt0-filter] "
                f"{batch_input_dir.parent.name} ({batch_index}/{len(batch_input_dirs)}): "
                f"active {result['active_before']} -> {result['active_after']} "
                f"(disabled {result['disabled']})"
            )
        print(
            "[cmt0-filter] Done: "
            f"disabled {total_disabled} active cells across {batches_changed} batches."
        )

    def _prefilter_batch_run_mask_max_cmt(
        self, batch_input_dir: Path, max_cmt: int
    ) -> dict[str, int]:
        run_mask_file = batch_input_dir / RUN_MASK_DESTINATION
        vegetation_file = batch_input_dir / VEGETATION_DESTINATION
        if not run_mask_file.exists():
            raise FileNotFoundError(
                f"Missing run-mask for max-cmt filter: {run_mask_file}"
            )
        if not vegetation_file.exists():
            raise FileNotFoundError(
                f"Missing vegetation for max-cmt filter: {vegetation_file}"
            )

        with open_dataset_for_read(run_mask_file) as run_mask_ds:
            run_mask_da, row_dim, col_dim = extract_run_mask_2d(
                run_mask_ds, run_mask_file.name, run_var=RUN_MASK_VARIABLE
            )
            active_before = np.isfinite(run_mask_da) & np.isclose(
                run_mask_da, RUN_ENABLED_VALUE
            )
            active_before_count = int(active_before.sum().item())

            with open_dataset_for_read(vegetation_file) as veg_ds:
                if VEG_CLASS_VARIABLE not in veg_ds:
                    raise KeyError(
                        f"{vegetation_file} missing '{VEG_CLASS_VARIABLE}' variable."
                    )
                veg_high = self._compute_veg_class_gt_mask(
                    veg_ds[VEG_CLASS_VARIABLE],
                    row_dim=row_dim,
                    col_dim=col_dim,
                    var_name=VEG_CLASS_VARIABLE,
                    max_cmt=max_cmt,
                )

            if veg_high.sizes != run_mask_da.sizes:
                raise ValueError(
                    f"veg_class grid {dict(veg_high.sizes)} does not match "
                    f"run grid {dict(run_mask_da.sizes)} in {batch_input_dir}."
                )

            disable_mask = active_before & veg_high
            disabled_count = int(disable_mask.sum().item())
            active_after_count = active_before_count - disabled_count

            if disabled_count == 0:
                return {
                    "active_before": active_before_count,
                    "active_after": active_after_count,
                    "disabled": disabled_count,
                }

            updated_values = np.where(
                active_before.values & ~disable_mask.values, RUN_ENABLED_VALUE, 0
            ).astype(run_mask_da.dtype, copy=False)
            run_mask_out = run_mask_ds.copy(deep=True)
            run_mask_out[RUN_MASK_VARIABLE] = xr.DataArray(
                updated_values,
                dims=run_mask_da.dims,
                coords=run_mask_da.coords,
                attrs=run_mask_da.attrs.copy(),
            )
            tmp_path = run_mask_file.with_suffix(".tmp.nc")
            run_mask_out.to_netcdf(
                tmp_path.as_posix(),
                engine="netcdf4",
                format=OUTPUT_NETCDF_FORMAT,
            )
            run_mask_out.close()

        tmp_path.replace(run_mask_file)
        return {
            "active_before": active_before_count,
            "active_after": active_after_count,
            "disabled": disabled_count,
        }

    def _prefilter_split_run_masks_max_cmt(
        self, batch_input_dirs: List[Path], max_cmt: int
    ) -> None:
        total_disabled = 0
        batches_changed = 0
        for batch_index, batch_input_dir in enumerate(batch_input_dirs, start=1):
            result = self._prefilter_batch_run_mask_max_cmt(
                batch_input_dir=batch_input_dir, max_cmt=max_cmt
            )
            total_disabled += result["disabled"]
            if result["disabled"] > 0:
                batches_changed += 1
            print(
                "  [max-cmt-filter] "
                f"{batch_input_dir.parent.name} ({batch_index}/{len(batch_input_dirs)}): "
                f"active {result['active_before']} -> {result['active_after']} "
                f"(disabled {result['disabled']}, max_cmt={max_cmt})"
            )
        print(
            "[max-cmt-filter] Done: "
            f"disabled {total_disabled} active cells across {batches_changed} batches "
            f"(veg_class > {max_cmt})."
        )

    def _load_staged_run_mask_array(self, staging_path: Path) -> np.ndarray:
        staged_run_mask = staging_path / RUN_MASK_DESTINATION
        with open_dataset_for_read(staged_run_mask) as ds:
            run_values = np.asarray(ds[RUN_MASK_VARIABLE].values, dtype=float)
        return np.where(np.isnan(run_values), 0, run_values)

    def _apply_cmt_filters_to_full_run_mask(
        self,
        run_mask_da: xr.DataArray,
        input_path: Path,
        *,
        cmt0_filter: bool,
        no_max_cmt: bool,
        max_cmt_val: int,
    ) -> xr.DataArray:
        run_values = np.asarray(run_mask_da.values, dtype=float)
        run_values = np.where(np.isnan(run_values), 0, run_values)
        active_before = count_active_cells(run_values)
        filtered = apply_run_mask_filters(
            run_values,
            input_path,
            cmt0_filter=cmt0_filter,
            no_max_cmt=no_max_cmt,
            max_cmt=max_cmt_val,
        )
        active_after = count_active_cells(filtered)
        if active_after == 0:
            raise ValueError(
                "No active cells remain in run-mask after CMT filters on full grid."
            )
        print(
            "[wiemip_split] Applied CMT filters on full grid before bbox/split: "
            f"active {active_before} -> {active_after}"
        )
        return xr.DataArray(
            filtered.astype(run_mask_da.dtype, copy=False),
            dims=run_mask_da.dims,
            coords=run_mask_da.coords,
            attrs=run_mask_da.attrs.copy(),
        )

    def _warn_restart_with_prerun_years(self) -> None:
        """Guard the common restart foot-gun.

        dvmdostem asserts that PR and EQ years are 0 when a restart file is set
        (see TEM.cpp advance_model). Setting restart_from (explicit path or
        scenario continuation) together with -p>0 or -e>0 makes every batch job
        abort immediately. Warn loudly before submission so users catch it here.
        """
        restart_from = getattr(self._args, "restart_from", None)
        scenario_continuation = getattr(self._args, "scenario_continuation", False)
        restart_active = bool(restart_from) or scenario_continuation
        if not restart_active:
            return
        p = int(getattr(self._args, "p", 0) or 0)
        e = int(getattr(self._args, "e", 0) or 0)
        if p > 0 or e > 0:
            import typer
            label = (
                f"restart_from={restart_from!r}"
                if restart_from
                else "scenario_continuation=True"
            )
            typer.secho(
                "[wiemip_split] WARNING: a restart is configured "
                f"({label}) but PR/EQ years are non-zero (-p {p} -e {e}). "
                "dvmdostem cannot run PR or EQ years when restarting and every "
                "batch will abort with "
                "'Cannot run PR years when restarting from a previous run'. "
                "For a restart, set -p 0 -e 0 (and -s 0 to start from a spin-up "
                "restart into the transient stage).",
                err=True,
                fg=typer.colors.YELLOW,
            )

    def execute(self) -> None:
        print("[wiemip_split] Starting integrated WIEMIP split workflow")
        self._warn_restart_with_prerun_years()
        slurm_partition = getattr(self._args, "slurm_partition", "spot")
        mpi_ranks = getattr(self._args, "mpi_ranks", None)
        if mpi_ranks is not None:
            max_cpus = {"spot": 8, "dask": 4, "compute": 15}.get(slurm_partition, 8)
            if int(mpi_ranks) > max_cpus:
                import typer
                typer.secho(
                    f"Error: The requested --mpi-ranks ({mpi_ranks}) exceeds the maximum number of CPUs ({max_cpus}) available per node for the '{slurm_partition}' partition.",
                    err=True, fg=typer.colors.RED
                )
                import sys
                sys.exit(1)

        if str(self.input_path).startswith("gcs://"):
            raise NotImplementedError(
                "bp batch wiemip_split currently supports local input paths only."
            )

        input_path = Path(interpret_path(self.input_path))
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        print(f"[wiemip_split] Input path: {input_path}")

        nbatches_arg = getattr(self._args, "nbatches", None)
        cells_per_batch = getattr(self._args, "cells_per_batch", None)
        if cells_per_batch is not None and int(cells_per_batch) < 1:
            raise ValueError("cells_per_batch must be >= 1")
        if nbatches_arg is not None and int(nbatches_arg) < 1:
            raise ValueError("nbatches must be >= 1")
        if cells_per_batch is None and nbatches_arg is None:
            raise ValueError("Specify --cells-per-batch or --nbatches.")
        if cells_per_batch is not None and nbatches_arg is not None:
            print(
                "[wiemip_split] Both --cells-per-batch and --nbatches were provided; "
                "using active-cell balancing."
            )
        nbatches = int(nbatches_arg) if nbatches_arg is not None else 0
        if cells_per_batch is not None:
            print(
                "[wiemip_split] Active cells per batch target: "
                f"{int(cells_per_batch)}"
            )
        elif nbatches:
            print(f"[wiemip_split] Requested batch count: {nbatches}")
        runmask_prefilter = bool(getattr(self._args, "runmask_prefilter", True))
        split_mode = str(getattr(self._args, "split_mode", SPLIT_MODE_Y_STRIPE)).strip().lower()
        if split_mode not in (SPLIT_MODE_Y_STRIPE, SPLIT_MODE_RECT):
            raise ValueError(
                f"Unsupported split_mode '{split_mode}'. "
                f"Use '{SPLIT_MODE_Y_STRIPE}' or '{SPLIT_MODE_RECT}'."
            )
        min_cells_per_batch = int(getattr(self._args, "min_cells_per_batch", 1))
        if min_cells_per_batch < 1:
            raise ValueError("min_cells_per_batch must be >= 1")
        cmt0_filter = bool(getattr(self._args, "cmt0_filter", False))
        no_max_cmt = bool(getattr(self._args, "no_max_cmt", False))
        max_cmt_val = int(getattr(self._args, "max_cmt", 74))
        max_cmt = None if no_max_cmt else max_cmt_val
        cmt_filters_before_split = cmt0_filter or (max_cmt is not None)
        print(f"[wiemip_split] Run-mask prefilter: {runmask_prefilter}")
        print(f"[wiemip_split] Split mode: {split_mode}")
        if split_mode == SPLIT_MODE_RECT:
            print(
                "[wiemip_split] Rect split min active cells per batch: "
                f"{min_cells_per_batch}"
            )
        print(f"[wiemip_split] CMT0 run-mask filter (veg_class==0): {cmt0_filter}")
        if max_cmt is not None:
            print(
                f"[wiemip_split] Max-CMT run-mask filter (veg_class > {max_cmt}): True "
                "(applied before split when enabled)"
            )
        else:
            print(
                "[wiemip_split] Max-CMT run-mask filter (veg_class > N): False "
                "(--no-max-cmt)"
            )

        print("[wiemip_split:1/8] Discovering run-mask source")
        _, _, run_mask_source = self._discover_inputs(input_path)
        print(f"[wiemip_split] Run-mask source: {run_mask_source}")
        print("[wiemip_split:2/8] Loading run-mask, applying CMT filters, computing bbox")
        with open_dataset_for_read(run_mask_source) as run_mask_ds:
            run_mask_da, run_mask_row_dim, run_mask_col_dim = extract_run_mask_2d(
                run_mask_ds, run_mask_source.name, run_var=RUN_MASK_VARIABLE
            )
            full_rows = int(run_mask_da.sizes[run_mask_row_dim])
            full_cols = int(run_mask_da.sizes[run_mask_col_dim])

        if cmt_filters_before_split:
            run_mask_da = self._apply_cmt_filters_to_full_run_mask(
                run_mask_da,
                input_path,
                cmt0_filter=cmt0_filter,
                no_max_cmt=no_max_cmt,
                max_cmt_val=max_cmt_val,
            )

        bbox = compute_active_bbox(run_mask_da, active_value=RUN_ENABLED_VALUE)
        row_min, row_max = bbox.row_start, bbox.row_end
        col_min, col_max = bbox.col_start, bbox.col_end
        bbox_rows, bbox_cols = bbox.n_rows, bbox.n_cols
        if cells_per_batch is None and nbatches > bbox_rows:
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
            schema_version=2,
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
            split_mode=split_mode,
        )
        write_split_metadata(self.base_batch_dir / SPLIT_METADATA_FILENAME, metadata)
        print(f"Wrote split metadata: {self.base_batch_dir / SPLIT_METADATA_FILENAME}")

        print("[wiemip_split:5/8] Planning split blocks from staged inputs")
        row_spatial_files, copy_files, _ = self._discover_inputs(staging_path)

        cropped_run_data = self._load_staged_run_mask_array(staging_path)
        blocks: List[Tuple[int, int, int, int]] = []

        if cells_per_batch is not None:
            target_cells = int(cells_per_batch)
            if split_mode == SPLIT_MODE_RECT:
                blocks = plan_rect_blocks_by_active_cells(
                    cropped_run_data,
                    target_cells,
                    min_active_cells=min_cells_per_batch,
                )
                avg_cells, min_cells, max_cells = summarize_active_cells_per_block(
                    blocks, cropped_run_data
                )
                print(
                    "[wiemip_split] Planned "
                    f"{len(blocks)} rect batches by "
                    f"~{target_cells} active cells/batch "
                    f"(avg/min/max: {avg_cells:.1f}/{min_cells}/{max_cells})"
                )
            else:
                blocks = plan_y_stripe_blocks_by_active_cells(
                    cropped_run_data, target_cells
                )
                avg_cells, min_cells, max_cells = summarize_active_cells_per_block(
                    blocks, cropped_run_data
                )
                print(
                    "[wiemip_split] Planned "
                    f"{len(blocks)} Y-stripe batches by "
                    f"~{target_cells} active cells/batch "
                    f"(avg/min/max: {avg_cells:.1f}/{min_cells}/{max_cells})"
                )
            nbatches = len(blocks)
        else:
            chunks = create_chunks(bbox_rows, nbatches)
            y_ranges = [(int(chunk.start), int(chunk.end)) for chunk in chunks]
            blocks = [(y_start, y_end, 0, bbox_cols) for y_start, y_end in y_ranges]
            if split_mode == SPLIT_MODE_RECT:
                raise ValueError(
                    "Rect split requires --cells-per-batch. "
                    "Equal-row --nbatches is only supported with y-stripe split."
                )
            print(
                f"[wiemip_split] Planned {nbatches} equal-row Y-stripe batches "
                f"({bbox_rows} cropped rows)"
            )

        if split_mode == SPLIT_MODE_RECT:
            layout_path = self.base_batch_dir / BATCH_LAYOUT_FILENAME
            write_batch_layout(
                layout_path,
                blocks=blocks,
                grid_y=bbox_rows,
                grid_x=bbox_cols,
            )
            metadata.blocks = [list(block) for block in blocks]
            write_split_metadata(self.base_batch_dir / SPLIT_METADATA_FILENAME, metadata)
            print(f"Wrote batch layout: {layout_path}")

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

        print(f"Found {len(split_specs)} spatial files to split into batches.")
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

        print("Splitting spatial input files into batch blocks")
        for src_file, destination_name, row_dim, col_dim, is_full_grid in split_specs:
            self._split_spatial_file_to_blocks(
                src_file=src_file,
                destination_name=destination_name,
                row_dim=row_dim,
                col_dim=col_dim,
                is_full_grid=is_full_grid,
                bbox=(row_min, row_max, col_min, col_max),
                blocks=blocks,
                batch_input_dirs=batch_input_dirs,
            )

        print("Validating split run-mask files")
        for block, batch_input_dir in zip(blocks, batch_input_dirs):
            start_row, end_row, start_col, end_col = block
            run_mask_file = batch_input_dir / RUN_MASK_DESTINATION
            if not run_mask_file.exists():
                raise FileNotFoundError(
                    f"Expected split run-mask file missing: {run_mask_file}"
                )
            self._validate_split_run_mask(
                run_mask_file=run_mask_file,
                expected_row_dim=OUTPUT_ROW_DIM,
                expected_col_dim=OUTPUT_COL_DIM,
                expected_rows=end_row - start_row,
                expected_cols=end_col - start_col,
            )

        num_prefilters = int(runmask_prefilter) + (
            int(cmt0_filter and not cmt_filters_before_split)
            + int(max_cmt is not None and not cmt_filters_before_split)
        )
        tail_total = 8 + num_prefilters
        step = 8

        if runmask_prefilter:
            print(
                f"[wiemip_split:{step}/{tail_total}] Pre-filtering split run-mask files "
                "using required climate vars"
            )
            self._prefilter_split_run_masks(
                batch_input_dirs=batch_input_dirs,
                required_vars=REQUIRED_CLIMATE_VARS,
            )
            step += 1
        else:
            print(
                "[wiemip_split] Skipping run-mask prefilter due to "
                "--no-runmask-prefilter."
            )

        if cmt0_filter and not cmt_filters_before_split:
            print(
                f"[wiemip_split:{step}/{tail_total}] Pre-filtering split run-mask files "
                "(disable cells where veg_class==0)"
            )
            self._prefilter_split_run_masks_cmt0(batch_input_dirs=batch_input_dirs)
            step += 1
        elif cmt0_filter:
            print(
                "[wiemip_split] Skipping post-split cmt0-filter "
                "(already applied before split)."
            )

        if max_cmt is not None and not cmt_filters_before_split:
            print(
                f"[wiemip_split:{step}/{tail_total}] Pre-filtering split run-mask files "
                f"(disable cells where veg_class>{max_cmt})"
            )
            self._prefilter_split_run_masks_max_cmt(
                batch_input_dirs=batch_input_dirs, max_cmt=max_cmt
            )
            step += 1
        elif max_cmt is not None:
            print(
                "[wiemip_split] Skipping post-split max-cmt filter "
                "(already applied before split)."
            )

        if num_prefilters == 0:
            setup_step = "[wiemip_split:8/8]"
        else:
            setup_step = f"[wiemip_split:{tail_total}/{tail_total}]"

        print(f"{setup_step} Creating runnable batch workdirs and configs")
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
