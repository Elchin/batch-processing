from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

from batch_processing.cmd.base import BaseCommand
from batch_processing.utils.utils import interpret_path
from batch_processing.utils.wiemip_processing import (
    RUN_ENABLED_VALUE,
    RUN_MASK_VARIABLE,
    detect_spatial_dims,
    extract_run_mask_2d,
    open_dataset_for_read,
)

RUN_STATUS_FILENAME = "run_status.nc"
RUN_STATUS_VARIABLE = "run_status"
SUCCESS_STATUS_VALUE = 100
RETRY_DIR_NAME = "retry"


class WiemipReRunMergeCommand(BaseCommand):
    """Merge retry output files back into a single WIEMIP batch output directory."""

    def __init__(self, args):
        super().__init__()
        self._args = args
        self.batch_path = self._resolve_batch_path(args.batch_path)
        self.output_dir = self.batch_path / "output"
        self.retry_output_dir = self.batch_path / RETRY_DIR_NAME / "output"
        self.input_run_mask_path = self.batch_path / "input" / "run-mask.nc"
        self.output_run_status_path = self.output_dir / RUN_STATUS_FILENAME
        self.retry_run_status_path = self.retry_output_dir / RUN_STATUS_FILENAME

    def _resolve_batch_path(self, batch_path: str) -> Path:
        direct_candidate = Path(interpret_path(batch_path))
        if direct_candidate.exists():
            return direct_candidate.resolve()

        exacloud_candidate = Path(self.exacloud_user_dir, batch_path)
        if exacloud_candidate.exists():
            return exacloud_candidate.resolve()

        return direct_candidate

    def _validate_paths(self) -> None:
        if not self.batch_path.exists() or not self.batch_path.is_dir():
            raise FileNotFoundError(f"Batch directory not found: {self.batch_path}")

        required_paths = [
            self.output_dir,
            self.retry_output_dir,
            self.input_run_mask_path,
            self.output_run_status_path,
            self.retry_run_status_path,
        ]
        missing = [path for path in required_paths if not path.exists()]
        if missing:
            missing_paths = ", ".join(path.as_posix() for path in missing)
            raise FileNotFoundError(f"Missing required retry merge inputs: {missing_paths}")

    def _extract_run_status_2d(self, ds: xr.Dataset, source_label: str) -> xr.DataArray:
        if RUN_STATUS_VARIABLE not in ds:
            raise KeyError(f"{source_label} must contain '{RUN_STATUS_VARIABLE}' variable.")
        run_status_da = ds[RUN_STATUS_VARIABLE]
        spatial_dims = detect_spatial_dims(run_status_da.dims)
        if spatial_dims is None:
            raise ValueError(
                f"{source_label}:{RUN_STATUS_VARIABLE} must include row/col dims."
            )
        row_dim, col_dim = spatial_dims
        for dim_name in run_status_da.dims:
            if dim_name in (row_dim, col_dim):
                continue
            if int(run_status_da.sizes[dim_name]) != 1:
                raise ValueError(
                    f"{source_label}:{RUN_STATUS_VARIABLE} contains non-singleton "
                    f"extra dimension '{dim_name}' with size "
                    f"{int(run_status_da.sizes[dim_name])}."
                )
            run_status_da = run_status_da.isel({dim_name: 0}, drop=True)
        return run_status_da.transpose(row_dim, col_dim)

    def _get_completion_counts(self, run_status_file: Path) -> Tuple[int, int, int]:
        with open_dataset_for_read(self.input_run_mask_path) as run_mask_ds:
            run_mask_da, _, _ = extract_run_mask_2d(
                run_mask_ds,
                self.input_run_mask_path.as_posix(),
                run_var=RUN_MASK_VARIABLE,
            )
            run_mask_values = np.asarray(run_mask_da.values)

        with open_dataset_for_read(run_status_file) as status_ds:
            run_status_da = self._extract_run_status_2d(
                status_ds, run_status_file.as_posix()
            )
            run_status_values = np.asarray(run_status_da.values)

        if run_status_values.shape != run_mask_values.shape:
            raise ValueError(
                "run_status and run-mask shapes do not match for completion check. "
                f"run_status={run_status_values.shape}, run-mask={run_mask_values.shape}"
            )

        active_mask = np.isfinite(run_mask_values) & np.isclose(
            run_mask_values, RUN_ENABLED_VALUE
        )
        completed_mask = np.isfinite(run_status_values) & np.isclose(
            run_status_values, SUCCESS_STATUS_VALUE
        )
        n_cells = int(np.sum(active_mask))
        m_completed = int(np.sum(active_mask & completed_mask))
        remaining = max(0, n_cells - m_completed)
        return m_completed, n_cells, remaining

    def _is_valid_retry_value_for_run_status_extra_var(
        self, retry_values: np.ndarray, retry_da: xr.DataArray
    ) -> np.ndarray:
        fill_value = retry_da.attrs.get("_FillValue", retry_da.encoding.get("_FillValue"))
        if fill_value is not None:
            if np.issubdtype(retry_values.dtype, np.floating):
                return (retry_values != 0) & (retry_values != fill_value) & ~np.isnan(
                    retry_values
                )
            return (retry_values != 0) & (retry_values != fill_value)
        if np.issubdtype(retry_values.dtype, np.floating):
            return (retry_values != 0) & ~np.isnan(retry_values)
        return retry_values != 0

    def _is_valid_retry_value_for_output_var(
        self, retry_values: np.ndarray, retry_da: xr.DataArray
    ) -> np.ndarray:
        fill_value = retry_da.attrs.get("_FillValue", retry_da.encoding.get("_FillValue"))
        if np.issubdtype(retry_values.dtype, np.floating):
            if fill_value is not None:
                return ~np.isnan(retry_values) & (retry_values != fill_value)
            return ~np.isnan(retry_values)

        if fill_value is not None:
            return retry_values != fill_value
        return np.ones_like(retry_values, dtype=bool)

    def _atomic_write_dataset(self, ds: xr.Dataset, target_path: Path) -> None:
        temp_path = target_path.with_suffix(".tmp.nc")
        try:
            ds.to_netcdf(temp_path.as_posix(), engine="netcdf4")
            temp_path.replace(target_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _merge_run_status(self) -> None:
        with open_dataset_for_read(self.output_run_status_path) as original_ds_in:
            original_ds = original_ds_in.load()
        with open_dataset_for_read(self.retry_run_status_path) as retry_ds_in:
            retry_ds = retry_ds_in.load()

        if RUN_STATUS_VARIABLE not in original_ds or RUN_STATUS_VARIABLE not in retry_ds:
            raise KeyError(
                f"Both run_status datasets must include '{RUN_STATUS_VARIABLE}'."
            )

        original_status = np.asarray(original_ds[RUN_STATUS_VARIABLE].values)
        retry_status = np.asarray(retry_ds[RUN_STATUS_VARIABLE].values)
        if original_status.shape != retry_status.shape:
            raise ValueError(
                "Original and retry run_status shapes do not match. "
                f"original={original_status.shape}, retry={retry_status.shape}"
            )

        newly_successful_mask = (retry_status == SUCCESS_STATUS_VALUE) & (
            original_status != SUCCESS_STATUS_VALUE
        )
        merged_status = np.array(original_status, copy=True)
        merged_status[newly_successful_mask] = SUCCESS_STATUS_VALUE
        original_ds[RUN_STATUS_VARIABLE].values[:] = merged_status

        for var_name in retry_ds.data_vars:
            if var_name == RUN_STATUS_VARIABLE or var_name not in original_ds.data_vars:
                continue

            retry_da = retry_ds[var_name]
            original_da = original_ds[var_name]
            if retry_da.shape != original_da.shape:
                print(
                    f"Skipping variable '{var_name}' in {RUN_STATUS_FILENAME} due to "
                    "shape mismatch."
                )
                continue

            retry_values = np.asarray(retry_da.values)
            original_values = np.asarray(original_da.values).copy()
            valid_mask = self._is_valid_retry_value_for_run_status_extra_var(
                retry_values, retry_da
            )
            original_values[valid_mask] = retry_values[valid_mask]
            original_ds[var_name].values[:] = original_values

        try:
            self._atomic_write_dataset(original_ds, self.output_run_status_path)
        finally:
            original_ds.close()
            retry_ds.close()

    def _merge_existing_output_file(self, original_file: Path, retry_file: Path) -> None:
        with open_dataset_for_read(original_file) as original_ds_in:
            original_ds = original_ds_in.load()
        with open_dataset_for_read(retry_file) as retry_ds_in:
            retry_ds = retry_ds_in.load()

        for var_name in retry_ds.data_vars:
            if var_name not in original_ds.data_vars:
                continue

            retry_da = retry_ds[var_name]
            original_da = original_ds[var_name]
            if retry_da.shape != original_da.shape:
                print(
                    f"Skipping variable '{var_name}' in {retry_file.name} due to "
                    "shape mismatch."
                )
                continue

            retry_values = np.asarray(retry_da.values)
            original_values = np.asarray(original_da.values).copy()

            if retry_values.ndim == 0:
                valid_mask = self._is_valid_retry_value_for_output_var(
                    np.asarray([retry_values]), retry_da
                )[0]
                if valid_mask:
                    original_ds[var_name].values = retry_values
                continue

            valid_mask = self._is_valid_retry_value_for_output_var(retry_values, retry_da)
            original_values[valid_mask] = retry_values[valid_mask]
            original_ds[var_name].values[:] = original_values

        try:
            self._atomic_write_dataset(original_ds, original_file)
        finally:
            original_ds.close()
            retry_ds.close()

    def _merge_other_output_files(self) -> None:
        retry_nc_files = sorted(
            path for path in self.retry_output_dir.glob("*.nc") if path.is_file()
        )
        for retry_file in retry_nc_files:
            if retry_file.name == RUN_STATUS_FILENAME:
                continue

            original_file = self.output_dir / retry_file.name
            if not original_file.exists():
                shutil.copy2(retry_file, original_file)
                print(f"Copied retry file: {retry_file.name}")
                continue

            try:
                self._merge_existing_output_file(original_file, retry_file)
                print(f"Merged retry file: {retry_file.name}")
            except Exception as exc:
                print(
                    f"Merge failed for {retry_file.name} ({exc}). "
                    "Falling back to copying retry file."
                )
                shutil.copy2(retry_file, original_file)

    def execute(self) -> None:
        self._validate_paths()
        m_before, n_cells, remaining_before = self._get_completion_counts(
            self.output_run_status_path
        )
        print(
            f"Before merge: completed {m_before}/{n_cells} active cells "
            f"(remaining {remaining_before})"
        )

        self._merge_run_status()
        self._merge_other_output_files()

        m_after, _, remaining_after = self._get_completion_counts(
            self.output_run_status_path
        )
        print(
            f"After merge: completed {m_after}/{n_cells} active cells "
            f"(remaining {remaining_after})"
        )
        if m_after < n_cells:
            print(
                "Retry merge completed with remaining incomplete cells. "
                "Outputs were partially merged."
            )
        else:
            print("Retry merge completed. Batch output is fully complete.")
