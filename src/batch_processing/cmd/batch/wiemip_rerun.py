from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from batch_processing.cmd.base import BaseCommand
from batch_processing.utils.utils import (
    extract_sbatch_job_id,
    interpret_path,
    submit_job,
)
from batch_processing.utils.wiemip_processing import (
    RUN_ENABLED_VALUE,
    RUN_MASK_VARIABLE,
    detect_spatial_dims,
    extract_run_mask_2d,
    open_dataset_for_read,
)

RUN_STATUS_VARIABLE = "run_status"
RUN_STATUS_SUCCESS_VALUE = 100
RETRY_DIR_NAME = "retry"


class WiemipReRunCommand(BaseCommand):
    """Create and submit a single-batch WIEMIP retry run."""

    def __init__(self, args):
        super().__init__()
        self._args = args
        self.batch_path = self._resolve_batch_path(args.batch_path)
        self.retry_path = self.batch_path / RETRY_DIR_NAME
        self.retry_run_mask_path = self.retry_path / "input" / "run-mask.nc"
        self.retry_config_path = self.retry_path / "config" / "config.js"
        self.retry_slurm_path = self.retry_path / "slurm_runner.sh"
        self.source_run_mask_path = self.batch_path / "input" / "run-mask.nc"
        self.source_run_status_path = self.batch_path / "output" / "run_status.nc"
        self.source_config_path = self.batch_path / "config" / "config.js"
        self.source_slurm_path = self.batch_path / "slurm_runner.sh"

    def _resolve_batch_path(self, batch_path: str) -> Path:
        direct_candidate = Path(interpret_path(batch_path))
        if direct_candidate.exists():
            return direct_candidate.resolve()

        exacloud_candidate = Path(self.exacloud_user_dir, batch_path)
        if exacloud_candidate.exists():
            return exacloud_candidate.resolve()

        return direct_candidate

    def _validate_batch_structure(self) -> None:
        if not self.batch_path.exists() or not self.batch_path.is_dir():
            raise FileNotFoundError(f"Batch directory not found: {self.batch_path}")

        required_files = [
            self.source_run_mask_path,
            self.source_run_status_path,
            self.source_config_path,
            self.source_slurm_path,
        ]
        missing = [path for path in required_files if not path.exists()]
        if missing:
            missing_as_text = ", ".join(path.as_posix() for path in missing)
            raise FileNotFoundError(
                f"Batch is missing required files for rerun: {missing_as_text}"
            )

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

    def _build_retry_run_mask(self) -> tuple[np.ndarray, str, str, int, int, int]:
        with open_dataset_for_read(self.source_run_mask_path) as run_mask_ds:
            run_mask_da, run_row_dim, run_col_dim = extract_run_mask_2d(
                run_mask_ds,
                self.source_run_mask_path.as_posix(),
                run_var=RUN_MASK_VARIABLE,
            )
            run_mask_values = np.asarray(run_mask_da.values)

        with open_dataset_for_read(self.source_run_status_path) as status_ds:
            run_status_da = self._extract_run_status_2d(
                status_ds, self.source_run_status_path.as_posix()
            )
            run_status_values = np.asarray(run_status_da.values)

        if run_status_values.shape != run_mask_values.shape:
            raise ValueError(
                "run_status and run-mask shapes do not match. "
                f"run_status={run_status_values.shape}, run-mask={run_mask_values.shape}"
            )

        enabled_mask = np.isfinite(run_mask_values) & np.isclose(
            run_mask_values, RUN_ENABLED_VALUE
        )
        completed_mask = np.isfinite(run_status_values) & np.isclose(
            run_status_values, RUN_STATUS_SUCCESS_VALUE
        )
        failed_mask = enabled_mask & ~completed_mask
        retry_run_mask = np.where(failed_mask, RUN_ENABLED_VALUE, 0).astype(
            run_mask_values.dtype, copy=False
        )

        enabled_cells = int(np.sum(enabled_mask))
        completed_cells = int(np.sum(enabled_mask & completed_mask))
        failed_cells = int(np.sum(failed_mask))
        return (
            retry_run_mask,
            run_row_dim,
            run_col_dim,
            enabled_cells,
            completed_cells,
            failed_cells,
        )

    def _copy_batch_to_retry(self) -> None:
        if self.retry_path.exists():
            if not getattr(self._args, "force", False):
                raise FileExistsError(
                    f"Retry directory already exists: {self.retry_path}. "
                    "Use --force to overwrite."
                )
            shutil.rmtree(self.retry_path)

        shutil.copytree(
            self.batch_path,
            self.retry_path,
            ignore=shutil.ignore_patterns(RETRY_DIR_NAME),
        )

    def _write_retry_run_mask(
        self, retry_run_mask: np.ndarray, row_dim: str, col_dim: str
    ) -> None:
        with open_dataset_for_read(self.retry_run_mask_path) as in_ds:
            if RUN_MASK_VARIABLE not in in_ds:
                raise KeyError(
                    f"{self.retry_run_mask_path} missing '{RUN_MASK_VARIABLE}' variable."
                )
            updated_ds = in_ds.load()

        original_da = updated_ds[RUN_MASK_VARIABLE]
        updated_da = xr.DataArray(
            retry_run_mask,
            dims=(row_dim, col_dim),
            coords={
                row_dim: updated_ds[row_dim].values,
                col_dim: updated_ds[col_dim].values,
            },
        )
        for dim_name in original_da.dims:
            if dim_name in (row_dim, col_dim):
                continue
            updated_da = updated_da.expand_dims(
                {dim_name: updated_ds[dim_name].values}, axis=0
            )
        updated_da = updated_da.transpose(*original_da.dims)
        updated_da = updated_da.astype(original_da.dtype, copy=False)
        updated_da.attrs = original_da.attrs.copy()
        updated_ds[RUN_MASK_VARIABLE] = updated_da

        temp_path = self.retry_run_mask_path.with_suffix(".tmp.nc")
        try:
            updated_ds.to_netcdf(temp_path.as_posix(), engine="netcdf4")
            temp_path.replace(self.retry_run_mask_path)
        finally:
            updated_ds.close()
            if temp_path.exists():
                temp_path.unlink()

    def _rewrite_retry_config(self) -> None:
        with open(self.retry_config_path, "r", encoding="utf-8") as file_obj:
            config_data = json.load(file_obj)

        source_path_options = {
            self.batch_path.as_posix(),
            str(self.batch_path),
            self.batch_path.resolve().as_posix(),
        }
        destination_path = self.retry_path.resolve().as_posix()

        def replace_paths(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {key: replace_paths(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [replace_paths(value) for value in obj]
            if isinstance(obj, str):
                updated = obj
                for source_path in source_path_options:
                    updated = updated.replace(source_path, destination_path)
                batch_match = re.search(r"batch_(\d+)", self.batch_path.name)
                if batch_match:
                    batch_index = batch_match.group(1)
                    updated = re.sub(
                        rf"/tmp/batch[_-]{batch_index}\b",
                        f"/tmp/batch_{batch_index}_retry",
                        updated,
                    )
                return updated
            return obj

        updated_data = replace_paths(config_data)
        with open(self.retry_config_path, "w", encoding="utf-8") as file_obj:
            json.dump(updated_data, file_obj, indent=4)

    def _rewrite_retry_slurm_runner(self) -> None:
        with open(self.retry_slurm_path, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()

        content = content.replace(
            self.batch_path.as_posix(), self.retry_path.resolve().as_posix()
        )

        # Ensure retry jobs are distinguishable in the scheduler.
        content = re.sub(
            r'(#SBATCH\s+--job-name="?)([^"\n]+)("?)(\n)',
            lambda match: (
                f"{match.group(1)}{match.group(2)}-retry{match.group(3)}{match.group(4)}"
                if not match.group(2).endswith("-retry")
                else match.group(0)
            ),
            content,
        )

        content = re.sub(
            r"^#SBATCH\s+(-p\s+\S+|--partition(?:=\S+|\s+\S+))\s*$",
            f"#SBATCH -p {self._args.partition}",
            content,
            flags=re.MULTILINE,
        )
        if not re.search(r"^#SBATCH\s+(-p|--partition)\b", content, flags=re.MULTILINE):
            lines = content.splitlines()
            insertion_index = 1 if lines and lines[0].startswith("#!") else 0
            lines.insert(insertion_index, f"#SBATCH -p {self._args.partition}")
            content = "\n".join(lines) + "\n"

        def replace_log_path(match: re.Match[str]) -> str:
            old_log_path = match.group(1)
            if old_log_path.endswith("-retry"):
                return match.group(0)
            return f"#SBATCH -o {old_log_path}-retry"

        content = re.sub(r"^#SBATCH -o\s+(.+)$", replace_log_path, content, flags=re.MULTILINE)

        with open(self.retry_slurm_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(content)

    def _submit_retry_job(self) -> None:
        submit_result = submit_job(self.retry_slurm_path.as_posix())
        if submit_result.returncode != 0:
            raise RuntimeError(
                f"Failed to submit retry job ({submit_result.returncode}). "
                f"stdout: {submit_result.stdout} stderr: {submit_result.stderr}"
            )
        combined_output = (
            (submit_result.stdout or "") + "\n" + (submit_result.stderr or "")
        ).strip()
        job_id = extract_sbatch_job_id(combined_output)
        if job_id:
            print(f"Submitted retry batch job: {job_id}")
        else:
            print("Submitted retry batch job.")
        if combined_output:
            print(combined_output)

    def execute(self) -> None:
        self._validate_batch_structure()
        (
            retry_run_mask,
            row_dim,
            col_dim,
            enabled_cells,
            completed_cells,
            failed_cells,
        ) = self._build_retry_run_mask()

        print(f"Batch: {self.batch_path}")
        print(
            f"Active cells: {enabled_cells}; completed cells: {completed_cells}; "
            f"cells to rerun: {failed_cells}"
        )
        if failed_cells == 0:
            print("No incomplete cells detected. Nothing to rerun.")
            return

        self._copy_batch_to_retry()
        self._write_retry_run_mask(retry_run_mask, row_dim=row_dim, col_dim=col_dim)
        self._rewrite_retry_config()
        self._rewrite_retry_slurm_runner()
        print(f"Retry batch prepared at {self.retry_path}")

        if getattr(self._args, "submit", True):
            self._submit_retry_job()
        else:
            print("Retry batch prepared without submission (--no-submit).")
