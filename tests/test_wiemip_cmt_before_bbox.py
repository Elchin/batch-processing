"""Tests for CMT-before-bbox wiemip_split staging semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from batch_processing.utils.split_planning import apply_run_mask_filters, count_active_cells
from batch_processing.utils.wiemip_processing import (
    RUN_ENABLED_VALUE,
    RUN_MASK_VARIABLE,
    compute_active_bbox,
)


def _write_run_mask(path: Path, run_values: np.ndarray) -> None:
    y_size, x_size = run_values.shape
    ds = xr.Dataset(
        data_vars={RUN_MASK_VARIABLE: (("Y", "X"), run_values)},
        coords={"Y": np.arange(y_size), "X": np.arange(x_size)},
    )
    ds.to_netcdf(path)


def test_cmt_filter_on_full_grid_then_crop_matches_vegetation_shape(tmp_path: Path):
    """CMT filters must run on full grid before bbox crop (123x720 vs 77x720)."""
    full_rows, full_cols = 123, 720
    run_values = np.zeros((full_rows, full_cols), dtype=float)
    run_values[10:87, 50:650] = RUN_ENABLED_VALUE

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_run_mask(input_dir / "run-mask.nc", run_values)

    veg = np.full((full_rows, full_cols), 50, dtype=np.int16)
    veg[20:30, 100:200] = 0
    veg[40:50, 300:400] = 80
    veg_ds = xr.Dataset(
        data_vars={"veg_class": (("Y", "X"), veg)},
        coords={"Y": np.arange(full_rows), "X": np.arange(full_cols)},
    )
    veg_ds.to_netcdf(input_dir / "vegetation.nc")

    filtered = apply_run_mask_filters(
        run_values,
        input_dir,
        cmt0_filter=True,
        no_max_cmt=False,
        max_cmt=74,
    )
    filtered_da = xr.DataArray(filtered, dims=("Y", "X"))
    bbox = compute_active_bbox(filtered_da, active_value=RUN_ENABLED_VALUE)

    cropped = filtered[
        bbox.row_start : bbox.row_end + 1,
        bbox.col_start : bbox.col_end + 1,
    ]
    assert cropped.shape[0] == bbox.n_rows
    assert cropped.shape[1] == bbox.n_cols
    assert count_active_cells(cropped) == count_active_cells(filtered)
    assert count_active_cells(cropped) < count_active_cells(run_values)
