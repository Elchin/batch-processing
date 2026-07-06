"""Tests for split planning recommendations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from batch_processing.utils.split_planning import (
    apply_run_mask_filters,
    build_split_plan,
    count_active_cells,
    plan_y_ranges_by_active_cells,
    plan_y_stripe_blocks_by_active_cells,
    predict_batch_count,
    suggest_cells_per_batch,
    summarize_active_cells_per_block,
)


def _write_run_mask(path: Path, run_values: np.ndarray) -> None:
    y_size, x_size = run_values.shape
    ds = xr.Dataset(
        data_vars={"run": (("Y", "X"), run_values)},
        coords={"Y": np.arange(y_size), "X": np.arange(x_size)},
    )
    ds.to_netcdf(path)


def test_suggest_cells_per_batch_reduces_batch_count(tmp_path: Path):
    run_values = np.zeros((20, 40), dtype=float)
    run_values[5:15, 10:30] = 1
    _write_run_mask(tmp_path / "run-mask.nc", run_values)

    small_batches, _, _ = suggest_cells_per_batch(run_values, 20, 40, target_batches=200)
    large_batches, _, _ = suggest_cells_per_batch(run_values, 20, 40, target_batches=20)

    assert large_batches > small_batches
    assert predict_batch_count(run_values, 20, 40, large_batches)[0] <= 25


def test_build_split_plan_from_input_dir(tmp_path: Path):
    run_values = np.zeros((123, 720), dtype=float)
    run_values[30:90, 100:600] = 1
    input_dir = tmp_path / "setup"
    input_dir.mkdir()
    _write_run_mask(input_dir / "run-mask.nc", run_values)

    plan = build_split_plan(
        input_dir,
        target_batches=100,
        mpi_ranks=8,
        total_years=2450,
    )

    assert plan.active_cells == count_active_cells(run_values)
    assert 50 <= plan.predicted_batches <= 150
    assert plan.suggested_cells_per_batch >= 32
    assert plan.avg_cells_per_batch / plan.mpi_ranks >= 2
    assert plan.suggested_cells_per_batch >= 32


def test_plan_y_ranges_by_active_cells_balances_sparse_rows():
    run_values = np.zeros((8, 2), dtype=float)
    run_values[0:4, :] = 1
    run_values[4:8, :] = 1

    ranges = plan_y_ranges_by_active_cells(run_values, target_active_cells=5)
    assert ranges == [(0, 3), (3, 6), (6, 8)]

    blocks = plan_y_stripe_blocks_by_active_cells(run_values, target_active_cells=5)
    avg, min_c, max_c = summarize_active_cells_per_block(blocks, run_values)
    assert min_c >= 4
    assert max_c <= 6
    assert avg == 16 / 3


def test_apply_run_mask_filters_with_vegetation(tmp_path: Path):
    run_values = np.ones((4, 4), dtype=float)
    input_dir = tmp_path / "setup"
    input_dir.mkdir()
    _write_run_mask(input_dir / "run-mask.nc", run_values)

    veg = np.full((4, 4), 50, dtype=np.int16)
    veg[0, 0] = 0
    veg[1, 1] = 80
    veg_ds = xr.Dataset(
        data_vars={"veg_class": (("Y", "X"), veg)},
        coords={"Y": np.arange(4), "X": np.arange(4)},
    )
    veg_ds.to_netcdf(input_dir / "vegetation.nc")

    filtered = apply_run_mask_filters(
        run_values,
        input_dir,
        cmt0_filter=True,
        no_max_cmt=False,
        max_cmt=74,
    )
    assert count_active_cells(filtered) == 14


def test_build_split_plan_with_pilot_timing(tmp_path: Path):
    run_values = np.ones((12, 12), dtype=float)
    input_dir = tmp_path / "setup"
    input_dir.mkdir()
    _write_run_mask(input_dir / "run-mask.nc", run_values)

    pilot_dir = tmp_path / "batch_0"
    (pilot_dir / "input").mkdir(parents=True)
    _write_run_mask(pilot_dir / "input" / "run-mask.nc", run_values[:6, :6])

    plan = build_split_plan(
        input_dir,
        target_batches=8,
        pilot_batch_dir=pilot_dir,
        pilot_hours=2.0,
        total_years=100,
        mpi_ranks=8,
        max_concurrent=4,
    )

    assert plan.pilot_cells == 36
    assert plan.estimated_hours_per_batch is not None
    assert plan.estimated_hours_per_batch > 0
    assert plan.estimated_total_hours is not None
