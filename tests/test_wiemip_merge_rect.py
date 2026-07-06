"""Tests for WIEMIP rect canvas merge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from batch_processing.cmd.batch.wiemip_merge import WiemipMergeCommand
from batch_processing.utils.wiemip_processing import (
    SPLIT_METADATA_FILENAME,
    SPLIT_MODE_RECT,
    WiemipSplitMetadata,
    write_batch_layout,
    write_split_metadata,
)


def _write_batch_output(
    batch_dir: Path,
    *,
    output_name: str,
    values: np.ndarray,
) -> None:
    output_dir = batch_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    y_size, x_size = values.shape
    ds = xr.Dataset(
        data_vars={"GPP": (("y", "x"), values.astype(np.float32))},
        coords={"y": np.arange(y_size), "x": np.arange(x_size)},
    )
    ds.to_netcdf(output_dir / output_name, engine="netcdf4")


def test_rect_canvas_merge_reconstructs_cropped_grid(tmp_path: Path, monkeypatch):
    split_root = tmp_path / "split"
    split_root.mkdir()
    input_root = tmp_path / "input"
    input_root.mkdir()
    run_mask = input_root / "run-mask.nc"
    xr.Dataset(
        data_vars={"run": (("Y", "X"), np.ones((4, 8), dtype=int))},
        coords={"Y": np.arange(4), "X": np.arange(8)},
    ).to_netcdf(run_mask)

    blocks = [(0, 2, 0, 4), (0, 2, 4, 8), (2, 4, 0, 8)]
    write_batch_layout(split_root / "batch_layout.json", blocks=blocks, grid_y=4, grid_x=8)
    metadata = WiemipSplitMetadata(
        schema_version=2,
        original_input_path=input_root.as_posix(),
        filtered_staging_path=(split_root / "_staging").as_posix(),
        run_mask_filename="run-mask.nc",
        row_dim="Y",
        col_dim="X",
        active_value=1,
        full_rows=4,
        full_cols=8,
        active_bbox={
            "row_start": 0,
            "row_end": 3,
            "col_start": 0,
            "col_end": 7,
        },
        file_mappings={},
        split_mode=SPLIT_MODE_RECT,
        blocks=[list(block) for block in blocks],
    )
    write_split_metadata(split_root / SPLIT_METADATA_FILENAME, metadata)

    expected = np.zeros((4, 8), dtype=np.float32)
    for batch_idx, block in enumerate(blocks):
        y0, y1, x0, x1 = block
        values = np.full((y1 - y0, x1 - x0), float(batch_idx + 1), dtype=np.float32)
        expected[y0:y1, x0:x1] = values
        batch_dir = split_root / f"batch_{batch_idx}"
        _write_batch_output(batch_dir, output_name="GPP_yearly_tr.nc", values=values)

    monkeypatch.setattr(
        "batch_processing.cmd.batch.wiemip_merge.BaseCommand.__init__",
        lambda self: None,
    )
    cmd = WiemipMergeCommand.__new__(WiemipMergeCommand)
    cmd.exacloud_user_dir = tmp_path
    cmd._args = type("Args", (), {"batches": split_root.name, "output_dir_name": "merged"})()
    cmd.base_batch_dir = split_root
    cmd.result_root_dir = split_root / "merged"
    cmd.filtered_result_dir = cmd.result_root_dir / "merged_filtered"
    cmd.restored_result_dir = cmd.result_root_dir / "merged_restored"
    cmd.filtered_result_dir.mkdir(parents=True, exist_ok=True)
    cmd.restored_result_dir.mkdir(parents=True, exist_ok=True)

    merged_path = cmd._merge_single_output_file_rect(
        "GPP_yearly_tr.nc",
        sorted((split_root).glob("batch_*")),
        blocks,
        4,
        8,
        cmd.filtered_result_dir,
    )
    assert merged_path is not None
    with xr.open_dataset(merged_path, decode_times=False) as merged:
        np.testing.assert_allclose(merged["GPP"].values, expected)
