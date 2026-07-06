"""Tests for WIEMIP rect split metadata and batch layout helpers."""

from __future__ import annotations

import json
from pathlib import Path

from batch_processing.utils.wiemip_processing import (
    BATCH_LAYOUT_FILENAME,
    SPLIT_MODE_RECT,
    SPLIT_MODE_Y_STRIPE,
    WiemipSplitMetadata,
    read_batch_layout,
    write_batch_layout,
    write_split_metadata,
)


def test_metadata_roundtrip_with_rect_blocks(tmp_path: Path):
    metadata = WiemipSplitMetadata(
        schema_version=2,
        original_input_path="/tmp/input",
        filtered_staging_path="/tmp/staging",
        run_mask_filename="run-mask.nc",
        row_dim="Y",
        col_dim="X",
        active_value=1,
        full_rows=10,
        full_cols=20,
        active_bbox={
            "row_start": 1,
            "row_end": 5,
            "col_start": 2,
            "col_end": 15,
        },
        file_mappings={"run-mask.nc": "run-mask.nc"},
        split_mode=SPLIT_MODE_RECT,
        blocks=[[0, 2, 0, 8], [2, 5, 0, 14]],
    )
    path = tmp_path / "wiemip_split_metadata.json"
    write_split_metadata(path, metadata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split_mode"] == SPLIT_MODE_RECT
    assert payload["blocks"] == [[0, 2, 0, 8], [2, 5, 0, 14]]


def test_write_and_read_batch_layout(tmp_path: Path):
    blocks = [(0, 3, 0, 10), (3, 6, 10, 20)]
    layout_path = tmp_path / BATCH_LAYOUT_FILENAME
    write_batch_layout(layout_path, blocks=blocks, grid_y=6, grid_x=20)
    loaded_blocks, grid_y, grid_x = read_batch_layout(layout_path)
    assert loaded_blocks == blocks
    assert grid_y == 6
    assert grid_x == 20


def test_metadata_defaults_to_y_stripe():
    metadata = WiemipSplitMetadata(
        schema_version=1,
        original_input_path="/tmp/input",
        filtered_staging_path="/tmp/staging",
        run_mask_filename="run-mask.nc",
        row_dim="Y",
        col_dim="X",
        active_value=1,
        full_rows=10,
        full_cols=20,
        active_bbox={
            "row_start": 0,
            "row_end": 9,
            "col_start": 0,
            "col_end": 19,
        },
        file_mappings={},
    )
    assert metadata.split_mode == SPLIT_MODE_Y_STRIPE
    assert metadata.uses_rect_split is False
