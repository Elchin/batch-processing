"""Tests for active-cell-balanced rect block planning."""

from __future__ import annotations

import numpy as np

from batch_processing.utils.split_planning import (
    cells_in_blocks,
    plan_rect_blocks_by_active_cells,
    plan_y_stripe_blocks_by_active_cells,
    summarize_active_cells_per_block,
)


def test_plan_rect_blocks_caps_dense_row():
    run_values = np.zeros((5, 12), dtype=float)
    run_values[2, :] = 1
    target = 5
    blocks = plan_rect_blocks_by_active_cells(run_values, target)
    counts = cells_in_blocks(blocks, run_values)
    assert max(counts) <= target
    assert sum(counts) == 12
    assert len(blocks) == 3


def test_plan_rect_blocks_respects_min_active_cells():
    run_values = np.zeros((4, 4), dtype=float)
    run_values[0, 0] = 1
    run_values[0, 1] = 1
    run_values[3, 3] = 1
    blocks = plan_rect_blocks_by_active_cells(run_values, target_active_cells=1, min_active_cells=2)
    counts = cells_in_blocks(blocks, run_values)
    assert sum(counts) == 3
    assert min(counts) >= 1


def test_rect_split_is_more_even_than_y_stripe_on_dense_row():
    run_values = np.zeros((3, 20), dtype=float)
    run_values[1, :] = 1
    target = 8
    rect_blocks = plan_rect_blocks_by_active_cells(run_values, target)
    stripe_blocks = plan_y_stripe_blocks_by_active_cells(run_values, target)
    _, _, rect_max = summarize_active_cells_per_block(rect_blocks, run_values)
    _, _, stripe_max = summarize_active_cells_per_block(stripe_blocks, run_values)
    assert rect_max < stripe_max
