"""Helpers to recommend WIEMIP / batch split parameters."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

Block = Tuple[int, int, int, int]


@dataclass
class SplitPlan:
  input_path: Path
  grid_y: int
  grid_x: int
  active_cells: int
  target_batches: int
  suggested_cells_per_batch: int
  predicted_batches: int
  avg_cells_per_batch: float
  min_cells_per_batch: int
  max_cells_per_batch: int
  suggested_nbatches: int
  active_y_rows: int
  mpi_ranks: int
  max_concurrent: int
  total_years: int
  estimated_hours_per_batch: Optional[float]
  estimated_total_hours: Optional[float]
  pilot_hours: Optional[float]
  pilot_cells: Optional[int]

  def example_split_command(self, batches_path: str = "/mnt/exacloud/$USER/my_split") -> str:
    return (
      "bp batch split "
      f"-i {self.input_path.as_posix()} "
      f"-b {batches_path} "
      f"--cells-per-batch {self.suggested_cells_per_batch} "
      f"-p ... -e ... -s ... -t ... -sp spot --mpi-ranks {self.mpi_ranks}"
    )

  def example_wiemip_split_command(
    self,
    batches_path: str = "/mnt/exacloud/$USER/my_split",
    *,
    split_mode: str = "y-stripe",
  ) -> str:
    mode_flag = ""
    if split_mode == "rect":
      mode_flag = " --split-mode rect"
    return (
      "bp batch wiemip_split "
      f"-i {self.input_path.as_posix()} "
      f"-b {batches_path} "
      f"--cells-per-batch {self.suggested_cells_per_batch}"
      f"{mode_flag} "
      f"-p ... -e ... -s ... -t ... -sp spot --mpi-ranks {self.mpi_ranks}"
    )


def _resolve_run_mask_path(input_path: Path) -> Path:
  if input_path.is_file():
    return input_path
  candidate = input_path / "run-mask.nc"
  if not candidate.is_file():
    raise FileNotFoundError(f"run-mask.nc not found under {input_path}")
  return candidate


def _load_run_mask_array(input_path: Path) -> Tuple[np.ndarray, int, int]:
  run_mask_path = _resolve_run_mask_path(input_path)
  with xr.open_dataset(run_mask_path, decode_times=False) as ds:
    if "run" not in ds:
      raise KeyError(f"{run_mask_path} does not contain variable 'run'")
    run_da = ds["run"]
    while run_da.ndim > 2:
      run_da = run_da.isel({run_da.dims[0]: 0}, drop=True)
    if run_da.ndim != 2:
      raise ValueError(f"Expected 2D run mask in {run_mask_path}, got dims {run_da.dims}")
    y_dim, x_dim = run_da.dims
    y_size = int(run_da.sizes[y_dim])
    x_size = int(run_da.sizes[x_dim])
    run_values = np.asarray(run_da.values, dtype=float)
  return run_values, y_size, x_size


def apply_run_mask_filters(
  run_data: np.ndarray,
  input_path: Path,
  *,
  cmt0_filter: bool,
  no_max_cmt: bool,
  max_cmt: int,
) -> np.ndarray:
  filtered = np.where(np.isnan(run_data), 0, run_data)
  veg_path = input_path / "vegetation.nc" if input_path.is_dir() else input_path.parent / "vegetation.nc"
  if not (cmt0_filter or not no_max_cmt) or not veg_path.is_file():
    return filtered

  with xr.open_dataset(veg_path, decode_times=False) as veg_ds:
    if "veg_class" not in veg_ds:
      return filtered
    veg_data = np.asarray(veg_ds["veg_class"].values)
    while veg_data.ndim > 2:
      veg_data = veg_data.take(0, axis=0)
    if cmt0_filter:
      filtered = np.where(veg_data == 0, 0, filtered)
    if not no_max_cmt:
      filtered = np.where(veg_data > max_cmt, 0, filtered)
  return filtered


# Backward-compatible alias used internally.
_apply_run_mask_filters = apply_run_mask_filters


def active_cell_mask(run_data: np.ndarray) -> np.ndarray:
  return np.isfinite(run_data) & np.isclose(run_data, 1.0)


def plan_y_ranges_by_active_cells(
  run_data: np.ndarray,
  target_active_cells: int,
) -> List[Tuple[int, int]]:
  """Return half-open Y row ranges [start, end) with ~target active cells each.

  Rows are never split across batches (Y-stripe constraint). A batch may exceed
  ``target_active_cells`` when a single row contains more active cells than the target.
  """
  if target_active_cells < 1:
    raise ValueError("target_active_cells must be >= 1")
  if run_data.ndim != 2:
    raise ValueError(f"Expected 2D run mask, got shape {run_data.shape}")

  active = active_cell_mask(run_data)
  total_active = int(np.sum(active))
  if total_active == 0:
    raise ValueError("run-mask contains no active cells after filters (run == 1)")

  n_rows, _ = run_data.shape
  active_per_row = np.sum(active, axis=1)

  ranges: List[Tuple[int, int]] = []
  batch_start = 0
  accum = 0
  for row in range(n_rows):
    accum += int(active_per_row[row])
    if accum >= target_active_cells and batch_start < row + 1:
      ranges.append((batch_start, row + 1))
      batch_start = row + 1
      accum = 0

  if batch_start < n_rows:
    trailing_active = int(np.sum(active[batch_start:, :]))
    if trailing_active > 0:
      ranges.append((batch_start, n_rows))

  if not ranges:
    raise ValueError("run-mask contains no active cells after filters (run == 1)")

  return ranges


def plan_y_stripe_blocks_by_active_cells(
  run_data: np.ndarray,
  target_active_cells: int,
) -> List[Block]:
  """Full-width Y stripes balanced by filtered active cell count."""
  if run_data.ndim != 2:
    raise ValueError(f"Expected 2D run mask, got shape {run_data.shape}")
  _, n_cols = run_data.shape
  y_ranges = plan_y_ranges_by_active_cells(run_data, target_active_cells)
  return [(y_start, y_end, 0, n_cols) for y_start, y_end in y_ranges]


def _active_in_block(run_data: np.ndarray, block: Block) -> int:
  y_start, y_end, x_start, x_end = block
  subset = run_data[y_start:y_end, x_start:x_end]
  return int(np.sum(active_cell_mask(subset)))


def _merge_two_blocks(a: Block, b: Block) -> Block:
  return (
    min(a[0], b[0]),
    max(a[1], b[1]),
    min(a[2], b[2]),
    max(a[3], b[3]),
  )


def _blocks_are_adjacent(a: Block, b: Block) -> bool:
  ay0, ay1, ax0, ax1 = a
  by0, by1, bx0, bx1 = b
  if ay0 == by0 and ay1 == by1:
    return ax1 == bx0 or bx1 == ax0
  if ax0 == bx0 and ax1 == bx1:
    return ay1 == by0 or by1 == ay0
  return False


def _merge_blocks_below_min(
  blocks: Sequence[Block],
  run_data: np.ndarray,
  min_active_cells: int,
) -> List[Block]:
  if min_active_cells <= 1 or len(blocks) <= 1:
    return list(blocks)

  out: List[Block] = list(blocks)
  while len(out) >= 2 and _active_in_block(run_data, out[-1]) < min_active_cells:
    last = out.pop()
    prev = out.pop()
    if not _blocks_are_adjacent(prev, last):
      out.append(prev)
      out.append(last)
      break
    out.append(_merge_two_blocks(prev, last))
  return out


def plan_rect_blocks_by_active_cells(
  run_data: np.ndarray,
  target_active_cells: int,
  *,
  min_active_cells: int = 1,
) -> List[Block]:
  """Return half-open rectangles with ~target active cells by splitting Y and X.

  Dense latitude rows may be divided across multiple X slices. Rectangles are
  contiguous in row-major order over the cropped grid.
  """
  if target_active_cells < 1:
    raise ValueError("target_active_cells must be >= 1")
  if min_active_cells < 1:
    raise ValueError("min_active_cells must be >= 1")
  if run_data.ndim != 2:
    raise ValueError(f"Expected 2D run mask, got shape {run_data.shape}")

  active = active_cell_mask(run_data)
  if int(np.sum(active)) == 0:
    raise ValueError("run-mask contains no active cells after filters (run == 1)")

  n_rows, n_cols = run_data.shape
  blocks: List[Block] = []
  y = 0
  while y < n_rows:
    while y < n_rows and not np.any(active[y, :]):
      y += 1
    if y >= n_rows:
      break

    y0 = y
    band = active[y0 : y0 + 1, :]
    y_end = y0 + 1
    while y_end < n_rows:
      next_row = active[y_end : y_end + 1, :]
      if not np.any(next_row):
        y_end += 1
        continue
      trial = np.vstack([band, next_row])
      if int(np.sum(trial)) <= target_active_cells or int(np.sum(band)) == 0:
        band = trial
        y_end += 1
      else:
        break

    x = 0
    while x < n_cols:
      while x < n_cols and not np.any(band[:, x]):
        x += 1
      if x >= n_cols:
        break
      x0 = x
      accum = 0
      x1 = x0
      while x1 < n_cols:
        col_cells = int(np.sum(band[:, x1]))
        if col_cells == 0:
          x1 += 1
          continue
        if accum + col_cells > target_active_cells and accum > 0:
          break
        accum += col_cells
        x1 += 1
      if x1 <= x0:
        x1 = min(n_cols, x0 + 1)
      block = (y0, y_end, x0, x1)
      if _active_in_block(run_data, block) > 0:
        blocks.append(block)
      x = x1
    y = y_end

  if not blocks:
    raise ValueError("run-mask contains no active cells after filters (run == 1)")

  return _merge_blocks_below_min(blocks, run_data, min_active_cells)


def summarize_active_cells_per_block(
  blocks: Sequence[Block],
  run_data: np.ndarray,
) -> Tuple[float, int, int]:
  counts = cells_in_blocks(blocks, run_data)
  if not counts:
    return 0.0, 0, 0
  return float(np.mean(counts)), int(min(counts)), int(max(counts))


def generate_blocks(y_size: int, x_size: int, cells_per_batch: int) -> List[Block]:
  cx = min(x_size, int(math.sqrt(cells_per_batch)))
  cy = min(y_size, max(1, cells_per_batch // cx))
  blocks: List[Block] = []
  for y_start in range(0, y_size, cy):
    y_end = min(y_size, y_start + cy)
    for x_start in range(0, x_size, cx):
      x_end = min(x_size, x_start + cx)
      blocks.append((y_start, y_end, x_start, x_end))
  return blocks


def filter_active_blocks(blocks: Sequence[Block], run_data: np.ndarray) -> List[Block]:
  active_blocks: List[Block] = []
  for y_start, y_end, x_start, x_end in blocks:
    subset = run_data[y_start:y_end, x_start:x_end]
    if np.any(np.isclose(subset, 1)):
      active_blocks.append((y_start, y_end, x_start, x_end))
  return active_blocks


def count_active_cells(run_data: np.ndarray) -> int:
  active = np.isfinite(run_data) & np.isclose(run_data, 1)
  return int(np.sum(active))


def count_active_y_rows(run_data: np.ndarray) -> int:
  active = np.isfinite(run_data) & np.isclose(run_data, 1)
  return int(np.sum(np.any(active, axis=1)))


def cells_in_blocks(blocks: Sequence[Block], run_data: np.ndarray) -> List[int]:
  counts: List[int] = []
  for y_start, y_end, x_start, x_end in blocks:
    subset = run_data[y_start:y_end, x_start:x_end]
    counts.append(int(np.sum(np.isfinite(subset) & np.isclose(subset, 1))))
  return counts


def predict_batch_count(
  run_data: np.ndarray,
  y_size: int,
  x_size: int,
  cells_per_batch: int,
) -> Tuple[int, List[Block]]:
  blocks = filter_active_blocks(
    generate_blocks(y_size, x_size, cells_per_batch),
    run_data,
  )
  return len(blocks), blocks


def suggest_cells_per_batch(
  run_data: np.ndarray,
  y_size: int,
  x_size: int,
  target_batches: int,
) -> Tuple[int, int, List[Block]]:
  active_cells = count_active_cells(run_data)
  if active_cells == 0:
    raise ValueError("run-mask contains no active cells (run == 1)")
  if target_batches < 1:
    raise ValueError("target_batches must be >= 1")

  target_batches = min(target_batches, active_cells)
  lo, hi = 1, active_cells
  best_cells = 1
  best_batches = active_cells
  best_blocks: List[Block] = []

  while lo <= hi:
    mid = (lo + hi) // 2
    batch_count, blocks = predict_batch_count(run_data, y_size, x_size, mid)
    if abs(batch_count - target_batches) <= abs(best_batches - target_batches):
      best_cells = mid
      best_batches = batch_count
      best_blocks = blocks
    if batch_count > target_batches:
      lo = mid + 1
    else:
      hi = mid - 1

  return best_cells, best_batches, best_blocks


def estimate_batch_walltime_hours(
  *,
  pilot_hours: float,
  pilot_cells: int,
  pilot_years: int,
  batch_cells: float,
  total_years: int,
) -> float:
  if pilot_hours <= 0 or pilot_cells <= 0 or pilot_years <= 0:
    raise ValueError("pilot_hours, pilot_cells, and pilot_years must be positive")
  hours_per_cell_year = pilot_hours / (pilot_cells * pilot_years)
  return hours_per_cell_year * batch_cells * total_years


def count_active_cells_from_batch_dir(batch_dir: Path) -> int:
  run_mask_path = batch_dir / "input" / "run-mask.nc"
  if not run_mask_path.is_file():
    raise FileNotFoundError(f"Missing batch run-mask: {run_mask_path}")
  run_data, _, _ = _load_run_mask_array(run_mask_path)
  return count_active_cells(run_data)


def build_split_plan(
  input_path: str | Path,
  *,
  target_batches: int = 100,
  target_walltime_hours: Optional[float] = None,
  mpi_ranks: int = 8,
  total_years: int = 0,
  pilot_batch_dir: Optional[str | Path] = None,
  pilot_hours: Optional[float] = None,
  pilot_cells_override: Optional[int] = None,
  cmt0_filter: bool = False,
  no_max_cmt: bool = False,
  max_cmt: int = 74,
  max_concurrent: int = 16,
) -> SplitPlan:
  resolved_input = Path(input_path).expanduser().resolve()
  run_data, y_size, x_size = _load_run_mask_array(resolved_input)
  if resolved_input.is_file():
    resolved_input = resolved_input.parent
  run_data = _apply_run_mask_filters(
    run_data,
    resolved_input,
    cmt0_filter=cmt0_filter,
    no_max_cmt=no_max_cmt,
    max_cmt=max_cmt,
  )

  active_cells = count_active_cells(run_data)
  active_y_rows = count_active_y_rows(run_data)
  cells_per_batch = max(1, int(round(active_cells / target_batches)))
  blocks = plan_y_stripe_blocks_by_active_cells(run_data, cells_per_batch)
  predicted_batches = len(blocks)
  per_batch_counts = cells_in_blocks(blocks, run_data)
  avg_cells = float(np.mean(per_batch_counts)) if per_batch_counts else 0.0
  min_cells = int(min(per_batch_counts)) if per_batch_counts else 0
  max_cells = int(max(per_batch_counts)) if per_batch_counts else 0

  suggested_nbatches = min(target_batches, max(1, active_y_rows))
  pilot_cells: Optional[int] = None
  estimated_hours: Optional[float] = None
  estimated_total: Optional[float] = None

  if pilot_cells_override is not None:
    pilot_cells = int(pilot_cells_override)
  elif pilot_batch_dir is not None:
    pilot_cells = count_active_cells_from_batch_dir(Path(pilot_batch_dir))
  if pilot_hours is not None and pilot_cells is not None and total_years > 0:
    estimated_hours = estimate_batch_walltime_hours(
      pilot_hours=pilot_hours,
      pilot_cells=pilot_cells,
      pilot_years=total_years,
      batch_cells=avg_cells,
      total_years=total_years,
    )
    estimated_total = estimated_hours * predicted_batches / max(1, max_concurrent)

  if target_walltime_hours is not None and estimated_hours is not None:
    if estimated_hours > target_walltime_hours * 1.25:
      scale = estimated_hours / target_walltime_hours
      revised_target = max(1, int(round(predicted_batches * scale)))
      cells_per_batch = max(1, int(round(active_cells / revised_target)))
      blocks = plan_y_stripe_blocks_by_active_cells(run_data, cells_per_batch)
      predicted_batches = len(blocks)
      per_batch_counts = cells_in_blocks(blocks, run_data)
      avg_cells = float(np.mean(per_batch_counts)) if per_batch_counts else 0.0
      min_cells = int(min(per_batch_counts)) if per_batch_counts else 0
      max_cells = int(max(per_batch_counts)) if per_batch_counts else 0
      suggested_nbatches = min(revised_target, max(1, active_y_rows))
      estimated_hours = estimate_batch_walltime_hours(
        pilot_hours=pilot_hours,
        pilot_cells=pilot_cells,
        pilot_years=total_years,
        batch_cells=avg_cells,
        total_years=total_years,
      )
      estimated_total = estimated_hours * predicted_batches / max(1, max_concurrent)

  return SplitPlan(
    input_path=resolved_input,
    grid_y=y_size,
    grid_x=x_size,
    active_cells=active_cells,
    target_batches=target_batches,
    suggested_cells_per_batch=cells_per_batch,
    predicted_batches=predicted_batches,
    avg_cells_per_batch=avg_cells,
    min_cells_per_batch=min_cells,
    max_cells_per_batch=max_cells,
    suggested_nbatches=suggested_nbatches,
    active_y_rows=active_y_rows,
    mpi_ranks=mpi_ranks,
    max_concurrent=max_concurrent,
    total_years=total_years,
    estimated_hours_per_batch=estimated_hours,
    estimated_total_hours=estimated_total,
    pilot_hours=pilot_hours,
    pilot_cells=pilot_cells,
  )


def format_split_plan_report(plan: SplitPlan, *, batches_path: str) -> str:
  lines = [
    "[SPLIT PLAN]",
    f"  input:                 {plan.input_path}",
    f"  grid:                  Y={plan.grid_y}, X={plan.grid_x}",
    f"  active cells:          {plan.active_cells}",
    f"  target batches:        {plan.target_batches}",
    "",
    "Recommended for `bp batch split` and `bp batch wiemip_split`:",
    f"  --cells-per-batch {plan.suggested_cells_per_batch}",
    f"  predicted batches (y-stripe): {plan.predicted_batches}",
    f"  cells/batch (avg/min/max): {plan.avg_cells_per_batch:.1f} / "
    f"{plan.min_cells_per_batch} / {plan.max_cells_per_batch}",
    f"  cells/MPI rank (avg):    "
    f"{plan.avg_cells_per_batch / max(1, plan.mpi_ranks):.1f}",
    "",
    "For faster walltime on dense latitude rows, try rect split:",
    f"  bp batch wiemip_split ... --split-mode rect "
    f"--cells-per-batch {plan.suggested_cells_per_batch}",
    "",
    "Legacy equal-row split (`bp batch wiemip_split` only):",
    f"  --nbatches {plan.suggested_nbatches}  "
    f"(active Y rows={plan.active_y_rows})",
  ]

  if plan.pilot_hours is not None and plan.pilot_cells is not None:
    if plan.pilot_cells < max(4, plan.mpi_ranks):
      lines.append(
        f"  [WARN] pilot batch has only {plan.pilot_cells} active cell(s); "
        "timing estimate may be unreliable. Use --pilot-cells to override."
      )
    if plan.estimated_hours_per_batch is not None:
      lines.extend(
        [
          "",
          "Pilot-based walltime estimate:",
          f"  pilot:                 {plan.pilot_hours:.2f} h for "
          f"{plan.pilot_cells} cells over {plan.total_years} model years",
          f"  est. batch walltime:     {plan.estimated_hours_per_batch:.2f} h",
        ]
      )
      if plan.estimated_total_hours is not None:
        lines.append(
          f"  est. experiment time:    {plan.estimated_total_hours:.1f} h "
          f"({plan.predicted_batches} batches, max_concurrent={plan.max_concurrent})"
        )

  lines.extend(
    [
      "",
      "Example commands:",
      f"  {plan.example_split_command(batches_path)}",
      f"  {plan.example_wiemip_split_command(batches_path)}",
      f"  {plan.example_wiemip_split_command(batches_path, split_mode='rect')}",
      "",
      "Submit (default: all jobs queued in Slurm immediately):",
      "  bp batch run -b <split_path>",
      "",
      "Optional throttling if startup storms are a problem:",
      "  bp batch run -b <split_path> --throttle --max-concurrent 16 --max-queue-depth 32",
    ]
  )
  return "\n".join(lines)
