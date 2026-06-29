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

  def example_wiemip_split_command(self, batches_path: str = "/mnt/exacloud/$USER/my_split") -> str:
    return (
      "bp batch wiemip_split "
      f"-i {self.input_path.as_posix()} "
      f"-b {batches_path} "
      f"--nbatches {self.suggested_nbatches} "
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


def _apply_run_mask_filters(
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
  cells_per_batch, predicted_batches, blocks = suggest_cells_per_batch(
    run_data, y_size, x_size, target_batches
  )
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
      cells_per_batch, predicted_batches, blocks = suggest_cells_per_batch(
        run_data, y_size, x_size, revised_target
      )
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
    "Recommended for `bp batch split`:",
    f"  --cells-per-batch {plan.suggested_cells_per_batch}",
    f"  predicted batches:     {plan.predicted_batches}",
    f"  cells/batch (avg/min/max): {plan.avg_cells_per_batch:.1f} / "
    f"{plan.min_cells_per_batch} / {plan.max_cells_per_batch}",
    f"  cells/MPI rank (avg):    {plan.avg_cells_per_batch / max(1, plan.mpi_ranks):.1f}",
    "",
    "Recommended for `bp batch wiemip_split`:",
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
      "",
      "Submit (default: all jobs queued in Slurm immediately):",
      "  bp batch run -b <split_path>",
      "",
      "Optional throttling if startup storms are a problem:",
      "  bp batch run -b <split_path> --throttle --max-concurrent 16 --max-queue-depth 32",
    ]
  )
  return "\n".join(lines)
