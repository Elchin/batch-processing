#!/usr/bin/env python3
"""Audit WIEMIP split batch status and print recovery commands."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from batch_processing.utils.utils import BatchSubmitOptions, mpirun_rank_flags, submit_batch_jobs

RUN_MASK_VAR = "run"
RUN_STATUS_VAR = "run_status"
RUN_SUCCESS_VALUE = 100
RUN_ENABLED_VALUE = 1
BATCH_DIR_PATTERN = re.compile(r"^batch_(\d+)$")
ROW_DIM_CANDIDATES = ("Y", "y", "latitude", "lat")
COL_DIM_CANDIDATES = ("X", "x", "longitude", "lon")


def normalize_path(path_str: str) -> Path:
    expanded = path_str.strip()
    if expanded.startswith("mnt/"):
        expanded = f"/{expanded}"
    return Path(expanded).expanduser().resolve()


def get_spatial_dims(dim_names) -> Tuple[str, str]:
    dim_set = set(dim_names)
    for row_dim in ROW_DIM_CANDIDATES:
        if row_dim not in dim_set:
            continue
        for col_dim in COL_DIM_CANDIDATES:
            if col_dim in dim_set:
                return row_dim, col_dim
    raise ValueError(f"Could not detect row/col dimensions from {tuple(dim_names)}")


def to_2d_spatial_array(data_array: xr.DataArray, source_label: str) -> xr.DataArray:
    row_dim, col_dim = get_spatial_dims(data_array.dims)
    array_2d = data_array
    for dim_name in list(array_2d.dims):
        if dim_name in (row_dim, col_dim):
            continue
        if int(array_2d.sizes[dim_name]) != 1:
            raise ValueError(
                f"{source_label} contains non-singleton extra dimension "
                f"{dim_name}={int(array_2d.sizes[dim_name])}"
            )
        array_2d = array_2d.isel({dim_name: 0}, drop=True)
    return array_2d.transpose(row_dim, col_dim)


def get_batch_dirs(split_path: Path) -> List[Path]:
    batch_dirs = [
        path
        for path in split_path.iterdir()
        if path.is_dir() and BATCH_DIR_PATTERN.match(path.name)
    ]
    return sorted(batch_dirs, key=lambda path: int(BATCH_DIR_PATTERN.match(path.name).group(1)))


def count_active_cells(run_mask_path: Path) -> int:
    with xr.open_dataset(run_mask_path, decode_times=False) as ds:
        run_da = to_2d_spatial_array(ds[RUN_MASK_VAR], run_mask_path.as_posix())
        run_values = np.asarray(run_da.values)
    active = np.isfinite(run_values) & np.isclose(run_values, RUN_ENABLED_VALUE)
    return int(np.sum(active))


def count_completed_cells(run_status_path: Path) -> int:
    with xr.open_dataset(run_status_path, decode_times=False) as ds:
        status_da = to_2d_spatial_array(ds[RUN_STATUS_VAR], run_status_path.as_posix())
        status_values = np.asarray(status_da.values)
    completed = np.isfinite(status_values) & np.isclose(status_values, RUN_SUCCESS_VALUE)
    return int(np.sum(completed))


def audit_split(split_path: Path) -> Dict[str, List[int]]:
    complete: List[int] = []
    inactive: List[int] = []
    incomplete: List[int] = []
    missing_status: List[int] = []
    empty_status: List[int] = []
    bad_slurm: List[int] = []

    for batch_dir in get_batch_dirs(split_path):
        batch_id = int(BATCH_DIR_PATTERN.match(batch_dir.name).group(1))
        run_mask_path = batch_dir / "input" / "run-mask.nc"
        run_status_path = batch_dir / "output" / "run_status.nc"
        slurm_path = batch_dir / "slurm_runner.sh"

        if slurm_path.is_file():
            slurm_text = slurm_path.read_text(encoding="utf-8")
            if (
                "mpirun" in slurm_text
                and "--use-hwthread-cpus" not in slurm_text
                and "mpirun -n" not in slurm_text
            ):
                bad_slurm.append(batch_id)

        if not run_mask_path.is_file():
            incomplete.append(batch_id)
            continue

        n_cells = count_active_cells(run_mask_path)
        if n_cells == 0:
            inactive.append(batch_id)
            continue

        if not run_status_path.is_file():
            missing_status.append(batch_id)
            incomplete.append(batch_id)
            continue
        if run_status_path.stat().st_size == 0:
            empty_status.append(batch_id)
            incomplete.append(batch_id)
            continue

        m_cells = count_completed_cells(run_status_path)
        if m_cells >= n_cells:
            complete.append(batch_id)
        else:
            incomplete.append(batch_id)

    return {
        "complete": complete,
        "inactive": inactive,
        "incomplete": sorted(set(incomplete)),
        "missing_status": missing_status,
        "empty_status": empty_status,
        "bad_slurm": bad_slurm,
    }


def clean_batch_outputs(split_path: Path, batch_ids: List[int], dry_run: bool) -> None:
    for batch_id in batch_ids:
        output_dir = split_path / f"batch_{batch_id}" / "output"
        if not output_dir.exists():
            continue
        print(f"[CLEAN] {output_dir}")
        if dry_run:
            continue
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()


def patch_slurm_scripts(
    split_path: Path,
    partition: str,
    mpi_ranks: Optional[int],
    dry_run: bool,
) -> int:
    rank_flags = mpirun_rank_flags(mpi_ranks)
    patched = 0
    for batch_dir in get_batch_dirs(split_path):
        slurm_path = batch_dir / "slurm_runner.sh"
        if not slurm_path.is_file():
            continue
        text = slurm_path.read_text(encoding="utf-8")
        original = text
        text = re.sub(r"#SBATCH -p \S+", f"#SBATCH -p {partition}", text, count=1)
        if re.search(r"mpirun (?:-n \d+|--use-hwthread-cpus) ", text):
            text = re.sub(
                r"mpirun (?:-n \d+|--use-hwthread-cpus) ",
                f"mpirun {rank_flags} ",
                text,
                count=1,
            )
        else:
            text = re.sub(
                r"mpirun -x HDF5_USE_FILE_LOCKING",
                f"mpirun {rank_flags} -x HDF5_USE_FILE_LOCKING",
                text,
                count=1,
            )
        if text != original:
            patched += 1
            print(f"[PATCH] {slurm_path}")
            if not dry_run:
                slurm_path.write_text(text, encoding="utf-8")
    return patched


def submit_batches(
    split_path: Path,
    batch_ids: List[int],
    dry_run: bool,
    throttle: bool = False,
    max_concurrent: int = 16,
    max_queue_depth: int = 32,
    submit_delay: float = 0.25,
    poll_interval: int = 30,
) -> None:
    script_paths = []
    for batch_id in batch_ids:
        script = split_path / f"batch_{batch_id}" / "slurm_runner.sh"
        if script.is_file():
            script_paths.append(script)
        else:
            print(f"[WARN] missing slurm script: {script}")

    submit_batch_jobs(
        script_paths,
        BatchSubmitOptions(
            submit_all=not throttle,
            max_concurrent=max_concurrent,
            max_queue_depth=max_queue_depth,
            submit_delay_seconds=submit_delay,
            poll_interval_seconds=poll_interval,
            dry_run=dry_run,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        required=True,
        help="Path to WIEMIP split directory (e.g. .../stable_split_veg1_new)",
    )
    parser.add_argument(
        "--action",
        choices=("audit", "clean", "patch-slurm", "submit"),
        default="audit",
        help="audit=report only; clean=remove failed batch outputs; patch-slurm=fix MPI/partition; submit=sbatch in chunks",
    )
    parser.add_argument("--partition", default="compute", help="Slurm partition for patch-slurm")
    parser.add_argument(
        "--mpi-ranks",
        type=int,
        default=None,
        help=(
            "Optional explicit MPI rank count for patch-slurm (mpirun -n N). "
            "If omitted, uses mpirun --use-hwthread-cpus."
        ),
    )
    parser.add_argument(
        "--throttle",
        action="store_true",
        help="Pause submission while the Slurm queue is full (default: submit all immediately)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="With --throttle: max running jobs before pausing submission",
    )
    parser.add_argument(
        "--max-queue-depth",
        type=int,
        default=32,
        help="With --throttle: max RUNNING+PENDING jobs before pausing submission",
    )
    parser.add_argument(
        "--submit-delay",
        type=float,
        default=0.25,
        help="Seconds to sleep between sbatch calls",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="With --throttle: seconds between queue checks",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying/submitting")
    args = parser.parse_args()

    split_path = normalize_path(args.split)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split path not found: {split_path}")

    report = audit_split(split_path)
    total_batches = len(get_batch_dirs(split_path))
    print(f"[AUDIT] split={split_path}")
    print(f"  total batches:     {total_batches}")
    print(f"  complete:          {len(report['complete'])}")
    print(f"  inactive (0 cells):{len(report['inactive'])}")
    print(f"  incomplete:        {len(report['incomplete'])}")
    print(f"  missing run_status:{len(report['missing_status'])}")
    print(f"  empty run_status:  {len(report['empty_status'])}")
    print(f"  bad slurm (no rank flags): {len(report['bad_slurm'])}")

    if report["incomplete"]:
        preview = ", ".join(f"batch_{i}" for i in report["incomplete"][:20])
        suffix = f" (+{len(report['incomplete']) - 20} more)" if len(report["incomplete"]) > 20 else ""
        print(f"  incomplete ids:    {preview}{suffix}")

    if args.action == "audit":
        if report["bad_slurm"]:
            print("\n[RECOMMEND] Patch slurm scripts (partition and/or MPI rank flags):")
            print(
                f"  python {Path(__file__).name} --split {split_path} "
                f"--action patch-slurm --partition compute"
            )
        if report["incomplete"]:
            print("\n[RECOMMEND] Recovery sequence:")
            print(
                f"  1) python {Path(__file__).name} --split {split_path} "
                f"--action patch-slurm --partition compute"
            )
            print(
                f"     (add --mpi-ranks 1 to force single-rank jobs if needed)"
            )
            print(
                f"  2) python {Path(__file__).name} --split {split_path} "
                f"--action clean"
            )
            print(
                f"  3) python {Path(__file__).name} --split {split_path} "
                f"--action submit"
            )
        return

    if args.action == "clean":
        clean_batch_outputs(split_path, report["incomplete"], dry_run=args.dry_run)
        return

    if args.action == "patch-slurm":
        patched = patch_slurm_scripts(
            split_path,
            partition=args.partition,
            mpi_ranks=args.mpi_ranks,
            dry_run=args.dry_run,
        )
        print(f"[PATCH] updated {patched} slurm_runner.sh files")
        return

    if args.action == "submit":
        submit_batches(
            split_path,
            report["incomplete"],
            dry_run=args.dry_run,
            throttle=args.throttle,
            max_concurrent=max(1, args.max_concurrent),
            max_queue_depth=max(1, args.max_queue_depth),
            submit_delay=max(0.0, args.submit_delay),
            poll_interval=max(1, args.poll_interval),
        )


if __name__ == "__main__":
    main()
