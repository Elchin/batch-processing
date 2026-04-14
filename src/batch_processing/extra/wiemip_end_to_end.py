#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import xarray as xr

DEFAULT_INPUT_PATH = (
    "/mnt/exacloud/ejafarov_woodwellclimate_org/wiemip/setup_GFDL-ESM4"
)
DEFAULT_SPLIT_PATH = (
    "/mnt/exacloud/ejafarov_woodwellclimate_org/wiemip/test_gfdl_split"
)
DEFAULT_PLOT_SCRIPT = os.path.expanduser(
    "~/Circumpolar_TEM_aux_scripts/plot_nc_all_files.py"
)
DEFAULT_SPLIT_PARTITION = ""
RUN_MASK_VAR = "run"
RUN_STATUS_VAR = "run_status"
RUN_SUCCESS_VALUE = 100
RUN_ENABLED_VALUE = 1
ROW_DIM_CANDIDATES = ("Y", "y", "latitude", "lat")
COL_DIM_CANDIDATES = ("X", "x", "longitude", "lon")
BATCH_DIR_PATTERN = re.compile(r"^batch_(\d+)$")


def normalize_path(path_str: str) -> Path:
    expanded = os.path.expanduser(path_str.strip())
    if expanded.startswith("mnt/"):
        expanded = f"/{expanded}"
    return Path(os.path.abspath(expanded))


def run_cmd(command: Sequence[str], dry_run: bool = False) -> subprocess.CompletedProcess | None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"[RUN] {printable}")
    if dry_run:
        return None
    return subprocess.run(command, check=True, text=True, capture_output=True)


def get_spatial_dims(dim_names: Iterable[str]) -> Tuple[str, str]:
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


def determine_nbatches(input_path: Path) -> int:
    run_mask_path = input_path / "run-mask.nc"
    if not run_mask_path.exists():
        raise FileNotFoundError(f"Missing run-mask.nc at {run_mask_path}")

    with xr.open_dataset(run_mask_path, decode_times=False) as ds:
        if RUN_MASK_VAR not in ds:
            raise KeyError(f"{run_mask_path} does not contain '{RUN_MASK_VAR}'")
        run_mask_da = to_2d_spatial_array(ds[RUN_MASK_VAR], run_mask_path.as_posix())
        run_values = np.asarray(run_mask_da.values)

    active_mask = np.isfinite(run_values) & np.isclose(run_values, RUN_ENABLED_VALUE)
    if not np.any(active_mask):
        raise ValueError("run-mask contains no active cells (run == 1)")

    active_rows = np.any(active_mask, axis=1)
    nbatches = int(np.sum(active_rows))
    if nbatches <= 0:
        raise ValueError("Calculated nbatches is 0, expected at least 1")
    return nbatches


def get_batch_dirs(split_path: Path) -> List[Path]:
    if not split_path.exists():
        return []
    batch_dirs = [
        path
        for path in split_path.iterdir()
        if path.is_dir() and BATCH_DIR_PATTERN.match(path.name)
    ]
    return sorted(batch_dirs, key=lambda path: int(BATCH_DIR_PATTERN.match(path.name).group(1)))


def batch_id_from_path(batch_path: Path) -> int:
    match = BATCH_DIR_PATTERN.match(batch_path.name)
    if not match:
        raise ValueError(f"Invalid batch directory name: {batch_path.name}")
    return int(match.group(1))


def count_active_cells(run_mask_path: Path) -> int:
    with xr.open_dataset(run_mask_path, decode_times=False) as ds:
        if RUN_MASK_VAR not in ds:
            raise KeyError(f"{run_mask_path} missing '{RUN_MASK_VAR}'")
        run_da = to_2d_spatial_array(ds[RUN_MASK_VAR], run_mask_path.as_posix())
        run_values = np.asarray(run_da.values)
    active = np.isfinite(run_values) & np.isclose(run_values, RUN_ENABLED_VALUE)
    return int(np.sum(active))


def count_completed_cells(run_status_path: Path) -> int:
    with xr.open_dataset(run_status_path, decode_times=False) as ds:
        if RUN_STATUS_VAR not in ds:
            raise KeyError(f"{run_status_path} missing '{RUN_STATUS_VAR}'")
        status_da = to_2d_spatial_array(ds[RUN_STATUS_VAR], run_status_path.as_posix())
        status_values = np.asarray(status_da.values)
    completed = np.isfinite(status_values) & np.isclose(status_values, RUN_SUCCESS_VALUE)
    return int(np.sum(completed))


def collect_incomplete_batches(split_path: Path) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
    incomplete: List[int] = []
    progress: Dict[int, Tuple[int, int]] = {}
    batch_dirs = get_batch_dirs(split_path)
    for batch_dir in batch_dirs:
        batch_id = batch_id_from_path(batch_dir)
        run_mask_path = batch_dir / "input" / "run-mask.nc"
        run_status_path = batch_dir / "output" / "run_status.nc"

        if not run_mask_path.exists():
            print(f"[WARN] Missing run-mask for batch_{batch_id}: {run_mask_path}")
            incomplete.append(batch_id)
            continue

        n_cells = count_active_cells(run_mask_path)
        if not run_status_path.exists():
            print(f"[WARN] Missing run_status for batch_{batch_id}: {run_status_path}")
            progress[batch_id] = (0, n_cells)
            incomplete.append(batch_id)
            continue

        m_cells = count_completed_cells(run_status_path)
        progress[batch_id] = (m_cells, n_cells)
        if m_cells < n_cells:
            incomplete.append(batch_id)
    return sorted(incomplete), progress


def wait_for_jobs(
    split_path: Path,
    batch_ids: Sequence[int],
    poll_seconds: int,
    retry_jobs: bool = False,
    dry_run: bool = False,
) -> None:
    if not batch_ids:
        print("[WAIT] No batch ids provided, skipping queue wait.")
        return

    split_name = split_path.name
    expected_names = {
        f"{split_name}-batch-{batch_id}{'-retry' if retry_jobs else ''}"
        for batch_id in batch_ids
    }
    print(
        f"[WAIT] Monitoring {len(expected_names)} job names "
        f"({'retry' if retry_jobs else 'initial'} pass)."
    )
    if dry_run:
        print("[WAIT] Dry-run mode, skipping queue polling.")
        return

    user = os.getenv("USER")
    if not user:
        raise EnvironmentError("USER environment variable is not set.")

    while True:
        result = subprocess.run(
            ["squeue", "-h", "-u", user, "-o", "%A|%j|%T"],
            check=True,
            text=True,
            capture_output=True,
        )
        active_names = set()
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            job_name = parts[1].strip()
            if job_name in expected_names:
                active_names.add(job_name)

        if not active_names:
            print("[WAIT] No matching jobs in queue. Continuing.")
            return

        print(
            f"[WAIT] {len(active_names)} matching jobs still running/pending. "
            f"Next check in {poll_seconds} seconds."
        )
        time.sleep(poll_seconds)


def format_incomplete_progress(
    incomplete_ids: Sequence[int], progress: Dict[int, Tuple[int, int]]
) -> str:
    chunks = []
    for batch_id in incomplete_ids:
        m_cells, n_cells = progress.get(batch_id, (0, 0))
        chunks.append(f"batch_{batch_id} ({m_cells}/{n_cells})")
    return ", ".join(chunks)


def run_rerun_pass(
    split_path: Path,
    batch_ids: Sequence[int],
    pass_index: int,
    poll_seconds: int,
    dry_run: bool,
) -> None:
    if not batch_ids:
        print(f"[PASS {pass_index}] No incomplete batches. Skipping rerun pass.")
        return

    print(f"[PASS {pass_index}] Rerunning {len(batch_ids)} incomplete batches.")
    before_counts: Dict[int, int] = {}
    for batch_id in batch_ids:
        batch_path = split_path / f"batch_{batch_id}"
        run_status_path = batch_path / "output" / "run_status.nc"
        if run_status_path.exists():
            before_counts[batch_id] = count_completed_cells(run_status_path)
        else:
            before_counts[batch_id] = 0
        run_cmd(["bp", "batch", "wiemip_re-run", batch_path.as_posix()], dry_run=dry_run)

    wait_for_jobs(
        split_path=split_path,
        batch_ids=batch_ids,
        retry_jobs=True,
        poll_seconds=poll_seconds,
        dry_run=dry_run,
    )

    for batch_id in batch_ids:
        batch_path = split_path / f"batch_{batch_id}"
        run_cmd(
            ["bp", "batch", "wiemip_rerun_merge", batch_path.as_posix()],
            dry_run=dry_run,
        )

    if dry_run:
        return

    for batch_id in batch_ids:
        batch_path = split_path / f"batch_{batch_id}"
        run_status_path = batch_path / "output" / "run_status.nc"
        if not run_status_path.exists():
            continue
        after_count = count_completed_cells(run_status_path)
        if after_count < before_counts[batch_id]:
            raise RuntimeError(
                f"Completion regressed for batch_{batch_id}: "
                f"{after_count} < {before_counts[batch_id]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WIEMIP end-to-end automation (split -> run -> rerun passes -> merge -> plot)."
    )
    parser.add_argument(
        "--input",
        "--input-path",
        dest="input_path",
        default=DEFAULT_INPUT_PATH,
        help=f"WIEMIP input setup directory (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--split",
        "--split-path",
        dest="split_path",
        default=DEFAULT_SPLIT_PATH,
        help=f"WIEMIP split output directory (default: {DEFAULT_SPLIT_PATH})",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Queue polling interval in seconds (default: 300).",
    )
    parser.add_argument(
        "--plot-script",
        default=DEFAULT_PLOT_SCRIPT,
        help=f"Plot script path (default: {DEFAULT_PLOT_SCRIPT})",
    )
    parser.add_argument(
        "-sp",
        "--slurm-partition",
        default=DEFAULT_SPLIT_PARTITION,
        help=(
            "Optional split partition/node type for `bp batch wiemip_split` "
            "(examples: dask, spot, compute)."
        ),
    )
    parser.add_argument(
        "-p",
        type=int,
        default=10,
        help="PRE-RUN years for split setup (default: 10).",
    )
    parser.add_argument(
        "-e",
        type=int,
        default=10,
        help="EQUILIBRIUM years for split setup (default: 10).",
    )
    parser.add_argument(
        "-s",
        type=int,
        default=10,
        help="SPINUP years for split setup (default: 10).",
    )
    parser.add_argument(
        "-t",
        type=int,
        default=10,
        help="TRANSIENT years for split setup (default: 10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and skip execution.",
    )
    args = parser.parse_args()

    input_path = normalize_path(args.input_path)
    split_path = normalize_path(args.split_path)
    plot_script = normalize_path(args.plot_script)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not (input_path / "run-mask.nc").exists():
        raise FileNotFoundError(f"Input run-mask missing: {input_path / 'run-mask.nc'}")
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be >= 1")

    print(f"[INFO] Input path: {input_path}")
    print(f"[INFO] Split path: {split_path}")

    # Step 0: Determine nbatches from active bbox row count.
    nbatches = determine_nbatches(input_path)
    max_batch_id = nbatches - 1
    print(f"[STEP 0] Computed nbatches={nbatches} (max batch id: {max_batch_id})")

    # Step 1: WIEMIP split.
    split_cmd = [
        "bp",
        "batch",
        "wiemip_split",
        "-i",
        input_path.as_posix(),
        "-b",
        split_path.as_posix(),
        "-N",
        str(nbatches),
        "--restart_from",
        "",
        "-p",
        str(args.p),
        "-e",
        str(args.e),
        "-s",
        str(args.s),
        "-t",
        str(args.t),
    ]
    if args.slurm_partition:
        split_cmd.extend(["-sp", args.slurm_partition])
    run_cmd(split_cmd, dry_run=args.dry_run)

    # Step 2: Submit all batches.
    run_cmd(["bp", "batch", "run", "-b", split_path.as_posix()], dry_run=args.dry_run)

    # Step 3: Wait until this run's batch jobs are out of queue.
    expected_initial_ids = list(range(nbatches))
    wait_for_jobs(
        split_path=split_path,
        batch_ids=expected_initial_ids,
        retry_jobs=False,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("[INFO] Dry-run mode complete. No filesystem/queue checks beyond this point.")
        return

    # Step 4: Find incomplete batches.
    incomplete_ids, progress = collect_incomplete_batches(split_path)
    if incomplete_ids:
        print(
            "[STEP 4] Incomplete batches found: "
            + format_incomplete_progress(incomplete_ids, progress)
        )
    else:
        print("[STEP 4] No incomplete batches after initial run.")

    # Steps 5-8: first rerun pass.
    run_rerun_pass(
        split_path=split_path,
        batch_ids=incomplete_ids,
        pass_index=1,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
    )

    # Step 9: optional second rerun pass.
    incomplete_after_pass1, progress_after_pass1 = collect_incomplete_batches(split_path)
    if incomplete_after_pass1:
        print(
            "[STEP 9] Remaining incomplete after pass 1: "
            + format_incomplete_progress(incomplete_after_pass1, progress_after_pass1)
        )
        run_rerun_pass(
            split_path=split_path,
            batch_ids=incomplete_after_pass1,
            pass_index=2,
            poll_seconds=args.poll_seconds,
            dry_run=args.dry_run,
        )
    else:
        print("[STEP 9] No second rerun pass needed.")

    final_incomplete_ids, final_progress = collect_incomplete_batches(split_path)
    if final_incomplete_ids:
        print(
            "[WARN] Batches still incomplete after two rerun passes: "
            + format_incomplete_progress(final_incomplete_ids, final_progress)
        )
    else:
        print("[INFO] All batches complete after rerun passes.")

    # Step 10: final WIEMIP merge.
    run_cmd(["bp", "batch", "wiemip_merge", "-b", split_path.as_posix()], dry_run=False)

    # Step 11: plot merged outputs.
    merged_restored = split_path / "wiemip_merged" / "merged_restored"
    if not merged_restored.exists():
        raise FileNotFoundError(f"Merged restored output directory not found: {merged_restored}")
    if not plot_script.exists():
        raise FileNotFoundError(f"Plot script not found: {plot_script}")

    run_cmd(
        [sys.executable, plot_script.as_posix(), merged_restored.as_posix()],
        dry_run=False,
    )
    print("[DONE] WIEMIP end-to-end workflow finished.")


if __name__ == "__main__":
    main()
