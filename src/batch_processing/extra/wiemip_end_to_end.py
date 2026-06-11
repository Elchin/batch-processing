#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import xarray as xr

WIEMIP_SPLIT_METADATA_FILENAME = "wiemip_split_metadata.json"
STAGING_INPUT_DIRNAME = "_wiemip_filtered_input"
OPTION2_EPILOG = """
Option 2 restart example (fresh setup masks + prior-split restart files):

  python wiemip_end_to_end.py \\
    --input /mnt/exacloud/$USER/wiemip/setup_stable \\
    --split /mnt/exacloud/$USER/wiemip/stable_split_veg1_restart \\
    --restart_from /mnt/exacloud/$USER/wiemip/stable_split_veg1 \\
    --restart_file restart-sp.nc \\
    --max-cmt 1 --cmt0-filter \\
    -sp dask -p 0 -e 0 -s 0 -t 20

Run-masks are built from --input via wiemip_split. --restart_from copies restart
NetCDFs only (not run-mask.nc or config.js from the source split).
"""


def _default_exacloud_wiemip_path(subpath: str) -> str:
    user = os.environ.get("USER", "YOURUSER")
    return f"/mnt/exacloud/{user}_woodwellclimate_org/wiemip/{subpath}"


DEFAULT_INPUT_PATH = _default_exacloud_wiemip_path("setup_GFDL-ESM4")
DEFAULT_SPLIT_PATH = _default_exacloud_wiemip_path("test_gfdl_split")
DEFAULT_PLOT_SCRIPT = os.path.expanduser(
    "~/Circumpolar_TEM_aux_scripts/plot_nc_all_files.py"
)
DEFAULT_SPLIT_PARTITION = ""
RUN_MASK_VAR = "run"


def _max_cmt_arg_type(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--max-cmt requires an integer (got {value!r})"
        ) from exc


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


def run_status_is_valid(run_status_path: Path) -> bool:
    """True when run_status.nc exists and is non-empty (valid NetCDF header)."""
    return run_status_path.is_file() and run_status_path.stat().st_size > 0


def count_completed_cells(run_status_path: Path) -> int:
    try:
        with xr.open_dataset(run_status_path, decode_times=False) as ds:
            if RUN_STATUS_VAR not in ds:
                raise KeyError(f"{run_status_path} missing '{RUN_STATUS_VAR}'")
            status_da = to_2d_spatial_array(ds[RUN_STATUS_VAR], run_status_path.as_posix())
            status_values = np.asarray(status_da.values)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Unreadable run_status file: {run_status_path} ({exc})") from exc
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
        if not run_status_is_valid(run_status_path):
            if run_status_path.exists():
                print(
                    f"[WARN] Invalid run_status for batch_{batch_id} "
                    f"(missing or empty): {run_status_path}"
                )
            else:
                print(f"[WARN] Missing run_status for batch_{batch_id}: {run_status_path}")
            progress[batch_id] = (0, n_cells)
            incomplete.append(batch_id)
            continue

        try:
            m_cells = count_completed_cells(run_status_path)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            progress[batch_id] = (0, n_cells)
            incomplete.append(batch_id)
            continue
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
    initial_grace_seconds: int = 120,
    stable_empty_polls: int = 2,
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

    start_time = time.time()
    empty_poll_streak = 0
    saw_active_jobs = False

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

        if active_names:
            saw_active_jobs = True
            empty_poll_streak = 0
            print(
                f"[WAIT] {len(active_names)} matching jobs still running/pending. "
                f"Next check in {poll_seconds} seconds."
            )
            time.sleep(poll_seconds)
            continue

        elapsed = time.time() - start_time
        if not saw_active_jobs and elapsed < initial_grace_seconds:
            print(
                f"[WAIT] No matching jobs in queue yet "
                f"({elapsed:.0f}s / {initial_grace_seconds}s grace). "
                f"Next check in {poll_seconds} seconds."
            )
            time.sleep(poll_seconds)
            continue

        empty_poll_streak += 1
        if empty_poll_streak >= stable_empty_polls:
            if saw_active_jobs:
                print("[WAIT] Queue clear after observing active jobs. Continuing.")
            else:
                print(
                    "[WARN] No matching jobs were observed in queue during grace period. "
                    "Jobs may have failed immediately or were never submitted. Continuing."
                )
            return

        print(
            f"[WAIT] Queue empty (check {empty_poll_streak}/{stable_empty_polls}). "
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


def _is_staging_input_path(path: Path) -> bool:
    """True if path looks like a prior split staging dir, not a fresh WIEMIP setup."""
    if path.name == STAGING_INPUT_DIRNAME:
        return True
    parent = path.parent
    metadata_file = parent / WIEMIP_SPLIT_METADATA_FILENAME
    if metadata_file.is_file() and get_batch_dirs(parent):
        return True
    return False


def _print_option2_restart_banner(
    input_path: Path,
    split_path: Path,
    restart_from_path: Path,
    restart_file: str,
    *,
    runmask_prefilter: bool,
    cmt0_filter: bool,
    no_max_cmt: bool,
    max_cmt: int,
) -> None:
    print("[INFO] Option 2 restart workflow:")
    print(f"  Run-masks + inputs: wiemip_split from --input ({input_path})")
    print(f"  Restart files only: --restart_from ({restart_from_path})")
    print(f"  New split output:   {split_path}")
    print(f"  Restart filename:   {restart_file}")
    print(
        "  CMT / climate prefilters during split: "
        f"runmask_prefilter={runmask_prefilter}, "
        f"cmt0_filter={cmt0_filter}, "
        f"max_cmt={'disabled' if no_max_cmt else max_cmt}"
    )


def _validate_option2_restart(
    input_path: Path,
    restart_from_path: Path,
    nbatches: int,
    restart_file: str,
) -> None:
    """Validate restart_from layout matches Option 2 expectations (fail-hard on batch count)."""
    source_batch_dirs = get_batch_dirs(restart_from_path)
    if not source_batch_dirs:
        raise FileNotFoundError(f"No batch_x dirs found under {restart_from_path}")

    source_count = len(source_batch_dirs)
    if source_count != nbatches:
        raise ValueError(
            f"--restart_from has {source_count} batch directories but --input run-mask "
            f"implies nbatches={nbatches}. Use the same setup and batch geometry for both, "
            f"or pick a matching source split."
        )

    if _is_staging_input_path(input_path):
        print(
            "[WARN] --input looks like a prior split staging directory "
            f"({input_path}). For Option 2, use a full WIEMIP setup directory "
            "(e.g. setup_stable with top-level run-mask.nc and vegetation.nc), "
            f"not {STAGING_INPUT_DIRNAME}."
        )

    missing_restart_batches: List[int] = []
    for src_batch_dir in source_batch_dirs:
        batch_id = batch_id_from_path(src_batch_dir)
        src_restart = src_batch_dir / "output" / restart_file
        if not src_restart.exists():
            missing_restart_batches.append(batch_id)

    if missing_restart_batches:
        preview = ", ".join(f"batch_{i}" for i in missing_restart_batches[:10])
        suffix = (
            f" (+{len(missing_restart_batches) - 10} more)"
            if len(missing_restart_batches) > 10
            else ""
        )
        print(
            f"[WARN] {len(missing_restart_batches)} source batch(es) missing "
            f"output/{restart_file}: {preview}{suffix}"
        )


def run_rerun_pass(
    split_path: Path,
    batch_ids: Sequence[int],
    pass_index: int,
    poll_seconds: int,
    dry_run: bool,
    initial_grace_seconds: int = 120,
) -> None:
    if not batch_ids:
        print(f"[PASS {pass_index}] No incomplete batches. Skipping rerun pass.")
        return

    print(f"[PASS {pass_index}] Rerunning {len(batch_ids)} incomplete batches.")
    before_counts: Dict[int, int] = {}
    for batch_id in batch_ids:
        batch_path = split_path / f"batch_{batch_id}"
        run_status_path = batch_path / "output" / "run_status.nc"
        if run_status_is_valid(run_status_path):
            try:
                before_counts[batch_id] = count_completed_cells(run_status_path)
            except ValueError:
                before_counts[batch_id] = 0
        else:
            before_counts[batch_id] = 0
        run_cmd(["bp", "batch", "wiemip_re-run", batch_path.as_posix()], dry_run=dry_run)

    wait_for_jobs(
        split_path=split_path,
        batch_ids=batch_ids,
        retry_jobs=True,
        poll_seconds=poll_seconds,
        dry_run=dry_run,
        initial_grace_seconds=initial_grace_seconds,
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
        description="WIEMIP end-to-end automation (split -> run -> rerun passes -> merge -> plot).",
        epilog=OPTION2_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "--input-path",
        dest="input_path",
        default=DEFAULT_INPUT_PATH,
        help=(
            "WIEMIP setup directory with top-level run-mask.nc and vegetation.nc "
            f"(default: {DEFAULT_INPUT_PATH}). Do not use a prior split's "
            f"{STAGING_INPUT_DIRNAME} folder."
        ),
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
        "--initial-grace-seconds",
        type=int,
        default=120,
        help=(
            "Seconds to keep polling before treating an empty queue as finished "
            "when no matching jobs were seen yet (default: 120)."
        ),
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
        "--runmask-prefilter",
        dest="runmask_prefilter",
        action="store_true",
        default=True,
        help=(
            "Enable WIEMIP split run-mask prefilter (default): disable active cells "
            "where required climate forcing vars are invalid."
        ),
    )
    parser.add_argument(
        "--no-runmask-prefilter",
        dest="runmask_prefilter",
        action="store_false",
        help="Disable WIEMIP split run-mask prefilter.",
    )
    parser.add_argument(
        "--cmt0-filter",
        dest="cmt0_filter",
        action="store_true",
        help=(
            "Enable WIEMIP split cmt0 run-mask prefilter: disable active cells "
            "where veg_class==0 in vegetation.nc."
        ),
    )
    parser.add_argument(
        "--no-cmt0-filter",
        dest="cmt0_filter",
        action="store_false",
        help="Disable WIEMIP split cmt0 run-mask prefilter (default).",
    )
    parser.set_defaults(cmt0_filter=False)
    parser.add_argument(
        "--no-max-cmt",
        dest="no_max_cmt",
        action="store_true",
        help=(
            "Disable WIEMIP split max-CMT run-mask prefilter "
            "(veg_class > N in vegetation.nc)."
        ),
    )
    parser.add_argument(
        "--max-cmt",
        dest="max_cmt",
        type=_max_cmt_arg_type,
        default=74,
        metavar="N",
        help=(
            "WIEMIP split: disable active cells where veg_class > N in vegetation.nc "
            "(runs after climate and --cmt0-filter prefilters). Default N is 74."
        ),
    )
    parser.add_argument(
        "--restart_from",
        "--restart-from",
        dest="restart_from",
        default=None,
        help=(
            "Prior split root with batch_x/ directories (Option 2 restart). Copies "
            "<restart_file> from each batch_x/output/ into the new split and sets "
            "IO.restart_from in config.js. Does not copy run-mask.nc or config.js "
            "from the source split; masks come from --input via wiemip_split."
        ),
    )
    parser.add_argument(
        "--restart_file",
        "--restart-file",
        dest="restart_file",
        default="restart-sp.nc",
        help="Filename of the restart file to copy from each source batch_x/output/ (default: restart-sp.nc).",
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
    if args.initial_grace_seconds < 0:
        raise ValueError("--initial-grace-seconds must be >= 0")
    restart_from_path: Path | None = None
    if args.restart_from:
        restart_from_path = normalize_path(args.restart_from)
        if not restart_from_path.exists():
            raise FileNotFoundError(f"--restart_from path does not exist: {restart_from_path}")

    print(f"[INFO] Input path: {input_path}")
    print(f"[INFO] Split path: {split_path}")

    # Step 0: Determine nbatches from active bbox row count.
    nbatches = determine_nbatches(input_path)
    max_batch_id = nbatches - 1
    print(f"[STEP 0] Computed nbatches={nbatches} (max batch id: {max_batch_id})")

    if restart_from_path is not None:
        _validate_option2_restart(
            input_path=input_path,
            restart_from_path=restart_from_path,
            nbatches=nbatches,
            restart_file=args.restart_file,
        )
        _print_option2_restart_banner(
            input_path=input_path,
            split_path=split_path,
            restart_from_path=restart_from_path,
            restart_file=args.restart_file,
            runmask_prefilter=args.runmask_prefilter,
            cmt0_filter=args.cmt0_filter,
            no_max_cmt=args.no_max_cmt,
            max_cmt=args.max_cmt,
        )

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
    if not args.runmask_prefilter:
        split_cmd.append("--no-runmask-prefilter")
    if args.cmt0_filter:
        split_cmd.append("--cmt0-filter")
    if args.no_max_cmt:
        split_cmd.append("--no-max-cmt")
    else:
        split_cmd.extend(["--max-cmt", str(args.max_cmt)])
    run_cmd(split_cmd, dry_run=args.dry_run)

    # Step 1.5: Seed restart files from source split run into new batch output dirs.
    if restart_from_path is not None:
        source_batch_dirs = get_batch_dirs(restart_from_path)
        print(
            f"[STEP 1.5] Restart-only: masks from split; IO.restart_from -> "
            f"{split_path}/batch_N/output/{args.restart_file}"
        )
        print(
            f"[STEP 1.5] Seeding '{args.restart_file}' from {restart_from_path} "
            f"({len(source_batch_dirs)} batches)"
        )
        seeded_count = 0
        missing_restart_count = 0
        for src_batch_dir in source_batch_dirs:
            batch_id = batch_id_from_path(src_batch_dir)
            src_restart = src_batch_dir / "output" / args.restart_file
            if not src_restart.exists():
                missing_restart_count += 1
                print(f"[WARN] Missing {args.restart_file} for batch_{batch_id}: {src_restart}")
                continue
            seeded_count += 1
            dst_batch_dir = split_path / f"batch_{batch_id}"
            dst_output_dir = dst_batch_dir / "output"
            dst_output_dir.mkdir(parents=True, exist_ok=True)
            dst_restart = dst_output_dir / args.restart_file
            if not args.dry_run:
                shutil.copy2(src_restart, dst_restart)
            config_file = dst_batch_dir / "config" / "config.js"
            if not config_file.exists():
                print(f"[WARN] config.js missing for batch_{batch_id}: {config_file}")
                continue
            if not args.dry_run:
                with open(config_file) as fh:
                    config_data = json.load(fh)
                config_data["IO"]["restart_from"] = dst_restart.as_posix()
                with open(config_file, "w") as fh:
                    json.dump(config_data, fh, indent=4)
            print(f"[RESTART] batch_{batch_id}: {src_restart} -> {dst_restart}")

        print(
            f"[STEP 1.5] Restart seed summary: {seeded_count} copied, "
            f"{missing_restart_count} missing source {args.restart_file}"
        )

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
        initial_grace_seconds=args.initial_grace_seconds,
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
        initial_grace_seconds=args.initial_grace_seconds,
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
            initial_grace_seconds=args.initial_grace_seconds,
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
