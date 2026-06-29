import errno
import json
import os
import random
import re
import string
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from subprocess import CompletedProcess
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from dask_jobqueue import SLURMCluster
import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from google.cloud import storage
from netCDF4 import Dataset
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

INPUT_FILES = [
    "co2.nc",
    "projected-co2.nc",
    "drainage.nc",
    "fri-fire.nc",
    "run-mask.nc",
    "soil-texture.nc",
    "topo.nc",
    "vegetation.nc",
    "historic-explicit-fire.nc",
    "projected-explicit-fire.nc",
    "projected-climate.nc",
    "historic-climate.nc",
]

INPUT_FILES_TO_COPY = ["co2.nc", "projected-co2.nc"]

IO_PATHS = {
    "parameter_dir": "parameters/",
    "output_dir": "output/",
    "output_spec_file": "config/output_spec.csv",
    "runmask_file": "input/run-mask.nc",
    "hist_climate_file": "input/historic-climate.nc",
    "proj_climate_file": "input/projected-climate.nc",
    "veg_class_file": "input/vegetation.nc",
    "drainage_file": "input/drainage.nc",
    "soil_texture_file": "input/soil-texture.nc",
    "co2_file": "input/co2.nc",
    "proj_co2_file": "input/projected-co2.nc",
    "topo_file": "input/topo.nc",
    "fri_fire_file": "input/fri-fire.nc",
    "hist_exp_fire_file": "input/historic-explicit-fire.nc",
    "proj_exp_fire_file": "input/projected-explicit-fire.nc",
    "restart_from":"output/",
}


@dataclass
class Chunk:
    id: int
    start: int
    end: int


def create_chunks(total_size: int, num_chunks: int) -> List[Chunk]:
    """
    Create chunk boundaries for slicing the dataset.

    Parameters:
    total_size (int): The total size of the dimension to be chunked.
    num_chunks (int): The number of chunks to create.

    Returns:
    List[Chunk]: A list of Chunk instances, each containing the chunk index,
        start index, and end index.
    """
    if num_chunks <= 0:
        raise ValueError("num_chunks must be a positive integer")

    chunk_size = total_size // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else total_size
        chunks.append(Chunk(i, start, end))

    return chunks


def run_command(command: list) -> None:
    """Executes a shell command."""
    subprocess.run(command, check=True)


def mkdir_p(path: str) -> None:
    """Provides similar functionality to bash mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def remove_file(file: Union[str, list]):
    """Remove the specified file or list of files.

    Parameters:
        file (str or list): File path or list of file paths to be removed.

    Returns:
        None
    """
    if isinstance(file, str):
        os.remove(file)
        return

    if isinstance(file, list):
        _ = [os.remove(f) for f in file]


def download_directory(bucket_name: str, blob_name: str, output_path: str) -> None:
    """Downloads a directory from Google Cloud Storage.

    Args:
        bucket_name (str): Bucket name
        blob_name (str): The full path of the desired directory

    Example:
        Consider the below `gsutil URI`:

        gs://wcrc-tfstate-9486302/slurm-lustre-dvmdostem-v5/slurm-lustre-dvmdostem-v5/primary

        In the above URI, `wcrc-tfstate-9486302` is the bucket name and
        `slurm-lustre-dvmdostem-v5/slurm-lustre-dvmdostem-v5/primary` is the blob_name.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=blob_name)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        absolute_directory = f"{output_path}/{directory}"
        Path(absolute_directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(f"{output_path}/{blob.name}")


def download_file(bucket_name: str, blob_name: str, output_file_name: str) -> None:
    """
    Downloads a file from a Google Cloud Storage bucket to a local file.

    This function retrieves a blob from the specified bucket in Google Cloud Storage
    and downloads it to a local file. The local file is saved with the specified output
    file name.

    Parameters:
    - bucket_name (str): The name of the Google Cloud Storage bucket from which to
    download the file.
    - blob_name (str): The name of the blob (file) within the bucket to download.
    - output_file_name (str): The name (including path) under which the file should be
    saved locally.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    blob.download_to_filename(output_file_name)


def clean_and_load_json(input: str) -> dict:
    """
    Cleans comments from JSON-formatted string and loads it into a Python object.

    Args:
        input (str): Input JSON-formatted string possibly containing comments.

    Returns:
        dict: Python dictionary representing the JSON data.
    """
    cleaned_str = re.sub("//.*\n", "\n", input)
    json_data = json.loads(cleaned_str)
    return json_data


def get_slurm_queue(params: list = []) -> str:
    command = ["squeue", "--me", "--noheader"]
    command.extend(params)

    return subprocess.check_output(command).decode("utf-8")


def static_map(monthly_GPP_tr, monthly_GPP_sc, output, file_name):
    # Calculate the GPP means for 2000-2020
    a = (
        monthly_GPP_tr.sel(time=slice("2000", "2015"))
        .resample(time="YS")
        .sum(dim="time")
    )
    b = (
        monthly_GPP_sc.sel(time=slice("2016", "2020"))
        .resample(time="YS")
        .sum(dim="time")
    )
    gpp_mean_2000_2020 = xr.concat([a, b], dim="time")
    gpp_mean_2000_2020 = gpp_mean_2000_2020.mean(dim="time", keepdims=True)

    # Calculate the GPP means for 2040-2060 and 2080-2100
    gpp_mean_2040_2060 = (
        monthly_GPP_sc.sel(time=slice("2040", "2060"))
        .resample(time="YS")
        .sum(dim="time")
        .mean(dim="time")
    )
    gpp_mean_2080_2100 = (
        monthly_GPP_sc.sel(time=slice("2080", "2100"))
        .resample(time="YS")
        .sum(dim="time")
        .mean(dim="time")
    )

    # Create a plot with 3 subplots with uniform colorbars
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4), constrained_layout=True)
    vmin = np.min(
        [gpp_mean_2000_2020.min(), gpp_mean_2040_2060.min(), gpp_mean_2080_2100.min()]
    )
    vmax = np.max(
        [gpp_mean_2000_2020.max(), gpp_mean_2040_2060.max(), gpp_mean_2080_2100.max()]
    )

    # Plot the mean GPP value for each time period
    colormap = "YlGn"
    gpp_mean_2000_2020.plot(
        ax=axes[0], cmap=colormap, add_colorbar=False, vmin=vmin, vmax=vmax
    )
    gpp_mean_2040_2060.plot(
        ax=axes[1], cmap=colormap, add_colorbar=False, vmin=vmin, vmax=vmax
    )
    gpp_mean_2080_2100.plot(
        ax=axes[2], cmap=colormap, add_colorbar=False, vmin=vmin, vmax=vmax
    )

    # Add titles and labels to the subplots
    axes[0].set_title("2000-2020")
    axes[1].set_title("2040-2060")
    axes[2].set_title("2080-2100")
    # axes[0].set_ylabel('Y')
    # axes[1].set_ylabel('Y')
    # axes[2].set_ylabel('Y')
    # axes[0].set_xlabel('X')
    # axes[1].set_xlabel('X')
    # axes[2].set_xlabel('X')

    # Add a colorbar to the figure
    fig.colorbar(
        axes[2].collections[0],
        ax=axes,
        orientation="horizontal",
        label="Average Yearly Spatial " + output,
    )
    fig.suptitle(("Mean " + output), fontsize=20)

    plt.savefig(file_name)


def static_timeseries(data_tr, data_sc, output, type_var, type_spread, file_name):
    """
    output = 'GPP' or other variable of interest contained in dataframe
    type_var = 'mean' or 'sum'
    type_spread = 'std' or 'var'
    """
    plt.style.use("bmh")
    if type_spread == "std":
        spreadtext = "Standard Deviation"
    else:
        spreadtext = "Variance"

    # Convert the time coordinate to a regular datetime format
    data_tr["time"] = [
        cftime.datetime(t.year, t.month, t.day) for t in data_tr.time.values
    ]
    data_sc["time"] = [
        cftime.datetime(t.year, t.month, t.day) for t in data_sc.time.values
    ]

    # Group the data by year and compute the mean for each year
    a = data_tr.sel(time=slice("2000", "2015")).groupby("time.year").mean(dim="time")
    b = data_sc.groupby("time.year").mean(dim="time")
    annual_means = xr.concat([a, b], dim="time")
    # annual_means = monthly_GPP_sc.groupby('time.year').mean(dim='time')

    df = annual_means.to_dataframe().reset_index()

    # Group the data by year and calculate the sum and variance of GPP
    df_grouped = df.groupby("year").agg({output: [type_var, type_spread]}).reset_index()

    # Extract the sum and variance columns
    gpp_sum = df_grouped[output][
        type_var
    ]  # this is mean of gpp over all locations - do we want sum or mean??
    gpp_std = df_grouped[output][
        type_spread
    ]  # this is std of each year over all locations

    # Create the plot
    fig, ax = plt.subplots()

    # Add the shaded region for the variance
    y1 = gpp_sum - (gpp_std)
    y2 = gpp_sum + (gpp_std)
    ax.fill_between(
        df_grouped["year"],
        y1,
        y2,
        color="#fcaa0f",
        alpha=0.25,
        interpolate=True,
        label=spreadtext,
    )
    ax.plot(
        df_grouped["year"],
        gpp_sum,
        color="#9f2a63",
        label="Mean " + output + " over all locations/year",
    )
    # ax.plot(time, gpp_var, label="Standard Deviation")

    # Set the axis labels and title
    # ax.set_yscale('log')
    ax.set_xlabel("Time")
    ax.set_ylabel("Averaged " + output)
    ax.set_title(output + " over Time with " + spreadtext)
    ax.legend()
    plt.savefig(file_name)


def get_progress_bar():
    return Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )


def get_project_root() -> Path:
    """Returns the project root."""
    return Path(__file__).parent.parent


def interpret_path(path: str) -> str:
    """Converts any given relative path to an absolute path."""
    if path.startswith("gcs://"):
        return path

    path = os.path.expanduser(path)

    return os.path.abspath(path)


def generate_random_string(N=5):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=N))


def get_dimensions(file_name: str) -> Tuple[int, int]:
    """Retrieve the dimensions sizes from the given NetCDF file using netCDF4."""
    with Dataset(file_name, "r") as dataset:
        x = dataset.dimensions["X"].size
        y = dataset.dimensions["Y"].size
    return x, y


def get_batch_number(path: Union[Path, str]) -> int:
    """Returns the batch number from the given path.

    An example argument would be like this:

    /mnt/exacloud/dteber_woodwellclimate_org/output/batch_0/output/restart-eq.nc

    The return value for the above path is 0.
    """
    match_found = re.search(r"batch_(\d+)", str(path))
    return int(match_found.group(1)) if match_found else -1


def get_batch_folders(path: Path) -> List[Path]:
    """
    Find all folders that match the pattern 'batch_[integer]' in the given path.
    
    Args:
        path (Path): A Path object representing the directory to search in
        
    Returns:
        list: A list of Path objects for folders matching the pattern
    """
    if not isinstance(path, Path):
        path = Path(path)
    
    batch_folders = []
    for item in path.iterdir():
        if item.is_dir():
            batch_num = get_batch_number(item)
            if batch_num >= 0:  # Valid batch number found
                batch_folders.append(item)
    
    batch_folders.sort(key=get_batch_number)
    
    return batch_folders


def mpirun_rank_flags(mpi_ranks: int | None) -> str:
    """Return mpirun rank flags for slurm_runner.sh.

    Default (mpi_ranks is None): one rank per hardware thread.
    Explicit mpi_ranks: fixed ``mpirun -n N``.
    """
    if mpi_ranks is None:
        return "--use-hwthread-cpus"
    return f"-n {max(1, int(mpi_ranks))}"


def render_slurm_job_script(template_name: str, values: dict) -> str:
    """Reads the specified template file and populates it with the given values.

    Args:
        template_name (str): Name of the template file located in the templates folder
                             at the root of the project.
        values (dict): A dictionary of key-value pairs for substitution in the template.
                       Keys represent placeholders in the template, and values are the
                       corresponding substitution values.

    Returns:
        str: The populated job script ready to be submitted to Slurm.

    Raises:
        FileNotFoundError: If the specified template file does not exist.

    """
    template_path = get_project_root() / "templates" / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"{template_path} doesn't exist.")

    with open(template_path) as file:
        template = Template(file.read())

    return template.substitute(values)


def read_text_file(path: str) -> str:
    """Reads and returns the content of a text file.

    Args:
        path (str): The file system path to the text file to be read.

    Returns:
        str: The content of the file as a string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The given file is not found: {path}")

    with open(path) as file:
        content = file.read()

    return content


def read_json_file(path: str) -> dict:
    """Reads and returns the content of a JSON file.

    Args:
        path (str): The file system path to the JSON file to be read.

    Returns:
        dict: The content of the file as a dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The given file is not found: {path}")

    with open(path) as file:
        content = json.load(file)

    return content


def write_text_file(path: str, content: str) -> None:
    """A self-explanatory function

    Args:
        path (str): The file system path where the content should be written.
        content (str): The content to write to the file.

    Returns:
        None
    """
    with open(path, "w") as file:
        file.write(content)


def write_json_file(path: str, content: dict, indent: int = 4) -> None:
    """Writes a dictionary to a file in JSON format with specified indentation.

    Args:
        path (str): The file system path where the JSON content should be written.
        content (dict): A dictionary representing the JSON data to be written
            to the file.
        indent (int, optional): The number of spaces to use as indentation in the
            JSON file. Defaults to 4.

    Returns:
        None
    """
    with open(path, "w") as file:
        json.dump(content, file, indent=indent)


RUN_MASK_VAR = "run"
RUN_STATUS_VAR = "run_status"
RUN_SUCCESS_VALUE = 100
RUN_ENABLED_VALUE = 1
SLURM_ACTIVE_STATES = frozenset({"RUNNING", "CONFIGURING", "COMPLETING", "CG"})


@dataclass
class BatchSubmitOptions:
    """Controls Slurm submission for batch runs."""

    submit_all: bool = True
    max_concurrent: int = 16
    max_queue_depth: Optional[int] = 32
    submit_delay_seconds: float = 0.25
    poll_interval_seconds: int = 30
    skip_complete: bool = False
    dry_run: bool = False


def count_user_slurm_jobs(*, active_only: bool = True) -> int:
    """Return the number of jobs in the current user's Slurm queue."""
    user = os.getenv("USER")
    if not user:
        return 0

    result = subprocess.run(
        ["squeue", "-h", "-u", user, "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 0

    states = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not active_only:
        return len(states)
    return sum(1 for state in states if state in SLURM_ACTIVE_STATES)


def wait_for_submit_capacity(
    max_concurrent: int,
    max_queue_depth: Optional[int],
    poll_interval_seconds: int,
    *,
    dry_run: bool = False,
) -> None:
    """Block until there is room to submit another job.

    Limits concurrently *running* jobs (Lustre/MPI startup load) and optionally
    total queue depth (RUNNING + PENDING) so all jobs can be submitted without
    waiting for long-running simulations to finish.
    """
    if dry_run:
        return

    while True:
        running = count_user_slurm_jobs(active_only=True)
        queued = count_user_slurm_jobs(active_only=False)
        over_running = max_concurrent > 0 and running >= max_concurrent
        over_queue = max_queue_depth is not None and queued >= max_queue_depth
        if not over_running and not over_queue:
            return

        reasons = []
        if over_running:
            reasons.append(f"running={running}/{max_concurrent}")
        if over_queue:
            reasons.append(f"queued={queued}/{max_queue_depth}")
        print(
            f"[SUBMIT] At capacity ({', '.join(reasons)}); "
            f"waiting {poll_interval_seconds}s before next sbatch..."
        )
        time.sleep(poll_interval_seconds)


def wait_for_queue_capacity(
    max_concurrent: int,
    poll_interval_seconds: int,
    *,
    dry_run: bool = False,
) -> None:
    """Block until active job count is below max_concurrent."""
    wait_for_submit_capacity(
        max_concurrent,
        max_queue_depth=None,
        poll_interval_seconds=poll_interval_seconds,
        dry_run=dry_run,
    )


def wait_for_job_ids(
    job_ids: Sequence[str],
    poll_interval_seconds: int,
    *,
    dry_run: bool = False,
) -> None:
    """Wait until the given Slurm job ids are no longer in the queue."""
    remaining = {job_id for job_id in job_ids if job_id}
    if not remaining or dry_run:
        return

    while remaining:
        result = subprocess.run(
            ["squeue", "-h", "-j", ",".join(sorted(remaining)), "-o", "%A"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"[WARN] squeue check failed: {result.stderr.strip()}")
            return

        still_active = {
            line.strip() for line in result.stdout.splitlines() if line.strip()
        }
        remaining &= still_active
        if remaining:
            print(
                f"[SUBMIT] Waiting for {len(remaining)} job(s) to finish; "
                f"next check in {poll_interval_seconds}s..."
            )
            time.sleep(poll_interval_seconds)


def is_batch_complete(batch_dir: Path) -> bool:
    """True when all active cells in run-mask have run_status == 100."""
    run_mask_path = batch_dir / "input" / "run-mask.nc"
    run_status_path = batch_dir / "output" / "run_status.nc"
    if not run_mask_path.is_file() or not run_status_path.is_file():
        return False
    if run_status_path.stat().st_size == 0:
        return False

    try:
        with xr.open_dataset(run_mask_path, decode_times=False) as ds:
            if RUN_MASK_VAR not in ds:
                return False
            run_values = np.asarray(ds[RUN_MASK_VAR].values)
        active = int(
            np.sum(np.isfinite(run_values) & np.isclose(run_values, RUN_ENABLED_VALUE))
        )
        if active == 0:
            return True

        with xr.open_dataset(run_status_path, decode_times=False) as ds:
            if RUN_STATUS_VAR not in ds:
                return False
            status_values = np.asarray(ds[RUN_STATUS_VAR].values)
        completed = int(
            np.sum(
                np.isfinite(status_values)
                & np.isclose(status_values, RUN_SUCCESS_VALUE)
            )
        )
        return completed >= active
    except (OSError, ValueError, KeyError):
        return False


def submit_job(path: str) -> CompletedProcess:
    """Submits a job script to the Slurm workload manager using the `sbatch` command.

    Args:
        path (str): The file system path to the job script to be submitted.

    Returns:
        CompletedProcess: An object representing the completed process, containing
                          information about the execution of the `sbatch` command,
                          including stdout, stderr, and the return code.

    Raises:
        FileNotFoundError: If the specified job script file does not exist.
        subprocess.CalledProcessError: If the `sbatch` command fails.
    """
    batch_dir = os.path.dirname(path)
    command = ["sbatch", f"--chdir={batch_dir}", path]
    return subprocess.run(command, text=True, capture_output=True)


def submit_batch_jobs(
    script_paths: Iterable[Union[str, Path]],
    options: Optional[BatchSubmitOptions] = None,
) -> Tuple[int, int, int]:
    """Submit batch slurm_runner scripts.

    By default (submit_all=True) every job is sbatch'd immediately so Slurm can
    queue them. Set submit_all=False to pause submission while the queue is full.

    Returns:
        Tuple of (submitted_count, failed_count, skipped_count).
    """
    opts = options or BatchSubmitOptions()
    paths = sorted(Path(path) for path in script_paths)
    if not paths:
        print("[SUBMIT] No slurm_runner scripts to submit.")
        return 0, 0, 0

    mode = "submit-all" if opts.submit_all else "throttle"
    print(
        f"[SUBMIT] mode={mode}, jobs={len(paths)}, "
        f"skip_complete={opts.skip_complete}"
    )
    if not opts.submit_all:
        print(
            f"[SUBMIT] throttle limits: max_concurrent={opts.max_concurrent}, "
            f"max_queue_depth={opts.max_queue_depth}"
        )

    submitted = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for index, script_path in enumerate(paths, start=1):
        batch_dir = script_path.parent
        if opts.skip_complete and is_batch_complete(batch_dir):
            skipped += 1
            continue

        if opts.dry_run:
            print(f"[SUBMIT] dry-run sbatch {script_path}")
            submitted += 1
            continue

        if not opts.submit_all:
            wait_for_submit_capacity(
                opts.max_concurrent,
                opts.max_queue_depth,
                opts.poll_interval_seconds,
                dry_run=opts.dry_run,
            )

        result = submit_job(script_path.as_posix())
        if result.returncode != 0:
            failed += 1
            message = (result.stderr or result.stdout or "").strip()
            print(f"[ERROR] sbatch failed for {script_path}: {message}")
            continue

        submitted += 1
        job_id = extract_sbatch_job_id((result.stdout or "") + "\n" + (result.stderr or ""))
        if job_id:
            print(f"[SUBMIT] ({index}/{len(paths)}) {script_path.parent.name} -> job {job_id}")
        else:
            print(
                f"[SUBMIT] ({index}/{len(paths)}) {script_path.parent.name}: "
                f"{(result.stdout or '').strip()}"
            )

        if opts.submit_delay_seconds > 0:
            time.sleep(opts.submit_delay_seconds)

    elapsed = time.time() - start_time
    print(
        f"[SUBMIT] Done in {elapsed:.1f}s: submitted={submitted}, "
        f"failed={failed}, skipped={skipped}"
    )
    return submitted, failed, skipped


def extract_sbatch_job_id(output: str) -> Optional[str]:
    """Extracts Slurm job id from sbatch output text."""
    if not output:
        return None
    submitted_match = re.search(r"Submitted batch job\s+(\d+)", output)
    if submitted_match:
        return submitted_match.group(1)
    fallback_match = re.search(r"\b(\d{4,})\b", output)
    if fallback_match:
        return fallback_match.group(1)
    return None


#def update_config(path: str, prefix_value: str) -> None:
def update_config(
    path: str,
    prefix_value: str,
    scenario_continuation: bool = False,
    restart_from: str | None = None,
) -> None:
    """Updates the 'IO' section of config.js with new paths.

    This function reads the JSON configuration file, modifies the 'IO' section
    by updating the paths with a new prefix, and then writes the updated
    configuration back to the file.

    Args:
        path (str): The file system path to the JSON configuration file to be updated.
        prefix_value (str): The new prefix to be added to the paths in the 'IO' section.
        scenario_continuation (bool): If True, set restart_from to output/restart-tr.nc.
        restart_from (str | None): If provided, set IO.restart_from to this exact value.

    Returns:
        None
    """
    config_data = read_json_file(path)
    for key, val in IO_PATHS.items():
        if key == "restart_from" and scenario_continuation:
            config_data["IO"][key] = f"{prefix_value}/output/restart-tr.nc"
        else:
            config_data["IO"][key] = f"{prefix_value}/{val}"
    if restart_from is not None:
        config_data["IO"]["restart_from"] = restart_from

    write_json_file(path, config_data)


def create_slurm_script(
    path: str, template_name: str, substitution_values: dict
) -> None:
    """Creates a Slurm job script by rendering a template and writing it to a file.

    This function uses a template and a set of substitution values to generate a
    Slurm job script, and then writes the resulting script to the specified path.

    Args:
        path (str): The file system path where the Slurm job script should be saved.
        template_name (str): The name of the template file located in the templates
            folder at the root of the project.
        substitution_values (dict): A dictionary of key-value pairs for substituting
            placeholders in the template.

    Returns:
        None
    """
    slurm_runner = render_slurm_job_script(template_name, substitution_values)
    write_text_file(path, slurm_runner)


def get_gcsfs():
    import gcsfs

    return gcsfs.GCSFileSystem(project="spherical-berm-323321", token=None)


def get_cluster(n_workers, walltime="06:00:00"):
    return SLURMCluster(
        queue="dask",
        n_workers=n_workers,
        interface="ens4",
        cores=4,
        memory="30GB",
        log_directory=f"{os.getenv('HOME')}/slurm_logs",
        python="/usr/bin/python3",
        walltime=walltime,
    )


def extract_variable_name(filename):
    """Extracts the variable name and stage name from the filename.
    
    Example:
        >>> extract_variable_name("ALD_yearly_eq.nc")
        ('ALD', 'eq')
    """
    parts = filename.split("_")
    if len(parts) >= 2:
        # Get first part and stage name (without .nc extension)
        stage_name = parts[-1].split('.')[0]
        return parts[0], stage_name
    return None


def get_email_from_username():
    """Helper function to get the current user's email from their username"""
    username = os.getenv("USER")
    email_username = username.split("_")[0]

    return f"{email_username}@woodwellclimate.org"


def send_email(to: str, subject: str, body: str, pdf_path: str = None):
    sender_email = "dteber@woodwellclimate.org"
    password = os.getenv("CLUSTER_SEND_EMAIL_PASSWORD_DOGUKAN")

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    if pdf_path:
        with open(pdf_path, "rb") as file:
            attachment = MIMEApplication(file.read(), _subtype="pdf")
            attachment.add_header(
                "Content-Disposition", 
                f"attachment; filename={os.path.basename(pdf_path)}"
            )
            message.attach(attachment)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)

        server.send_message(message)

    print(f"Email sent successfully to {to}")
