import dask
import os
import re
import shutil
import subprocess
import dask.distributed
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from typing import List

from batch_processing.cmd.base import BaseCommand
from batch_processing.utils.split_planning import (
    apply_run_mask_filters,
    count_active_cells,
    filter_active_blocks,
    plan_y_stripe_blocks_by_active_cells,
    summarize_active_cells_per_block,
)
from batch_processing.utils.utils import (
    create_slurm_script,
    interpret_path,
    mpirun_rank_flags,
    update_config,
    get_gcsfs,
    get_cluster,
)

# todo: this list doesn't include co2.nc and projected-co2.c files
# give a better name and refactor
INPUT_FILES = [
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
INPUT_FILES_TO_SPLIT = [
    "drainage.zarr",
    "fri-fire.zarr",
    "run-mask.zarr",
    "soil-texture.zarr",
    "topo.zarr",
    "vegetation.zarr",
    "historic-explicit-fire.zarr",
    "projected-explicit-fire.zarr",
    "projected-climate.zarr",
    "historic-climate.zarr",
]
BATCH_DIRS: List[Path] = []
BATCH_INPUT_DIRS: List[Path] = []


class BatchSplitCommand(BaseCommand):
    def __init__(self, args):
        super().__init__()
        # todo: remove self._args and create class variables for every argument
        self._args = args
        self.base_batch_dir = Path(self.exacloud_user_dir, args.batches)
        self.log_path = Path(self.base_batch_dir, "logs")

        self.log_path.mkdir(exist_ok=True, parents=True)

        self.input_path = args.input_path

        # Patch setup_working_directory.py to include restart_from in sort_order
        self._patch_setup_working_directory()

    def _patch_setup_working_directory(self):
        """
        Patches setup_working_directory.py to add 'restart_from' to sort_order
        if it's missing. This fixes compatibility with newer dvm-dos-tem versions.
        """
        setup_script_path = os.path.join(
            self.dvmdostem_scripts_path, "util", "setup_working_directory.py"
        )
        
        if not os.path.exists(setup_script_path):
            return
        
        with open(setup_script_path, "r") as f:
            content = f.read()
        
        # Check if restart_from is already in the file
        if '"restart_from"' in content:
            return
        
        # Add restart_from after output_interval in the sort_order list
        if '"output_interval",' in content:
            content = content.replace(
                '"output_interval",',
                '"output_interval",\n    "restart_from",'
            )
            
            with open(setup_script_path, "w") as f:
                f.write(content)
            
            print("Patched setup_working_directory.py to include 'restart_from' in sort_order")

    def _run_utils(self, batch_dir, batch_input_dir):
        # todo: instead of running this file, implement what this file does
        # inside bp.
        # later, delete the last portion of the execute() code which removes
        # duplicated input files.
        # doing that should save us some time.
        setup_script_path = os.path.join(
            self.dvmdostem_scripts_path, "util", "setup_working_directory.py"
        )
        subprocess.run(
            [
                setup_script_path,
                batch_dir,
                "--input-data-path",
                batch_input_dir,
                "--copy-inputs",
            ]
        )

    def _configure(self, index: int, batch_dir: Path) -> None:
        config_file = batch_dir / "config" / "config.js"
        #update_config(path=config_file.as_posix(), prefix_value=batch_dir)
        scenario_continuation = getattr(self._args, "scenario_continuation", False)
        restart_from = getattr(self._args, "restart_from", None)
        update_config(
            path=config_file.as_posix(),
            prefix_value=batch_dir,
            scenario_continuation=scenario_continuation,
            restart_from=restart_from,
        )
        mpi_ranks = getattr(self._args, "mpi_ranks", None)

        if self._args.job_name_prefix:
            job_name = f"{self._args.job_name_prefix}-{self.base_batch_dir.name}-batch-{index}"
        else:
            job_name = f"{self.base_batch_dir.name}-batch-{index}"

        additional_flags = "--no-output-cleanup" if getattr(self._args, 'restart_run', False) else ""
        flags_before_max_output = (
            "--no-output-cleanup" if scenario_continuation else ""
        )

        substitution_values = {
            "job_name": job_name,
            "partition": self._args.slurm_partition,
            "dvmdostem_binary": self.dvmdostem_bin_path,
            "batch_dir": batch_dir,
            "log_file_path": self.log_path / f"batch-{index}",
            "log_level": self._args.log_level,
            "p": self._args.p,
            "e": self._args.e,
            "s": self._args.s,
            "t": self._args.t,
            "n": self._args.n,
            "additional_flags": additional_flags,
            "flags_before_max_output": flags_before_max_output,
            "mpirun_rank_flags": mpirun_rank_flags(mpi_ranks),
        }

        script_path = batch_dir / "slurm_runner.sh"
        create_slurm_script(
            script_path.as_posix(), "slurm_runner.sh", substitution_values
        )

    def _split_with_xarray_local(
        self, blocks: list, input_path: Path
    ) -> None:
        import dask
        n_years = getattr(self._args, "n", 0)
        files_to_split = [f for f in INPUT_FILES if n_years > 0 or not f.startswith("projected-")]
        
        chunk_y = blocks[0][1] - blocks[0][0] if blocks else 1
        chunk_x = blocks[0][3] - blocks[0][2] if blocks else 1

        cmt0_filter = getattr(self._args, "cmt0_filter", False)
        no_max_cmt = getattr(self._args, "no_max_cmt", False)
        max_cmt_val = getattr(self._args, "max_cmt", 74)
        prefiltered_run_data = getattr(self, "_prefiltered_run_data", None)
        apply_cmt_on_write = (
            (cmt0_filter or not no_max_cmt)
            and prefiltered_run_data is None
        )
        
        veg_ds = None
        if apply_cmt_on_write and "vegetation.nc" in files_to_split:
            veg_path = input_path / "vegetation.nc"
            if veg_path.exists():
                veg_ds = xr.open_dataset(veg_path, engine="netcdf4", decode_times=False)

        for input_file in files_to_split:
            src_input_path = input_path / input_file
            if not src_input_path.exists():
                print(f"Warning: {src_input_path} does not exist, skipping.")
                continue
            print("splitting ", src_input_path, "using xarray")
            
            if input_file in [
                "historic-climate.nc",
                "historic-explicit-fire.nc",
                "projected-climate.nc",
                "projected-explicit-fire.nc",
            ]:
                chunk_dict = {"Y": chunk_y, "X": chunk_x, "time": -1}
            else:
                chunk_dict = {"Y": chunk_y, "X": chunk_x}

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    ds = xr.open_dataset(src_input_path, engine="netcdf4", chunks=chunk_dict, decode_times=False)
                except Exception as e:
                    print(f"Fallback for {input_file} due to {e}")
                    ds = xr.open_dataset(src_input_path, engine="h5netcdf", chunks=chunk_dict, decode_times=False)

            if input_file == "run-mask.nc" and prefiltered_run_data is not None:
                ds = ds.load()
                ds["run"].values = prefiltered_run_data
            elif input_file == "run-mask.nc" and apply_cmt_on_write and veg_ds is not None and "veg_class" in veg_ds:
                ds = ds.load()
                veg_data = veg_ds["veg_class"].values
                import numpy as np
                if cmt0_filter:
                    ds["run"].values = np.where(veg_data == 0, 0, ds["run"].values)
                if not no_max_cmt:
                    ds["run"].values = np.where(veg_data > max_cmt_val, 0, ds["run"].values)

            delayed_writes = []
            for index, (y_start, y_end, x_start, x_end) in enumerate(blocks):
                path = os.path.join(BATCH_INPUT_DIRS[index], input_file)
                subset = ds.isel({"Y": slice(y_start, y_end), "X": slice(x_start, x_end)})
                # Always use engine="netcdf4" to avoid H5DSis_scale bug
                delayed_obj = subset.to_netcdf(path, engine="netcdf4", compute=False)
                delayed_writes.append(delayed_obj)
                
            batch_size = 125
            for i in range(0, len(delayed_writes), batch_size):
                print(f"Computing batch number {(i // batch_size) + 1} of {(len(delayed_writes) + batch_size - 1) // batch_size}")
                batch = delayed_writes[i : i + batch_size]
                dask.compute(*batch)
                
            ds.close()
            print("done splitting ", input_file)

        if veg_ds is not None:
            veg_ds.close()

    def _split_with_dask(self, bucket_path, blocks):
        cluster = get_cluster(n_workers=100)
        client = dask.distributed.Client(cluster)
        client.wait_for_workers(50)
        print(f"Dashboard link: {client.dashboard_link}")
        fs = get_gcsfs()
        
        # Calculate chunk sizes based on the first block (assuming relatively uniform blocks)
        chunk_y = blocks[0][1] - blocks[0][0] if blocks else 1
        chunk_x = blocks[0][3] - blocks[0][2] if blocks else 1
        
        n_years = getattr(self._args, "n", 0)
        files_to_split = [f for f in INPUT_FILES_TO_SPLIT if n_years > 0 or not f.startswith("projected-")]
        
        cmt0_filter = getattr(self._args, "cmt0_filter", False)
        no_max_cmt = getattr(self._args, "no_max_cmt", False)
        max_cmt_val = getattr(self._args, "max_cmt", 74)
        
        prefiltered_run_data = getattr(self, "_prefiltered_run_data", None)
        apply_cmt_on_write = (
            (cmt0_filter or not no_max_cmt)
            and prefiltered_run_data is None
        )
        
        veg_ds = None
        if apply_cmt_on_write and "vegetation.zarr" in files_to_split:
            try:
                veg_mapping = fs.get_mapper(os.path.join(bucket_path, "vegetation.zarr"), check=True)
                veg_ds = xr.open_zarr(veg_mapping, decode_times=False)
            except Exception as e:
                print(f"Warning: could not open vegetation.zarr for cmt filtering: {e}")

        for input_file in files_to_split:
            print(f"Processing {input_file}")
            try:
                bucket_mapping = fs.get_mapper(
                    os.path.join(bucket_path, input_file), check=True
                )
            except Exception as e:
                print(f"Warning: could not open {input_file} in bucket, skipping. ({e})")
                continue
                
            ds = xr.open_zarr(bucket_mapping, decode_times=False)
            if input_file in [
                "historic-climate.zarr",
                "historic-explicit-fire.zarr",
                "projected-climate.zarr",
                "projected-explicit-fire.zarr",
            ]:
                chunk_dict = {"Y": chunk_y, "X": chunk_x, "time": -1}
            else:
                chunk_dict = {"Y": chunk_y, "X": chunk_x}

            ds = ds.chunk(chunk_dict)

            if input_file == "run-mask.zarr" and prefiltered_run_data is not None:
                ds = ds.load()
                ds["run"].values = prefiltered_run_data
            elif input_file == "run-mask.zarr" and apply_cmt_on_write and veg_ds is not None and "veg_class" in veg_ds:
                ds = ds.load()
                veg_data = veg_ds["veg_class"].values
                import numpy as np
                if cmt0_filter:
                    ds["run"].values = np.where(veg_data == 0, 0, ds["run"].values)
                if not no_max_cmt:
                    ds["run"].values = np.where(veg_data > max_cmt_val, 0, ds["run"].values)

            # I know this is ugly but passing `ds` as an argument makes things painfully slow
            @dask.delayed
            def _process_data(y_start, y_end, x_start, x_end, output_path):
                subset = ds.isel({"Y": slice(y_start, y_end), "X": slice(x_start, x_end)})
                obj = subset.to_netcdf(output_path, engine="h5netcdf")
                return obj

            delayed_objs = [
                _process_data(
                    b[0], b[1], b[2], b[3],
                    os.path.join(
                        self.base_batch_dir,
                        f"batch_{i}",
                        "input",
                        f"{input_file[:len(input_file)-5]}.nc",
                    ),
                )
                for i, b in enumerate(blocks)
            ]
            batch_size = 125
            for i in range(0, len(blocks), batch_size):
                print(f"Computing batch number {(i // batch_size) + 1}")
                batch = delayed_objs[i : i + batch_size]
                dask.compute(*batch)

            ds.close()

        if veg_ds is not None:
            veg_ds.close()

        cluster.close()

    def execute(self):
        slurm_partition = getattr(self._args, "slurm_partition", "spot")
        mpi_ranks = getattr(self._args, "mpi_ranks", None)
        if mpi_ranks is not None:
            max_cpus = {"spot": 8, "dask": 4, "compute": 15}.get(slurm_partition, 8)
            if int(mpi_ranks) > max_cpus:
                import typer
                typer.secho(
                    f"Error: The requested --mpi-ranks ({mpi_ranks}) exceeds the maximum number of CPUs ({max_cpus}) available per node for the '{slurm_partition}' partition.",
                    err=True, fg=typer.colors.RED
                )
                import sys
                sys.exit(1)

        reading_remote_data = False
        if self.input_path.startswith("gcs://"):
            self.input_path = self.input_path.replace("gcs://", "")
            reading_remote_data = True
        else:
            self.input_path = Path(interpret_path(self.input_path))

        fs = get_gcsfs()

        if reading_remote_data:
            path = fs.get_mapper(
                os.path.join(self.input_path, "run-mask.zarr"), check=True
            )
            ds = xr.open_zarr(path)
        else:
            #ds = xr.open_dataset(self.input_path / "run-mask.nc", engine="h5netcdf")
            ds = xr.open_dataset(
                self.input_path / "run-mask.nc",
                engine="h5netcdf",
                driver_kwds={"backend": "pyfive"},
            )

        X, Y = ds.X.size, ds.Y.size
        print("Dimension size of X:", X)
        print("Dimension size of Y:", Y)
        
        import numpy as np
        run_data = None
        if "run" in ds:
            run_data = np.asarray(ds["run"].values, dtype=float)
            while run_data.ndim > 2:
                run_data = run_data.take(0, axis=0)
            run_data = np.where(np.isnan(run_data), 0, run_data)

            cmt0_filter = getattr(self._args, "cmt0_filter", False)
            no_max_cmt = getattr(self._args, "no_max_cmt", False)
            max_cmt_val = getattr(self._args, "max_cmt", 74)
            active_before = count_active_cells(run_data)
            run_data = apply_run_mask_filters(
                run_data,
                self.input_path if not reading_remote_data else Path(self.input_path),
                cmt0_filter=cmt0_filter,
                no_max_cmt=no_max_cmt,
                max_cmt=max_cmt_val,
            )
            active_after = count_active_cells(run_data)
            if active_after != active_before:
                print(
                    f"Applied CMT filters on full grid before split: "
                    f"active {active_before} -> {active_after}"
                )
            elif cmt0_filter or not no_max_cmt:
                print(
                    f"CMT filters enabled; active cells unchanged ({active_after})."
                )
            self._prefiltered_run_data = run_data

        cells_per_batch = getattr(self._args, "cells_per_batch", None)
        blocks = []

        if cells_per_batch is not None and int(cells_per_batch) > 0:
            if reading_remote_data:
                raise NotImplementedError(
                    "bp batch split --cells-per-batch currently supports "
                    "local input paths only."
                )
            if run_data is None:
                raise ValueError("run-mask.nc must contain variable 'run'")
            blocks = plan_y_stripe_blocks_by_active_cells(
                run_data, int(cells_per_batch)
            )
            avg_cells, min_cells, max_cells = summarize_active_cells_per_block(
                blocks, run_data
            )
            print(
                f"\nPlanned {len(blocks)} full-width Y-stripe batches by "
                f"~{int(cells_per_batch)} active cells/batch "
                f"(avg/min/max: {avg_cells:.1f}/{min_cells}/{max_cells})"
            )
        else:
            SPLIT_DIMENSION = "Y"
            print(f"\nSplitting across {SPLIT_DIMENSION} dimension")
            print("Dimension size:", Y)
            for y in range(Y):
                blocks.append((y, y + 1, 0, X))

        # Filter active blocks
        if run_data is not None:
            blocks = filter_active_blocks(blocks, run_data)
        DIMENSION_SIZE = len(blocks)
        print(f"Filtered to {DIMENSION_SIZE} active batches")

        ds.close()

        print("Cleaning up the existing directories")
        if self.base_batch_dir.exists():
            pattern = re.compile(r"^batch_\d+$")
            to_be_removed = [
                d
                for d in self.base_batch_dir.iterdir()
                if d.is_dir() and pattern.match(d.name)
            ]

            with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                executor.map(lambda elem: shutil.rmtree(elem), to_be_removed)

        print("Set up batch directories")
        self.base_batch_dir.mkdir(exist_ok=True)
        self.log_path.mkdir(exist_ok=True)

        # write layout to json
        import json
        with open(self.base_batch_dir / "batch_layout.json", "w") as f:
            json.dump({"blocks": blocks, "X": X, "Y": Y}, f)

        for index in range(DIMENSION_SIZE):
            path = self.base_batch_dir / f"batch_{index}"
            BATCH_DIRS.append(path)

            path = path / "input"
            BATCH_INPUT_DIRS.append(path)

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            executor.map(lambda elem: os.makedirs(elem), BATCH_INPUT_DIRS)

        n_years = getattr(self._args, "n", 0)
        co2_files = ["co2.zarr/"]
        if n_years > 0:
            co2_files.append("projected-co2.zarr/")
            
        if reading_remote_data:
            for co2_file in co2_files:
                src = os.path.join(self.input_path, co2_file)
                dst = os.path.join(self.base_batch_dir, co2_file)
                try:
                    fs.get(src, dst, recursive=True)

                    ds = xr.open_zarr(dst)
                    co2_file_path = Path(co2_file)
                    ds.to_netcdf(
                        os.path.join(self.base_batch_dir, f"{co2_file_path.stem}.nc"),
                        engine="h5netcdf",
                    )
                    ds.close()
                    shutil.rmtree(dst)
                except Exception as e:
                    print(f"Warning: Failed to process {co2_file}: {e}")

        # co2.nc and projected-co2.nc doesn't have X and Y dimensions. So, we copy
        # them instead of splitting.
        print("Copy co2.nc and projected-co2.nc files (if needed)")
        co2_dest = self.input_path
        if reading_remote_data:
            co2_dest = self.base_batch_dir

        for batch_dir in BATCH_INPUT_DIRS:
            src_co2 = co2_dest / "co2.nc"
            dst_co2 = batch_dir / "co2.nc"
            if src_co2.exists():
                shutil.copy(src_co2, dst_co2)

            if n_years > 0:
                src_projected_co2 = co2_dest / "projected-co2.nc"
                dst_projected_co2 = batch_dir / "projected-co2.nc"
                if src_projected_co2.exists():
                    shutil.copy(src_projected_co2, dst_projected_co2)

        if reading_remote_data:
            co2_nc = os.path.join(co2_dest, "co2.nc")
            if os.path.exists(co2_nc):
                os.remove(co2_nc)
            proj_co2_nc = os.path.join(co2_dest, "projected-co2.nc")
            if os.path.exists(proj_co2_nc):
                os.remove(proj_co2_nc)

        print("Split input files")
        if reading_remote_data:
            self._split_with_dask(self.input_path, blocks)
        else:
            self._split_with_xarray_local(blocks, self.input_path)

        print("Set up the batch simulation")
        for batch_dir, batch_input_dir in zip(BATCH_DIRS, BATCH_INPUT_DIRS):
            self._run_utils(batch_dir, batch_input_dir)

        print("Configure each batch")
        for index, batch_dir in enumerate(BATCH_DIRS):
            self._configure(index, batch_dir)

        # we have to do this otherwise there would be two inputs folders:
        # input/ and inputs/
        #
        # inputs/ folder is created because we are calling setup_working_directory.py
        print("Delete duplicated inputs files")
        duplicated_input_paths = self.base_batch_dir.glob("*/inputs")
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            executor.map(lambda elem: shutil.rmtree(elem), duplicated_input_paths)
