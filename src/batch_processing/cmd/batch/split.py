import dask
import os
import re
import shutil
import subprocess
import dask.distributed
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from batch_processing.cmd.base import BaseCommand
from batch_processing.utils.utils import (
    create_slurm_script,
    interpret_path,
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

RUN_MASK_DESTINATION = "run-mask.nc"
VEGETATION_DESTINATION = "vegetation.nc"
VEG_CLASS_VARIABLE = "veg_class"
RUN_MASK_VARIABLE = "run"
RUN_ENABLED_VALUE = 1
OUTPUT_NETCDF_FORMAT = "NETCDF4_CLASSIC"
ROW_DIM_CANDIDATES = ("Y", "y", "latitude", "lat")
COL_DIM_CANDIDATES = ("X", "x", "longitude", "lon")


def _open_dataset_for_read(path: Path, decode_cf: bool = False) -> xr.Dataset:
    path_str = path.as_posix()
    try:
        return xr.open_dataset(
            path_str,
            engine="h5netcdf",
            decode_times=False,
            decode_cf=decode_cf,
        )
    except Exception:
        return xr.open_dataset(
            path_str,
            engine="netcdf4",
            decode_times=False,
            decode_cf=decode_cf,
        )


def _detect_spatial_dims(dim_names: Iterable[str]) -> Optional[Tuple[str, str]]:
    dim_name_set = set(dim_names)
    for row_dim in ROW_DIM_CANDIDATES:
        if row_dim not in dim_name_set:
            continue
        for col_dim in COL_DIM_CANDIDATES:
            if col_dim in dim_name_set:
                return row_dim, col_dim
    return None


def _extract_run_mask_2d(
    ds: xr.Dataset,
    source_label: str,
    run_var: str = RUN_MASK_VARIABLE,
) -> tuple[xr.DataArray, str, str]:
    if run_var not in ds:
        raise KeyError(f"{source_label} must contain '{run_var}' variable.")

    run_da = ds[run_var]
    spatial_dims = _detect_spatial_dims(run_da.dims)
    if spatial_dims is None:
        raise ValueError(
            f"{source_label}:{run_var} must include row/col dims from "
            f"{ROW_DIM_CANDIDATES} x {COL_DIM_CANDIDATES}. Found {tuple(run_da.dims)}."
        )
    row_dim, col_dim = spatial_dims

    for dim_name in run_da.dims:
        if dim_name in (row_dim, col_dim):
            continue
        if int(run_da.sizes[dim_name]) != 1:
            raise ValueError(
                f"{source_label}:{run_var} contains non-singleton extra dimension "
                f"'{dim_name}' with size {int(run_da.sizes[dim_name])}."
            )
        run_da = run_da.isel({dim_name: 0}, drop=True)

    run_da = run_da.transpose(row_dim, col_dim)
    return run_da, row_dim, col_dim


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
        update_config(path=config_file.as_posix(), prefix_value=batch_dir)

        if self._args.job_name_prefix:
            job_name = f"{self._args.job_name_prefix}-{self.base_batch_dir.name}-batch-{index}"
        else:
            job_name = f"{self.base_batch_dir.name}-batch-{index}"

        additional_flags = "--no-output-cleanup" if getattr(self._args, 'restart_run', False) else ""
        scenario_continuation = getattr(self._args, "scenario_continuation", False)
        flags_before_max_output = (
            "--no-output-cleanup" if scenario_continuation else ""
        )
        mpi_ranks = max(1, int(getattr(self._args, "mpi_ranks", 1)))

        substitution_values = {
            "job_name": job_name,
            "partition": self._args.slurm_partition,
            "dvmdostem_binary": self.dvmdostem_bin_path,
            "log_file_path": self.log_path / f"batch-{index}",
            "log_level": self._args.log_level,
            "config_path": config_file,
            "p": self._args.p,
            "e": self._args.e,
            "s": self._args.s,
            "t": self._args.t,
            "n": self._args.n,
            "additional_flags": additional_flags,
            "flags_before_max_output": flags_before_max_output,
            "mpi_ranks": mpi_ranks,
        }

        script_path = batch_dir / "slurm_runner.sh"
        create_slurm_script(
            script_path.as_posix(), "slurm_runner.sh", substitution_values
        )

    def _split_with_nco(
        self, start_index: int, end_index: int, input_path: Path, split_dimension: str
    ) -> None:
        for input_file in INPUT_FILES:
            src_input_path = input_path / input_file
            print("splitting ", src_input_path)
            for index in range(start_index, end_index):
                path = os.path.join(BATCH_INPUT_DIRS[index], input_file)
                subprocess.run(
                    [
                        "ncks",
                        "-O",
                        "-h",
                        "-d",
                        f"{split_dimension},{index}",
                        src_input_path,
                        path,
                    ]
                )
            print("done splitting ", input_file)

    def _split_with_dask(self, bucket_path):
        cluster = get_cluster(n_workers=100)
        client = dask.distributed.Client(cluster)
        client.wait_for_workers(50)
        print(f"Dashboard link: {client.dashboard_link}")
        fs = get_gcsfs()
        for input_file in INPUT_FILES_TO_SPLIT:
            print(f"Processing {input_file}")
            bucket_mapping = fs.get_mapper(
                os.path.join(bucket_path, input_file), check=True
            )
            ds = xr.open_zarr(bucket_mapping, decode_times=False)
            if input_file in [
                "historic-climate.zarr",
                "historic-explicit-fire.zarr",
                "projected-climate.zarr",
                "projected-explicit-fire.zarr",
            ]:
                chunk_dict = {"Y": 1, "X": -1, "time": -1}
            else:
                chunk_dict = {"Y": 1, "X": -1}

            ds = ds.chunk(chunk_dict)
            y_dim = ds.Y.size

            # I know this is ugly but passing `ds` as an argument makes things painfully slow
            @dask.delayed
            def _process_data(col_index, output_path):
                subset = ds.isel({"Y": col_index}).expand_dims("Y")
                obj = subset.to_netcdf(output_path, engine="h5netcdf")
                return obj

            delayed_objs = [
                _process_data(
                    i,
                    os.path.join(
                        self.base_batch_dir,
                        f"batch_{i}",
                        "input",
                        f"{input_file[:len(input_file)-5]}.nc",
                    ),
                )
                for i in range(y_dim)
            ]
            batch_size = 125
            for i in range(0, y_dim, batch_size):
                print(f"Computing batch number {(i // batch_size) + 1}")
                batch = delayed_objs[i : i + batch_size]
                dask.compute(*batch)

            ds.close()

        cluster.close()

    def _compute_veg_class_zero_mask(
        self, da: xr.DataArray, row_dim: str, col_dim: str, var_name: str
    ) -> xr.DataArray:
        if row_dim not in da.dims or col_dim not in da.dims:
            raise ValueError(
                f"Vegetation variable '{var_name}' must include '{row_dim}' and "
                f"'{col_dim}' dims. Found {tuple(da.dims)}."
            )

        da_work = da
        for dim_name in list(da_work.dims):
            if dim_name in (row_dim, col_dim):
                continue
            sz = int(da_work.sizes[dim_name])
            if sz == 1:
                da_work = da_work.isel({dim_name: 0}, drop=True)
            elif sz < 1:
                raise ValueError(
                    f"{var_name}: dimension '{dim_name}' has invalid size {sz}."
                )

        zero = da_work == 0
        reduce_dims = [d for d in zero.dims if d not in (row_dim, col_dim)]
        if reduce_dims:
            zero = zero.any(dim=reduce_dims)
        return zero.transpose(row_dim, col_dim)

    def _compute_veg_class_gt_mask(
        self,
        da: xr.DataArray,
        row_dim: str,
        col_dim: str,
        var_name: str,
        max_cmt: int,
    ) -> xr.DataArray:
        if row_dim not in da.dims or col_dim not in da.dims:
            raise ValueError(
                f"Vegetation variable '{var_name}' must include '{row_dim}' and "
                f"'{col_dim}' dims. Found {tuple(da.dims)}."
            )

        da_work = da
        for dim_name in list(da_work.dims):
            if dim_name in (row_dim, col_dim):
                continue
            sz = int(da_work.sizes[dim_name])
            if sz == 1:
                da_work = da_work.isel({dim_name: 0}, drop=True)
            elif sz < 1:
                raise ValueError(
                    f"{var_name}: dimension '{dim_name}' has invalid size {sz}."
                )

        high = da_work > max_cmt
        reduce_dims = [d for d in high.dims if d not in (row_dim, col_dim)]
        if reduce_dims:
            high = high.any(dim=reduce_dims)
        return high.transpose(row_dim, col_dim)

    def _write_filtered_run_mask(
        self,
        run_mask_file: Path,
        run_mask_ds: xr.Dataset,
        run_mask_da: xr.DataArray,
        active_before: xr.DataArray,
        disable_mask: xr.DataArray,
    ) -> None:
        updated_values = np.where(
            active_before.values & ~disable_mask.values, RUN_ENABLED_VALUE, 0
        ).astype(run_mask_da.dtype, copy=False)
        run_mask_out = run_mask_ds.copy(deep=True)
        run_mask_out[RUN_MASK_VARIABLE] = xr.DataArray(
            updated_values,
            dims=run_mask_da.dims,
            coords=run_mask_da.coords,
            attrs=run_mask_da.attrs.copy(),
        )
        tmp_path = run_mask_file.with_suffix(".tmp.nc")
        run_mask_out.to_netcdf(
            tmp_path.as_posix(),
            engine="netcdf4",
            format=OUTPUT_NETCDF_FORMAT,
        )
        run_mask_out.close()
        tmp_path.replace(run_mask_file)

    def _prefilter_batch_run_mask_cmt0(self, batch_input_dir: Path) -> dict[str, int]:
        run_mask_file = batch_input_dir / RUN_MASK_DESTINATION
        vegetation_file = batch_input_dir / VEGETATION_DESTINATION
        if not run_mask_file.exists():
            raise FileNotFoundError(f"Missing run-mask for cmt0-filter: {run_mask_file}")
        if not vegetation_file.exists():
            raise FileNotFoundError(
                f"Missing vegetation for cmt0-filter: {vegetation_file}"
            )

        with _open_dataset_for_read(run_mask_file) as run_mask_ds:
            run_mask_da, row_dim, col_dim = _extract_run_mask_2d(
                run_mask_ds, run_mask_file.name, run_var=RUN_MASK_VARIABLE
            )
            active_before = np.isfinite(run_mask_da) & np.isclose(
                run_mask_da, RUN_ENABLED_VALUE
            )
            active_before_count = int(active_before.sum().item())

            with _open_dataset_for_read(vegetation_file) as veg_ds:
                if VEG_CLASS_VARIABLE not in veg_ds:
                    raise KeyError(
                        f"{vegetation_file} missing '{VEG_CLASS_VARIABLE}' variable."
                    )
                veg_zero = self._compute_veg_class_zero_mask(
                    veg_ds[VEG_CLASS_VARIABLE],
                    row_dim=row_dim,
                    col_dim=col_dim,
                    var_name=VEG_CLASS_VARIABLE,
                )

            if veg_zero.sizes != run_mask_da.sizes:
                raise ValueError(
                    f"veg_class grid {dict(veg_zero.sizes)} does not match "
                    f"run grid {dict(run_mask_da.sizes)} in {batch_input_dir}."
                )

            disable_mask = active_before & veg_zero
            disabled_count = int(disable_mask.sum().item())
            active_after_count = active_before_count - disabled_count

            if disabled_count == 0:
                return {
                    "active_before": active_before_count,
                    "active_after": active_after_count,
                    "disabled": disabled_count,
                }

            self._write_filtered_run_mask(
                run_mask_file, run_mask_ds, run_mask_da, active_before, disable_mask
            )

        return {
            "active_before": active_before_count,
            "active_after": active_after_count,
            "disabled": disabled_count,
        }

    def _prefilter_split_run_masks_cmt0(self, batch_input_dirs: List[Path]) -> None:
        total_disabled = 0
        batches_changed = 0
        for batch_index, batch_input_dir in enumerate(batch_input_dirs, start=1):
            result = self._prefilter_batch_run_mask_cmt0(batch_input_dir=batch_input_dir)
            total_disabled += result["disabled"]
            if result["disabled"] > 0:
                batches_changed += 1
            print(
                "  [cmt0-filter] "
                f"{batch_input_dir.parent.name} ({batch_index}/{len(batch_input_dirs)}): "
                f"active {result['active_before']} -> {result['active_after']} "
                f"(disabled {result['disabled']})"
            )
        print(
            "[cmt0-filter] Done: "
            f"disabled {total_disabled} active cells across {batches_changed} batches."
        )

    def _prefilter_batch_run_mask_max_cmt(
        self, batch_input_dir: Path, max_cmt: int
    ) -> dict[str, int]:
        run_mask_file = batch_input_dir / RUN_MASK_DESTINATION
        vegetation_file = batch_input_dir / VEGETATION_DESTINATION
        if not run_mask_file.exists():
            raise FileNotFoundError(
                f"Missing run-mask for max-cmt filter: {run_mask_file}"
            )
        if not vegetation_file.exists():
            raise FileNotFoundError(
                f"Missing vegetation for max-cmt filter: {vegetation_file}"
            )

        with _open_dataset_for_read(run_mask_file) as run_mask_ds:
            run_mask_da, row_dim, col_dim = _extract_run_mask_2d(
                run_mask_ds, run_mask_file.name, run_var=RUN_MASK_VARIABLE
            )
            active_before = np.isfinite(run_mask_da) & np.isclose(
                run_mask_da, RUN_ENABLED_VALUE
            )
            active_before_count = int(active_before.sum().item())

            with _open_dataset_for_read(vegetation_file) as veg_ds:
                if VEG_CLASS_VARIABLE not in veg_ds:
                    raise KeyError(
                        f"{vegetation_file} missing '{VEG_CLASS_VARIABLE}' variable."
                    )
                veg_high = self._compute_veg_class_gt_mask(
                    veg_ds[VEG_CLASS_VARIABLE],
                    row_dim=row_dim,
                    col_dim=col_dim,
                    var_name=VEG_CLASS_VARIABLE,
                    max_cmt=max_cmt,
                )

            if veg_high.sizes != run_mask_da.sizes:
                raise ValueError(
                    f"veg_class grid {dict(veg_high.sizes)} does not match "
                    f"run grid {dict(run_mask_da.sizes)} in {batch_input_dir}."
                )

            disable_mask = active_before & veg_high
            disabled_count = int(disable_mask.sum().item())
            active_after_count = active_before_count - disabled_count

            if disabled_count == 0:
                return {
                    "active_before": active_before_count,
                    "active_after": active_after_count,
                    "disabled": disabled_count,
                }

            self._write_filtered_run_mask(
                run_mask_file, run_mask_ds, run_mask_da, active_before, disable_mask
            )

        return {
            "active_before": active_before_count,
            "active_after": active_after_count,
            "disabled": disabled_count,
        }

    def _prefilter_split_run_masks_max_cmt(
        self, batch_input_dirs: List[Path], max_cmt: int
    ) -> None:
        total_disabled = 0
        batches_changed = 0
        for batch_index, batch_input_dir in enumerate(batch_input_dirs, start=1):
            result = self._prefilter_batch_run_mask_max_cmt(
                batch_input_dir=batch_input_dir, max_cmt=max_cmt
            )
            total_disabled += result["disabled"]
            if result["disabled"] > 0:
                batches_changed += 1
            print(
                "  [max-cmt-filter] "
                f"{batch_input_dir.parent.name} ({batch_index}/{len(batch_input_dirs)}): "
                f"active {result['active_before']} -> {result['active_after']} "
                f"(disabled {result['disabled']}, max_cmt={max_cmt})"
            )
        print(
            "[max-cmt-filter] Done: "
            f"disabled {total_disabled} active cells across {batches_changed} batches "
            f"(veg_class > {max_cmt})."
        )

    def execute(self):
        global BATCH_DIRS, BATCH_INPUT_DIRS
        BATCH_DIRS = []
        BATCH_INPUT_DIRS = []

        cmt0_filter = bool(getattr(self._args, "cmt0_filter", False))
        no_max_cmt = bool(getattr(self._args, "no_max_cmt", False))
        max_cmt_val = int(getattr(self._args, "max_cmt", 74))
        max_cmt = None if no_max_cmt else max_cmt_val
        print(f"[batch split] CMT0 run-mask filter (veg_class==0): {cmt0_filter}")
        if max_cmt is not None:
            print(
                f"[batch split] Max-CMT run-mask filter (veg_class > {max_cmt}): True"
            )
        else:
            print(
                "[batch split] Max-CMT run-mask filter (veg_class > N): False "
                "(--no-max-cmt)"
            )

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
            ds = xr.open_dataset(
                self.input_path / "run-mask.nc",
                engine="h5netcdf",
                driver_kwds={"backend": "pyfive"},
            )

        X, Y = ds.X.size, ds.Y.size
        print("Dimension size of X:", X)
        print("Dimension size of Y:", Y)

        # always split across y dimension
        SPLIT_DIMENSION, DIMENSION_SIZE = "Y", Y

        print(f"\nSplitting accros {SPLIT_DIMENSION} dimension")
        print("Dimension size:", DIMENSION_SIZE)

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
        for index in range(DIMENSION_SIZE):
            path = self.base_batch_dir / f"batch_{index}"
            BATCH_DIRS.append(path)

            path = path / "input"
            BATCH_INPUT_DIRS.append(path)

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            executor.map(lambda elem: os.makedirs(elem), BATCH_INPUT_DIRS)

        co2_files = ["co2.zarr/", "projected-co2.zarr/"]
        if reading_remote_data:
            for co2_file in co2_files:
                src = os.path.join(self.input_path, co2_file)
                dst = os.path.join(self.base_batch_dir, co2_file)
                fs.get(src, dst, recursive=True)

                ds = xr.open_zarr(dst)
                co2_file = Path(co2_file)
                ds.to_netcdf(
                    os.path.join(self.base_batch_dir, f"{co2_file.stem}.nc"),
                    engine="h5netcdf",
                )
                ds.close()
                shutil.rmtree(dst)

        # co2.nc and projected-co2.nc doesn't have X and Y dimensions. So, we copy
        # them instead of splitting.
        print("Copy co2.nc and projected-co2.nc files")
        co2_dest = self.input_path
        if reading_remote_data:
            co2_dest = self.base_batch_dir

        for batch_dir in BATCH_INPUT_DIRS:
            src_co2 = co2_dest / "co2.nc"
            dst_co2 = batch_dir / "co2.nc"
            shutil.copy(src_co2, dst_co2)

            src_projected_co2 = co2_dest / "projected-co2.nc"
            dst_projected_co2 = batch_dir / "projected-co2.nc"
            shutil.copy(src_projected_co2, dst_projected_co2)

            src_ch4 = co2_dest / "ch4.nc"
            dst_ch4 = batch_dir / "ch4.nc"
            shutil.copy(src_ch4, dst_ch4)

            src_projected_ch4 = co2_dest / "projected-ch4.nc"
            dst_projected_ch4 = batch_dir / "projected-ch4.nc"
            shutil.copy(src_projected_ch4, dst_projected_ch4)

        if reading_remote_data:
            os.remove(os.path.join(co2_dest, "co2.nc"))
            os.remove(os.path.join(co2_dest, "projected-co2.nc"))
            os.remove(os.path.join(co2_dest, "ch4.nc"))
            os.remove(os.path.join(co2_dest, "projected-ch4.nc"))

        print("Split input files")
        if reading_remote_data:
            self._split_with_dask(self.input_path)
        else:
            self._split_with_nco(0, DIMENSION_SIZE, self.input_path, SPLIT_DIMENSION)

        if cmt0_filter:
            print("Pre-filtering split run-mask files (disable cells where veg_class==0)")
            self._prefilter_split_run_masks_cmt0(BATCH_INPUT_DIRS)
        if max_cmt is not None:
            print(
                f"Pre-filtering split run-mask files "
                f"(disable cells where veg_class>{max_cmt})"
            )
            self._prefilter_split_run_masks_max_cmt(BATCH_INPUT_DIRS, max_cmt)

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
