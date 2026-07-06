from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import xarray as xr

from batch_processing.cmd.base import BaseCommand
from batch_processing.utils.utils import get_batch_number
from batch_processing.utils.wiemip_processing import (
    BATCH_LAYOUT_FILENAME,
    RUN_MASK_VARIABLE,
    SPLIT_METADATA_FILENAME,
    SPLIT_MODE_RECT,
    SPLIT_MODE_Y_STRIPE,
    WiemipSplitMetadata,
    extract_run_mask_2d,
    open_dataset_for_read,
    read_batch_layout,
    read_split_metadata,
    restore_filtered_dataset_to_full_grid,
)

CONCAT_DIM_CANDIDATES = ("Y", "y", "latitude", "lat", "active_cell")
COL_DIM_CANDIDATES = ("X", "x", "longitude", "lon")
ROW_DIMS = ("Y", "y", "latitude", "lat")
COL_DIMS = ("X", "x", "longitude", "lon")
Block = Tuple[int, int, int, int]


class WiemipMergeCommand(BaseCommand):
    """
    Merge WIEMIP batch outputs back into full-domain files.

    Y-stripe splits are merged by concatenating along Y. Rect splits use a 2D
    canvas placement driven by ``batch_layout.json``.
    """

    def __init__(self, args):
        super().__init__()
        self._args = args
        self.base_batch_dir = Path(self.exacloud_user_dir, args.batches)
        self.result_root_dir = self.base_batch_dir / args.output_dir_name
        self.filtered_result_dir = self.result_root_dir / "merged_filtered"
        self.restored_result_dir = self.result_root_dir / "merged_restored"
        self.filtered_result_dir.mkdir(parents=True, exist_ok=True)
        self.restored_result_dir.mkdir(parents=True, exist_ok=True)

    def _get_available_batches(self) -> List[Path]:
        batch_dirs = [
            path
            for path in self.base_batch_dir.iterdir()
            if path.is_dir() and path.name.startswith("batch_")
        ]
        return sorted(batch_dirs, key=lambda p: get_batch_number(p.name))

    def _get_output_files(self, batch_dirs: List[Path]) -> List[str]:
        if not batch_dirs:
            return []

        output_files: Set[str] = set()
        skipped_non_nc: Set[str] = set()
        found_output_dir = False

        for batch_dir in batch_dirs:
            output_dir = batch_dir / "output"
            if not output_dir.exists():
                continue
            found_output_dir = True
            for path in output_dir.iterdir():
                if not path.is_file():
                    continue
                if path.suffix.lower() == ".nc":
                    output_files.add(path.name)
                else:
                    skipped_non_nc.add(path.name)

        if skipped_non_nc:
            print(
                "Skipping non-NetCDF output files from merge: "
                f"{', '.join(sorted(skipped_non_nc))}"
            )

        if not found_output_dir:
            return []

        return sorted(output_files)

    def _resolve_concat_dim(self, sample_file: Path) -> Optional[str]:
        with open_dataset_for_read(sample_file, decode_cf=False) as ds:
            for concat_dim in CONCAT_DIM_CANDIDATES:
                if concat_dim in ds.dims:
                    return concat_dim
        return None

    def _resolve_col_dim(self, sample_file: Path) -> Optional[str]:
        with open_dataset_for_read(sample_file, decode_cf=False) as ds:
            for col_dim in COL_DIM_CANDIDATES:
                if col_dim in ds.dims:
                    return col_dim
        return None

    def _validate_constant_dimension(
        self, files: List[str], dim_name: str, output_file: str
    ) -> None:
        expected_size: Optional[int] = None
        for file_path in files:
            with open_dataset_for_read(Path(file_path), decode_cf=False) as ds:
                if dim_name not in ds.dims:
                    raise ValueError(
                        f"{output_file}: missing '{dim_name}' in {file_path}."
                    )
                dim_size = int(ds.sizes[dim_name])
            if expected_size is None:
                expected_size = dim_size
                continue
            if dim_size != expected_size:
                raise ValueError(
                    f"{output_file}: inconsistent '{dim_name}' size across batches. "
                    f"Expected {expected_size}, found {dim_size} in {file_path}."
                )

    def _resolve_split_layout(
        self, metadata: WiemipSplitMetadata
    ) -> Tuple[str, Optional[List[Block]], Optional[int], Optional[int]]:
        split_mode = metadata.split_mode or SPLIT_MODE_Y_STRIPE
        layout_path = self.base_batch_dir / BATCH_LAYOUT_FILENAME
        if split_mode == SPLIT_MODE_RECT:
            if layout_path.exists():
                blocks, grid_y, grid_x = read_batch_layout(layout_path)
                return split_mode, blocks, grid_y, grid_x
            if metadata.blocks:
                blocks = [tuple(int(v) for v in block) for block in metadata.blocks]
                return split_mode, blocks, metadata.bbox.n_rows, metadata.bbox.n_cols
            raise FileNotFoundError(
                "Rect split merge requires batch_layout.json or blocks in metadata. "
                f"Expected: {layout_path}"
            )
        return split_mode, None, None, None

    def _resolve_spatial_dim_names(self, sample_ds: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
        row_dim = next((name for name in ROW_DIMS if name in sample_ds.dims), None)
        col_dim = next((name for name in COL_DIMS if name in sample_ds.dims), None)
        return row_dim, col_dim

    def _merge_single_output_file_y_stripe(
        self, output_file: str, batch_dirs: List[Path], target_dir: Path
    ) -> Optional[Path]:
        files = []
        missing_batches = []
        for batch_dir in batch_dirs:
            file_path = batch_dir / "output" / output_file
            if file_path.exists():
                files.append(file_path.as_posix())
            else:
                missing_batches.append(batch_dir.name)

        if not files:
            print(f"Skipping {output_file}: no source files found.")
            return None

        if missing_batches:
            print(
                f"Merging {output_file} with missing batches: {', '.join(missing_batches)}"
            )

        concat_dim = self._resolve_concat_dim(Path(files[0]))
        col_dim = self._resolve_col_dim(Path(files[0]))
        target = target_dir / output_file

        if concat_dim is None:
            if len(files) == 1:
                with open_dataset_for_read(Path(files[0]), decode_cf=False) as ds:
                    ds.to_netcdf(target.as_posix(), engine="netcdf4")
                print(f"Wrote {target}")
                return target

            raise ValueError(
                f"Could not determine merge dimension for {output_file}. "
                f"Tried {CONCAT_DIM_CANDIDATES}."
            )

        if concat_dim != "active_cell" and col_dim is not None:
            self._validate_constant_dimension(files, col_dim, output_file)

        try:
            ds = xr.open_mfdataset(
                files,
                engine="h5netcdf",
                combine="nested",
                concat_dim=concat_dim,
                data_vars="minimal",
                coords="minimal",
                compat="override",
                decode_cf=False,
                decode_times=False,
            )
        except Exception:
            ds = xr.open_mfdataset(
                files,
                engine="netcdf4",
                combine="nested",
                concat_dim=concat_dim,
                data_vars="minimal",
                coords="minimal",
                compat="override",
                decode_cf=False,
                decode_times=False,
            )
        try:
            ds.to_netcdf(target.as_posix(), engine="netcdf4")
            print(f"Wrote {target}")
        finally:
            ds.close()
        return target

    def _merge_single_output_file_rect(
        self,
        output_file: str,
        batch_dirs: List[Path],
        blocks: List[Block],
        grid_y: int,
        grid_x: int,
        target_dir: Path,
    ) -> Optional[Path]:
        available: List[tuple[int, Path]] = []
        for batch_dir in batch_dirs:
            file_path = batch_dir / "output" / output_file
            if file_path.exists():
                available.append((get_batch_number(batch_dir.name), file_path))

        if not available:
            print(f"Skipping {output_file}: no source files found.")
            return None

        if len(available) != len(blocks):
            print(
                f"[WARN] {output_file}: {len(available)} batch files vs "
                f"{len(blocks)} layout blocks."
            )

        sample_path = available[0][1]
        with open_dataset_for_read(sample_path, decode_cf=False) as sample_ds:
            row_dim, col_dim = self._resolve_spatial_dim_names(sample_ds)
            if row_dim is None or col_dim is None:
                if len(available) == 1:
                    target = target_dir / output_file
                    with open_dataset_for_read(sample_path, decode_cf=False) as ds:
                        ds.to_netcdf(target.as_posix(), engine="netcdf4")
                    print(f"Wrote {target}")
                    return target
                raise ValueError(
                    f"{output_file} lacks spatial dims for rect canvas merge."
                )

            canvas_coords = {
                row_dim: np.arange(grid_y),
                col_dim: np.arange(grid_x),
            }
            for dim_name, dim_size in sample_ds.sizes.items():
                if dim_name in (row_dim, col_dim):
                    continue
                canvas_coords[dim_name] = sample_ds[dim_name].values

            canvas_vars = {}
            for var_name, var_da in sample_ds.data_vars.items():
                dims = list(var_da.dims)
                shape = []
                for dim_name in dims:
                    if dim_name == row_dim:
                        shape.append(grid_y)
                    elif dim_name == col_dim:
                        shape.append(grid_x)
                    else:
                        shape.append(int(sample_ds.sizes[dim_name]))
                fill_value = var_da.attrs.get("_FillValue")
                if fill_value is None:
                    fill_arr = np.full(shape, np.nan, dtype=var_da.dtype)
                else:
                    fill_arr = np.full(shape, fill_value, dtype=var_da.dtype)
                canvas_vars[var_name] = (tuple(dims), fill_arr)

            canvas = xr.Dataset(canvas_vars, coords=canvas_coords, attrs=sample_ds.attrs.copy())
            for var_name in canvas.data_vars:
                canvas[var_name].attrs = sample_ds[var_name].attrs.copy()

        for batch_idx, file_path in available:
            if batch_idx >= len(blocks):
                print(
                    f"[WARN] Skipping {file_path}: batch index {batch_idx} "
                    "outside layout blocks."
                )
                continue
            y_start, y_end, x_start, x_end = blocks[batch_idx]
            with open_dataset_for_read(file_path, decode_cf=False) as batch_ds:
                batch_row_dim, batch_col_dim = self._resolve_spatial_dim_names(batch_ds)
                if batch_row_dim is None or batch_col_dim is None:
                    continue
                rename_map = {}
                if batch_row_dim != row_dim:
                    rename_map[batch_row_dim] = row_dim
                if batch_col_dim != col_dim:
                    rename_map[batch_col_dim] = col_dim
                batch_work = batch_ds.rename(rename_map) if rename_map else batch_ds
                for var_name in canvas.data_vars:
                    src_da = batch_work[var_name]
                    if row_dim not in src_da.dims or col_dim not in src_da.dims:
                        continue
                    dest_da = canvas[var_name]
                    dest_arr = dest_da.values
                    src_arr = src_da.values
                    dest_row_axis = dest_da.dims.index(row_dim)
                    dest_col_axis = dest_da.dims.index(col_dim)
                    src_row_axis = src_da.dims.index(row_dim)
                    src_col_axis = src_da.dims.index(col_dim)
                    dest_slices: list[slice | int] = [slice(None)] * dest_arr.ndim
                    dest_slices[dest_row_axis] = slice(y_start, y_end)
                    dest_slices[dest_col_axis] = slice(x_start, x_end)
                    src_slices: list[slice | int] = [slice(None)] * src_arr.ndim
                    src_slices[src_row_axis] = slice(0, y_end - y_start)
                    src_slices[src_col_axis] = slice(0, x_end - x_start)
                    dest_arr[tuple(dest_slices)] = src_arr[tuple(src_slices)]
                    canvas[var_name].values = dest_arr

        target = target_dir / output_file
        canvas.to_netcdf(target.as_posix(), engine="netcdf4")
        canvas.close()
        print(f"Wrote {target}")
        return target

    def _get_metadata_path(self) -> Path:
        return self.base_batch_dir / SPLIT_METADATA_FILENAME

    def _resolve_template_path(self, original_input_path: Path, output_file: str) -> Optional[Path]:
        candidate = original_input_path / output_file
        if candidate.exists():
            return candidate
        return None

    def execute(self) -> None:
        if not self.base_batch_dir.exists():
            raise FileNotFoundError(f"Batch path does not exist: {self.base_batch_dir}")

        metadata_path = self._get_metadata_path()
        if not metadata_path.exists():
            raise FileNotFoundError(
                "Missing WIEMIP split metadata. Run 'bp batch wiemip_split' first. "
                f"Expected: {metadata_path}"
            )
        metadata = read_split_metadata(metadata_path)
        split_mode, blocks, grid_y, grid_x = self._resolve_split_layout(metadata)
        original_input_path = Path(metadata.original_input_path)
        run_mask_path = original_input_path / metadata.run_mask_filename
        if not run_mask_path.exists():
            raise FileNotFoundError(
                f"Run-mask required for restore not found at {run_mask_path}"
            )

        with open_dataset_for_read(run_mask_path) as run_mask_ds:
            run_mask_da, run_mask_row_dim, run_mask_col_dim = extract_run_mask_2d(
                run_mask_ds,
                run_mask_path.name,
                run_var=RUN_MASK_VARIABLE,
            )
            run_mask_da = run_mask_da.load()

        batch_dirs = self._get_available_batches()
        if not batch_dirs:
            raise FileNotFoundError(
                f"No batch directories found under {self.base_batch_dir}"
            )

        print(f"Found {len(batch_dirs)} batches.")
        print(f"Merge mode: {split_mode}")
        output_files = self._get_output_files(batch_dirs)
        if not output_files:
            raise FileNotFoundError(
                f"No NetCDF output files found under {self.base_batch_dir / 'batch_*/output'}"
            )

        print(f"Merging {len(output_files)} NetCDF output files")
        for output_file in output_files:
            if split_mode == SPLIT_MODE_RECT:
                if blocks is None or grid_y is None or grid_x is None:
                    raise ValueError("Rect merge layout is missing block definitions.")
                filtered_path = self._merge_single_output_file_rect(
                    output_file,
                    batch_dirs,
                    blocks,
                    grid_y,
                    grid_x,
                    self.filtered_result_dir,
                )
            else:
                filtered_path = self._merge_single_output_file_y_stripe(
                    output_file, batch_dirs, self.filtered_result_dir
                )
            if filtered_path is None:
                continue

            template_path = self._resolve_template_path(original_input_path, output_file)
            template_ds = None
            if template_path is not None:
                template_ds = open_dataset_for_read(template_path, decode_cf=False)
            try:
                with open_dataset_for_read(filtered_path, decode_cf=False) as filtered_ds:
                    restored_ds = restore_filtered_dataset_to_full_grid(
                        filtered_ds=filtered_ds,
                        run_mask_da=run_mask_da,
                        run_row_dim=run_mask_row_dim,
                        run_col_dim=run_mask_col_dim,
                        bbox=metadata.bbox,
                        template_ds=template_ds,
                        active_value=metadata.active_value,
                    )
                    restored_path = self.restored_result_dir / output_file
                    restored_ds.to_netcdf(restored_path.as_posix(), engine="netcdf4")
                    restored_ds.close()
                    print(f"Wrote {restored_path}")
            finally:
                if template_ds is not None:
                    template_ds.close()
