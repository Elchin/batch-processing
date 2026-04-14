from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import xarray as xr

ROW_DIM_CANDIDATES = ("Y", "y", "latitude", "lat")
COL_DIM_CANDIDATES = ("X", "x", "longitude", "lon")
RUN_MASK_VARIABLE = "run"
RUN_ENABLED_VALUE = 1
SPLIT_METADATA_FILENAME = "wiemip_split_metadata.json"


@dataclass(frozen=True)
class ActiveBBox:
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def n_rows(self) -> int:
        return self.row_end - self.row_start + 1

    @property
    def n_cols(self) -> int:
        return self.col_end - self.col_start + 1


@dataclass
class WiemipSplitMetadata:
    schema_version: int
    original_input_path: str
    filtered_staging_path: str
    run_mask_filename: str
    row_dim: str
    col_dim: str
    active_value: int
    full_rows: int
    full_cols: int
    active_bbox: dict[str, int]
    file_mappings: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "original_input_path": self.original_input_path,
            "filtered_staging_path": self.filtered_staging_path,
            "run_mask_filename": self.run_mask_filename,
            "row_dim": self.row_dim,
            "col_dim": self.col_dim,
            "active_value": self.active_value,
            "full_rows": self.full_rows,
            "full_cols": self.full_cols,
            "active_bbox": self.active_bbox,
            "file_mappings": self.file_mappings,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WiemipSplitMetadata":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            original_input_path=str(payload["original_input_path"]),
            filtered_staging_path=str(payload["filtered_staging_path"]),
            run_mask_filename=str(payload.get("run_mask_filename", "run-mask.nc")),
            row_dim=str(payload["row_dim"]),
            col_dim=str(payload["col_dim"]),
            active_value=int(payload.get("active_value", RUN_ENABLED_VALUE)),
            full_rows=int(payload["full_rows"]),
            full_cols=int(payload["full_cols"]),
            active_bbox=dict(payload["active_bbox"]),
            file_mappings=dict(payload.get("file_mappings", {})),
        )

    @property
    def bbox(self) -> ActiveBBox:
        return ActiveBBox(
            row_start=int(self.active_bbox["row_start"]),
            row_end=int(self.active_bbox["row_end"]),
            col_start=int(self.active_bbox["col_start"]),
            col_end=int(self.active_bbox["col_end"]),
        )


def write_split_metadata(path: Path, metadata: WiemipSplitMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(metadata.to_dict(), fp, indent=2)


def read_split_metadata(path: Path) -> WiemipSplitMetadata:
    with open(path, "r", encoding="utf-8") as fp:
        return WiemipSplitMetadata.from_dict(json.load(fp))


def open_dataset_for_read(path: Path, decode_cf: bool = False) -> xr.Dataset:
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


def detect_spatial_dims(dim_names: Iterable[str]) -> Optional[Tuple[str, str]]:
    dim_name_set = set(dim_names)
    for row_dim in ROW_DIM_CANDIDATES:
        if row_dim not in dim_name_set:
            continue
        for col_dim in COL_DIM_CANDIDATES:
            if col_dim in dim_name_set:
                return row_dim, col_dim
    return None


def extract_run_mask_2d(
    ds: xr.Dataset,
    source_label: str,
    run_var: str = RUN_MASK_VARIABLE,
) -> tuple[xr.DataArray, str, str]:
    if run_var not in ds:
        raise KeyError(f"{source_label} must contain '{run_var}' variable.")

    run_da = ds[run_var]
    spatial_dims = detect_spatial_dims(run_da.dims)
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


def compute_active_bbox(
    run_mask_da: xr.DataArray, active_value: int = RUN_ENABLED_VALUE
) -> ActiveBBox:
    run_values = np.asarray(run_mask_da.values)
    active_mask = np.isfinite(run_values) & np.isclose(run_values, active_value)
    active_indices = np.argwhere(active_mask)
    if active_indices.size == 0:
        raise ValueError("run-mask contains no enabled run==1 cells.")
    return ActiveBBox(
        row_start=int(active_indices[:, 0].min()),
        row_end=int(active_indices[:, 0].max()),
        col_start=int(active_indices[:, 1].min()),
        col_end=int(active_indices[:, 1].max()),
    )


def _bbox_indexer(row_dim: str, col_dim: str, bbox: ActiveBBox) -> dict[str, slice]:
    return {
        row_dim: slice(bbox.row_start, bbox.row_end + 1),
        col_dim: slice(bbox.col_start, bbox.col_end + 1),
    }


def filter_dataset_to_cropped_mask(
    in_ds: xr.Dataset,
    run_mask_da: xr.DataArray,
    run_row_dim: str,
    run_col_dim: str,
    ds_row_dim: str,
    ds_col_dim: str,
    bbox: ActiveBBox,
    active_value: int = RUN_ENABLED_VALUE,
) -> xr.Dataset:
    full_rows = int(run_mask_da.sizes[run_row_dim])
    full_cols = int(run_mask_da.sizes[run_col_dim])
    bbox_rows = bbox.n_rows
    bbox_cols = bbox.n_cols
    ds_rows = int(in_ds.sizes[ds_row_dim])
    ds_cols = int(in_ds.sizes[ds_col_dim])

    if ds_rows == full_rows and ds_cols == full_cols:
        ds_work = in_ds.isel(_bbox_indexer(ds_row_dim, ds_col_dim, bbox))
        mask_work = run_mask_da.isel(_bbox_indexer(run_row_dim, run_col_dim, bbox))
    elif ds_rows == bbox_rows and ds_cols == bbox_cols:
        ds_work = in_ds
        mask_work = run_mask_da.isel(_bbox_indexer(run_row_dim, run_col_dim, bbox))
    else:
        raise ValueError(
            f"Unexpected spatial shape {ds_row_dim}/{ds_col_dim}={ds_rows}/{ds_cols}. "
            f"Expected full {full_rows}/{full_cols} or bbox {bbox_rows}/{bbox_cols}."
        )

    if (run_row_dim, run_col_dim) != (ds_row_dim, ds_col_dim):
        mask_work = mask_work.rename({run_row_dim: ds_row_dim, run_col_dim: ds_col_dim})
    mask_work = mask_work.assign_coords(
        {
            ds_row_dim: ds_work[ds_row_dim].values,
            ds_col_dim: ds_work[ds_col_dim].values,
        }
    )

    active_mask = xr.where(
        np.isfinite(mask_work) & np.isclose(mask_work, active_value), True, False
    )
    out_ds = xr.Dataset(coords=ds_work.coords, attrs=in_ds.attrs.copy())
    for var_name in ds_work.data_vars:
        var_da = ds_work[var_name]
        if ds_row_dim in var_da.dims and ds_col_dim in var_da.dims:
            out_ds[var_name] = var_da.where(active_mask)
        else:
            out_ds[var_name] = var_da
    return out_ds


def restore_filtered_dataset_to_full_grid(
    filtered_ds: xr.Dataset,
    run_mask_da: xr.DataArray,
    run_row_dim: str,
    run_col_dim: str,
    bbox: ActiveBBox,
    template_ds: Optional[xr.Dataset] = None,
    active_value: int = RUN_ENABLED_VALUE,
) -> xr.Dataset:
    spatial_dims = detect_spatial_dims(filtered_ds.dims)
    if spatial_dims is None:
        return filtered_ds.copy()

    ds_row_dim, ds_col_dim = spatial_dims
    full_rows = int(run_mask_da.sizes[run_row_dim])
    full_cols = int(run_mask_da.sizes[run_col_dim])
    ds_rows = int(filtered_ds.sizes[ds_row_dim])
    ds_cols = int(filtered_ds.sizes[ds_col_dim])

    if (ds_row_dim, ds_col_dim) != (run_row_dim, run_col_dim):
        filtered_work = filtered_ds.rename({ds_row_dim: run_row_dim, ds_col_dim: run_col_dim})
    else:
        filtered_work = filtered_ds

    if ds_rows == bbox.n_rows and ds_cols == bbox.n_cols:
        filtered_work = filtered_work.assign_coords(
            {
                run_row_dim: run_mask_da[run_row_dim]
                .isel({run_row_dim: slice(bbox.row_start, bbox.row_end + 1)})
                .values,
                run_col_dim: run_mask_da[run_col_dim]
                .isel({run_col_dim: slice(bbox.col_start, bbox.col_end + 1)})
                .values,
            }
        )
        filtered_work = filtered_work.reindex(
            {
                run_row_dim: run_mask_da[run_row_dim].values,
                run_col_dim: run_mask_da[run_col_dim].values,
            }
        )
    elif ds_rows == full_rows and ds_cols == full_cols:
        filtered_work = filtered_work.assign_coords(
            {
                run_row_dim: run_mask_da[run_row_dim].values,
                run_col_dim: run_mask_da[run_col_dim].values,
            }
        )
    else:
        raise ValueError(
            f"Unexpected filtered spatial shape {run_row_dim}/{run_col_dim}={ds_rows}/{ds_cols}. "
            f"Expected bbox {bbox.n_rows}/{bbox.n_cols} or full {full_rows}/{full_cols}."
        )

    active_mask_full = xr.where(
        np.isfinite(run_mask_da) & np.isclose(run_mask_da, active_value), True, False
    )
    restored_vars: dict[str, xr.DataArray] = {}
    for var_name in filtered_work.data_vars:
        da = filtered_work[var_name]
        if run_row_dim in da.dims and run_col_dim in da.dims:
            if template_ds is not None and var_name in template_ds.data_vars:
                tmpl = template_ds[var_name]
                restored = tmpl.where(~active_mask_full, da)
                restored = restored.astype(tmpl.dtype)
                restored.attrs = tmpl.attrs.copy()
            else:
                restored = da.where(active_mask_full)
                restored.attrs = filtered_work[var_name].attrs.copy()
            restored_vars[var_name] = restored
        else:
            restored_vars[var_name] = da

    out_ds = xr.Dataset(restored_vars)
    out_ds = out_ds.assign_coords(
        {
            run_row_dim: run_mask_da[run_row_dim].values,
            run_col_dim: run_mask_da[run_col_dim].values,
        }
    )
    for coord_name, coord in filtered_work.coords.items():
        if coord_name in (run_row_dim, run_col_dim):
            continue
        if coord_name not in out_ds.coords:
            out_ds = out_ds.assign_coords({coord_name: coord})

    if template_ds is not None:
        out_ds.attrs = template_ds.attrs.copy()
    else:
        out_ds.attrs = filtered_work.attrs.copy()
    out_ds.attrs["restored_from_filtered"] = "true"
    return out_ds
