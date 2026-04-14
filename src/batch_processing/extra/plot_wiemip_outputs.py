from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from netCDF4 import Dataset

DEFAULT_INPUT = (
    "/mnt/exacloud/ejafarov_woodwellclimate_org/wiemip/test_gfdl_split/"
    "wiemip_merged/merged_restored/GPP_yearly_tr.nc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot WIEMIP NetCDF outputs as a summary figure "
            "(first map, last map, and mean-over-time)."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT,
        help=(
            "Path to a NetCDF file or a directory with NetCDF files. "
            f"Default: {DEFAULT_INPUT}"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output PDF path. If omitted, writes to <input>.pdf for files "
            "or summary_plots.pdf inside the input directory."
        ),
    )
    parser.add_argument(
        "--variable",
        "-v",
        default=None,
        help=(
            "Variable name in NetCDF to plot. If omitted, inferred from "
            "filename (text before first underscore) or first data variable."
        ),
    )
    return parser.parse_args()


def infer_variable_name(nc_path: Path, explicit_name: str | None, nc: Dataset) -> str:
    if explicit_name:
        return explicit_name

    inferred = nc_path.name.split("_")[0]
    if inferred in nc.variables:
        return inferred

    for variable_name in nc.variables:
        if variable_name in nc.dimensions:
            continue
        return variable_name

    raise ValueError(f"Could not infer variable from {nc_path}")


def find_time_dim(var_dims: tuple[str, ...], dimensions: dict) -> str:
    for candidate in ("time", "Time", "TIME", "t"):
        if candidate in var_dims:
            return candidate

    for dim in var_dims:
        if dim.lower().startswith("time"):
            return dim

    for dim in var_dims:
        if dimensions[dim].size > 1:
            return dim

    raise ValueError(f"Could not detect time dimension from {var_dims}")


def reduce_to_time_y_x(data: np.ndarray, var_dims: tuple[str, ...], time_dim: str) -> np.ndarray:
    spatial_candidates = [dim for dim in var_dims if dim != time_dim]
    if len(spatial_candidates) < 2:
        raise ValueError(
            f"Variable must have at least 2 spatial dimensions in addition "
            f"to time. Found dimensions: {var_dims}"
        )

    # Follow reference script behavior: keep first index for extra dims.
    selected_data = data
    selected_dims = list(var_dims)
    for dim in list(selected_dims):
        if dim == time_dim:
            continue
        if len([d for d in selected_dims if d != time_dim]) <= 2:
            break
        axis = selected_dims.index(dim)
        selected_data = np.take(selected_data, indices=0, axis=axis)
        selected_dims.pop(axis)

    time_axis = selected_dims.index(time_dim)
    non_time_dims = [dim for dim in selected_dims if dim != time_dim]
    if len(non_time_dims) != 2:
        raise ValueError(
            "Unable to reduce variable to [time, y, x]. "
            f"Remaining dimensions: {selected_dims}"
        )

    y_axis = selected_dims.index(non_time_dims[0])
    x_axis = selected_dims.index(non_time_dims[1])
    transposed = np.transpose(selected_data, axes=[time_axis, y_axis, x_axis])
    return transposed


def clean_data(data: np.ndarray, variable) -> np.ndarray:
    values = data
    if isinstance(values, np.ma.MaskedArray):
        values = np.ma.filled(values, np.nan)
    values = np.asarray(values, dtype=float)

    if hasattr(variable, "_FillValue"):
        values = np.where(values == variable._FillValue, np.nan, values)
    if hasattr(variable, "missing_value"):
        values = np.where(values == variable.missing_value, np.nan, values)

    values = np.where(np.isclose(values, -9999.0), np.nan, values)
    values[~np.isfinite(values)] = np.nan
    return values


def maybe_annual_average(data: np.ndarray) -> tuple[np.ndarray, str]:
    time_size = data.shape[0]
    if time_size > 500 and time_size % 12 == 0:
        years = time_size // 12
        reshaped = data.reshape(years, 12, data.shape[1], data.shape[2])
        averaged = np.nanmean(reshaped, axis=1)
        return averaged, " (annual mean from monthly)"
    return data, ""


def trim_nan_borders(data: np.ndarray) -> np.ndarray:
    """
    Remove outer rows/cols that are NaN for all timesteps.
    """
    valid_mask_2d = np.any(np.isfinite(data), axis=0)

    row_idx = np.where(np.any(valid_mask_2d, axis=1))[0]
    col_idx = np.where(np.any(valid_mask_2d, axis=0))[0]

    if row_idx.size == 0 or col_idx.size == 0:
        return data

    row_start, row_end = int(row_idx[0]), int(row_idx[-1]) + 1
    col_start, col_end = int(col_idx[0]), int(col_idx[-1]) + 1
    return data[:, row_start:row_end, col_start:col_end]


def rotate_map_90_ccw(slice_2d: np.ndarray) -> np.ndarray:
    return np.rot90(slice_2d, k=1)


def make_figure(
    data: np.ndarray,
    variable_name: str,
    units: str,
    source_name: str,
    averaging_note: str,
) -> plt.Figure:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError(f"{variable_name} has no finite values.")

    vmin = float(np.nanpercentile(finite, 2))
    vmax = float(np.nanpercentile(finite, 98))
    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    mean_series = np.nanmean(data, axis=(1, 2))
    std_series = np.nanstd(data, axis=(1, 2))
    time_steps = np.arange(data.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    label = f"{variable_name} ({units})" if units else variable_name

    # Rotate only the two map images (left and center panels) by 90 deg CCW.
    first = rotate_map_90_ccw(data[0])
    last = rotate_map_90_ccw(data[-1])

    im0 = axes[0].imshow(first, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{variable_name} - first timestep")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig.colorbar(im0, ax=axes[0], label=label)

    im1 = axes[1].imshow(last, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{variable_name} - last timestep")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    fig.colorbar(im1, ax=axes[1], label=label)

    axes[2].plot(time_steps, mean_series, color="tab:blue", label=f"Mean {variable_name}")
    axes[2].fill_between(
        time_steps,
        mean_series - std_series,
        mean_series + std_series,
        color="tab:blue",
        alpha=0.2,
        label="±1 std dev",
    )
    axes[2].set_xlabel("Time index")
    axes[2].set_ylabel(label)
    axes[2].set_title(f"{variable_name} over time{averaging_note}")
    axes[2].legend(loc="best")

    fig.suptitle(source_name, fontsize=10)
    fig.tight_layout()
    return fig


def plot_single_file(nc_path: Path, variable_override: str | None) -> plt.Figure:
    with Dataset(nc_path, "r") as nc:
        variable_name = infer_variable_name(nc_path, variable_override, nc)
        if variable_name not in nc.variables:
            raise KeyError(
                f"Variable '{variable_name}' not found in {nc_path}. "
                f"Available variables: {list(nc.variables.keys())}"
            )

        variable = nc.variables[variable_name]
        raw = variable[:]
        cleaned = clean_data(raw, variable)
        time_dim = find_time_dim(variable.dimensions, nc.dimensions)
        reduced = reduce_to_time_y_x(cleaned, variable.dimensions, time_dim)
        reduced, averaging_note = maybe_annual_average(reduced)
        reduced = trim_nan_borders(reduced)

        if not np.any(np.isfinite(reduced)):
            raise ValueError(f"{variable_name} in {nc_path} has no valid data.")

        units = getattr(variable, "units", "")
        return make_figure(
            reduced,
            variable_name,
            units,
            source_name=nc_path.name,
            averaging_note=averaging_note,
        )


def collect_netcdf_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".nc":
            raise ValueError(f"Input file is not NetCDF: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.glob("*.nc"))
        if not files:
            raise ValueError(f"No .nc files found in directory: {input_path}")
        return files

    raise FileNotFoundError(f"Input path not found: {input_path}")


def resolve_output_path(input_path: Path, output: str | None) -> Path:
    if output:
        return Path(output).expanduser()
    if input_path.is_file():
        return input_path.with_suffix(".plots.pdf")
    return input_path / "summary_plots.pdf"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()
    files = collect_netcdf_files(input_path)
    output_pdf = resolve_output_path(input_path, args.output)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for nc_file in files:
            try:
                fig = plot_single_file(nc_file, args.variable)
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {nc_file.name}: {exc}")
                continue
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Added plot for {nc_file.name}")

    print(f"Saved plots to {output_pdf}")


if __name__ == "__main__":
    main()
