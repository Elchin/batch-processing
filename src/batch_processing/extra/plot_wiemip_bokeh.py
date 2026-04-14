from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, ColorBar, ColumnDataSource, Div, LinearColorMapper, Slider
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

DEFAULT_INPUT_PATH = (
    "/mnt/exacloud/ejafarov_woodwellclimate_org/wiemip/test_gfdl_split/"
    "wiemip_merged/merged_restored/GPP_yearly_tr.nc"
)
DEFAULT_VARIABLE = "GPP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Bokeh map viewer for WIEMIP NetCDF files."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to NetCDF file (default: %(default)s).",
    )
    parser.add_argument(
        "--variable",
        default=DEFAULT_VARIABLE,
        help="Variable name to display (default: %(default)s).",
    )
    parser.add_argument(
        "--time-dim",
        default="time",
        help="Time dimension name (default: %(default)s).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=400,
        help="Animation interval in milliseconds (default: %(default)s).",
    )
    return parser.parse_args()


def infer_time_dim(data_array: xr.DataArray, requested: str) -> str:
    if requested in data_array.dims:
        return requested

    for candidate in ("time", "Time", "TIME", "t"):
        if candidate in data_array.dims:
            return candidate

    raise ValueError(
        f"Could not find a time dimension. Requested '{requested}', "
        f"available dims: {data_array.dims}"
    )


def to_time_label(time_values: np.ndarray, index: int) -> str:
    if len(time_values) == 0:
        return str(index)

    value = time_values[index]
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="D")
    return str(value)


def prepare_data_array(data_array: xr.DataArray, time_dim: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    da = data_array.squeeze(drop=True)

    non_time_dims = [dim for dim in da.dims if dim != time_dim]
    if len(non_time_dims) < 2:
        raise ValueError(
            f"Variable must have at least 2 spatial dimensions. Found dims: {da.dims}"
        )

    # If additional dimensions exist (for example pft/layer), pin them to index 0.
    if len(non_time_dims) > 2:
        selection = {dim: 0 for dim in non_time_dims[:-2]}
        da = da.isel(selection)

    spatial_dims = [dim for dim in da.dims if dim != time_dim]
    if len(spatial_dims) != 2:
        raise ValueError(f"Unable to reduce variable to time + 2D map. Final dims: {da.dims}")

    y_dim, x_dim = spatial_dims
    da = da.transpose(time_dim, y_dim, x_dim)

    values = da.values
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    values = np.asarray(values, dtype=float)

    fill_value = da.attrs.get("_FillValue")
    if fill_value is not None:
        values[values == fill_value] = np.nan

    missing_value = da.attrs.get("missing_value")
    if missing_value is not None:
        values[values == missing_value] = np.nan

    values[~np.isfinite(values)] = np.nan

    if time_dim in da.coords:
        time_values = np.asarray(da[time_dim].values)
    else:
        time_values = np.arange(values.shape[0])

    return values, time_values, y_dim, x_dim


args = parse_args()
input_path = Path(args.input).expanduser()
if not input_path.exists():
    raise FileNotFoundError(f"NetCDF file does not exist: {input_path}")

with xr.open_dataset(input_path, decode_times=True) as dataset:
    if args.variable not in dataset.variables:
        available = ", ".join(dataset.data_vars.keys())
        raise KeyError(
            f"Variable '{args.variable}' not found in {input_path}. Available data vars: {available}"
        )
    variable_data = dataset[args.variable].load()

time_dim = infer_time_dim(variable_data, args.time_dim)
data, time_values, y_dim, x_dim = prepare_data_array(variable_data, time_dim)

num_steps = data.shape[0]
if num_steps == 0:
    raise ValueError("No time steps found in selected variable.")

finite = data[np.isfinite(data)]
if finite.size == 0:
    color_low, color_high = 0.0, 1.0
else:
    color_low, color_high = np.nanpercentile(finite, [2, 98])
    if np.isclose(color_low, color_high):
        color_low -= 1.0
        color_high += 1.0

source = ColumnDataSource(data={"image": [data[0]]})
color_mapper = LinearColorMapper(
    palette=Viridis256,
    low=float(color_low),
    high=float(color_high),
    nan_color="#d9d9d9",
)

plot = figure(
    title=f"{args.variable} time index 0",
    x_axis_label=x_dim,
    y_axis_label=y_dim,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    sizing_mode="stretch_both",
)
plot.image(
    image="image",
    x=0,
    y=0,
    dw=data.shape[2],
    dh=data.shape[1],
    source=source,
    color_mapper=color_mapper,
)
plot.add_layout(ColorBar(color_mapper=color_mapper), "right")
plot.x_range.range_padding = 0
plot.y_range.range_padding = 0

time_slider = Slider(
    title="Time step",
    start=0,
    end=num_steps - 1,
    value=0,
    step=1,
    width=480,
)
play_button = Button(label="Play", button_type="success", width=80)
prev_button = Button(label="Prev", button_type="default", width=70)
next_button = Button(label="Next", button_type="default", width=70)
time_info = Div(text="", width=900)


def update_frame(index: int) -> None:
    source.data = {"image": [data[index]]}
    time_label = to_time_label(time_values, index)
    time_info.text = (
        f"<b>{time_dim}</b>: {time_label} | "
        f"frame {index + 1}/{num_steps}"
    )
    plot.title.text = f"{args.variable} at {time_label}"


def on_time_change(attr: str, old: int, new: int) -> None:
    del attr, old
    update_frame(new)


time_slider.on_change("value", on_time_change)

state = {"callback_id": None}


def animate() -> None:
    time_slider.value = (time_slider.value + 1) % num_steps


def toggle_animation() -> None:
    callback_id = state["callback_id"]
    if callback_id is None:
        state["callback_id"] = curdoc().add_periodic_callback(animate, args.interval_ms)
        play_button.label = "Pause"
        play_button.button_type = "warning"
    else:
        curdoc().remove_periodic_callback(callback_id)
        state["callback_id"] = None
        play_button.label = "Play"
        play_button.button_type = "success"


def go_previous() -> None:
    time_slider.value = (time_slider.value - 1) % num_steps


def go_next() -> None:
    time_slider.value = (time_slider.value + 1) % num_steps


play_button.on_click(toggle_animation)
prev_button.on_click(go_previous)
next_button.on_click(go_next)

controls = row(play_button, prev_button, next_button, time_slider, sizing_mode="stretch_width")
layout = column(controls, time_info, plot, sizing_mode="stretch_both")

document = curdoc()
document.title = f"{args.variable} time viewer"
document.add_root(layout)

update_frame(0)
