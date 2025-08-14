import os

from dashboard.styles import DASHBOARD_RAW_CSS

import panel as pn
import holoviews as hv
import geoviews as gv
import datetime
import pandas as pd
import xarray as xr
import numpy as np
from typing import Optional, Any

# Set resource paths once at the top, relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")
FAVICON_PATH = os.path.join(RESOURCES_DIR, "favicon.ico")
LOGO_PATH = os.path.join(RESOURCES_DIR, "logo.png")
ABOUT_PDF_PATH = os.path.join(RESOURCES_DIR, "about.pdf")
print(ABOUT_PDF_PATH)
# Use pn.extension to set favicon (works with panel serve)
pn.extension(
    favicon=FAVICON_PATH,
    raw_css=[DASHBOARD_RAW_CSS],
)

gv.extension("bokeh")
hv.extension("bokeh")


"""

# For sharing HTML, use relative paths for all resources
FAVICON_PATH = "resources/favicon.ico"
LOGO_PATH = "resources/logo.png"
ABOUT_PDF_PATH = "resources/about.pdf"
"""


def create_fflies_dashboard(
    netcdf_path: str,
    save_path: Optional[str] = None,
    favicon_path: Optional[str] = None,
    logo_path: Optional[str] = None,
    about_pdf_path: Optional[str] = None,
) -> pn.Column:
    """
    Create the interactive dashboard layout for FfliesOutput.plot.

    Args:
        netcdf_path: Path to the NetCDF file (must contain required metadata).
        save_path: Optional path to save the dashboard HTML.
        favicon_path: Optional path to favicon.
        logo_path: Optional path to logo.
        about_pdf_path: Optional path to about PDF.

    Returns:
        Panel Column layout for the dashboard.
    """
    ds = xr.open_dataset(netcdf_path)
    # da = next(iter(ds.data_vars.values()))
    da = ds["days_to_completion"]  # Use the specific variable name from the dataset
    most_likely = ds["most_likely_completion_date"]
    latest_likely = ds["latest_likely_completion_date"]
    range_likely = ds["range_of_completion_dates"]
    species = ds.attrs.get("species", "Unknown")
    detection_date = ds.attrs.get("detection_date", "2000-01-01")
    latitude = float(ds.attrs.get("latitude", ds.coords["latitude"].values.mean()))
    longitude = float(ds.attrs.get("longitude", ds.coords["longitude"].values.mean()))
    generations = ds.coords["generation"].values

    # Use provided paths if given, else use the relative ones
    favicon = favicon_path or FAVICON_PATH
    logo = logo_path or LOGO_PATH
    about_pdf = about_pdf_path or ABOUT_PDF_PATH

    da = da.sortby(["latitude", "longitude"])
    years = da.coords["year"].values
    year_labels = {f"sim{y}": y for y in da.coords["year"].values}
    custom_layers = {
        "Most Likely Completion Date": "most_likely",
        "Latest Likely Completion Date": "latest_likely",
        "Range (All Years)": "range",
    }
    year_options = {**custom_layers, **year_labels}
    select_styles = {
        "color": "#1E1E1D",
        "font-size": "16px",
        "font-family": "Inter, Arial, sans-serif",
        "background": "#E7E7D6",
        "border": "1px solid #AEAC92",
        "border-radius": "6px",
        "padding": "4px 8px",
        "margin-bottom": "10px",
    }
    slider_styles = {
        "color": "#1E1E1D",
        "font-size": "16px",
        "font-family": "Inter, Arial, sans-serif",
        "background": "#E7E7D6",
        "border": "1px solid #AEAC92",
        "border-radius": "6px",
        "padding": "4px 8px",
        "margin-bottom": "10px",
    }
    year_select = pn.widgets.Select(
        name="Layer / Year",
        options=year_options,
        value="most_likely",  # changed from "mean" to "most_likely"
        styles=select_styles,
        width=160,
    )
    gen_select = pn.widgets.Select(
        name="Generation",
        options=generations.tolist(),
        value=generations[2] if len(generations) > 2 else generations[0],
        styles=select_styles,
        width=160,
    )
    # Transparency button with color depending on state, with bounding box
    alpha_button = pn.widgets.Toggle(
        name="Transparency",
        value=False,  # False: alpha=0.9, True: alpha=0.5
        button_type="default",  # Use 'default' to avoid built-in color
        width=160,
        styles=slider_styles,
    )

    def update_alpha_button_color(event):
        # Use only custom styles for color, keep button_type 'default'
        if event.new:
            alpha_button.button_type = "default"
            alpha_button.button_style = "solid"
            alpha_button.styles = {
                **slider_styles,
                "background": "#e87b4d",
                "color": "#fff",
                "border": "1px solid #e87b4d",
            }
        else:
            alpha_button.button_type = "default"
            alpha_button.button_style = "solid"
            alpha_button.styles = {
                **slider_styles,
                "background": "#AEAC92",
                "color": "#1E1E1D",
                "border": "1px solid #AEAC92",
            }

    # Set initial color
    update_alpha_button_color(type("evt", (), {"new": alpha_button.value})())
    alpha_button.param.watch(update_alpha_button_color, "value")

    # Compute global min/max for all generations for consistent color mapping
    global_min = float(da.min())
    global_max = float(da.max())
    global_most_likely_min = float(latest_likely.min())
    global_most_likely_max = float(latest_likely.max())
    global_latest_likely_min = float(latest_likely.min())
    global_latest_likely_max = float(latest_likely.max())
    global_range_min = float(range_likely.min())
    global_range_max = float(range_likely.max())

    clim_per_gen = {}
    clim_most_likely_per_gen = {}
    clim_latest_likely_per_gen = {}
    clim_range_per_gen = {}
    for gen in generations:
        gen_key = gen.item() if hasattr(gen, "item") else gen
        # Use global min/max for all generations
        clim_per_gen[gen_key] = (global_min, global_max)
        clim_most_likely_per_gen[gen_key] = (
            global_most_likely_min,
            global_most_likely_max,
        )
        clim_latest_likely_per_gen[gen_key] = (
            global_latest_likely_min,
            global_latest_likely_max,
        )
        clim_range_per_gen[gen_key] = (global_range_min, global_range_max)

    def make_plot(year_or_stat, generation, low_transparency):
        gen_key = generation.item() if hasattr(generation, "item") else generation
        alpha = 0.65 if low_transparency else 0.94

        if year_or_stat == "most_likely":
            sliced = most_likely.sel(generation=generation)
            clim = clim_most_likely_per_gen[gen_key]
            cmap = "Viridis"
        elif year_or_stat == "latest_likely":
            sliced = latest_likely.sel(generation=generation)
            clim = clim_latest_likely_per_gen[gen_key]
            cmap = "Viridis"
        elif year_or_stat == "range":
            sliced = range_likely.sel(generation=generation)
            clim = clim_range_per_gen[gen_key]
            cmap = "Magma"
        else:
            try:
                sliced = da.sel(year=year_or_stat, generation=generation)
            except KeyError:
                sliced = da.sel(year=years[0], generation=generation)
            clim = clim_per_gen[gen_key]
            cmap = "Cividis"

        # --- Add Completion Date calculation ---
        # detection_date is a string, convert to pandas.Timestamp
        det_date = pd.to_datetime(detection_date)
        # Ensure sliced is a DataArray of days to completion
        days_to_completion = sliced
        # Calculate completion date as a DataArray of strings (ISO format)
        completion_date = xr.apply_ufunc(
            lambda days: (det_date + pd.to_timedelta(days, unit="D")).strftime(
                "%Y-%m-%d"
            ),
            days_to_completion,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[str],
        )
        # Compose a new DataArray with both days and completion date as variables
        # We'll use a tuple for vdims, but holoviews expects a Dataset for multiple vdims
        ds = xr.Dataset(
            {
                "Days to Completion": days_to_completion,
                "Completion Date": completion_date,
            }
        )

        img = gv.Image(
            ds,
            kdims=["longitude", "latitude"],
            vdims=["Days to Completion", "Completion Date"],
        ).opts(
            cmap=cmap,
            alpha=alpha,
            colorbar=True,
            tools=["hover", "tap"],  # Ensure tap tool is present
            clim=clim,
            bgcolor="#E7E7D6",
            xaxis=None,
            yaxis=None,
            framewise=True,
            hooks=[map_tap_hook],  # <-- Add the tap hook here
        )
        tiles = gv.tile_sources.OSM.opts(alpha=1.0, bgcolor="#E7E7D6")
        point = gv.Points(
            [(longitude, latitude)], kdims=["longitude", "latitude"]
        ).opts(
            color="#e87b4d",
            size=12,
            marker="o",
            line_color="black",
            nonselection_alpha=1.0,
        )
        circle_radius_km = 15
        circle_radius_deg = circle_radius_km / 111.32
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = longitude + circle_radius_deg * np.cos(theta)
        circle_y = latitude + circle_radius_deg * np.sin(theta)
        circle_path = gv.Path(
            [np.column_stack([circle_x, circle_y])], kdims=["longitude", "latitude"]
        ).opts(color="#e87b4d", line_width=2, alpha=1.0)
        return tiles * img * point * circle_path

    plot_pane = pn.bind(
        make_plot,
        year_or_stat=year_select,
        generation=gen_select,
        low_transparency=alpha_button,
    )
    # Modern left menu with a card and a title

    model_run_date = datetime.date.today().isoformat()  # get today's date

    # Path to your explanatory PDF (adjust as needed)
    # ABOUT_PDF_PATH = "../../resources/about.pdf"  # or a full URL if hosted elsewhere

    # Place logo and about above the rest of the layout, not in a Row with the menu
    import base64

    def get_base64_image(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    # In your dashboard code:
    logo_data_uri = get_base64_image(logo)
    logo_pane = pn.pane.HTML(
        f"<img src='{logo_data_uri}' width='180' style='margin:0;display:block;align-items:start;'/>",
        width=180,
        margin=(0, 0, 0, 0),
        align="start",
    )
    about_link = pn.pane.HTML(
        f"<a href='{about_pdf}' target='_blank' style='color:#1E1E1D;font-size:16px;text-decoration:underline;'>About</a>",
        width=60,
        margin=(0, 30, 0, 0),
        align="end",
    )
    # Place logo and about in their own row, above the rest of the layout, and remove menu left margin
    logo_row = pn.Row(
        logo_pane,
        pn.Spacer(sizing_mode="stretch_width"),
        about_link,
        sizing_mode="stretch_width",
        height=120,
    )

    # Step 1: Define descriptions for each map layer/year
    layer_descriptions = {
        "most_likely": (
            "The most likely (mean) completion dates across all sub-simulated years."
        ),
        "latest_likely": (
            "Shows the latest likely completion dates (maximum across sub-simulations)."
        ),
        "range": (
            "The range (max-min) of completion dates across all sub-simulations. "
            "Use to identify areas with high inter-annual variation in completion dates."
        ),
        # Add a default for years (simXXXX)
        "default": ("Shows the completion dates for the selected sub-simulation year."),
    }

    # Step 2: Function to get the description based on selection
    def get_layer_description(year_or_stat):
        if year_or_stat in layer_descriptions:
            return layer_descriptions[year_or_stat]
        else:
            return layer_descriptions["default"]

    # Step 3: Use a single Markdown pane and update its content via a watcher (no bind/depends)
    description_md = pn.pane.Markdown("", margin=(0, 0, 10, 0))

    def update_description(event=None):
        # Use .value directly to avoid event issues
        y = year_select.value
        description_md.object = f"<span style='color:#1E1E1D; font-size:14px'>{get_layer_description(y)}</span>"

    # Initial set
    update_description()
    # Remove value_throttled watcher, only use value
    year_select.param.watch(update_description, "value")
    # Remove alpha_button watcher, only update on year change

    menu = pn.Card(
        pn.pane.Markdown(
            f"<div style='color:#1E1E1D;font-size:18px;'><strong>Species: {species} </strong><br>"
            f"<b>Detection Date:</b> {detection_date}<br>"
            f"<b>Model Run Date:</b> {model_run_date}<br>"
            f"<b>Location:</b> ({latitude}, {longitude})<br></div>",
            sizing_mode="stretch_width",
        ),
        year_select,
        gen_select,
        alpha_button,
        description_md,  # Use the single Markdown pane here
        title="Options",
        width=300,
        margin=(10, 10, 20, 10),
        sizing_mode="stretch_height",
        css_classes=["fflies-light-card"],
    )
    # --- Add: Scatterplot modal setup ---
    scatter_modal = pn.Column(sizing_mode="stretch_both")
    scatter_modal.visible = False  # Initially hidden

    # Add a close button for the scatterplot modal
    close_scatter_btn = pn.widgets.Button(
        name="Close",
        button_type="primary",
        width=80,
        margin=(0, 0, 10, 0),
    )

    def close_scatterplot(event):
        scatter_modal.visible = False

    close_scatter_btn.on_click(close_scatterplot)

    def show_scatterplot(lon, lat):
        # Extract completion dates for all years and generations at (lon, lat)
        # da: dims ('year', 'generation', 'latitude', 'longitude')
        # Find nearest indices
        lon_idx = np.argmin(np.abs(da.coords["longitude"].values - lon))
        lat_idx = np.argmin(np.abs(da.coords["latitude"].values - lat))
        years = da.coords["year"].values
        generations = da.coords["generation"].values

        # Get days to completion for all years/generations at this location
        vals = da.isel(longitude=lon_idx, latitude=lat_idx)
        # Convert days to completion dates
        # det_date = pd.to_datetime(detection_date)
        # completion_dates = det_date + pd.to_timedelta(vals.values, unit="D")
        # Prepare data for scatterplot
        data = []
        for gen_idx, gen in enumerate(generations):
            for year_idx, year in enumerate(years):
                data.append(
                    {
                        "Generation": int(gen),
                        "Year": int(year),
                        "Completion Date": vals[year_idx, gen_idx],
                        "Days to Completion": vals.values[year_idx, gen_idx],
                    }
                )
        df = pd.DataFrame(data)
        # Scatterplot: x=Year, y=Completion Date, color=Generation
        scatter = hv.Scatter(
            df,
            kdims=["Year"],
            vdims=["Completion Date", "Generation", "Days to Completion"],
            group="Completion Dates",
        ).opts(
            color="Generation",
            cmap="Category10",
            size=8,
            tools=["hover"],
            xlabel="Year",
            ylabel="Completion Date",
            title=f"Completion Dates at ({lon:.3f}, {lat:.3f})",
            width=500,
            height=350,
            marker="o",
        )
        scatter_modal.clear()
        scatter_modal.append(close_scatter_btn)
        scatter_modal.append(pn.pane.HoloViews(scatter, sizing_mode="stretch_both"))
        scatter_modal.visible = True

    # --- Add: Click callback for the map ---
    # Instead of .on_event, use HoloViews hooks for tap/click events

    def map_tap_hook(plot, element):
        # Only attach callback if plot has a figure and tools
        if hasattr(plot, "state") and hasattr(plot.state, "tools"):
            # Find the TapTool
            for tool in plot.state.tools:
                if tool.__class__.__name__ == "TapTool":
                    # Attach the callback only once
                    if not hasattr(tool, "_fflies_callback_attached"):

                        def _callback(attr, old, new):
                            # Get the coordinates from the tap event
                            if tool.callback is not None and hasattr(
                                tool, "computed_renderers"
                            ):
                                # This is a placeholder; actual tap coordinates are in plot.state
                                # Instead, use plot.state.last_mouse_position if available
                                if hasattr(plot.state, "last_mouse_position"):
                                    x, y = plot.state.last_mouse_position
                                    show_scatterplot(x, y)

                        tool._fflies_callback_attached = True
                        # Attach the callback to the tool's event
                        # But Bokeh TapTool does not expose a direct callback for Python
                        # Instead, use plot.state.on_event if available
                        if hasattr(plot.state, "on_event"):
                            plot.state.on_event(
                                "tap", lambda event: show_scatterplot(event.x, event.y)
                            )
                        # If not available, skip (Panel server only)

    # Add the hook to the plot
    plot_pane_hv = pn.pane.HoloViews(plot_pane, sizing_mode="stretch_both")

    # Main layout: logo at the very top, then the rest of the dashboard
    layout = pn.Column(
        logo_row,
        pn.Row(
            menu,
            pn.Column(
                plot_pane_hv,
                scatter_modal,  # Add the modal to the layout
                sizing_mode="stretch_both",
                min_height=800,
            ),
            sizing_mode="stretch_both",
            min_height=800,
        ),
        sizing_mode="stretch_both",
        min_height=800,
    )
    if save_path is not None:
        layout.save(save_path, embed=True)
    return layout


# Recommendation for sharing HTML files:
# 1. Place all resources (favicon.ico, logo.png, about.pdf, etc.) in a subfolder (e.g., 'resources') next to your saved HTML files.
# 2. When saving the HTML, use relative paths for resources (e.g., "resources/logo.png").
# 3. Update all resource references in the dashboard to use these relative paths.

# Example: Assume your output folder structure is:
# /some/output/folder/
#   ├── my_dashboard.html
#   └── resources/
#         ├── favicon.ico
#         ├── logo.png
#         └── about.pdf

# Then, set:
# FAVICON_PATH = "resources/favicon.ico"
# LOGO_PATH = "resources/logo.png"
# ABOUT_PDF_PATH = "resources/about.pdf"

# When saving the HTML, copy the resources folder next to the HTML file.
# This way, the HTML and all resources are portable and will work on any machine or web server.


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Serve the FruitFlyPheno dashboard for a NetCDF file."
    )
    parser.add_argument(
        "netcdf_path", type=str, help="Path to the NetCDF file to visualize."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Port to serve the dashboard on (default: 5006).",
    )
    args = parser.parse_args()

    dashboard = create_fflies_dashboard(args.netcdf_path)
    import panel as pn

    pn.serve(dashboard, port=args.port, show=True, title="FruitFlyPheno Dashboard")


if __name__ == "__main__":
    main()
