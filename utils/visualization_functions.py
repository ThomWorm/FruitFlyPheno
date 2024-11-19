import xarray as xr
import geoviews as gv
import holoviews as hv
from holoviews import opts
import numpy as np
from bokeh.models import HoverTool, DatetimeTickFormatter
import xarray as xr
import geoviews as gv
import holoviews as hv
from holoviews import opts
import numpy as np
from bokeh.models import HoverTool, DatetimeTickFormatter


def plot_threshold_reached_dates(data_array, export_html=False, output_file="map.html"):
    hv.extension("bokeh")
    hv.Dimension.type_formatters[np.datetime64] = "%Y-%m-%d"

    # Assuming data_array is already defined as an xarray.DataArray of datetime64[ns]
    dataset = xr.Dataset({"gen_comp_date": data_array})

    kdims = ["latitude", "longitude"]
    vdims = ["gen_comp_date"]

    xr_dataset = gv.Dataset(dataset, kdims=kdims, vdims=vdims)

    # Convert the dataset to a GeoViews Image
    image = xr_dataset.to(gv.Image, ["longitude", "latitude"])

    # Define a custom hover tool to display formatted dates
    hover = HoverTool(
        tooltips=[
            ("Gen Comp Date", "@image{%F}"),
        ],
        formatters={
            "@image": "datetime",  # use 'datetime' formatter for gen_comp_date
        },
    )

    # Define options for the plot
    image = image.opts(
        opts.Image(
            cmap="viridis",
            colorbar=True,
            colorbar_position="right",
            width=1200,
            height=1200,
            tools=[hover],
            title="Generation Completion Date with OSM Basemap",
            cformatter=DatetimeTickFormatter(days="%Y-%m-%d"),
            alpha=0.75,
        )
    )

    # Overlay with an OSM basemap
    tiles = gv.tile_sources.OSM
    plot = tiles * image

    # Display the plot
    if export_html:
        hv.save(plot, output_file)
        print(f"Map saved to {output_file}")
    else:
        hv.output(plot)
