import datetime
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from typing import Optional
import matplotlib.figure as mpl_fig
from dataclasses import dataclass, field
from xarray import DataArray

import os
import sys

sys.path.append(os.path.abspath(".."))


# from notebooks.inputs import get_recent_weather_data
# from notebooks.simulations import calculate_predicted_f3_days_linear
@dataclass
class fflies_output_class:
    finish_date_list: Optional[list] = field(
        default_factory=list
    )  # Can be a DataFrame or None
    
    value: Optional[int] = None  # Can be an int or None
    array: Optional[DataArray] = field(
        default_factory=DataArray
    )  # Can be an xarray DataArray or None

    def plot(self):
        if self.figure is not None:
            plt.show(self.figure)
        else:
            print("No figure to plot.")

    def safeTiff(self, filename):
        if self.array is not None:
            self.array.rio.to_raster(filename)
        else:
            print("No array to save as TIFF.")

    # def serve


def report_stats(model_output, coordinates):
    if type(coordinates) is list:
        coordinates = coordinates[0]
    completion_at_coords = model_output.sel(
        latitude=coordinates[0], longitude=coordinates[1], method="nearest"
    ).values.item()
    print(int(completion_at_coords), " days to F3 completion at ", coordinates)


def plot_xr_with_point_and_circle(
    data,
    point_coords,
    circle_radius_km=15,
    alpha=0.8,
    generate_webpage=False,
    generate_tiff=False,
):
    """
    Plots an xarray DataArray on a map with a point and a circle overlay.
    Parameters:
    -----------
    data : xarray.DataArray
        The data to be plotted. It should be geospatial data compatible with the PlateCarree projection.
    point_coords : tuple
        A tuple containing the latitude and longitude of the point to be plotted (lat, lon).
    circle_radius_km : float, optional
        The radius of the circle to be drawn around the point, in kilometers. Default is 15 km.
    alpha : float, optional
        The transparency level of the data overlay. Default is 0.8.
    generate_webpage : bool, optional
        If True, generates a webpage with the plot as a png. Default is False.
    generate_tiff : bool, optional
        If True, generates a tiff file of the plot for download. Default is False.
    Returns:
    --------
    None
        This function does not return any value. It displays the plot.
    Notes:
    ------
    - The function uses OpenStreetMap (OSM) as the basemap.
    - The circle radius is converted from kilometers to degrees using an approximate conversion factor.
    - The plot includes a colorbar labeled "Days to F3" and a title indicating the start date of the calculation.
    """

    # data is an xarray DataArray
    point_lon = point_coords[1]
    point_lat = point_coords[0]
    # Create a plot with a basemap
    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    # ax.set_extent(extent, crs=ccrs.PlateCarree())
    vmax = data.max()
    # Add an OSM basemap
    osm = cimgt.OSM()
    ax.add_image(osm, 10)
    # Plot the data
    data.plot(
        ax=ax,
        cmap="viridis",
        alpha=alpha,
        transform=ccrs.PlateCarree(),
        add_colorbar=True,
        vmin=-1,
        vmax=vmax,
        cbar_kwargs={"label": "Days to F3"},
    )

    # create date from day of year

    date = datetime.datetime(2001, 1, 1) + datetime.timedelta(240 - 1)
    # Plot the point
    ax.plot(point_lon, point_lat, "ro", markersize=10, transform=ccrs.PlateCarree())

    # Plot the circle
    circle_radius_deg = (
        circle_radius_km / 111.32
    )  # Convert radius from km to degrees (approximation)
    circle = Circle(
        (point_lon, point_lat),
        circle_radius_deg,
        color="red",
        fill=False,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(circle)

    # Add labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Days to F3 Completion beginning on " + date.strftime("%Y-%m-%d"))

    # Show the plot

    plt.show()
