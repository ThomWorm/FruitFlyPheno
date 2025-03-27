import os
import xarray as xr
import datetime
from datetime import timedelta
import sys
import json

sys.path.append(os.path.abspath(".."))
from utils.degree_day_equations import *
from archive.net_cdf_functions import *
from utils.processing_functions import *


# from utils.visualization_functions import *

import pandas as pd

# from visualization_functions import *
import numpy as np

from custom_errors import *


import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


def validate_inputs(start_dates, coordinates, historical_data_buffer):
    """
    Validates the input parameters for a data processing or prediction model.

    This function ensures that the number of start dates matches the number of
    coordinates, checks if the coordinates list is not empty, and verifies that
    the date range is valid for fetching historical data.

    Parameters:
    -----------
    start_dates : list of pandas.Timestamp
        A list of start dates for the data processing or prediction.
    coordinates : list of tuples
        A list of coordinate pairs (latitude, longitude) corresponding to the start dates.
    historical_data_buffer : int
        The number of days to extend the start date to determine the end date for
        historical data fetching.

    Returns:
    --------
    all_historical_data : bool
        A boolean indicating whether the model can rely entirely on historical data
        (True) or if predictions are required due to insufficient historical data (False).

    Raises:
    -------
    ValueError
        If the number of coordinates does not match the number of start dates.
        If the coordinates list is empty.
        If any start date is earlier than "2000-01-01".


    """

    if len(coordinates) != len(start_dates):
        raise ValueError("Number of coordinates and dates do not match")
    if not coordinates:
        raise ValueError("No coordinates supplied")
    # Check if the date range is valid for fetching data
    for start_date in start_dates:
        end_date = start_date + pd.Timedelta(days=historical_data_buffer)

        if start_date < pd.Timestamp("2000-01-01"):
            raise ValueError("Start date is too early")

        elif end_date > pd.Timestamp.now() - pd.Timedelta(days=2):
            all_historical_data = False
            return all_historical_data

        else:
            all_historical_data = True
            return all_historical_data


def get_bounding_box(coordinates):
    """Computes bounding box for a list of coordinates."""
    if len(coordinates) > 1:
        lats, lons = zip(*coordinates)
        return (min(lons), max(lons), min(lats), max(lats))
    lats, lons = coordinates[0]
    return (lons - 0.3, lons + 0.3, lats - 0.3, lats + 0.3)


def load_species_params(target_species, data_path):
    """Loads species-specific parameters from a JSON file."""
    with open(data_path + "fly_models.json") as f:
        fly_models = json.load(f)
    return fly_models.get(target_species)


def fetch_weather_data(start_date, end_date, bbox, days_of_data):
    """Fetches PRISM weather data for the given date range and bounding box."""
    n_days = (end_date - start_date).days
    return fetch_ncss_data(
        start_date=start_date.strftime("%Y-%m-%d"), n_days=n_days, bbox=bbox
    )


def check_data_at_point(data, coordinates):
    """
    Checks if data is available at the specified coordinates.
    Parameters:
        data (xarray.DataArray): The dataset to check.
        coordinates (list of tuple): A list of (latitude, longitude) pairs.
    Raises:
        ValueError: If no data is available at any of the specified coordinates.
    """

    for coord in coordinates:
        sample = data.sel(
            latitude=coord[0], longitude=coord[1], method="nearest"
        ).values

        if np.any(np.isnan(sample)):
            raise ValueError("No data available at coordinates", coord)


def report_stats(model_output, coordinates):
    if type(coordinates) == list:
        coordinates = coordinates[0]
    completion_at_coords = model_output.sel(
        latitude=coordinates[0], longitude=coordinates[1], method="nearest"
    ).values.item()
    print(int(completion_at_coords), " days to F3 completion at ", coordinates)


def select_point_data(data, coordinates):
    if type(coordinates) == list:
        coordinates = coordinates[0]
    return data.sel(latitude=coordinates[0], longitude=coordinates[1], method="nearest")


def get_recent_weather_data(DD_data, date, iteration_coords):
    """Fetch recent weather data for the given date and coordinates."""
    return (
        DD_data.sel(t=slice(date, pd.Timestamp.now() - timedelta(days=2)))
        .sel(
            latitude=iteration_coords[0],
            longitude=iteration_coords[1],
            method="nearest",
        )
        .values
    )


def process_start_dates():
    """Processes start dates to calculate predicted F3 days for fruit fly phenology.

    Args:
        DD_data (DataFrame): Degree-day data used for calculations.
        start_dates (list of datetime): List of start dates for processing.
        coordinates (list of tuples): List of coordinate pairs (latitude, longitude) corresponding to each start date.
        prediction_years (int): Number of years to consider for predictions.
        fly_params (dict): Parameters related to fruit fly development and behavior.

    Returns:
        list: A list of predicted F3 days for each start date and coordinate pair.
    DD_data, start_dates, coordinates, prediction_years, fly_params"
    """
    for i, date in enumerate(start_dates):
        month_day = date.strftime("%m-%d")
        date_iteration_first_year = date - timedelta(days=prediction_years * 365)
        iteration_coords = coordinates[i]

        # Fetch recent weather data
        recent_weather_data = get_recent_weather_data(DD_data, date, iteration_coords)

        # Calculate predicted F3 days
        predicted_f3_days = calculate_predicted_f3_days_linear(
            DD_data,
            recent_weather_data,
            date_iteration_first_year,
            prediction_years,
            iteration_coords,
            fly_params,
        )


def calculate_predicted_f3_days_linear(
    DD_data,
    recent_weather_data,
    date_iteration_first_year,
    prediction_years,
    iteration_coords,
    fly_params,
):
    """
    Calculate predicted F3 days for a given set of coordinates and weather data.

    This function predicts the F3 generation completion dates for fruit flies
    based on degree-day (DD) data, recent weather data, and specific parameters
    for a given number of prediction years.

    Args:
        DD_data (pd.DataFrame): Degree-day data used for predictions.
        recent_weather_data (pd.DataFrame): Recent weather data to supplement
            the degree-day calculations.
        date_iteration_first_year (datetime.date): The starting date for the
            first year's iteration.
        prediction_years (int): The number of years to predict F3 days for.
        iteration_coords (tuple): Coordinates (latitude, longitude) for the
            location of interest.
        fly_params (dict): Parameters for the fruit fly model, including:
            - "dd_threshold" (float): The degree-day threshold for F3 generation.

    Returns:
        list[datetime.date]: A list of predicted F3 generation completion dates
        for each year in the prediction range.
    """

    predicted_f3_days = []
    for n in range(prediction_years):
        tmp_data = prediction_model_make_temp_data(
            DD_data,
            recent_weather_data,
            date_iteration_first_year,
            n,
            iteration_coords,
        )

        finish_date = fflies_model_2(
            tmp_data,
            0,
            fly_params["dd_threshold"],
        )
        predicted_f3_days.append(finish_date)
    return predicted_f3_days


def fflies_model_2(data, start, threshold, generations=3):
    """
    Simulates a model to calculate the number of days required for a cumulative sum
    of data values to reach a specified threshold, starting from a given position.
    Parameters:
    -----------
    data : xarray.DataArray or numpy.ndarray
        The input data array containing numerical values. If a numpy array is provided,
        it will be converted to an xarray DataArray.
    start : int
        The starting index in the data array from which the calculation begins.
    threshold : float
        The target cumulative sum value to be reached. This value is scaled by the
        number of generations.
    generations : int, optional
        The scaling factor for the threshold. Default is 3.
    Returns:
    --------
    int
        The number of elapsed days required to reach or exceed the threshold cumulative sum.
    Raises:
    -------
    HistoricalDataBufferError
        If the cumulative sum does not reach the threshold within the data range,
        an error is raised suggesting increasing the historical data buffer.
    ValueError
        If the data contains NaN values at the specified indices, the function
        returns -1 to indicate invalid input.
    Notes:
    ------
    - The function assumes the input data is a one-dimensional array.
    - If the cumulative sum does not reach the threshold, a detailed error message
      is printed before raising the exception.
    """
    # Ensure data is an xarray DataArray
    if isinstance(data, np.ndarray):
        data = xr.DataArray(data)

    threshold = threshold * generations
    # threshold = threshold * generations
    # Initialize cumulative sum and elapsed days
    cumsum = 0
    elapsed_days = 0

    # Iterate through the data array starting from the given start position
    for i in range(start, len(data)):
        # Add the value of the current position to the cumsum
        if np.isnan(data[i]):
            return -1
        cumsum += data[i]
        # Increment the elapsed days
        elapsed_days += 1
        # If the cumsum is greater than or equal to the threshold, return the number of elapsed days
        if cumsum >= threshold:
            return elapsed_days

    # If the threshold is not reached, return the total number of days
    print("cumsum not reached" + str(cumsum) + str(elapsed_days))
    raise HistoricalDataBufferError(
        "Threshold not reached - you may be calculating for a cold area. \n Try increasing the historical data buffer by 100 days."
    )


def apply_fflies_model_run_distributed(data, date, dd_threshold=754, generations=3):
    # Apply the wrapper function over the x and y dimensions using multi-threading or dask
    result = xr.apply_ufunc(
        fflies_model_2,
        data,
        date,
        dd_threshold,
        generations,
        input_core_dims=[["t"], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int],
    )
    result.name = "days_to_f3"
    result = result.where(result != -1, np.nan)
    return result


def plot_xr_with_point_and_circle(data, point_coords, circle_radius_km=15, alpha=0.8):
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
