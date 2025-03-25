import os
import datetime
from datetime import timedelta, date
import os
import sys
import pickle
import gc
import pandas as pd
import numpy as np
from dask import delayed
import time
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from matplotlib.patches import Circle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

# -------------------
from custom_errors import *
from inputs import *
from outputs import *
from degree_day_equations import *


def da_calculate_degree_days(LTT, UTT, data):
    # returns a data array with the degree days
    if len(data.dims) == 1:
        # print("calculating degree days for 1D data array")
        degree_days_vec = xr.apply_ufunc(
            vsingle_sine_horizontal_cutoff,
            data["tmin"],
            data["tmax"],
            kwargs={"LTT": LTT, "UTT": UTT},
            dask="allowed",
        )
        degree_days_vec = xr.DataArray(
            degree_days_vec, coords=data.coords, dims=data.dims, name="degree_days"
        )

        return degree_days_vec

    elif len(data.dims) == 3:
        # print("calculating degree days for 3D data array")
        degree_days_vec = xr.apply_ufunc(
            vsingle_sine_horizontal_cutoff,
            data["tmin"].values,
            data["tmax"].values,
            kwargs={"LTT": LTT, "UTT": UTT},
            # vectorize=True,  # Ensure function is applied element-wise
            input_core_dims=[
                ["t", "longitude", "latitude"],
                ["t", "longitude", "latitude"],
            ],  # Adjust to your dimensions
            output_core_dims=[
                ["t", "longitude", "latitude"]
            ],  # Adjust to your dimensions
            # dask="allowed",
            output_dtypes=[],
        )
        if np.isnan(degree_days_vec).any():
            warnings.warn("Development error: NaN values in degree days")
            # degree_days_vec = np.nan_to_num(degree_days_vec, nan=0)
        degree_days_vec = xr.DataArray(
            degree_days_vec,
            coords=data.coords,
            dims=data.dims,
            name="degree_days",
        )

        return degree_days_vec


def select_point_data(data, coordinates):
    if type(coordinates) == list:
        coordinates = coordinates[0]
    return data.sel(latitude=coordinates[0], longitude=coordinates[1], method="nearest")


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


def all_historical_model_run(
    coordinates,
    start_dates,
    fly_params,
    historical_data_buffer,
    cache_path=None,
    context_map=False,
    retry_count=0,
):
    coordinates_bbox = get_bounding_box(coordinates)

    first_date = min(start_dates)
    last_date = max(start_dates) + timedelta(
        days=historical_data_buffer
    )  # this will be removed when we switch to server - no need to call buffer
    if last_date > pd.Timestamp.now() - timedelta(days=2):
        raise PredictionNeededError("")
    n_days_data = (last_date - first_date).days
    first_date_str = first_date.strftime("%Y-%m-%d")

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as cache_file:
            raw_PRISM = pickle.load(cache_file)
    else:
        # Fetch historical data
        raw_PRISM = fetch_ncss_data(
            start_date=first_date_str, n_days=n_days_data, bbox=coordinates_bbox
        )

        # Save fetched data to cache
        if cache_path:
            with open(cache_path, "wb") as cache_file:
                pickle.dump(raw_PRISM, cache_file)

    DD_data = da_calculate_degree_days(fly_params["LTT"], fly_params["UTT"], raw_PRISM)
    del raw_PRISM
    gc.collect()

    check_data_at_point(DD_data, coordinates)

    ###############
    ## Run Model ##
    ###############
    try:
        # if we receive multiple points, we will just output the completion dates
        if len(coordinates) == 1 and context_map == True:

            time_index = np.argwhere(
                DD_data.t.values == np.datetime64(start_dates[0])
            ).flatten()[0]

            model_output = apply_fflies_model_run_distributed(
                DD_data, time_index, fly_params["dd_threshold"]
            )
            report_stats(model_output, coordinates)
            return plot_xr_with_point_and_circle(model_output, coordinates[0])
        else:
            for i, coord in enumerate(coordinates):
                time_index = np.argwhere(
                    DD_data.t.values == np.datetime64(start_dates[i])
                ).flatten()[0]
                print("time index", time_index)
                point_data = select_point_data(DD_data, coord)

                start_time = time.time()
                model_output_test = fflies_model_2(
                    point_data, time_index, fly_params["dd_threshold"]
                )
                print(model_output_test)
                end_time = time.time()
                print(f"fflies_model_1 execution time: {end_time - start_time} seconds")

                start_time = time.time()
                model_output = apply_fflies_model_run_distributed(
                    DD_data, time_index, fly_params["dd_threshold"]
                )
                end_time = time.time()
                print(
                    f"apply_fflies_model_run_distributed execution time: {end_time - start_time} seconds"
                )

                report_stats(model_output, coord)
            return None

    except HistoricalDataBufferError:
        if retry_count < 2:
            print(
                "insufficient accumulation of growing degree days over \n "
                + str(historical_data_buffer)
                + " days. Increasing buffer by 100 and retrying. \n"
                "You can try setting the historical_data_buffer to a higher value."
            )
            return all_historical_model_run(
                coordinates,
                start_dates,
                fly_params,
                historical_data_buffer + 100 * (retry_count + 1),
                cache_path,
                context_map,
                retry_count + 1,
            )
        else:
            print(
                "Historical Data Error encountered again. Exiting after two increases. Is this a very cold area?"
                " insufficient accumulation over"
                + str(historical_data_buffer + 300)
                + " days"
            )
            raise


def prediction_model_run(
    coordinates,
    start_dates,
    fly_params,
    n_days_data,
    cache_path,
    produce_plot=False,
    prediction_years=5,
):

    ######
    # Model Setup
    ######
    bbox = get_bounding_box(coordinates)
    first_date = min(start_dates)
    historical_data_first_year = first_date.year - prediction_years

    """
    replace to fetch data from server when available

    data = fetch_ncss_data(
        start_date=
        n_days=n_days_data,
        bbox=get_bounding_box(coordinates),
    )
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as cache_file:
            raw_PRISM = pickle.load(cache_file)

    DD_data = da_calculate_degree_days(fly_params["LTT"], fly_params["UTT"], raw_PRISM)
    del raw_PRISM
    gc.collect()

    ##Model Run
    for i, date in enumerate(start_dates):

        date_iteration_first_year = date - timedelta(days=prediction_years * 365)
        iteration_coords = coordinates[i]
        recent_weather_data = get_recent_weather_data(DD_data, date, iteration_coords)
        predicted_f3_days = calculate_predicted_f3_days_linear(
            DD_data,
            recent_weather_data,
            date_iteration_first_year,
            prediction_years,
            iteration_coords,
            fly_params,
        )

    print("Predicted F3 days for ", iteration_coords, ":", predicted_f3_days)
