import os
from datetime import timedelta
import pickle
import gc
import pandas as pd
import numpy as np
import time
import xarray as xr
import warnings
'''
# -------------------
from utils.custom_errors import PredictionNeededError, HistoricalDataBufferError
from utils.inputs import (
    fetch_ncss_data,
    get_recent_weather_data,
    check_data_at_point,
    prediction_model_make_temp_data,
    get_bounding_box,
)
from utils.degree_day_equations import vsingle_sine_horizontal_cutoff
from utils.outputs import fflies_output_class, report_stats

import sys

sys.path.append(os.path.abspath(".."))


def da_calculate_degree_days(LTT, UTT, data):
    """
    Calculate degree days using the single sine horizontal cutoff method.
    This function computes degree days for a given dataset based on the
    lower temperature threshold (LTT) and upper temperature threshold (UTT).
    It supports both 1D and 3D data arrays.
    Args:
        LTT (float): Lower temperature threshold for degree day calculation.
        UTT (float): Upper temperature threshold for degree day calculation.
        data (xarray.DataArray): Input data containing minimum and maximum
            temperatures. The data should have dimensions:
            - For 1D: ["t"]
            - For 3D: ["t", "longitude", "latitude"]
    Returns:
        xarray.DataArray: A DataArray containing the calculated degree days
        with the same dimensions and coordinates as the input data.
    Raises:
        UserWarning: If NaN values are detected in the calculated degree days
        for 3D data arrays.
    Notes:
        - The function uses `xr.apply_ufunc` to apply the
          `vsingle_sine_horizontal_cutoff` function element-wise.
        - For 3D data arrays, ensure the input dimensions match the expected
          structure.
    """

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
    if type(coordinates) is list:
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
        if len(coordinates) == 1 and context_map:
            time_index = np.argwhere(
                DD_data.t.values == np.datetime64(start_dates[0])
            ).flatten()[0]

            model_output = apply_fflies_model_run_distributed(
                DD_data, time_index, fly_params["dd_threshold"]
            )
            return fflies_output_class(array=model_output)

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

                return fflies_output_class(array=model_output)

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
    # bbox = get_bounding_box(coordinates)
    # first_date = min(start_dates)
    # historical_data_first_year = first_date.year - prediction_years

    """
    replace to fetch data from server when available

    data = fetch_ncss_data(
        start_date=
        n_days=n_days_data,
        bbox=get_bounding_box(coordinates),
    )
    """

    if cache_path:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cache_path = os.path.join(project_root, cache_path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as cache_file:
                raw_PRISM = pickle.load(cache_file)

        elif not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache file not found at {cache_path}. Please check the path."
            )

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
    return fflies_output_class(finish_date_list=predicted_f3_days)


'''
##########################

import numpy as np
from typing import Dict, Tuple, Union, List
import xarray as xr
from utils.degree_day_equations import single_sine_horizontal_cutoff


def fflies_core(
    tmin_1d: np.ndarray,
    tmax_1d: np.ndarray,
    start_day: int,
    stages: List[Dict],
    generations: int = 3,
) -> Dict[str, Union[Tuple[int, int, float], Tuple[str, int, float]]]:
    """
    Returns:
    - For complete generations: ('completed', days, accumulated_dd)
    - For incomplete: ('stage_X_gen_Y', current_days, partial_dd)
    """

     # Check for missing data
    if np.any(np.isnan(tmin_1d)) or np.any(np.isnan(tmax_1d)):
        return np.nan
    
    current_day = start_day
    total_days = len(tmin_1d)

    for gen in range(1, generations + 1):
        stage_accumulator = 0.0

        for stage_idx, stage in enumerate(stages):
            stage_dd = 0.0
            days_in_stage = 0
            while current_day < total_days:
                # Calculate degree days for all remaining days
                dd = single_sine_horizontal_cutoff(
                    tmin_1d[current_day],
                    tmax_1d[current_day],
                    stage["LTT"],
                    stage["UTT"],
                )

                stage_dd += dd
                stage_accumulator += dd
                days_in_stage += 1
                current_day += 1

                if stage_accumulator >= stage["dd_threshold"]:
                    '''
                    print(
                        f"Stage {stage_idx + 1} of generation {gen} completed on day {current_day}"
                    )
                    '''
                    break

            if stage_accumulator < stage["dd_threshold"]:
                # Incomplete stage - triggers when the loop ends without reaching the threshold
                return -1.0 #biological incompletion


        # Generation completed
        if gen == generations:
            days_elapsed = current_day - start_day
            # here we have fully completed the generations and so we are on the next generation
            return float(days_elapsed) # may wish to revise output structure
        
    raise ValueError("generation accumulation failed") #should not reach here

def fflies_spatial_wrapper(
    tmin_xr: xr.DataArray,  # (time, lat, lon)
    tmax_xr: xr.DataArray,  # (time, lat, lon)
    start_day: int,
    stages: List[Dict],
    generations: int = 3,
) -> xr.Dataset:
    
    """Simplified wrapper matching core outputs"""
    results= xr.apply_ufunc(
        fflies_core,
        tmin_xr,
        tmax_xr,
        input_core_dims=[["t"], ["t"]],
        kwargs={"start_day": start_day, "stages": stages, "generations": generations},
        output_core_dims=[[], ],
        output_dtypes=[float],
        vectorize=True,
        dask="parallelized",
        exclude_dims={"t"},
    )

    return xr.Dataset({
        'days_to_f3_completion': results.where(results >= 0),  # Only show valid completions
        'incomplete_development': (results == -1),
        'missing_data': np.isnan(results)  # Original missing data
    })

def fflies_prediction(
    current_data: xr.Dataset,  # tmin/tmax (time, lat, lon) - days since start_date
    historical_data: xr.Dataset,  # tmin/tmax (year, time, lat, lon)
    stages: List[Dict],
    detection_date: pd.Timestamp,

    generations: int = 3,
    start_year: int = 2021,
    end_year: int = 2023,

) -> xr.Dataset:
    """
    Predicts development using:
    1. Current year data until available
    2. Continues with each historical year's data
    
    Returns xr.Dataset with (year, lat, lon) containing:
    - total_days: Days from start to completion
    - completed: Boolean whether full development occurred
    """


    #### setup outputs
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)
    shape = (n_years, len(current_data.latitude), len(current_data.longitude))
    
    outputs = xr.Dataset({
        'days_to_f3_completion': (('year', 'latitude', 'longitude'), np.full(shape, np.nan, dtype=np.float32)),
        'incomplete_development': (('year', 'latitude', 'longitude'), np.full(shape, False, dtype=bool)),
        'missing_data': (('year', 'latitude', 'longitude'), np.full(shape, False, dtype=bool))
    }, coords={
        'year': years,
        'latitude': current_data.latitude,
        'longitude': current_data.longitude
    })
    
  
    #####

    
    detection_day_of_year = detection_date.dayofyear
    days_recent_data = len(current_data.t)
    for year in range(start_year, end_year + 1):
        historical_date = pd.Timestamp(year=year, month=detection_date.month, day=detection_date.day)
        historical_index = historical_data.get_index("t").get_loc(historical_date)
        historical_tmin = historical_data["tmin"].isel(t=slice(historical_index + days_recent_data, None))
        historical_tmax = historical_data["tmax"].isel(t=slice(historical_index + days_recent_data, None))

        # if efficiency become an issue, we can use np arrays to avoid copying
        model_run_tmax = xr.concat(
            [current_data["tmax"], historical_tmax], dim="t"
        )
        model_run_tmin = xr.concat(
            [current_data["tmin"], historical_tmin], dim="t"
        )

        #run model
        result = fflies_spatial_wrapper(
            model_run_tmin,
            model_run_tmax,
            start_day=0,
            stages=stages,
            generations=generations,
        )
        outputs['days_to_f3_completion'].loc[dict(year=year)] = result['days_to_f3_completion']
        outputs['incomplete_development'].loc[dict(year=year)] = result['incomplete_development']
        outputs['missing_data'].loc[dict(year=year)] = result['missing_data']
    
    return outputs