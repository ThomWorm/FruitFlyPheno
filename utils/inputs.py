import pandas as pd
import numpy as np
import requests
import json
import datetime
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import xarray as xr
from io import BytesIO
import os
import sys

sys.path.append(os.path.abspath(".."))
print(os.path.abspath(".."))  # Debugging line to check the path

# Replace 'some_module' with the actual module name


def fetch_single_day_ncss(ncss_url):
    """
    Fetch data for a single day from a THREDDS server using NCSS.

    Parameters:
    - ncss_url: str, the NCSS URL to fetch data from

    Returns:
    - xarray.Dataset containing the requested data for the specified day
    """
    # Fetch the data using requests
    response = requests.get(ncss_url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Load the dataset into xarray
    ds = xr.open_dataset(BytesIO(response.content), engine="h5netcdf")

    return ds


def fetch_ncss_data(
    start_date,
    n_days=None,
    bbox=None,
    variables=["tmin", "tmax"],
    point=None,
    base_url="https://thredds.climate.ncsu.edu/thredds/ncss/grid/prism/daily/combo",
):
    """
    Fetch data from a THREDDS server using NCSS and collect it into an xarray Dataset.

    Parameters:
    - base_url: str, base URL of the THREDDS server
    - start_date: str, start date in the format 'YYYY-MM-DD'
    - n_days: int or None, number of days to fetch data for. If None, fetch all data to the present.
    - bbox: tuple or None, bounding box in the format (lon_min, lon_max, lat_min, lat_max)
    - variables: list or None, list of variables to fetch

    Returns:
    - xarray.Dataset containing the requested data
    """
    # Convert start_date to datetime
    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    # Calculate end_date
    if n_days is None:
        end_date = datetime.now() - timedelta(days=2)
    else:
        end_date = start_date + timedelta(days=n_days)

    # Generate list of dates
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Initialize an empty list to store NCSS URLs
    ncss_urls = []

    # Loop through each date and construct the NCSS URL
    for date in dates:
        date_str = date.strftime("%Y-%m-%dT00:00:00Z")
        year = date.strftime("%Y")
        url = f"{base_url}/{year}/PRISM_combo_{date.strftime('%Y%m%d')}.nc"

        # Construct the NCSS URL
        var_params = "&".join([f"var={var}" for var in variables])
        if bbox:
            ncss_url = (
                f"{url}?{var_params}"
                f"&north={bbox[3]}&west={bbox[0]}&east={bbox[1]}&south={bbox[2]}"
                f"&horizStride=1&time_start={date_str}&time_end={date_str}&accept=netcdf4ext&addLatLon=true"
            )
        elif point:
            ncss_url = (
                f"{url}?{var_params}"
                f"&north={point[1]}&west={point[0]}&east={point[0]}&south={point[1]}"
                f"&horizStride=1&time_start={date_str}&time_end={date_str}&accept=netcdf4ext&addLatLon=true"
            )
        else:
            raise ValueError("Either bbox or point must be provided.")
            # Append the NCSS URL to the list

        ncss_urls.append(ncss_url)

    # Initialize an empty list to store datasets
    datasets = [None] * len(ncss_urls)

    # Use ThreadPoolExecutor to fetch data in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {
            executor.submit(fetch_single_day_ncss, url): i
            for i, url in enumerate(ncss_urls)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                ds = future.result()
                datasets[index] = ds
            except Exception:
                try:
                    # wait 5 seconds
                    time.sleep(3)
                    ds = future.result()
                    datasets.append(ds)
                except Exception as e:
                    print(e)
                    print(f"Error fetching data for URL {ncss_urls[index]}: {e}")

    # Combine all datasets into a single xarray Dataset
    combined_ds = xr.concat(datasets, dim="t", join="override")

    return combined_ds


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


import os
import json


def load_species_params(target_species, data_path=None):
    """Loads species-specific parameters from a JSON file."""
    # Determine the project root (one level up from the utils folder)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Default data_path to the "data" folder in the project root
    if data_path is None:
        data_path = os.path.join(project_root, "data")
    else:
        data_path = os.path.join(project_root, data_path)
    # Load the JSON file
    with open(os.path.join(data_path, "fly_models.json")) as f:
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
    if type(coordinates) is list:
        coordinates = coordinates[0]

    completion_at_coords = model_output.sel(
        latitude=coordinates[0], longitude=coordinates[1], method="nearest"
    ).values.item()

    print(int(completion_at_coords), " days to F3 completion at ", coordinates)


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


'''
def process_start_dates(
    DD_data, start_dates, coordinates, prediction_years, fly_params
):
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
'''


def prediction_model_make_temp_data(
    DD_data, recent_weather_data, date_iteration_first_year, n, iteration_coords
):
    """
    Generate a temporary dataset by combining degree-day (DD) data and recent weather data.
    This function extracts a subset of the DD data starting from a specific date and location,
    and substitutes the first `n` days of the extracted data with the provided recent weather data.
    Args:
        DD_data (xarray.DataArray): The degree-day data array containing temperature-related information.
        recent_weather_data (numpy.ndarray or similar): The recent weather data to substitute into the DD data.
        date_iteration_first_year (datetime.datetime): The starting date of the iteration's first year.
        n (int): The number of years to offset from the starting date.
        iteration_coords (tuple): A tuple containing the latitude and longitude coordinates
                                  (latitude, longitude) for selecting the data.
    Returns:
        numpy.ndarray: A modified array where the first `n` days are replaced with recent weather data.
    """
    tmp_data = (
        DD_data.sel(
            t=slice(
                date_iteration_first_year + timedelta(days=n * 365),
                None,
            )
        )
        .sel(
            latitude=iteration_coords[0],
            longitude=iteration_coords[1],
            method="nearest",
        )
        .values
    )

    recent_weather_length = len(recent_weather_data)
    # substitute the first n days of tmp_data with the recent weather data
    tmp_data[:recent_weather_length] = recent_weather_data

    return tmp_data
