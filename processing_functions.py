import numpy as np
import netCDF4 as nc
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from degree_day_equations import *


def compute_mean(sub_array):
    return np.mean(sub_array, axis=1)


def process_netcdf_day_temp_means(
    input_dir,
    start_year,
    stop_year,
    month_day,
    output_dir,
    output_filename,
    variable_name,
    num_cores,
):
    # Setup
    num_process = num_cores

    # Create list of years and corresponding file paths
    years_list = range(start_year, stop_year + 1)
    file_list = [
        input_dir + str(year) + "/PRISM_combo_" + str(year) + month_day + ".nc"
        for year in years_list
    ]

    base_array = None
    lats = None
    lons = None

    # Process each file and concatenate the data
    for i, each_file in enumerate(file_list):
        # Open the file and read the data
        ds = nc.Dataset(each_file, "r")
        arr = ds.variables[variable_name][:]
        if i == 0:
            lats = ds.variables["latitude"][:]  # Extract latitude
            lons = ds.variables["longitude"][:]  # Extract longitude
        ds.close()

        # Remove singleton dimensions and flatten the array
        print(arr.shape)
        arr = np.squeeze(arr)
        print("post squeeze")
        print(arr.shape)
        arr_flat = arr.reshape((-1, 1))

        # Initialize or concatenate the base array
        if base_array is None:
            base_array = arr_flat
        else:
            base_array = np.concatenate((base_array, arr_flat), axis=1)

    # Compute mean for each grid point
    N = base_array.shape[1]  # Number of time steps
    P = num_process  # Number of partitions
    partitions = list(
        zip(
            np.linspace(0, base_array.shape[0], P, dtype=int)[:-1],
            np.linspace(0, base_array.shape[0], P, dtype=int)[1:],
        )
    )
    partitions[-1] = (
        partitions[-1][0],
        base_array.shape[0],
    )  # Ensure the last partition ends correctly

    # Use ProcessPoolExecutor to process partitions in parallel
    with ProcessPoolExecutor(max_workers=num_process) as executor:
        result = executor.map(compute_mean, [base_array[i:j, :] for i, j in partitions])

    # Combine results and reshape
    means = np.concatenate(list(result), axis=0)
    means = means.reshape(len(lats), len(lons))

    return means


def subset_dataset_by_coords(dataset, lat, lon, window_size=None):
    # if window_size is not None:

    if window_size is not None:
        # Define the window boundaries
        lat_min = lat - window_size
        lat_max = lat + window_size
        lon_min = lon - window_size
        lon_max = lon + window_size

        # Create boolean masks for the coordinate ranges
        mask_lat = (dataset.latitude >= lat_min) & (dataset.latitude <= lat_max)
        mask_lon = (dataset.longitude >= lon_min) & (dataset.longitude <= lon_max)

        # Create a boolean mask that combines latitude and longitude masks
        # ombined_mask = mask_lat[:, np.newaxis] & mask_lon[np.newaxis, :]

        # Apply the mask using .where() and drop the data points outside the window
        subset = dataset.where(mask_lat & mask_lon, drop=True).fillna(np.nan)

        return subset
    else:
        unchunked_dataset = dataset.chunk(None)
        return unchunked_dataset.sel(latitude=lat, longitude=lon, method="nearest")


def da_calculate_degree_days(LTT, UTT, data):
    # returns a data array with the degree days
    if len(data.dims) == 1:
        print("calculating degree days for 1D data array")
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
        print("calculating degree days for 3D data array")
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
            dask="allowed",  # Use Dask if your dataset is large
        )
        degree_days_vec = xr.DataArray(
            degree_days_vec,
            coords=data.coords,
            dims=data.dims,
            name="degree_days",
        )

        return degree_days_vec


def day_cumsum_reaches_threshold(
    degree_days, start_index, start_time_values, threshold
):
    cumsum = np.cumsum(degree_days[start_index:])
    threshold_reached = np.where(cumsum >= threshold)[0]
    if len(threshold_reached) == 0:
        warnings.warn("Development error: Threshold not reached")
        return 0
    first_reached_index = threshold_reached[0]
    result_date = start_time_values[start_index + first_reached_index]
    return result_date


def create_start_date_array(degree_days, date):

    if len(degree_days.dims) == 3:
        latitude = degree_days["latitude"]
        longitude = degree_days["longitude"]
        lat_shape = latitude.shape[0]
        lon_shape = longitude.shape[0]
        start_dates = xr.DataArray(
            np.full((lat_shape, lon_shape), np.datetime64(date, "ns")),
            coords=[latitude, longitude],
            dims=["latitude", "longitude"],
        )

    elif len(degree_days.dims) == 1:
        latitude = degree_days.coords.get("latitude", np.array([0]))
        longitude = degree_days.coords.get("longitude", np.array([0]))
        start_dates = xr.DataArray(
            np.datetime64(date, "ns"),  # Single start date for 1D input
            coords={"latitude": latitude, "longitude": longitude},
            dims=[],
        )
    return start_dates


def create_start_index_array(start_dates, time_values):
    try:
        start_indices = np.array(
            [np.where(time_values == d)[0][0] for d in start_dates.values.flatten()]
        ).reshape(start_dates.shape)
        return start_indices
    except IndexError:
        print(
            "Error: Start date not found in time dimension when creating start index array"
        )
        return None
