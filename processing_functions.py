import numpy as np
import netCDF4 as nc
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from degree_day_equations import *
import warnings


def subset_dataset_by_coords(dataset, lat, lon, window_size=None):
    if window_size is not None:
        # Define the window boundaries
        lat_min = lat - window_size
        lat_max = lat + window_size
        lon_min = lon - window_size
        lon_max = lon + window_size

        # Create boolean masks for the coordinate ranges
        mask_lat = (dataset.latitude >= lat_min) & (dataset.latitude <= lat_max)
        mask_lon = (dataset.longitude >= lon_min) & (dataset.longitude <= lon_max)

        # Apply the mask using .where() and drop the data points outside the window
        print("masking 3d")
        subset = dataset.where(mask_lat & mask_lon, drop=True)
        nan_mask = subset.isnull().any(dim="t")
        # combine tmin and tmax into one mask
        nan_mask = nan_mask.tmin | nan_mask.tmax

        # subset = subset.fillna(-9999)
        # Check if the subset contains only NaN values

        # Create a 2D mask for NaN values in the subset
        # .any(dim="t")
        return subset, nan_mask
    else:
        print("masking 1d")
        subset = dataset.sel(latitude=lat, longitude=lon, method="nearest")
        # Check if the subset contains only NaN values
        # print any nan values
        print(subset.isnull(keep_attrs=True))

        # Create a mask for NaN values in the subset
        nan_mask = subset.isnull().any(dim="t")
        nan_mask = nan_mask.tmin | nan_mask.tmax
        return subset, nan_mask


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


def day_cumsum_reaches_threshold(
    degree_days, start_index, start_time_values, threshold
):
    # if degree_days contains na, return na
    if np.isnan(degree_days[start_index]):
        return np.nan

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
        print(f"start_dates: {start_dates}")

        return None


def create_start_index_array(start_dates, time_values):
    try:
        # Handle NaT values by assigning an invalid index (-1)
        start_indices = np.array(
            [
                -1 if np.isnat(d) else np.where(time_values == d)[0][0]
                for d in start_dates.values.flatten()
            ]
        ).reshape(start_dates.shape)

        return start_indices
    except IndexError:
        print(
            "Error: Start date not found in time dimension when creating start index array"
        )
        print(f"start_dates: {start_dates}")
        return None


def day_cumsum_reaches_threshold(
    degree_days, start_index, start_time_values, threshold
):
    # if degree_days contains NaN values, return NaT
    if np.isnan(degree_days).any():
        # print dimensions
        return 0
    cumsum = np.cumsum(degree_days[start_index:])
    if start_index == -1:
        return 0
    threshold_reached = np.where(cumsum >= threshold)[0]
    if len(threshold_reached) == 0:
        print(degree_days)
        warnings.warn("Development error: Threshold not reached")
        return 0
    # elisef start_index is None
    elif start_index is None:
        print(degree_days)
        warnings.warn("Development error: Start index is None")
        return 0

    else:
        first_reached_index = threshold_reached[0]
        result_date = start_time_values[start_index + first_reached_index]
        return result_date


def convert_to_datetime64_ns(value):
    if np.isnan(value):
        return np.datetime64("NaT", "ns")
    elif value == 0:
        return np.datetime64("NaT", "ns")
    return np.datetime64(int(value), "ns")


si = []


def compute_dd_threshold_reached_days(degree_days, start_dates, threshold):
    start_indices = create_start_index_array(start_dates, degree_days.t.values)
    si.append(start_indices)
    result_raw = xr.apply_ufunc(
        day_cumsum_reaches_threshold,
        degree_days,
        start_indices,
        degree_days.t.values.astype("datetime64[ns]"),
        threshold,
        input_core_dims=[["t"], [], ["t"], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["int64"],
    )

    result = xr.apply_ufunc(
        np.vectorize(convert_to_datetime64_ns),
        result_raw,
        dask="parallelized",
        output_dtypes=[np.datetime64],
    )

    return result
