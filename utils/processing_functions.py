import numpy as np
import netCDF4 as nc
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from .degree_day_equations import *
import warnings
from glob import glob
from .net_cdf_functions import *



def subset_dataset_by_coords(dataset, lat, lon, lat_min = None, lat_max = None, lon_min = None, lon_max = None, window_size=None):
    if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
        # Create boolean masks for the coordinate ranges
        mask_lat = (dataset.latitude >= lat_min) & (dataset.latitude <= lat_max)
        mask_lon = (dataset.longitude >= lon_min) & (dataset.longitude <= lon_max)

        # Apply the mask using .where() and drop the data points outside the window
        subset = dataset.where(mask_lat & mask_lon, drop=True)
        nan_mask = subset.isnull().any(dim="t")
        # combine tmin and tmax into one mask
        nan_mask = nan_mask.tmin | nan_mask.tmax

        # subset = subset.fillna(-9999)
        # Check if the subset contains only NaN values

        # Create a 2D mask for NaN values in the subset
        # .any(dim="t")
        return subset, nan_mask
    elif window_size is not None and lat is not None and lon is not None:
        # Define the window boundaries
        lat_min = lat - window_size
        lat_max = lat + window_size
        lon_min = lon - window_size
        lon_max = lon + window_size

        # Create boolean masks for the coordinate ranges
        mask_lat = (dataset.latitude >= lat_min) & (dataset.latitude <= lat_max)
        mask_lon = (dataset.longitude >= lon_min) & (dataset.longitude <= lon_max)

        # Apply the mask using .where() and drop the data points outside the window
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
        print("subsetting dataset by coordinates")
        subset = dataset.sel(latitude=lat, longitude=lon, method="nearest")
        # Check if the subset contains only NaN values
        # print any nan values

        # Create a mask for NaN values in the subset    
        print("masking")
        nan_mask = subset.isnull().any(dim="t")
        print("mask2")
        nan_mask = nan_mask.tmin | nan_mask.tmax
        return subset, nan_mask

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
        warnings.warn("Development error: Threshold not reached")
        return 0
    # elisef start_index is None
    elif start_index is None:
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


def compute_dd_threshold_reached_dates(degree_days, start_dates, threshold):
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
    #print(result)
    result = result.rename("gen_comp_date")  # Rename the output data

    return result

def days_between(start_date, end_date):
    # Convert the dates to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate the difference between the dates
    difference = end_date - start_date

    # Return the number of days
    return difference.days

def calculate_maturation_date(dd_data, start_date, threshold, generations):
    """
    Calculate the date when the threshold is reached for the specified number of generations
    """
    start_date_array = create_start_date_array(dd_data, start_date)
    threshold_reached_days = compute_dd_threshold_reached_dates(
        dd_data, start_date_array, threshold
    )

    for _ in range(generations - 1):
        threshold_reached_days = compute_dd_threshold_reached_dates(
            dd_data, threshold_reached_days, threshold
        )

    # Return an xarray with the date when the threshold is reached for the last time as one value and also the timedelta from the start date
    return threshold_reached_days

def fflies_model_run(start_date, degree_day_data,  dd_threshold, window = None, use_current_years_weather = True, generations = 3 ):
    #S

    
    #degree_day_data_current = degree_day_data.sel(t = slice(start_date, None))
    degree_day_data_current = degree_day_data
    ####################################################################
    #compute the threshold reached date for every year in the historical past and add to an array
    
    #create start day that strips the year from the start date
    #start_day = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%m-%d")
    start_day_str = start_date[5:]
    print("running DD accumulations for historical data, 20 yrs")
    for year in range(2000, 2020):

        
        if use_current_years_weather == False:
            number_days_recent_data = 0

        else:  
            number_days_recent_data = len(degree_day_data_current.t) - 1

        start_day_loop = np.datetime64(str(year) + "-" + start_day_str)
        finish_day_loop = start_day_loop + np.timedelta64(number_days_recent_data, "D")

        start_dates_array = create_start_date_array(degree_day_data, start_day_loop)
        temporary_degree_day_data = degree_day_data.sel(t=slice(start_day_loop, None)).copy()


        # replace the first n values of the temporary degree day data with the current degree day data
        # where n is the length of the current degree day data


        if number_days_recent_data > 0:
            temporary_degree_day_data.loc[{"t": slice(start_day_loop, finish_day_loop)}] = (
                degree_day_data_current.values
            )
        

        f3_maturation = calculate_maturation_date(temporary_degree_day_data, start_day_loop, dd_threshold, generations)
        #to the f_3_dates_array add the f3_timedelta as a separate value
        # Convert the 'gen_comp_date' to datetime
        # Convert the 'gen_comp_date' to datetime
        f3_maturation = f3_maturation.to_dataset(name="gen_comp_date")
        # Convert the numpy ndarray to a pandas Series
        
        f3_maturation['timedelta'] = ((f3_maturation['gen_comp_date'].data - start_day_loop) / (24 * 60 * 60 * 1e9)).astype(int)
        f3_maturation = f3_maturation.assign_coords(t = year)
        #if model_output object does not exist, make it f_3 maturation
        if 'model_output' not in locals():
            model_output = f3_maturation
        else:
            model_output = xr.concat([model_output, f3_maturation], dim='t')



    return model_output

def plot_values(model_1_date, species_name, model_start_date, model_output):
    # Create a figure and axis
    fig, ax = plt.subplots()
    values = model_output['timedelta'].data.tolist()
    model_1_day = days_between(model_start_date, model_1_date)
    # Plot the values as dots
    ax.scatter(values, [0]*len(values), alpha=0.5, label = "Annual Predicted Dates")
    ax.set_xlabel("Predicted Days Until F3 Maturation")

    # Plot the additional value in red
    ax.axvline(x=model_1_day, color='red',label='Model 1 Predicted date' )

    ax.legend()

    # Hide the Y axis
    ax.yaxis.set_visible(False)
    ax.set_title(species_name)

    ax.text(0.01, 0.95, f"Model Start Date: {model_start_date}", transform=ax.transAxes)
    ax.text(0.01, 0.90, f"Model_1 Finish Date: {model_1_date}", transform=ax.transAxes)
    # Show the plot
    return fig




def testing_load_data(data_path, start_date, LTT, UTT, lat = None, lon = None, lat_min = None, lat_max = None, lon_min = None, lon_max = None, window = None):
    def _preprocess(ds, lat, lon, LTT, UTT):
        dd = ds.sel(latitude=lat, longitude=lon, method="nearest")
        dd["degree_days"] = single_sine_horizontal_cutoff(
            dd["tmin"].values, dd['tmax'].values, LTT, UTT
        )
        dd.drop_vars(["tmin", "tmax"])
        return dd
    def _preprocess_simple(ds, lat, lon):
        return ds.sel(latitude=lat, longitude=lon, method="nearest")
    print("downloading recent weather data")
    check_and_download_missing_files(start_date, data_path)
    print("loading PRISM data")
    if "prism_year_stack" not in locals() and 'degree_day_data' not in locals():
        prism_year_stack = xr.open_mfdataset(
            data_path + "PRISM/*/PRISM_temp_*.nc",
            combine="nested",
            drop_variables=["ppt", "tmean"],
            data_vars="minimal",
            #coords="minimal",
            #compat="override",
            concat_dim='t',
            #decode_times=False,
            #decode_cf=False,
            #decode_coords=False,
            engine='netcdf4',
            parallel=False,
           # preprocess=lambda ds: _preprocess_simple(ds, lat, lon),
            #chunks = {'latitude': 1, 'longitude': 1, 't': 1000}
        )
        print("cropping PRISM data")
        if lat is not None and lon is not None:
            prism_year_stack, na_mask = subset_dataset_by_coords(prism_year_stack, lat, lon, window)
        elif lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
            prism_year_stack, na_mask = subset_dataset_by_coords(prism_year_stack, lat, lon, lat_min, lat_max, lon_min, lon_max)

        print("loading PRISM data")
        prism_year_stack.load()
        print("calculating degree days")

        degree_day_data = da_calculate_degree_days(LTT, UTT, prism_year_stack)
        prism_year_stack.close()
    #split off data after start date into separate xarray
        return degree_day_data
    else:
        return degree_day_data
    

def fflies_model_1(data, start, threshold):
    #initialize variables
    cumsum = 0
    elapsed_days = 0
    #iterate through the data array starting from the given start position
    for i in range(start, len(data)):
        #add the value of the current position to the cumsum
        cumsum += data[i]
        #increment the elapsed days
        elapsed_days += 1
        #if the cumsum is greater than or equal to the threshold, return the number of elapsed days
        if cumsum >= threshold:
            return elapsed_days
    #if the end of the array is reached, start over from the beginning and keep counting
    for i in range(0, start):
        cumsum += data[i]
        elapsed_days += 1
        if cumsum >= threshold:
            return elapsed_days
    return elapsed_days
# Loop through each fly species

def fflies_model_1(data, start, threshold):
    # Ensure data is an xarray DataArray
    if isinstance(data, np.ndarray):
        data = xr.DataArray(data)
    
    # Initialize cumulative sum and elapsed days
    cumsum = 0
    elapsed_days = 0
    
    # Iterate through the data array starting from the given start position
    for i in range(start, len(data)):
        # Add the value of the current position to the cumsum
        cumsum += data[i]
        # Increment the elapsed days
        elapsed_days += 1
        # If the cumsum is greater than or equal to the threshold, return the number of elapsed days
        if cumsum >= threshold:
            return elapsed_days
    
    # If the end of the array is reached, start over from the beginning and keep counting
    for i in range(0, start):
        cumsum += data[i]
        elapsed_days += 1
        if cumsum >= threshold:
            return elapsed_days
    
    # If the threshold is not reached, return the total number of days
    return elapsed_days

def apply_fflies_model_run(data, date, dd_threshold=754):
    # Apply the wrapper function over the x and y dimensions
    result = xr.apply_ufunc(
        fflies_model_1,
        data,
        date,
        dd_threshold,
        input_core_dims=[['day_of_year'], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int]
    )
    return result