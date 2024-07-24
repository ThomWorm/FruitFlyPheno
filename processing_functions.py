import numpy as np
import netCDF4 as nc
from concurrent.futures import ProcessPoolExecutor


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


# Example usage
output_array = process_netcdf_day(
    "data/test/PRISM/",
    2000,
    2020,
    "0101",
    "data/test/derived",
    "PRISM_mean_2000-2020",
    variable_name="tmax",
    num_cores=6,
)
