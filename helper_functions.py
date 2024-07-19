import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import timedelta, date
from pydap.client import open_url
import xarray as xr
import netCDF4 as nc


def plot_netcdf(filepath, variable_name):
    """
    Plot a specified variable from a NetCDF file.

    Parameters:
    filepath (str): Path to the NetCDF file.
    variable_name (str): Name of the variable to plot.
    """
    try:
        # Open the NetCDF file
        dataset = nc.Dataset(filepath)

        # Check if the variable exists in the dataset
        if variable_name not in dataset.variables:
            print(f"Variable '{variable_name}' not found in the dataset.")
            return

        # Extract the variable data
        variable_data = dataset.variables[variable_name][:]

        # Get latitude and longitude variables
        lon_name = "longitude" if "longitude" in dataset.variables else "lon"
        lat_name = "latitude" if "latitude" in dataset.variables else "lat"

        if lon_name not in dataset.variables or lat_name not in dataset.variables:
            print("Longitude and latitude variables not found in the dataset.")
            return

        lon = dataset.variables[lon_name][:]
        lat = dataset.variables[lat_name][:]

        # Create a meshgrid for plotting
        lon, lat = np.meshgrid(lon, lat)

        # Plot the variable data
        plt.figure(figsize=(10, 6))
        plt.contourf(
            lon, lat, variable_data[0, :, :], cmap="viridis"
        )  # Adjust index [0, :, :] as needed
        plt.colorbar(label=f"{variable_name} (units)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"{variable_name} Distribution")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


def fetch_data_for_day(year, date):
    # set base url to https://tds.climate.ncsu.edu/thredds/catalog/prism/daily/combo/2020/catalog.html

    base_url = "https://tds.climate.ncsu.edu/thredds/dodsC/prism/daily/combo/{}/PRISM_combo_{}.nc"
    # base_url = "https://safaris-tds.cipm.info/thredds/dodsC/prismrh/{}/PRISM_rh_{}.nc"

    date_str = date.strftime("%Y%m%d")

    url = base_url.format(year, date_str)
    dataset = open_url(url)
    return url, dataset


def fetch_and_save_data(year, start_day_of_year, end_day_of_year, base_save_dir):

    save_dir = os.path.join(base_save_dir, "PRISM", str(year))

    os.makedirs(save_dir, exist_ok=True)

    start_date = date(year, 1, 1) + timedelta(days=start_day_of_year - 1)

    end_date = date(year, 1, 1) + timedelta(days=end_day_of_year - 1)

    current_date = start_date
    # if file exists, skip

    while current_date <= end_date:
        url, dataset = fetch_data_for_day(year, current_date)

        # Use xarray to save the data to a .nc file

        ds = xr.open_dataset(url)

        date_str = current_date.strftime("%Y%m%d")

        save_path = os.path.join(save_dir, f"PRISM_combo_{date_str}.nc")

        ds.to_netcdf(save_path)

        print(f"Saved data for {date_str} to {save_path}")

        current_date += timedelta(days=1)


# Example usage
# plot_netcdf('samp.nc', 'temperature')


def clip_netcdf(filepath, output_filepath, lat_center, lon_center, buffer_degrees):
    # Open the NetCDF file
    dataset = nc.Dataset(filepath, "r")

    # Extract latitude and longitude variables
    lats = dataset.variables["lat"][:]
    lons = dataset.variables["lon"][:]

    # Calculate buffer boundaries
    lat_min = lat_center - buffer_degrees
    lat_max = lat_center + buffer_degrees
    lon_min = lon_center - buffer_degrees
    lon_max = lon_center + buffer_degrees

    # Find indices within the buffer
    lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_indices = np.where((lons >= lon_min) & (lons <= lon_max))[0]

    # Extract the subset of data within the buffer
    clipped_data = {}
    for var_name in dataset.variables:
        var = dataset.variables[var_name]
        if "lat" in var.dimensions and "lon" in var.dimensions:
            lat_dim = var.dimensions.index("lat")
            lon_dim = var.dimensions.index("lon")
            slices = [slice(None)] * var.ndim
            slices[lat_dim] = slice(lat_indices[0], lat_indices[-1] + 1)
            slices[lon_dim] = slice(lon_indices[0], lon_indices[-1] + 1)
            clipped_data[var_name] = var[tuple(slices)]
        else:
            clipped_data[var_name] = var[:]

    # Create new NetCDF file with clipped data
    with nc.Dataset(output_filepath, "w", format="NETCDF4") as dst:
        # Copy dimensions
        for name, dimension in dataset.dimensions.items():
            length = len(dimension)
            if name == "lat":
                length = len(lat_indices)
            elif name == "lon":
                length = len(lon_indices)
            dst.createDimension(name, length)

        # Copy variables
        for var_name, var_data in clipped_data.items():
            var = dataset.variables[var_name]
            new_var = dst.createVariable(var_name, var.datatype, var.dimensions)
            new_var[:] = var_data

        # Copy global attributes
        dst.setncatts(dataset.__dict__)

    # Close the source NetCDF file
    dataset.close()


"""
# Example usage
#filepath = "data/P"
output_filepath = "path/to/your/output.nc"
lat_center = 36.73614  # Latitude in decimal degrees
lon_center = -119.78899  # Longitude in decimal degrees
buffer_degrees = 1  # Buffer in degrees

clip_netcdf(filepath, output_filepath, lat_center, lon_center, buffer_degrees)
"""
