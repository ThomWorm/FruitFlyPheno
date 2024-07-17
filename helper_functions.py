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
    #base_url = "https://safaris-tds.cipm.info/thredds/dodsC/prismrh/{}/PRISM_rh_{}.nc"

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
    #if file exists, skip

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
