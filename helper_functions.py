import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np


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


# Example usage
# plot_netcdf('samp.nc', 'temperature')
