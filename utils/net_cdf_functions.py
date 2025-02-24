from glob import glob

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta, date
from pydap.client import open_url
import xarray as xr
import netCDF4 as nc
import warnings
from utils.net_cdf_functions import *
from .degree_day_equations import single_sine_horizontal_cutoff

import time
import pandas as pd
from datetime import datetime, timedelta
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        print(dataset)
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
    #silence warnings
    warnings.filterwarnings("ignore")

    base_url = "https://tds.climate.ncsu.edu/thredds/dodsC/prism/daily/combo/{}/PRISM_combo_{}.nc"
    # base_url = "https://safaris-tds.cipm.info/thredds/dodsC/prismrh/{}/PRISM_rh_{}.nc"

    date_str = date.strftime("%Y%m%d")
 
    url = base_url.format(year, date_str)
   

    dataset = open_url(url)
    return url, dataset


def check_and_download_missing_files(model_start_date, base_save_dir):
    #silence warnings
    warnings.filterwarnings("ignore")

    today = date.today()
    yesterday = today - timedelta(days=1)

    current_date = datetime.strptime(model_start_date, "%Y-%m-%d").date()
    missing_files = []
    added_files = []
    while current_date <= yesterday:
        year = current_date.year
        date_str = current_date.strftime("%Y%m%d")
        save_dir = os.path.join(base_save_dir, "PRISM", str(year))
        file_path = os.path.join(save_dir, f"PRISM_temp_{date_str}.nc")
        if not os.path.exists(file_path):
            print(f"Missing file for {file_path}")
            missing_files.append((year, current_date))

        current_date += timedelta(days=1)

    print(f"Total missing files to download: {len(missing_files)}")

    for year, file_date in missing_files:
        print(f"Downloading data for {file_date} {year}")
        try:
            url, dataaset = fetch_data_for_day(year, file_date)
        except Exception as e:
            print(f"Error downloading data for {file_date} {year}: {e}")
            print('skipping update')
            break


        # Use xarray to save the data to a .nc file

        # convert date to string and remove dashes

        date_str_sv = file_date.strftime("%Y%m%d")
        date_str_sv = date_str_sv.replace("-", "")
        try:
            ds = xr.open_dataset(url)

        # date_str = current_date.strftime("%Y%m%d")
            save_path = os.path.join(base_save_dir, "PRISM", str(year), f"PRISM_combo_{date_str_sv}.nc")

            ds.to_netcdf(save_path)
            added_files.append(save_path)
            print(f"Saved data for {date_str_sv} to {save_path}")

        except Exception as e:
            print(f"Error downloading data for {file_date} {year}: {e}")
            print("continuing to next date")

    clean_netcdf_files(base_save_dir, added_files)
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
        # strip dashes
        date_str = date_str.replace("-", "")

        save_path = os.path.join(save_dir, f"PRISM_combo_{date_str}.nc")

        ds.to_netcdf(save_path)

        print(f"Saved data for {date_str} to {save_path}")

        current_date += timedelta(days=1)


# Example usage
# plot_netcdf('samp.nc', 'temperature')



def clean_netcdf_files(folder_path, file_list = None):
    # Counter for the number of files needing modification
    files_modified = 0
    if file_list is not None:
        files = file_list
        for file in files:
            if "combo" in file and file.endswith(".nc"):
                # Construct the full file path
                file_path = file
                
                # Open the netCDF file using a context manager
                with xr.open_dataset(file_path) as ds:
                    # Drop the ppt and tmean variables if they exist
                    if 'ppt' in ds and 'tmean' in ds:
                        ds = ds.drop_vars(['ppt', 'tmean'])
                        
                        # Construct the new file name
                        new_file_name = file.replace("combo", "temp")
                        
                        
                        # Save the modified dataset to a new file
                        ds.to_netcdf(new_file_name)
                        
                        # Increment the counter
                        files_modified += 1

                # Delete the original "combo" netCDF file
                os.remove(file_path)
    else:
    # Walk through the directory and subdirectories
        for root, dirs, files in os.walk(folder_path):
            print(f"Processing folder: {root}")
            for file in files:
                if "combo" in file and file.endswith(".nc"):
                    # Construct the full file path
                    file_path = os.path.join(root, file)
                    
                    # Open the netCDF file using a context manager
                    with xr.open_dataset(file_path) as ds:
                        # Drop the ppt and tmean variables if they exist
                        if 'ppt' in ds and 'tmean' in ds:
                            ds = ds.drop_vars(['ppt', 'tmean'])
                            
                            # Construct the new file name
                            new_file_name = file.replace("combo", "temp")
                            new_file_path = os.path.join(root, new_file_name)
                            
                            # Save the modified dataset to a new file
                            ds.to_netcdf(new_file_path)
                            
                            # Increment the counter
                            files_modified += 1

                    # Delete the original "combo" netCDF file
                    os.remove(file_path)

    # Print the number of files that were modified
    print(f"Number of files modified: {files_modified}")
"""
# Example usage
#filepath = "data/P"
output_filepath = "path/to/your/output.nc"
lat_center = 36.73614  # Latitude in decimal degrees
lon_center = -119.78899  # Longitude in decimal degrees
buffer_degrees = 1  # Buffer in degrees

clip_netcdf(filepath, output_filepath, lat_center, lon_center, buffer_degrees)
"""
'''
def read_netcdfs(files, coords, LTT, UTT):
    def process_one_path(path):
        # Use a context manager to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # Initialize an empty list to store data for each point or bounding box
            point_datasets = []
            
            if isinstance(coords, tuple) and len(coords) == 2:
                # Single point case
                lat, lon = coords
                ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")
                ds_point["degree_days"] = single_sine_horizontal_cutoff(
                    ds_point["tmin"], ds_point["tmax"], LTT, UTT
                )
                ds_point = ds_point.drop_vars(["tmin", "tmax"])
                ds_point.load()
                point_datasets.append(ds_point)
            elif isinstance(coords, dict) and all(k in coords for k in ["lat_min", "lat_max", "lon_min", "lon_max"]):
                # Bounding box case
                lat_min, lat_max = coords["lat_min"], coords["lat_max"]
                lon_min, lon_max = coords["lon_min"], coords["lon_max"]
                ds_bbox = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
                ds_bbox["degree_days"] = single_sine_horizontal_cutoff(
                    ds_bbox["tmin"], ds_bbox["tmax"], LTT, UTT
                )
                ds_bbox = ds_bbox.drop_vars(["tmin", "tmax"])
                ds_bbox.load()
                point_datasets.append(ds_bbox)
            else:
                raise ValueError("Invalid coordinates format. Provide a tuple for a single point or a dictionary for a bounding box.")
            
            # Concatenate data for all points or bounding boxes along a new dimension 'point'
            ds_combined = xr.concat(point_datasets, dim="point")
            return ds_combined

    # Get a sorted list of file paths
    paths = sorted(glob(files))
    # Process each file and store the datasets in a list
    datasets = [process_one_path(p) for p in paths]
    # Concatenate all datasets along the specified dimension
    combined = xr.concat(datasets, "t")
    return combined
'''

def read_netcdfs(files, points, LTT, UTT):
    def process_one_path(path):
        # Use a context manager to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # Initialize an empty list to store data for each point
            point_datasets = []
            for lat, lon in points:
                # Select data for the current point
                ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")
                # Calculate degree days
                ds_point["degree_days"] = single_sine_horizontal_cutoff(
                    ds_point["tmin"], ds_point["tmax"], LTT, UTT
                )
                # Select only degree days
                ds_point = ds_point.drop_vars(["tmin", "tmax"])
                # Load all data from the transformed dataset to ensure we can
                # use it after closing each original file
                ds_point.load()
                # Append the processed dataset for the current point
                point_datasets.append(ds_point)
            # Concatenate data for all points along a new dimension 'point'
            ds_combined = xr.concat(point_datasets, dim="point")
            return ds_combined

    # Get a sorted list of file paths
    paths = sorted(glob(files))
    # Process each file and store the datasets in a list
    datasets = [process_one_path(p) for p in paths]
    # Concatenate all datasets along the specified dimension
    combined = xr.concat(datasets, "t")
    return combined


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
    ds = xr.open_dataset(BytesIO(response.content), engine='h5netcdf')
    
    return ds

def fetch_ncss_data( start_date, n_days=None, bbox=None, variables=['tmin', 'tmax'], point = None, base_url = "https://thredds.climate.ncsu.edu/thredds/ncss/grid/prism/daily/combo",):
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
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Calculate end_date
    if n_days is None:
        end_date = datetime.now() - timedelta(days=2)
    else:
        end_date = start_date + timedelta(days=n_days)
    
    # Generate list of dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize an empty list to store NCSS URLs
    ncss_urls = []
    
    # Loop through each date and construct the NCSS URL
    for date in dates:
        date_str = date.strftime('%Y-%m-%dT00:00:00Z')
        year = date.strftime('%Y')
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
        future_to_index = {executor.submit(fetch_single_day_ncss, url): i for i, url in enumerate(ncss_urls)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                ds = future.result()
                datasets[index] = ds
            except Exception as e:

                try:
                    #wait 5 seconds
                    time.sleep(3)
                    ds = future.result()
                    datasets.append(ds)
                except:
                   print(e)
                   print(f"Error fetching data for URL {ncss_urls[index]}: {e}")
    
    # Combine all datasets into a single xarray Dataset
    combined_ds = xr.concat(datasets, dim='t', join = 'override')
    
    return combined_ds