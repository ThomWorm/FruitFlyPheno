import xarray as xr
import xpublish
import numpy as np
import pandas as pd

output_zarr_store = r"simple_zarr_dataset.zarr"  # Replace with the desired output path

data = xr.open_zarr(output_zarr_store)


rest = xpublish.SingleDatasetRest(data)
# Start the server and keep it running
if __name__ == "__main__":
    print("Serving simple dataset...")
    rest.serve(port=8000)
