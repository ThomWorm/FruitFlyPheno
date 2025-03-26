import xarray as xr
import xpublish
import numpy as np
import pandas as pd

data = xr.Dataset(
    {
        "temperature": (("time", "lat", "lon"), 20 * np.random.rand(4, 3, 2)),
    },
    coords={
        "time": pd.date_range("2023-01-01", periods=4),
        "lat": [10, 20, 30],
        "lon": [100, 110],
    },
)
print(type(data))

d2 = xr.open_zarr(
    r"/home/thom/Desktop/CIPM/FruitFlyPheno/tests/simple_zarr_dataset.zarr",
    consolidated=True,
)
print(type(d2))
rest = xpublish.SingleDatasetRest(d2)
# Start the server and keep it running
if __name__ == "__main__":
    print("Serving simple dataset...")
    rest.serve(port=8000)
