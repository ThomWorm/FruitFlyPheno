import xarray as xr
import xpublish

output_zarr_store = (
    r"~/Desktop/CIPM/FruitFlyPheno/data/zarr/"  # Replace with the desired output path
)

data = xr.open_zarr(output_zarr_store)

rest = xpublish.SingleDatasetRest(data)
print("serving")


# Start the server and keep it running
if __name__ == "__main__":
    print("Serving dataset...")
    rest.serve(host="0.0.0.0", port=8001)
