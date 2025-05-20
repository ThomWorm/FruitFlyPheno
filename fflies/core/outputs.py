from dataclasses import dataclass
import xarray as xr
import json
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class FfliesOutput:
    data: xr.Dataset  # Accepts an xarray Dataset or DataArray
    latitude: float  # Latitude for the data
    longitude: float  # Longitude for the data

    def create_json(self, filename: str):
        """
        Save the xarray Dataset/DataArray as a JSON file.

        Parameters:
            filename (str): The name of the JSON file to create.
        """
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.data.to_dict(), f, indent=4)
        print(f"JSON file created at: {output_path}")

    def plot(self, variable: str, filename: str):
        """
        Generate a plot for a specific variable in the Dataset/DataArray.

        Parameters:
            variable (str): The variable to plot (must exist in the Dataset).
            filename (str): The name of the file to save the plot.
        """
        if variable not in self.data:
            raise ValueError(f"Variable '{variable}' not found in the Dataset.")

        # Generate the plot
        self.data[variable].plot()
        plt.title(f"Plot of {variable}")
        plt.savefig(self.output_dir / filename)
        plt.close()
        print(f"Plot saved at: {self.output_dir / filename}")

    def _check_local_variation(self) -> bool:
        """
        Check if the data has local variation based on latitude and longitude.

        Returns:
            bool: True if local variation is present, False otherwise.
        """
        # Placeholder for actual logic to check local variation
        return True
