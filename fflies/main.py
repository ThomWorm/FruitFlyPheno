# main.py


from fflies.core import (
    WeatherDataHandler,
    FfliesOutput,
    fflies_prediction_wrapper,
    fflies_spatial_wrapper,
    dict_to_table,
    dict_to_table_multi,
)
from fflies.io_handlers import load_config, get_user_input
from fflies.utils import load_species_params
import pandas as pd
import panel as pn
import sys
import json
import pickle
import os
import xarray as xr

import time


def validate_inputs(
    weather_data: WeatherDataHandler, input, detection_date: str, species_params: dict
):
    """
    Validate the required inputs for the model.

    Parameters:
    -----------
    weather_data : WeatherDataHandler
        The weather data handler object or loaded weather data.
    detection_date : str
        The detection date in 'YYYY-MM-DD' format.
    species_params : dict
        The species parameters dictionary.

    Raises:
    -------
    ValueError
        If any required input is missing or invalid.
    """
    # check if input is within weather bbox
    if not weather_data.is_within_bbox(input["latitude"], input["longitude"]):
        raise ValueError(
            f"Input coordinates ({input['latitude']}, {input['longitude']}) are outside the PRISM data bounding box."
        )
    # check if detection date > jan 1 2000, <= today
    detection_dt = pd.to_datetime(detection_date)
    if (
        detection_dt < pd.Timestamp(year=2000, month=1, day=1)
        or detection_dt > pd.Timestamp.now()
    ):
        raise ValueError(
            "Detection date must be between Jan 1, 2000 and today. Please check the input."
        )
    # check that input['species'] is in species_params
    if input["species"] not in species_params:
        raise ValueError(
            f"Species '{input['species']}' not found in species parameters. Please check the input."
        )
    # check if generations is a positive integer
    if not isinstance(input.get("generations"), int) or input["generations"] <= 0:
        raise ValueError(
            "Generations must be a positive integer. Please check the input."
        )


def fflies_model(
    input_json=None,
    print_results=False,
    output_path="outputs",
    unique_id=None,  # unique id imported from input JSON, but can be overridden by CLI
    exec_dashboard=False,
    use_pickle=False,  # Load/save model results from/to a pickle file for faster plotting development
    predict_from_date=None,  # New parameter for prediction cutoff date
):
    """
    Main entry point for FruitFlyPheno pipeline.

    Parameters:
    -----------
    input_json : str or None
        Path to input JSON file. If None, uses test input.
    plot : bool
        (Deprecated) No longer used. Plotting is handled in a separate dashboard module.
    save_plot : str or None
        (Deprecated) No longer used. Plotting is handled in a separate dashboard module.
    print_results : bool
        If True, prints the output JSON to the terminal in a formatted, readable table.
    use_pickle : bool
        If True, loads/saves model results from/to a pickle file for faster plotting development.
    output_path : str
        Path to save the output JSON file. Mandatory.
    """
    # Load configuration
    config = load_config("../config/settings.yaml")
    if input_json is None:
        raise ValueError(
            "No input JSON provided. Please specify an input JSON file using --input."
        )
    else:
        with open(input_json, "r") as f:
            inputs = json.load(f)

    if exec_dashboard and len(inputs) != 1:
        raise ValueError(
            "Dashboard execution is only supported for a single input. Please provide a single input JSON."
        )
    weather = WeatherDataHandler()

    # MAIN LOOP
    for input in inputs:
        # ----------------------------
        # 1. SETUP
        # ----------------------------
        # Check if the input is valid

        # Require unique_id in input
        if not input.get("unique_id"):
            raise ValueError("Missing required parameter: unique_id.")

        # Extract parameters
        detection_date = input["detection_date"]
        # output_formats = input["output_formats"]
        # TODO: replace with actual validation logic
        if (
            not input.get("detection_date")
            or not input.get("species")
            or not input.get("generations")
        ):
            raise ValueError(
                "Missing required parameters: detection_date, species, generations."
            )
        species_params = load_species_params(species=input["species"])
        # ----------------------------
        # weather loading
        # ----------------------------
        # Fetch weather data from detection date to the present
        detection_dt = pd.to_datetime(input["detection_date"])
        start_date = detection_dt.strftime("%Y-%m-%d")
        results = None
        all_historical = 1  # Default to historical unless prediction is run

        print(
            "Fetching weather data for:",
            input["species"],
            "from",
            start_date,
            "to now",
        )
        if exec_dashboard:
            buffer = True
        else:
            buffer = False
        weather_data = weather.fetch_data_gridmet(
            latitude=input["latitude"],
            longitude=input["longitude"],
            time_range=(start_date, pd.Timestamp.now()),
            use_buffer=buffer,
        )
        weather_data = weather_data.load()  # Load the dataset immediately
        # convert K to C
        weather_data["tmax"] = weather_data["tmax"] - 273.15
        weather_data["tmin"] = weather_data["tmin"] - 273.15
        print("Weather data loaded")

        # Remove weather data past the predict_from_date if specified
        if predict_from_date:
            predict_from_dt = pd.to_datetime(predict_from_date)
            weather_data = weather_data.sel(t=slice(None, predict_from_dt))
            print(f"Weather data truncated to {predict_from_date}")

        # ----------------------------
        # 2. MODELLING
        # ----------------------------
        # setup
        detection_date = pd.to_datetime(
            input["detection_date"], errors="coerce", format="%Y-%m-%d"
        )
        date_index = weather_data["t"].get_index("t").get_loc(input["detection_date"])
        detection_date = pd.to_datetime(input["detection_date"])
        # check if detection is already complete
        results = fflies_spatial_wrapper(
            tmin_xr=weather_data["tmin"],
            tmax_xr=weather_data["tmax"],
            start_day=date_index,
            stages=species_params,
            generations=input["generations"],
        )
        # if detection is not complete, fetch historical data and run prediction
        if results["incomplete_development"].any():
            print(
                "Incomplete development detected, fetching historical data for prediction."
            )
            start_year = detection_dt.year - 20
            historical_start_date = pd.Timestamp(
                year=start_year, month=1, day=1
            ).strftime("%Y-%m-%d")
            historical_weather_data = weather.fetch_data_gridmet(
                latitude=input["latitude"],
                longitude=input["longitude"],
                time_range=(historical_start_date, start_date),
                use_buffer=False,
            )
            historical_weather_data = historical_weather_data.load()
            historical_weather_data["tmax"] = historical_weather_data["tmax"] - 273.15
            historical_weather_data["tmin"] = historical_weather_data["tmin"] - 273.15
            # Combine historical and recent weather data
            weather_data = xr.concat([historical_weather_data, weather_data], dim="t")
            print("Historical weather data added for prediction.")
            results = fflies_prediction_wrapper(
                current_data=weather_data.isel(t=slice(date_index, None)),
                historical_data=weather_data,
                stages=species_params,
                detection_date=detection_date,
                generations=input["generations"],
                start_year=start_year,
                end_year=detection_dt.year - 1,
            )
            all_historical = 0
        else:
            print("No incomplete development detected, using historical results.")

        # ----------------------------
        # 3. POST-PROCESSING
        # ----------------------------
        # Convert days_to_completion to a date by adding to the detection_date
        # save results to a netcdf to open in a notebook
        # Add metadata and save netcdf

        # Attach metadata to the results xarray Dataset

        results.attrs["latitude"] = input["latitude"]
        results.attrs["longitude"] = input["longitude"]
        results.attrs["detection_date"] = input["detection_date"]

        results.to_netcdf("phillip_testing/santaclara_testing_off_results.nc")

        output = FfliesOutput(
            data=results,
            detection_date=input["detection_date"],
            generations=input["generations"],
            species=input["species"],
            all_historical=all_historical,
            latitude=input["latitude"],
            longitude=input["longitude"],
        )
        # Collect outputs for all inputs
        if "all_outputs" not in locals():
            all_outputs = []
        all_outputs.append(
            {"species": input["species"], "output": output.create_json()}
        )

    # Save or print all outputs as a single JSON
    # Use unique_id from the first input for filename
    unique_id = inputs[0].get("unique_id", "unknown")
    output_filename = f"results_{input_json.split('/')[-1].replace('.json','')}.json"
    output_file = os.path.join(output_path, output_filename)
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2)
    if exec_dashboard:
        # Write NetCDF output using the first output object
        netcdf_filename = (
            f"results_{unique_id}.nc"
            if len(all_outputs) > 1
            else f"{inputs[0]['species']}_{unique_id}_results.nc"
        )
        netcdf_file = os.path.join(output_path, netcdf_filename)
        # Use the first output object for NetCDF export
        output.to_netcdf(netcdf_file)
    if print_results:
        try:
            from tabulate import tabulate

            print(tabulate(dict_to_table_multi(all_outputs), headers=["Key", "Value"]))
        except ImportError:
            print(json.dumps(all_outputs, indent=2))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run FFLIES SAFARIS pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSON file. If not provided, uses test input.",
    )

    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print output JSON to terminal in a formatted, readable table.",
    )
    parser.add_argument(
        "--save-exec-dashboard",
        action="store_true",
        help="Flag to save netCDF for the dashboard with metadata.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs",  # Set default value to "outputs"
        help="Directory path to save the output JSON file. Defaults to 'outputs'.",
    )
    parser.add_argument(
        "--predict-from-date",
        type=str,
        default=None,
        help="Specify a cutoff date (YYYY-MM-DD) to remove weather data past this date and run prediction from detection date.",
    )

    args = parser.parse_args()

    return fflies_model(
        input_json=args.input,
        print_results=args.print_results,
        unique_id=getattr(args, "unique_id", None),
        exec_dashboard=getattr(args, "exec_dashboard", False),
        output_path=args.output_path,
        predict_from_date=args.predict_from_date,  # Pass the new argument
    )


if __name__ == "__main__":
    sys.exit(main())
