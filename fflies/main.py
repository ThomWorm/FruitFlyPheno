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

import time


def fflies_model(
    input_json=None,
    print_results=False,
    output_path=None,
    # use_pickle=False,
    unique_id=None,  # unique id imported from input JSON, but can be overridden by CLI
    exec_dashboard=False,
    use_pickle=False,  # Load/save model results from/to a pickle file for faster plotting development
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

    # CLI/GUI/web form returns a list of dicts
    weather = WeatherDataHandler(cache_dir=config["weather"]["cache_dir"])

    # Remove plot-related checks
    # if plot and len(inputs) > 1:
    #     raise ValueError("Plotting is only supported for a single input.")

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
        # Set start_date to Jan 1st, 20 years before detection_date

        detection_dt = pd.to_datetime(input["detection_date"])
        start_year = detection_dt.year - 20
        start_date = pd.Timestamp(year=start_year, month=1, day=1).strftime("%Y-%m-%d")
        pickle_filename = f"{input['species']}_results.pkl"
        results = None
        all_historical = 1  # Default to historical unless prediction is run
        if use_pickle and os.path.exists(pickle_filename):
            with open(pickle_filename, "rb") as pf:
                results = pickle.load(pf)
            print(f"Loaded model results from {pickle_filename}")
        else:
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
                use_buffer=buffer,  # Plotting buffer no longer relevant
            )
            # Ensure t is a single chunk for apply_ufunc compatibility
            weather_data = weather_data.load()
            # convert weather data from kelvin to celsius
            weather_data["tmax"] = weather_data["tmax"] - 273.15
            weather_data["tmin"] = weather_data["tmin"] - 273.15
            print("Weather data loaded")
            # ----------------------------
            # 2. MODELLING
            # ----------------------------
            test_idx = (
                weather_data["t"].get_index("t").get_loc(input["detection_date"])
            )  # TODO double check that loc off of a string works _ I think it does
            detection_date = pd.to_datetime(input["detection_date"])
            results = fflies_spatial_wrapper(
                weather_data["tmax"], weather_data["tmin"], test_idx, species_params
            )
            if results["incomplete_development"].any():
                print("Incomplete development detected, running prediction.")
                detection_dt = pd.to_datetime(input["detection_date"])
                results = fflies_prediction_wrapper(
                    current_data=weather_data.isel(t=slice(test_idx, None)),
                    historical_data=weather_data,
                    stages=species_params,
                    detection_date=detection_date,
                    generations=input["generations"],
                    start_year=detection_dt.year - 20,
                    end_year=detection_dt.year - 1,
                )
                all_historical = 0
            else:
                print("No incomplete development detected, using historical results.")
            if use_pickle:
                with open(pickle_filename, "wb") as pf:
                    pickle.dump(results, pf)
                print(f"Saved model results to {pickle_filename}")
        # ----------------------------
        # 3. POST-PROCESSING
        # ----------------------------
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
    output_filename = (
        f"results_{unique_id}.json"
        if len(all_outputs) > 1
        else f"{inputs[0]['species']}_{unique_id}_results.json"
    )
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

    parser = argparse.ArgumentParser(description="Run FruitFlyPheno main pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSON file. If not provided, uses test input.",
    )
    # Remove plot-related arguments
    # parser.add_argument(
    #     "--plot",
    #     action="store_true",
    #     help="Flag to plot results instead of saving JSON. If used in Colab/Jupyter, displays inline.",
    # )
    # parser.add_argument(
    #     "--save-plot",
    #     type=str,
    #     default=None,
    #     help="If provided, saves the plot as an HTML file to this path (only used with --plot).",
    # )
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print output JSON to terminal in a formatted, readable table.",
    )
    parser.add_argument(
        "--use-pickle",
        action="store_true",
        help="Load/save model results from/to a pickle file for faster plotting development.",
    )
    parser.add_argument(
        "--unique-id",
        type=str,
        default=None,
        help="Unique ID for this run. Overrides unique_id in input JSON if provided.",
    )
    parser.add_argument(
        "--exec-dashboard",
        action="store_true",
        help="Flag to execute the dashboard after running the pipeline.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory path to save the output JSON file. Mandatory.",
    )

    args = parser.parse_args()

    return fflies_model(
        input_json=args.input,
        print_results=args.print_results,
        use_pickle=args.use_pickle,
        unique_id=getattr(args, "unique_id", None),
        exec_dashboard=getattr(args, "exec_dashboard", False),
        output_path=args.output_path,
    )


if __name__ == "__main__":
    sys.exit(main())
