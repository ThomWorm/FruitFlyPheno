# main.py


from core import (
    WeatherDataHandler,
    FfliesOutput,
    fflies_prediction_wrapper,
    fflies_spatial_wrapper,
)
from io_handlers import load_config, get_user_input
from utils import load_species_params
import pandas as pd
import panel as pn
import sys
import json

import time


def main(input_json=None, plot=False, save_plot=False, plot_save_path=None):
    # Load configuration
    config = load_config("../config/settings.yaml")
    if input_json == "test" or input_json is None:
        input_json = "test"
        inputs = get_user_input(test_mode=True)
    else:
        with open(input_json, "r") as f:
            inputs = json.load(f)

        # CLI/GUI/web form returns a list of dicts
    weather = WeatherDataHandler(cache_dir=config["weather"]["cache_dir"])

    if plot and len(inputs) > 1:
        raise ValueError("Plotting is only supported for a single input.")

    for input in inputs:
        # ----------------------------
        # 1. SETUP
        # ----------------------------
        # Check if the input is valid

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

        if input_json is None:
            weather_data = weather.load_cached()
        else:
            detection_dt = pd.to_datetime(input["detection_date"])
            start_year = detection_dt.year - 20
            start_date = pd.Timestamp(year=start_year, month=1, day=1).strftime(
                "%Y-%m-%d"
            )
            print("Fetching weather data for:", input["species"])
            print("Start date:", start_date)
            weather_data = weather.fetch_data_gridmet(
                latitude=input["latitude"],
                longitude=input["longitude"],
                time_range=(start_date, pd.Timestamp.now()),
                use_buffer=plot,  # Use buffer if plotting
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
        print("Running spatial wrapper for:", input["species"])
        print("Detection date:", input["detection_date"])
        test_idx = (
            weather_data["t"].get_index("t").get_loc(input["detection_date"])
        )  # TODO double check that loc off of a string works _ I think it does
        detection_date = pd.to_datetime(input["detection_date"])
        results = fflies_spatial_wrapper(
            weather_data["tmax"], weather_data["tmin"], test_idx, species_params
        )
        all_historical = 1
        if results["incomplete_development"].any():
            print("Incomplete development detected, running prediction wrapper.")
            detection_dt = pd.to_datetime(input["detection_date"])
            results = fflies_prediction_wrapper(
                current_data=weather_data.isel(t=slice(test_idx, None)),
                historical_data=weather_data,
                stages=species_params,
                detection_date=detection_date,
                generations=input["generations"],
                start_year=detection_dt.year
                - 20,  # TODO replace with years calculated from the data
                end_year=detection_dt.year
                - 1,  # TODO replace with years calculated from the data
            )
            all_historical = 0
        else:
            print("No incomplete development detected, using spatial results.")
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
        if plot:
            save_path = None
            if save_plot:
                if plot_save_path:
                    save_path = plot_save_path
                else:
                    save_path = f"{input['species']}_results_plot.html"
            plot_panel = output.plot(save_path=save_path)
            server = plot_panel.show(open=True)
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                server.stop()
        else:
            # Output JSON file per input
            output.create_json(filename=f"{input['species']}_results.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FruitFlyPheno main pipeline.")
    parser.add_argument(
        "--input", type=str, default=None, help="Path to input JSON file."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Flag to plot results instead of saving JSON.",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Flag to save the plot as an HTML file (only used with --plot).",
    )
    parser.add_argument(
        "--plot-save-path",
        type=str,
        default=None,
        help="Path to save the plot HTML file (used with --save-plot).",
    )
    args = parser.parse_args()

    # If --input is not provided, pass None to main (will use test input)
    main(
        input_json=args.input,
        plot=args.plot,
        save_plot=args.save_plot,
        plot_save_path=args.plot_save_path,
    )
