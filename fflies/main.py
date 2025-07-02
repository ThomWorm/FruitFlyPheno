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
import pickle
import os

import time


def is_notebook():
    """
    Returns True if running in a Jupyter/Colab notebook, else False.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "Shell":
            return True  # Google Colab
        else:
            return False  # Other type (likely terminal)
    except Exception:
        return False


def main(input_json=None, plot=False, save_plot=None, print_json=False, use_pickle=False):
    """
    Main entry point for FruitFlyPheno pipeline.

    Parameters:
    -----------
    input_json : str or None
        Path to input JSON file, or 'test' to use test input. If None, uses test input.
    plot : bool
        If True, display interactive plot inline (Colab/Jupyter) or in browser (local).
    save_plot : str or None
        If provided, saves the plot as an HTML file to this path. If None, does not save.
    print_json : bool
        If True, prints the output JSON to the terminal in a formatted, readable table.
    """
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

        detection_dt = pd.to_datetime(input["detection_date"])
        start_year = detection_dt.year - 20
        start_date = pd.Timestamp(year=start_year, month=1, day=1).strftime(
            "%Y-%m-%d"
        )
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
        if plot:
            save_path = save_plot if save_plot else None
            plot_panel = output.plot(save_path=save_path)
            # Always load the Panel extension before plotting
            pn.extension("bokeh")
            if is_notebook():
                from IPython.display import display

                display(plot_panel)
            else:
                from core.outputs import serve_panel
                serve_panel(plot_panel, port=5006, open_browser=True)
                print("Panel server started on http://localhost:5006. Press Ctrl+C to stop.")
        else:
            # Output JSON file per input
            output_json = output.create_json()
            with open(f"{input['species']}_results.json", "w") as f:
                json.dump(output_json, f, indent=2)
            if print_json:
                try:
                    from tabulate import tabulate

                    def dict_to_table(d):
                        # Flatten dict for tabulate
                        rows = []
                        for k, v in d.items():
                            if isinstance(v, dict):
                                for k2, v2 in v.items():
                                    if isinstance(v2, dict):
                                        for k3, v3 in v2.items():
                                            rows.append([f"{k}.{k2}.{k3}", v3])
                                    else:
                                        rows.append([f"{k}.{k2}", v2])
                            else:
                                rows.append([k, v])
                        return rows

                    print(
                        tabulate(dict_to_table(output_json), headers=["Key", "Value"])
                    )
                except ImportError:
                    print(json.dumps(output_json, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FruitFlyPheno main pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSON file. If not provided, uses test input.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Flag to plot results instead of saving JSON. If used in Colab/Jupyter, displays inline.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="If provided, saves the plot as an HTML file to this path (only used with --plot).",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print output JSON to terminal in a formatted, readable table.",
    )
    parser.add_argument(
        "--use-pickle",
        action="store_true",
        help="Load/save model results from/to a pickle file for faster plotting development."
    )
    args = parser.parse_args()

    # If --input is not provided, pass None to main (will use test input)
    main(
        input_json=args.input,
        plot=args.plot,
        save_plot=args.save_plot,
        print_json=args.print_json,
        use_pickle=args.use_pickle,
    )
