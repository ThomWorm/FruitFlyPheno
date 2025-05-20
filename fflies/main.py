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


def main():
    # Load configuration
    config = load_config("../config/settings.yaml")
    inputs = get_user_input(test_mode=True)  # CLI/GUI/web form returns a list of dicts

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
            not input["detection_date"]
            or not input["species"]
            or not input["generations"]
        ):
            raise ValueError(
                "Missing required parameters: detection_date, species, generations."
            )

        species_params = load_species_params(species=input["species"])
        # ----------------------------
        # weather loading
        # ----------------------------
        # TODO : replace with ZARR server calls when made
        # unless ZARR is much slower than expected, extract all 20 years of data from the server
        # weather_data = config["weather"]["data_path"]

        weather = WeatherDataHandler(
            cache_dir=config["weather"]["cache_dir"],
            latitude=input["latitude"],
            longitude=input["longitude"],
            credentials={},
        )
        weather_data = (
            weather.load_cached()
        )  # TODO: replace with weather.fetch_remote_data() when server is ready

        # ----------------------------
        # 2. MODELLING
        # ----------------------------
        test_idx = weather_data["t"].get_index("t").get_loc(input["detection_date"])
        detection_date = pd.to_datetime(input["detection_date"])
        results = fflies_spatial_wrapper(
            weather_data["tmax"], weather_data["tmin"], test_idx, species_params
        )
        all_historical = 1
        if results["incomplete_development"].any():
            results = fflies_prediction_wrapper(
                current_data=weather_data.isel(t=slice(test_idx, None)),
                historical_data=weather_data,
                stages=species_params,
                detection_date=detection_date,
                generations=input["generations"],
                start_year=2021,  # TODO replace with years calculated from the data
                end_year=2024,
            )
            all_historical = 0

        # ----------------------------
        # 3. POST-PROCESSING
        # ----------------------------
        output = FfliesOutput(
            data=results,
            detection_date=detection_date,
            generations=input["generations"],
            species=input["species"],
            all_historical=all_historical,
        )

    """
    # ----------------------------
    # 5. OUTPUT GENERATION
    # ----------------------------
    if "plot" in inputs["output_formats"]:
        output.generate_plots(
            results,
            filename=f"{inputs['species']}_generation_{inputs['target_date']}.png",
        )

    if "json" in inputs["output_formats"]:
        output.save_json(results, filename=f"{inputs['species']}_results.json")

    if "report" in inputs["output_formats"]:
        output.generate_report(
            results, species=inputs["species"], date=inputs["target_date"]
        )

    print("Pipeline executed successfully!")
    """


if __name__ == "__main__":
    main()
