# main.py
# Adjusted imports to reflect the new location of main.py in src
from utils import fflies_prediction, fflies_spatial_wrapper
from src import (
    WeatherDataHandler,
    load_species_params,
    get_user_input,
    load_config,
)


def main():
    # Load configuration
    config = load_config("../config/settings.yaml")
    inputs = get_user_input(test_mode=True)  # CLI/GUI/web form

    for input in inputs:
        # Check if the input is valid

        # Extract parameters
        detection_date = input.get("detection_date")
        species = input.get("species")
        generations = input.get("generations")
        # output_formats = input.get("output_formats", [])
        input_latitude = input.get("latitude")
        input_longitude = input.get("longitude")
        # TODO: replace with actual validation logic
        if not detection_date or not species or not generations:
            raise ValueError(
                "Missing required parameters: detection_date, species, generations."
            )

        species_params = load_species_params(
            species=species, data_path="../config/fly_models.json"
        )
        # ----------------------------
        # weather loading
        # ----------------------------
        # TODO : replace with ZARR server calls when made
        # unless ZARR is much slower than expected, extract all 20 years of data from the server
        # weather_data = config["weather"]["data_path"]

        weather = WeatherDataHandler(config["weather"], input_latitude, input_longitude)
        weather_data = weather.load_cached()

        # ----------------------------
        # 2. MODELLING
        # ----------------------------
        test_idx = weather_data["t"].get_index("t").get_loc(detection_date)
        recent_weather_run = fflies_spatial_wrapper(
            weather_data["tmax"], weather_data["tmin"], test_idx, species_params
        )
        if recent_weather_run["incomplete_development"].any():
            prediction_run = fflies_prediction(
                current_data=weather_data.isel(t=slice(test_idx, None)),
                historical_data=weather_data,
                stages=species_params,
                detection_date=detection_date,
                generations=generations,
                start_year=2021,  # TODO replace with years calculated from the data
                end_year=2024,
            )

        return 0

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
