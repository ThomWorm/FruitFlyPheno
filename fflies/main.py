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
import csv
import warnings

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


def _apply_temperature_quality_checks(
    ds: xr.Dataset,
    source_label: str,
    missing_day_threshold: int = 20,
) -> tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
    """
    Apply temperature quality checks and return:
    - sanitized dataset
    - per-cell/day QC failure mask (1 for failed QC, 0 otherwise)
    - per-cell invalid-cell mask (True where missing days exceed threshold)

    Rules:
    - Missing temperature days count as QC failures.
    - If tmin > tmax on an individual day, set tmin = tmax so the day contributes 0 DD.
    - If a cell has more than `missing_day_threshold` missing days, the entire cell is invalidated.
    """
    missing = ds["tmin"].isnull() | ds["tmax"].isnull()
    invalid_order = (
        ds["tmin"].notnull()
        & ds["tmax"].notnull()
        & (ds["tmin"] > ds["tmax"])
    )

    qc_missing_count = int(missing.sum().item())
    qc_invalid_order_count = int(invalid_order.sum().item())
    
    qc_fail_mask = (missing | invalid_order).astype("int16")
    qc_fail_count = int(qc_fail_mask.sum().item())

    if qc_fail_count > 0:
        days_with_qc_issues = int(qc_fail_mask.any(dim=["latitude", "longitude"]).sum().item())
        warnings.warn(
            (
                f"{source_label}: found {qc_fail_count} cell-day record(s) failing QC "
                f"across {days_with_qc_issues} day(s). \n"
                f"Missing days: {qc_missing_count}, \n"
                f"tmin>tmax days: {qc_invalid_order_count}."

            ),
            RuntimeWarning,
            stacklevel=2,
        )

    # Correct impossible ordering so the day contributes 0 degree days.
    if bool(invalid_order.any()):
        ds["tmin"] = xr.where(invalid_order, ds["tmax"], ds["tmin"])

    missing_days_per_cell = missing.sum(dim="t")
    invalid_cell_mask = missing_days_per_cell > missing_day_threshold
    invalid_cell_count = int(invalid_cell_mask.sum().item())

    if invalid_cell_count > 0:
        warnings.warn(
            (
                f"{source_label}: {invalid_cell_count} cell(s) exceeded the missing-day "
                f"threshold of {missing_day_threshold} and were set to NaN."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        ds["tmin"] = ds["tmin"].where(~invalid_cell_mask)
        ds["tmax"] = ds["tmax"].where(~invalid_cell_mask)

    return ds, qc_fail_mask, invalid_cell_mask


def fflies_model(
    input_json=None,
    print_results=False,
    output_path="outputs",
    unique_id=None,  # unique id imported from input JSON, but can be overridden by CLI
    exec_dashboard=False,
    use_pickle=False,  # Load/save model results from/to a pickle file for faster plotting development
    predict_from_date=None,  # New parameter for prediction cutoff date
    output_format="csv",  # New parameter: "csv", "json", or "both"
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
        Path to save the output file(s). Mandatory.
    output_format : str
        Format for output files: "csv" (default), "json", or "both".
    """
    # Load configuration
    config = load_config("../config/settings.yaml")
    weather_cfg = config.get("weather", {})
    fly_models_cfg = config.get("fly_models", {})
    weather_source = weather_cfg.get("source", "gridmet")
    history_years = int(weather_cfg.get("history_years", 20))
    recent_buffer_deg = weather_cfg.get("recent_buffer_deg", 0.1)
    dashboard_buffer_deg = weather_cfg.get("dashboard_buffer_deg", recent_buffer_deg)
    historical_buffer_deg = weather_cfg.get("historical_buffer_deg", recent_buffer_deg)
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
    weather = WeatherDataHandler(
        cache_dir=config.get("weather", {}).get("cache_dir"),
    )
    if weather_cfg.get("gridmet_urls"):
        weather.gridmet_urls = weather_cfg["gridmet_urls"]

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
        species_params = load_species_params(
            species=input["species"],
            models_path=fly_models_cfg.get("path"),
        )
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
        if weather_source == "cache":
            recent_weather_data = weather.load_cached()
            recent_qc_fail_mask = xr.zeros_like(recent_weather_data["tmin"], dtype="int16")
            recent_invalid_cell_mask = xr.zeros_like(recent_weather_data["tmin"].isel(t=0), dtype=bool)
        else:
            buffer_deg = dashboard_buffer_deg if exec_dashboard else recent_buffer_deg
            recent_weather_data = weather.fetch_data_gridmet(
                latitude=input["latitude"],
                longitude=input["longitude"],
                time_range=(start_date, pd.Timestamp.now()),
                use_buffer=exec_dashboard,
                buffer_deg=buffer_deg,
            )
            recent_weather_data = recent_weather_data.load()  # Load the dataset immediately
            # convert K to C
            recent_weather_data["tmax"] = recent_weather_data["tmax"] - 273.15
            recent_weather_data["tmin"] = recent_weather_data["tmin"] - 273.15

            # Remove weather data past the predict_from_date if specified
            if predict_from_date:
                predict_from_dt = pd.to_datetime(predict_from_date)
                recent_weather_data = recent_weather_data.sel(t=slice(None, predict_from_dt))
                print(f"Weather data truncated to {predict_from_date}")

            (
                recent_weather_data,
                recent_qc_fail_mask,
                recent_invalid_cell_mask,
            ) = _apply_temperature_quality_checks(recent_weather_data, "Recent weather")
        print("Weather data loaded")

        # ----------------------------
        # 2. MODELLING
        # ----------------------------
        # setup
        detection_date = pd.to_datetime(
            input["detection_date"], errors="coerce", format="%Y-%m-%d"
        )
        date_index = recent_weather_data["t"].get_index("t").get_loc(input["detection_date"])
        detection_date = pd.to_datetime(input["detection_date"])
        # check if detection is already complete
        results = fflies_spatial_wrapper(
            tmin_xr=recent_weather_data["tmin"],
            tmax_xr=recent_weather_data["tmax"],
            start_day=date_index,
            stages=species_params,
            generations=input["generations"],
            qc_fail_mask_xr=recent_qc_fail_mask,
        )
        # if detection is not complete, fetch historical data and run prediction
        if results["incomplete_development"].any():
            print(
                "Incomplete development detected, fetching historical data for prediction."
            )
            
            start_year = detection_dt.year - history_years
            historical_start_date = pd.Timestamp(
                year=start_year, month=1, day=1
            ).strftime("%Y-%m-%d")
            historical_weather_data = weather.fetch_data_gridmet(
                latitude=input["latitude"],
                longitude=input["longitude"],
                time_range=(historical_start_date, start_date),
                use_buffer=False,
                buffer_deg=historical_buffer_deg,
            )
            historical_weather_data = historical_weather_data.load()
            historical_weather_data["tmax"] = historical_weather_data["tmax"] - 273.15
            historical_weather_data["tmin"] = historical_weather_data["tmin"] - 273.15
            (
                historical_weather_data,
                historical_qc_fail_mask,
                historical_invalid_cell_mask,
            ) = _apply_temperature_quality_checks(
                historical_weather_data, "Historical weather"
            )
            # Combine historical and recent weather data
            print("Historical weather data added for prediction.")
            results = fflies_prediction_wrapper(
                current_data=recent_weather_data,
                historical_data=historical_weather_data,
                stages=species_params,
                detection_date=detection_date,
                generations=input["generations"],
                start_year=start_year,
                end_year=detection_dt.year - 1,
                current_qc_fail_mask=recent_qc_fail_mask,
                historical_qc_fail_mask=historical_qc_fail_mask,
            )
            combined_invalid_cell_mask = recent_invalid_cell_mask | historical_invalid_cell_mask
            qc_layer = results["data_quality_fail_days"].copy()
            results = results.where(~combined_invalid_cell_mask)
            results["data_quality_fail_days"] = qc_layer
            all_historical = 0
        else:
            print("No incomplete development detected, using historical results.")
            # Add a year dimension to match the structure from prediction wrapper
            # This ensures consistent data structure for downstream processing
            qc_layer = recent_qc_fail_mask.sum(dim="t").expand_dims(year=[detection_dt.year])
            results = results.expand_dims(year=[detection_dt.year])
            results["data_quality_fail_days"] = qc_layer
            qc_layer = results["data_quality_fail_days"].copy()
            results = results.where(~recent_invalid_cell_mask)
            results["data_quality_fail_days"] = qc_layer

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

        output = FfliesOutput(
            data=results,
            detection_date=input["detection_date"],
            generations=input["generations"],
            species=input["species"],
            all_historical=all_historical,
            latitude=input["latitude"],
            longitude=input["longitude"],
            unique_id=input.get("unique_id", ""),
        )
        # Collect outputs for all inputs
        if "all_outputs" not in locals():
            all_outputs = []
            all_csv_data = []  # For CSV format
            all_output_objects = []
        output_dict = output.create_json()
        all_outputs.append({"species": input["species"], "output": output_dict})
        all_csv_data.extend(output.create_csv())  # Add CSV row(s)
        all_output_objects.append(output)

    # Save outputs based on output_format parameter
    # Use unique_id from the first input for filename
    unique_id = inputs[0].get("unique_id", "unknown")
    base_filename = f"results_{input_json.split('/')[-1].replace('.json','')}"
    
    # Save CSV output (default or if requested)
    if output_format in ["csv", "both"]:
        csv_filename = f"{base_filename}.csv"
        csv_file = os.path.join(output_path, csv_filename)
        
        # Write CSV file
        if all_csv_data:
            with open(csv_file, "w", newline="") as f:
                fieldnames = all_csv_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_csv_data)
            print(f"CSV output saved to: {csv_file}")
    
    # Save JSON output (if requested)
    if output_format in ["json", "both"]:
        json_filename = f"{base_filename}.json"
        json_file = os.path.join(output_path, json_filename)
        with open(json_file, "w") as f:
            json.dump(all_outputs, f, indent=2)
        print(f"JSON output saved to: {json_file}")
    if exec_dashboard:
        # Write NetCDF output(s) for dashboard production.
        # Save one .nc per input so multi-input runs remain traceable.
        os.makedirs(output_path, exist_ok=True)
        for output_obj in all_output_objects:
            out_unique_id = output_obj.unique_id if output_obj.unique_id else "unknown"
            netcdf_filename = f"{output_obj.species}_{out_unique_id}_results.nc"
            netcdf_file = os.path.join(output_path, netcdf_filename)
            output_obj.to_netcdf(netcdf_file)
            print(f"NetCDF output saved to: {netcdf_file}")
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
    parser.add_argument(
        "--output-format",
        type=str,
        default="csv",
        choices=["csv", "json", "both"],
        help="Output format: 'csv' (default), 'json', or 'both'.",
    )

    args = parser.parse_args()

    return fflies_model(
        input_json=args.input,
        print_results=args.print_results,
        unique_id=getattr(args, "unique_id", None),
        exec_dashboard=getattr(args, "save_exec_dashboard", False),
        output_path=args.output_path,
        predict_from_date=args.predict_from_date,  # Pass the new argument
        output_format=args.output_format,  # Pass the output format argument
    )


if __name__ == "__main__":
    sys.exit(main())
