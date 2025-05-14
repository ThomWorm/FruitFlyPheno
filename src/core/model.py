# src/core/model.py
import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, List, Optional
from utils.degree_day_equations import single_sine_horizontal_cutoff


from typing import Dict, Tuple, Union, List
from utils.degree_day_equations import single_sine_horizontal_cutoff

import numpy as np


def fflies_core(
    tmin_1d: np.ndarray,
    tmax_1d: np.ndarray,
    start_day: int,
    stages: List[Dict],
    generations: int = 3,
) -> np.ndarray:
    """
    Returns:
    - A NumPy array of results for each generation:
      - For completed generations: days elapsed
      - For incomplete generations: -1.0
    """
    current_day = start_day
    total_days = len(tmin_1d)
    generation_results = np.full(generations, -1.0, dtype=float)  # Initialize with -1.0

    for gen in range(1, generations + 1):
        stage_accumulator = 0.0

        for stage_idx, stage in enumerate(stages):
            stage_dd = 0.0
            days_in_stage = 0
            while current_day < total_days:
                # Calculate degree days for all remaining days
                dd = single_sine_horizontal_cutoff(
                    tmin_1d[current_day],
                    tmax_1d[current_day],
                    stage["LTT"],
                    stage["UTT"],
                )

                stage_dd += dd
                stage_accumulator += dd
                days_in_stage += 1
                current_day += 1

                if stage_accumulator >= stage["dd_threshold"]:
                    break

            if stage_accumulator < stage["dd_threshold"]:
                # Incomplete stage - triggers when the loop ends without reaching the threshold
                generation_results[gen - 1] = -1.0  # Mark as incomplete
                return generation_results  # Return results up to this point

        # Generation completed
        days_elapsed = current_day - start_day
        generation_results[gen - 1] = float(
            days_elapsed
        )  # Append result for this generation

    return generation_results  # Return results for all generations


def fflies_spatial_wrapper(
    tmin_xr: xr.DataArray,  # (time, lat, lon)
    tmax_xr: xr.DataArray,  # (time, lat, lon)
    start_day: int,
    stages: List[Dict],
    generations: int = 3,
) -> xr.Dataset:
    """Simplified wrapper matching core outputs"""
    results = xr.apply_ufunc(
        fflies_core,
        tmin_xr,
        tmax_xr,
        input_core_dims=[["t"], ["t"]],
        kwargs={"start_day": start_day, "stages": stages, "generations": generations},
        output_core_dims=[["generation"]],
        output_dtypes=[float],
        vectorize=True,
        dask="parallelized",
        exclude_dims={"t"},
    )

    return xr.Dataset(
        {
            "days_to_completion": results.where(
                results >= 0
            ),  # Only show valid completions
            "incomplete_development": (results == -1),
            "missing_data": np.isnan(results),  # Original missing data
        },
        coords={
            "generation": np.arange(1, generations + 1),
        },
    )


def fflies_prediction_wrapper(
    current_data: xr.Dataset,  # tmin/tmax (time, lat, lon)
    historical_data: xr.Dataset,  # tmin/tmax (year, time, lat, lon)
    stages: List[Dict],
    detection_date: pd.Timestamp,
    generations: int = 3,
    start_year: int = 2021,
    end_year: int = 2023,
) -> xr.Dataset:
    """
    Predicts development using:
    1. Current year data until available
    2. Continues with each historical year's data

    Returns xr.Dataset with (year, generation, lat, lon) containing:
    - total_days: Days from start to completion
    - completed: Boolean whether full development occurred
    """
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)
    shape = (
        n_years,
        generations,
        len(current_data.latitude),
        len(current_data.longitude),
    )

    outputs = xr.Dataset(
        {
            "days_to_completion": (
                ("year", "generation", "latitude", "longitude"),
                np.full(shape, np.nan, dtype=np.float32),
            ),
            "incomplete_development": (
                ("year", "generation", "latitude", "longitude"),
                np.full(shape, False, dtype=bool),
            ),
            "missing_data": (
                ("year", "generation", "latitude", "longitude"),
                np.full(shape, False, dtype=bool),
            ),
        },
        coords={
            "year": years,
            "generation": np.arange(1, generations + 1),
            "latitude": current_data.latitude,
            "longitude": current_data.longitude,
        },
    )

    detection_day_of_year = detection_date.dayofyear
    days_recent_data = len(current_data.t)
    for year in range(start_year, end_year + 1):
        historical_date = pd.Timestamp(
            year=year, month=detection_date.month, day=detection_date.day
        )
        historical_index = historical_data.get_index("t").get_loc(historical_date)
        historical_tmin = historical_data["tmin"].isel(
            t=slice(historical_index + days_recent_data, None)
        )
        historical_tmax = historical_data["tmax"].isel(
            t=slice(historical_index + days_recent_data, None)
        )

        model_run_tmax = xr.concat([current_data["tmax"], historical_tmax], dim="t")
        model_run_tmin = xr.concat([current_data["tmin"], historical_tmin], dim="t")

        result = fflies_spatial_wrapper(
            model_run_tmin,
            model_run_tmax,
            start_day=0,
            stages=stages,
            generations=generations,
        )
        outputs["days_to_completion"].loc[dict(year=year)] = result[
            "days_to_completion"
        ]
        outputs["incomplete_development"].loc[dict(year=year)] = result[
            "incomplete_development"
        ]
        outputs["missing_data"].loc[dict(year=year)] = result["missing_data"]

    return outputs


'''
class DegreeDayModel:
    def __init__(self, stages_library: Dict[str, List[Dict]]):
        """
        Args:
            stages_library: Dictionary of {species_name: [stage1_params, stage2_params]}
        """
        self.stages_library = stages_library

    def run(
        self,
        current_data: xr.Dataset,
        historical_data: Optional[xr.Dataset] = None,
        detection_date: Optional[pd.Timestamp] = None,
        species: str = "default",
        generations: int = 3,
        prediction_years: tuple = (2021, 2023),
    ) -> xr.Dataset:
        """
        Main execution method that handles both recent and prediction runs

        Args:
            current_data: tmin/tmax (time, lat, lon)
            historical_data: tmin/tmax (year, time, lat, lon)
            detection_date: When to switch to historical data
            species: Key for stages_library
            generations: Number of generations to model
            prediction_years: Range of years for historical prediction

        Returns:
            xr.Dataset with completion data
        """
        stages = self.stages_library[species]

        # Run initial model
        recent_results = self._run_recent(
            current_data=current_data, stages=stages, generations=generations
        )

        # Check if prediction needed
        if historical_data is not None and detection_date is not None:
            if recent_results["incomplete_development"].any():
                prediction = self._run_prediction(
                    current_data=current_data,
                    historical_data=historical_data,
                    stages=stages,
                    detection_date=detection_date,
                    generations=generations,
                    years=prediction_years,
                )
                return self._combine_results(recent_results, prediction)

        return recent_results

    def _run_recent(
        self, current_data: xr.Dataset, stages: List[Dict], generations: int
    ) -> xr.Dataset:
        """Run model on recent data only"""
        return self._spatial_wrapper(
            tmin=current_data["tmin"],
            tmax=current_data["tmax"],
            stages=stages,
            generations=generations,
        )

    def _run_prediction(
        self,
        current_data: xr.Dataset,
        historical_data: xr.Dataset,
        stages: List[Dict],
        detection_date: pd.Timestamp,
        generations: int,
        years: tuple,
    ) -> xr.Dataset:
        """Run historical prediction for incomplete areas"""
        start_year, end_year = years
        return self._prediction_wrapper(
            current_data=current_data,
            historical_data=historical_data,
            stages=stages,
            detection_date=detection_date,
            generations=generations,
            start_year=start_year,
            end_year=end_year,
        )

    def _core_calculation(
        self,
        tmin_1d: np.ndarray,
        tmax_1d: np.ndarray,
        start_day: int,
        stages: List[Dict],
        generations: int = 3,
    ) -> float:
        """Core degree day accumulation logic"""
        if np.any(np.isnan(tmin_1d)) or np.any(np.isnan(tmax_1d)):
            return np.nan

        current_day = start_day
        total_days = len(tmin_1d)

        for gen in range(1, generations + 1):
            stage_accumulator = 0.0

            for stage in stages:
                while current_day < total_days:
                    dd = single_sine_horizontal_cutoff(
                        tmin_1d[current_day],
                        tmax_1d[current_day],
                        stage["LTT"],
                        stage["UTT"],
                    )
                    stage_accumulator += dd
                    current_day += 1

                    if stage_accumulator >= stage["dd_threshold"]:
                        break

                if stage_accumulator < stage["dd_threshold"]:
                    return -1.0  # Incomplete development

        return float(current_day - start_day)  # Successful completion

    def _spatial_wrapper(
        self,
        tmin: xr.DataArray,
        tmax: xr.DataArray,
        stages: List[Dict],
        generations: int = 3,
    ) -> xr.Dataset:
        """Vectorized spatial processing"""
        results = xr.apply_ufunc(
            self._core_calculation,
            tmin,
            tmax,
            input_core_dims=[["time"], ["time"]],
            kwargs={"start_day": 0, "stages": stages, "generations": generations},
            output_core_dims=[[]],
            output_dtypes=[float],
            vectorize=True,
            dask="parallelized",
            exclude_dims={"time"},
        )

        return xr.Dataset(
            {
                "days_to_completion": results.where(results >= 0),
                "incomplete_development": (results == -1),
                "missing_data": np.isnan(results),
            }
        )

    def _prediction_wrapper(
        self,
        current_data: xr.Dataset,
        historical_data: xr.Dataset,
        stages: List[Dict],
        detection_date: pd.Timestamp,
        generations: int,
        start_year: int,
        end_year: int,
    ) -> xr.Dataset:
        """Historical prediction workflow"""
        years = np.arange(start_year, end_year + 1)
        shape = (len(years), len(current_data.latitude), len(current_data.longitude))

        outputs = xr.Dataset(
            {
                "days_to_completion": (
                    ("year", "latitude", "longitude"),
                    np.full(shape, np.nan, dtype=np.float32),
                ),
                "incomplete_development": (
                    ("year", "latitude", "longitude"),
                    np.full(shape, False, dtype=bool),
                ),
                "missing_data": (
                    ("year", "latitude", "longitude"),
                    np.full(shape, False, dtype=bool),
                ),
            },
            coords={
                "year": years,
                "latitude": current_data.latitude,
                "longitude": current_data.longitude,
            },
        )

        days_recent_data = len(current_data.time)

        for year in years:
            hist_date = pd.Timestamp(
                year=year, month=detection_date.month, day=detection_date.day
            )
            hist_idx = historical_data.time.get_loc(hist_date)

            combined_tmin = xr.concat(
                [
                    current_data["tmin"],
                    historical_data["tmin"].isel(
                        time=slice(hist_idx + days_recent_data, None)
                    ),
                ],
                dim="time",
            )

            combined_tmax = xr.concat(
                [
                    current_data["tmax"],
                    historical_data["tmax"].isel(
                        time=slice(hist_idx + days_recent_data, None)
                    ),
                ],
                dim="time",
            )

            result = self._spatial_wrapper(
                tmin=combined_tmin,
                tmax=combined_tmax,
                stages=stages,
                generations=generations,
            )

            outputs["days_to_completion"].loc[{"year": year}] = result[
                "days_to_completion"
            ]
            outputs["incomplete_development"].loc[{"year": year}] = result[
                "incomplete_development"
            ]
            outputs["missing_data"].loc[{"year": year}] = result["missing_data"]

        return outputs

    def _combine_results(
        self, recent: xr.Dataset, historical: xr.Dataset
    ) -> xr.Dataset:
        """Merge recent and historical results"""
        # For simplicity, return historical results with recent metadata
        historical.attrs.update(recent.attrs)
        return historical
'''
