# fflies/__init__.py

# Top-level exports (API surface)
from .core import (
    DegreeDayModel,
    WeatherDataHandler,
    fflies_prediction_wrapper,
    fflies_spatial_wrapper,
    FfliesOutput,
)
from .io_handlers import load_config, OutputGenerator, get_user_input
from .utils import load_species_params

# Optional but useful:
__all__ = [
    "DegreeDayModel",
    "WeatherDataHandler",
    "load_config",
    "OutputGenerator",
    "load_species_params",
    "get_user_input",
    "fflies_prediction_wrapper",
    "fflies_spatial_wrapper",
    "FfliesOutput",
]
