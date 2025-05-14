# src/__init__.py

# Top-level exports (API surface)
from .core import DegreeDayModel, WeatherDataHandler
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
]
