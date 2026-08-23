"""
AEGIS AI prediction package.

Provides prediction contracts and the prediction engine.
"""

from aegis.prediction.models import (
    PredictionDirection,
    PredictionResult,
)
from aegis.prediction.engine import (
    BaselinePredictor,
    PredictionError,
)

__all__ = [
    "PredictionDirection",
    "PredictionResult",
    "BaselinePredictor",
    "PredictionError",
]