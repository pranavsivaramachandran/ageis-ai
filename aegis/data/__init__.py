"""
AEGIS AI data processing pipeline.

Provides validation and normalization of market data:

- validation: Higher-level validation boundary around canonical models
- normalization: Provider-independent conversion to canonical Tick/OHLC

Public API:
    validate_tick, validate_ohlc, is_tick_stale,
    normalize_tick, normalize_ohlc, ValidationResult
"""

from aegis.data.validation import (
    ValidationResult,
    is_tick_stale,
    validate_ohlc,
    validate_tick,
)
from aegis.data.normalization import normalize_ohlc, normalize_tick

__all__ = [
    "ValidationResult",
    "validate_tick",
    "validate_ohlc",
    "is_tick_stale",
    "normalize_tick",
    "normalize_ohlc",
]
