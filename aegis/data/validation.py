"""
Higher-level validation boundary for canonical market data models.

This module wraps the Pydantic-level validation in Tick/OHLC with a
clean functional API that:

1. Returns a structured ValidationResult instead of raising on every
   consumer call-site.
2. Provides deterministic staleness detection through an injectable
   reference_time.
3. Reuses the canonical Tick and OHLC models instead of duplicating
   their field-level validation rules.

The module does not connect to providers, brokers, or execution systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import ValidationError

from aegis.interfaces.market_data import OHLC, Tick


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of a market-data validation check.

    Attributes:
        is_valid:
            True when the data passes validation.
        error:
            Human-readable validation error when validation fails.
            None when validation succeeds.
    """

    is_valid: bool
    error: Optional[str] = None


def validate_tick(tick: Tick) -> ValidationResult:
    """
    Validate a canonical Tick instance.

    Tick already performs its structural validation through Pydantic,
    including:

    - positive bid/ask prices
    - bid <= ask
    - valid timestamp
    - UTC timestamp
    - non-negative volume

    This function provides a higher-level validation boundary for
    consumers that prefer a ValidationResult instead of exceptions.

    Args:
        tick:
            Canonical Tick instance to validate.

    Returns:
        ValidationResult indicating whether the tick is valid.
    """

    try:
        Tick.model_validate(tick.model_dump())
    except ValidationError as exc:
        return ValidationResult(
            is_valid=False,
            error=str(exc),
        )

    return ValidationResult(is_valid=True)


def validate_ohlc(ohlc: OHLC) -> ValidationResult:
    """
    Validate a canonical OHLC instance.

    OHLC already performs structural validation through Pydantic,
    including:

    - positive OHLC prices
    - valid timestamp
    - valid timeframe
    - high >= open
    - high >= close
    - low <= open
    - low <= close
    - high >= low
    - non-negative volume

    This function provides a higher-level validation boundary for
    consumers that prefer a ValidationResult instead of exceptions.

    Args:
        ohlc:
            Canonical OHLC instance to validate.

    Returns:
        ValidationResult indicating whether the candle is valid.
    """

    try:
        OHLC.model_validate(ohlc.model_dump())
    except ValidationError as exc:
        return ValidationResult(
            is_valid=False,
            error=str(exc),
        )

    return ValidationResult(is_valid=True)


def is_tick_stale(
    tick: Tick,
    max_age_seconds: float,
    reference_time: Optional[datetime] = None,
) -> bool:
    """
    Determine whether a tick is stale.

    The function accepts an injectable reference_time so callers and
    tests can perform deterministic age calculations.

    Args:
        tick:
            Canonical Tick whose timestamp will be evaluated.

        max_age_seconds:
            Maximum acceptable age of the tick in seconds.
            Must be non-negative.

        reference_time:
            Reference point in time used to calculate the tick age.
            If omitted, the current UTC time is used.

            The reference time must be timezone-aware and represent UTC.

    Returns:
        True if the tick is strictly older than max_age_seconds.
        False otherwise.

    Raises:
        ValueError:
            If max_age_seconds is negative.

            Also raised if reference_time is naive or does not
            represent UTC.
    """

    if max_age_seconds < 0:
        raise ValueError("max_age_seconds cannot be negative")

    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    if reference_time.tzinfo is None:
        raise ValueError(
            "reference_time must be timezone-aware UTC"
        )

    if reference_time.utcoffset() != timedelta(0):
        raise ValueError(
            "reference_time must represent UTC"
        )

    age = reference_time - tick.timestamp

    return age > timedelta(seconds=max_age_seconds)