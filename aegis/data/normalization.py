"""
Provider-independent normalization of raw market data into canonical models.

This module converts raw dictionaries (as received from data providers)
into validated canonical Tick and OHLC instances with:

- Decimal precision for all financial values (no Decimal(float))
- Timezone-aware UTC timestamps (naive timestamps rejected)
- Normalized symbol strings (strip, uppercase, preserve structure)
- Timeframe normalization for OHLC candles

The module does not connect to providers, brokers, or execution systems.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from pydantic import ValidationError

from aegis.interfaces.errors import MalformedDataError
from aegis.interfaces.market_data import OHLC, Tick, Timeframe


def normalize_symbol(raw: str) -> str:
    """
    Normalize a symbol string.

    Rules:
    - Strip leading/trailing whitespace.
    - Convert to uppercase.
    - Preserve existing structure (do NOT insert separators).

    Examples:
        " usd/inr " -> "USD/INR"
        "usd/inr"   -> "USD/INR"
        "USDINR"    -> "USDINR"
        "NIFTY"     -> "NIFTY"

    Args:
        raw: Raw symbol string.

    Returns:
        Normalized symbol string.

    Raises:
        MalformedDataError: If the symbol is empty after stripping.
    """
    normalized = raw.strip().upper()

    if not normalized:
        raise MalformedDataError("Symbol is empty after normalization")

    return normalized


def _parse_timestamp(raw: Any) -> datetime:
    """
    Parse and normalize a timestamp to timezone-aware UTC.

    Accepts:
    - ISO 8601 strings with timezone info (including 'Z' suffix)
    - timezone-aware datetime objects

    Rejects:
    - Naive timestamps (no timezone info)
    - Malformed timestamp strings

    Args:
        raw: Raw timestamp value (string or datetime).

    Returns:
        Timezone-aware datetime in UTC.

    Raises:
        MalformedDataError:
            If the timestamp is naive, malformed, or cannot be parsed.
    """
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            raise MalformedDataError(
                "Timezone-naive datetime rejected: "
                "timestamps must include timezone information"
            )
        return raw.astimezone(timezone.utc)

    if not isinstance(raw, str):
        raise MalformedDataError(
            f"Unsupported timestamp type: {type(raw).__name__}"
        )

    try:
        # Replace 'Z' suffix with '+00:00' for fromisoformat compatibility
        iso_str = raw.strip()
        if iso_str.endswith("Z") or iso_str.endswith("z"):
            iso_str = iso_str[:-1] + "+00:00"

        parsed = datetime.fromisoformat(iso_str)
    except (ValueError, TypeError) as exc:
        raise MalformedDataError(
            f"Malformed timestamp '{raw}': {exc}"
        ) from exc

    if parsed.tzinfo is None:
        raise MalformedDataError(
            "Timezone-naive timestamp rejected: "
            f"'{raw}' does not include timezone information"
        )

    return parsed.astimezone(timezone.utc)


def _safe_decimal(value: Any, field_name: str) -> Decimal:
    """
    Convert a value to Decimal safely.

    Accepts string representations of numbers.
    Rejects malformed values.

    Args:
        value: The raw value to convert.
        field_name: Name of the field (for error messages).

    Returns:
        A Decimal instance.

    Raises:
        MalformedDataError: If the value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value

    if isinstance(value, float):
        # Avoid Decimal(float) precision issues — convert via string
        raise MalformedDataError(
            f"Field '{field_name}': raw float values are not accepted. "
            f"Use a string representation instead."
        )

    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MalformedDataError(
            f"Field '{field_name}': cannot convert '{value}' to Decimal: {exc}"
        ) from exc


def normalize_tick(raw: dict) -> Tick:
    """
    Normalize a raw dictionary into a canonical Tick instance.

    Required fields: symbol, timestamp, bid, ask.
    Optional fields: volume.

    All prices are converted to Decimal from string representations.
    Timestamps are normalized to UTC.
    Symbols are stripped and uppercased.

    Args:
        raw: Raw dictionary with tick data.

    Returns:
        A validated canonical Tick instance.

    Raises:
        MalformedDataError:
            If required fields are missing, values are malformed,
            or the data fails canonical Tick validation.
    """
    if not isinstance(raw, dict):
        raise MalformedDataError(
            f"Expected dict, got {type(raw).__name__}"
        )

    # --- symbol ---
    raw_symbol = raw.get("symbol")
    if raw_symbol is None:
        raise MalformedDataError("Missing required field: 'symbol'")
    symbol = normalize_symbol(str(raw_symbol))

    # --- timestamp ---
    raw_timestamp = raw.get("timestamp")
    if raw_timestamp is None:
        raise MalformedDataError("Missing required field: 'timestamp'")
    timestamp = _parse_timestamp(raw_timestamp)

    # --- bid ---
    raw_bid = raw.get("bid")
    if raw_bid is None:
        raise MalformedDataError("Missing required field: 'bid'")
    bid = _safe_decimal(raw_bid, "bid")

    # --- ask ---
    raw_ask = raw.get("ask")
    if raw_ask is None:
        raise MalformedDataError("Missing required field: 'ask'")
    ask = _safe_decimal(raw_ask, "ask")

    # --- volume (optional) ---
    raw_volume = raw.get("volume")
    volume: Optional[Decimal] = None
    if raw_volume is not None:
        volume = _safe_decimal(raw_volume, "volume")

    # --- construct and validate via Pydantic ---
    try:
        return Tick(
            symbol=symbol,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            volume=volume,
        )
    except (ValidationError, ValueError) as exc:
        raise MalformedDataError(
            f"Tick validation failed: {exc}"
        ) from exc


def normalize_ohlc(raw: dict) -> OHLC:
    """
    Normalize a raw dictionary into a canonical OHLC instance.

    Required fields: symbol, timestamp, timeframe, open, high, low, close.
    Optional fields: volume.

    All prices are converted to Decimal from string representations.
    Timestamps are normalized to UTC.
    Symbols are stripped and uppercased.
    Timeframes are matched case-insensitively.

    Args:
        raw: Raw dictionary with OHLC data.

    Returns:
        A validated canonical OHLC instance.

    Raises:
        MalformedDataError:
            If required fields are missing, values are malformed,
            or the data fails canonical OHLC validation.
    """
    if not isinstance(raw, dict):
        raise MalformedDataError(
            f"Expected dict, got {type(raw).__name__}"
        )

    # --- symbol ---
    raw_symbol = raw.get("symbol")
    if raw_symbol is None:
        raise MalformedDataError("Missing required field: 'symbol'")
    symbol = normalize_symbol(str(raw_symbol))

    # --- timestamp ---
    raw_timestamp = raw.get("timestamp")
    if raw_timestamp is None:
        raise MalformedDataError("Missing required field: 'timestamp'")
    timestamp = _parse_timestamp(raw_timestamp)

    # --- timeframe ---
    raw_timeframe = raw.get("timeframe")
    if raw_timeframe is None:
        raise MalformedDataError("Missing required field: 'timeframe'")

    try:
        timeframe = Timeframe(str(raw_timeframe).strip().upper())
    except ValueError as exc:
        raise MalformedDataError(
            f"Invalid timeframe '{raw_timeframe}': {exc}"
        ) from exc

    # --- prices ---
    price_fields = ("open", "high", "low", "close")
    prices: dict[str, Decimal] = {}
    for field in price_fields:
        raw_price = raw.get(field)
        if raw_price is None:
            raise MalformedDataError(
                f"Missing required field: '{field}'"
            )
        prices[field] = _safe_decimal(raw_price, field)

    # --- volume (optional) ---
    raw_volume = raw.get("volume")
    volume: Optional[Decimal] = None
    if raw_volume is not None:
        volume = _safe_decimal(raw_volume, "volume")

    # --- construct and validate via Pydantic ---
    try:
        return OHLC(
            symbol=symbol,
            timestamp=timestamp,
            timeframe=timeframe,
            open=prices["open"],
            high=prices["high"],
            low=prices["low"],
            close=prices["close"],
            volume=volume,
        )
    except (ValidationError, ValueError) as exc:
        raise MalformedDataError(
            f"OHLC validation failed: {exc}"
        ) from exc