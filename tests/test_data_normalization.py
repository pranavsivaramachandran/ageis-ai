"""
Tests for aegis.data.normalization module.

Validates the provider-independent normalization layer that converts
raw dictionaries into canonical Tick/OHLC models with safe financial
precision and timezone handling.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from aegis.interfaces.errors import MalformedDataError
from aegis.interfaces.market_data import OHLC, Tick, Timeframe
from aegis.data.normalization import normalize_ohlc, normalize_tick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    """Return a UTC ISO-8601 timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _valid_tick_dict(**overrides) -> dict:
    defaults = {
        "symbol": "USD/INR",
        "timestamp": _utc_iso(),
        "bid": "83.2500",
        "ask": "83.2600",
    }
    defaults.update(overrides)
    return defaults


def _valid_ohlc_dict(**overrides) -> dict:
    defaults = {
        "symbol": "USD/INR",
        "timestamp": _utc_iso(),
        "timeframe": "M15",
        "open": "83.2500",
        "high": "83.3000",
        "low": "83.2000",
        "close": "83.2800",
    }
    defaults.update(overrides)
    return defaults


# ===================================================================
# normalize_tick — happy path
# ===================================================================

class TestNormalizeTick:
    """Tests for normalize_tick()."""

    def test_valid_tick_dict(self):
        tick = normalize_tick(_valid_tick_dict())
        assert isinstance(tick, Tick)
        assert tick.symbol == "USD/INR"
        assert tick.bid == Decimal("83.2500")
        assert tick.ask == Decimal("83.2600")
        assert tick.timestamp.tzinfo is not None

    def test_decimal_precision_preserved(self):
        raw = _valid_tick_dict(bid="1.123456789", ask="1.123456790")
        tick = normalize_tick(raw)
        assert tick.bid == Decimal("1.123456789")
        assert tick.ask == Decimal("1.123456790")

    def test_volume_optional_defaults_none(self):
        raw = _valid_tick_dict()
        raw.pop("volume", None)
        tick = normalize_tick(raw)
        assert tick.volume is None

    def test_volume_present_converted(self):
        raw = _valid_tick_dict(volume="15000")
        tick = normalize_tick(raw)
        assert tick.volume == Decimal("15000")


# ===================================================================
# normalize_tick — timestamp handling
# ===================================================================

class TestNormalizeTickTimestamps:
    """Timestamp conversion and rejection tests."""

    def test_timezone_offset_converted_to_utc(self):
        """A timestamp with +05:30 offset must be normalized to UTC."""
        raw = _valid_tick_dict(timestamp="2025-06-15T17:30:00+05:30")
        tick = normalize_tick(raw)
        assert tick.timestamp.utcoffset() == timedelta(0)
        assert tick.timestamp == datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_utc_z_suffix_accepted(self):
        raw = _valid_tick_dict(timestamp="2025-06-15T12:00:00Z")
        tick = normalize_tick(raw)
        assert tick.timestamp.utcoffset() == timedelta(0)

    def test_naive_timestamp_rejected(self):
        raw = _valid_tick_dict(timestamp="2025-06-15T12:00:00")
        with pytest.raises(MalformedDataError, match="[Tt]imezone"):
            normalize_tick(raw)

    def test_malformed_timestamp_rejected(self):
        raw = _valid_tick_dict(timestamp="not-a-date")
        with pytest.raises(MalformedDataError):
            normalize_tick(raw)

    def test_datetime_object_accepted(self):
        """Passing a datetime object directly (not a string)."""
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        raw = _valid_tick_dict(timestamp=ts)
        tick = normalize_tick(raw)
        assert tick.timestamp == ts


# ===================================================================
# normalize_tick — symbol normalization
# ===================================================================

class TestNormalizeTickSymbol:
    """Symbol string normalization tests."""

    def test_symbol_whitespace_normalized(self):
        raw = _valid_tick_dict(symbol="  usd/inr  ")
        tick = normalize_tick(raw)
        assert tick.symbol == "USD/INR"

    def test_symbol_without_separator_preserved(self):
        """A symbol without a separator must remain unchanged.

        The normalization layer must not invent a financial
        instrument mapping (e.g. USDINR must NOT become USD/INR).
        """
        raw = _valid_tick_dict(symbol="USDINR")
        tick = normalize_tick(raw)
        assert tick.symbol == "USDINR"

    def test_symbol_already_canonical(self):
        raw = _valid_tick_dict(symbol="USD/INR")
        tick = normalize_tick(raw)
        assert tick.symbol == "USD/INR"

    def test_symbol_lowercase_uppercased(self):
        raw = _valid_tick_dict(symbol="eur/usd")
        tick = normalize_tick(raw)
        assert tick.symbol == "EUR/USD"

    def test_non_pair_symbol_preserved(self):
        """Symbols that aren't 6-char pairs keep their form (just uppercased)."""
        raw = _valid_tick_dict(symbol="nifty50")
        tick = normalize_tick(raw)
        assert tick.symbol == "NIFTY50"


# ===================================================================
# normalize_tick — error cases
# ===================================================================

class TestNormalizeTickErrors:
    """Rejection and error handling tests."""

    def test_missing_required_field_rejected(self):
        raw = _valid_tick_dict()
        del raw["bid"]
        with pytest.raises(MalformedDataError):
            normalize_tick(raw)

    def test_malformed_price_rejected(self):
        raw = _valid_tick_dict(bid="abc")
        with pytest.raises(MalformedDataError):
            normalize_tick(raw)

    def test_missing_symbol_rejected(self):
        raw = _valid_tick_dict()
        del raw["symbol"]
        with pytest.raises(MalformedDataError):
            normalize_tick(raw)

    def test_empty_symbol_rejected(self):
        raw = _valid_tick_dict(symbol="")
        with pytest.raises(MalformedDataError):
            normalize_tick(raw)

    def test_invalid_data_never_silently_passes(self):
        """Negative bid must be caught, never silently accepted."""
        raw = _valid_tick_dict(bid="-1.0")
        with pytest.raises(MalformedDataError):
            normalize_tick(raw)


# ===================================================================
# normalize_ohlc — happy path
# ===================================================================

class TestNormalizeOhlc:
    """Tests for normalize_ohlc()."""

    def test_valid_ohlc_dict(self):
        ohlc = normalize_ohlc(_valid_ohlc_dict())
        assert isinstance(ohlc, OHLC)
        assert ohlc.symbol == "USD/INR"
        assert ohlc.timeframe == Timeframe.M15
        assert ohlc.open == Decimal("83.2500")

    def test_decimal_precision_preserved_ohlc(self):
        raw = _valid_ohlc_dict(
            open="1.123456789",
            high="1.223456789",
            low="1.023456789",
            close="1.123456789",
        )
        ohlc = normalize_ohlc(raw)
        assert ohlc.open == Decimal("1.123456789")

    def test_volume_optional_ohlc(self):
        raw = _valid_ohlc_dict()
        raw.pop("volume", None)
        ohlc = normalize_ohlc(raw)
        assert ohlc.volume is None

    def test_timeframe_case_insensitive(self):
        raw = _valid_ohlc_dict(timeframe="m15")
        ohlc = normalize_ohlc(raw)
        assert ohlc.timeframe == Timeframe.M15

    def test_timeframe_h1(self):
        raw = _valid_ohlc_dict(timeframe="h1")
        ohlc = normalize_ohlc(raw)
        assert ohlc.timeframe == Timeframe.H1


# ===================================================================
# normalize_ohlc — error cases
# ===================================================================

class TestNormalizeOhlcErrors:
    """Rejection and error handling tests for OHLC normalization."""

    def test_missing_timeframe_rejected(self):
        raw = _valid_ohlc_dict()
        del raw["timeframe"]
        with pytest.raises(MalformedDataError):
            normalize_ohlc(raw)

    def test_invalid_timeframe_rejected(self):
        raw = _valid_ohlc_dict(timeframe="X99")
        with pytest.raises(MalformedDataError):
            normalize_ohlc(raw)

    def test_naive_timestamp_rejected_ohlc(self):
        raw = _valid_ohlc_dict(timestamp="2025-06-15T12:00:00")
        with pytest.raises(MalformedDataError, match="[Tt]imezone"):
            normalize_ohlc(raw)

    def test_malformed_price_rejected_ohlc(self):
        raw = _valid_ohlc_dict(open="xyz")
        with pytest.raises(MalformedDataError):
            normalize_ohlc(raw)

    def test_missing_close_rejected(self):
        raw = _valid_ohlc_dict()
        del raw["close"]
        with pytest.raises(MalformedDataError):
            normalize_ohlc(raw)
