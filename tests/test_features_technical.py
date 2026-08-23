"""
Tests for aegis.features.technical module.

Validates deterministic, provider-independent technical feature
functions against known reference values and edge cases.
"""

from datetime import datetime, timezone
from decimal import Decimal
import math

import pytest

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.features.technical import (
    atr,
    bollinger_bands,
    ema,
    macd,
    momentum,
    rolling_volatility,
    rsi,
    simple_returns,
    sma,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(
    close: str,
    high: str | None = None,
    low: str | None = None,
    open_: str | None = None,
    index: int = 0,
) -> OHLC:
    """Create a canonical OHLC candle with sensible defaults."""
    from datetime import timedelta

    c = Decimal(close)
    h = Decimal(high) if high else c + Decimal("0.5")
    l = Decimal(low) if low else c - Decimal("0.5")
    o = Decimal(open_) if open_ else c

    # Ensure OHLC constraints: high >= open, close; low <= open, close
    h = max(h, o, c)
    l = min(l, o, c)
    if l <= Decimal("0"):
        l = Decimal("0.01")

    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

    return OHLC(
        symbol="USD/INR",
        timestamp=base_date + timedelta(days=index),
        timeframe=Timeframe.D1,
        open=o,
        high=h,
        low=l,
        close=c,
    )


def _make_candles(closes: list[str]) -> list[OHLC]:
    """Create a list of OHLC candles from close prices."""
    return [_make_candle(c, index=i) for i, c in enumerate(closes)]


def _make_candles_ohlc(
    data: list[tuple[str, str, str, str]],
) -> list[OHLC]:
    """Create candles from (open, high, low, close) tuples."""
    from datetime import timedelta

    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i, (o, h, l, c) in enumerate(data):
        candles.append(
            OHLC(
                symbol="USD/INR",
                timestamp=base_date + timedelta(days=i),
                timeframe=Timeframe.D1,
                open=Decimal(o),
                high=Decimal(h),
                low=Decimal(l),
                close=Decimal(c),
            )
        )
    return candles


# ===================================================================
# Simple Returns
# ===================================================================

class TestSimpleReturns:
    """Tests for simple_returns()."""

    def test_valid_returns(self):
        candles = _make_candles(["100", "110", "105"])
        result = simple_returns(candles)
        assert result is not None
        assert len(result) == 2
        assert result[0] == pytest.approx(0.1, abs=1e-10)
        assert result[1] == pytest.approx(-0.04545454545, abs=1e-8)

    def test_insufficient_data(self):
        candles = _make_candles(["100"])
        assert simple_returns(candles) is None

    def test_empty_list(self):
        assert simple_returns([]) is None

    def test_two_candles_minimum(self):
        candles = _make_candles(["100", "200"])
        result = simple_returns(candles)
        assert result is not None
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0, abs=1e-10)

    def test_deterministic(self):
        candles = _make_candles(["100", "110", "105", "115"])
        r1 = simple_returns(candles)
        r2 = simple_returns(candles)
        assert r1 == r2


# ===================================================================
# SMA
# ===================================================================

class TestSMA:
    """Tests for sma()."""

    def test_valid_sma(self):
        candles = _make_candles(["10", "20", "30", "40", "50"])
        result = sma(candles, period=3)
        # SMA of last 3: (30 + 40 + 50) / 3 = 40
        assert result == pytest.approx(40.0, abs=1e-10)

    def test_sma_period_equals_data(self):
        candles = _make_candles(["10", "20", "30"])
        result = sma(candles, period=3)
        assert result == pytest.approx(20.0, abs=1e-10)

    def test_insufficient_data(self):
        candles = _make_candles(["10", "20"])
        assert sma(candles, period=3) is None

    def test_period_one(self):
        candles = _make_candles(["42"])
        result = sma(candles, period=1)
        assert result == pytest.approx(42.0, abs=1e-10)

    def test_invalid_period(self):
        candles = _make_candles(["10", "20"])
        assert sma(candles, period=0) is None
        assert sma(candles, period=-1) is None

    def test_deterministic(self):
        candles = _make_candles(["10", "20", "30", "40"])
        assert sma(candles, 3) == sma(candles, 3)


# ===================================================================
# EMA
# ===================================================================

class TestEMA:
    """Tests for ema()."""

    def test_valid_ema(self):
        candles = _make_candles(["10", "20", "30"])
        result = ema(candles, period=2)
        # Seed = SMA(first 2) = 15
        # k = 2/3
        # EMA = 30 * 2/3 + 15 * 1/3 = 20 + 5 = 25
        assert result == pytest.approx(25.0, abs=1e-10)

    def test_ema_equals_sma_at_seed(self):
        """When data length == period, EMA == SMA."""
        candles = _make_candles(["10", "20", "30"])
        result = ema(candles, period=3)
        expected_sma = sma(candles, period=3)
        assert result == pytest.approx(expected_sma, abs=1e-10)

    def test_insufficient_data(self):
        candles = _make_candles(["10"])
        assert ema(candles, period=2) is None

    def test_period_one(self):
        candles = _make_candles(["10", "20", "30"])
        result = ema(candles, period=1)
        # k = 2/2 = 1.0, so EMA always equals last close
        assert result == pytest.approx(30.0, abs=1e-10)

    def test_invalid_period(self):
        candles = _make_candles(["10", "20"])
        assert ema(candles, period=0) is None

    def test_deterministic(self):
        candles = _make_candles(["10", "20", "30", "40", "50"])
        assert ema(candles, 3) == ema(candles, 3)


# ===================================================================
# RSI
# ===================================================================

class TestRSI:
    """Tests for rsi()."""

    def test_valid_rsi(self):
        # 15 candles needed for period=14 (14 changes)
        prices = [str(44 + i) for i in range(15)]
        candles = _make_candles(prices)
        result = rsi(candles, period=14)
        assert result is not None
        # All gains, no losses → RSI = 100
        assert result == pytest.approx(100.0, abs=1e-10)

    def test_all_losses_rsi(self):
        prices = [str(60 - i) for i in range(15)]
        candles = _make_candles(prices)
        result = rsi(candles, period=14)
        assert result is not None
        # All losses, no gains → RSI = 0
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_rsi_range_0_to_100(self):
        prices = ["50", "52", "51", "53", "50", "48", "50", "52",
                   "51", "49", "50", "52", "54", "53", "55"]
        candles = _make_candles(prices)
        result = rsi(candles, period=14)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_insufficient_data(self):
        candles = _make_candles(["50"] * 14)  # Need 15
        assert rsi(candles, period=14) is None

    def test_period_one(self):
        candles = _make_candles(["50", "55"])
        result = rsi(candles, period=1)
        assert result is not None
        assert result == pytest.approx(100.0, abs=1e-10)

    def test_deterministic(self):
        prices = [str(50 + i % 5) for i in range(20)]
        candles = _make_candles(prices)
        assert rsi(candles, 14) == rsi(candles, 14)

    def test_no_change_rsi(self):
        """All same prices: no gains, no losses → RSI = 50."""
        candles = _make_candles(["50"] * 16)
        result = rsi(candles, period=14)
        assert result is not None
        assert result == pytest.approx(50.0, abs=1e-10)


# ===================================================================
# MACD
# ===================================================================

class TestMACD:
    """Tests for macd()."""

    def test_valid_macd(self):
        # Need slow + signal - 1 = 26 + 9 - 1 = 34 candles
        prices = [str(50 + i * 0.5) for i in range(35)]
        candles = _make_candles(prices)
        result = macd(candles)
        assert result is not None
        macd_line, signal_line, histogram = result
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)
        assert histogram == pytest.approx(macd_line - signal_line, abs=1e-10)

    def test_insufficient_data(self):
        candles = _make_candles(["50"] * 33)  # Need 34
        assert macd(candles) is None

    def test_fast_greater_than_slow_rejected(self):
        candles = _make_candles(["50"] * 50)
        assert macd(candles, fast=26, slow=12) is None

    def test_invalid_periods(self):
        candles = _make_candles(["50"] * 50)
        assert macd(candles, fast=0, slow=26, signal=9) is None
        assert macd(candles, fast=12, slow=0, signal=9) is None
        assert macd(candles, fast=12, slow=26, signal=0) is None

    def test_histogram_identity(self):
        prices = [str(50 + (i % 10) * 0.3) for i in range(40)]
        candles = _make_candles(prices)
        result = macd(candles)
        assert result is not None
        ml, sl, hist = result
        assert hist == pytest.approx(ml - sl, abs=1e-10)

    def test_deterministic(self):
        prices = [str(50 + i * 0.1) for i in range(40)]
        candles = _make_candles(prices)
        assert macd(candles) == macd(candles)


# ===================================================================
# ATR
# ===================================================================

class TestATR:
    """Tests for atr()."""

    def test_valid_atr(self):
        # Use consistent OHLC data
        data = [
            ("50", "52", "49", "51"),
            ("51", "53", "50", "52"),
            ("52", "54", "51", "53"),
        ]
        candles = _make_candles_ohlc(data)
        result = atr(candles, period=2)
        assert result is not None
        assert result > 0.0

    def test_insufficient_data(self):
        data = [("50", "52", "49", "51")]
        candles = _make_candles_ohlc(data)
        assert atr(candles, period=1) is None  # Need period+1=2

    def test_zero_range_atr(self):
        """When all prices are identical, ATR approaches zero."""
        data = [("50", "50", "50", "50")] * 16
        candles = _make_candles_ohlc(data)
        result = atr(candles, period=14)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_atr_always_non_negative(self):
        data = [
            ("50", "55", "45", "52"),
            ("52", "54", "48", "50"),
            ("50", "53", "47", "51"),
            ("51", "56", "46", "53"),
        ]
        candles = _make_candles_ohlc(data)
        result = atr(candles, period=3)
        assert result is not None
        assert result >= 0.0

    def test_deterministic(self):
        data = [("50", "52", "49", "51")] * 20
        candles = _make_candles_ohlc(data)
        assert atr(candles, 14) == atr(candles, 14)


# ===================================================================
# Bollinger Bands
# ===================================================================

class TestBollingerBands:
    """Tests for bollinger_bands()."""

    def test_valid_bollinger(self):
        candles = _make_candles(["10", "12", "11", "13", "14"])
        result = bollinger_bands(candles, period=3)
        assert result is not None
        upper, middle, lower = result
        # Middle = SMA(3) of last 3 = (11+13+14)/3 ≈ 12.667
        assert middle == pytest.approx(12.666666, abs=1e-4)
        assert upper > middle
        assert lower < middle

    def test_band_symmetry(self):
        """Upper and lower bands should be equidistant from middle."""
        candles = _make_candles(["10", "20", "30", "40", "50"])
        result = bollinger_bands(candles, period=5)
        assert result is not None
        upper, middle, lower = result
        assert upper - middle == pytest.approx(middle - lower, abs=1e-10)

    def test_zero_std_dev(self):
        """All same prices → upper == middle == lower."""
        candles = _make_candles(["50"] * 5)
        result = bollinger_bands(candles, period=5)
        assert result is not None
        upper, middle, lower = result
        assert upper == pytest.approx(50.0, abs=1e-10)
        assert middle == pytest.approx(50.0, abs=1e-10)
        assert lower == pytest.approx(50.0, abs=1e-10)

    def test_insufficient_data(self):
        candles = _make_candles(["10", "20"])
        assert bollinger_bands(candles, period=3) is None

    def test_deterministic(self):
        candles = _make_candles(["10", "20", "30", "40", "50"])
        assert bollinger_bands(candles, 3) == bollinger_bands(candles, 3)


# ===================================================================
# Momentum
# ===================================================================

class TestMomentum:
    """Tests for momentum()."""

    def test_valid_momentum(self):
        candles = _make_candles(["100", "110", "120"])
        result = momentum(candles, period=2)
        # close[-1] - close[-3] = 120 - 100 = 20
        assert result == pytest.approx(20.0, abs=1e-10)

    def test_negative_momentum(self):
        candles = _make_candles(["120", "110", "100"])
        result = momentum(candles, period=2)
        assert result == pytest.approx(-20.0, abs=1e-10)

    def test_zero_momentum(self):
        candles = _make_candles(["100", "110", "100"])
        result = momentum(candles, period=2)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data(self):
        candles = _make_candles(["100"])
        assert momentum(candles, period=1) is None  # Need 2

    def test_deterministic(self):
        candles = _make_candles(["100", "110", "105", "115"])
        assert momentum(candles, 2) == momentum(candles, 2)


# ===================================================================
# Rolling Volatility
# ===================================================================

class TestRollingVolatility:
    """Tests for rolling_volatility()."""

    def test_valid_volatility(self):
        candles = _make_candles(["100", "110", "105", "115"])
        result = rolling_volatility(candles, period=3)
        assert result is not None
        assert result > 0.0

    def test_zero_volatility(self):
        """Constant prices → zero returns → zero volatility."""
        candles = _make_candles(["100"] * 5)
        result = rolling_volatility(candles, period=4)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data(self):
        candles = _make_candles(["100", "110"])
        assert rolling_volatility(candles, period=2) is None  # Need 3

    def test_volatility_non_negative(self):
        candles = _make_candles(["50", "55", "48", "52", "50"])
        result = rolling_volatility(candles, period=4)
        assert result is not None
        assert result >= 0.0

    def test_deterministic(self):
        candles = _make_candles(["100", "110", "105", "115", "108"])
        assert rolling_volatility(candles, 3) == rolling_volatility(candles, 3)


# ===================================================================
# Cross-cutting: no silent invalid values
# ===================================================================

class TestNoSilentInvalidValues:
    """Verify that no function silently produces NaN or Inf."""

    def test_no_nan_in_returns(self):
        candles = _make_candles(["100", "110", "105"])
        result = simple_returns(candles)
        assert result is not None
        for val in result:
            if val is not None:
                assert not math.isnan(val)
                assert not math.isinf(val)

    def test_no_nan_in_sma(self):
        candles = _make_candles(["100", "200", "300"])
        result = sma(candles, 3)
        assert result is not None
        assert not math.isnan(result)
        assert not math.isinf(result)

    def test_no_nan_in_ema(self):
        candles = _make_candles(["100", "200", "300"])
        result = ema(candles, 3)
        assert result is not None
        assert not math.isnan(result)
        assert not math.isinf(result)

    def test_no_nan_in_rsi(self):
        prices = [str(50 + i) for i in range(16)]
        candles = _make_candles(prices)
        result = rsi(candles, 14)
        assert result is not None
        assert not math.isnan(result)
        assert not math.isinf(result)

    def test_no_nan_in_bollinger(self):
        candles = _make_candles(["100"] * 20)
        result = bollinger_bands(candles, 20)
        assert result is not None
        for val in result:
            assert not math.isnan(val)
            assert not math.isinf(val)
