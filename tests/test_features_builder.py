"""
Tests for aegis.features.builder module.

Validates the FeatureBuilder orchestration and FeatureVector output,
including symbol/timestamp/timeframe preservation, insufficient data
handling, and integration with technical feature functions.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.features.builder import FeatureBuilder, FeatureVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(
    close: str,
    index: int = 0,
    symbol: str = "USD/INR",
    timeframe: Timeframe = Timeframe.D1,
) -> OHLC:
    """Create a canonical OHLC candle with sensible defaults."""
    from datetime import timedelta

    c = Decimal(close)
    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return OHLC(
        symbol=symbol,
        timestamp=base_date + timedelta(days=index),
        timeframe=timeframe,
        open=c,
        high=c + Decimal("1"),
        low=max(c - Decimal("1"), Decimal("0.01")),
        close=c,
    )


def _make_candles(
    closes: list[str],
    symbol: str = "USD/INR",
    timeframe: Timeframe = Timeframe.D1,
) -> list[OHLC]:
    """Create a list of OHLC candles from close prices."""
    return [
        _make_candle(c, index=i, symbol=symbol, timeframe=timeframe)
        for i, c in enumerate(closes)
    ]


# ===================================================================
# FeatureVector
# ===================================================================

class TestFeatureVector:
    """Tests for the FeatureVector dataclass."""

    def test_feature_vector_is_frozen(self):
        """FeatureVector should be immutable."""
        vec = FeatureVector(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            symbol="USD/INR",
            timeframe=Timeframe.D1,
        )
        with pytest.raises(AttributeError):
            vec.symbol = "EUR/USD"  # type: ignore[misc]

    def test_feature_vector_defaults_to_none(self):
        """All optional features default to None."""
        vec = FeatureVector(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            symbol="USD/INR",
            timeframe=Timeframe.D1,
        )
        assert vec.returns is None
        assert vec.sma_value is None
        assert vec.ema_value is None
        assert vec.rsi_value is None
        assert vec.macd_line is None
        assert vec.macd_signal is None
        assert vec.macd_histogram is None
        assert vec.atr_value is None
        assert vec.bollinger_upper is None
        assert vec.bollinger_middle is None
        assert vec.bollinger_lower is None
        assert vec.momentum_value is None
        assert vec.volatility is None


# ===================================================================
# FeatureBuilder — basic construction
# ===================================================================

class TestFeatureBuilder:
    """Tests for FeatureBuilder.build()."""

    def test_returns_none_for_empty_candles(self):
        builder = FeatureBuilder()
        assert builder.build([]) is None

    def test_returns_none_for_single_candle(self):
        builder = FeatureBuilder()
        candles = _make_candles(["50"])
        assert builder.build(candles) is None

    def test_minimum_candles_property(self):
        builder = FeatureBuilder()
        assert builder.minimum_candles == 2

    def test_recommended_candles_property(self):
        builder = FeatureBuilder()
        # Default MACD slow=26, signal=9 → 34
        assert builder.recommended_candles >= 34

    def test_build_with_minimum_data(self):
        """With just 2 candles, only returns should be available."""
        builder = FeatureBuilder()
        candles = _make_candles(["50", "55"])
        result = builder.build(candles)
        assert result is not None
        assert isinstance(result, FeatureVector)
        assert result.returns is not None
        assert len(result.returns) == 1

    def test_build_returns_feature_vector(self):
        # Enough data for all features
        prices = [str(50 + i * 0.5) for i in range(40)]
        candles = _make_candles(prices)
        builder = FeatureBuilder()
        result = builder.build(candles)
        assert result is not None
        assert isinstance(result, FeatureVector)


# ===================================================================
# FeatureBuilder — metadata preservation
# ===================================================================

class TestFeatureBuilderMetadata:
    """Tests for symbol/timestamp/timeframe preservation."""

    def test_symbol_preserved(self):
        candles = _make_candles(["50", "55"], symbol="EUR/USD")
        builder = FeatureBuilder()
        result = builder.build(candles)
        assert result is not None
        assert result.symbol == "EUR/USD"

    def test_timestamp_from_latest_candle(self):
        candles = _make_candles(["50", "55", "60"])
        builder = FeatureBuilder()
        result = builder.build(candles)
        assert result is not None
        # Latest candle has index=2 → Jan 3
        assert result.timestamp == datetime(2026, 1, 3, tzinfo=timezone.utc)

    def test_timeframe_preserved(self):
        candles = _make_candles(
            ["50", "55"], timeframe=Timeframe.H1
        )
        builder = FeatureBuilder()
        result = builder.build(candles)
        assert result is not None
        assert result.timeframe == Timeframe.H1

    def test_mixed_symbols_raises(self):
        c1 = _make_candle("50", index=0, symbol="USD/INR")
        c2 = _make_candle("55", index=1, symbol="EUR/USD")
        builder = FeatureBuilder()
        with pytest.raises(ValueError, match="Mixed symbols"):
            builder.build([c1, c2])

    def test_mixed_timeframes_raises(self):
        c1 = _make_candle("50", index=0, timeframe=Timeframe.D1)
        c2 = _make_candle("55", index=1, timeframe=Timeframe.H1)
        builder = FeatureBuilder()
        with pytest.raises(ValueError, match="Mixed timeframes"):
            builder.build([c1, c2])


# ===================================================================
# FeatureBuilder — feature computation integration
# ===================================================================

class TestFeatureBuilderIntegration:
    """Integration tests: builder correctly delegates to feature functions."""

    def _large_candles(self, n: int = 40) -> list[OHLC]:
        """Create enough candles for all features."""
        prices = [str(50 + i * 0.5) for i in range(n)]
        return _make_candles(prices)

    def test_all_features_populated_with_sufficient_data(self):
        candles = self._large_candles(40)
        builder = FeatureBuilder()
        result = builder.build(candles)
        assert result is not None

        # All features should be computed with 40 candles
        assert result.returns is not None
        assert result.sma_value is not None
        assert result.ema_value is not None
        assert result.rsi_value is not None
        assert result.macd_line is not None
        assert result.macd_signal is not None
        assert result.macd_histogram is not None
        assert result.atr_value is not None
        assert result.bollinger_upper is not None
        assert result.bollinger_middle is not None
        assert result.bollinger_lower is not None
        assert result.momentum_value is not None
        assert result.volatility is not None

    def test_partial_features_with_limited_data(self):
        """With 5 candles, only some features should be computed."""
        candles = _make_candles(["50", "55", "52", "58", "54"])
        builder = FeatureBuilder(sma_period=3, ema_period=3)
        result = builder.build(candles)
        assert result is not None

        # Returns should work (5 candles → 4 returns)
        assert result.returns is not None
        assert len(result.returns) == 4

        # SMA/EMA with period=3 should work
        assert result.sma_value is not None
        assert result.ema_value is not None

        # MACD needs 34 candles by default → None
        assert result.macd_line is None

    def test_custom_periods(self):
        candles = self._large_candles(40)
        builder = FeatureBuilder(
            sma_period=5,
            ema_period=5,
            rsi_period=7,
            macd_fast=5,
            macd_slow=10,
            macd_signal=3,
            atr_period=7,
            bollinger_period=10,
            momentum_period=5,
            volatility_period=10,
        )
        result = builder.build(candles)
        assert result is not None
        assert result.sma_value is not None
        assert result.rsi_value is not None
        assert result.macd_line is not None

    def test_deterministic_build(self):
        candles = self._large_candles(40)
        builder = FeatureBuilder()
        r1 = builder.build(candles)
        r2 = builder.build(candles)
        assert r1 is not None and r2 is not None
        assert r1.sma_value == r2.sma_value
        assert r1.ema_value == r2.ema_value
        assert r1.rsi_value == r2.rsi_value
        assert r1.returns == r2.returns

    def test_no_silent_none_features_with_sufficient_data(self):
        """With 40 candles and default periods, nothing should be None."""
        candles = self._large_candles(40)
        builder = FeatureBuilder()
        result = builder.build(candles)
        assert result is not None

        features = [
            result.returns, result.sma_value, result.ema_value,
            result.rsi_value, result.macd_line, result.macd_signal,
            result.macd_histogram, result.atr_value,
            result.bollinger_upper, result.bollinger_middle,
            result.bollinger_lower, result.momentum_value,
            result.volatility,
        ]
        for feat in features:
            assert feat is not None, f"Feature unexpectedly None: {feat}"
