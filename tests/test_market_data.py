from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from aegis.interfaces.market_data import OHLC, Tick, Timeframe


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TestTick:
    def test_valid_tick(self):
        tick = Tick(
            symbol="EUR/USD",
            timestamp=utc_now(),
            bid=Decimal("1.10000"),
            ask=Decimal("1.10020"),
            volume=Decimal("1000"),
        )

        assert tick.symbol == "EUR/USD"
        assert tick.bid == Decimal("1.10000")
        assert tick.ask == Decimal("1.10020")

    def test_mid_price(self):
        tick = Tick(
            symbol="EUR/USD",
            timestamp=utc_now(),
            bid=Decimal("1.10000"),
            ask=Decimal("1.10020"),
        )

        assert tick.mid == Decimal("1.10010")

    def test_bid_cannot_exceed_ask(self):
        with pytest.raises(ValidationError):
            Tick(
                symbol="EUR/USD",
                timestamp=utc_now(),
                bid=Decimal("1.10100"),
                ask=Decimal("1.10000"),
            )

    def test_tick_prices_must_be_positive(self):
        with pytest.raises(ValidationError):
            Tick(
                symbol="EUR/USD",
                timestamp=utc_now(),
                bid=Decimal("-1"),
                ask=Decimal("1.1"),
            )

    def test_tick_volume_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            Tick(
                symbol="EUR/USD",
                timestamp=utc_now(),
                bid=Decimal("1.1"),
                ask=Decimal("1.2"),
                volume=Decimal("-1"),
            )

    def test_fresh_tick_is_not_stale(self):
        tick = Tick(
            symbol="EUR/USD",
            timestamp=utc_now(),
            bid=Decimal("1.1"),
            ask=Decimal("1.2"),
        )

        assert tick.is_stale(max_age_seconds=60) is False

    def test_old_tick_is_stale(self):
        tick = Tick(
            symbol="EUR/USD",
            timestamp=utc_now() - timedelta(minutes=5),
            bid=Decimal("1.1"),
            ask=Decimal("1.2"),
        )

        assert tick.is_stale(max_age_seconds=60) is True


class TestOHLC:
    def valid_candle(self):
        return OHLC(
            symbol="EUR/USD",
            timestamp=utc_now(),
            timeframe=Timeframe.M15,
            open=Decimal("1.10000"),
            high=Decimal("1.10200"),
            low=Decimal("1.09800"),
            close=Decimal("1.10100"),
            volume=Decimal("5000"),
        )

    def test_valid_ohlc(self):
        candle = self.valid_candle()

        assert candle.symbol == "EUR/USD"
        assert candle.timeframe == Timeframe.M15
        assert candle.high >= candle.open
        assert candle.high >= candle.close
        assert candle.low <= candle.open
        assert candle.low <= candle.close

    def test_high_cannot_be_below_open(self):
        with pytest.raises(ValidationError):
            OHLC(
                symbol="EUR/USD",
                timestamp=utc_now(),
                timeframe=Timeframe.M15,
                open=Decimal("1.10500"),
                high=Decimal("1.10200"),
                low=Decimal("1.09800"),
                close=Decimal("1.10000"),
            )

    def test_high_cannot_be_below_close(self):
        with pytest.raises(ValidationError):
            OHLC(
                symbol="EUR/USD",
                timestamp=utc_now(),
                timeframe=Timeframe.M15,
                open=Decimal("1.10000"),
                high=Decimal("1.10200"),
                low=Decimal("1.09800"),
                close=Decimal("1.10500"),
            )

    def test_low_cannot_be_above_open(self):
        with pytest.raises(ValidationError):
            OHLC(
                symbol="EUR/USD",
                timestamp=utc_now(),
                timeframe=Timeframe.M15,
                open=Decimal("1.10000"),
                high=Decimal("1.10500"),
                low=Decimal("1.10200"),
                close=Decimal("1.10300"),
            )

    def test_low_cannot_be_above_close(self):
        with pytest.raises(ValidationError):
            OHLC(
                symbol="EUR/USD",
                timestamp=utc_now(),
                timeframe=Timeframe.M15,
                open=Decimal("1.10000"),
                high=Decimal("1.10500"),
                low=Decimal("1.09800"),
                close=Decimal("1.09700"),
            )

    def test_high_cannot_be_below_low(self):
        with pytest.raises(ValidationError):
            OHLC(
                symbol="EUR/USD",
                timestamp=utc_now(),
                timeframe=Timeframe.M15,
                open=Decimal("1.10000"),
                high=Decimal("1.09900"),
                low=Decimal("1.10100"),
                close=Decimal("1.10000"),
            )

    def test_ohlc_prices_must_be_positive(self):
        with pytest.raises(ValidationError):
            OHLC(
                symbol="EUR/USD",
                timestamp=utc_now(),
                timeframe=Timeframe.M15,
                open=Decimal("-1"),
                high=Decimal("1.1"),
                low=Decimal("1.0"),
                close=Decimal("1.05"),
            )