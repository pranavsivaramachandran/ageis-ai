"""
Feature builder and feature vector for AEGIS AI.

Orchestrates computation of technical features from canonical OHLC data
and packages results into a typed FeatureVector.

The FeatureVector is a data carrier for computed features — it is NOT
a trading signal, decision, or recommendation by itself.

This module does not:
- Connect to providers or brokers
- Submit orders or make trading decisions
- Bypass the event architecture
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import hashlib

from pydantic import BaseModel, Field

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


@dataclass(frozen=True)
class FeatureVector:
    """
    Typed container for computed technical features.

    All feature values are Optional[float] (or Optional[list] for returns).
    A None value indicates the feature could not be computed due to
    insufficient data or numerical impossibility.

    This is a data carrier — not a trading signal.

    Attributes:
        timestamp: UTC timestamp of the most recent candle.
        symbol: Instrument symbol.
        timeframe: Candle timeframe.
        last_close: Closing price of the most recent candle.
        returns: List of simple returns, or None.
        sma_value: Simple Moving Average, or None.
        ema_value: Exponential Moving Average, or None.
        rsi_value: Relative Strength Index (0-100), or None.
        macd_line: MACD line value, or None.
        macd_signal: MACD signal line value, or None.
        macd_histogram: MACD histogram value, or None.
        atr_value: Average True Range, or None.
        bollinger_upper: Bollinger upper band, or None.
        bollinger_middle: Bollinger middle band (SMA), or None.
        bollinger_lower: Bollinger lower band, or None.
        momentum_value: Price momentum, or None.
        volatility: Rolling volatility, or None.
    """

    timestamp: datetime
    symbol: str
    timeframe: Timeframe
    last_close: float

    # Feature values
    returns: Optional[list[float]] = None
    sma_value: Optional[float] = None
    ema_value: Optional[float] = None
    rsi_value: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr_value: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    momentum_value: Optional[float] = None
    volatility: Optional[float] = None

class FeatureBuilderConfig(BaseModel):
    """Configuration for FeatureBuilder parameters."""
    sma_period: int = Field(default=20, gt=0)
    ema_period: int = Field(default=20, gt=0)
    rsi_period: int = Field(default=14, gt=0)
    macd_fast: int = Field(default=12, gt=0)
    macd_slow: int = Field(default=26, gt=0)
    macd_signal: int = Field(default=9, gt=0)
    atr_period: int = Field(default=14, gt=0)
    bollinger_period: int = Field(default=20, gt=0)
    bollinger_std: float = Field(default=2.0, gt=0)
    momentum_period: int = Field(default=10, gt=0)
    volatility_period: int = Field(default=20, gt=0)

    model_config = {"frozen": True}

    @property
    def identity(self) -> str:
        """Deterministic identity for this feature builder configuration."""
        content = f"{self.sma_period}|{self.ema_period}|{self.rsi_period}|{self.macd_fast}|{self.macd_slow}|{self.macd_signal}|{self.atr_period}|{self.bollinger_period}|{self.bollinger_std}|{self.momentum_period}|{self.volatility_period}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


class FeatureBuilder:
    """
    Builds a FeatureVector from a list of canonical OHLC candles.

    Orchestrates individual feature functions and captures their
    results into a typed FeatureVector. Each feature computation
    is independent — a failure in one does not prevent others from
    being computed.

    Default periods:
        SMA: 20, EMA: 20, RSI: 14, MACD: 12/26/9,
        ATR: 14, Bollinger: 20/2.0, Momentum: 10, Volatility: 20

    Usage:
        builder = FeatureBuilder()
        vector = builder.build(candles)
    """

    def __init__(
        self,
        config: Optional[FeatureBuilderConfig] = None,
        **kwargs
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = FeatureBuilderConfig(**kwargs)

        self.sma_period = self.config.sma_period
        self.ema_period = self.config.ema_period
        self.rsi_period = self.config.rsi_period
        self.macd_fast = self.config.macd_fast
        self.macd_slow = self.config.macd_slow
        self.macd_signal = self.config.macd_signal
        self.atr_period = self.config.atr_period
        self.bollinger_period = self.config.bollinger_period
        self.bollinger_std = self.config.bollinger_std
        self.momentum_period = self.config.momentum_period
        self.volatility_period = self.config.volatility_period

    @property
    def minimum_candles(self) -> int:
        """
        Minimum number of candles required to compute at least one feature.

        Returns 2 (minimum for simple returns).
        """
        return 2

    @property
    def recommended_candles(self) -> int:
        """
        Recommended number of candles to compute all features.

        Based on the most data-hungry feature (MACD: slow + signal - 1).
        """
        return max(
            self.sma_period,
            self.ema_period,
            self.rsi_period + 1,
            self.macd_slow + self.macd_signal - 1,
            self.atr_period + 1,
            self.bollinger_period,
            self.momentum_period + 1,
            self.volatility_period + 1,
        )

    def build(self, candles: list[OHLC]) -> Optional[FeatureVector]:
        """
        Build a FeatureVector from canonical OHLC candles.

        Args:
            candles: List of canonical OHLC candles, ordered oldest first.
                     Must contain at least 2 candles (minimum for returns).
                     All candles must share the same symbol and timeframe.

        Returns:
            A FeatureVector with all computable features populated,
            or None if fewer than 2 candles are provided.

        Raises:
            ValueError: If candles have mixed symbols or timeframes.
        """
        if len(candles) < self.minimum_candles:
            return None

        # Validate consistent symbol and timeframe
        symbol = candles[-1].symbol
        timeframe = candles[-1].timeframe
        timestamp = candles[-1].timestamp
        last_close = float(candles[-1].close)

        for c in candles:
            if c.symbol != symbol:
                raise ValueError(
                    f"Mixed symbols in candle list: "
                    f"'{c.symbol}' vs '{symbol}'"
                )
            if c.timeframe != timeframe:
                raise ValueError(
                    f"Mixed timeframes in candle list: "
                    f"'{c.timeframe}' vs '{timeframe}'"
                )

        # Compute features independently
        returns_val = simple_returns(candles)
        sma_val = sma(candles, self.sma_period)
        ema_val = ema(candles, self.ema_period)
        rsi_val = rsi(candles, self.rsi_period)

        macd_result = macd(
            candles, self.macd_fast, self.macd_slow, self.macd_signal
        )
        macd_l = macd_result[0] if macd_result else None
        macd_s = macd_result[1] if macd_result else None
        macd_h = macd_result[2] if macd_result else None

        atr_val = atr(candles, self.atr_period)

        bb_result = bollinger_bands(
            candles, self.bollinger_period, self.bollinger_std
        )
        bb_upper = bb_result[0] if bb_result else None
        bb_middle = bb_result[1] if bb_result else None
        bb_lower = bb_result[2] if bb_result else None

        mom_val = momentum(candles, self.momentum_period)
        vol_val = rolling_volatility(candles, self.volatility_period)

        return FeatureVector(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            last_close=last_close,
            returns=returns_val,
            sma_value=sma_val,
            ema_value=ema_val,
            rsi_value=rsi_val,
            macd_line=macd_l,
            macd_signal=macd_s,
            macd_histogram=macd_h,
            atr_value=atr_val,
            bollinger_upper=bb_upper,
            bollinger_middle=bb_middle,
            bollinger_lower=bb_lower,
            momentum_value=mom_val,
            volatility=vol_val,
        )