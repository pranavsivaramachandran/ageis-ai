"""
AEGIS AI feature engineering package.

Provides deterministic, provider-independent technical feature computation
from canonical OHLC data.

Public API:
    FeatureVector, FeatureBuilder — feature orchestration
    sma, ema, rsi, macd, atr, bollinger_bands, momentum,
    rolling_volatility, simple_returns — individual feature functions
"""

from aegis.features.builder import FeatureBuilder, FeatureVector
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

__all__ = [
    "FeatureVector",
    "FeatureBuilder",
    "sma",
    "ema",
    "rsi",
    "macd",
    "atr",
    "bollinger_bands",
    "momentum",
    "rolling_volatility",
    "simple_returns",
]
