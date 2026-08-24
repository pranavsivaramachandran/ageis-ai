"""
Deterministic, provider-independent technical feature functions.

Each function operates on a list of canonical OHLC candles and returns
computed feature values. All functions are:

- Deterministic: same input → same output
- Provider-independent: operates on canonical OHLC only
- Numerically stable: guards against division by zero, insufficient data
- Transparent: documents minimum data length and warm-up periods

Conventions:
- Functions return None when given insufficient data.
- Functions return float values for computed features (standard for
  technical analysis, avoids Decimal performance issues in sliding-window
  math while preserving input precision from OHLC Decimal fields).
- Division by zero produces None, not NaN or Inf.
- No external dependencies beyond stdlib.
- No connection to providers, brokers, or execution systems.

Feature functions are FEATURES, not trading signals or decisions.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Optional

from aegis.interfaces.market_data import OHLC


def _closes(candles: list[OHLC]) -> list[float]:
    """Extract close prices as floats from canonical OHLC candles."""
    return [float(c.close) for c in candles]


def _highs(candles: list[OHLC]) -> list[float]:
    """Extract high prices as floats from canonical OHLC candles."""
    return [float(c.high) for c in candles]


def _lows(candles: list[OHLC]) -> list[float]:
    """Extract low prices as floats from canonical OHLC candles."""
    return [float(c.low) for c in candles]


# ===================================================================
# Simple Returns
# ===================================================================

def simple_returns(candles: list[OHLC]) -> Optional[list[float]]:
    """
    Compute simple returns from close prices.

    Return[i] = (close[i] - close[i-1]) / close[i-1]

    Note: The returned list contains values corresponding to candles[1:].
    The first candle provides the baseline for the first return.
    If 5 candles are provided, 4 returns are computed.

    Minimum data: 2 candles.
    Returns: List of (n-1) return values, or None if insufficient data.
             Individual returns are None if the previous close is zero.
    """
    if len(candles) < 2:
        return None

    closes = _closes(candles)
    returns: list[float] = []

    for i in range(1, len(closes)):
        if closes[i - 1] == 0.0:
            returns.append(None)  # type: ignore[arg-type]
        else:
            returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

    return returns


# ===================================================================
# Simple Moving Average (SMA)
# ===================================================================

def sma(candles: list[OHLC], period: int) -> Optional[float]:
    """
    Compute the Simple Moving Average of close prices.

    SMA = sum(close[n-period:n]) / period

    Minimum data: period candles.
    Warm-up: period candles.

    Args:
        candles: List of canonical OHLC candles.
        period: Number of periods for the average. Must be >= 1.

    Returns:
        The SMA value, or None if insufficient data or invalid period.
    """
    if period < 1 or len(candles) < period:
        return None

    closes = _closes(candles)
    return sum(closes[-period:]) / period


# ===================================================================
# Exponential Moving Average (EMA)
# ===================================================================

def ema(candles: list[OHLC], period: int) -> Optional[float]:
    """
    Compute the Exponential Moving Average of close prices.

    Uses the standard multiplier: k = 2 / (period + 1).
    Initial EMA seed = SMA of the first `period` values.

    Minimum data: period candles.
    Warm-up: period candles (SMA seed), then EMA over remaining.

    Args:
        candles: List of canonical OHLC candles.
        period: Number of periods. Must be >= 1.

    Returns:
        The EMA value, or None if insufficient data or invalid period.
    """
    if period < 1 or len(candles) < period:
        return None

    closes = _closes(candles)
    k = 2.0 / (period + 1)

    # Seed with SMA of first `period` values
    ema_value = sum(closes[:period]) / period

    # Apply EMA over remaining values
    for price in closes[period:]:
        ema_value = price * k + ema_value * (1.0 - k)

    return ema_value


def _ema_series(values: list[float], period: int) -> Optional[list[float]]:
    """
    Internal: compute a full EMA series for use by MACD and others.

    Returns a list of EMA values with the same length as the input.
    The first `period - 1` values are None (warm-up period).

    Returns None if insufficient data.
    """
    if period < 1 or len(values) < period:
        return None

    k = 2.0 / (period + 1)

    result: list[Optional[float]] = [None] * (period - 1)

    # Seed with SMA of first `period` values
    seed = sum(values[:period]) / period
    result.append(seed)

    ema_val = seed
    for price in values[period:]:
        ema_val = price * k + ema_val * (1.0 - k)
        result.append(ema_val)

    return result  # type: ignore[return-value]


# ===================================================================
# RSI (Relative Strength Index)
# ===================================================================

def rsi(candles: list[OHLC], period: int = 14) -> Optional[float]:
    """
    Compute the Relative Strength Index using Wilder's smoothing.

    RSI = 100 - (100 / (1 + RS))
    RS = avg_gain / avg_loss

    Uses Wilder's smoothing method:
    - Initial avg_gain/avg_loss = simple average of first `period` changes
    - Subsequent values smoothed: avg = (prev_avg * (period-1) + current) / period

    Minimum data: period + 1 candles (need `period` price changes).

    Args:
        candles: List of canonical OHLC candles.
        period: RSI period. Must be >= 1. Default 14.

    Returns:
        RSI value (0-100), or None if insufficient data.
        Returns 100.0 if avg_loss is zero (all gains).
        Returns 0.0 if avg_gain is zero (all losses).
    """
    if period < 1 or len(candles) < period + 1:
        return None

    closes = _closes(candles)

    # Calculate price changes
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Initial averages from first `period` changes
    gains = [max(c, 0.0) for c in changes[:period]]
    losses = [abs(min(c, 0.0)) for c in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder's smoothing for remaining changes
    for change in changes[period:]:
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    if avg_gain == 0.0:
        return 0.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ===================================================================
# MACD (Moving Average Convergence Divergence)
# ===================================================================

def macd(
    candles: list[OHLC],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[tuple[float, float, float]]:
    """
    Compute MACD line, signal line, and histogram.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(signal) of MACD Line
    Histogram = MACD Line - Signal Line

    Minimum data: slow + signal - 1 candles.

    Args:
        candles: List of canonical OHLC candles.
        fast: Fast EMA period. Must be >= 1.
        slow: Slow EMA period. Must be >= fast.
        signal: Signal EMA period. Must be >= 1.

    Returns:
        Tuple of (macd_line, signal_line, histogram), or None if
        insufficient data or invalid parameters.
    """
    if fast < 1 or slow < 1 or signal < 1:
        return None
    if fast > slow:
        return None

    min_data = slow + signal - 1
    if len(candles) < min_data:
        return None

    closes = _closes(candles)

    fast_ema = _ema_series(closes, fast)
    slow_ema = _ema_series(closes, slow)

    if fast_ema is None or slow_ema is None:
        return None

    # MACD line: only valid where both EMAs are available
    macd_line_values: list[float] = []
    for i in range(len(closes)):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line_values.append(fast_ema[i] - slow_ema[i])

    if len(macd_line_values) < signal:
        return None

    # Signal line = EMA of MACD line
    signal_ema = _ema_series(macd_line_values, signal)
    if signal_ema is None:
        return None

    # Latest values
    macd_val = macd_line_values[-1]
    signal_val = signal_ema[-1]

    if signal_val is None:
        return None

    histogram = macd_val - signal_val

    return (macd_val, signal_val, histogram)


# ===================================================================
# ATR (Average True Range)
# ===================================================================

def atr(candles: list[OHLC], period: int = 14) -> Optional[float]:
    """
    Compute the Average True Range.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = Wilder's smoothed average of True Range over `period` values.

    Minimum data: period + 1 candles (need `period` true ranges).

    Args:
        candles: List of canonical OHLC candles.
        period: ATR period. Must be >= 1. Default 14.

    Returns:
        ATR value, or None if insufficient data.
    """
    if period < 1 or len(candles) < period + 1:
        return None

    highs = _highs(candles)
    lows = _lows(candles)
    closes = _closes(candles)

    # Calculate true ranges
    true_ranges: list[float] = []
    for i in range(1, len(candles)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    # Initial ATR = simple average of first `period` true ranges
    atr_value = sum(true_ranges[:period]) / period

    # Wilder's smoothing for remaining
    for tr in true_ranges[period:]:
        atr_value = (atr_value * (period - 1) + tr) / period

    return atr_value


# ===================================================================
# Bollinger Bands
# ===================================================================

def bollinger_bands(
    candles: list[OHLC],
    period: int = 20,
    num_std: float = 2.0,
) -> Optional[tuple[float, float, float]]:
    """
    Compute Bollinger Bands.

    Middle = SMA(period)
    Upper = Middle + num_std * std_dev
    Lower = Middle - num_std * std_dev

    Minimum data: period candles.

    Args:
        candles: List of canonical OHLC candles.
        period: SMA period. Must be >= 1. Default 20.
        num_std: Number of standard deviations. Default 2.0.

    Returns:
        Tuple of (upper, middle, lower), or None if insufficient data.
        If standard deviation is zero (all same prices), upper == middle == lower.
    """
    if period < 1 or len(candles) < period:
        return None

    closes = _closes(candles)
    window = closes[-period:]

    middle = sum(window) / period

    # Population standard deviation
    variance = sum((x - middle) ** 2 for x in window) / period
    std_dev = math.sqrt(variance)

    upper = middle + num_std * std_dev
    lower = middle - num_std * std_dev

    return (upper, middle, lower)


# ===================================================================
# Momentum
# ===================================================================

def momentum(candles: list[OHLC], period: int = 10) -> Optional[float]:
    """
    Compute price momentum.

    Momentum = close[current] - close[current - period]

    Minimum data: period + 1 candles.

    Args:
        candles: List of canonical OHLC candles.
        period: Lookback period. Must be >= 1. Default 10.

    Returns:
        Momentum value, or None if insufficient data.
    """
    if period < 1 or len(candles) < period + 1:
        return None

    closes = _closes(candles)
    return closes[-1] - closes[-1 - period]


# ===================================================================
# Rolling Volatility
# ===================================================================

def rolling_volatility(
    candles: list[OHLC],
    period: int = 20,
) -> Optional[float]:
    """
    Compute rolling volatility as the standard deviation of simple returns.

    Volatility = std_dev(returns[-period:])

    Minimum data: period + 1 candles (need `period` returns).

    Args:
        candles: List of canonical OHLC candles.
        period: Number of returns to use. Must be >= 1. Default 20.

    Returns:
        Volatility value, or None if insufficient data.
        Returns 0.0 if all returns are identical (zero variance).
        Returns None if any return cannot be computed (zero close price).
    """
    if period < 1 or len(candles) < period + 1:
        return None

    closes = _closes(candles)

    # Compute the last `period` returns
    start_idx = len(closes) - period - 1
    returns: list[float] = []
    for i in range(start_idx + 1, len(closes)):
        if closes[i - 1] == 0.0:
            return None
        returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

    if len(returns) < period:
        return None

    mean_return = sum(returns) / len(returns)

    # Population standard deviation
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)
