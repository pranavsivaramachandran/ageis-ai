# AEGIS AI - Known Limitations

This document tracks known limitations and acceptable technical debt within the AEGIS AI codebase, as identified during the extreme deep code audit and hardening passes.

## 1. Feature Representation
- **Floating-point Math**: Technical features (e.g., MACD, RSI, ATR) use standard Python `float` arithmetic for speed during sliding-window processing, rather than `Decimal`. This is an acceptable performance trade-off, but may introduce microscopic floating-point drift over thousands of iterations.
- **Baseline Predictor**: The `BaselinePredictor` schema requires zero features. If a completely broken data feed results in an empty feature vector, the predictor will silently return `NEUTRAL` rather than halting the system.

## 2. In-Memory Registry
- **No Persistence**: The model registry (`aegis.experiments.models.ModelRegistry`) uses an in-memory deterministic retrieval system but has no persistence layer (e.g., database or disk storage). This is adequate for the current scope (Sprint 8) but must be addressed before distributed training or deployment.

## 3. Backtest Metrics
- **Intra-Trade Drawdown vs Final Drawdown**: While intra-candle PnL is now tracked, specific metrics related to maximum favorable excursion (MFE) and maximum adverse excursion (MAE) per trade are not yet explicitly captured for advanced trade analytics.

## 4. Backtest Engine Constraints
- **Single Instrument**: The current `BacktestEngine` is designed to simulate a single instrument (e.g., EUR/USD) per run. Portfolio-level backtesting across multiple correlated instruments is not natively supported without instantiating multiple engines and aggregating manually.
- **Fixed Slippage**: Slippage is calculated as a fixed percentage (`slippage_percent`). It does not model order book depth or variable liquidity conditions dynamically.
