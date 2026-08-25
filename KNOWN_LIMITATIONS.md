# AEGIS AI - Known Limitations

This document tracks known limitations and acceptable technical debt within the AEGIS AI codebase, as identified during the final Sprint 15 audit.

## 1. Feature Representation
- **Floating-point Math**: Technical features (e.g., MACD, RSI, ATR) use standard Python `float` arithmetic for speed during sliding-window processing, rather than `Decimal`. This is an acceptable performance trade-off for research, but may introduce microscopic floating-point drift over thousands of iterations compared to institutional pricing engines.
- **Baseline Predictor**: The `BaselinePredictor` schema requires zero features. If a completely broken data feed results in an empty feature vector, the predictor will silently return `NEUTRAL` rather than halting the system.

## 2. Backtest Metrics
- **Intra-Trade Drawdown vs Final Drawdown**: While intra-candle PnL is now tracked, specific metrics related to maximum favorable excursion (MFE) and maximum adverse excursion (MAE) per trade are not yet explicitly captured for advanced trade analytics.

## 3. Backtest Engine Constraints
- **Single Instrument**: The current `BacktestEngine` is designed to simulate a single instrument (e.g., EUR/USD) per run. Portfolio-level backtesting across multiple correlated instruments is not natively supported without instantiating multiple engines and aggregating manually.
- **Fixed Slippage**: Slippage is calculated as a fixed percentage (`slippage_percent`). It does not model order book depth or variable liquidity conditions dynamically.

## 4. Execution Restrictions
- **No Live Trading**: AEGIS AI is strictly an **offline, historical prediction research platform**. The core `ExecutionMode` is hardcoded to `PREDICTION_ONLY`. There are no broker connections, no API key handlers for live execution, and no infrastructure for streaming real-time ticks.
- **Model Promotion ≠ Live Deployment**: While the system fully supports a "Champion/Challenger" model registry with robust health monitoring and alert generation, promoting a Champion simply marks it as the default model for downstream offline analytics. It does **not** trigger any real-world deployments.
