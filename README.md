# AEGIS AI

**AEGIS AI** is a strict, offline, historical ML-driven research platform for quantitative predictive modeling.

## ⚠️ Mission Critical Safety Boundary
AEGIS AI operates strictly in `PREDICTION_ONLY` mode. It is designed to evaluate historical prediction models, perform backtesting, run walk-forward evaluations, test robustness, govern champion/challenger lifecycles, and monitor prediction drift offline.

**It does NOT:**
- Execute live trades
- Connect to brokers
- Manage real capital
- Serve models to live applications

Any attempt to use this platform for live trading is strictly prohibited and architecturally unsupported by design.

## Architecture

The system is organized into modular layers that strictly separate concerns:

1. **Data & Features**: Ingests historical OHLC data, applies validation, and generates technical features deterministically.
2. **ML Pipeline**: Constructs datasets, applies labelling logic, trains models, performs probability calibration, selects top variants, and merges them into robust ensembles.
3. **Risk Engine**: Evaluates raw model predictions against sizing, maximum drawdown, and exposure limits to generate secure `RiskDecisions`.
4. **Backtest & Walk-Forward**: Evaluates the combined ML-Risk pipeline iteratively across sliding temporal windows, explicitly isolating train/validation data from unseen test periods.
5. **Robustness Evaluator**: Re-runs walk-forward experiments under varied scenarios (e.g. high slippage) without contaminating base historical test results.
6. **Governance & Registry**: Manages model artifacts via a strictly verified, persistent SQLite registry. Reproducibility receipts are generated and hashed to ensure models were created securely.
7. **Champion Health Monitoring**: Profiles selected Champion models and evaluates their health against new out-of-sample data, tracking Data Drift, Prediction Drift, Confidence Drift, and Performance Drift.

## Installation & Running

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the full test suite
python -m pytest tests/ -v

# Run the dry-run command to verify the system boots safely
python main.py --dry-run

# Run the end-to-end Sprint 15 Demonstration
python run_final_aegis_demo.py
```

## Reproducibility & Artifact Integrity

AEGIS models are heavily protected. Every promoted model computes a deterministic SHA-256 fingerprint of its configuration, dataset identities, and training seed. Models loaded from the filesystem are hashed and matched against the secure SQLite registry. Tampered artifacts (e.g. models maliciously modified on disk) are instantly rejected by the load sequence.
