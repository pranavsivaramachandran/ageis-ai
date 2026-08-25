import math
from aegis.governance.monitoring.data_drift import calculate_data_drift
from aegis.governance.monitoring.prediction_drift import calculate_prediction_drift
from aegis.governance.monitoring.performance_drift import calculate_performance_drift

def test_data_drift_mean_std_shift():
    ref = {"f1": {"mean": 1.0, "std": 0.5, "missing_rate": 0.0}}
    obs = {"f1": {"mean": 1.5, "std": 1.0, "missing_rate": 0.0}}
    
    drift = calculate_data_drift(ref, obs)
    assert "f1" in drift
    
    d = drift["f1"]
    assert d["status"] == "CALCULATED"
    assert math.isclose(d["mean_shift"], 0.5) # abs(1.5 - 1.0)/1.0 = 0.5
    assert math.isclose(d["std_shift"], 1.0) # abs(1.0 - 0.5)/0.5 = 1.0
    assert d["missingness_delta"] == 0.0

def test_data_drift_nan_inf_safety():
    ref = {"f1": {"mean": 1.0, "std": 0.5}}
    obs = {"f1": {"mean": float('nan'), "std": float('inf')}}
    
    drift = calculate_data_drift(ref, obs)
    d = drift["f1"]
    assert d["mean_shift"] is None
    assert d["std_shift"] is None

def test_data_drift_zero_reference():
    ref = {"f1": {"mean": 0.0, "std": 0.0}}
    obs = {"f1": {"mean": 1.0, "std": 1.0}}
    
    drift = calculate_data_drift(ref, obs, epsilon=1e-8)
    d = drift["f1"]
    # 1.0 / 1e-8 = 1e8
    assert d["mean_shift"] == 1e8
    assert d["std_shift"] == 1e8

def test_prediction_drift_divergence():
    ref = {"prob_BUY": 0.5, "prob_SELL": 0.3, "prob_NEUTRAL": 0.2, "mean_confidence": 0.8}
    obs = {"prob_BUY": 0.6, "prob_SELL": 0.2, "prob_NEUTRAL": 0.2, "mean_confidence": 0.7}
    
    drift = calculate_prediction_drift(ref, obs)
    
    # divergence = (abs(0.6-0.5) + abs(0.2-0.3) + abs(0.2-0.2))/3 = (0.1 + 0.1 + 0.0)/3 = 0.06666
    assert math.isclose(drift["prediction_divergence"], 0.066666, rel_tol=1e-4)
    assert math.isclose(drift["confidence_shift"], -0.1)

def test_prediction_drift_invalid_class():
    ref = {"prob_BUY": 1.0, "prob_SELL": 0.0, "prob_NEUTRAL": 0.0, "mean_confidence": 0.8}
    obs = {"prob_BUY": float('nan'), "prob_SELL": float('inf'), "prob_NEUTRAL": 0.0, "mean_confidence": float('nan')}
    
    drift = calculate_prediction_drift(ref, obs)
    assert drift["status"] == "CORRUPT"
    assert drift["prediction_divergence"] is None
    assert drift["confidence_shift"] is None

def test_prediction_drift_nan_buy():
    ref = {"prob_BUY": 1.0, "prob_SELL": 0.0, "prob_NEUTRAL": 0.0, "mean_confidence": 0.8}
    obs = {"prob_BUY": float('nan'), "prob_SELL": 0.0, "prob_NEUTRAL": 0.0, "mean_confidence": 0.8}
    drift = calculate_prediction_drift(ref, obs)
    assert drift["status"] == "CORRUPT"

def test_prediction_drift_inf_sell():
    ref = {"prob_BUY": 1.0, "prob_SELL": 0.0, "prob_NEUTRAL": 0.0, "mean_confidence": 0.8}
    obs = {"prob_BUY": 1.0, "prob_SELL": float('inf'), "prob_NEUTRAL": 0.0, "mean_confidence": 0.8}
    drift = calculate_prediction_drift(ref, obs)
    assert drift["status"] == "CORRUPT"

def test_prediction_drift_nan_confidence():
    ref = {"prob_BUY": 1.0, "prob_SELL": 0.0, "prob_NEUTRAL": 0.0, "mean_confidence": 0.8}
    obs = {"prob_BUY": 1.0, "prob_SELL": 0.0, "prob_NEUTRAL": 0.0, "mean_confidence": float('nan')}
    drift = calculate_prediction_drift(ref, obs)
    assert drift["status"] == "CORRUPT"
def test_performance_drift():
    ref = {"mean_f1": 0.8, "accuracy": 0.85, "win_rate": 0.6, "max_drawdown": 0.1}
    obs = {"mean_f1": 0.7, "accuracy": 0.80, "win_rate": 0.5, "max_drawdown": 0.2}
    
    drift = calculate_performance_drift(ref, obs)
    
    assert math.isclose(drift["mean_f1_degradation"], 0.1)
    assert math.isclose(drift["accuracy_degradation"], 0.05)
    assert math.isclose(drift["win_rate_degradation"], 0.1)
    assert math.isclose(drift["max_drawdown_increase"], 0.1)

def test_performance_drift_missing():
    ref = {"mean_f1": 0.8}
    obs = {"accuracy": 0.8}
    
    drift = calculate_performance_drift(ref, obs)
    
    assert drift["mean_f1_degradation"] is None
    assert drift["accuracy_degradation"] is None
