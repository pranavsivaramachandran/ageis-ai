import math
from typing import Dict, Any

def calculate_performance_drift(reference_perf: Dict[str, float], 
                                observation_perf: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculates deterministic degradation in performance metrics.
    Metrics: f1, accuracy, max_drawdown, win_rate
    """
    drift_results = {}
    
    metrics = {
        "mean_f1": {"type": "higher_is_better"},
        "accuracy": {"type": "higher_is_better"},
        "win_rate": {"type": "higher_is_better"},
        "max_drawdown": {"type": "lower_is_better"}
    }
    
    for metric, config in metrics.items():
        ref_val = reference_perf.get(metric)
        obs_val = observation_perf.get(metric)
        
        if ref_val is not None and obs_val is not None:
             if not (math.isnan(ref_val) or math.isnan(obs_val) or math.isinf(ref_val) or math.isinf(obs_val)):
                 if config["type"] == "higher_is_better":
                     # degradation is positive when performance drops
                     drift_results[f"{metric}_degradation"] = ref_val - obs_val
                 else:
                     # degradation is positive when value increases
                     drift_results[f"{metric}_increase"] = obs_val - ref_val
             else:
                 if config["type"] == "higher_is_better":
                     drift_results[f"{metric}_degradation"] = None
                 else:
                     drift_results[f"{metric}_increase"] = None
        else:
             if config["type"] == "higher_is_better":
                 drift_results[f"{metric}_degradation"] = None
             else:
                 drift_results[f"{metric}_increase"] = None
                 
    return drift_results
