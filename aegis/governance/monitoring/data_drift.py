import math
from typing import Dict, Any

def calculate_data_drift(reference_stats: Dict[str, Dict[str, float]], 
                         observation_stats: Dict[str, Dict[str, float]], 
                         epsilon: float = 1e-8) -> Dict[str, Any]:
    """
    Deterministically calculates data drift for features between reference and observation.
    Returns a dictionary of drift metrics per feature.
    """
    drift_results = {}
    
    for feature_name, ref_stat in reference_stats.items():
        if feature_name not in observation_stats:
            drift_results[feature_name] = {"status": "MISSING_IN_OBSERVATION"}
            continue
            
        obs_stat = observation_stats[feature_name]
        
        # Calculate mean shift
        ref_mean = ref_stat.get("mean")
        obs_mean = obs_stat.get("mean")
        
        mean_shift = None
        if ref_mean is not None and obs_mean is not None:
            if not (math.isnan(ref_mean) or math.isnan(obs_mean) or math.isinf(ref_mean) or math.isinf(obs_mean)):
                mean_shift = abs(obs_mean - ref_mean) / max(abs(ref_mean), epsilon)
            
        # Calculate std shift
        ref_std = ref_stat.get("std")
        obs_std = obs_stat.get("std")
        
        std_shift = None
        if ref_std is not None and obs_std is not None:
            if not (math.isnan(ref_std) or math.isnan(obs_std) or math.isinf(ref_std) or math.isinf(obs_std)):
                std_shift = abs(obs_std - ref_std) / max(abs(ref_std), epsilon)
            
        # Calculate missingness delta
        ref_missing = ref_stat.get("missing_rate", 0.0)
        obs_missing = obs_stat.get("missing_rate", 0.0)
        
        missingness_delta = obs_missing - ref_missing
        if math.isnan(missingness_delta) or math.isinf(missingness_delta):
            missingness_delta = None
        
        drift_results[feature_name] = {
            "mean_shift": mean_shift,
            "std_shift": std_shift,
            "missingness_delta": missingness_delta,
            "status": "CALCULATED"
        }
        
    return drift_results
