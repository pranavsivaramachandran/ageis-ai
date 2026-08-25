import math
from typing import Dict, Any

def calculate_prediction_drift(reference_preds: Dict[str, float], 
                               observation_preds: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculates deterministic drift in prediction class distributions and confidence.
    """
    classes = ["BUY", "SELL", "NEUTRAL"]
    divergence = 0.0
    valid_classes = 0
    is_corrupt = False
    
    for cls in classes:
        ref_prob = reference_preds.get(f"prob_{cls}")
        obs_prob = observation_preds.get(f"prob_{cls}")
        
        if ref_prob is not None and obs_prob is not None:
             if math.isnan(ref_prob) or math.isnan(obs_prob) or math.isinf(ref_prob) or math.isinf(obs_prob):
                 is_corrupt = True
                 break
             else:
                 divergence += abs(obs_prob - ref_prob)
                 valid_classes += 1
                 
    if not is_corrupt:
        # confidence shift
        ref_conf = reference_preds.get("mean_confidence")
        obs_conf = observation_preds.get("mean_confidence")
        
        conf_shift = None
        if ref_conf is not None and obs_conf is not None:
            if math.isnan(ref_conf) or math.isnan(obs_conf) or math.isinf(ref_conf) or math.isinf(obs_conf):
                is_corrupt = True
            else:
                conf_shift = obs_conf - ref_conf

    if is_corrupt:
        return {
            "status": "CORRUPT",
            "prediction_divergence": None,
            "confidence_shift": None
        }

    if valid_classes > 0:
        divergence = divergence / valid_classes
    else:
        divergence = None
        
    return {
        "status": "VALID",
        "prediction_divergence": divergence,
        "confidence_shift": conf_shift
    }

