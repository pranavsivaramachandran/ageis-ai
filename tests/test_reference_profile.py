from datetime import datetime, timezone
import pytest
from aegis.governance.monitoring.models import ReferenceProfile, MonitoringWindow

def test_reference_profile_identity_deterministic():
    stats = {
        "feature1": {"mean": 1.0, "std": 0.5, "missing_rate": 0.0}
    }
    pred_stats = {"prob_BUY": 0.3, "prob_SELL": 0.3, "prob_NEUTRAL": 0.4}
    
    prof1 = ReferenceProfile(
        champion_identity="champ1",
        champion_version=1,
        experiment_identity="exp1",
        feature_schema_identity="schema1",
        feature_config_identity="fc1",
        target_identity="t1",
        reference_window_identity="w1",
        feature_statistics=stats,
        prediction_statistics=pred_stats
    )
    
    prof2 = ReferenceProfile(
        champion_identity="champ1",
        champion_version=1,
        experiment_identity="exp1",
        feature_schema_identity="schema1",
        feature_config_identity="fc1",
        target_identity="t1",
        reference_window_identity="w1",
        feature_statistics=stats,
        prediction_statistics=pred_stats
    )
    
    assert prof1.identity == prof2.identity
    assert prof1.created_at != prof2.created_at # likely different due to tick
    
def test_reference_profile_isolation():
    stats1 = {"feature1": {"mean": 1.0}}
    prof1 = ReferenceProfile(
        champion_identity="champ1",
        champion_version=1,
        experiment_identity="exp1",
        feature_schema_identity="schema1",
        feature_config_identity="fc1",
        target_identity="t1",
        reference_window_identity="w1",
        feature_statistics=stats1
    )
    
    # modify stats dictionary -> shouldn't affect hash if passed carefully
    # actually Pydantic 2 copies on assign or deep validation, but anyway, ReferenceProfile is frozen
    with pytest.raises(Exception): # pydantic ValidationError for frozen
        prof1.feature_statistics = {"feature1": {"mean": 2.0}}
