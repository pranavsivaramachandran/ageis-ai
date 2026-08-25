"""
AEGIS AI - Sprint 12 Demonstration
Model Governance, Experiment Tracking, and Model Promotion
"""

import sys
import logging
from datetime import datetime, timezone

from aegis.governance.models import (
    ModelIdentity, ExperimentRecord, PromotionPolicy, ExperimentStatus, GovernanceStatus
)
from aegis.governance.evaluator import GovernanceEvaluator
from aegis.prediction.registry import ModelRegistry
from aegis.prediction.model_interface import PredictionModel, FeatureSchema
from aegis.core.config import ExecutionMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sprint_12_demo")

class DummyMLModel(PredictionModel):
    def __init__(self, model_id, version):
        self._model_id = model_id
        self._version = version
    @property
    def model_id(self): return self._model_id
    @property
    def version(self): return int(self._version)
    @property
    def schema(self): return FeatureSchema()
    def is_ready(self): return True
    def predict(self, fv): return None


def run_demo():
    logger.info("Starting AEGIS AI Sprint 12 Governance Demonstration")
    logger.info(f"Current Execution Mode: {ExecutionMode.PREDICTION_ONLY.name}")
    assert ExecutionMode.PREDICTION_ONLY.name == "PREDICTION_ONLY"

    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    
    policy = PromotionPolicy(
        policy_id="strict_research_v1",
        min_walk_forward_windows=4,
        min_mean_f1=0.55,
        max_drawdown_pct=0.15,
        requires_baseline_beat=True
    )
    logger.info(f"Active Promotion Policy: {policy.policy_id} (min_f1={policy.min_mean_f1}, max_dd={policy.max_drawdown_pct})")

    # 1. Register candidate A (Fails)
    candidate_a = DummyMLModel("rf_classifier_a", 1)
    registry.register(candidate_a)
    logger.info(f"Registered Candidate A: {candidate_a.model_id}-v{candidate_a.version}")
    
    exp_a = ExperimentRecord(
        experiment_id="exp-a", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=[candidate_a.model_id],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={
            "walk_forward_windows": 5, "mean_f1": 0.50, # Fails F1 requirement
            "max_drawdown": 0.10, "beats_baseline": True
        }
    )
    
    logger.info("Evaluating Candidate A...")
    decision_a = evaluator.evaluate_candidate(exp_a, policy, {}, candidate_a.version, is_reproducible=True, is_safe=True)
    logger.info(f"Decision for Candidate A: {decision_a.decision.name} - Reason: {decision_a.reason}")
    registry.reject(candidate_a.model_id, candidate_a.version, decision_a)
    
    # 2. Register candidate B (Passes)
    candidate_b = DummyMLModel("rf_classifier_b", 1)
    registry.register(candidate_b)
    logger.info(f"Registered Candidate B: {candidate_b.model_id}-v{candidate_b.version}")
    
    exp_b = ExperimentRecord(
        experiment_id="exp-b", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=[candidate_b.model_id],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={
            "walk_forward_windows": 5, "mean_f1": 0.62,
            "max_drawdown": 0.12, "beats_baseline": True
        }
    )
    
    logger.info("Evaluating Candidate B...")
    decision_b = evaluator.evaluate_candidate(exp_b, policy, {}, candidate_b.version, is_reproducible=True, is_safe=True)
    logger.info(f"Decision for Candidate B: {decision_b.decision.name}")
    
    registry.promote(candidate_b.model_id, candidate_b.version, decision_b)
    champion = registry.get_champion()
    logger.info(f"New Champion Promoted: {champion.model_id}-v{champion.version}")
    
    # 3. Register challenger C (Passes, supersedes B)
    candidate_c = DummyMLModel("xgb_classifier_c", 1)
    registry.register(candidate_c, GovernanceStatus.CHALLENGER)
    logger.info(f"Registered Challenger C: {candidate_c.model_id}-v{candidate_c.version}")
    
    exp_c = ExperimentRecord(
        experiment_id="exp-c", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=[candidate_c.model_id],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={
            "walk_forward_windows": 6, "mean_f1": 0.68,
            "max_drawdown": 0.08, "beats_baseline": True
        }
    )
    
    logger.info("Evaluating Challenger C against Policy...")
    decision_c = evaluator.evaluate_candidate(exp_c, policy, {}, candidate_c.version, is_reproducible=True, is_safe=True)
    logger.info(f"Decision for Challenger C: {decision_c.decision.name}")
    
    registry.promote(candidate_c.model_id, candidate_c.version, decision_c)
    
    # 4. Show audit / registry state
    logger.info("--- Registry State ---")
    for full_id in registry.list_models():
        model_id, version_str = full_id.split("-v")
        status = registry.get_status(model_id, int(version_str))
        logger.info(f"Model: {full_id} | Status: {status.name}")
        
    logger.info("Safety Check: Execution Mode is strictly PREDICTION_ONLY.")
    logger.info("Demo complete.")

if __name__ == "__main__":
    run_demo()
