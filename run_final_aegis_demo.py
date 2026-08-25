"""
AEGIS AI - Sprint 15 Final Demonstration
End-to-end integration proving offline, reproducible research capabilities.
"""

import os
import shutil
import tempfile
import hashlib
import joblib
import logging
import time
from decimal import Decimal
from datetime import datetime, timezone, timedelta

from aegis.core.config import ExecutionMode
from aegis.interfaces.market_data import Timeframe, OHLC

from aegis.features.builder import FeatureBuilder, FeatureBuilderConfig
from aegis.ml.labels import TargetConfig, TargetGenerator
from aegis.prediction.model_interface import FeatureSchema, PredictionModel
from aegis.ml.training import TrainerConfig
from aegis.risk.engine import RiskManagementEngine
from aegis.backtest.models import SimulationConfig
from aegis.evaluation.models import WalkForwardConfig, WalkForwardStrategy
from aegis.evaluation.walk_forward import WalkForwardEvaluator
from aegis.evaluation.robustness import RobustnessEvaluator, RobustnessScenario

from aegis.db import session as db_session_module
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aegis.governance.models import ArtifactMetadata, GovernanceStatus, PromotionDecision, PromotionDecisionType
from aegis.governance.reproducibility import ReproducibilityVerifier
from aegis.prediction.registry import ModelRegistry

from aegis.governance.monitoring.models import ReferenceProfile, MonitoringWindow, MonitoringPolicy
from aegis.governance.monitoring.engine import MonitoringEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sprint_15_final_demo")

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

def compute_hash(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for b in iter(lambda: f.read(4096), b""):
            sha.update(b)
    return sha.hexdigest()

def generate_dummy_history():
    history = []
    base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    val = 100.0
    for i in range(500):
        if i % 10 < 5:
            val *= 1.01
        else:
            val *= 0.99
        history.append(OHLC(
            symbol="BTC/USD", timestamp=base_time + timedelta(hours=i),
            timeframe=Timeframe.H1, open=Decimal(str(val)),
            high=Decimal(str(val * 1.02)), low=Decimal(str(val * 0.98)),
            close=Decimal(str(val)), volume=Decimal("1000")
        ))
    return history

def run_demo():
    logger.info("=" * 70)
    logger.info("AEGIS AI SPRINT 15 FINAL DEMONSTRATION")
    logger.info("=" * 70)
    logger.info(f"Execution Mode: {ExecutionMode.PREDICTION_ONLY.name}")
    
    fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    test_engine = create_engine(f"sqlite:///{temp_db_path}")
    db_session_module.engine = test_engine
    db_session_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db_session_module.init_db()
    
    artifact_dir = tempfile.mkdtemp(prefix="aegis_artifacts_")
    registry = ModelRegistry(artifact_dir=artifact_dir)
    
    logger.info("\n[1] Historical Data Loading")
    history = generate_dummy_history()
    logger.info(f"Loaded {len(history)} deterministic historical candles.")
    
    logger.info("\n[2-13] End-to-End Walk-Forward Evaluation")
    fb_config = FeatureBuilderConfig(
        sma_period=10, ema_period=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9,
        atr_period=14, bollinger_period=20, momentum_period=10, volatility_period=10
    )
    feature_builder = FeatureBuilder(fb_config)
    target_generator = TargetGenerator(TargetConfig(target_horizon_candles=5, threshold=0.005))
    schema = FeatureSchema(schema_version=1, required_features=["sma_value"])
    trainer_config = TrainerConfig(random_state=42)
    risk_engine = RiskManagementEngine()
    sim_config = SimulationConfig(initial_capital=Decimal("10000.0"), commission_per_trade=Decimal("0.0"), slippage_percent=Decimal("0.0"))
    
    wf_evaluator = WalkForwardEvaluator(feature_builder, target_generator, schema, trainer_config, risk_engine, sim_config)
    wf_config = WalkForwardConfig(strategy=WalkForwardStrategy.ROLLING, train_size=100, validation_size=20, test_size=50, step_size=50, minimum_train_samples=50)
    
    wf_report = wf_evaluator.evaluate(history, wf_config, "final_demo_wf")
    logger.info(f"Walk-Forward Complete. Mean ML PnL: {wf_report.mean_ml_pnl}")
    
    rb_evaluator = RobustnessEvaluator(base_evaluator=wf_evaluator)
    scenarios = [RobustnessScenario(scenario_name="High_Slippage", slippage_percent=Decimal("0.005"))]
    rb_report = rb_evaluator.evaluate(history, wf_config, "final_demo_rb", scenarios)
    logger.info(f"Robustness Complete. Scenarios checked: {len(rb_report.scenarios)}")
    
    logger.info("\n[14-16] Governance & Champion Registration")
    model_a = DummyMLModel("final_rf", 1)
    artifact_path = os.path.join(artifact_dir, "final_rf-v1.joblib")
    joblib.dump(model_a, artifact_path)
    hsh = compute_hash(artifact_path)
    
    meta = ArtifactMetadata(
        artifact_format="joblib", model_identity="final_rf", version=1,
        fingerprint="fingerprint_123", training_experiment="final_demo_wf",
        feature_schema_version=1, dataset_identity="ds_demo",
        training_date=datetime.now(timezone.utc), random_seed=42, integrity_hash=hsh
    )
    registry.register(model_a, GovernanceStatus.CANDIDATE, metadata=meta)
    
    registry.promote("final_rf", 1, PromotionDecision(
        model_identity="final_rf", model_version=1, decision=PromotionDecisionType.PROMOTE,
        reason="Passed all offline tests", policy_identity="strict_v1"
    ))
    champ = registry.get_champion()
    logger.info(f"Promoted CHAMPION: {champ.model_id}-v{champ.version}")
    
    logger.info("\n[17-19] Monitoring, Health & Drift")
    reference = ReferenceProfile(
        champion_identity="final_rf", champion_version=1, experiment_identity="exp",
        feature_schema_identity="f", feature_config_identity="fc", target_identity="t",
        reference_window_identity="ref", feature_statistics={}, prediction_statistics={}, performance_statistics={}
    )
    obs = MonitoringWindow(
        start_time=datetime.now(timezone.utc), end_time=datetime.now(timezone.utc),
        sample_count=100, labeled_sample_count=100, champion_identity="final_rf", champion_version=1,
        observation_fingerprint="obs", feature_statistics={}, prediction_statistics={}, performance_statistics={}
    )
    policy = MonitoringPolicy(policy_id="p1")
    health = MonitoringEngine.assess_health(reference, obs, policy)
    logger.info(f"Champion Health State: {health.state.name}")
    
    logger.info("\n[20-21] Persistence & Restart Test")
    logger.info("Simulating restart (new registry instance)...")
    registry_restarted = ModelRegistry(artifact_dir=artifact_dir)
    restored_champ = registry_restarted.get_champion()
    logger.info(f"Restored CHAMPION after restart: {restored_champ.model_id}-v{restored_champ.version}")
    
    logger.info("\n[22] Final Safety Demonstration: Intentional Failure Path")
    logger.info("Corrupting artifact on disk...")
    with open(artifact_path, "ab") as f:
        f.write(b"corrupt")
    
    try:
        registry_restarted.get_champion()
        logger.error("VULNERABILITY: Tampered artifact was loaded!")
    except RuntimeError as e:
        logger.info(f"Safety Gate Active: Tampered artifact successfully blocked. Reason: {e}")
        
    logger.info("\n[23] Final Cleanup")
    shutil.rmtree(artifact_dir)
    test_engine.dispose()
    os.remove(temp_db_path)
    logger.info("Demo complete.")

if __name__ == "__main__":
    run_demo()
