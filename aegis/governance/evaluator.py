"""
Governance evaluation logic for AEGIS AI.
"""

from datetime import datetime, timezone
from aegis.governance.models import (
    ExperimentRecord, PromotionPolicy, PromotionDecision, PromotionDecisionType, ChampionHealth, ExperimentStatus
)
from aegis.governance.reproducibility import ReproducibilityReceipt, VerificationStatus
from aegis.prediction.registry import ModelRegistry

class GovernanceEvaluator:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def evaluate_candidate(self, experiment: ExperimentRecord, policy: PromotionPolicy, evidence: dict, model_version: int, reproducibility_receipt: ReproducibilityReceipt, is_safe: bool) -> PromotionDecision:
        candidate_id = experiment.candidate_model_ids[0] if experiment.candidate_model_ids else "unknown"
        
        # Check explicit gates
        if experiment.status == ExperimentStatus.FAILED:
            return self._reject(candidate_id, model_version, policy, "Experiment has FAILED status", evidence)
            
        if not is_safe and policy.require_safety_check:
            return self._reject(candidate_id, model_version, policy, "Safety check failed", evidence)
            
        if reproducibility_receipt.status != VerificationStatus.VERIFIED and policy.require_reproducibility:
            return self._reject(candidate_id, model_version, policy, f"Reproducibility check failed. Status: {reproducibility_receipt.status.value}", evidence)

        # Evaluate metrics
        metrics = experiment.overall_metrics
        if "walk_forward_windows" in metrics and metrics["walk_forward_windows"] < policy.min_walk_forward_windows:
            return self._reject(candidate_id, model_version, policy, f"Insufficient walk-forward windows", evidence)

        if "mean_f1" in metrics and metrics["mean_f1"] < policy.min_mean_f1:
            return self._reject(candidate_id, model_version, policy, f"Mean F1 below required", evidence)

        if "max_drawdown" in metrics and metrics["max_drawdown"] > policy.max_drawdown_pct:
            return self._reject(candidate_id, model_version, policy, f"Max drawdown exceeds allowed", evidence)

        if policy.requires_baseline_beat and "beats_baseline" in metrics and not metrics["beats_baseline"]:
             return self._reject(candidate_id, model_version, policy, "Model did not beat baseline", evidence)

        # Check existing champion
        champion = self.registry.get_champion()
        prev_champ_id = f"{champion.model_id}-v{champion.version}" if champion else None
        
        return PromotionDecision(
            model_identity=candidate_id,
            model_version=model_version,
            decision=PromotionDecisionType.PROMOTE,
            reason="Candidate satisfies all promotion policy criteria.",
            policy_identity=policy.identity,
            evidence_summary=evidence,
            timestamp=datetime.now(timezone.utc),
            previous_champion=prev_champ_id,
            new_champion=f"{candidate_id}-v{model_version}"
        )

    def _reject(self, candidate_id: str, version: int, policy: PromotionPolicy, reason: str, evidence: dict) -> PromotionDecision:
        return PromotionDecision(
            model_identity=candidate_id,
            model_version=version,
            decision=PromotionDecisionType.REJECT,
            reason=reason,
            policy_identity=policy.identity,
            evidence_summary=evidence,
            timestamp=datetime.now(timezone.utc),
            previous_champion=None,
            new_champion=None
        )

