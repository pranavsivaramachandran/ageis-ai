"""
Risk Management Engine for AEGIS AI.

Evaluates predictions against configured risk limits and generates
a deterministic RiskDecision. Does not submit orders.
"""

from decimal import Decimal
from typing import Optional

from aegis.core.config import config
from aegis.prediction.models import PredictionDirection, PredictionResult
from aegis.risk.models import RiskDecision, RiskStatus


class RiskManagementEngine:
    """
    Stateless evaluator of predictions against risk configuration.
    """

    def evaluate_prediction(
        self,
        prediction: PredictionResult,
        current_daily_loss: Decimal = Decimal("0"),
        current_weekly_loss: Decimal = Decimal("0"),
        current_monthly_loss: Decimal = Decimal("0"),
        risk_distance: Optional[Decimal] = None,
    ) -> RiskDecision:
        """
        Evaluate a prediction against all risk parameters.

        Args:
            prediction: The PredictionResult to evaluate.
            current_daily_loss: Accumulated REALIZED loss for the current day.
            current_weekly_loss: Accumulated REALIZED loss for the current week.
            current_monthly_loss: Accumulated REALIZED loss for the current month.
            risk_distance: Price distance to stop-loss (used for position sizing).

        Returns:
            A deterministic RiskDecision.
        """
        
        # 1. Base validation
        if prediction.direction == PredictionDirection.NEUTRAL:
            return self._reject(prediction, "Prediction direction is NEUTRAL.")
            
        # 2. Confidence check
        if prediction.confidence < config.RISK_MIN_CONFIDENCE:
            return self._reject(
                prediction,
                f"Confidence {prediction.confidence} is below minimum {config.RISK_MIN_CONFIDENCE}."
            )
            
        # 3. Loss limits check
        if current_daily_loss >= config.RISK_MAX_DAILY_LOSS:
            return self._reject(prediction, "Maximum daily loss limit reached.")
            
        if current_weekly_loss >= config.RISK_MAX_WEEKLY_LOSS:
            return self._reject(prediction, "Maximum weekly loss limit reached.")
            
        if current_monthly_loss >= config.RISK_MAX_MONTHLY_LOSS:
            return self._reject(prediction, "Maximum monthly loss limit reached.")
            
        # 4. Position sizing
        if risk_distance is None:
            return self._reject(prediction, "risk_distance is required for position sizing.")
            
        if risk_distance <= Decimal("0"):
            return self._reject(prediction, "Risk distance must be strictly positive.")
            
        risk_amount = config.RISK_MAX_RISK_PER_TRADE
            
        # Prevent division by zero mathematically although checked above
        if risk_distance.is_zero():
            return self._reject(prediction, "Risk distance cannot be zero.")
            
        position_size = risk_amount / risk_distance
        
        if position_size > config.RISK_MAX_POSITION_SIZE:
            return self._reject(
                prediction,
                f"Calculated position size {position_size} exceeds maximum {config.RISK_MAX_POSITION_SIZE}."
            )
            
        if position_size <= Decimal("0"):
            return self._reject(prediction, "Calculated position size must be strictly positive.")
                
        return RiskDecision(
            symbol=prediction.symbol,
            timestamp=prediction.timestamp,
            timeframe=prediction.timeframe,
            prediction_direction=prediction.direction,
            confidence=prediction.confidence,
            status=RiskStatus.APPROVED,
            reason="Passes all risk checks.",
            risk_amount=risk_amount,
            position_size=position_size,
        )
        
    def _reject(self, prediction: PredictionResult, reason: str) -> RiskDecision:
        """Helper to construct a REJECTED RiskDecision."""
        return RiskDecision(
            symbol=prediction.symbol,
            timestamp=prediction.timestamp,
            timeframe=prediction.timeframe,
            prediction_direction=prediction.direction,
            confidence=prediction.confidence,
            status=RiskStatus.REJECTED,
            reason=reason,
            risk_amount=None,
            position_size=None,
        )
