"""
AEGIS AI — Baseline Development Predictor (Prediction Engine).

=================================================================
THIS IS A BASELINE / DEVELOPMENT PREDICTOR.
IT IS NOT A FINAL AI MODEL.
IT DOES NOT CLAIM PROFITABILITY.
IT MUST BE REPLACED BY A TRAINED ML MODEL IN A FUTURE SPRINT.
=================================================================

Architecture:

    FeatureVector  →  BaselinePredictor  →  PredictionResult

The engine:
- Consumes a FeatureVector (from aegis.features.builder)
- Returns a PredictionResult (from aegis.prediction.models)
- Is deterministic: identical input → identical output
- Uses ONLY the supplied FeatureVector — no external data access

The engine does NOT:
- Access a broker, market data provider, or external API
- Submit, modify, or cancel orders
- Mutate any global state
- Access future prices or historical data beyond the FeatureVector
- Claim to be a profitable trading strategy

Evidence-Scoring Design
-----------------------
Each available feature contributes a directional vote in [-1.0, +1.0].
Only features that are not None participate. The final score is the
weighted average of all available votes.

Direction:
    score >  +threshold  →  BUY
    score <  -threshold  →  SELL
    otherwise            →  NEUTRAL

Confidence:
    abs(score) mapped to [0.0, 1.0], clamped at boundaries.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from aegis.features.builder import FeatureVector
from aegis.prediction.models import PredictionDirection, PredictionResult
from aegis.prediction.model_interface import PredictionModel, FeatureSchema


class PredictionError(Exception):
    """Raised when the prediction engine encounters invalid inputs."""
    pass


# ===================================================================
# Evidence scoring configuration
# ===================================================================

# Direction threshold: score must exceed this to be non-NEUTRAL
_DIRECTION_THRESHOLD = 0.15

# Feature weights (relative importance — normalized at scoring time)
_WEIGHTS = {
    "rsi": 1.5,
    "macd": 1.5,
    "sma": 1.0,
    "ema": 1.0,
    "momentum": 1.0,
    "bollinger": 1.2,
}


class BaselinePredictor(PredictionModel):
    """
    Deterministic evidence-scoring baseline predictor.

    BASELINE / DEVELOPMENT PREDICTOR — NOT A FINAL AI MODEL.

    Consumes a FeatureVector and produces a PredictionResult by
    scoring multiple technical features independently and combining
    their directional votes into a single prediction.

    Usage:
        predictor = BaselinePredictor()
        result = predictor.predict(feature_vector)
    """

    def __init__(
        self,
        direction_threshold: float = _DIRECTION_THRESHOLD,
    ) -> None:
        """
        Args:
            direction_threshold: Minimum absolute score to emit BUY/SELL.
                                 Below this, NEUTRAL is produced.
        """
        if not (0.0 < direction_threshold < 1.0):
            raise PredictionError(
                f"direction_threshold must be in (0, 1), got {direction_threshold}"
            )
        self._threshold = direction_threshold

    @property
    def model_id(self) -> str:
        return "baseline"
        
    @property
    def version(self) -> int:
        return 1
        
    @property
    def schema(self) -> FeatureSchema:
        return FeatureSchema(
            schema_version=1,
            required_features=[]
        )
        
    def is_ready(self) -> bool:
        return True

    def predict(self, fv: FeatureVector) -> PredictionResult:
        """
        Produce a deterministic prediction from a FeatureVector.
        
        Before predicting, the FeatureVector is validated against the model's schema.

        Args:
            fv: A FeatureVector containing computed technical features.

        Returns:
            A PredictionResult with direction, confidence, and reasoning.

        Raises:
            PredictionError: If the FeatureVector violates the schema (missing/invalid inputs).
        """
        try:
            self.schema.validate_features(fv)
        except ValueError as e:
            raise PredictionError(str(e))
            
        votes = self._collect_votes(fv)

        if not votes:
            # No usable features → NEUTRAL with zero confidence
            return PredictionResult(
                symbol=fv.symbol,
                timestamp=fv.timestamp,
                timeframe=fv.timeframe,
                direction=PredictionDirection.NEUTRAL,
                confidence=Decimal("0"),
                reasoning="No usable features available for prediction",
                model_name="baseline_development_predictor",
            )

        # Weighted average score
        total_weight = sum(w for _, _, w in votes)
        weighted_score = sum(v * w for _, v, w in votes) / total_weight

        # Determine direction
        if weighted_score > self._threshold:
            direction = PredictionDirection.BUY
        elif weighted_score < -self._threshold:
            direction = PredictionDirection.SELL
        else:
            direction = PredictionDirection.NEUTRAL

        # Confidence: abs(score) clamped to [0, 1]
        raw_confidence = min(abs(weighted_score), 1.0)
        confidence = Decimal(str(raw_confidence)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Build reasoning
        reasoning = self._build_reasoning(votes, weighted_score, direction)

        return PredictionResult(
            symbol=fv.symbol,
            timestamp=fv.timestamp,
            timeframe=fv.timeframe,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            model_name="baseline_development_predictor",
        )

    # ===============================================================
    # Private methods
    # ===============================================================

    def _validate_features(self, fv: FeatureVector) -> None:
        """Reject FeatureVectors containing NaN or Inf values. Handled by schema."""
        pass

    def _collect_votes(
        self, fv: FeatureVector
    ) -> list[tuple[str, float, float]]:
        """
        Collect directional votes from available features.

        Returns a list of (feature_name, vote, weight) tuples where:
            vote ∈ [-1.0, +1.0]
            weight > 0

        Only features that are not None contribute.
        """
        votes: list[tuple[str, float, float]] = []

        # --- RSI ---
        if fv.rsi_value is not None:
            vote = self._score_rsi(fv.rsi_value)
            votes.append(("RSI", vote, _WEIGHTS["rsi"]))

        # --- MACD Histogram ---
        if fv.macd_histogram is not None:
            vote = self._score_macd(fv, fv.macd_histogram)
            votes.append(("MACD", vote, _WEIGHTS["macd"]))

        # --- SMA vs last close ---
        if fv.sma_value is not None:
            vote = self._score_ma_relationship(fv, fv.sma_value)
            if vote is not None:
                votes.append(("SMA", vote, _WEIGHTS["sma"]))

        # --- EMA vs last close ---
        if fv.ema_value is not None:
            vote = self._score_ma_relationship(fv, fv.ema_value)
            if vote is not None:
                votes.append(("EMA", vote, _WEIGHTS["ema"]))

        # --- Momentum ---
        if fv.momentum_value is not None:
            vote = self._score_momentum(fv, fv.momentum_value)
            votes.append(("Momentum", vote, _WEIGHTS["momentum"]))

        # --- Bollinger Position ---
        if (
            fv.bollinger_upper is not None
            and fv.bollinger_lower is not None
            and fv.bollinger_middle is not None
        ):
            vote = self._score_bollinger(fv)
            if vote is not None:
                votes.append(("Bollinger", vote, _WEIGHTS["bollinger"]))

        return votes

    @staticmethod
    def _score_rsi(rsi_value: float) -> float:
        """
        Score RSI on a continuous scale.

        RSI < 30: strong bullish (oversold)   →  up to +1.0
        RSI 30–50: mild bullish               →  0 to +0.5
        RSI 50: neutral                       →  0.0
        RSI 50–70: mild bearish               →  0 to -0.5
        RSI > 70: strong bearish (overbought) →  up to -1.0
        """
        if rsi_value <= 30.0:
            # Scale from +0.5 at 30 to +1.0 at 0
            return min(0.5 + (30.0 - rsi_value) / 60.0, 1.0)
        elif rsi_value <= 50.0:
            # Scale from 0.0 at 50 to +0.5 at 30
            return (50.0 - rsi_value) / 40.0
        elif rsi_value <= 70.0:
            # Scale from 0.0 at 50 to -0.5 at 70
            return -(rsi_value - 50.0) / 40.0
        else:
            # Scale from -0.5 at 70 to -1.0 at 100
            return max(-0.5 - (rsi_value - 70.0) / 60.0, -1.0)

    @staticmethod
    def _score_macd(fv: FeatureVector, histogram: float) -> float:
        """
        Score MACD histogram direction.

        Positive histogram → bullish vote
        Negative histogram → bearish vote

        Normalized by last_close to be scale-invariant, then bounded via tanh.
        Limitation: True MACD normalization requires historical variance of the histogram.
        Using last_close is a safe, simple approximation for a baseline predictor.
        """
        if histogram == 0.0 or fv.last_close == 0.0:
            return 0.0
        
        normalized = histogram / fv.last_close
        return math.tanh(normalized * 1000.0)

    @staticmethod
    def _score_ma_relationship(fv: FeatureVector, ma_value: float) -> Optional[float]:
        """
        Score the relationship between the latest implied close and a moving average.

        Close > MA → bullish (+)
        Close < MA → bearish (-)
        Close == MA → neutral (0)

        The deviation is expressed as a fraction of MA and bounded via tanh.
        """
        if ma_value == 0.0:
            return None

        deviation = (fv.last_close - ma_value) / ma_value
        return math.tanh(deviation * 100.0)  # Scale for sensitivity

    @staticmethod
    def _score_momentum(fv: FeatureVector, momentum_value: float) -> float:
        """
        Score momentum direction.

        Positive momentum → bullish
        Negative momentum → bearish
        
        Normalized by last_close to ensure scale invariance across instruments.
        """
        if momentum_value == 0.0 or fv.last_close == 0.0:
            return 0.0
            
        normalized = momentum_value / fv.last_close
        return math.tanh(normalized * 100.0)

    @staticmethod
    def _score_bollinger(fv: FeatureVector) -> Optional[float]:
        """
        Score the position within Bollinger Bands.

        last_close < bollinger_lower -> bullish/reversion evidence
        last_close > bollinger_upper -> bearish/reversion evidence
        last_close near/inside the bands -> neutral unless another clearly justified deterministic condition applies

        This remains a BASELINE / DEVELOPMENT PREDICTOR.
        """
        band_width = fv.bollinger_upper - fv.bollinger_lower
        if band_width <= 0:
            return None

        if fv.last_close < fv.bollinger_lower:
            return 1.0  # Bullish reversion
        elif fv.last_close > fv.bollinger_upper:
            return -1.0 # Bearish reversion
        else:
            return 0.0  # Neutral inside bands

    def _build_reasoning(
        self,
        votes: list[tuple[str, float, float]],
        score: float,
        direction: PredictionDirection,
    ) -> str:
        """Build a human-readable reasoning string."""
        bullish_features = [name for name, v, _ in votes if v > 0]
        bearish_features = [name for name, v, _ in votes if v < 0]
        neutral_features = [name for name, v, _ in votes if v == 0]
        
        parts = ["BASELINE PREDICTOR (not a trading recommendation)."]
        if bullish_features:
            parts.append(f"bullish: {', '.join(bullish_features)};")
        if bearish_features:
            parts.append(f"bearish: {', '.join(bearish_features)};")
        if neutral_features:
            parts.append(f"neutral: {', '.join(neutral_features)};")
            
        parts.append(f"Weighted score: {score:+.4f}.")
        parts.append(f"Direction: {direction.value}.")
        
        return " ".join(parts)

class PredictionEngine:
    """
    Higher-level prediction engine that delegates to a registered PredictionModel.
    """
    
    def __init__(self, model: PredictionModel):
        self.model = model
        
    def predict(self, fv: FeatureVector) -> PredictionResult:
        """Forward the prediction request to the underlying model."""
        return self.model.predict(fv)
