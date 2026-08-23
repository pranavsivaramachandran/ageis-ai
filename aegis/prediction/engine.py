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


class BaselinePredictor:
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

    def predict(self, fv: FeatureVector) -> PredictionResult:
        """
        Produce a deterministic prediction from a FeatureVector.

        Args:
            fv: A FeatureVector containing computed technical features.

        Returns:
            A PredictionResult with direction, confidence, and reasoning.

        Raises:
            PredictionError: If the FeatureVector contains invalid values
                             (NaN/Inf in numeric fields).
        """
        self._validate_features(fv)

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
        total_weight = sum(w for _, w in votes)
        weighted_score = sum(v * w for v, w in votes) / total_weight

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
        """Reject FeatureVectors containing NaN or Inf values."""
        numeric_fields = [
            ("sma_value", fv.sma_value),
            ("ema_value", fv.ema_value),
            ("rsi_value", fv.rsi_value),
            ("macd_line", fv.macd_line),
            ("macd_signal", fv.macd_signal),
            ("macd_histogram", fv.macd_histogram),
            ("atr_value", fv.atr_value),
            ("bollinger_upper", fv.bollinger_upper),
            ("bollinger_middle", fv.bollinger_middle),
            ("bollinger_lower", fv.bollinger_lower),
            ("momentum_value", fv.momentum_value),
            ("volatility", fv.volatility),
        ]

        for name, value in numeric_fields:
            if value is not None and (math.isnan(value) or math.isinf(value)):
                raise PredictionError(
                    f"FeatureVector contains invalid {name}: {value}"
                )

        # Also validate returns list if present
        if fv.returns is not None:
            for i, r in enumerate(fv.returns):
                if r is not None and (math.isnan(r) or math.isinf(r)):
                    raise PredictionError(
                        f"FeatureVector contains invalid return at index {i}: {r}"
                    )

    def _collect_votes(
        self, fv: FeatureVector
    ) -> list[tuple[float, float]]:
        """
        Collect directional votes from available features.

        Returns a list of (vote, weight) tuples where:
            vote ∈ [-1.0, +1.0]
            weight > 0

        Only features that are not None contribute.
        """
        votes: list[tuple[float, float]] = []

        # --- RSI ---
        if fv.rsi_value is not None:
            vote = self._score_rsi(fv.rsi_value)
            votes.append((vote, _WEIGHTS["rsi"]))

        # --- MACD Histogram ---
        if fv.macd_histogram is not None:
            vote = self._score_macd(fv.macd_histogram)
            votes.append((vote, _WEIGHTS["macd"]))

        # --- SMA vs last close ---
        if fv.sma_value is not None and fv.returns is not None:
            vote = self._score_ma_relationship(fv, fv.sma_value)
            if vote is not None:
                votes.append((vote, _WEIGHTS["sma"]))

        # --- EMA vs last close ---
        if fv.ema_value is not None and fv.returns is not None:
            vote = self._score_ma_relationship(fv, fv.ema_value)
            if vote is not None:
                votes.append((vote, _WEIGHTS["ema"]))

        # --- Momentum ---
        if fv.momentum_value is not None:
            vote = self._score_momentum(fv.momentum_value)
            votes.append((vote, _WEIGHTS["momentum"]))

        # --- Bollinger Position ---
        if (
            fv.bollinger_upper is not None
            and fv.bollinger_lower is not None
            and fv.bollinger_middle is not None
            and fv.returns is not None
        ):
            vote = self._score_bollinger(fv)
            if vote is not None:
                votes.append((vote, _WEIGHTS["bollinger"]))

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
    def _score_macd(histogram: float) -> float:
        """
        Score MACD histogram direction.

        Positive histogram → bullish vote
        Negative histogram → bearish vote

        The magnitude is bounded by a sigmoid-like mapping to stay in [-1, 1].
        """
        if histogram == 0.0:
            return 0.0
        # Use tanh for smooth bounded mapping
        return math.tanh(histogram)

    @staticmethod
    def _score_ma_relationship(fv: FeatureVector, ma_value: float) -> Optional[float]:
        """
        Score the relationship between the latest implied close and a moving average.

        Close > MA → bullish (+)
        Close < MA → bearish (-)

        The deviation is expressed as a fraction of MA and bounded via tanh.
        """
        if ma_value == 0.0:
            return None

        # Reconstruct the latest close from returns:
        # We don't have a raw close, but we can infer relative position.
        # The last return = (close - prev_close) / prev_close
        # Instead, compare directly: if returns[-1] is available and positive,
        # and close > MA, it's bullish.
        # A simpler approach: use returns to estimate close relative to MA.
        # Since we lack a raw close in FeatureVector, we check if returns
        # are consistently positive (price trending up vs MA typically rises slower).
        if fv.returns is None or len(fv.returns) == 0:
            return None

        # Use the sum of recent returns as a proxy for price trend vs MA.
        # Positive cumulative returns → likely above MA → bullish.
        recent = fv.returns[-3:] if len(fv.returns) >= 3 else fv.returns
        valid_returns = [r for r in recent if r is not None]
        if not valid_returns:
            return None

        cumulative = sum(valid_returns)
        return math.tanh(cumulative * 10.0)  # Scale for sensitivity

    @staticmethod
    def _score_momentum(momentum_value: float) -> float:
        """
        Score momentum direction.

        Positive momentum → bullish
        Negative momentum → bearish
        Bounded via tanh.
        """
        return math.tanh(momentum_value)

    @staticmethod
    def _score_bollinger(fv: FeatureVector) -> Optional[float]:
        """
        Score the position within Bollinger Bands.

        Near lower band → bullish (potential bounce)
        Near upper band → bearish (potential reversal)

        Uses the %B indicator: (price - lower) / (upper - lower).
        Since we lack a raw close, we use the midpoint relationship
        and recent returns as a proxy.
        """
        band_width = fv.bollinger_upper - fv.bollinger_lower
        if band_width <= 0:
            return None

        if fv.returns is None or len(fv.returns) == 0:
            return None

        # Use returns to determine relative position:
        # negative returns → price moving toward lower band → bullish
        # positive returns → price moving toward upper band → bearish
        recent = fv.returns[-3:] if len(fv.returns) >= 3 else fv.returns
        valid_returns = [r for r in recent if r is not None]
        if not valid_returns:
            return None

        avg_return = sum(valid_returns) / len(valid_returns)

        # Invert: falling price near lower band = bullish potential
        # Estimate %B-like position from returns
        # Strong negative returns → likely near lower band → bullish
        # Strong positive returns → likely near upper band → bearish
        return -math.tanh(avg_return * 15.0)

    def _build_reasoning(
        self,
        votes: list[tuple[float, float]],
        score: float,
        direction: PredictionDirection,
    ) -> str:
        """Build a human-readable reasoning string."""
        n_votes = len(votes)
        bullish = sum(1 for v, _ in votes if v > 0)
        bearish = sum(1 for v, _ in votes if v < 0)

        return (
            f"BASELINE PREDICTOR (not a trading recommendation). "
            f"Evidence: {n_votes} features scored, "
            f"{bullish} bullish, {bearish} bearish. "
            f"Weighted score: {score:+.4f}. "
            f"Direction: {direction.value}."
        )
