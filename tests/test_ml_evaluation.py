"""
Tests for ML Evaluation metrics.
"""

import pytest
from datetime import datetime, timezone

from aegis.interfaces.market_data import Timeframe
from aegis.features.builder import FeatureVector
from aegis.prediction.models import PredictionDirection
from aegis.ml.dataset import MLDataset, MLSample
from aegis.ml.evaluation import MLEvaluator

class DummyModel:
    def __init__(self, predictions):
        # A dictionary mapping timestamp to a predicted direction
        self.predictions = predictions
        
    def predict(self, fv: FeatureVector):
        from aegis.prediction.models import PredictionResult
        from decimal import Decimal
        return PredictionResult(
            symbol=fv.symbol,
            timestamp=fv.timestamp,
            timeframe=fv.timeframe,
            direction=self.predictions[fv.timestamp],
            confidence=Decimal("0.9"),
            model_name="dummy",
            reasoning="test"
        )


class TestMLEvaluator:
    def test_metrics_calculation(self):
        t1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)
        t4 = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
        
        # True labels: BUY, BUY, SELL, NEUTRAL
        samples = [
            MLSample(t1, "BTC", Timeframe.H1, [1.0], PredictionDirection.BUY, FeatureVector(t1, "BTC", Timeframe.H1, 100.0)),
            MLSample(t2, "BTC", Timeframe.H1, [2.0], PredictionDirection.BUY, FeatureVector(t2, "BTC", Timeframe.H1, 101.0)),
            MLSample(t3, "BTC", Timeframe.H1, [3.0], PredictionDirection.SELL, FeatureVector(t3, "BTC", Timeframe.H1, 99.0)),
            MLSample(t4, "BTC", Timeframe.H1, [4.0], PredictionDirection.NEUTRAL, FeatureVector(t4, "BTC", Timeframe.H1, 100.0)),
        ]
        dataset = MLDataset(samples=samples)
        
        # Predictions: BUY, SELL, SELL, NEUTRAL
        preds = {
            t1: PredictionDirection.BUY,
            t2: PredictionDirection.SELL, # Incorrect
            t3: PredictionDirection.SELL,
            t4: PredictionDirection.NEUTRAL
        }
        model = DummyModel(preds)
        
        metrics = MLEvaluator.evaluate(model, dataset)
        
        # 3 out of 4 correct -> Accuracy 0.75
        assert metrics.accuracy == 0.75
        
        # Check confusion matrix exists
        assert len(metrics.confusion_matrix) == len(metrics.classes)
        assert len(metrics.confusion_matrix[0]) == len(metrics.classes)
        
    def test_empty_dataset(self):
        dataset = MLDataset(samples=[])
        model = DummyModel({})
        with pytest.raises(ValueError, match="empty dataset"):
            MLEvaluator.evaluate(model, dataset)
