import pytest
from pydantic import BaseModel
from datetime import datetime, timezone
from decimal import Decimal

from aegis.ml.selection import ModelSelector, ModelSelectionResult
from aegis.ml.evaluation import MLMetrics

class MockModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        
def test_model_selection_best_f1():
    models = [MockModel("model_A"), MockModel("model_B")]
    metrics = {
        "model_A": MLMetrics(accuracy=0.8, precision_macro=0.8, recall_macro=0.8, f1_macro=0.8, confusion_matrix=[], classes=[]),
        "model_B": MLMetrics(accuracy=0.9, precision_macro=0.9, recall_macro=0.9, f1_macro=0.9, confusion_matrix=[], classes=[]),
    }
    
    selector = ModelSelector(metric_name="f1_macro")
    result = selector.select(models, metrics)
    
    assert result.selected_model.model_id == "model_B"
    assert result.selection_metric == "f1_macro"
    assert result.best_score == 0.9
    assert result.reason == "Highest f1_macro"

def test_model_selection_tie_breaker_balanced_accuracy():
    models = [MockModel("model_A"), MockModel("model_B")]
    # Same F1, but model_B has higher balanced accuracy (represented by recall_macro in typical sklearn classification_report macro avg)
    metrics = {
        "model_A": MLMetrics(accuracy=0.8, precision_macro=0.8, recall_macro=0.8, f1_macro=0.8, confusion_matrix=[], classes=[]),
        "model_B": MLMetrics(accuracy=0.8, precision_macro=0.7, recall_macro=0.9, f1_macro=0.8, confusion_matrix=[], classes=[]),
    }
    
    selector = ModelSelector(metric_name="f1_macro")
    result = selector.select(models, metrics)
    
    assert result.selected_model.model_id == "model_B"
    assert result.reason == "Tie broken by highest balanced accuracy (recall_macro)"

def test_model_selection_tie_breaker_alphabetical():
    models = [MockModel("model_B"), MockModel("model_A")]
    # Identical F1 and recall_macro
    metrics = {
        "model_A": MLMetrics(accuracy=0.8, precision_macro=0.8, recall_macro=0.8, f1_macro=0.8, confusion_matrix=[], classes=[]),
        "model_B": MLMetrics(accuracy=0.8, precision_macro=0.8, recall_macro=0.8, f1_macro=0.8, confusion_matrix=[], classes=[]),
    }
    
    selector = ModelSelector(metric_name="f1_macro")
    result = selector.select(models, metrics)
    
    # model_A is alphabetically first
    assert result.selected_model.model_id == "model_A"
    assert result.reason == "Tie broken by deterministic alphabetical sort"
