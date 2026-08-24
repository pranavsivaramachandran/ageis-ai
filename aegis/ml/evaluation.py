"""
ML Evaluation for AEGIS AI.

Computes classical ML metrics (Accuracy, Precision, Recall, F1)
for the model.
"""

from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from aegis.ml.dataset import MLDataset
from aegis.ml.models import MLPredictionModel
from aegis.prediction.models import PredictionDirection


@dataclass(frozen=True)
class MLMetrics:
    """Metrics for ML model evaluation."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    classes: list[str]


class MLEvaluator:
    """Evaluates an ML model against a dataset."""
    
    @staticmethod
    def evaluate(model: MLPredictionModel, dataset: MLDataset) -> MLMetrics:
        """
        Evaluate model on the given dataset.
        
        Args:
            model: Trained MLPredictionModel.
            dataset: MLDataset (Validation or Test).
            
        Returns:
            MLMetrics containing computed evaluation scores.
        """
        if not dataset:
            raise ValueError("Cannot evaluate on an empty dataset.")
            
        y_true = []
        y_pred = []
        
        for sample in dataset.samples:
            y_true.append(sample.target.value)
            
            # Predict using the full interface
            result = model.predict(sample.raw_feature_vector)
            y_pred.append(result.direction.value)
            
        # Compute metrics
        # Use macro averaging to treat all classes equally, helpful for imbalance
        labels = [d.value for d in PredictionDirection]
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        return MLMetrics(
            accuracy=float(acc),
            precision_macro=float(prec),
            recall_macro=float(rec),
            f1_macro=float(f1),
            confusion_matrix=cm.tolist(),
            classes=labels
        )
