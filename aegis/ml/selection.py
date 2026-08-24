"""
Model selection based on validation metrics.
"""
from typing import List, Dict, Any

from aegis.ml.evaluation import MLMetrics


class ModelSelectionResult:
    def __init__(self, selected_model: Any, selection_metric: str, best_score: float, reason: str):
        self.selected_model = selected_model
        self.selection_metric = selection_metric
        self.best_score = best_score
        self.reason = reason


class ModelSelector:
    """
    Selects the best model from candidates using validation metrics.
    """
    def __init__(self, metric_name: str = "f1_macro"):
        self.metric_name = metric_name
        
    def select(self, candidates: List[Any], metrics: Dict[str, MLMetrics]) -> ModelSelectionResult:
        """
        Selects the best candidate based on the metric.
        Tie-breaking:
        1. Target metric
        2. balanced accuracy (recall_macro)
        3. model ID (alphabetical)
        """
        if not candidates:
            raise ValueError("No candidates to select from")
            
        best_candidate = None
        best_score = -1.0
        best_balanced_acc = -1.0
        reason = ""
        
        for candidate in candidates:
            cand_id = candidate.model_id
            if cand_id not in metrics:
                continue
                
            cand_metrics = metrics[cand_id]
            score = getattr(cand_metrics, self.metric_name)
            balanced_acc = cand_metrics.recall_macro # Often used as balanced accuracy
            
            # 1. Compare target metric
            if score > best_score:
                best_candidate = candidate
                best_score = score
                best_balanced_acc = balanced_acc
                reason = f"Highest {self.metric_name}"
            # 2. Tie-break on balanced accuracy
            elif score == best_score:
                if balanced_acc > best_balanced_acc:
                    best_candidate = candidate
                    best_balanced_acc = balanced_acc
                    reason = "Tie broken by highest balanced accuracy (recall_macro)"
                # 3. Tie-break on model ID alphabetically
                elif balanced_acc == best_balanced_acc:
                    if best_candidate is None or cand_id < best_candidate.model_id:
                        best_candidate = candidate
                        reason = "Tie broken by deterministic alphabetical sort"
                        
        if best_candidate is None:
            raise ValueError("Could not select a model (missing metrics?)")
            
        return ModelSelectionResult(
            selected_model=best_candidate,
            selection_metric=self.metric_name,
            best_score=best_score,
            reason=reason
        )
