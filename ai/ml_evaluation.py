"""
ML Evaluation Module
Comprehensive metrics for model performance.
"""
import numpy as np
from typing import Dict, List, Optional

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for positive class (optional)
    
    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, confusion_matrix
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available"}

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            result["roc_auc"] = 0.5
    else:
        result["roc_auc"] = 0.5

    return result


def get_feature_importance_mi(X, y, feature_names: List[str], n_features: int = 25) -> Dict:
    """
    Get feature importance using mutual information.
    Returns dict of feature_name -> importance score.
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
        importance = mutual_info_classif(X, y, random_state=42)
        indices = np.argsort(importance)[::-1][:n_features]
        return {feature_names[i]: float(importance[i]) for i in indices}
    except Exception:
        return {}
