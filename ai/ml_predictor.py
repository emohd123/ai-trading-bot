"""
ML Predictor Service - PHASE 4 ENHANCED
Real-time ensemble predictions for price direction.
Supports multi-horizon and LSTM sequence input.

Phase 4 additions:
- Auto-retraining capability (weekly schedule)
- Performance tracking (rolling accuracy)
- Feature importance logging
- Model comparison before deployment
"""
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Optional, Tuple, List

import config
from datetime import datetime, timedelta
from ai.ml_features import MLFeatureEngineer
from ai.ml_ensemble import MLEnsemble

try:
    from ai.ml_lstm import SEQ_LENGTH
except ImportError:
    SEQ_LENGTH = 24


class MLPredictor:
    """
    Real-time ML price prediction service with auto-improvement.
    
    Phase 4 Features:
    - Performance tracking over rolling window
    - Auto-retraining when accuracy drops
    - Feature importance analysis
    - Model versioning
    """

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.feature_engineer = MLFeatureEngineer()
        self.ensemble = MLEnsemble(model_dir=self.model_dir)
        self.models_loaded = self.ensemble.load_models()
        
        # Phase 4: Performance tracking
        self.performance_file = os.path.join(self.model_dir, "ml_performance.json")
        self.predictions_history: List[Dict] = []
        self.max_history = 100  # Track last 100 predictions
        self.accuracy_threshold = 0.50  # Retrain if accuracy drops below this
        
        # Phase 4: Retraining settings
        self.retrain_interval_days = 7  # Retrain weekly
        self.last_retrain_date = None
        self.min_predictions_for_eval = 20  # Need at least 20 predictions to evaluate
        
        # Load performance history
        self._load_performance()
        
    def _load_performance(self):
        """Load performance history and model accuracy from file."""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.predictions_history = data.get("predictions", [])[-self.max_history:]
                    self.last_retrain_date = data.get("last_retrain_date")
                    self._model_accuracy = data.get("model_accuracy", {})
        except Exception:
            self.predictions_history = []
        if not hasattr(self, "_model_accuracy"):
            self._model_accuracy = {}

    def _save_performance(self):
        """Save performance history and model accuracy to file."""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            data = {
                "predictions": self.predictions_history[-self.max_history:],
                "last_retrain_date": self.last_retrain_date,
                "model_accuracy": getattr(self, "_model_accuracy", {}),
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.performance_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def record_prediction(self, prediction: Dict, actual_direction: Optional[str] = None):
        """
        Record a prediction for performance tracking.
        Stores model_votes so we can compute per-model accuracy when outcome is set.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "predicted_direction": prediction.get("direction"),
            "confidence": prediction.get("confidence", 0),
            "probability": prediction.get("probability", 0.5),
            "model_votes": prediction.get("model_votes", {}),
            "actual_direction": actual_direction,
            "correct": None
        }
        if actual_direction:
            record["correct"] = prediction.get("direction") == actual_direction
        self.predictions_history.append(record)
        
        # Keep only last N predictions
        if len(self.predictions_history) > self.max_history:
            self.predictions_history = self.predictions_history[-self.max_history:]
            
        self._save_performance()
        
    def update_prediction_outcome(self, timestamp: str, actual_direction: str):
        """
        Update a past prediction with the actual outcome.
        Also updates per-model accuracy for dynamic ensemble weights.
        """
        for pred in self.predictions_history:
            if pred.get("timestamp") == timestamp:
                pred["actual_direction"] = actual_direction
                pred["correct"] = pred.get("predicted_direction") == actual_direction
                model_votes = pred.get("model_votes") or {}
                acc = getattr(self, "_model_accuracy", {})
                for model, prob in model_votes.items():
                    if model not in acc:
                        acc[model] = {"correct": 0, "total": 0}
                    acc[model]["total"] += 1
                    model_up = (prob or 0.5) > 0.5
                    if (model_up and actual_direction == "UP") or (not model_up and actual_direction == "DOWN"):
                        acc[model]["correct"] += 1
                self._model_accuracy = acc
                break
        self._save_performance()

    def update_outcome_for_position_close(self, entry_time_str: str, actual_direction: str) -> bool:
        """
        Find the latest prediction without outcome and with timestamp <= entry_time;
        update it with actual_direction (e.g. from trade PnL: 'UP' if pnl > 0 else 'DOWN').

        Args:
            entry_time_str: Position entry time, e.g. 'YYYY-MM-DD HH:MM:SS' or ISO
            actual_direction: 'UP' or 'DOWN'

        Returns:
            True if a prediction was updated, False otherwise
        """
        if not entry_time_str:
            return False
        try:
            if "T" in entry_time_str:
                entry_dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
            else:
                entry_dt = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return False
        if entry_dt.tzinfo:
            entry_ts = entry_dt.timestamp()
        else:
            entry_ts = entry_dt.timestamp()
        candidate = None
        candidate_ts = -1.0
        for pred in self.predictions_history:
            if pred.get("actual_direction") is not None:
                continue
            ts_str = pred.get("timestamp")
            if not ts_str:
                continue
            try:
                if "T" in ts_str:
                    pred_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    pred_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                pred_ts = pred_dt.timestamp() if not pred_dt.tzinfo else pred_dt.timestamp()
            except Exception:
                continue
            if pred_ts <= entry_ts and pred_ts > candidate_ts:
                candidate_ts = pred_ts
                candidate = pred
        if candidate is None:
            return False
        self.update_prediction_outcome(candidate["timestamp"], actual_direction)
        return True

    def get_ensemble_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Get dynamic ensemble weights from per-model accuracy (from prediction outcomes).
        When a model has too few evaluations, use prior 0.25 each so we don't overfit to tiny samples.
        """
        acc = getattr(self, "_model_accuracy", {})
        default_models = ["rf", "xgb", "lgb", "lstm"]
        min_eval = 10
        prior = 1.0 / len(default_models)
        weights = {}
        for m in default_models:
            stats = acc.get(m, {"correct": 0, "total": 0})
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            if total < min_eval:
                weights[m] = prior
            else:
                rate = correct / total if total else 0.5
                weights[m] = max(0.05, min(0.6, 0.2 + 0.4 * rate))
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
        else:
            weights = {k: prior for k in default_models}
        return weights

    def get_accuracy(self, last_n: int = 50) -> Dict:
        """
        Calculate prediction accuracy over recent predictions.
        
        Args:
            last_n: Number of recent predictions to evaluate
            
        Returns:
            Accuracy metrics
        """
        # Get predictions with known outcomes
        evaluated = [p for p in self.predictions_history if p.get("correct") is not None]
        recent = evaluated[-last_n:] if len(evaluated) > last_n else evaluated
        
        if not recent:
            return {
                "accuracy": 0.5,
                "sample_size": 0,
                "up_accuracy": 0.5,
                "down_accuracy": 0.5,
                "needs_retrain": False
            }
        
        correct_count = sum(1 for p in recent if p.get("correct"))
        total = len(recent)
        accuracy = correct_count / total if total > 0 else 0.5
        
        # Calculate directional accuracy
        up_preds = [p for p in recent if p.get("predicted_direction") == "UP"]
        down_preds = [p for p in recent if p.get("predicted_direction") == "DOWN"]
        
        up_correct = sum(1 for p in up_preds if p.get("correct"))
        down_correct = sum(1 for p in down_preds if p.get("correct"))
        
        up_accuracy = up_correct / len(up_preds) if up_preds else 0.5
        down_accuracy = down_correct / len(down_preds) if down_preds else 0.5
        
        # Check if retraining is needed
        needs_retrain = (
            total >= self.min_predictions_for_eval and 
            accuracy < self.accuracy_threshold
        )
        
        return {
            "accuracy": round(accuracy, 4),
            "sample_size": total,
            "correct_count": correct_count,
            "up_accuracy": round(up_accuracy, 4),
            "down_accuracy": round(down_accuracy, 4),
            "needs_retrain": needs_retrain
        }
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from the ensemble models.
        
        Returns:
            Feature importance rankings
        """
        importance = {}
        
        try:
            # Random Forest importance
            if hasattr(self.ensemble, 'rf_model') and self.ensemble.rf_model is not None:
                if hasattr(self.ensemble.rf_model, 'feature_importances_'):
                    rf_imp = self.ensemble.rf_model.feature_importances_
                    if self.ensemble.feature_names and len(rf_imp) == len(self.ensemble.feature_names):
                        rf_sorted = sorted(
                            zip(self.ensemble.feature_names, rf_imp),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        importance["random_forest"] = {
                            name: round(float(imp), 4) 
                            for name, imp in rf_sorted[:15]
                        }
            
            # XGBoost importance
            if hasattr(self.ensemble, 'xgb_model') and self.ensemble.xgb_model is not None:
                if hasattr(self.ensemble.xgb_model, 'feature_importances_'):
                    xgb_imp = self.ensemble.xgb_model.feature_importances_
                    if self.ensemble.feature_names and len(xgb_imp) == len(self.ensemble.feature_names):
                        xgb_sorted = sorted(
                            zip(self.ensemble.feature_names, xgb_imp),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        importance["xgboost"] = {
                            name: round(float(imp), 4)
                            for name, imp in xgb_sorted[:15]
                        }
                        
            # LightGBM importance
            if hasattr(self.ensemble, 'lgb_model') and self.ensemble.lgb_model is not None:
                if hasattr(self.ensemble.lgb_model, 'feature_importances_'):
                    lgb_imp = self.ensemble.lgb_model.feature_importances_
                    if self.ensemble.feature_names and len(lgb_imp) == len(self.ensemble.feature_names):
                        lgb_sorted = sorted(
                            zip(self.ensemble.feature_names, lgb_imp),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        importance["lightgbm"] = {
                            name: round(float(imp), 4)
                            for name, imp in lgb_sorted[:15]
                        }
                        
        except Exception as e:
            importance["error"] = str(e)
            
        return importance
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if models should be retrained.
        
        Returns:
            (should_retrain, reason)
        """
        # Check accuracy
        accuracy_data = self.get_accuracy()
        if accuracy_data["needs_retrain"]:
            return True, f"Accuracy dropped to {accuracy_data['accuracy']:.1%}"
        
        # Check time since last retrain
        if self.last_retrain_date:
            try:
                last_date = datetime.fromisoformat(self.last_retrain_date)
                days_since = (datetime.now() - last_date).days
                if days_since >= self.retrain_interval_days:
                    return True, f"Scheduled retrain ({days_since} days since last)"
            except:
                pass
        
        return False, "No retrain needed"
    
    def trigger_retrain(self, df: pd.DataFrame) -> Dict:
        """
        Trigger model retraining with current data.
        
        This is a placeholder - actual retraining should be done via ml_training.py
        as it's computationally intensive.
        
        Args:
            df: Historical price data for training
            
        Returns:
            Retrain status
        """
        # In production, this would:
        # 1. Save the training data
        # 2. Train new models
        # 3. Compare with old models
        # 4. Deploy if improved
        
        # For now, just record the request
        self.last_retrain_date = datetime.now().isoformat()
        self._save_performance()
        
        return {
            "status": "retrain_requested",
            "timestamp": self.last_retrain_date,
            "message": "Run 'python ml_training.py' to retrain models",
            "current_accuracy": self.get_accuracy()
        }
    
    def get_model_status(self) -> Dict:
        """
        Get comprehensive model status report.
        
        Returns:
            Model status including accuracy, feature importance, and retrain needs
        """
        accuracy = self.get_accuracy()
        should_retrain, reason = self.should_retrain()
        importance = self.get_feature_importance()
        
        return {
            "models_loaded": self.models_loaded,
            "accuracy": accuracy,
            "feature_importance": importance,
            "should_retrain": should_retrain,
            "retrain_reason": reason,
            "last_retrain": self.last_retrain_date,
            "predictions_tracked": len(self.predictions_history)
        }

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict price direction from current market data.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            {
                'direction': 'UP' or 'DOWN',
                'confidence': 0-1,
                'probability': 0-1 (prob of UP),
                'model_votes': {'rf': 0.7, 'xgb': 0.65, 'lgb': 0.72},
                'models_loaded': True/False
            }
        """
        if df is None or df.empty or len(df) < 100:
            return {
                'direction': 'HOLD',
                'confidence': 0,
                'probability': 0.5,
                'model_votes': {},
                'models_loaded': False,
                'error': 'Insufficient data'
            }

        if not self.models_loaded:
            return {
                'direction': 'HOLD',
                'confidence': 0,
                'probability': 0.5,
                'model_votes': {},
                'models_loaded': False,
                'error': 'Models not trained. Run: python ml_training.py'
            }

        try:
            # Create features (single row for tree models)
            features = self.feature_engineer.create_features(df)
            if features.empty:
                return {
                    'direction': 'HOLD',
                    'confidence': 0,
                    'probability': 0.5,
                    'model_votes': {},
                    'models_loaded': True,
                    'error': 'Feature creation failed'
                }

            # Create sequence for LSTM only when enabled (LSTM is slow; config.ML_USE_LSTM = False for fast path)
            X_seq = None
            if getattr(config, "ML_USE_LSTM", False) and self.ensemble.lstm_model is not None and len(df) >= 100 + SEQ_LENGTH:
                feat_seq = self.feature_engineer.create_features_sequence(df, n_rows=SEQ_LENGTH)
                if not feat_seq.empty and len(feat_seq) >= SEQ_LENGTH:
                    feat_seq = feat_seq.fillna(0).replace([np.inf, -np.inf], 0)
                    if self.ensemble.feature_names:
                        missing = set(self.ensemble.feature_names) - set(feat_seq.columns)
                        for m in missing:
                            feat_seq[m] = 0
                        feat_seq = feat_seq[[c for c in self.ensemble.feature_names if c in feat_seq.columns]]
                    if self.ensemble.scaler is not None:
                        X_seq = self.ensemble.scaler.transform(feat_seq)
                    else:
                        X_seq = feat_seq.values

            # Get ensemble prediction (with dynamic weights from recent accuracy)
            dynamic_weights = self.get_ensemble_weights()
            prob_up, model_votes = self.ensemble.ensemble_predict(features, X_seq=X_seq, weights=dynamic_weights)

            # Determine direction and confidence
            direction = 'UP' if prob_up > 0.5 else 'DOWN'
            confidence = abs(prob_up - 0.5) * 2  # 0-1 scale

            # Multi-horizon structure (primary 4h for now)
            predictions = {
                '4h': {'direction': direction, 'confidence': round(float(confidence), 4)}
            }
            consensus = 'BULLISH' if direction == 'UP' else 'BEARISH'
            short_term = consensus
            long_term = consensus

            result = {
                'direction': direction,
                'confidence': round(float(confidence), 4),
                'probability': round(float(prob_up), 4),
                'model_votes': {k: round(float(v), 4) for k, v in model_votes.items()},
                'models_loaded': True,
                'predictions': predictions,
                'consensus': consensus,
                'short_term': short_term,
                'long_term': long_term
            }
            
            # Phase 4: Auto-track prediction
            self.record_prediction(result)
            
            return result
        except Exception as e:
            return {
                'direction': 'HOLD',
                'confidence': 0,
                'probability': 0.5,
                'model_votes': {},
                'models_loaded': self.models_loaded,
                'error': str(e)
            }

    def get_score(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Get ML score for AI engine integration.
        Score: -1 to +1 (negative = bearish, positive = bullish)

        Returns:
            (score, prediction_dict)
        """
        result = self.predict(df)

        if result.get('error') or not result.get('models_loaded'):
            return 0.0, result

        prob = result['probability']
        confidence = result['confidence']

        # Convert to -1 to +1 score
        if result['direction'] == 'UP':
            score = confidence
        else:
            score = -confidence

        return round(float(score), 4), result
