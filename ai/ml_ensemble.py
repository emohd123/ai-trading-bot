"""
ML Ensemble Module
Random Forest, XGBoost, LightGBM, LSTM models for price prediction.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import os

# Optional imports - models may not be installed
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from ai.ml_lstm import predict_lstm, SEQ_LENGTH
    LSTM_AVAILABLE = True
except (ImportError, Exception):
    LSTM_AVAILABLE = False


def create_random_forest():
    """Create Random Forest model"""
    if not SKLEARN_AVAILABLE:
        return None
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )


def create_xgboost():
    """Create XGBoost model"""
    if not XGBOOST_AVAILABLE:
        return None
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )


def create_lightgbm():
    """Create LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        return None
    return lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )


class MLEnsemble:
    """Ensemble of ML models for price prediction"""

    def __init__(self, model_dir: str = None):
        import config
        self.model_dir = model_dir or config.MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)

        self.rf_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.feature_names = None

        # Ensemble weights (4 models: RF, XGB, LGB, LSTM)
        self.weights = {
            'rf': 0.25,
            'xgb': 0.25,
            'lgb': 0.25,
            'lstm': 0.25
        }

    def load_models(self) -> bool:
        """Load trained models from disk"""
        import joblib

        loaded = False
        try:
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                loaded = True
        except Exception:
            pass

        try:
            rf_path = os.path.join(self.model_dir, 'rf_model.pkl')
            if os.path.exists(rf_path) and SKLEARN_AVAILABLE:
                self.rf_model = joblib.load(rf_path)
                loaded = True
        except Exception:
            pass

        try:
            xgb_path = os.path.join(self.model_dir, 'xgb_model.pkl')
            if os.path.exists(xgb_path) and XGBOOST_AVAILABLE:
                self.xgb_model = joblib.load(xgb_path)
                loaded = True
        except Exception:
            pass

        try:
            lgb_path = os.path.join(self.model_dir, 'lgb_model.pkl')
            if os.path.exists(lgb_path) and LIGHTGBM_AVAILABLE:
                self.lgb_model = joblib.load(lgb_path)
                loaded = True
        except Exception:
            pass

        try:
            meta_path = os.path.join(self.model_dir, 'meta.json')
            if os.path.exists(meta_path):
                import json
                with open(meta_path, encoding='utf-8') as f:
                    meta = json.load(f)
                    self.feature_names = meta.get('feature_names') or meta.get('selected_features')
        except Exception:
            pass

        # Load LSTM
        if LSTM_AVAILABLE:
            try:
                import tensorflow as tf
                lstm_path = os.path.join(self.model_dir, 'lstm_model.keras')
                if os.path.exists(lstm_path):
                    self.lstm_model = tf.keras.models.load_model(lstm_path)
                    loaded = True
            except Exception:
                pass

        return loaded

    def predict_proba(self, X: pd.DataFrame, X_seq: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get probability predictions from each model.
        X_seq: optional (n_rows, n_features) for LSTM - last 24 rows of scaled features.
        Returns dict: {'rf': prob_up, 'xgb': prob_up, 'lgb': prob_up, 'lstm': prob_up}
        """
        if X.empty or len(X) == 0:
            base = {'rf': 0.5, 'xgb': 0.5, 'lgb': 0.5, 'lstm': 0.5}
            return {k: v for k, v in base.items() if k in self.weights}

        # Ensure correct column order
        if self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            for m in missing:
                X[m] = 0
            X = X[[c for c in self.feature_names if c in X.columns]]
        else:
            X = X.fillna(0)

        # Scale if scaler exists
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X)
            except Exception:
                X_scaled = X.values
        else:
            X_scaled = X.values

        # LightGBM prefers DataFrame with feature names (avoids sklearn warning)
        feature_cols = list(X.columns) if isinstance(X, pd.DataFrame) else None
        X_for_lgb = pd.DataFrame(X_scaled, columns=feature_cols) if feature_cols else X_scaled

        predictions = {}

        if self.rf_model is not None:
            try:
                proba = self.rf_model.predict_proba(X_scaled)
                predictions['rf'] = float(proba[0][1]) if len(proba.shape) > 1 and proba.shape[1] > 1 else 0.5
            except Exception:
                predictions['rf'] = 0.5

        if self.xgb_model is not None:
            try:
                proba = self.xgb_model.predict_proba(X_scaled)
                predictions['xgb'] = float(proba[0][1]) if len(proba.shape) > 1 and proba.shape[1] > 1 else 0.5
            except Exception:
                predictions['xgb'] = 0.5

        if self.lgb_model is not None:
            try:
                proba = self.lgb_model.predict_proba(X_for_lgb)
                predictions['lgb'] = float(proba[0][1]) if len(proba.shape) > 1 and proba.shape[1] > 1 else 0.5
            except Exception:
                predictions['lgb'] = 0.5

        if self.lstm_model is not None and LSTM_AVAILABLE and X_seq is not None and len(X_seq) >= SEQ_LENGTH:
            try:
                X_last = X_seq[-SEQ_LENGTH:] if len(X_seq) > SEQ_LENGTH else X_seq
                lstm_prob = predict_lstm(self.lstm_model, X_last, scaler=None, seq_length=SEQ_LENGTH)
                predictions['lstm'] = float(lstm_prob)
            except Exception:
                predictions['lstm'] = 0.5
        elif 'lstm' in self.weights:
            predictions['lstm'] = 0.5

        if not predictions:
            return {'rf': 0.5, 'xgb': 0.5, 'lgb': 0.5, 'lstm': 0.5}

        return predictions

    def ensemble_predict(self, X: pd.DataFrame, X_seq: Optional[np.ndarray] = None, weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Weighted ensemble prediction.
        X_seq: optional sequence for LSTM (last 24 rows of scaled features). When None, LSTM weight is redistributed to rf/xgb/lgb.
        weights: optional per-model weights (e.g. from dynamic accuracy); when None use self.weights.
        Returns (probability_up, model_votes_dict)
        """
        votes = self.predict_proba(X, X_seq=X_seq)

        use_weights = weights if weights is not None else dict(self.weights)
        if X_seq is None and "lstm" in votes:
            use_weights = {k: v for k, v in use_weights.items() if k != "lstm"}
            total_w = sum(use_weights.values())
            if total_w > 0:
                use_weights = {k: v / total_w for k, v in use_weights.items()}
            else:
                use_weights = {k: 1.0 / 3 for k in ["rf", "xgb", "lgb"] if k in votes}

        total_weight = 0
        weighted_sum = 0
        for model, prob in votes.items():
            w = use_weights.get(model, 0)
            weighted_sum += prob * w
            total_weight += w

        if total_weight > 0:
            final_prob = weighted_sum / total_weight
        else:
            final_prob = 0.5

        return float(np.clip(final_prob, 0, 1)), votes
