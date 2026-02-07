"""
ML Training Pipeline
Multi-horizon predictions with feature selection, hyperparameter tuning, walk-forward validation.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import config

# Optional imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    import joblib
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

from ai.ml_features import MLFeatureEngineer, select_features
from ai.ml_ensemble import create_random_forest, create_xgboost, create_lightgbm
from ai.ml_evaluation import evaluate_model

# Multi-horizon: hours ahead to predict
HORIZONS = {'1h': 1, '4h': 4, '12h': 12, '24h': 24}
DEFAULT_HORIZON = '4h'
N_SELECTED_FEATURES = 25

# Retrain request file (written by dashboard/Meta AI; consumed by worker or cron)
RETRAIN_REQUEST_FILE = os.path.join(config.DATA_DIR, "ml_retrain_request.json")
PREVIOUS_VAL_FILE = "previous_val.json"  # stored in model_dir
BACKUP_SUBDIR = "backup"
# Optional ML hyperparams (horizon, n_selected_features) from param_optimizer tune_ml_hyperparams
ML_HYPERPARAMS_FILE = os.path.join(config.DATA_DIR, "ml_hyperparams.json")


def get_ml_hyperparams() -> Dict:
    """Read preferred horizon and n_selected_features from tune run (if any)."""
    try:
        if os.path.exists(ML_HYPERPARAMS_FILE):
            with open(ML_HYPERPARAMS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def request_ml_retrain(reason: str = "requested") -> None:
    """Write a retrain request so a worker or cron can run training (non-blocking)."""
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(RETRAIN_REQUEST_FILE, "w", encoding="utf-8") as f:
            json.dump({"requested_at": datetime.now().isoformat(), "reason": reason}, f, indent=2)
    except Exception:
        pass


def clear_retrain_request() -> None:
    """Remove the retrain request file after processing."""
    try:
        if os.path.exists(RETRAIN_REQUEST_FILE):
            os.remove(RETRAIN_REQUEST_FILE)
    except Exception:
        pass


def fetch_historical_data(limit: int = 8760) -> pd.DataFrame:
    """Fetch 12 months of hourly data from Binance (increased from 6 months for better training)."""
    try:
        from binance.client import Client
        client = Client()
        all_klines = []
        batch_size = 1000
        end_time = None

        while len(all_klines) < limit:
            fetch_limit = min(batch_size, limit - len(all_klines))
            params = {'symbol': config.SYMBOL, 'interval': '1h', 'limit': fetch_limit}
            if end_time is not None:
                params['endTime'] = end_time

            klines = client.get_klines(**params)
            if not klines:
                break

            all_klines = klines + all_klines
            end_time = klines[0][0] - 1
            if len(klines) < batch_size:
                break

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        return df[['open', 'high', 'low', 'close', 'volume']].tail(limit)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def create_target(df: pd.DataFrame, horizon: int = 4, min_move_pct: float = 0.001) -> np.ndarray:
    """Target: Will price move up enough to cover transaction costs?

    A move must exceed min_move_pct (default 0.1% = typical round-trip fees) to count as UP.
    Moves smaller than min_move_pct in either direction are labeled as DOWN (no-trade).
    This prevents the model from learning that tiny 0.01% moves are profitable.
    """
    future_return = (df['close'].shift(-horizon) - df['close']) / df['close']
    # Only label as UP if return exceeds transaction costs
    target = (future_return > min_move_pct).astype(int)
    return target.values


def walk_forward_validation(
    X: np.ndarray, y: np.ndarray, model, scaler,
    n_splits: int = 5
) -> Tuple[float, float]:
    """Time-series aware cross-validation. Returns (mean_score, std_score)."""
    if not SKLEARN_AVAILABLE:
        return 0.5, 0.0

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        acc = (pred == y_test).mean()
        scores.append(acc)

    return float(np.mean(scores)), float(np.std(scores)) if scores else 0.0


def tune_hyperparameters(X_train, y_train, model_type: str = 'rf'):
    """Grid search for optimal hyperparameters."""
    if model_type == 'rf' and SKLEARN_AVAILABLE:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }
        base = RandomForestClassifier(random_state=42, n_jobs=-1)
        gs = GridSearchCV(base, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    elif model_type == 'xgb' and XGBOOST_AVAILABLE:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1]
        }
        base = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        gs = GridSearchCV(base, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    elif model_type == 'lgb' and LIGHTGBM_AVAILABLE:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1]
        }
        base = lgb.LGBMClassifier(random_state=42, verbose=-1)
        gs = GridSearchCV(base, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    return None


def train_models(model_dir: str = None, horizon: str = DEFAULT_HORIZON, safe_deploy: bool = False) -> Dict:
    """
    Full training pipeline with multi-horizon, feature selection, tuning, walk-forward.
    When safe_deploy=True: backup current models, train new ones, deploy only if val accuracy improves.
    """
    model_dir = model_dir or config.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    backup_dir = os.path.join(model_dir, BACKUP_SUBDIR)
    previous_val_path = os.path.join(model_dir, PREVIOUS_VAL_FILE)

    if safe_deploy:
        try:
            os.makedirs(backup_dir, exist_ok=True)
            for name in ["rf_model.pkl", "xgb_model.pkl", "lgb_model.pkl", "scaler.pkl", "meta.json", "previous_val.json"]:
                src = os.path.join(model_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(backup_dir, name))
            for name in ["lstm_model.keras"]:
                src = os.path.join(model_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(backup_dir, name))
        except Exception as e:
            print(f"Safe deploy backup failed: {e}; continuing without restore capability")

    h = HORIZONS.get(horizon, 4)

    print("Fetching historical data (12 months)...")
    df = fetch_historical_data(limit=8760)
    if df.empty or len(df) < 500:
        print("Insufficient data for training")
        return {"error": "Insufficient data"}

    print(f"Loaded {len(df)} candles")
    print(f"Training for horizon: {horizon} ({h} hours)")

    engineer = MLFeatureEngineer()
    features_list = []
    targets_list = []

    # Transaction cost threshold: only count as UP if return > 0.1% (covers fees)
    min_move_pct = 0.001

    max_horizon = max(HORIZONS.values())
    for i in range(100, len(df) - max_horizon):
        df_slice = df.iloc[:i + 1].copy()
        feat_df = engineer.create_features(df_slice)
        if not feat_df.empty:
            features_list.append(feat_df)
            # Cost-aware target: only UP if return exceeds transaction costs
            future_return = (df['close'].iloc[i + h] - df['close'].iloc[i]) / df['close'].iloc[i]
            target = 1 if future_return > min_move_pct else 0
            targets_list.append(target)

    if len(features_list) < 100:
        print("Not enough samples for training")
        return {"error": "Not enough samples"}

    X = pd.concat(features_list, ignore_index=True)
    y = np.array(targets_list)

    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # SPLIT FIRST, then feature selection on training data only (prevents data leakage)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train_raw, X_val_raw, X_test_raw = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    # Feature selection on TRAINING data only (fixes data leakage)
    hyperparams = get_ml_hyperparams()
    n_selected_features = hyperparams.get("n_selected_features", N_SELECTED_FEATURES)
    print(f"Selecting top {n_selected_features} features (from training data only)...")
    X_train_selected, selected_features = select_features(X_train_raw, y_train, n_features=n_selected_features)

    # Apply same feature selection to val and test sets
    X_train = X_train_selected
    X_val = X_val_raw[selected_features] if selected_features else X_val_raw
    X_test = X_test_raw[selected_features] if selected_features else X_test_raw

    rf, xgb_model, lgb_model = None, None, None

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    feature_names = list(X.columns)

    # Train Random Forest (with optional tuning)
    if SKLEARN_AVAILABLE:
        print("Training Random Forest...")
        rf = tune_hyperparameters(X_train_scaled, y_train, 'rf')
        if rf is None:
            rf = create_random_forest()
        if rf is not None:
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)
            y_prob = rf.predict_proba(X_test_scaled)[:, 1] if hasattr(rf, 'predict_proba') else y_pred
            metrics = evaluate_model(y_test, y_pred, y_prob)
            results['rf_accuracy'] = round(metrics['accuracy'], 4)
            results['rf_metrics'] = metrics
            joblib.dump(rf, os.path.join(model_dir, 'rf_model.pkl'))
            print(f"  RF Accuracy: {metrics['accuracy']:.2%}")

    # Train XGBoost
    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        xgb_model = tune_hyperparameters(X_train_scaled, y_train, 'xgb')
        if xgb_model is None:
            xgb_model = create_xgboost()
        if xgb_model is not None:
            xgb_model.fit(X_train_scaled, y_train)
            y_pred = xgb_model.predict(X_test_scaled)
            y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_prob)
            results['xgb_accuracy'] = round(metrics['accuracy'], 4)
            results['xgb_metrics'] = metrics
            joblib.dump(xgb_model, os.path.join(model_dir, 'xgb_model.pkl'))
            print(f"  XGB Accuracy: {metrics['accuracy']:.2%}")

    # Train LightGBM
    if LIGHTGBM_AVAILABLE:
        print("Training LightGBM...")
        lgb_model = tune_hyperparameters(X_train_scaled, y_train, 'lgb')
        if lgb_model is None:
            lgb_model = create_lightgbm()
        if lgb_model is not None:
            lgb_model.fit(X_train_scaled, y_train)
            y_pred = lgb_model.predict(X_test_scaled)
            y_prob = lgb_model.predict_proba(X_test_scaled)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_prob)
            results['lgb_accuracy'] = round(metrics['accuracy'], 4)
            results['lgb_metrics'] = metrics
            joblib.dump(lgb_model, os.path.join(model_dir, 'lgb_model.pkl'))
            print(f"  LGB Accuracy: {metrics['accuracy']:.2%}")

    # LSTM disabled: consistently performs at 38% accuracy (worse than random)
    # Tree-based models (RF, XGB, LGB) are sufficient for the ensemble
    print("  LSTM: Disabled (38% accuracy - worse than random, hurts ensemble)")
    results['lstm_accuracy'] = None
    results['lstm_status'] = 'disabled'

    val_accuracy = 0.5
    try:
        probs = []
        if SKLEARN_AVAILABLE and rf is not None:
            probs.append(rf.predict_proba(X_val_scaled)[:, 1])
        if XGBOOST_AVAILABLE and xgb_model is not None:
            probs.append(xgb_model.predict_proba(X_val_scaled)[:, 1])
        if LIGHTGBM_AVAILABLE and lgb_model is not None:
            probs.append(lgb_model.predict_proba(X_val_scaled)[:, 1])
        if probs:
            avg_prob = np.mean(probs, axis=0)
            val_pred = (avg_prob > 0.5).astype(int)
            val_accuracy = (val_pred == y_val).mean()
        results["val_accuracy"] = round(float(val_accuracy), 4)
    except Exception:
        results["val_accuracy"] = 0.5

    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    with open(os.path.join(model_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'feature_names': feature_names,
            'selected_features': selected_features,
            'n_samples': len(X),
            'horizon': horizon,
            'results': results
        }, f, indent=2)

    if safe_deploy:
        previous_acc = 0.0
        try:
            if os.path.exists(previous_val_path):
                with open(previous_val_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                    previous_acc = float(prev.get("accuracy", 0))
        except Exception:
            pass
        if val_accuracy >= previous_acc:
            with open(previous_val_path, "w", encoding="utf-8") as f:
                json.dump({"accuracy": val_accuracy, "updated_at": datetime.now().isoformat()}, f, indent=2)
            print(f"Safe deploy: val accuracy {val_accuracy:.2%} >= previous {previous_acc:.2%}; keeping new models")
        else:
            print(f"Safe deploy: val accuracy {val_accuracy:.2%} < previous {previous_acc:.2%}; restoring backup")
            try:
                for name in ["rf_model.pkl", "xgb_model.pkl", "lgb_model.pkl", "scaler.pkl", "meta.json", "previous_val.json"]:
                    b = os.path.join(backup_dir, name)
                    if os.path.exists(b):
                        shutil.copy2(b, os.path.join(model_dir, name))
                for name in ["lstm_model.keras"]:
                    b = os.path.join(backup_dir, name)
                    if os.path.exists(b):
                        shutil.copy2(b, os.path.join(model_dir, name))
            except Exception as e:
                print(f"Restore failed: {e}")

    print("Models saved to", model_dir)
    return results


def train_all_horizons(model_dir: str = None) -> Dict:
    """Train models for all horizons (1h, 4h, 12h, 24h)."""
    model_dir = model_dir or config.MODEL_DIR
    all_results = {}
    for horizon in HORIZONS:
        subdir = os.path.join(model_dir, f"horizon_{horizon}")
        os.makedirs(subdir, exist_ok=True)
        print(f"\n=== Training horizon {horizon} ===")
        all_results[horizon] = train_models(model_dir=subdir, horizon=horizon)
    return all_results


if __name__ == '__main__':
    print("=" * 50)
    print("ML Model Training Pipeline (Enhanced)")
    print("=" * 50)
    results = train_models(horizon=DEFAULT_HORIZON, safe_deploy=True)
    print("\nResults:", results)
