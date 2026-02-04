"""
Position Size Learner - ML-based optimal position sizing
Learns optimal trade amounts from historical performance under different market conditions.
"""
import os
import json
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from collections import defaultdict
import statistics

import config

logger = logging.getLogger(__name__)

# Optional ML imports
try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class PositionSizeLearner:
    """
    Learns optimal position sizes from historical trade performance.
    Uses statistical analysis and optional ML regression to predict optimal size.
    """
    
    LEARNING_FILE = os.path.join(config.DATA_DIR, "position_size_learning.json")
    
    def __init__(self):
        """Initialize position size learner"""
        # Trade entries (conditions + size used)
        self.trade_entries = []  # List of {trade_id, conditions, position_size, timestamp}
        
        # Trade outcomes (linked to entries)
        self.trade_outcomes = {}  # {trade_id: {pnl_percent, pnl_dollar, exit_type}}
        
        # Learned optimal sizes per condition
        self.optimal_sizes = {}  # {condition_key: {size_mult, samples, avg_profit, profit_factor}}
        
        # ML model (optional)
        self.ml_model = None
        self.ml_model_trained = False
        
        # Load saved state
        self._load_state()
    
    def _load_state(self):
        """Load learned sizes from file"""
        try:
            if os.path.exists(self.LEARNING_FILE):
                with open(self.LEARNING_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.optimal_sizes = data.get('optimal_sizes', {})
                self.trade_entries = data.get('trade_entries', [])[-200:]  # Keep last 200
                self.trade_outcomes = data.get('trade_outcomes', {})
                self.ml_model_trained = data.get('ml_model_trained', False)
                
                logger.info(f"Loaded position size learning: {len(self.optimal_sizes)} condition groups")
        except Exception as e:
            logger.warning(f"Could not load position size learning: {e}")
            self.optimal_sizes = {}
            self.trade_entries = []
            self.trade_outcomes = {}
    
    def _save_state(self):
        """Save learned sizes to file"""
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            data = {
                'optimal_sizes': self.optimal_sizes,
                'trade_entries': self.trade_entries[-200:],  # Keep last 200
                'trade_outcomes': self.trade_outcomes,
                'ml_model_trained': self.ml_model_trained,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.LEARNING_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save position size learning: {e}")
    
    def record_trade_entry(
        self,
        trade_id: int,
        conditions: Dict,
        position_size: float
    ) -> None:
        """
        Record entry conditions and position size used.
        
        Args:
            trade_id: Unique trade ID
            conditions: Dict with regime, volatility, confidence, confluence, streak_multiplier
            position_size: Actual position size in USDT
        """
        entry_record = {
            "trade_id": trade_id,
            "timestamp": datetime.now().isoformat(),
            "position_size": position_size,
            "conditions": {
                "regime": conditions.get("regime", "unknown"),
                "volatility": conditions.get("volatility", "medium"),
                "volatility_ratio": conditions.get("volatility_ratio", 1.0),
                "confidence": conditions.get("confidence", 0.5),
                "confluence": conditions.get("confluence", 0),
                "streak_multiplier": conditions.get("streak_multiplier", 1.0),
                "confidence_tier": self._get_confidence_tier(conditions.get("confidence", 0.5)),
                "volatility_tier": self._get_volatility_tier(conditions.get("volatility_ratio", 1.0))
            }
        }
        
        self.trade_entries.append(entry_record)
        self._save_state()
    
    def record_trade_outcome(
        self,
        trade_id: int,
        pnl_percent: float,
        pnl_dollar: float,
        exit_type: str
    ) -> None:
        """
        Record trade outcome and link to entry.
        
        Args:
            trade_id: Trade ID matching entry
            pnl_percent: Profit/loss percentage
            pnl_dollar: Profit/loss in dollars
            exit_type: How trade exited (profit_target, stop_loss, etc.)
        """
        self.trade_outcomes[trade_id] = {
            "pnl_percent": pnl_percent,
            "pnl_dollar": pnl_dollar,
            "exit_type": exit_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Trigger learning after outcome recorded
        self.learn_optimal_sizes()
        self._save_state()
    
    def _get_confidence_tier(self, confidence: float) -> str:
        """Convert confidence value to tier"""
        if confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _get_volatility_tier(self, vol_ratio: float) -> str:
        """Convert volatility ratio to tier"""
        if vol_ratio >= 2.0:
            return "extreme"
        elif vol_ratio >= 1.5:
            return "high"
        elif vol_ratio >= 1.2:
            return "medium"
        else:
            return "low"
    
    def _get_condition_key(self, conditions: Dict) -> str:
        """Generate key for grouping trades by conditions"""
        regime = conditions.get("regime", "unknown")
        vol_tier = conditions.get("volatility_tier", "medium")
        conf_tier = conditions.get("confidence_tier", "medium")
        return f"{regime}_{vol_tier}_{conf_tier}"
    
    def learn_optimal_sizes(self) -> None:
        """
        Analyze historical trades to find optimal position sizes per condition.
        Groups trades by condition combinations and finds size that maximizes profit.
        """
        if len(self.trade_entries) < 5:
            return  # Need at least 5 trades to learn
        
        # Build dataset: entry conditions + size used + outcome
        complete_trades = []
        for entry in self.trade_entries:
            trade_id = entry["trade_id"]
            if trade_id in self.trade_outcomes:
                outcome = self.trade_outcomes[trade_id]
                complete_trades.append({
                    **entry,
                    "outcome": outcome
                })
        
        if len(complete_trades) < 5:
            return  # Need completed trades
        
        # Group by condition key
        groups = defaultdict(list)
        for trade in complete_trades:
            key = self._get_condition_key(trade["conditions"])
            groups[key].append(trade)
        
        # Analyze each group to find optimal size
        base_size = config.TRADE_AMOUNT_USDT
        min_samples = getattr(config, 'POSITION_SIZE_MIN_SAMPLES', 5)
        
        for condition_key, trades in groups.items():
            if len(trades) < min_samples:
                continue
            
            # Group by position size multiplier (relative to base)
            size_groups = defaultdict(list)
            for trade in trades:
                size_mult = trade["position_size"] / base_size
                # Round to nearest 0.1x for grouping
                size_mult_rounded = round(size_mult * 10) / 10
                size_groups[size_mult_rounded].append(trade)
            
            # Find size multiplier with best performance
            best_mult = None
            best_metric = float('-inf')
            best_samples = 0
            best_avg_profit = 0
            best_profit_factor = 0
            best_win_rate = 0
            
            for size_mult, size_trades in size_groups.items():
                if len(size_trades) < 2:
                    continue  # Need at least 2 trades per size
                
                # Calculate performance metrics
                profits = [t["outcome"]["pnl_dollar"] for t in size_trades]
                profit_pcts = [t["outcome"]["pnl_percent"] for t in size_trades]
                
                avg_profit = statistics.mean(profits) if profits else 0
                avg_profit_pct = statistics.mean(profit_pcts) if profit_pcts else 0
                win_rate = sum(1 for p in profit_pcts if p > 0) / len(profit_pcts) if profit_pcts else 0
                
                # Profit factor: total wins / total losses
                total_wins = sum(p for p in profits if p > 0)
                total_losses = abs(sum(p for p in profits if p < 0))
                profit_factor = total_wins / total_losses if total_losses > 0 else (total_wins if total_wins > 0 else 0)
                
                # Combined metric: profit factor weighted by win rate
                metric = profit_factor * win_rate * avg_profit_pct
                
                if metric > best_metric:
                    best_metric = metric
                    best_mult = size_mult
                    best_samples = len(size_trades)
                    best_avg_profit = avg_profit
                    best_profit_factor = profit_factor
                    best_win_rate = win_rate
            
            if best_mult is not None:
                # Store optimal size for this condition
                self.optimal_sizes[condition_key] = {
                    "size_mult": best_mult,
                    "samples": best_samples,
                    "avg_profit": best_avg_profit,
                    "profit_factor": best_profit_factor,
                    "win_rate": best_win_rate,
                    "last_updated": datetime.now().isoformat()
                }
        
        logger.info(f"Learned optimal sizes for {len(self.optimal_sizes)} condition groups")
        
        # Auto-train ML model if enough data (every 20 trades)
        if len(complete_trades) >= 20 and not self.ml_model_trained:
            try:
                self.train_ml_model()
            except Exception as e:
                logger.warning(f"Auto ML training failed: {e}")
    
    def get_optimal_multiplier(self, conditions: Dict) -> Optional[float]:
        """
        Get learned optimal position size multiplier for given conditions.
        
        Args:
            conditions: Dict with regime, volatility, confidence, etc.
        
        Returns:
            Optimal multiplier (0.3 to 1.5) or None if not enough data
        """
        if not getattr(config, 'POSITION_SIZE_LEARNING_ENABLED', True):
            return None
        
        # Add tiers to conditions
        conditions_with_tiers = dict(conditions)
        conditions_with_tiers["confidence_tier"] = self._get_confidence_tier(
            conditions.get("confidence", 0.5)
        )
        conditions_with_tiers["volatility_tier"] = self._get_volatility_tier(
            conditions.get("volatility_ratio", 1.0)
        )
        
        condition_key = self._get_condition_key(conditions_with_tiers)
        min_samples = getattr(config, 'POSITION_SIZE_MIN_SAMPLES', 5)
        
        if condition_key in self.optimal_sizes:
            optimal = self.optimal_sizes[condition_key]
            if optimal["samples"] >= min_samples:
                # Enforce safety bounds
                size_mult = max(0.3, min(1.5, optimal["size_mult"]))
                return size_mult
        
        return None
    
    def train_ml_model(self) -> bool:
        """
        Train ML regression model to predict optimal position size.
        Uses XGBoost if available.
        
        Returns:
            True if model trained successfully
        """
        if not XGBOOST_AVAILABLE or not NUMPY_AVAILABLE:
            return False
        
        # Build training dataset
        complete_trades = []
        for entry in self.trade_entries:
            trade_id = entry["trade_id"]
            if trade_id in self.trade_outcomes:
                outcome = self.trade_outcomes[trade_id]
                complete_trades.append({
                    **entry,
                    "outcome": outcome
                })
        
        if len(complete_trades) < 20:
            return False  # Need more data for ML
        
        try:
            # Prepare features
            X = []
            y = []
            
            base_size = config.TRADE_AMOUNT_USDT
            for trade in complete_trades:
                cond = trade["conditions"]
                
                # Features: regime (one-hot), volatility_ratio, confidence, confluence, streak_mult
                features = [
                    cond.get("volatility_ratio", 1.0),
                    cond.get("confidence", 0.5),
                    cond.get("confluence", 0) / 10.0,  # Normalize 0-1
                    cond.get("streak_multiplier", 1.0),
                    1.0 if cond.get("regime") == "trending_up" else 0.0,
                    1.0 if cond.get("regime") == "trending_down" else 0.0,
                    1.0 if cond.get("regime") == "ranging" else 0.0,
                    1.0 if cond.get("regime") == "high_volatility" else 0.0,
                ]
                
                # Target: optimal size multiplier (what size would have been best)
                # For now, use actual size if profitable, smaller if loss
                actual_mult = trade["position_size"] / base_size
                pnl_pct = trade["outcome"]["pnl_percent"]
                
                # If profitable, this size was good (or could be larger)
                # If loss, smaller size would be better
                if pnl_pct > 0:
                    target_mult = min(1.5, actual_mult * 1.1)  # Slightly larger
                else:
                    target_mult = max(0.3, actual_mult * 0.9)  # Slightly smaller
                
                X.append(features)
                y.append(target_mult)
            
            if len(X) < 20:
                return False
            
            # Train XGBoost regression
            X_array = np.array(X)
            y_array = np.array(y)
            
            self.ml_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.ml_model.fit(X_array, y_array)
            self.ml_model_trained = True
            
            logger.info("Trained ML model for position sizing")
            self._save_state()
            return True
            
        except Exception as e:
            logger.warning(f"ML model training failed: {e}")
            return False
    
    def predict_optimal_size_ml(self, conditions: Dict) -> Optional[float]:
        """
        Use ML model to predict optimal position size.
        
        Args:
            conditions: Current market conditions
        
        Returns:
            Predicted optimal multiplier or None
        """
        if not self.ml_model_trained or self.ml_model is None:
            return None
        
        try:
            # Prepare features same as training
            features = [
                conditions.get("volatility_ratio", 1.0),
                conditions.get("confidence", 0.5),
                conditions.get("confluence", 0) / 10.0,
                conditions.get("streak_multiplier", 1.0),
                1.0 if conditions.get("regime") == "trending_up" else 0.0,
                1.0 if conditions.get("regime") == "trending_down" else 0.0,
                1.0 if conditions.get("regime") == "ranging" else 0.0,
                1.0 if conditions.get("regime") == "high_volatility" else 0.0,
            ]
            
            X = np.array([features])
            pred = self.ml_model.predict(X)[0]
            
            # Enforce bounds
            return max(0.3, min(1.5, float(pred)))
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        complete_count = sum(1 for e in self.trade_entries if e["trade_id"] in self.trade_outcomes)
        
        return {
            "total_entries": len(self.trade_entries),
            "complete_trades": complete_count,
            "learned_conditions": len(self.optimal_sizes),
            "ml_model_trained": self.ml_model_trained
        }
