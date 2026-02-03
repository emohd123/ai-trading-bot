"""
Parameter Optimizer - Self-Tuning System
Automatically adjusts bot parameters based on performance
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics

import config


class ParameterOptimizer:
    """
    Self-tuning parameter optimizer
    Analyzes recent performance and adjusts parameters to improve results
    """
    
    OPTIMIZER_STATE_FILE = os.path.join("data", "optimizer_state.json")
    
    def __init__(self):
        # Parameters that can be tuned and their bounds
        self.tunable_params = {
            "PROFIT_TARGET": {"min": 0.003, "max": 0.03, "step": 0.001},
            "STOP_LOSS": {"min": 0.005, "max": 0.025, "step": 0.001},
            "MIN_PROFIT": {"min": 0.002, "max": 0.015, "step": 0.001},
            "BUY_THRESHOLD": {"min": 0.20, "max": 0.55, "step": 0.02},
            "SELL_THRESHOLD": {"min": -0.45, "max": -0.15, "step": 0.02},
            "MIN_CONFLUENCE_BUY": {"min": 3, "max": 8, "step": 1, "is_int": True},
            "MIN_CONFIDENCE_BUY": {"min": 0.30, "max": 0.65, "step": 0.05},
            "TRAILING_ACTIVATION": {"min": 0.005, "max": 0.02, "step": 0.002},
        }
        
        # Performance tracking
        self.param_history = {}  # {param_name: [{value, performance, timestamp}]}
        self.adjustment_log = []
        
        # Optimization state
        self.last_optimization = None
        self.optimization_interval_hours = 6  # How often to optimize
        
        # Load saved state
        self._load_state()
    
    def _load_state(self):
        """Load optimizer state from file"""
        try:
            if os.path.exists(self.OPTIMIZER_STATE_FILE):
                with open(self.OPTIMIZER_STATE_FILE, 'r') as f:
                    state = json.load(f)
                
                self.param_history = state.get("param_history", {})
                self.adjustment_log = state.get("adjustment_log", [])
                
                last_opt = state.get("last_optimization")
                if last_opt:
                    self.last_optimization = datetime.fromisoformat(last_opt)
                    
        except Exception as e:
            print(f"[OPTIMIZER] Error loading state: {e}")
    
    def _save_state(self):
        """Save optimizer state to file"""
        try:
            state = {
                "param_history": self.param_history,
                "adjustment_log": self.adjustment_log[-100:],  # Keep last 100
                "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(self.OPTIMIZER_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            print(f"[OPTIMIZER] Error saving state: {e}")
    
    def should_optimize(self) -> bool:
        """Check if it's time to run optimization"""
        if self.last_optimization is None:
            return True
        
        elapsed = (datetime.now() - self.last_optimization).total_seconds() / 3600
        return elapsed >= self.optimization_interval_hours
    
    def record_trade_performance(
        self, 
        is_win: bool,
        pnl_percent: float,
        exit_type: str,
        market_regime: str = None
    ):
        """
        Record trade performance for parameter analysis
        
        Args:
            is_win: Whether trade was profitable
            pnl_percent: Profit/loss percentage
            exit_type: How trade exited (profit_target, stop_loss, etc.)
            market_regime: Current market regime
        """
        # Get current parameter values
        current_params = self._get_current_params()
        
        # Record each parameter's performance
        for param_name, value in current_params.items():
            if param_name not in self.param_history:
                self.param_history[param_name] = []
            
            self.param_history[param_name].append({
                "value": value,
                "is_win": is_win,
                "pnl_percent": pnl_percent,
                "exit_type": exit_type,
                "regime": market_regime,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep last 100 records per parameter
            self.param_history[param_name] = self.param_history[param_name][-100:]
        
        self._save_state()
    
    def _get_current_params(self) -> Dict:
        """Get current values of tunable parameters"""
        params = {}
        for param_name in self.tunable_params:
            if hasattr(config, param_name):
                params[param_name] = getattr(config, param_name)
        return params
    
    def analyze_parameter(self, param_name: str) -> Dict:
        """
        Analyze performance for a specific parameter
        
        Returns:
            Analysis results including optimal value suggestion
        """
        history = self.param_history.get(param_name, [])
        
        if len(history) < 5:
            return {"status": "insufficient_data", "samples": len(history)}
        
        # Group by value ranges
        param_config = self.tunable_params[param_name]
        step = param_config["step"]
        
        value_groups = {}
        for record in history:
            # Round to step
            value = record["value"]
            if param_config.get("is_int"):
                rounded = int(value)
            else:
                rounded = round(value / step) * step
            
            if rounded not in value_groups:
                value_groups[rounded] = {"wins": 0, "losses": 0, "total_pnl": 0}
            
            if record["is_win"]:
                value_groups[rounded]["wins"] += 1
            else:
                value_groups[rounded]["losses"] += 1
            value_groups[rounded]["total_pnl"] += record["pnl_percent"]
        
        # Calculate win rate and avg PnL for each value
        value_scores = []
        for value, data in value_groups.items():
            total = data["wins"] + data["losses"]
            if total >= 2:  # Need at least 2 samples
                win_rate = data["wins"] / total * 100
                avg_pnl = data["total_pnl"] / total
                
                # Score: weighted combination of win rate and PnL
                score = (win_rate * 0.6) + (avg_pnl * 10 * 0.4)
                
                value_scores.append({
                    "value": value,
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "samples": total,
                    "score": score
                })
        
        if not value_scores:
            return {"status": "insufficient_data", "samples": len(history)}
        
        # Find best and current values
        current_value = getattr(config, param_name, None)
        best = max(value_scores, key=lambda x: x["score"])
        
        return {
            "status": "ok",
            "param_name": param_name,
            "current_value": current_value,
            "best_value": best["value"],
            "best_win_rate": best["win_rate"],
            "best_avg_pnl": best["avg_pnl"],
            "best_samples": best["samples"],
            "all_values": value_scores,
            "should_adjust": best["value"] != current_value and best["samples"] >= 3
        }
    
    def optimize_all(self) -> Dict:
        """
        Analyze and optimize all parameters
        
        Returns:
            Optimization results and recommendations
        """
        self.last_optimization = datetime.now()
        results = {
            "timestamp": datetime.now().isoformat(),
            "adjustments": [],
            "recommendations": [],
            "status": "ok"
        }
        
        for param_name in self.tunable_params:
            analysis = self.analyze_parameter(param_name)
            
            if analysis.get("status") != "ok":
                continue
            
            if analysis.get("should_adjust"):
                current = analysis["current_value"]
                best = analysis["best_value"]
                
                # Apply gradual adjustment (move 50% towards best value)
                if current is not None and best is not None:
                    param_config = self.tunable_params[param_name]
                    
                    if param_config.get("is_int"):
                        new_value = int(round((current + best) / 2))
                    else:
                        new_value = (current + best) / 2
                    
                    # Ensure within bounds
                    new_value = max(param_config["min"], min(param_config["max"], new_value))
                    
                    results["recommendations"].append({
                        "param": param_name,
                        "current": current,
                        "recommended": new_value,
                        "best_observed": best,
                        "win_rate_at_best": analysis["best_win_rate"],
                        "samples": analysis["best_samples"]
                    })
        
        self._save_state()
        return results
    
    def apply_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Apply recommended parameter adjustments
        
        Args:
            recommendations: List of recommendations from optimize_all()
            
        Returns:
            List of applied adjustments
        """
        applied = []
        
        for rec in recommendations:
            param_name = rec["param"]
            new_value = rec["recommended"]
            old_value = rec["current"]
            
            try:
                # Apply to config
                setattr(config, param_name, new_value)
                
                adjustment = {
                    "param": param_name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "timestamp": datetime.now().isoformat()
                }
                
                applied.append(adjustment)
                self.adjustment_log.append(adjustment)
                
                print(f"[OPTIMIZER] Adjusted {param_name}: {old_value} -> {new_value}")
                
            except Exception as e:
                print(f"[OPTIMIZER] Error adjusting {param_name}: {e}")
        
        self._save_state()
        return applied
    
    def auto_optimize(self) -> Dict:
        """
        Automatically analyze and apply optimizations
        
        Returns:
            Summary of optimizations performed
        """
        if not self.should_optimize():
            return {"status": "skipped", "reason": "Too soon since last optimization"}
        
        print("[OPTIMIZER] Running auto-optimization...")
        
        # Get recommendations
        results = self.optimize_all()
        
        if results["recommendations"]:
            # Apply recommendations
            applied = self.apply_recommendations(results["recommendations"])
            results["adjustments"] = applied
            
            print(f"[OPTIMIZER] Applied {len(applied)} adjustments")
        else:
            print("[OPTIMIZER] No adjustments needed")
        
        return results
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization state"""
        return {
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "total_adjustments": len(self.adjustment_log),
            "recent_adjustments": self.adjustment_log[-5:],
            "data_points": {
                param: len(history) 
                for param, history in self.param_history.items()
            }
        }
    
    def tune_for_regime(self, regime: str) -> Dict:
        """
        Get optimized parameters for a specific market regime
        
        Args:
            regime: Market regime (trending_up, trending_down, ranging, high_volatility)
            
        Returns:
            Optimized parameters for the regime
        """
        # Filter history by regime
        regime_history = {}
        for param_name, history in self.param_history.items():
            regime_records = [r for r in history if r.get("regime") == regime]
            if regime_records:
                regime_history[param_name] = regime_records
        
        # Analyze for regime-specific values
        regime_params = {}
        for param_name, records in regime_history.items():
            if len(records) >= 3:
                wins = [r for r in records if r["is_win"]]
                if wins:
                    # Use average of winning trade parameter values
                    avg_value = statistics.mean(r["value"] for r in wins)
                    regime_params[param_name] = avg_value
        
        return {
            "regime": regime,
            "optimized_params": regime_params,
            "data_points": len(regime_history.get(list(self.tunable_params.keys())[0], []))
        }


# Global optimizer instance
_optimizer = None

def get_optimizer() -> ParameterOptimizer:
    """Get global optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = ParameterOptimizer()
    return _optimizer


def record_trade(is_win: bool, pnl_percent: float, exit_type: str, regime: str = None):
    """Convenience function to record trade for optimization"""
    get_optimizer().record_trade_performance(is_win, pnl_percent, exit_type, regime)


def auto_tune() -> Dict:
    """Convenience function to run auto-optimization"""
    return get_optimizer().auto_optimize()
