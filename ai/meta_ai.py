"""
Meta AI Controller - The Brain of the Autonomous Trading System
Coordinates all subsystems, sets goals, and drives self-improvement
"""

import logging
import json

logger = logging.getLogger(__name__)
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Import all subsystems
from ai.backtester import Backtester, BacktestResult
from ai.strategy_evolver import StrategyEvolver, quick_evolve
from ai.param_optimizer import ParameterOptimizer, get_optimizer, auto_tune
from ai.deep_analyzer import DeepAnalyzer, get_analyzer, analyze_trades
from core.self_healing import SelfHealer
import config


class AIState(Enum):
    """States the Meta AI can be in"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    EVOLVING = "evolving"
    LEARNING = "learning"
    TRADING = "trading"


class Goal:
    """Represents an AI goal with target and progress tracking"""
    
    def __init__(self, name: str, target: float, metric: str, 
                 priority: int = 1, deadline_days: int = None):
        self.name = name
        self.target = target
        self.metric = metric
        self.priority = priority
        self.current_value = 0
        self.created_at = datetime.now()
        self.deadline = datetime.now() + timedelta(days=deadline_days) if deadline_days else None
        self.achieved = False
        self.progress_history = []
    
    def update_progress(self, value: float):
        """Update current progress towards goal"""
        self.current_value = value
        self.progress_history.append({
            "timestamp": datetime.now().isoformat(),
            "value": value
        })
        
        # Check if achieved
        if value >= self.target:
            self.achieved = True
    
    def get_progress_percent(self) -> float:
        """Get progress as percentage"""
        if self.target == 0:
            return 100 if self.achieved else 0
        return min(100, (self.current_value / self.target) * 100)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "target": self.target,
            "current": self.current_value,
            "metric": self.metric,
            "progress_percent": round(self.get_progress_percent(), 1),
            "achieved": self.achieved,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None
        }


class MetaAI:
    """
    Meta AI Controller - Coordinates all autonomous systems
    
    This is the "brain" that:
    - Sets and tracks goals
    - Decides when to optimize, evolve, or analyze
    - Coordinates all subsystems
    - Maintains awareness of overall system state
    - Drives continuous improvement
    """
    
    STATE_FILE = os.path.join("data", "meta_ai_state.json")
    
    def __init__(self):
        # State
        self.state = AIState.IDLE
        self.awareness = {}  # Current understanding of system state
        self.goals = []
        self.action_log = []
        
        # Subsystems
        self.analyzer = None
        self.optimizer = None
        self.evolver = None
        self.backtester = None
        self.healer = None
        
        # Schedules
        self.last_analysis = None
        self.last_optimization = None
        self.last_evolution = None
        self.last_health_check = None
        
        # Intervals (in hours)
        self.analysis_interval = 4
        self.optimization_interval = 6
        self.evolution_interval = 24
        self.health_check_interval = 1
        
        # Self-improvement tracking
        self.improvement_log = []
        self.version = 1
        
        # Thread control
        self._running = False
        self._thread = None
        
        # Load state
        self._load_state()
        
        # Initialize subsystems
        self._init_subsystems()
        
        # Set default goals
        self._init_default_goals()
    
    def _init_subsystems(self):
        """Initialize all subsystems"""
        try:
            self.analyzer = get_analyzer()
            self.optimizer = get_optimizer()
            self.evolver = StrategyEvolver(population_size=8)
            self.backtester = Backtester()
            logger.info("META-AI: Subsystems initialized")
        except Exception as e:
            logger.error("META-AI: Error initializing subsystems: %s", e)
    
    def _init_default_goals(self):
        """Set default improvement goals"""
        if not self.goals:
            self.goals = [
                Goal("win_rate", 65, "percentage", priority=1),
                Goal("profit_factor", 1.5, "ratio", priority=1),
                Goal("max_drawdown", 10, "percentage_max", priority=2),
                Goal("daily_profit", 0.5, "percentage", priority=2),
                Goal("trades_learned", 50, "count", priority=3)
            ]
    
    def _load_state(self):
        """Load saved state"""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                
                # Restore timestamps
                for key in ['last_analysis', 'last_optimization', 'last_evolution', 'last_health_check']:
                    val = state.get(key)
                    if val:
                        setattr(self, key, datetime.fromisoformat(val))
                
                self.action_log = state.get("action_log", [])[-100:]
                self.improvement_log = state.get("improvement_log", [])[-50:]
                self.version = state.get("version", 1)
                self.awareness = state.get("awareness", {})
                
                # Restore goals
                for goal_data in state.get("goals", []):
                    goal = Goal(
                        goal_data["name"],
                        goal_data["target"],
                        goal_data["metric"],
                        goal_data.get("priority", 1)
                    )
                    goal.current_value = goal_data.get("current", 0)
                    goal.achieved = goal_data.get("achieved", False)
                    self.goals.append(goal)
                
                logger.info("META-AI: State loaded: Version %s", self.version)
                
        except Exception as e:
            logger.warning("META-AI: Error loading state: %s", e)
    
    def _save_state(self):
        """Save current state"""
        try:
            state = {
                "version": self.version,
                "state": self.state.value,
                "awareness": self.awareness,
                "goals": [g.to_dict() for g in self.goals],
                "action_log": self.action_log[-100:],
                "improvement_log": self.improvement_log[-50:],
                "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
                "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
                "last_evolution": self.last_evolution.isoformat() if self.last_evolution else None,
                "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(self.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.warning("META-AI: Error saving state: %s", e)
    
    def log_action(self, action: str, details: Dict = None):
        """Log an action taken by the AI"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details or {},
            "state": self.state.value
        }
        self.action_log.append(entry)
        logger.info("META-AI: %s", action)
    
    def update_awareness(self):
        """Update AI's awareness of current system state"""
        self.state = AIState.ANALYZING
        
        try:
            # Load trade history (use config.DATA_DIR so paths match dashboard)
            trades = []
            trade_history_path = os.path.join(config.DATA_DIR, "trade_history.json")
            if os.path.exists(trade_history_path):
                with open(trade_history_path, 'r', encoding='utf-8') as f:
                    trades = json.load(f)
            
            sells = [t for t in trades if t.get("type") == "SELL"]
            wins = [t for t in sells if t.get("pnl", 0) > 0]
            
            # Load AI learning state
            learning = {}
            learning_path = os.path.join(config.DATA_DIR, "ai_learning.json")
            if os.path.exists(learning_path):
                with open(learning_path, 'r', encoding='utf-8') as f:
                    learning = json.load(f)
            
            # Calculate current metrics
            self.awareness = {
                "timestamp": datetime.now().isoformat(),
                "total_trades": len(sells),
                "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
                "total_profit": sum(t.get("pnl", 0) for t in sells),
                "trades_learned": len(learning.get("trade_history", [])),
                "current_streak": learning.get("win_streak", 0) - learning.get("loss_streak", 0),
                "indicator_accuracy": learning.get("indicator_accuracy", {}),
                "best_indicator": max(learning.get("indicator_accuracy", {}).items(), 
                                     key=lambda x: x[1])[0] if learning.get("indicator_accuracy") else None,
                "worst_indicator": min(learning.get("indicator_accuracy", {}).items(),
                                      key=lambda x: x[1])[0] if learning.get("indicator_accuracy") else None,
                "recent_performance": self._get_recent_performance(sells),
                "system_health": "healthy"
            }
            
            # ML accuracy and retrain status (optional)
            try:
                from ai.ml_predictor import MLPredictor
                _ml = MLPredictor()
                acc = _ml.get_accuracy()
                should_r, reason = _ml.should_retrain()
                self.awareness["ml_accuracy"] = acc
                self.awareness["ml_needs_retrain"] = should_r
                self.awareness["ml_retrain_reason"] = reason or ""
            except Exception:
                self.awareness["ml_accuracy"] = {}
                self.awareness["ml_needs_retrain"] = False
                self.awareness["ml_retrain_reason"] = ""
            
            # Update goal progress
            self._update_goal_progress()
            
            self.log_action("Updated awareness", {"win_rate": self.awareness["win_rate"]})
            
        except Exception as e:
            self.awareness["error"] = str(e)
            logger.warning("META-AI: Error updating awareness: %s", e)
        
        self.state = AIState.IDLE
        self._save_state()
    
    def _get_recent_performance(self, sells: List[Dict], n: int = 10) -> Dict:
        """Get performance of last n trades"""
        recent = sells[:n] if len(sells) >= n else sells
        if not recent:
            return {"trades": 0}
        
        wins = [t for t in recent if t.get("pnl", 0) > 0]
        return {
            "trades": len(recent),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(recent) * 100, 1),
            "total_pnl": round(sum(t.get("pnl", 0) for t in recent), 2)
        }
    
    def _update_goal_progress(self):
        """Update progress on all goals"""
        for goal in self.goals:
            if goal.metric == "percentage" and goal.name == "win_rate":
                goal.update_progress(self.awareness.get("win_rate", 0))
            elif goal.metric == "count" and goal.name == "trades_learned":
                goal.update_progress(self.awareness.get("trades_learned", 0))
            # Add more goal types as needed
    
    def decide_next_action(self) -> str:
        """Decide what action to take next based on current state"""
        now = datetime.now()
        
        # Priority 1: Health check if overdue
        if self.last_health_check is None or \
           (now - self.last_health_check).total_seconds() / 3600 >= self.health_check_interval:
            return "health_check"
        
        # Priority 2: Analysis if overdue
        if self.last_analysis is None or \
           (now - self.last_analysis).total_seconds() / 3600 >= self.analysis_interval:
            return "analyze"
        
        # Priority 3: ML retrain if accuracy dropped or requested
        if self.awareness.get("ml_needs_retrain"):
            return "ml_retrain"
        
        # Priority 4: Optimization if overdue and have enough data
        if self.awareness.get("trades_learned", 0) >= 10:
            if self.last_optimization is None or \
               (now - self.last_optimization).total_seconds() / 3600 >= self.optimization_interval:
                return "optimize"
        
        # Priority 5: Evolution if overdue and have enough data
        if self.awareness.get("trades_learned", 0) >= 20:
            if self.last_evolution is None or \
               (now - self.last_evolution).total_seconds() / 3600 >= self.evolution_interval:
                return "evolve"
        
        # Default: Check goals
        return "check_goals"
    
    def execute_action(self, action: str) -> Dict:
        """Execute a decided action"""
        result = {"action": action, "status": "started"}
        
        try:
            if action == "health_check":
                result = self._do_health_check()
            elif action == "analyze":
                result = self._do_analysis()
            elif action == "ml_retrain":
                result = self._do_ml_retrain()
            elif action == "optimize":
                result = self._do_optimization()
            elif action == "evolve":
                result = self._do_evolution()
            elif action == "check_goals":
                result = self._check_goals()
            else:
                result["status"] = "unknown_action"
                
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error("META-AI: Error executing %s: %s", action, e)
        
        self._save_state()
        return result
    
    def _do_health_check(self) -> Dict:
        """Run health check"""
        self.state = AIState.ANALYZING
        self.log_action("Running health check")
        
        # Update awareness
        self.update_awareness()
        
        self.last_health_check = datetime.now()
        self.state = AIState.IDLE
        
        return {
            "action": "health_check",
            "status": "complete",
            "awareness": self.awareness
        }
    
    def _do_analysis(self) -> Dict:
        """Run deep analysis"""
        self.state = AIState.ANALYZING
        self.log_action("Running deep analysis")
        
        if self.analyzer:
            analysis = self.analyzer.analyze_all()
            
            self.awareness["last_analysis"] = {
                "weaknesses": len(analysis.get("identified_weaknesses", [])),
                "strengths": len(analysis.get("identified_strengths", [])),
                "suggestions": analysis.get("improvement_suggestions", [])[:3]
            }
        
        self.last_analysis = datetime.now()
        self.state = AIState.IDLE
        
        return {
            "action": "analyze",
            "status": "complete",
            "insights": self.awareness.get("last_analysis", {})
        }
    
    def _do_ml_retrain(self) -> Dict:
        """Request ML retrain (writes request file; worker or cron runs training)."""
        self.state = AIState.LEARNING
        reason = self.awareness.get("ml_retrain_reason", "meta_ai") or "meta_ai"
        self.log_action("Requesting ML retrain", {"reason": reason})
        try:
            from ai.ml_training import request_ml_retrain
            request_ml_retrain(reason="meta_ai")
        except Exception as e:
            logger.warning("META-AI: Could not request ML retrain: %s", e)
        self.state = AIState.IDLE
        return {
            "action": "ml_retrain",
            "status": "complete",
            "reason": reason
        }
    
    def _do_optimization(self) -> Dict:
        """Run parameter optimization"""
        self.state = AIState.OPTIMIZING
        self.log_action("Running parameter optimization")
        
        result = {"action": "optimize", "status": "complete", "adjustments": []}
        
        if self.optimizer:
            opt_result = self.optimizer.auto_optimize()
            result["adjustments"] = opt_result.get("adjustments", [])
            
            if result["adjustments"]:
                self.improvement_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "optimization",
                    "changes": len(result["adjustments"])
                })
        
        self.last_optimization = datetime.now()
        self.state = AIState.IDLE
        
        return result
    
    def _do_evolution(self) -> Dict:
        """Run strategy evolution"""
        self.state = AIState.EVOLVING
        self.log_action("Running strategy evolution")
        
        result = {"action": "evolve", "status": "complete"}
        
        try:
            if self.evolver:
                # Load previous state
                self.evolver.load_state()
                
                # Run evolution (fewer generations for regular runs)
                best = self.evolver.run_evolution(generations=2, days=7)
                
                if best:
                    result["best_score"] = best.fitness
                    result["best_params"] = best.to_params()
                    
                    # If significantly better, apply parameters
                    if best.fitness > 50:  # Threshold for "good enough"
                        self._apply_evolved_params(best.to_params())
                        result["applied"] = True
                
                self.evolver.save_state()
                
        except Exception as e:
            result["error"] = str(e)
        
        self.last_evolution = datetime.now()
        self.state = AIState.IDLE
        
        return result
    
    def _apply_evolved_params(self, params: Dict):
        """Apply evolved parameters to config"""
        applied = []
        for key, value in params.items():
            if hasattr(config, key) and key != "INDICATOR_WEIGHTS":
                old = getattr(config, key)
                setattr(config, key, value)
                applied.append({"param": key, "old": old, "new": value})
        
        if applied:
            self.log_action("Applied evolved parameters", {"count": len(applied)})
            self.improvement_log.append({
                "timestamp": datetime.now().isoformat(),
                "type": "evolution",
                "changes": len(applied)
            })
    
    def _check_goals(self) -> Dict:
        """Check progress on goals and take action if needed"""
        result = {"action": "check_goals", "goals": []}
        
        for goal in self.goals:
            goal_status = {
                "name": goal.name,
                "progress": goal.get_progress_percent(),
                "achieved": goal.achieved
            }
            result["goals"].append(goal_status)
            
            # If goal is not being met, consider action
            if not goal.achieved and goal.get_progress_percent() < 50:
                if goal.name == "win_rate" and goal.priority == 1:
                    # Trigger optimization sooner
                    self.optimization_interval = max(2, self.optimization_interval - 1)
        
        return result
    
    def think(self) -> Dict:
        """
        Main thinking loop - decide and execute next action
        This is what gives the AI "awareness"
        """
        # Update awareness first
        self.update_awareness()
        
        # Decide what to do
        action = self.decide_next_action()
        
        # Execute
        result = self.execute_action(action)
        
        return {
            "awareness": self.awareness,
            "action_taken": action,
            "result": result,
            "goals": [g.to_dict() for g in self.goals],
            "next_actions": {
                "analysis_in": self._time_until("analysis"),
                "optimization_in": self._time_until("optimization"),
                "evolution_in": self._time_until("evolution")
            }
        }
    
    def _time_until(self, action: str) -> str:
        """Get time until next scheduled action"""
        now = datetime.now()
        
        if action == "analysis":
            last = self.last_analysis
            interval = self.analysis_interval
        elif action == "optimization":
            last = self.last_optimization
            interval = self.optimization_interval
        elif action == "evolution":
            last = self.last_evolution
            interval = self.evolution_interval
        else:
            return "unknown"
        
        if last is None:
            return "now"
        
        elapsed = (now - last).total_seconds() / 3600
        remaining = interval - elapsed
        
        if remaining <= 0:
            return "now"
        elif remaining < 1:
            return f"{int(remaining * 60)} minutes"
        else:
            return f"{remaining:.1f} hours"
    
    def start_autonomous_loop(self):
        """Start the autonomous thinking loop in background"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._autonomous_loop, daemon=True)
        self._thread.start()
        self.log_action("Started autonomous loop")
    
    def stop_autonomous_loop(self):
        """Stop the autonomous loop"""
        self._running = False
        self.log_action("Stopped autonomous loop")
    
    def _autonomous_loop(self):
        """Background loop for autonomous operation"""
        while self._running:
            try:
                # Think every 30 minutes
                self.think()
            except Exception as e:
                logger.error("META-AI: Error in autonomous loop: %s", e)
            
            # Sleep for 30 minutes
            for _ in range(30 * 60):
                if not self._running:
                    break
                time.sleep(1)
    
    def get_status(self) -> Dict:
        """Get current Meta AI status"""
        return {
            "state": self.state.value,
            "version": self.version,
            "awareness": self.awareness,
            "goals": [g.to_dict() for g in self.goals],
            "last_actions": self.action_log[-5:],
            "improvements": len(self.improvement_log),
            "autonomous_running": self._running,
            "schedules": {
                "analysis": self._time_until("analysis"),
                "optimization": self._time_until("optimization"),
                "evolution": self._time_until("evolution")
            }
        }
    
    def force_action(self, action: str) -> Dict:
        """Force execute a specific action"""
        self.log_action(f"Force executing: {action}")
        return self.execute_action(action)


# Global Meta AI instance
_meta_ai = None

def get_meta_ai() -> MetaAI:
    """Get global Meta AI instance"""
    global _meta_ai
    if _meta_ai is None:
        _meta_ai = MetaAI()
    return _meta_ai


def start_autonomous_ai():
    """Start the autonomous AI system"""
    ai = get_meta_ai()
    ai.start_autonomous_loop()
    return ai


def ai_think() -> Dict:
    """Trigger a thinking cycle"""
    return get_meta_ai().think()


def ai_status() -> Dict:
    """Get AI status"""
    return get_meta_ai().get_status()
