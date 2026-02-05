"""
Deep Performance Analyzer - AI Self-Reflection System
Analyzes trading patterns, identifies weaknesses, and generates insights
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

import config


class DeepAnalyzer:
    """
    Deep analysis of trading performance
    Identifies patterns, weaknesses, and improvement opportunities
    """
    
    ANALYSIS_FILE = os.path.join(config.DATA_DIR, "deep_analysis.json")
    
    def __init__(self):
        self.trade_history = []
        self.insights = []
        self.patterns = {}
        self.weaknesses = []
        self.strengths = []
        
        # Load existing analysis
        self._load_analysis()
    
    def _load_analysis(self):
        """Load saved analysis"""
        try:
            if os.path.exists(self.ANALYSIS_FILE):
                with open(self.ANALYSIS_FILE, 'r') as f:
                    data = json.load(f)
                
                self.insights = data.get("insights", [])
                self.patterns = data.get("patterns", {})
                self.weaknesses = data.get("weaknesses", [])
                self.strengths = data.get("strengths", [])
                
        except Exception as e:
            print(f"[ANALYZER] Error loading analysis: {e}")
    
    def _save_analysis(self):
        """Save analysis to file"""
        try:
            data = {
                "insights": self.insights[-50:],
                "patterns": self.patterns,
                "weaknesses": self.weaknesses,
                "strengths": self.strengths,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.ANALYSIS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[ANALYZER] Error saving analysis: {e}")
    
    def load_trade_history(self) -> List[Dict]:
        """Load trade history from file"""
        try:
            path = os.path.join(config.DATA_DIR, "trade_history.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.trade_history = json.load(f)
            return self.trade_history
        except Exception:
            return []
    
    def analyze_all(self) -> Dict:
        """
        Run comprehensive analysis on all trading data
        
        Returns:
            Complete analysis report
        """
        self.load_trade_history()
        
        if not self.trade_history:
            return {"status": "no_data"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_trades": 0,
            "performance_summary": {},
            "time_analysis": {},
            "regime_analysis": {},
            "exit_type_analysis": {},
            "pattern_insights": [],
            "identified_weaknesses": [],
            "identified_strengths": [],
            "improvement_suggestions": []
        }
        
        # Run all analyses
        report["performance_summary"] = self._analyze_overall_performance()
        report["time_analysis"] = self._analyze_by_time()
        report["regime_analysis"] = self._analyze_by_regime()
        report["exit_type_analysis"] = self._analyze_by_exit_type()
        report["pattern_insights"] = self._identify_patterns()
        report["identified_weaknesses"] = self._identify_weaknesses()
        report["identified_strengths"] = self._identify_strengths()
        report["improvement_suggestions"] = self._generate_suggestions()
        report["total_trades"] = len([t for t in self.trade_history if t.get("type") == "SELL"])
        
        # Store for future reference
        self.insights.append({
            "timestamp": report["timestamp"],
            "summary": {
                "win_rate": report["performance_summary"].get("win_rate", 0),
                "total_profit": report["performance_summary"].get("total_profit", 0),
                "weaknesses_count": len(report["identified_weaknesses"]),
                "strengths_count": len(report["identified_strengths"])
            }
        })
        
        self.weaknesses = report["identified_weaknesses"]
        self.strengths = report["identified_strengths"]
        
        self._save_analysis()
        
        return report
    
    def _analyze_overall_performance(self) -> Dict:
        """Analyze overall trading performance"""
        sells = [t for t in self.trade_history if t.get("type") == "SELL"]
        
        if not sells:
            return {}
        
        wins = [t for t in sells if t.get("pnl", 0) > 0]
        losses = [t for t in sells if t.get("pnl", 0) <= 0]
        
        total_profit = sum(t.get("pnl", 0) for t in wins)
        total_loss = sum(t.get("pnl", 0) for t in losses)
        
        pnl_values = [t.get("pnl_percent", 0) for t in sells]
        
        return {
            "total_trades": len(sells),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "net_profit": round(total_profit + total_loss, 2),
            "avg_win": round(total_profit / len(wins), 2) if wins else 0,
            "avg_loss": round(total_loss / len(losses), 2) if losses else 0,
            "profit_factor": round(abs(total_profit / total_loss), 2) if total_loss != 0 else 0,
            "best_trade": round(max(pnl_values), 2) if pnl_values else 0,
            "worst_trade": round(min(pnl_values), 2) if pnl_values else 0,
            "avg_trade": round(statistics.mean(pnl_values), 2) if pnl_values else 0,
            "std_dev": round(statistics.stdev(pnl_values), 2) if len(pnl_values) > 1 else 0
        }
    
    def _analyze_by_time(self) -> Dict:
        """Analyze performance by time of day and day of week"""
        sells = [t for t in self.trade_history if t.get("type") == "SELL"]
        
        by_hour = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
        by_day = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
        
        for trade in sells:
            try:
                time_str = trade.get("time", "")
                trade_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                hour = trade_time.hour
                day = trade_time.strftime("%A")

                pnl = trade.get("pnl", 0)
                is_win = pnl > 0

                by_hour[hour]["wins" if is_win else "losses"] += 1
                by_hour[hour]["pnl"] += pnl

                by_day[day]["wins" if is_win else "losses"] += 1
                by_day[day]["pnl"] += pnl

            except (ValueError, TypeError, KeyError):
                continue
        
        # Find best/worst hours
        hour_stats = []
        for hour, data in by_hour.items():
            total = data["wins"] + data["losses"]
            if total >= 2:
                hour_stats.append({
                    "hour": hour,
                    "win_rate": round(data["wins"] / total * 100, 1),
                    "total_pnl": round(data["pnl"], 2),
                    "trades": total
                })
        
        hour_stats.sort(key=lambda x: x["win_rate"], reverse=True)
        
        # Find best/worst days
        day_stats = []
        for day, data in by_day.items():
            total = data["wins"] + data["losses"]
            if total >= 2:
                day_stats.append({
                    "day": day,
                    "win_rate": round(data["wins"] / total * 100, 1),
                    "total_pnl": round(data["pnl"], 2),
                    "trades": total
                })
        
        day_stats.sort(key=lambda x: x["win_rate"], reverse=True)
        
        return {
            "best_hours": hour_stats[:3] if hour_stats else [],
            "worst_hours": hour_stats[-3:] if hour_stats else [],
            "best_days": day_stats[:2] if day_stats else [],
            "worst_days": day_stats[-2:] if day_stats else [],
            "by_hour": dict(by_hour),
            "by_day": dict(by_day)
        }
    
    def _analyze_by_regime(self) -> Dict:
        """Analyze performance by market regime"""
        sells = [t for t in self.trade_history if t.get("type") == "SELL"]
        
        by_regime = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
        
        for trade in sells:
            regime = trade.get("regime", "unknown")
            pnl = trade.get("pnl", 0)
            is_win = pnl > 0
            
            by_regime[regime]["wins" if is_win else "losses"] += 1
            by_regime[regime]["pnl"] += pnl
        
        regime_stats = []
        for regime, data in by_regime.items():
            total = data["wins"] + data["losses"]
            if total >= 2:
                regime_stats.append({
                    "regime": regime,
                    "win_rate": round(data["wins"] / total * 100, 1),
                    "total_pnl": round(data["pnl"], 2),
                    "trades": total,
                    "avg_pnl": round(data["pnl"] / total, 2)
                })
        
        regime_stats.sort(key=lambda x: x["win_rate"], reverse=True)
        
        return {
            "by_regime": regime_stats,
            "best_regime": regime_stats[0] if regime_stats else None,
            "worst_regime": regime_stats[-1] if regime_stats else None
        }
    
    def _analyze_by_exit_type(self) -> Dict:
        """Analyze performance by exit type"""
        sells = [t for t in self.trade_history if t.get("type") == "SELL"]
        
        by_exit = defaultdict(lambda: {"count": 0, "wins": 0, "losses": 0, "pnl": 0})
        
        for trade in sells:
            exit_type = trade.get("exit_type", "unknown")
            pnl = trade.get("pnl", 0)
            is_win = pnl > 0
            
            by_exit[exit_type]["count"] += 1
            by_exit[exit_type]["wins" if is_win else "losses"] += 1
            by_exit[exit_type]["pnl"] += pnl
        
        exit_stats = []
        for exit_type, data in by_exit.items():
            total = data["count"]
            if total >= 1:
                exit_stats.append({
                    "exit_type": exit_type,
                    "count": total,
                    "win_rate": round(data["wins"] / total * 100, 1) if total > 0 else 0,
                    "total_pnl": round(data["pnl"], 2),
                    "avg_pnl": round(data["pnl"] / total, 2)
                })
        
        exit_stats.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "by_exit_type": exit_stats,
            "most_common": exit_stats[0] if exit_stats else None,
            "most_profitable": max(exit_stats, key=lambda x: x["avg_pnl"]) if exit_stats else None,
            "least_profitable": min(exit_stats, key=lambda x: x["avg_pnl"]) if exit_stats else None
        }
    
    def _identify_patterns(self) -> List[Dict]:
        """Identify recurring patterns in trades"""
        patterns = []
        sells = [t for t in self.trade_history if t.get("type") == "SELL"]
        
        if len(sells) < 5:
            return patterns
        
        # Pattern 1: Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in sells:
            if trade.get("pnl", 0) <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        if max_consecutive_losses >= 3:
            patterns.append({
                "pattern": "consecutive_losses",
                "description": f"Had {max_consecutive_losses} consecutive losses",
                "severity": "high" if max_consecutive_losses >= 5 else "medium",
                "recommendation": "Consider reducing position size after 2 consecutive losses"
            })
        
        # Pattern 2: Stop loss frequency
        stop_losses = [t for t in sells if t.get("exit_type") == "stop_loss"]
        stop_loss_rate = len(stop_losses) / len(sells) * 100
        
        if stop_loss_rate > 40:
            patterns.append({
                "pattern": "frequent_stop_losses",
                "description": f"Stop loss rate is {stop_loss_rate:.1f}%",
                "severity": "high",
                "recommendation": "Consider widening stop loss or improving entry timing"
            })
        
        # Pattern 3: Small wins, big losses
        wins = [t for t in sells if t.get("pnl", 0) > 0]
        losses = [t for t in sells if t.get("pnl", 0) <= 0]
        
        if wins and losses:
            avg_win = statistics.mean(t.get("pnl_percent", 0) for t in wins)
            avg_loss = abs(statistics.mean(t.get("pnl_percent", 0) for t in losses))
            
            if avg_loss > avg_win * 1.5:
                patterns.append({
                    "pattern": "asymmetric_risk_reward",
                    "description": f"Average loss ({avg_loss:.2f}%) is much larger than average win ({avg_win:.2f}%)",
                    "severity": "high",
                    "recommendation": "Tighten stop loss or let winners run longer"
                })
        
        # Pattern 4: Time-based patterns
        morning_trades = [t for t in sells if self._get_hour(t) in range(6, 12)]
        afternoon_trades = [t for t in sells if self._get_hour(t) in range(12, 18)]
        night_trades = [t for t in sells if self._get_hour(t) in range(18, 24) or self._get_hour(t) in range(0, 6)]
        
        def win_rate(trades):
            if not trades:
                return 0
            return len([t for t in trades if t.get("pnl", 0) > 0]) / len(trades) * 100
        
        time_rates = {
            "morning": (win_rate(morning_trades), len(morning_trades)),
            "afternoon": (win_rate(afternoon_trades), len(afternoon_trades)),
            "night": (win_rate(night_trades), len(night_trades))
        }
        
        # Find significant time differences
        for period, (rate, count) in time_rates.items():
            if count >= 3 and rate < 40:
                patterns.append({
                    "pattern": f"weak_{period}_performance",
                    "description": f"{period.title()} win rate is only {rate:.1f}% ({count} trades)",
                    "severity": "medium",
                    "recommendation": f"Consider avoiding trades during {period} hours"
                })
        
        self.patterns = {p["pattern"]: p for p in patterns}
        return patterns
    
    def _get_hour(self, trade: Dict) -> int:
        """Extract hour from trade timestamp"""
        try:
            time_str = trade.get("time", "")
            trade_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            return trade_time.hour
        except (ValueError, TypeError, AttributeError):
            return 12
    
    def _identify_weaknesses(self) -> List[Dict]:
        """Identify trading weaknesses"""
        weaknesses = []
        sells = [t for t in self.trade_history if t.get("type") == "SELL"]
        
        if len(sells) < 5:
            return weaknesses
        
        perf = self._analyze_overall_performance()
        
        # Weakness 1: Low win rate
        if perf.get("win_rate", 0) < 50:
            weaknesses.append({
                "weakness": "low_win_rate",
                "description": f"Win rate is {perf['win_rate']}% (below 50%)",
                "impact": "high",
                "suggestion": "Improve entry signal quality - require higher confluence"
            })
        
        # Weakness 2: Poor risk/reward
        if perf.get("profit_factor", 0) < 1.2:
            weaknesses.append({
                "weakness": "poor_risk_reward",
                "description": f"Profit factor is {perf['profit_factor']} (should be > 1.5)",
                "impact": "high",
                "suggestion": "Adjust profit targets and stop losses"
            })
        
        # Weakness 3: High drawdown potential
        if perf.get("worst_trade", 0) < -2:
            weaknesses.append({
                "weakness": "large_losses",
                "description": f"Worst trade was {perf['worst_trade']}%",
                "impact": "medium",
                "suggestion": "Consider tighter stop losses or position sizing"
            })
        
        # Weakness 4: Inconsistency
        if perf.get("std_dev", 0) > 1.5:
            weaknesses.append({
                "weakness": "inconsistent_results",
                "description": f"High result variance (std dev: {perf['std_dev']}%)",
                "impact": "medium",
                "suggestion": "Focus on more consistent setups"
            })
        
        return weaknesses
    
    def _identify_strengths(self) -> List[Dict]:
        """Identify trading strengths"""
        strengths = []
        sells = [t for t in self.trade_history if t.get("type") == "SELL"]
        
        if len(sells) < 5:
            return strengths
        
        perf = self._analyze_overall_performance()
        regime = self._analyze_by_regime()
        time = self._analyze_by_time()
        
        # Strength 1: Good win rate
        if perf.get("win_rate", 0) >= 60:
            strengths.append({
                "strength": "high_win_rate",
                "description": f"Win rate is {perf['win_rate']}% (above 60%)",
                "leverage": "Can be more aggressive with position sizing"
            })
        
        # Strength 2: Strong profit factor
        if perf.get("profit_factor", 0) >= 1.8:
            strengths.append({
                "strength": "excellent_risk_reward",
                "description": f"Profit factor is {perf['profit_factor']} (above 1.8)",
                "leverage": "Strategy is mathematically sound"
            })
        
        # Strength 3: Best performing regime
        if regime.get("best_regime") and regime["best_regime"]["win_rate"] >= 70:
            best = regime["best_regime"]
            strengths.append({
                "strength": "regime_expertise",
                "description": f"Excellent in {best['regime']} regime ({best['win_rate']}% win rate)",
                "leverage": f"Increase position size during {best['regime']} regime"
            })
        
        # Strength 4: Best performing time
        if time.get("best_hours") and time["best_hours"][0]["win_rate"] >= 70:
            best_hour = time["best_hours"][0]
            strengths.append({
                "strength": "time_expertise",
                "description": f"Strong performance at hour {best_hour['hour']} ({best_hour['win_rate']}%)",
                "leverage": f"Focus trading around hour {best_hour['hour']}"
            })
        
        return strengths
    
    def _generate_suggestions(self) -> List[Dict]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        for weakness in self.weaknesses:
            suggestions.append({
                "priority": "high" if weakness["impact"] == "high" else "medium",
                "area": weakness["weakness"],
                "suggestion": weakness["suggestion"],
                "expected_impact": f"Fix: {weakness['description']}"
            })
        
        for strength in self.strengths:
            suggestions.append({
                "priority": "medium",
                "area": strength["strength"],
                "suggestion": strength["leverage"],
                "expected_impact": f"Leverage: {strength['description']}"
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 2))
        
        return suggestions
    
    def get_quick_insights(self) -> Dict:
        """Get quick insights without full analysis"""
        self.load_trade_history()
        sells = [t for t in self.trade_history if t.get("type") == "SELL"]
        
        if len(sells) < 3:
            return {"status": "insufficient_data", "trades": len(sells)}
        
        recent = sells[:10]  # Last 10 trades
        wins = len([t for t in recent if t.get("pnl", 0) > 0])
        
        return {
            "recent_win_rate": round(wins / len(recent) * 100, 1),
            "total_trades": len(sells),
            "last_trade_profitable": sells[0].get("pnl", 0) > 0 if sells else None,
            "weaknesses_count": len(self.weaknesses),
            "strengths_count": len(self.strengths)
        }


# Global analyzer instance
_analyzer = None

def get_analyzer() -> DeepAnalyzer:
    """Get global analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = DeepAnalyzer()
    return _analyzer


def analyze_trades() -> Dict:
    """Convenience function to run full analysis"""
    return get_analyzer().analyze_all()
