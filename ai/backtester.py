"""
Backtesting Engine - Test strategies on historical data
Allows the AI to evaluate strategies before using them live
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

from core.binance_client import BinanceClient
from ai.analyzer import TechnicalAnalyzer
from ai.ai_engine import AIEngine, Decision
from market.market_regime import RegimeDetector
import config


class BacktestResult:
    """Container for backtest results"""
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.initial_balance = 0
        self.final_balance = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.win_rate = 0
        self.profit_factor = 0
        self.avg_win = 0
        self.avg_loss = 0
        self.avg_trade_duration = 0
        self.best_trade = 0
        self.worst_trade = 0
        
    def to_dict(self) -> Dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "total_profit": round(self.total_profit, 2),
            "total_loss": round(self.total_loss, 2),
            "net_profit": round(self.total_profit - abs(self.total_loss), 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "best_trade": round(self.best_trade, 2),
            "worst_trade": round(self.worst_trade, 2),
            "avg_trade_duration_hours": round(self.avg_trade_duration, 1),
            "initial_balance": self.initial_balance,
            "final_balance": round(self.final_balance, 2),
            "return_pct": round((self.final_balance - self.initial_balance) / self.initial_balance * 100, 2)
        }
    
    def score(self) -> float:
        """Calculate a composite score for this backtest result"""
        # Higher is better
        score = 0
        
        # Win rate contribution (0-30 points)
        score += min(30, self.win_rate * 0.4)
        
        # Profit factor contribution (0-25 points)
        if self.profit_factor > 0:
            score += min(25, self.profit_factor * 10)
        
        # Sharpe ratio contribution (0-25 points)
        if self.sharpe_ratio > 0:
            score += min(25, self.sharpe_ratio * 10)
        
        # Drawdown penalty (0 to -20 points)
        score -= min(20, self.max_drawdown * 2)
        
        # Trade count bonus (need enough trades for significance)
        if self.total_trades >= 20:
            score += 10
        elif self.total_trades >= 10:
            score += 5
        
        # Return contribution (0-10 points)
        return_pct = (self.final_balance - self.initial_balance) / max(self.initial_balance, 1) * 100
        score += min(10, max(-10, return_pct))
        
        return max(0, score)


class Backtester:
    """
    Backtesting engine for testing trading strategies
    """
    
    def __init__(self):
        self.client = BinanceClient()
        self.analyzer = TechnicalAnalyzer()
        self.regime_detector = RegimeDetector()
        
        # Cache for historical data
        self._data_cache = {}
        
    def fetch_historical_data(
        self, 
        days: int = 30, 
        interval: str = "1h"
    ) -> pd.DataFrame:
        """Fetch historical price data for backtesting"""
        cache_key = f"{days}_{interval}"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        try:
            # Calculate how many candles we need
            if interval == "1h":
                limit = days * 24
            elif interval == "15m":
                limit = days * 24 * 4
            elif interval == "4h":
                limit = days * 6
            else:
                limit = days * 24
            
            limit = min(limit, 1000)  # Binance limit
            
            df = self.client.get_historical_klines(
                symbol=config.SYMBOL,
                interval=interval,
                limit=limit
            )
            
            if not df.empty:
                self._data_cache[cache_key] = df
                
            return df
            
        except Exception as e:
            print(f"[BACKTEST] Error fetching data: {e}")
            return pd.DataFrame()
    
    def run_backtest(
        self,
        strategy_params: Dict = None,
        days: int = 14,
        initial_balance: float = 100,
        trade_amount: float = 30
    ) -> BacktestResult:
        """
        Run a backtest with given strategy parameters
        
        Args:
            strategy_params: Override default config parameters
            days: Number of days to backtest
            initial_balance: Starting balance in USDT
            trade_amount: Amount per trade in USDT
        """
        result = BacktestResult()
        result.initial_balance = initial_balance
        
        # Fetch historical data
        df = self.fetch_historical_data(days=days, interval="1h")
        if df.empty or len(df) < 50:
            print("[BACKTEST] Not enough data for backtest")
            return result
        
        # Apply strategy parameters
        original_config = self._apply_params(strategy_params)
        
        # Create fresh AI engine for backtest
        ai_engine = AIEngine()
        
        # Simulation state
        balance = initial_balance
        position = None
        equity_curve = [balance]
        trades = []
        peak_equity = balance
        max_drawdown = 0
        
        # Walk through historical data
        for i in range(50, len(df)):
            # Get data window for analysis
            window = df.iloc[i-50:i+1].copy()
            current_price = float(window['close'].iloc[-1])
            current_time = window.index[-1]
            
            # Run technical analysis
            try:
                analysis = self.analyzer.analyze(window)
                if "error" in analysis:
                    continue
                    
                analysis["current_price"] = current_price
                
                # Get regime
                regime_data = self.regime_detector.detect_regime(window)
                
                # Get AI decision
                ai_decision = ai_engine.get_decision(analysis, window, regime_data)
                decision = ai_decision.get("decision", Decision.HOLD)
                ai_score = ai_decision.get("score", 0)
                
            except Exception as e:
                continue
            
            # Position management
            if position is None:
                # Look for entry
                if decision == Decision.BUY and balance >= trade_amount:
                    quantity = trade_amount / current_price
                    position = {
                        "entry_price": current_price,
                        "quantity": quantity,
                        "entry_time": current_time,
                        "entry_score": ai_score
                    }
                    balance -= trade_amount
                    
            else:
                # Check for exit
                entry_price = position["entry_price"]
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                should_exit = False
                exit_reason = ""
                
                # Profit target
                profit_target = getattr(config, 'PROFIT_TARGET', 0.015) * 100
                if pnl_pct >= profit_target:
                    should_exit = True
                    exit_reason = "profit_target"
                
                # Min profit (quick exit)
                min_profit = getattr(config, 'MIN_PROFIT', 0.005) * 100
                if pnl_pct >= min_profit and decision == Decision.SELL:
                    should_exit = True
                    exit_reason = "min_profit"
                
                # Stop loss
                stop_loss = getattr(config, 'STOP_LOSS', 0.01) * 100
                if pnl_pct <= -stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # AI sell signal
                if decision == Decision.SELL and pnl_pct > 0:
                    should_exit = True
                    exit_reason = "ai_signal"
                
                if should_exit:
                    exit_price = current_price
                    pnl = (exit_price - entry_price) * position["quantity"]
                    
                    trade = {
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": position["quantity"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "entry_time": position["entry_time"],
                        "exit_time": current_time,
                        "duration_hours": (current_time - position["entry_time"]).total_seconds() / 3600
                    }
                    trades.append(trade)
                    
                    balance += trade_amount + pnl
                    position = None
            
            # Update equity curve
            current_equity = balance
            if position:
                unrealized_pnl = (current_price - position["entry_price"]) * position["quantity"]
                current_equity += trade_amount + unrealized_pnl
            
            equity_curve.append(current_equity)
            
            # Track drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown = (peak_equity - current_equity) / peak_equity * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Close any remaining position at last price
        if position:
            exit_price = float(df['close'].iloc[-1])
            pnl = (exit_price - position["entry_price"]) * position["quantity"]
            trade = {
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "quantity": position["quantity"],
                "pnl": pnl,
                "pnl_pct": (exit_price - position["entry_price"]) / position["entry_price"] * 100,
                "exit_reason": "end_of_test",
                "entry_time": position["entry_time"],
                "exit_time": df.index[-1],
                "duration_hours": 0
            }
            trades.append(trade)
            balance += trade_amount + pnl
        
        # Restore original config
        self._restore_params(original_config)
        
        # Calculate results
        result.trades = trades
        result.equity_curve = equity_curve
        result.final_balance = balance
        result.total_trades = len(trades)
        result.max_drawdown = max_drawdown
        
        if trades:
            wins = [t for t in trades if t["pnl"] > 0]
            losses = [t for t in trades if t["pnl"] <= 0]
            
            result.winning_trades = len(wins)
            result.losing_trades = len(losses)
            result.win_rate = len(wins) / len(trades) * 100
            
            result.total_profit = sum(t["pnl"] for t in wins)
            result.total_loss = sum(t["pnl"] for t in losses)
            
            if result.total_loss != 0:
                result.profit_factor = abs(result.total_profit / result.total_loss)
            else:
                result.profit_factor = result.total_profit if result.total_profit > 0 else 0
            
            if wins:
                result.avg_win = result.total_profit / len(wins)
                result.best_trade = max(t["pnl_pct"] for t in wins)
            
            if losses:
                result.avg_loss = result.total_loss / len(losses)
                result.worst_trade = min(t["pnl_pct"] for t in losses)
            
            durations = [t["duration_hours"] for t in trades if t["duration_hours"] > 0]
            if durations:
                result.avg_trade_duration = sum(durations) / len(durations)
            
            # Calculate Sharpe ratio (simplified)
            returns = [t["pnl_pct"] for t in trades]
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    result.sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized
        
        return result
    
    def _apply_params(self, params: Dict) -> Dict:
        """Apply strategy parameters and return original values"""
        if not params:
            return {}
        
        original = {}
        for key, value in params.items():
            if hasattr(config, key):
                original[key] = getattr(config, key)
                setattr(config, key, value)
        
        return original
    
    def _restore_params(self, original: Dict):
        """Restore original config values"""
        for key, value in original.items():
            setattr(config, key, value)
    
    def compare_strategies(
        self, 
        strategies: List[Dict],
        days: int = 14
    ) -> List[Tuple[Dict, BacktestResult]]:
        """
        Compare multiple strategies and rank them
        
        Args:
            strategies: List of strategy parameter dicts
            days: Number of days to backtest
            
        Returns:
            List of (strategy, result) tuples sorted by score
        """
        results = []
        
        for i, strategy in enumerate(strategies):
            print(f"[BACKTEST] Testing strategy {i+1}/{len(strategies)}...")
            result = self.run_backtest(strategy_params=strategy, days=days)
            results.append((strategy, result))
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x[1].score(), reverse=True)
        
        return results
    
    def quick_test(self, params: Dict = None) -> Dict:
        """Run a quick backtest and return summary"""
        result = self.run_backtest(strategy_params=params, days=7)
        return {
            "score": result.score(),
            "metrics": result.to_dict()
        }


# Save/load backtest results
BACKTEST_HISTORY_FILE = os.path.join("data", "backtest_history.json")

def save_backtest_result(strategy: Dict, result: BacktestResult):
    """Save backtest result to history"""
    try:
        history = []
        if os.path.exists(BACKTEST_HISTORY_FILE):
            with open(BACKTEST_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "result": result.to_dict(),
            "score": result.score()
        }
        history.append(entry)
        
        # Keep last 100 results
        history = history[-100:]
        
        with open(BACKTEST_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        print(f"[BACKTEST] Error saving result: {e}")


def get_best_strategy() -> Optional[Dict]:
    """Get the best performing strategy from history"""
    try:
        if not os.path.exists(BACKTEST_HISTORY_FILE):
            return None
            
        with open(BACKTEST_HISTORY_FILE, 'r') as f:
            history = json.load(f)
        
        if not history:
            return None
        
        # Find highest scoring strategy
        best = max(history, key=lambda x: x.get("score", 0))
        return best.get("strategy")
        
    except Exception as e:
        print(f"[BACKTEST] Error loading best strategy: {e}")
        return None
