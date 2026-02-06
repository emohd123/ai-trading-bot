"""
Performance Tracker - Advanced Metrics and Analytics
Tracks Sharpe ratio, Sortino ratio, MAE/MFE, drawdown, and more.

Phase 5: Monitoring & Analytics
"""
import os
import json
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import deque
import statistics
import math

import config

logger = logging.getLogger(__name__)

# Optional numpy for advanced calculations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class PerformanceTracker:
    """
    Tracks advanced performance metrics for trading analysis.
    """
    
    STATE_FILE = os.path.join(config.DATA_DIR, "performance_metrics.json")
    
    def __init__(self):
        """Initialize performance tracker"""
        # Trade tracking
        self.trades = []  # List of completed trades
        self.equity_curve = []  # {timestamp, value, pnl}
        
        # Return series for ratio calculations
        self.returns = deque(maxlen=365)  # Daily returns (1 year)
        self.hourly_returns = deque(maxlen=24*30)  # Hourly returns (30 days)
        
        # Peak tracking for drawdown
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_duration = 0  # hours
        self.drawdown_start = None
        
        # MAE/MFE tracking (Maximum Adverse/Favorable Excursion)
        self.mae_values = []  # Max drawdown during each trade
        self.mfe_values = []  # Max profit during each trade
        
        # Performance by category
        self.performance_by_regime = {}
        self.performance_by_hour = {h: [] for h in range(24)}
        self.performance_by_day = {d: [] for d in range(7)}  # 0=Monday
        self.performance_by_exit = {}
        
        # Initial capital tracking
        self.initial_capital = 0.0
        self.start_date = None
        
        self._load_state()
    
    def _load_state(self):
        """Load performance state from file"""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.trades = data.get('trades', [])[-500:]  # Keep last 500
                self.equity_curve = data.get('equity_curve', [])[-1000:]
                self.peak_equity = data.get('peak_equity', 0)
                self.max_drawdown = data.get('max_drawdown', 0)
                self.initial_capital = data.get('initial_capital', 0)
                self.performance_by_regime = data.get('performance_by_regime', {})
                self.performance_by_exit = data.get('performance_by_exit', {})
                
                start = data.get('start_date')
                if start:
                    self.start_date = datetime.fromisoformat(start)
                
                # Rebuild return series
                for eq in self.equity_curve[-365:]:
                    if eq.get('return_pct'):
                        self.returns.append(eq['return_pct'])
                
                logger.info(f"Loaded performance data: {len(self.trades)} trades")
        except Exception as e:
            logger.warning(f"Could not load performance state: {e}")
    
    def _save_state(self):
        """Save performance state to file"""
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            data = {
                'trades': self.trades[-500:],
                'equity_curve': self.equity_curve[-1000:],
                'peak_equity': self.peak_equity,
                'max_drawdown': self.max_drawdown,
                'initial_capital': self.initial_capital,
                'start_date': self.start_date.isoformat() if self.start_date else None,
                'performance_by_regime': self.performance_by_regime,
                'performance_by_exit': self.performance_by_exit,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save performance state: {e}")
    
    def set_initial_capital(self, capital: float):
        """Set initial capital for return calculations"""
        if self.initial_capital == 0:
            self.initial_capital = capital
            self.current_equity = capital
            self.peak_equity = capital
            self.start_date = datetime.now()
            self._save_state()
    
    def update_equity(self, current_value: float):
        """Update equity curve with current portfolio value"""
        now = datetime.now()
        
        # Calculate return
        if self.current_equity > 0:
            return_pct = (current_value - self.current_equity) / self.current_equity
        else:
            return_pct = 0
        
        # Update equity
        self.current_equity = current_value
        
        # Update peak and drawdown
        if current_value > self.peak_equity:
            self.peak_equity = current_value
            self.drawdown_start = None
        else:
            if self.peak_equity > 0:
                current_dd = (self.peak_equity - current_value) / self.peak_equity
                if current_dd > self.max_drawdown:
                    self.max_drawdown = current_dd
                
                if self.drawdown_start is None:
                    self.drawdown_start = now
                else:
                    dd_duration = (now - self.drawdown_start).total_seconds() / 3600
                    if dd_duration > self.max_drawdown_duration:
                        self.max_drawdown_duration = dd_duration
        
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': now.isoformat(),
            'value': current_value,
            'peak': self.peak_equity,
            'drawdown_pct': (self.peak_equity - current_value) / self.peak_equity if self.peak_equity > 0 else 0,
            'return_pct': return_pct
        })
        
        # Add to returns
        self.returns.append(return_pct)
        self.hourly_returns.append(return_pct)
    
    def record_trade(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl_dollar: float,
        pnl_percent: float,
        exit_type: str,
        regime: str = "unknown",
        duration_minutes: float = 0,
        highest_price: float = 0,
        lowest_price: float = 0,
        entry_time: str = None
    ):
        """Record a completed trade with detailed metrics"""
        now = datetime.now()
        
        # Calculate MAE/MFE
        if highest_price > 0 and lowest_price > 0 and entry_price and entry_price > 0:
            mfe = ((highest_price - entry_price) / entry_price) * 100  # Max favorable
            mae = ((entry_price - lowest_price) / entry_price) * 100  # Max adverse
        else:
            mfe = max(0, pnl_percent)
            mae = max(0, -pnl_percent)
        
        self.mae_values.append(mae)
        self.mfe_values.append(mfe)
        
        trade = {
            'timestamp': now.isoformat(),
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl_dollar': pnl_dollar,
            'pnl_percent': pnl_percent,
            'exit_type': exit_type,
            'regime': regime,
            'duration_minutes': duration_minutes,
            'mae': mae,
            'mfe': mfe,
            'hour': now.hour,
            'day_of_week': now.weekday()
        }
        
        self.trades.append(trade)
        
        # Update performance by category
        # By regime
        if regime not in self.performance_by_regime:
            self.performance_by_regime[regime] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        self.performance_by_regime[regime]['trades'] += 1
        if pnl_dollar > 0:
            self.performance_by_regime[regime]['wins'] += 1
        self.performance_by_regime[regime]['total_pnl'] += pnl_dollar
        
        # By exit type
        if exit_type not in self.performance_by_exit:
            self.performance_by_exit[exit_type] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        self.performance_by_exit[exit_type]['trades'] += 1
        if pnl_dollar > 0:
            self.performance_by_exit[exit_type]['wins'] += 1
        self.performance_by_exit[exit_type]['total_pnl'] += pnl_dollar
        
        # By hour and day
        self.performance_by_hour[now.hour].append(pnl_percent)
        self.performance_by_day[now.weekday()].append(pnl_percent)
        
        self._save_state()
    
    def calculate_sharpe_ratio(self, period_days: int = 30, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe Ratio.
        Sharpe = (avg_return - risk_free) / std_return
        
        Args:
            period_days: Number of days to calculate over
            risk_free_rate: Annual risk-free rate (default 0)
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(self.returns) < 10:
            return 0.0
        
        returns = list(self.returns)[-period_days:]
        
        if len(returns) < 2:
            return 0.0
        
        try:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualize (assuming daily returns)
            daily_rf = risk_free_rate / 365
            sharpe = (avg_return - daily_rf) / std_return
            
            # Annualize Sharpe
            annualized_sharpe = sharpe * math.sqrt(365)
            
            return round(annualized_sharpe, 3)
        except Exception:
            return 0.0
    
    def calculate_sortino_ratio(self, period_days: int = 30, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation only).
        Sortino = (avg_return - risk_free) / downside_deviation
        
        Returns:
            Annualized Sortino ratio
        """
        if len(self.returns) < 10:
            return 0.0
        
        returns = list(self.returns)[-period_days:]
        
        if len(returns) < 2:
            return 0.0
        
        try:
            avg_return = statistics.mean(returns)
            daily_rf = risk_free_rate / 365
            
            # Calculate downside deviation (only negative returns)
            negative_returns = [r for r in returns if r < daily_rf]
            
            if len(negative_returns) < 2:
                return float('inf') if avg_return > 0 else 0.0
            
            downside_dev = statistics.stdev(negative_returns)
            
            if downside_dev == 0:
                return float('inf') if avg_return > 0 else 0.0
            
            sortino = (avg_return - daily_rf) / downside_dev
            
            # Annualize
            annualized_sortino = sortino * math.sqrt(365)
            
            return round(annualized_sortino, 3)
        except Exception:
            return 0.0
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio.
        Calmar = Annual Return / Max Drawdown
        
        Returns:
            Calmar ratio
        """
        if self.max_drawdown == 0 or self.initial_capital == 0:
            return 0.0
        
        if not self.start_date:
            return 0.0
        
        try:
            # Calculate annualized return
            days_trading = (datetime.now() - self.start_date).days
            if days_trading < 1:
                return 0.0
            
            total_return = (self.current_equity - self.initial_capital) / self.initial_capital
            annual_return = total_return * (365 / days_trading)
            
            calmar = annual_return / self.max_drawdown
            
            return round(calmar, 3)
        except Exception:
            return 0.0
    
    def calculate_profit_factor(self) -> float:
        """
        Calculate Profit Factor.
        Profit Factor = Gross Profit / Gross Loss
        
        Returns:
            Profit factor (>1 is profitable)
        """
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t['pnl_dollar'] for t in self.trades if t['pnl_dollar'] > 0)
        gross_loss = abs(sum(t['pnl_dollar'] for t in self.trades if t['pnl_dollar'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return round(gross_profit / gross_loss, 3)
    
    def calculate_expectancy(self) -> float:
        """
        Calculate Trade Expectancy (expected value per trade).
        Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        
        Returns:
            Expected profit per trade
        """
        if not self.trades:
            return 0.0
        
        wins = [t['pnl_dollar'] for t in self.trades if t['pnl_dollar'] > 0]
        losses = [abs(t['pnl_dollar']) for t in self.trades if t['pnl_dollar'] < 0]
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        loss_rate = 1 - win_rate
        
        avg_win = statistics.mean(wins) if wins else 0
        avg_loss = statistics.mean(losses) if losses else 0
        
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        return round(expectancy, 4)
    
    def get_mae_mfe_analysis(self) -> Dict:
        """
        Analyze MAE/MFE patterns.
        Helps understand if stops are too tight or profits taken too early.
        """
        if len(self.mae_values) < 5:
            return {'status': 'insufficient_data'}
        
        avg_mae = statistics.mean(self.mae_values)
        avg_mfe = statistics.mean(self.mfe_values)
        max_mae = max(self.mae_values)
        max_mfe = max(self.mfe_values)
        
        # Calculate efficiency: how much of MFE was captured
        if avg_mfe > 0:
            recent_trades = self.trades[-20:]
            captured_profits = [t['pnl_percent'] for t in recent_trades if t['pnl_percent'] > 0]
            avg_captured = statistics.mean(captured_profits) if captured_profits else 0
            efficiency = avg_captured / avg_mfe if avg_mfe > 0 else 0
        else:
            efficiency = 0
        
        # Recommendations
        recommendations = []
        
        stop_loss_config = getattr(config, 'STOP_LOSS', 0.0075) * 100
        if avg_mae < stop_loss_config * 0.5:
            recommendations.append("Stop loss may be too tight - trades hitting stop before recovering")
        
        if avg_mfe > avg_mae * 2 and efficiency < 0.5:
            recommendations.append("Profits being taken too early - consider trailing stops")
        
        return {
            'avg_mae': round(avg_mae, 3),
            'avg_mfe': round(avg_mfe, 3),
            'max_mae': round(max_mae, 3),
            'max_mfe': round(max_mfe, 3),
            'mfe_mae_ratio': round(avg_mfe / avg_mae, 2) if avg_mae > 0 else 0,
            'profit_efficiency': round(efficiency * 100, 1),
            'recommendations': recommendations
        }
    
    def get_performance_by_hour(self) -> Dict:
        """Get performance breakdown by hour of day"""
        result = {}
        for hour, pnls in self.performance_by_hour.items():
            if pnls:
                result[hour] = {
                    'trades': len(pnls),
                    'avg_pnl': round(statistics.mean(pnls), 3),
                    'win_rate': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1)
                }
        return result
    
    def get_performance_by_day(self) -> Dict:
        """Get performance breakdown by day of week"""
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        result = {}
        for day_num, pnls in self.performance_by_day.items():
            if pnls:
                result[days[day_num]] = {
                    'trades': len(pnls),
                    'avg_pnl': round(statistics.mean(pnls), 3),
                    'win_rate': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1)
                }
        return result
    
    def get_full_metrics(self) -> Dict:
        """Get all performance metrics"""
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['pnl_dollar'] > 0)
        losing_trades = total_trades - winning_trades
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades * 100
            total_pnl = sum(t['pnl_dollar'] for t in self.trades)
            avg_win = statistics.mean([t['pnl_dollar'] for t in self.trades if t['pnl_dollar'] > 0]) if winning_trades > 0 else 0
            avg_loss = abs(statistics.mean([t['pnl_dollar'] for t in self.trades if t['pnl_dollar'] < 0])) if losing_trades > 0 else 0
        else:
            win_rate = 0
            total_pnl = 0
            avg_win = 0
            avg_loss = 0
        
        # Calculate return
        if self.initial_capital > 0:
            total_return = (self.current_equity - self.initial_capital) / self.initial_capital * 100
        else:
            total_return = 0
        
        return {
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 1),
                'total_pnl': round(total_pnl, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'total_return_pct': round(total_return, 2)
            },
            'ratios': {
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'sortino_ratio': self.calculate_sortino_ratio(),
                'calmar_ratio': self.calculate_calmar_ratio(),
                'profit_factor': self.calculate_profit_factor(),
                'expectancy': self.calculate_expectancy()
            },
            'drawdown': {
                'current_drawdown_pct': round((self.peak_equity - self.current_equity) / self.peak_equity * 100, 2) if self.peak_equity > 0 else 0,
                'max_drawdown_pct': round(self.max_drawdown * 100, 2),
                'max_drawdown_duration_hours': round(self.max_drawdown_duration, 1),
                'peak_equity': round(self.peak_equity, 2),
                'current_equity': round(self.current_equity, 2)
            },
            'mae_mfe': self.get_mae_mfe_analysis(),
            'by_regime': self.performance_by_regime,
            'by_exit_type': self.performance_by_exit,
            'by_hour': self.get_performance_by_hour(),
            'by_day': self.get_performance_by_day(),
            'equity_curve_points': len(self.equity_curve)
        }
    
    def get_equity_curve_data(self, limit: int = 100) -> List[Dict]:
        """Get equity curve data for charting"""
        return self.equity_curve[-limit:]


# Singleton instance
_performance_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get or create singleton performance tracker"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker
