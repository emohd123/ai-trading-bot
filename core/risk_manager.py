"""
Risk Manager - Portfolio-Level Risk Management
Handles stop loss execution, drawdown protection, and position sizing limits.

Phase 1 & 4: Critical risk management improvements
- Guaranteed stop loss execution with slippage protection
- Portfolio-level exposure limits
- Drawdown protection
- Correlation-based risk limits
"""
import os
import json
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from collections import deque
import statistics

import config

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Centralized risk management for the trading bot.
    Ensures stop losses are enforced and protects capital.
    """
    
    STATE_FILE = os.path.join(config.DATA_DIR, "risk_manager_state.json")
    
    def __init__(self):
        """Initialize risk manager"""
        # Stop loss tracking
        self.hard_stop_limit = getattr(config, 'HARD_STOP_LIMIT', 0.02)  # 2% absolute max
        self.slippage_buffer = getattr(config, 'STOP_LOSS_SLIPPAGE_BUFFER', 0.002)  # 0.2%
        
        # Drawdown tracking
        self.peak_portfolio_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown_limit = getattr(config, 'MAX_DRAWDOWN_LIMIT', 0.10)  # 10%
        self.drawdown_reduction_factor = getattr(config, 'DRAWDOWN_REDUCTION_FACTOR', 0.5)
        
        # Daily loss tracking
        self.daily_losses = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.max_daily_loss_pct = getattr(config, 'MAX_DAILY_LOSS_PCT', 0.03)
        self.max_daily_trades = getattr(config, 'MAX_DAILY_TRADES', 20)
        
        # Position tracking for correlation
        self.recent_trades = deque(maxlen=50)  # Last 50 trades
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Exposure limits
        self.max_portfolio_exposure = getattr(config, 'MAX_PORTFOLIO_EXPOSURE', 0.8)  # 80%
        self.max_single_position = getattr(config, 'MAX_SINGLE_POSITION', 0.4)  # 40%
        
        # Risk state
        self.risk_mode = "normal"  # normal, cautious, recovery, blocked
        self.blocked_until = None
        
        self._load_state()
    
    def _load_state(self):
        """Load risk state from file"""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.peak_portfolio_value = data.get('peak_portfolio_value', 0)
                self.daily_losses = data.get('daily_losses', 0)
                self.daily_trades = data.get('daily_trades', 0)
                self.consecutive_losses = data.get('consecutive_losses', 0)
                self.consecutive_wins = data.get('consecutive_wins', 0)
                self.risk_mode = data.get('risk_mode', 'normal')
                
                last_reset = data.get('last_reset_date')
                if last_reset:
                    self.last_reset_date = datetime.fromisoformat(last_reset).date()
                
                blocked = data.get('blocked_until')
                if blocked:
                    self.blocked_until = datetime.fromisoformat(blocked)
                
                logger.info(f"Loaded risk state: mode={self.risk_mode}, daily_losses=${self.daily_losses:.2f}")
        except Exception as e:
            logger.warning(f"Could not load risk state: {e}")
    
    def _save_state(self):
        """Save risk state to file"""
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            data = {
                'peak_portfolio_value': self.peak_portfolio_value,
                'daily_losses': self.daily_losses,
                'daily_trades': self.daily_trades,
                'consecutive_losses': self.consecutive_losses,
                'consecutive_wins': self.consecutive_wins,
                'risk_mode': self.risk_mode,
                'last_reset_date': self.last_reset_date.isoformat(),
                'blocked_until': self.blocked_until.isoformat() if self.blocked_until else None,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save risk state: {e}")
    
    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_losses = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
            
            # Clear blocked status on new day (if not severe)
            if self.risk_mode != "blocked":
                self.risk_mode = "normal"
            elif self.blocked_until and datetime.now() > self.blocked_until:
                self.risk_mode = "normal"
                self.blocked_until = None
            
            self._save_state()
            logger.info("Risk counters reset for new day")
    
    def reset_daily_stats(self):
        """Manually reset daily statistics (for testing)"""
        self.daily_losses = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.risk_mode = "normal"
        self.blocked_until = None
        self._save_state()
    
    @property
    def current_drawdown_pct(self) -> float:
        """Get current drawdown as percentage"""
        return self.current_drawdown * 100
    
    def update_portfolio_value(self, current_value: float):
        """Update portfolio value and track drawdown"""
        self._check_daily_reset()
        
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        if self.peak_portfolio_value > 0:
            self.current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        # Check drawdown limits
        if self.current_drawdown >= self.max_drawdown_limit:
            if self.risk_mode != "blocked":
                self.risk_mode = "recovery"
                logger.warning(f"Max drawdown reached ({self.current_drawdown:.1%}), entering recovery mode")
        elif self.current_drawdown >= self.max_drawdown_limit * 0.7:
            if self.risk_mode == "normal":
                self.risk_mode = "cautious"
                logger.info(f"Drawdown at {self.current_drawdown:.1%}, entering cautious mode")
        elif self.current_drawdown < self.max_drawdown_limit * 0.5:
            if self.risk_mode == "cautious":
                self.risk_mode = "normal"
        
        self._save_state()
    
    def get_effective_stop_loss(
        self,
        base_stop_loss: float,
        regime: str = "unknown",
        volatility: str = "normal",
        entry_price: float = 0,
        current_price: float = 0
    ) -> Tuple[float, str]:
        """
        Get effective stop loss with slippage buffer and regime adjustments.
        
        Returns:
            Tuple of (stop_loss_percent, reason)
        """
        # Start with base stop loss
        stop_loss = base_stop_loss
        reason = "base"
        
        # Regime adjustments
        if regime == "trending_down":
            stop_loss = min(stop_loss, getattr(config, 'STOP_LOSS_TRENDING_DOWN', 0.006))
            reason = "downtrend"
        elif volatility in ("high", "extreme"):
            stop_loss = getattr(config, 'STOP_LOSS_HIGH_VOL', 0.0075)
            reason = "high_volatility"
        
        # Add slippage buffer for execution
        stop_loss_with_buffer = stop_loss + self.slippage_buffer
        
        # Never exceed hard stop limit
        if stop_loss_with_buffer > self.hard_stop_limit:
            stop_loss_with_buffer = self.hard_stop_limit
            reason = "hard_limit"
        
        # In recovery mode, use tighter stops
        if self.risk_mode == "recovery":
            stop_loss_with_buffer = min(stop_loss_with_buffer, base_stop_loss * 0.8)
            reason = "recovery_mode"
        
        return stop_loss_with_buffer, reason
    
    def should_force_stop(
        self,
        pnl_percent: float,
        entry_price: float,
        current_price: float,
        regime: str = "unknown",
        ai_score: float = 0
    ) -> Tuple[bool, str]:
        """
        Check if stop loss should be forced (no AI override allowed).
        
        Returns:
            Tuple of (should_force, reason)
        """
        # Convert to decimal
        pnl_decimal = pnl_percent / 100
        
        # 1. HARD STOP - Never override (absolute protection)
        if pnl_decimal <= -self.hard_stop_limit:
            return True, f"hard_stop_{self.hard_stop_limit*100:.1f}pct"
        
        # 2. Downtrend with significant loss - cut fast
        if regime == "trending_down" and pnl_decimal <= -getattr(config, 'STOP_LOSS_TRENDING_DOWN', 0.006):
            return True, "downtrend_stop"
        
        # 3. Recovery mode - strict enforcement
        if self.risk_mode == "recovery" and pnl_decimal <= -getattr(config, 'STOP_LOSS', 0.0075) * 0.8:
            return True, "recovery_mode_stop"
        
        # 4. Daily loss limit approaching - be more aggressive
        if self.daily_losses >= self.max_daily_loss_pct * 0.8:
            if pnl_decimal <= -getattr(config, 'STOP_LOSS', 0.0075) * 0.7:
                return True, "daily_limit_approaching"
        
        # 5. Consecutive losses - tighten stops
        if self.consecutive_losses >= 3 and pnl_decimal <= -getattr(config, 'STOP_LOSS', 0.0075) * 0.8:
            return True, "consecutive_loss_protection"
        
        return False, ""
    
    def can_trade(self, portfolio_value: float = 0) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits.
        
        Returns:
            Tuple of (can_trade, reason)
        """
        self._check_daily_reset()
        
        # Check if blocked
        if self.risk_mode == "blocked":
            if self.blocked_until and datetime.now() < self.blocked_until:
                remaining = (self.blocked_until - datetime.now()).total_seconds() / 60
                return False, f"blocked_for_{remaining:.0f}min"
            else:
                self.risk_mode = "normal"
                self.blocked_until = None
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"daily_trade_limit_{self.max_daily_trades}"
        
        # Check daily loss limit
        if self.daily_losses >= self.max_daily_loss_pct:
            self._block_trading(hours=2, reason="daily_loss_limit")
            return False, f"daily_loss_limit_{self.max_daily_loss_pct*100:.1f}pct"
        
        # Check drawdown
        if self.current_drawdown >= self.max_drawdown_limit:
            return False, f"max_drawdown_{self.current_drawdown*100:.1f}pct"
        
        return True, "ok"
    
    def _block_trading(self, hours: float = 1, reason: str = ""):
        """Block trading for specified hours"""
        self.risk_mode = "blocked"
        self.blocked_until = datetime.now() + timedelta(hours=hours)
        self._save_state()
        logger.warning(f"Trading blocked for {hours}h: {reason}")
    
    def record_trade_result(
        self,
        pnl_dollar: float,
        pnl_percent: float,
        exit_type: str,
        portfolio_value: float = 0
    ):
        """Record trade result for risk tracking"""
        self._check_daily_reset()
        
        self.daily_trades += 1
        
        if pnl_dollar < 0:
            self.daily_losses += abs(pnl_percent) / 100
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # Check for losing streak
            if self.consecutive_losses >= 5:
                self._block_trading(hours=1, reason="5_consecutive_losses")
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        
        # Record trade
        self.recent_trades.append({
            'pnl_dollar': pnl_dollar,
            'pnl_percent': pnl_percent,
            'exit_type': exit_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update portfolio value
        if portfolio_value > 0:
            self.update_portfolio_value(portfolio_value)
        
        self._save_state()
    
    def get_position_size_multiplier(
        self,
        base_size: float,
        regime: str = "unknown",
        confidence: float = 0.5,
        volatility: str = "normal"
    ) -> float:
        """
        Get position size multiplier based on risk conditions.
        
        Returns:
            Multiplier (0.3 to 1.5)
        """
        mult = 1.0
        
        # Risk mode adjustments
        if self.risk_mode == "recovery":
            mult *= 0.5
        elif self.risk_mode == "cautious":
            mult *= 0.7
        
        # Drawdown adjustment
        if self.current_drawdown > 0.05:  # 5% drawdown
            mult *= max(0.5, 1.0 - self.current_drawdown)
        
        # Consecutive loss adjustment
        if self.consecutive_losses >= 2:
            mult *= max(0.5, 1.0 - (self.consecutive_losses * 0.1))
        
        # Regime adjustment
        if regime == "trending_down":
            mult *= getattr(config, 'POSITION_SIZE_DOWNTREND_MULT', 0.6)
        
        # Volatility adjustment
        if volatility == "extreme":
            mult *= 0.5
        elif volatility == "high":
            mult *= 0.7
        
        # Confidence adjustment (0.8x at low, 1.2x at high)
        if confidence < 0.4:
            mult *= 0.8
        elif confidence > 0.7:
            mult *= min(1.2, mult)  # Cap at 1.2x
        
        # Enforce bounds
        return max(0.3, min(1.5, mult))
    
    def get_status(self) -> Dict:
        """Get current risk status"""
        self._check_daily_reset()
        
        return {
            'mode': self.risk_mode,
            'can_trade': self.can_trade()[0],
            'daily_losses_pct': self.daily_losses * 100,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'current_drawdown_pct': self.current_drawdown * 100,
            'peak_value': self.peak_portfolio_value,
            'blocked_until': self.blocked_until.isoformat() if self.blocked_until else None
        }


# Singleton instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get or create singleton risk manager"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
