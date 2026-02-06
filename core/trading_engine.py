"""
Trading Engine - Centralized Trading Logic
Unified trading execution with trailing profits, smart stops, and risk management.

Phase 1 & 2: Critical performance and code quality improvements
- Trailing profit targets (let winners run)
- Smart stop execution with guaranteed limits
- Position management
- Trade execution
"""
import logging
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime
from enum import Enum

import config
from core.risk_manager import get_risk_manager

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Trade exit reasons"""
    PROFIT_TARGET = "profit_target"
    TRAILING_PROFIT = "trailing_profit"
    MIN_PROFIT = "min_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    HARD_STOP = "hard_stop"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    AI_SELL = "ai_sell"
    BREAKEVEN = "breakeven_stop"


class TrailingProfitTracker:
    """
    Tracks trailing profit targets to let winners run.
    In strong trends, trails profit instead of taking fixed target.
    """
    
    def __init__(
        self,
        entry_price: float,
        activation_pct: float = 0.01,  # 1% to activate
        trail_pct: float = 0.005,       # 0.5% trail
        min_profit_pct: float = 0.0025  # 0.25% minimum
    ):
        self.entry_price = entry_price
        self.activation_pct = activation_pct
        self.trail_pct = trail_pct
        self.min_profit_pct = min_profit_pct
        
        self.highest_price = entry_price
        self.trailing_active = False
        self.trailing_stop_price = 0.0
        self.locked_profit_pct = 0.0
    
    def update(self, current_price: float, regime: str = "unknown", volatility: str = "normal") -> Dict:
        """
        Update trailing profit tracker with current price.
        
        Returns dict with:
        - trailing_active: bool
        - trailing_stop_price: float
        - locked_profit_pct: float
        - should_exit: bool
        - exit_reason: str
        """
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        # Adjust parameters based on regime
        if regime == "trending_up":
            # In uptrend, let profits run more
            effective_activation = self.activation_pct * 1.5  # 1.5% to activate
            effective_trail = self.trail_pct * 0.8  # Tighter trail
        elif regime == "trending_down":
            # In downtrend, take profits faster
            effective_activation = self.activation_pct * 0.5  # 0.5% to activate
            effective_trail = self.trail_pct * 1.5  # Wider trail (faster exit)
        elif volatility in ("high", "extreme"):
            # In high volatility, use tighter trail
            effective_activation = self.activation_pct * 0.5  # Faster activation
            effective_trail = self.trail_pct * 1.5  # Wider trail
        else:
            effective_activation = self.activation_pct
            effective_trail = self.trail_pct
        
        result = {
            'trailing_active': self.trailing_active,
            'trailing_stop_price': self.trailing_stop_price,
            'locked_profit_pct': self.locked_profit_pct,
            'should_exit': False,
            'exit_reason': ''
        }
        
        # Check if we should activate trailing
        if not self.trailing_active and pnl_pct >= effective_activation:
            self.trailing_active = True
            self.highest_price = current_price
            self.trailing_stop_price = current_price * (1 - effective_trail)
            self.locked_profit_pct = (self.trailing_stop_price - self.entry_price) / self.entry_price
            logger.info(f"Trailing profit activated at +{pnl_pct*100:.2f}%, stop at ${self.trailing_stop_price:,.2f}")
        
        # If trailing is active, update highest and stop
        if self.trailing_active:
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.trailing_stop_price = current_price * (1 - effective_trail)
                self.locked_profit_pct = (self.trailing_stop_price - self.entry_price) / self.entry_price
            
            # Check if trailing stop hit
            if current_price <= self.trailing_stop_price:
                result['should_exit'] = True
                result['exit_reason'] = 'trailing_profit'
        
        result['trailing_active'] = self.trailing_active
        result['trailing_stop_price'] = self.trailing_stop_price
        result['locked_profit_pct'] = self.locked_profit_pct * 100
        
        return result


class PositionManager:
    """
    Manages individual position tracking with trailing profits and stops.
    """
    
    def __init__(self, position_data: Dict):
        self.entry_price = position_data.get('entry_price', 0)
        self.quantity = position_data.get('quantity', 0)
        self.entry_time = position_data.get('time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.trade_id = position_data.get('trade_id', 0)
        self.regime_at_entry = position_data.get('regime', 'unknown')
        
        # Tracking
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        self.trailing_profit_tracker = None
        self.breakeven_activated = False
        self.stop_delay_count = 0
        
        # Initialize trailing profit tracker
        self._init_trailing_tracker()
    
    def _init_trailing_tracker(self):
        """Initialize trailing profit tracker with config values"""
        activation = getattr(config, 'TRAILING_ACTIVATION', 0.005)
        trail = getattr(config, 'TRAILING_STOP_PCT', 0.003)
        min_profit = getattr(config, 'MIN_PROFIT', 0.0025)
        
        self.trailing_profit_tracker = TrailingProfitTracker(
            entry_price=self.entry_price,
            activation_pct=activation,
            trail_pct=trail,
            min_profit_pct=min_profit
        )
    
    def check_exit(
        self,
        current_price: float,
        regime: str = "unknown",
        volatility: str = "normal",
        ai_score: float = 0,
        ai_decision: str = "HOLD"
    ) -> Tuple[bool, ExitReason, str]:
        """
        Check if position should exit.
        
        Returns:
            Tuple of (should_exit, reason, details)
        """
        risk_mgr = get_risk_manager()
        
        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # Update price tracking
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price:
            self.lowest_price = current_price
        
        # === 1. HARD STOP - Never override ===
        force_stop, force_reason = risk_mgr.should_force_stop(
            pnl_percent=pnl_pct,
            entry_price=self.entry_price,
            current_price=current_price,
            regime=regime,
            ai_score=ai_score
        )
        
        if force_stop:
            return True, ExitReason.HARD_STOP, f"Forced stop: {force_reason}"
        
        # === 2. QUICK PROFIT MODE ===
        # Exit quickly if profit target reached within time limit (for fast gainers)
        quick_profit_enabled = getattr(config, 'QUICK_PROFIT_ENABLED', True)
        if quick_profit_enabled:
            quick_profit_target = getattr(config, 'QUICK_PROFIT_TARGET', 0.005) * 100  # 0.5%
            quick_profit_time_limit = getattr(config, 'QUICK_PROFIT_TIME_LIMIT', 30)  # 30 minutes
            
            # Parse entry_time (may be string or datetime)
            if isinstance(self.entry_time, str):
                try:
                    entry_dt = datetime.strptime(self.entry_time, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    entry_dt = datetime.now()  # Fallback
            else:
                entry_dt = self.entry_time
            
            position_age_minutes = (datetime.now() - entry_dt).total_seconds() / 60
            if position_age_minutes < quick_profit_time_limit and pnl_pct >= quick_profit_target:
                return True, ExitReason.PROFIT_TARGET, f"Quick profit +{pnl_pct:.2f}% (reached in {position_age_minutes:.1f} min)"
        
        # === 3. PROFIT TARGET CHECK ===
        # In volatile markets, use lower targets
        if volatility in ("high", "extreme"):
            profit_target = getattr(config, 'PROFIT_TARGET_HIGH_VOL', 0.005) * 100
            min_profit = getattr(config, 'MIN_PROFIT_HIGH_VOL', 0.0015) * 100
        else:
            profit_target = getattr(config, 'PROFIT_TARGET', 0.015) * 100
            min_profit = getattr(config, 'MIN_PROFIT', 0.0025) * 100
        
        # In uptrend, let profits run more
        if regime == "trending_up" and pnl_pct > profit_target:
            # Check trailing profit
            trail_result = self.trailing_profit_tracker.update(current_price, regime, volatility)
            if trail_result['should_exit']:
                return True, ExitReason.TRAILING_PROFIT, f"Trailing profit +{trail_result['locked_profit_pct']:.2f}%"
            # Don't exit yet, let it run
        elif pnl_pct >= profit_target:
            return True, ExitReason.PROFIT_TARGET, f"Target reached +{pnl_pct:.2f}%"
        
        # Check minimum profit
        if pnl_pct >= min_profit:
            # In downtrend or high volatility, take min profit
            if regime == "trending_down" or volatility in ("high", "extreme"):
                return True, ExitReason.MIN_PROFIT, f"Quick profit +{pnl_pct:.2f}% ({regime})"
        
        # === 4. TRAILING PROFIT (for positions in profit) ===
        if pnl_pct > 0:
            trail_result = self.trailing_profit_tracker.update(current_price, regime, volatility)
            if trail_result['should_exit']:
                return True, ExitReason.TRAILING_PROFIT, f"Trailing profit +{trail_result['locked_profit_pct']:.2f}%"
        
        # === 5. BREAKEVEN STOP ===
        breakeven_pct = getattr(config, 'BREAKEVEN_ACTIVATION', 0.005) * 100
        breakeven_buffer = getattr(config, 'BREAKEVEN_BUFFER', 0.001)
        
        if not self.breakeven_activated and pnl_pct >= breakeven_pct:
            self.breakeven_activated = True
            logger.info(f"Breakeven activated at +{pnl_pct:.2f}%")
        
        if self.breakeven_activated:
            breakeven_price = self.entry_price * (1 + breakeven_buffer)
            if current_price <= breakeven_price:
                return True, ExitReason.BREAKEVEN, f"Breakeven stop at ${breakeven_price:,.2f}"
        
        # === 6. STOP LOSS CHECK ===
        # Min-hold: skip regular stop in first N minutes (hard stop already applied in section 1)
        if getattr(config, 'MIN_HOLD_ENABLED', False):
            # Parse entry_time (may be string or datetime)
            if isinstance(self.entry_time, str):
                try:
                    entry_dt = datetime.strptime(self.entry_time, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    entry_dt = datetime.now()  # Fallback
            else:
                entry_dt = self.entry_time
            
            age_min = (datetime.now() - entry_dt).total_seconds() / 60
            if age_min < getattr(config, 'MIN_HOLD_MINUTES', 20):
                pass  # Skip regular stop during min-hold window
            else:
                base_stop = getattr(config, 'STOP_LOSS', 0.01)
                effective_stop, stop_reason = risk_mgr.get_effective_stop_loss(
                    base_stop, regime, volatility, self.entry_price, current_price
                )
                if pnl_pct <= -effective_stop * 100:
                    return True, ExitReason.STOP_LOSS, f"Stop loss {stop_reason}: {pnl_pct:.2f}%"
        else:
            base_stop = getattr(config, 'STOP_LOSS', 0.01)
            effective_stop, stop_reason = risk_mgr.get_effective_stop_loss(
                base_stop, regime, volatility, self.entry_price, current_price
            )
            if pnl_pct <= -effective_stop * 100:
                return True, ExitReason.STOP_LOSS, f"Stop loss {stop_reason}: {pnl_pct:.2f}%"
        
        # === 6. TIME-BASED EXIT ===
        try:
            # Parse entry_time (may be string or datetime)
            if isinstance(self.entry_time, str):
                try:
                    entry_dt = datetime.strptime(self.entry_time, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    entry_dt = datetime.now()  # Fallback
            else:
                entry_dt = self.entry_time
            
            age_hours = (datetime.now() - entry_dt).total_seconds() / 3600
            
            max_hours = getattr(config, 'MAX_POSITION_AGE_HOURS', 4)
            stale_threshold = getattr(config, 'STALE_LOSS_THRESHOLD', -0.003) * 100
            
            if age_hours > max_hours and pnl_pct < stale_threshold:
                return True, ExitReason.TIME_EXIT, f"Stale position ({age_hours:.1f}h) with {pnl_pct:.2f}%"
        except Exception:
            pass
        
        # === 7. AI SELL SIGNAL ===
        # When in loss, require stronger bearish signal (SELL_THRESHOLD_IN_LOSS) to avoid locking small losses
        if ai_decision == "SELL" and pnl_pct < 0:
            sell_threshold = getattr(config, 'SELL_THRESHOLD_IN_LOSS', -0.35)
            if ai_score < sell_threshold:
                return True, ExitReason.AI_SELL, f"AI SELL signal ({ai_score:.2f})"
        
        return False, None, ""
    
    def get_status(self, current_price: float) -> Dict:
        """Get position status"""
        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        pnl_dollar = (current_price - self.entry_price) * self.quantity
        
        # Get trailing info
        trail_result = self.trailing_profit_tracker.update(current_price, "unknown", "normal")
        
        return {
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'trade_id': self.trade_id,
            'pnl_percent': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'breakeven_activated': self.breakeven_activated,
            'trailing_active': trail_result['trailing_active'],
            'trailing_stop_price': trail_result['trailing_stop_price'],
            'locked_profit_pct': trail_result['locked_profit_pct'],
            'entry_time': self.entry_time
        }


class TradingEngine:
    """
    Centralized trading engine for execution and position management.
    """
    
    def __init__(self, binance_client=None):
        self.client = binance_client
        self.positions: List[PositionManager] = []
        self.risk_manager = get_risk_manager()
        self.max_positions = getattr(config, 'MAX_POSITIONS', 2)
    
    def set_client(self, client):
        """Set Binance client"""
        self.client = client
    
    def load_positions(self, positions_data: List[Dict]):
        """Load existing positions"""
        self.positions = []
        for pos_data in positions_data:
            self.positions.append(PositionManager(pos_data))
    
    def can_open_position(self) -> Tuple[bool, str]:
        """Check if new position can be opened"""
        # Check risk manager
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            return False, reason
        
        # Check position limit
        if len(self.positions) >= self.max_positions:
            return False, f"max_positions_{self.max_positions}"
        
        return True, "ok"
    
    def calculate_position_size(
        self,
        base_amount: float,
        regime: str = "unknown",
        confidence: float = 0.5,
        volatility: str = "normal",
        learned_mult: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate position size with risk adjustments.
        
        Returns:
            Tuple of (adjusted_amount, details)
        """
        # Get risk-based multiplier
        risk_mult = self.risk_manager.get_position_size_multiplier(
            base_amount, regime, confidence, volatility
        )
        
        # Blend with learned multiplier if available
        blend_ratio = getattr(config, 'POSITION_SIZE_BLEND_RATIO', 0.7)
        
        if learned_mult is not None:
            effective_mult = (learned_mult * blend_ratio) + (risk_mult * (1 - blend_ratio))
        else:
            effective_mult = risk_mult
        
        # Apply multiplier
        adjusted_amount = base_amount * effective_mult
        
        # Enforce min/max bounds
        min_amount = base_amount * 0.3
        max_amount = base_amount * 1.5
        adjusted_amount = max(min_amount, min(max_amount, adjusted_amount))
        
        details = {
            'base_amount': base_amount,
            'risk_mult': risk_mult,
            'learned_mult': learned_mult,
            'effective_mult': effective_mult,
            'adjusted_amount': adjusted_amount,
            'risk_mode': self.risk_manager.risk_mode
        }
        
        return adjusted_amount, details
    
    def check_all_positions(
        self,
        current_price: float,
        regime: str = "unknown",
        volatility: str = "normal",
        ai_score: float = 0,
        ai_decision: str = "HOLD"
    ) -> List[Tuple[int, ExitReason, str]]:
        """
        Check all positions for exits.
        
        Returns:
            List of (position_index, exit_reason, details) for positions to close
        """
        exits = []
        
        for i, pos in enumerate(self.positions):
            should_exit, reason, details = pos.check_exit(
                current_price=current_price,
                regime=regime,
                volatility=volatility,
                ai_score=ai_score,
                ai_decision=ai_decision
            )
            
            if should_exit:
                exits.append((i, reason, details))
        
        return exits
    
    def record_trade_exit(
        self,
        pnl_dollar: float,
        pnl_percent: float,
        exit_type: str,
        portfolio_value: float = 0
    ):
        """Record trade exit with risk manager"""
        self.risk_manager.record_trade_result(
            pnl_dollar=pnl_dollar,
            pnl_percent=pnl_percent,
            exit_type=exit_type,
            portfolio_value=portfolio_value
        )
    
    def get_all_positions_status(self, current_price: float) -> List[Dict]:
        """Get status of all positions"""
        return [pos.get_status(current_price) for pos in self.positions]
    
    def get_risk_status(self) -> Dict:
        """Get risk manager status"""
        return self.risk_manager.get_status()


# Singleton instance
_trading_engine: Optional[TradingEngine] = None


def get_trading_engine() -> TradingEngine:
    """Get or create singleton trading engine"""
    global _trading_engine
    if _trading_engine is None:
        _trading_engine = TradingEngine()
    return _trading_engine
