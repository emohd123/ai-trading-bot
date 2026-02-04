"""
Test Suite for Trading Engine
Tests risk management, position management, and trade execution.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from core.trading_engine import (
    TradingEngine,
    PositionManager,
    TrailingProfitTracker,
    ExitReason,
    get_trading_engine
)
from core.risk_manager import RiskManager, get_risk_manager


class TestTrailingProfitTracker(unittest.TestCase):
    """Test trailing profit tracking"""
    
    def setUp(self):
        self.entry_price = 100000.0  # $100k BTC
        self.tracker = TrailingProfitTracker(
            entry_price=self.entry_price,
            activation_pct=0.01,  # 1%
            trail_pct=0.005,      # 0.5%
            min_profit_pct=0.0025
        )
    
    def test_initial_state(self):
        """Test initial state is inactive"""
        self.assertFalse(self.tracker.trailing_active)
        self.assertEqual(self.tracker.highest_price, self.entry_price)
    
    def test_activation(self):
        """Test trailing activates at threshold"""
        # Price up 0.5% - not enough
        result = self.tracker.update(101500)  # +1.5%
        self.assertTrue(result['trailing_active'])
        
        # Reset and test below threshold
        tracker2 = TrailingProfitTracker(self.entry_price, 0.01, 0.005, 0.0025)
        result2 = tracker2.update(100500)  # +0.5%
        self.assertFalse(result2['trailing_active'])
    
    def test_trailing_stop_updates(self):
        """Test trailing stop follows price up"""
        # Activate at 1.5%
        self.tracker.update(101500)
        initial_stop = self.tracker.trailing_stop_price
        
        # Price goes higher
        self.tracker.update(102000)
        self.assertGreater(self.tracker.trailing_stop_price, initial_stop)
        self.assertEqual(self.tracker.highest_price, 102000)
    
    def test_exit_on_pullback(self):
        """Test exit triggered on pullback"""
        # Activate
        self.tracker.update(101500)
        # Price continues up
        self.tracker.update(103000)  # +3%
        
        # Pullback below trailing stop
        stop = self.tracker.trailing_stop_price
        result = self.tracker.update(stop - 10)
        
        self.assertTrue(result['should_exit'])
        self.assertEqual(result['exit_reason'], 'trailing_profit')


class TestRiskManager(unittest.TestCase):
    """Test risk management"""
    
    def setUp(self):
        # Create a fresh risk manager for each test
        self.risk_mgr = RiskManager()
        self.risk_mgr.daily_losses = 0
        self.risk_mgr.daily_trades = 0
        self.risk_mgr.consecutive_losses = 0
        self.risk_mgr.risk_mode = "normal"
    
    def test_can_trade_normal(self):
        """Test trading allowed in normal conditions"""
        can_trade, reason = self.risk_mgr.can_trade()
        self.assertTrue(can_trade)
        self.assertEqual(reason, "ok")
    
    def test_daily_loss_limit(self):
        """Test daily loss limit blocks trading"""
        self.risk_mgr.daily_losses = 0.04  # 4% > 3% limit
        
        can_trade, reason = self.risk_mgr.can_trade()
        self.assertFalse(can_trade)
        self.assertIn("daily_loss", reason)
    
    def test_daily_trade_limit(self):
        """Test daily trade limit blocks trading"""
        self.risk_mgr.daily_trades = 25  # > 20 limit
        
        can_trade, reason = self.risk_mgr.can_trade()
        self.assertFalse(can_trade)
        self.assertIn("trade_limit", reason)
    
    def test_force_stop_hard_limit(self):
        """Test hard stop is forced"""
        force, reason = self.risk_mgr.should_force_stop(
            pnl_percent=-2.5,  # -2.5% > 2% hard limit
            entry_price=100000,
            current_price=97500,
            regime="trending_up"
        )
        
        self.assertTrue(force)
        self.assertIn("hard_stop", reason)
    
    def test_downtrend_tighter_stop(self):
        """Test downtrend forces stop earlier"""
        force, reason = self.risk_mgr.should_force_stop(
            pnl_percent=-0.7,  # > 0.6% downtrend stop
            entry_price=100000,
            current_price=99300,
            regime="trending_down"
        )
        
        self.assertTrue(force)
        self.assertEqual(reason, "downtrend_stop")
    
    def test_position_size_multiplier(self):
        """Test position size adjustments"""
        # Normal conditions
        mult = self.risk_mgr.get_position_size_multiplier(
            base_size=40,
            regime="ranging",
            confidence=0.5,
            volatility="normal"
        )
        self.assertGreaterEqual(mult, 0.3)
        self.assertLessEqual(mult, 1.5)
        
        # Downtrend should reduce
        mult_down = self.risk_mgr.get_position_size_multiplier(
            base_size=40,
            regime="trending_down",
            confidence=0.5,
            volatility="normal"
        )
        self.assertLess(mult_down, mult)


class TestPositionManager(unittest.TestCase):
    """Test position management"""
    
    def setUp(self):
        self.position_data = {
            'entry_price': 100000,
            'quantity': 0.0004,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'trade_id': 1,
            'regime': 'ranging'
        }
        self.pos = PositionManager(self.position_data)
    
    def test_profit_target_exit(self):
        """Test exit on profit target"""
        # Price up 2% (> 1.5% target)
        should_exit, reason, details = self.pos.check_exit(
            current_price=102000,
            regime="ranging",
            volatility="normal"
        )
        
        # Should exit with profit_target or trailing_profit
        self.assertTrue(should_exit)
        self.assertIn(reason, [ExitReason.PROFIT_TARGET, ExitReason.TRAILING_PROFIT, ExitReason.MIN_PROFIT])
    
    def test_stop_loss_exit(self):
        """Test exit on stop loss"""
        should_exit, reason, details = self.pos.check_exit(
            current_price=99000,  # -1%
            regime="ranging",
            volatility="normal"
        )
        
        self.assertTrue(should_exit)
        self.assertIn(reason, [ExitReason.STOP_LOSS, ExitReason.HARD_STOP])
    
    def test_breakeven_protection(self):
        """Test breakeven activation"""
        # First, move into profit to activate breakeven
        self.pos.check_exit(current_price=100700, regime="ranging")  # +0.7%
        self.assertTrue(self.pos.breakeven_activated)
        
        # Now price drops to just above entry
        should_exit, reason, _ = self.pos.check_exit(
            current_price=100050,  # Just above breakeven
            regime="ranging"
        )
        
        # Should not exit yet (still above breakeven)
        # or should exit with breakeven
        if should_exit:
            self.assertEqual(reason, ExitReason.BREAKEVEN)


class TestTradingEngine(unittest.TestCase):
    """Test trading engine"""
    
    def setUp(self):
        self.engine = TradingEngine()
    
    def test_can_open_position(self):
        """Test position opening check"""
        self.engine.positions = []
        can_open, reason = self.engine.can_open_position()
        
        self.assertTrue(can_open)
    
    def test_max_positions_limit(self):
        """Test max positions limit"""
        # Add max positions
        for i in range(self.engine.max_positions):
            self.engine.positions.append(PositionManager({
                'entry_price': 100000,
                'quantity': 0.0004,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'trade_id': i + 1,
                'regime': 'ranging'
            }))
        
        can_open, reason = self.engine.can_open_position()
        self.assertFalse(can_open)
        self.assertIn("max_positions", reason)
    
    def test_position_size_calculation(self):
        """Test position size calculation"""
        amount, details = self.engine.calculate_position_size(
            base_amount=40,
            regime="ranging",
            confidence=0.5,
            volatility="normal"
        )
        
        self.assertGreater(amount, 0)
        self.assertIn('risk_mult', details)
        self.assertIn('adjusted_amount', details)


if __name__ == '__main__':
    print("=" * 50)
    print("Running Trading Engine Tests")
    print("=" * 50)
    
    unittest.main(verbosity=2)
