"""
Unit tests for Risk Manager module.
Tests portfolio-level risk management, drawdown protection, and position sizing.
"""
import unittest
import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_manager import RiskManager


class TestRiskManagerBasics(unittest.TestCase):
    """Test basic risk manager functionality"""
    
    def setUp(self):
        """Create fresh RiskManager for each test"""
        self.rm = RiskManager()
        self.rm.reset_daily_stats()
    
    def test_initial_state(self):
        """Test risk manager initializes correctly"""
        self.assertEqual(self.rm.consecutive_losses, 0)
        self.assertEqual(self.rm.consecutive_wins, 0)
        self.assertIn(self.rm.risk_mode, ["normal", "cautious", "recovery", "blocked"])
    
    def test_can_trade_initial(self):
        """Test can_trade returns True initially"""
        can_trade, reason = self.rm.can_trade(portfolio_value=1000)
        self.assertTrue(can_trade)
    
    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement"""
        # Record losses up to limit
        for i in range(5):
            self.rm.record_trade_result(
                pnl_dollar=-25,  # $25 loss each
                pnl_percent=-2.5,
                exit_type="stop_loss",
                portfolio_value=1000
            )
        
        # After 5 losses, should be blocked (either due to daily loss or consecutive losses)
        can_trade, reason = self.rm.can_trade(portfolio_value=1000)
        # Should be blocked - either blocked_for_XXmin or daily_loss_limit
        self.assertFalse(can_trade)


class TestDrawdownProtection(unittest.TestCase):
    """Test drawdown tracking and protection"""
    
    def setUp(self):
        self.rm = RiskManager()
        self.rm.reset_daily_stats()
    
    def test_drawdown_tracking(self):
        """Test that drawdown is tracked correctly"""
        # Start at $1000
        self.rm.update_portfolio_value(1000)
        
        # Drop to $900 (10% drawdown)
        self.rm.update_portfolio_value(900)
        
        self.assertGreaterEqual(self.rm.current_drawdown_pct, 9.0)
    
    def test_recovery_mode_activation(self):
        """Test that recovery mode activates on high drawdown"""
        self.rm.update_portfolio_value(1000)
        self.rm.update_portfolio_value(850)  # 15% drawdown
        
        # Should be in recovery mode (not "conservative")
        self.assertEqual(self.rm.risk_mode, "recovery")


class TestPositionSizing(unittest.TestCase):
    """Test position size calculations"""
    
    def setUp(self):
        self.rm = RiskManager()
        self.rm.reset_daily_stats()
    
    def test_base_position_size(self):
        """Test position size multiplier in normal conditions"""
        mult = self.rm.get_position_size_multiplier(
            base_size=100,
            regime="trending_up",
            confidence=0.7,
            volatility="normal"
        )
        self.assertGreater(mult, 0)
        self.assertLessEqual(mult, 2.0)
    
    def test_reduced_size_in_downtrend(self):
        """Test position size reduced in downtrend"""
        mult_up = self.rm.get_position_size_multiplier(
            base_size=100,
            regime="trending_up",
            confidence=0.7,
            volatility="normal"
        )
        mult_down = self.rm.get_position_size_multiplier(
            base_size=100,
            regime="trending_down",
            confidence=0.7,
            volatility="normal"
        )
        self.assertLessEqual(mult_down, mult_up)
    
    def test_reduced_size_high_volatility(self):
        """Test position size reduced in high volatility"""
        mult_normal = self.rm.get_position_size_multiplier(
            base_size=100,
            regime="ranging",
            confidence=0.7,
            volatility="normal"
        )
        mult_high = self.rm.get_position_size_multiplier(
            base_size=100,
            regime="ranging",
            confidence=0.7,
            volatility="high"
        )
        self.assertLessEqual(mult_high, mult_normal)


class TestStopLossCalculation(unittest.TestCase):
    """Test dynamic stop loss calculations"""
    
    def setUp(self):
        self.rm = RiskManager()
    
    def test_effective_stop_loss(self):
        """Test effective stop loss calculation"""
        stop_pct, reason = self.rm.get_effective_stop_loss(
            base_stop_loss=0.01,  # 1%
            regime="ranging",
            volatility="normal",
            entry_price=100,
            current_price=99
        )
        
        # Should return a valid stop loss percentage
        self.assertGreater(stop_pct, 0)
        self.assertLess(stop_pct, 0.10)  # Less than 10%
    
    def test_force_stop_trigger(self):
        """Test force stop triggers at hard limit"""
        should_stop, reason = self.rm.should_force_stop(
            pnl_percent=-5.0,  # 5% loss (exceeds hard stop limit of 2%)
            entry_price=100,
            current_price=95,
            regime="trending_down",
            ai_score=0.3
        )
        
        # 5% loss should trigger force stop (hard_stop or downtrend_stop)
        self.assertTrue(should_stop)
        self.assertIn("stop", reason.lower())


class TestConsecutiveTracking(unittest.TestCase):
    """Test consecutive wins/losses tracking"""
    
    def setUp(self):
        self.rm = RiskManager()
        self.rm.reset_daily_stats()
    
    def test_consecutive_losses_tracking(self):
        """Test consecutive losses are tracked"""
        for i in range(3):
            self.rm.record_trade_result(
                pnl_dollar=-10,
                pnl_percent=-1.0,
                exit_type="stop_loss",
                portfolio_value=1000
            )
        
        self.assertEqual(self.rm.consecutive_losses, 3)
        self.assertEqual(self.rm.consecutive_wins, 0)
    
    def test_consecutive_wins_tracking(self):
        """Test consecutive wins are tracked"""
        for i in range(3):
            self.rm.record_trade_result(
                pnl_dollar=10,
                pnl_percent=1.0,
                exit_type="profit_target",
                portfolio_value=1000
            )
        
        self.assertEqual(self.rm.consecutive_wins, 3)
        self.assertEqual(self.rm.consecutive_losses, 0)
    
    def test_streak_reset_on_opposite(self):
        """Test streak resets on opposite result"""
        # Win 3 times
        for i in range(3):
            self.rm.record_trade_result(10, 1.0, "profit", 1000)
        
        # Lose once
        self.rm.record_trade_result(-10, -1.0, "stop_loss", 1000)
        
        self.assertEqual(self.rm.consecutive_losses, 1)
        self.assertEqual(self.rm.consecutive_wins, 0)


if __name__ == '__main__':
    print("=" * 60)
    print("Running Risk Manager Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
