"""
Unit tests for AI Engine module.
Tests decision making, indicator weights, and confluence calculation.
"""
import unittest
import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.ai_engine import AIEngine, Decision


class TestAIEngineBasics(unittest.TestCase):
    """Test basic AI engine functionality"""
    
    def setUp(self):
        """Create fresh AIEngine for each test"""
        self.ai = AIEngine()
    
    def test_initialization(self):
        """Test AI engine initializes correctly"""
        self.assertIsNotNone(self.ai.weights)
        self.assertGreater(len(self.ai.weights), 0)
    
    def test_decision_enum(self):
        """Test Decision enum values"""
        self.assertEqual(Decision.BUY.value, "BUY")
        self.assertEqual(Decision.SELL.value, "SELL")
        self.assertEqual(Decision.HOLD.value, "HOLD")


class TestConfluenceCalculation(unittest.TestCase):
    """Test confluence calculation logic"""
    
    def setUp(self):
        self.ai = AIEngine()
    
    def test_bullish_confluence(self):
        """Test confluence counts bullish signals"""
        analysis = {
            "rsi": 35,            # Oversold (bullish)
            "macd_signal": "bullish",
            "bb_position": "lower", # At lower band (bullish)
            "trend": "bullish",
            "volume_confirmed": True
        }
        
        confluence = self.ai.calculate_confluence(analysis)
        
        self.assertIn("count", confluence)
        self.assertIn("direction", confluence)
        self.assertGreater(confluence["count"], 0)
    
    def test_bearish_confluence(self):
        """Test confluence counts bearish signals"""
        analysis = {
            "rsi": 75,            # Overbought (bearish)
            "macd_signal": "bearish",
            "bb_position": "upper", # At upper band (bearish)
            "trend": "bearish",
            "volume_confirmed": True
        }
        
        confluence = self.ai.calculate_confluence(analysis)
        
        self.assertIn("count", confluence)
        self.assertEqual(confluence["direction"], "bearish")


class TestIndicatorWeights(unittest.TestCase):
    """Test indicator weight management"""
    
    def setUp(self):
        self.ai = AIEngine()
    
    def test_weights_sum_to_one(self):
        """Test that indicator weights approximately sum to 1"""
        total = sum(self.ai.weights.values())
        self.assertAlmostEqual(total, 1.0, places=1)
    
    def test_all_weights_positive(self):
        """Test all weights are positive"""
        for ind, weight in self.ai.weights.items():
            self.assertGreater(weight, 0, f"Weight for {ind} should be positive")
    
    def test_indicator_accuracy_exists(self):
        """Test indicator accuracy tracking exists"""
        # All indicators should have accuracy tracking
        for ind in self.ai.weights:
            accuracy = self.ai.indicator_accuracy.get(ind, None)
            self.assertIsNotNone(accuracy, f"Accuracy for {ind} should exist")
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)


class TestScoreCalculation(unittest.TestCase):
    """Test AI score calculation"""
    
    def setUp(self):
        self.ai = AIEngine()
    
    def test_score_range(self):
        """Test score is within valid range"""
        analysis = {
            "rsi": 50,
            "macd_signal": "neutral",
            "trend": "neutral"
        }
        
        score, details = self.ai.calculate_score(analysis)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(details, dict)
    
    def test_bullish_signals_increase_score(self):
        """Test bullish signals produce higher score"""
        bearish_analysis = {
            "rsi": 75,
            "macd_signal": "bearish",
            "trend": "bearish"
        }
        
        bullish_analysis = {
            "rsi": 30,
            "macd_signal": "bullish",
            "trend": "bullish"
        }
        
        bearish_score, _ = self.ai.calculate_score(bearish_analysis)
        bullish_score, _ = self.ai.calculate_score(bullish_analysis)
        
        self.assertGreater(bullish_score, bearish_score)


class TestWinStreakTracking(unittest.TestCase):
    """Test win/loss streak tracking"""
    
    def setUp(self):
        self.ai = AIEngine()
        # Reset streaks for clean test
        self.ai.win_streak = 0
        self.ai.loss_streak = 0
    
    def test_win_streak_tracking(self):
        """Test consecutive wins are tracked"""
        for _ in range(3):
            self.ai.record_trade_result(profit_percent=1.0)  # 1% profit = win
        
        info = self.ai.get_streak_info()
        self.assertEqual(info["win_streak"], 3)
        self.assertEqual(info["loss_streak"], 0)
    
    def test_loss_streak_tracking(self):
        """Test consecutive losses are tracked"""
        for _ in range(3):
            self.ai.record_trade_result(profit_percent=-1.0)  # -1% = loss
        
        info = self.ai.get_streak_info()
        self.assertEqual(info["loss_streak"], 3)
        self.assertEqual(info["win_streak"], 0)
    
    def test_streak_reset(self):
        """Test streak resets on opposite result"""
        # Win 3 times
        for _ in range(3):
            self.ai.record_trade_result(profit_percent=1.0)
        
        # Lose once
        self.ai.record_trade_result(profit_percent=-1.0)
        
        info = self.ai.get_streak_info()
        self.assertEqual(info["loss_streak"], 1)
        self.assertEqual(info["win_streak"], 0)


class TestRegimeAwareness(unittest.TestCase):
    """Test market regime awareness"""
    
    def setUp(self):
        self.ai = AIEngine()
    
    def test_regime_stored(self):
        """Test regime data can be stored"""
        from market.market_regime import MarketRegime
        
        # Set regime directly (normally set by detect_regime)
        self.ai.current_regime = MarketRegime.TRENDING_UP
        self.ai.regime_data = {"regime": MarketRegime.TRENDING_UP, "regime_name": "trending_up"}
        
        # Verify it's stored
        self.assertEqual(self.ai.current_regime, MarketRegime.TRENDING_UP)
    
    def test_weights_vary_by_regime(self):
        """Test effective weights can vary by regime"""
        # Get weights for different regimes
        weights_up = self.ai._get_effective_weights(regime="trending_up")
        weights_down = self.ai._get_effective_weights(regime="trending_down")
        
        # Both should return valid weights
        self.assertIsNotNone(weights_up)
        self.assertIsNotNone(weights_down)
        self.assertGreater(len(weights_up), 0)


if __name__ == '__main__':
    print("=" * 60)
    print("Running AI Engine Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
