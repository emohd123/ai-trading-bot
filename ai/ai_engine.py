"""
AI Decision Engine - PHASE 4 ENHANCED VERSION
Smart trading decisions with:
- Confluence scoring (multiple indicators must agree)
- Divergence detection (early reversal signals)
- Regime-aware thresholds (adapts to market conditions)
- Confidence levels (know how strong the signal is)

PHASE 2 FEATURES:
- Win Streak Adaptation: Trade bigger when winning, smaller when losing
- Anti-Whipsaw Cooldown: Don't re-enter immediately after stop loss
- Stochastic Integration: Faster oversold/overbought detection
- Bollinger Squeeze Alerts: Detect imminent breakouts

PHASE 3 FEATURES:
- Combined Momentum (RSI+Stochastic): Fixes 90% correlation issue, now 5 indicators
- Extended Divergence: 20-candle lookback with strength levels
- 15m Timeframe Refinement: Better entry timing
- Persistent Learning: Save/load trade history and weights
- Fibonacci Level Awareness: Enhanced S/R signals
- Candlestick Pattern Recognition: Early reversal detection

PHASE 4 NEW FEATURES:
- Ichimoku Cloud: Complete trend system with cloud support/resistance
- Money Flow Index (MFI): Volume-weighted RSI for money flow
- Williams %R: Fast momentum oscillator
- CCI: Mean deviation for trend strength
- Order Book Analysis: Bid/ask imbalance and wall detection
- Sentiment Analysis: Fear & Greed Index, funding rate
- Correlation Analysis: BTC dominance, market context
"""
import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
import config
from market.market_regime import RegimeDetector, MarketRegime

# ML predictor (optional - may not be trained)
try:
    from ai.ml_predictor import MLPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Phase 4: External data sources (optional)
try:
    from market.order_book import OrderBookAnalyzer
    ORDER_BOOK_AVAILABLE = True
except ImportError:
    ORDER_BOOK_AVAILABLE = False

try:
    from market.sentiment import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    from market.correlation import CorrelationAnalyzer
    CORRELATION_AVAILABLE = True
except ImportError:
    CORRELATION_AVAILABLE = False


class Decision(Enum):
    """Trading decision types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AIEngine:
    """
    Enhanced AI-powered decision engine for trading
    Now with smart features that adapt automatically!
    """

    def __init__(self):
        """Initialize AI engine with config settings"""
        self.weights = config.INDICATOR_WEIGHTS
        self.buy_threshold = config.BUY_THRESHOLD
        self.sell_threshold = config.SELL_THRESHOLD

        # Initialize regime detector for smart adaptation
        self.regime_detector = RegimeDetector()
        self.current_regime = None
        self.regime_data = None

        # Confluence settings (will be adjusted by regime) - now 5 indicators
        self.min_confluence = getattr(config, 'MIN_CONFLUENCE', 3)

        # === PHASE 2: Win Streak Tracking ===
        self.trade_history = []  # List of recent trade results
        self.max_history = getattr(config, 'MAX_TRADE_HISTORY', 20)  # Track last 20 trades for learning
        self.win_streak = 0      # Current winning streak
        self.loss_streak = 0     # Current losing streak
        self.streak_multiplier = 1.0  # Position size adjustment

        # === PHASE 2: Anti-Whipsaw Cooldown ===
        self.cooldown_until = None  # Timestamp when cooldown ends
        self.cooldown_minutes = getattr(config, 'COOLDOWN_AFTER_LOSS_MINUTES', 15)  # Loss avoidance: wait longer after stop
        self.last_stop_loss_time = None

        # Validate weights sum to 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            for key in self.weights:
                self.weights[key] /= total_weight

        # Adaptive weights (learn from trade outcomes) - ALL 10 indicators
        self.indicator_accuracy = {ind: 0.5 for ind in self.weights}  # Start neutral
        
        # === PHASE 7: Enhanced Indicator Learning ===
        # Directional accuracy - track bullish vs bearish separately
        self.directional_accuracy = {
            ind: {"bullish": 0.5, "bearish": 0.5, "bullish_samples": 0, "bearish_samples": 0}
            for ind in self.weights
        }
        
        # Per-indicator sample counts for statistical significance
        self.indicator_samples = {ind: 0 for ind in self.weights}
        
        # Regime-specific indicator accuracy
        self.regime_indicator_accuracy = {}  # e.g., {"trending_up": {"momentum": 0.6, ...}}

        # === PHASE 6: Advanced Learning Structures ===
        # Entry condition performance tracking
        self.condition_performance = {}  # e.g., {"rsi_oversold": {"wins": 5, "losses": 1}}
        
        # Optimal stop loss learning by regime/volatility
        self.stop_loss_learning = {}  # e.g., {"high_volatility": {"avg_optimal": 1.8, "samples": 5}}
        
        # Loss reason tracking
        self.loss_reasons = {}  # e.g., {"stopped_too_early": 3, "entered_against_trend": 2}
        
        # Indicator combo performance
        self.indicator_combos = {}  # e.g., {"rsi_bullish+macd_bullish": {"wins": 8, "losses": 2}}

        # === PHASE 3: Learning Persistence ===
        self.learning_file = getattr(config, 'LEARNING_FILE', 'ai_learning.json')
        self._load_learning_state()

        # === PHASE 3: 15m Refinement Data ===
        self.last_15m_analysis = None

        # === ML Prediction (optional) ===
        self.ml_predictor = None
        if ML_AVAILABLE:
            try:
                self.ml_predictor = MLPredictor()
            except Exception:
                pass
        
        # === PHASE 4: External Data Sources ===
        self.order_book_analyzer = None
        if ORDER_BOOK_AVAILABLE:
            try:
                self.order_book_analyzer = OrderBookAnalyzer()
            except Exception:
                pass
        
        self.sentiment_analyzer = None
        if SENTIMENT_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
            except Exception:
                pass
        
        self.correlation_analyzer = None
        if CORRELATION_AVAILABLE:
            try:
                self.correlation_analyzer = CorrelationAnalyzer()
            except Exception:
                pass
        
        # Phase 4: External data cache
        self._order_book_data = None
        self._sentiment_data = None
        self._correlation_data = None

    def _adjust_indicator_scores_for_regime(self, scores: Dict[str, float], analysis: Dict, regime_name: str = None) -> Dict[str, float]:
        """
        Adjust indicator scores based on regime context for accurate interpretation.
        
        In uptrends: Overbought conditions are less bearish (continuation signal)
        In downtrends: Oversold conditions are less bullish (continuation signal)
        This makes AI scores accurate for all coins all the time.
        """
        if not regime_name:
            return scores
        
        adjusted = scores.copy()
        
        if regime_name == "trending_up":
            # In uptrends: Overbought indicators are less bearish (trend continuation)
            # Adjust bearish signals to be less negative
            
            # RSI/MFI/Williams: Overbought in uptrend = continuation, not reversal
            rsi_data = analysis.get("rsi", {})
            if rsi_data.get("signal") == "overbought":
                # Reduce bearishness: -1.0 becomes -0.3 (still cautious but not blocking)
                adjusted["momentum"] = max(adjusted.get("momentum", 0), -0.3)
            
            mfi_data = analysis.get("mfi", {})
            if mfi_data.get("signal") == "overbought":
                adjusted["mfi"] = max(adjusted.get("mfi", 0), -0.3)
            
            williams_data = analysis.get("williams_r", {})
            if williams_data.get("signal") == "overbought":
                adjusted["williams_r"] = max(adjusted.get("williams_r", 0), -0.3)
            
            # Bollinger: Above upper band in uptrend = strong momentum, not reversal
            bb_data = analysis.get("bollinger", {})
            if bb_data.get("signal") == "above_upper":
                adjusted["bollinger"] = max(adjusted.get("bollinger", 0), -0.2)
            
            # MACD: Bearish crossover in uptrend might be pullback, not reversal
            macd_data = analysis.get("macd", {})
            if macd_data.get("signal") in ["bearish", "bearish_crossover"]:
                # Reduce bearishness by 50%
                adjusted["macd"] = adjusted.get("macd", 0) * 0.5
            
            # CCI: Overbought in uptrend = continuation
            cci_data = analysis.get("cci", {})
            if cci_data.get("signal") == "overbought":
                adjusted["cci"] = max(adjusted.get("cci", 0), -0.3)
        
        elif regime_name == "trending_down":
            # In downtrends: Oversold indicators are less bullish (trend continuation)
            # Adjust bullish signals to be less positive
            
            rsi_data = analysis.get("rsi", {})
            if rsi_data.get("signal") == "oversold":
                # Reduce bullishness: +1.0 becomes +0.3
                adjusted["momentum"] = min(adjusted.get("momentum", 0), 0.3)
            
            mfi_data = analysis.get("mfi", {})
            if mfi_data.get("signal") == "oversold":
                adjusted["mfi"] = min(adjusted.get("mfi", 0), 0.3)
            
            williams_data = analysis.get("williams_r", {})
            if williams_data.get("signal") == "oversold":
                adjusted["williams_r"] = min(adjusted.get("williams_r", 0), 0.3)
            
            # Bollinger: Below lower band in downtrend = strong down momentum
            bb_data = analysis.get("bollinger", {})
            if bb_data.get("signal") == "below_lower":
                adjusted["bollinger"] = min(adjusted.get("bollinger", 0), 0.2)
            
            # MACD: Bullish crossover in downtrend might be dead cat bounce
            macd_data = analysis.get("macd", {})
            if macd_data.get("signal") in ["bullish", "bullish_crossover"]:
                adjusted["macd"] = adjusted.get("macd", 0) * 0.5
        
        return adjusted
    
    def _get_effective_weights(self, regime: str = None) -> Dict[str, float]:
        """
        Get effective indicator weights (config or adaptive).
        PHASE 7 Enhanced:
        - Uses regime-specific base weights when available
        - Applies learned accuracy with statistical significance check
        - Blends directional accuracy based on current signal direction
        
        Args:
            regime: Current market regime (trending_up, trending_down, ranging, high_volatility)
        """
        min_trades_for_learning = getattr(config, 'MIN_SAMPLES_FOR_LEARNING', 8)
        min_deviation = getattr(config, 'MIN_ACCURACY_DEVIATION', 0.08)
        
        if not getattr(config, 'ADAPTIVE_WEIGHTS_ENABLED', False) or len(self.trade_history) < 5:
            # Use regime-specific base weights if available
            if regime:
                regime_weights = getattr(config, 'REGIME_BASE_WEIGHTS', {})
                if regime in regime_weights:
                    return dict(regime_weights[regime])
            return dict(self.weights)
        
        # Start with base weights (regime-specific if available)
        base_weights = dict(self.weights)
        if regime:
            regime_weights = getattr(config, 'REGIME_BASE_WEIGHTS', {})
            if regime in regime_weights:
                base_weights = dict(regime_weights[regime])
        
        # Calculate effective weights
        effective = {}
        neutral_prior = 1.0 / len(self.weights) if self.weights else 0.1
        for ind in self.weights:
            base_weight = base_weights.get(ind, self.weights.get(ind, 0.1))
            samples = self.indicator_samples.get(ind, 0) if hasattr(self, "indicator_samples") else 0

            # Get accuracy (prefer regime-specific when enough samples)
            accuracy = self.indicator_accuracy.get(ind, 0.5)
            if regime and hasattr(self, "regime_indicator_accuracy"):
                regime_acc = self.regime_indicator_accuracy.get(regime, {})
                if ind in regime_acc:
                    if samples >= min_trades_for_learning:
                        accuracy = 0.9 * regime_acc[ind] + 0.1 * accuracy
                    else:
                        accuracy = 0.7 * regime_acc[ind] + 0.3 * accuracy

            # Very low samples: nudge toward neutral prior so 1–2 trades don't move weights wildly
            if samples < 3:
                effective[ind] = 0.7 * base_weight + 0.3 * neutral_prior
            elif samples < min_trades_for_learning:
                effective[ind] = base_weight
            elif abs(accuracy - 0.5) < min_deviation:
                effective[ind] = base_weight
            else:
                effective[ind] = base_weight * (0.5 + accuracy)
        
        # Apply time decay to older trades when calculating overall learning confidence
        if getattr(config, 'LEARNING_TIME_DECAY', True) and self.trade_history:
            decay_rate = getattr(config, 'DECAY_RATE_PER_DAY', 0.95)
            min_decay = getattr(config, 'MIN_DECAY_FACTOR', 0.3)
            now = datetime.now()
            
            # Calculate average age-weighted confidence
            total_weight = 0
            total_decayed_weight = 0
            for trade in self.trade_history:
                ts = trade.get('timestamp')
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                if ts:
                    age_days = (now - ts).days
                    decay = max(min_decay, decay_rate ** age_days)
                    total_weight += 1
                    total_decayed_weight += decay
            
            # If average decay is low, reduce confidence in learned weights
            if total_weight > 0:
                avg_decay = total_decayed_weight / total_weight
                if avg_decay < 0.7:
                    # Blend back towards base weights
                    blend_factor = avg_decay / 0.7  # 0 to 1
                    for ind in effective:
                        base = base_weights.get(ind, self.weights.get(ind, 0.1))
                        effective[ind] = blend_factor * effective[ind] + (1 - blend_factor) * base
        
        # Normalize to sum to 1.0
        total = sum(effective.values())
        if total > 0:
            return {k: v / total for k, v in effective.items()}
        return dict(self.weights)

    # === PHASE 3: Learning Persistence Methods ===

    def _load_learning_state(self):
        """Load learning state from file (trade history, indicator accuracy, streaks)"""
        try:
            if os.path.exists(self.learning_file):
                with open(self.learning_file, 'r') as f:
                    data = json.load(f)

                # Load trade history
                self.trade_history = data.get('trade_history', [])
                # Convert timestamp strings back to datetime
                for trade in self.trade_history:
                    if 'timestamp' in trade and isinstance(trade['timestamp'], str):
                        trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])

                # Load indicator accuracy (ensure all 10 indicators are present)
                loaded_accuracy = data.get('indicator_accuracy', {})
                self.indicator_accuracy = {ind: loaded_accuracy.get(ind, 0.5) for ind in self.weights}
                
                # PHASE 7: Load enhanced learning structures
                # Directional accuracy (bullish/bearish separately)
                loaded_directional = data.get('directional_accuracy', {})
                self.directional_accuracy = {
                    ind: loaded_directional.get(ind, {"bullish": 0.5, "bearish": 0.5, "bullish_samples": 0, "bearish_samples": 0})
                    for ind in self.weights
                }
                
                # Per-indicator sample counts
                loaded_samples = data.get('indicator_samples', {})
                self.indicator_samples = {ind: loaded_samples.get(ind, 0) for ind in self.weights}
                
                # Regime-specific indicator accuracy
                self.regime_indicator_accuracy = data.get('regime_indicator_accuracy', {})
                
                # PHASE 5: Load regime-specific accuracy
                self.regime_accuracy = data.get('regime_accuracy', {})

                # Load streaks
                self.win_streak = data.get('win_streak', 0)
                self.loss_streak = data.get('loss_streak', 0)
                self.streak_multiplier = data.get('streak_multiplier', 1.0)
                
                # PHASE 6: Load advanced learning structures
                self.condition_performance = data.get('condition_performance', {})
                self.stop_loss_learning = data.get('stop_loss_learning', {})
                self.loss_reasons = data.get('loss_reasons', {})
                self.indicator_combos = data.get('indicator_combos', {})

                print(f"[OK] Loaded learning state: {len(self.trade_history)} trades, W{self.win_streak}/L{self.loss_streak}")
        except Exception as e:
            logger.warning("Could not load learning state: %s", e)
            # Initialize with defaults
            self.indicator_accuracy = {ind: 0.5 for ind in self.weights}
            self.directional_accuracy = {
                ind: {"bullish": 0.5, "bearish": 0.5, "bullish_samples": 0, "bearish_samples": 0}
                for ind in self.weights
            }
            self.indicator_samples = {ind: 0 for ind in self.weights}
            self.regime_indicator_accuracy = {}

    def _save_learning_state(self):
        """Save learning state to file for persistence across restarts"""
        try:
            # Prepare data for serialization
            trade_history_serializable = []
            for trade in self.trade_history:
                trade_copy = dict(trade)
                if 'timestamp' in trade_copy and isinstance(trade_copy['timestamp'], datetime):
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                trade_history_serializable.append(trade_copy)

            data = {
                'trade_history': trade_history_serializable,
                'indicator_accuracy': self.indicator_accuracy,
                # PHASE 7: Enhanced learning structures
                'directional_accuracy': getattr(self, 'directional_accuracy', {}),
                'indicator_samples': getattr(self, 'indicator_samples', {}),
                'regime_indicator_accuracy': getattr(self, 'regime_indicator_accuracy', {}),
                'regime_accuracy': getattr(self, 'regime_accuracy', {}),  # PHASE 5
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak,
                'streak_multiplier': self.streak_multiplier,
                # PHASE 6: Advanced learning structures
                'condition_performance': self.condition_performance,
                'stop_loss_learning': self.stop_loss_learning,
                'loss_reasons': self.loss_reasons,
                'indicator_combos': self.indicator_combos,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.learning_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info("Saved learning state: %d trades", len(self.trade_history))
        except Exception as e:
            logger.warning("Could not save learning state: %s", e)

    def set_15m_analysis(self, analysis_15m: Dict):
        """
        Store 15m timeframe analysis for entry refinement

        Args:
            analysis_15m: Analysis dict from 15m candles
        """
        self.last_15m_analysis = analysis_15m

    # === PHASE 2: Win Streak Methods ===

    def record_trade_result(
        self,
        profit_percent: float,
        exit_type: str = "unknown",
        indicator_scores_at_entry: Optional[Dict[str, float]] = None,
        regime: str = None,
        entry_conditions: Dict = None,
        indicator_combo_key: str = None,
        max_drawdown: float = None,
        volatility: str = None
    ):
        """
        Record a trade result to track win/loss streaks and adaptive weights.
        Call this after every trade closes!

        Args:
            profit_percent: The P/L percentage of the trade
            exit_type: How the trade exited (profit_target, stop_loss, ai_signal, etc.)
            indicator_scores_at_entry: Dict of indicator -> score from when we entered (for adaptive weights)
            regime: Market regime when trade was made (for regime-specific tracking)
            entry_conditions: Dict of market conditions when trade was entered (PHASE 6)
            indicator_combo_key: Key representing the indicator combo at entry (PHASE 6)
            max_drawdown: Maximum drawdown % during the trade (PHASE 6)
            volatility: Volatility level at entry (PHASE 6)
        """
        is_win = profit_percent > 0
        trade_record = {
            "profit_percent": profit_percent,
            "is_win": is_win,
            "exit_type": exit_type,
            "timestamp": datetime.now(),
            "regime": regime or "unknown"
        }

        self.trade_history.append(trade_record)

        # Keep only last N trades
        if len(self.trade_history) > self.max_history:
            self.trade_history = self.trade_history[-self.max_history:]

        # Update streaks
        if is_win:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

            # If stopped out, activate cooldown
            if exit_type in ["stop_loss", "hard_stop"]:
                self._activate_cooldown()

        # Calculate streak multiplier for position sizing
        self._update_streak_multiplier()

        # === PHASE 7: Enhanced Adaptive Learning ===
        if (
            getattr(config, 'ADAPTIVE_WEIGHTS_ENABLED', False)
            and indicator_scores_at_entry
        ):
            # Get per-indicator thresholds (or use default 0.25)
            ind_thresholds = getattr(config, 'INDICATOR_THRESHOLDS', {})
            
            # Calculate profit weight (larger profits/losses matter more)
            profit_weight = 1.0
            if getattr(config, 'PROFIT_WEIGHTED_LEARNING', True):
                profit_weight_cap = getattr(config, 'PROFIT_WEIGHT_CAP', 2.0)
                loss_weight_factor = getattr(config, 'LOSS_WEIGHT_FACTOR', 0.5)
                # Weight by profit magnitude (capped)
                profit_weight = min(profit_weight_cap, 1.0 + abs(profit_percent))
                # Asymmetric: losses weighted less to prevent over-reacting
                if not is_win:
                    profit_weight *= loss_weight_factor
            
            # Calculate time decay factor (recent trades matter more)
            decay_factor = 1.0
            if getattr(config, 'LEARNING_TIME_DECAY', True):
                # This trade is brand new, so decay = 1.0
                # Decay is applied when USING the learning, not recording
                pass
            
            for ind, score in indicator_scores_at_entry.items():
                if ind not in self.indicator_accuracy:
                    # Ensure Phase 4 indicators are tracked too
                    self.indicator_accuracy[ind] = 0.5
                    if not hasattr(self, 'directional_accuracy') or ind not in self.directional_accuracy:
                        if not hasattr(self, 'directional_accuracy'):
                            self.directional_accuracy = {}
                        self.directional_accuracy[ind] = {"bullish": 0.5, "bearish": 0.5, "bullish_samples": 0, "bearish_samples": 0}
                    if not hasattr(self, 'indicator_samples') or ind not in self.indicator_samples:
                        if not hasattr(self, 'indicator_samples'):
                            self.indicator_samples = {}
                        self.indicator_samples[ind] = 0
                
                # Use indicator-specific threshold
                threshold = ind_thresholds.get(ind, 0.25)
                
                # Determine if indicator agreed (bullish signal)
                agreed = score > threshold
                is_bullish_signal = score > 0  # Any positive = bullish direction
                
                # Correct: (agreed and won) or (disagreed and lost)
                correct = (agreed and is_win) or (not agreed and not is_win)
                
                # Calculate accuracy delta with profit weighting
                base_delta = 1.0 if correct else 0.0
                learning_rate = 0.25  # Slightly slower than before for stability
                weighted_delta = base_delta * profit_weight
                
                # Update overall accuracy
                old_acc = self.indicator_accuracy.get(ind, 0.5)
                new_acc = (1 - learning_rate) * old_acc + learning_rate * (weighted_delta / max(profit_weight, 0.1))
                # Clamp to [0.1, 0.9] to prevent extreme values
                self.indicator_accuracy[ind] = max(0.1, min(0.9, new_acc))
                
                # Update sample count
                self.indicator_samples[ind] = self.indicator_samples.get(ind, 0) + 1
                
                # === Directional Accuracy Update ===
                if getattr(config, 'DIRECTIONAL_LEARNING', True):
                    if is_bullish_signal:
                        direction = "bullish"
                        # For bullish signals: correct if we won
                        dir_correct = is_win
                    else:
                        direction = "bearish"
                        # For bearish signals: correct if we would have sold (and price dropped)
                        # Since we bought, bearish signal was wrong if we won, right if we lost
                        dir_correct = not is_win
                    
                    dir_data = self.directional_accuracy.get(ind, {"bullish": 0.5, "bearish": 0.5, "bullish_samples": 0, "bearish_samples": 0})
                    old_dir_acc = dir_data.get(direction, 0.5)
                    new_dir_acc = (1 - learning_rate) * old_dir_acc + learning_rate * (1.0 if dir_correct else 0.0)
                    dir_data[direction] = max(0.1, min(0.9, new_dir_acc))
                    dir_data[f"{direction}_samples"] = dir_data.get(f"{direction}_samples", 0) + 1
                    self.directional_accuracy[ind] = dir_data
                
                # === Regime-Specific Accuracy Update ===
                if regime:
                    if not hasattr(self, 'regime_indicator_accuracy'):
                        self.regime_indicator_accuracy = {}
                    if regime not in self.regime_indicator_accuracy:
                        self.regime_indicator_accuracy[regime] = {i: 0.5 for i in self.weights}
                    
                    old_regime_acc = self.regime_indicator_accuracy[regime].get(ind, 0.5)
                    new_regime_acc = (1 - learning_rate) * old_regime_acc + learning_rate * (1.0 if correct else 0.0)
                    self.regime_indicator_accuracy[regime][ind] = max(0.1, min(0.9, new_regime_acc))
            
            # PHASE 5: Track regime-specific performance (overall, not per indicator)
            if regime:
                regime_key = f"regime_{regime}"
                if not hasattr(self, 'regime_accuracy'):
                    self.regime_accuracy = {}
                if regime_key not in self.regime_accuracy:
                    self.regime_accuracy[regime_key] = {"wins": 0, "losses": 0}
                if is_win:
                    self.regime_accuracy[regime_key]["wins"] += 1
                else:
                    self.regime_accuracy[regime_key]["losses"] += 1

        # === PHASE 6: Advanced Learning ===
        
        # Update entry condition performance
        if entry_conditions:
            self.update_condition_performance(entry_conditions, is_win)
        
        # Update indicator combo tracking
        if indicator_combo_key:
            self.update_indicator_combo(indicator_combo_key, is_win)
        
        # Update optimal stop loss learning
        if max_drawdown is not None and regime and volatility:
            # Calculate what stop would have been optimal
            # For wins: optimal stop = max_drawdown (could have held through)
            # For losses: optimal stop = slightly wider than actual (to avoid early stop)
            if is_win:
                optimal_stop = max_drawdown
            else:
                # If we lost, the stop was probably too tight
                optimal_stop = abs(profit_percent) + 0.3  # Add buffer
            self.update_stop_loss_learning(regime, volatility, optimal_stop)
        
        # Analyze loss reason
        if not is_win and entry_conditions:
            exit_data = {
                "exit_type": exit_type,
                "profit_percent": profit_percent
            }
            loss_reason = self.analyze_loss(entry_conditions, exit_data)
            trade_record["loss_reason"] = loss_reason

        # PHASE 3: Save learning state to file for persistence
        self._save_learning_state()

    def _update_streak_multiplier(self):
        """
        Update position size multiplier based on recent performance
        - Win streak >= 2: Increase position by 20%
        - Loss streak >= 2: Decrease position by 50%
        - Otherwise: Normal size
        """
        if self.win_streak >= 3:
            self.streak_multiplier = 1.3  # Hot hand: +30%
        elif self.win_streak >= 2:
            self.streak_multiplier = 1.2  # Winning: +20%
        elif self.loss_streak >= 3:
            self.streak_multiplier = 0.4  # Cold: -60%
        elif self.loss_streak >= 2:
            self.streak_multiplier = 0.5  # Losing: -50%
        elif self.loss_streak >= 1:
            self.streak_multiplier = 0.75  # One loss: -25%
        else:
            self.streak_multiplier = 1.0  # Normal

    def get_streak_info(self) -> Dict:
        """Get current streak information for dashboard"""
        recent_wins = sum(1 for t in self.trade_history if t.get("is_win", False))
        recent_losses = len(self.trade_history) - recent_wins

        return {
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "streak_multiplier": self.streak_multiplier,
            "recent_trades": len(self.trade_history),
            "recent_wins": recent_wins,
            "recent_losses": recent_losses,
            "win_rate": round(recent_wins / len(self.trade_history) * 100, 1) if self.trade_history else 0,
            "status": self._get_streak_status()
        }

    def _get_streak_status(self) -> str:
        """Get human-readable streak status"""
        if self.win_streak >= 3:
            return "🔥🔥🔥 On Fire!"
        elif self.win_streak >= 2:
            return "🔥🔥 Hot Streak"
        elif self.win_streak == 1:
            return "🔥 Winning"
        elif self.loss_streak >= 3:
            return "❄️❄️❄️ Cold Streak"
        elif self.loss_streak >= 2:
            return "❄️❄️ Cooling Off"
        elif self.loss_streak == 1:
            return "❄️ One Loss"
        else:
            return "➡️ Neutral"

    # === PHASE 2: Anti-Whipsaw Cooldown Methods ===

    def _activate_cooldown(self):
        """Activate cooldown period after stop loss"""
        self.last_stop_loss_time = datetime.now()
        self.cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_minutes)

    def is_in_cooldown(self) -> bool:
        """Check if we're still in cooldown period"""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until

    def get_cooldown_remaining(self) -> int:
        """Get remaining cooldown time in seconds"""
        if self.cooldown_until is None:
            return 0
        remaining = (self.cooldown_until - datetime.now()).total_seconds()
        return max(0, int(remaining))

    def clear_cooldown(self):
        """Manually clear cooldown (e.g., when regime changes)"""
        self.cooldown_until = None

    def get_cooldown_info(self) -> Dict:
        """Get cooldown status for dashboard"""
        return {
            "active": self.is_in_cooldown(),
            "remaining_seconds": self.get_cooldown_remaining(),
            "remaining_minutes": round(self.get_cooldown_remaining() / 60, 1),
            "last_stop_loss": self.last_stop_loss_time.isoformat() if self.last_stop_loss_time else None,
            "cooldown_duration": self.cooldown_minutes
        }

    # === PHASE 6: Advanced Learning Methods ===

    def capture_entry_conditions(self, analysis: Dict, bot_state: Dict = None) -> Dict:
        """
        Capture market conditions at entry for learning.
        Called when executing a BUY to remember what conditions led to the trade.
        """
        conditions = {}
        
        # RSI condition
        rsi_val = analysis.get("rsi", {}).get("value", 50)
        if rsi_val < 30:
            conditions["rsi_state"] = "oversold"
        elif rsi_val > 70:
            conditions["rsi_state"] = "overbought"
        else:
            conditions["rsi_state"] = "neutral"
        conditions["rsi_value"] = rsi_val
        
        # Volatility level
        atr = analysis.get("atr", {})
        vol_level = atr.get("volatility_level", "medium")
        conditions["volatility"] = vol_level
        
        # Trend direction
        ema = analysis.get("ema", {})
        if ema.get("signal") == "bullish":
            conditions["trend"] = "up"
        elif ema.get("signal") == "bearish":
            conditions["trend"] = "down"
        else:
            conditions["trend"] = "sideways"
        
        # Support/Resistance position
        sr = analysis.get("support_resistance", {})
        sr_signal = sr.get("signal", "neutral")
        if "support" in sr_signal:
            conditions["sr_position"] = "at_support"
        elif "resistance" in sr_signal:
            conditions["sr_position"] = "at_resistance"
        else:
            conditions["sr_position"] = "middle"
        
        # Bollinger position
        bb = analysis.get("bollinger", {})
        bb_pos = bb.get("position", 50)
        if bb_pos < 20:
            conditions["bb_position"] = "near_lower"
        elif bb_pos > 80:
            conditions["bb_position"] = "near_upper"
        else:
            conditions["bb_position"] = "middle"
        
        # Hour of day (for time pattern learning)
        conditions["hour"] = datetime.now().hour
        
        # Day of week
        conditions["day_of_week"] = datetime.now().strftime("%A")
        
        # Market regime
        conditions["regime"] = bot_state.get("market_regime", "unknown") if bot_state else "unknown"
        
        # Fear & Greed (if available)
        if bot_state:
            sentiment = bot_state.get("sentiment", {})
            fg = sentiment.get("fear_greed_value", 50)
            if fg < 25:
                conditions["fear_greed"] = "extreme_fear"
            elif fg < 45:
                conditions["fear_greed"] = "fear"
            elif fg < 55:
                conditions["fear_greed"] = "neutral"
            elif fg < 75:
                conditions["fear_greed"] = "greed"
            else:
                conditions["fear_greed"] = "extreme_greed"
        
        # Confluence count
        conditions["confluence_count"] = bot_state.get("confluence", {}).get("count", 0) if bot_state else 0
        
        return conditions

    def get_entry_condition_score(self, conditions: Dict) -> float:
        """
        Get a score modifier based on learned entry condition performance.
        Returns a multiplier (0.5 to 1.5) to adjust confidence.
        """
        if not self.condition_performance:
            return 1.0  # No data yet
        
        total_score = 0
        count = 0
        
        # Check each condition against learned performance
        condition_keys = [
            f"rsi_{conditions.get('rsi_state', 'neutral')}",
            f"vol_{conditions.get('volatility', 'medium')}",
            f"trend_{conditions.get('trend', 'sideways')}",
            f"sr_{conditions.get('sr_position', 'middle')}",
            f"bb_{conditions.get('bb_position', 'middle')}",
            f"fg_{conditions.get('fear_greed', 'neutral')}",
            f"regime_{conditions.get('regime', 'unknown')}",
        ]
        
        for key in condition_keys:
            if key in self.condition_performance:
                perf = self.condition_performance[key]
                wins = perf.get("wins", 0)
                losses = perf.get("losses", 0)
                total = wins + losses
                if total >= 2:  # Need at least 2 samples
                    win_rate = wins / total
                    # Convert win rate to score: 50% = 1.0, 75% = 1.25, 25% = 0.75
                    total_score += 0.5 + win_rate
                    count += 1
        
        if count == 0:
            return 1.0
        
        avg_score = total_score / count
        # Clamp to reasonable range
        return max(0.5, min(1.5, avg_score))

    def update_condition_performance(self, conditions: Dict, is_win: bool):
        """
        Update condition performance tracking after a trade closes.
        """
        condition_keys = [
            f"rsi_{conditions.get('rsi_state', 'neutral')}",
            f"vol_{conditions.get('volatility', 'medium')}",
            f"trend_{conditions.get('trend', 'sideways')}",
            f"sr_{conditions.get('sr_position', 'middle')}",
            f"bb_{conditions.get('bb_position', 'middle')}",
            f"fg_{conditions.get('fear_greed', 'neutral')}",
            f"regime_{conditions.get('regime', 'unknown')}",
            f"hour_{conditions.get('hour', 12)}",
        ]
        
        for key in condition_keys:
            if key not in self.condition_performance:
                self.condition_performance[key] = {"wins": 0, "losses": 0}
            
            if is_win:
                self.condition_performance[key]["wins"] += 1
            else:
                self.condition_performance[key]["losses"] += 1

    def analyze_loss(self, entry_conditions: Dict, exit_data: Dict, bot_state: Dict = None) -> str:
        """
        Analyze WHY a loss occurred and categorize it.
        Returns the primary loss reason.
        """
        loss_reason = "unknown"
        
        exit_type = exit_data.get("exit_type", "unknown")
        profit_pct = exit_data.get("profit_percent", 0)
        
        # Check for stopped too early (price recovered after stop)
        if exit_type in ["stop_loss", "hard_stop"]:
            # If we have post-exit price data showing recovery
            post_exit_recovery = exit_data.get("price_recovered", False)
            if post_exit_recovery:
                loss_reason = "stopped_too_early"
            
            # Check if entered against trend
            elif entry_conditions.get("trend") == "down":
                loss_reason = "entered_against_trend"
            
            # Check for high volatility whipsaw
            elif entry_conditions.get("volatility") == "high":
                loss_reason = "high_volatility_whipsaw"
            
            # Check for weak confluence
            elif entry_conditions.get("confluence_count", 0) < 5:
                loss_reason = "weak_confluence"
            
            # Check regime
            elif entry_conditions.get("regime") in ["trending_down", "high_volatility"]:
                loss_reason = "wrong_regime"
            
            else:
                loss_reason = "normal_stop"
        
        # Update loss reason tracking
        if loss_reason not in self.loss_reasons:
            self.loss_reasons[loss_reason] = 0
        self.loss_reasons[loss_reason] += 1
        
        return loss_reason

    def get_loss_warnings(self, conditions: Dict) -> List[str]:
        """
        Check if current conditions match common loss patterns.
        Returns list of warnings.
        """
        warnings = []
        
        # Check each condition against loss patterns
        if conditions.get("trend") == "down":
            against_trend_losses = self.loss_reasons.get("entered_against_trend", 0)
            if against_trend_losses >= 2:
                warnings.append(f"Warning: {against_trend_losses} losses from entering against trend")
        
        if conditions.get("volatility") == "high":
            vol_losses = self.loss_reasons.get("high_volatility_whipsaw", 0)
            if vol_losses >= 2:
                warnings.append(f"Warning: {vol_losses} losses from high volatility")
        
        if conditions.get("confluence_count", 0) < 5:
            weak_losses = self.loss_reasons.get("weak_confluence", 0)
            if weak_losses >= 2:
                warnings.append(f"Warning: {weak_losses} losses from weak confluence")
        
        if conditions.get("regime") in ["trending_down", "high_volatility"]:
            regime_losses = self.loss_reasons.get("wrong_regime", 0)
            if regime_losses >= 2:
                warnings.append(f"Warning: {regime_losses} losses in unfavorable regime")
        
        return warnings

    def update_stop_loss_learning(self, regime: str, volatility: str, optimal_stop: float):
        """
        Update learned optimal stop loss for regime/volatility combination.
        """
        key = f"{regime}_{volatility}"
        
        if key not in self.stop_loss_learning:
            self.stop_loss_learning[key] = {"avg_optimal": optimal_stop, "samples": 1}
        else:
            # Running average
            old_avg = self.stop_loss_learning[key]["avg_optimal"]
            samples = self.stop_loss_learning[key]["samples"]
            new_avg = (old_avg * samples + optimal_stop) / (samples + 1)
            self.stop_loss_learning[key]["avg_optimal"] = new_avg
            self.stop_loss_learning[key]["samples"] = samples + 1

    def get_recommended_stop(self, regime: str, volatility: str) -> float:
        """
        Get recommended stop loss based on learned data.
        Returns percentage (e.g., 1.5 for 1.5%)
        """
        key = f"{regime}_{volatility}"
        
        if key in self.stop_loss_learning:
            data = self.stop_loss_learning[key]
            if data.get("samples", 0) >= 3:  # Need at least 3 samples
                return data["avg_optimal"]
        
        # Fallback to config defaults
        if regime == "trending_down":
            return getattr(config, 'STOP_LOSS_TRENDING_DOWN', 0.008) * 100
        elif volatility == "high":
            return getattr(config, 'STOP_LOSS_HIGH_VOL', 0.012) * 100
        else:
            return getattr(config, 'STOP_LOSS', 0.01) * 100

    def get_indicator_combo_key(self, analysis: Dict) -> str:
        """
        Generate a key representing the current indicator combination.
        """
        signals = []
        
        # RSI
        rsi = analysis.get("rsi", {})
        if rsi.get("signal") == "oversold":
            signals.append("rsi_bullish")
        elif rsi.get("signal") == "overbought":
            signals.append("rsi_bearish")
        
        # MACD
        macd = analysis.get("macd", {})
        if macd.get("signal") == "bullish":
            signals.append("macd_bullish")
        elif macd.get("signal") == "bearish":
            signals.append("macd_bearish")
        
        # Bollinger
        bb = analysis.get("bollinger", {})
        if bb.get("signal") in ["oversold", "near_lower"]:
            signals.append("bb_bullish")
        elif bb.get("signal") in ["overbought", "near_upper"]:
            signals.append("bb_bearish")
        
        # Support/Resistance
        sr = analysis.get("support_resistance", {})
        if "support" in sr.get("signal", ""):
            signals.append("sr_support")
        elif "resistance" in sr.get("signal", ""):
            signals.append("sr_resistance")
        
        # Stochastic
        stoch = analysis.get("stochastic", {})
        if stoch.get("signal") in ["bullish", "bullish_crossover"]:
            signals.append("stoch_bullish")
        elif stoch.get("signal") in ["bearish", "bearish_crossover"]:
            signals.append("stoch_bearish")
        
        return "+".join(sorted(signals)) if signals else "no_signals"

    def update_indicator_combo(self, combo_key: str, is_win: bool):
        """
        Update indicator combo performance tracking.
        """
        if combo_key not in self.indicator_combos:
            self.indicator_combos[combo_key] = {"wins": 0, "losses": 0}
        
        if is_win:
            self.indicator_combos[combo_key]["wins"] += 1
        else:
            self.indicator_combos[combo_key]["losses"] += 1

    def get_combo_score(self, combo_key: str) -> float:
        """
        Get win rate for the current indicator combo.
        Returns a score multiplier (0.5 to 1.5).
        """
        if combo_key not in self.indicator_combos:
            return 1.0
        
        data = self.indicator_combos[combo_key]
        wins = data.get("wins", 0)
        losses = data.get("losses", 0)
        total = wins + losses
        
        if total < 3:  # Need at least 3 samples
            return 1.0
        
        win_rate = wins / total
        # Convert: 50% = 1.0, 80% = 1.3, 20% = 0.7
        return 0.5 + win_rate

    def get_learning_summary(self) -> Dict:
        """
        Get a summary of all learning data for dashboard display.
        PHASE 7: Enhanced with indicator learning stats
        """
        # Best conditions
        best_conditions = []
        for key, data in self.condition_performance.items():
            total = data.get("wins", 0) + data.get("losses", 0)
            if total >= 3:
                win_rate = data["wins"] / total * 100
                best_conditions.append({"condition": key, "win_rate": win_rate, "samples": total})
        best_conditions.sort(key=lambda x: x["win_rate"], reverse=True)
        
        # Best combos
        best_combos = []
        for key, data in self.indicator_combos.items():
            total = data.get("wins", 0) + data.get("losses", 0)
            if total >= 3:
                win_rate = data["wins"] / total * 100
                best_combos.append({"combo": key, "win_rate": win_rate, "samples": total})
        best_combos.sort(key=lambda x: x["win_rate"], reverse=True)
        
        # Top loss reasons
        loss_reasons_sorted = sorted(self.loss_reasons.items(), key=lambda x: x[1], reverse=True)
        
        # === PHASE 7: Enhanced Indicator Learning Stats ===
        # Per-indicator accuracy with confidence level
        indicator_stats = []
        min_samples = getattr(config, 'MIN_SAMPLES_FOR_LEARNING', 8)
        for ind in self.weights:
            accuracy = self.indicator_accuracy.get(ind, 0.5)
            samples = self.indicator_samples.get(ind, 0) if hasattr(self, 'indicator_samples') else 0
            
            # Confidence: high if enough samples and clear signal
            if samples >= min_samples and abs(accuracy - 0.5) >= 0.08:
                confidence = "high"
            elif samples >= 5:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Directional accuracy
            dir_data = self.directional_accuracy.get(ind, {}) if hasattr(self, 'directional_accuracy') else {}
            bullish_acc = dir_data.get("bullish", 0.5)
            bearish_acc = dir_data.get("bearish", 0.5)
            bullish_samples = dir_data.get("bullish_samples", 0)
            bearish_samples = dir_data.get("bearish_samples", 0)
            
            indicator_stats.append({
                "name": ind,
                "accuracy": round(accuracy * 100, 1),
                "samples": samples,
                "confidence": confidence,
                "bullish_accuracy": round(bullish_acc * 100, 1),
                "bearish_accuracy": round(bearish_acc * 100, 1),
                "bullish_samples": bullish_samples,
                "bearish_samples": bearish_samples,
                "effective_weight": round(self.weights.get(ind, 0) * 100, 1)
            })
        
        # Sort by accuracy (best first)
        indicator_stats.sort(key=lambda x: x["accuracy"], reverse=True)
        
        # Regime performance summary
        regime_stats = []
        for regime_key, data in getattr(self, 'regime_accuracy', {}).items():
            regime_name = regime_key.replace("regime_", "")
            wins = data.get("wins", 0)
            losses = data.get("losses", 0)
            total = wins + losses
            if total > 0:
                regime_stats.append({
                    "regime": regime_name,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(wins / total * 100, 1),
                    "total": total
                })
        
        # Best/worst indicators
        best_indicators = indicator_stats[:3]
        worst_indicators = sorted(indicator_stats, key=lambda x: x["accuracy"])[:3]
        
        return {
            "total_trades_learned": len(self.trade_history),
            # PHASE 7: Enhanced indicator learning
            "indicator_stats": indicator_stats,
            "best_indicators": best_indicators,
            "worst_indicators": worst_indicators,
            "regime_stats": regime_stats,
            # Existing learning data
            "condition_performance": self.condition_performance,
            "best_conditions": best_conditions[:5],
            "worst_conditions": best_conditions[-5:] if len(best_conditions) > 5 else [],
            "stop_loss_learning": self.stop_loss_learning,
            "loss_reasons": dict(loss_reasons_sorted[:5]),
            "indicator_combos": self.indicator_combos,
            "best_combos": best_combos[:5],
        }

    def calculate_confluence(self, analysis: Dict) -> Dict:
        """
        Calculate how many indicators agree on direction
        This prevents false signals from single indicators!

        PHASE 4: Now uses 10 indicators (6 original + 4 new)

        Returns:
            Confluence data with count, direction, and agreeing indicators
        """
        bullish_indicators = []
        bearish_indicators = []
        neutral_indicators = []

        # PHASE 4: Check 10 indicators (including ML prediction + new Phase 4 indicators)
        indicators = {
            "Momentum": analysis.get("momentum", {}),
            "MACD": analysis.get("macd", {}),
            "Bollinger": analysis.get("bollinger", {}),
            "EMA": analysis.get("ema", {}),
            "S/R": analysis.get("support_resistance", {}),
            "ML": analysis.get("ml_prediction", {}),
            # Phase 4 NEW
            "Ichimoku": analysis.get("ichimoku", {}),
            "MFI": analysis.get("mfi", {}),
            "Williams": analysis.get("williams_r", {}),
            "CCI": analysis.get("cci", {})
        }

        # Also check Fibonacci for extra boost (not counted in confluence but affects score)
        fib_data = analysis.get("fibonacci", {})
        fib_boost = 0
        if fib_data.get("at_fib") and fib_data.get("score", 0) != 0:
            fib_boost = fib_data.get("score", 0) * 0.2  # 20% of Fib score as boost

        # Check candle patterns for extra boost
        candle_data = analysis.get("candle_patterns", {})
        candle_boost = 0
        if candle_data.get("pattern_count", 0) > 0:
            candle_boost = candle_data.get("score", 0) * 0.15  # 15% of candle score

        for name, data in indicators.items():
            score = data.get("score", 0)
            if score > 0.25:  # Bullish
                bullish_indicators.append(name)
            elif score < -0.25:  # Bearish
                bearish_indicators.append(name)
            else:  # Neutral
                neutral_indicators.append(name)

        # Determine overall confluence
        bullish_count = len(bullish_indicators)
        bearish_count = len(bearish_indicators)

        if bullish_count > bearish_count:
            direction = "bullish"
            confluence_count = bullish_count
            agreeing = bullish_indicators
        elif bearish_count > bullish_count:
            direction = "bearish"
            confluence_count = bearish_count
            agreeing = bearish_indicators
        else:
            direction = "neutral"
            confluence_count = 0
            agreeing = []

        # PHASE 4: Determine strength (10 indicators with ML + Phase 4)
        if confluence_count >= 8:
            strength = "very_strong"
        elif confluence_count >= 6:
            strength = "strong"
        elif confluence_count >= 4:
            strength = "moderate"
        elif confluence_count >= 2:
            strength = "weak"
        else:
            strength = "none"

        return {
            "direction": direction,
            "count": confluence_count,
            "total": 10,  # 10 indicators with ML + Phase 4
            "agreeing_indicators": agreeing,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": len(neutral_indicators),
            "strength": strength,
            "fib_boost": round(fib_boost, 3),
            "candle_boost": round(candle_boost, 3)
        }

    def detect_divergence(self, df: pd.DataFrame, analysis: Dict) -> Dict:
        """
        Detect RSI divergence - when price and RSI disagree
        This is an EARLY reversal signal!

        PHASE 3: Extended to 20-candle lookback with strength levels:
        - 15+ candles = Strong divergence (+0.25)
        - 10-15 candles = Moderate divergence (+0.15)
        - <10 candles = Weak divergence (+0.10)

        Bullish divergence: Price making lower lows but RSI making higher lows
        Bearish divergence: Price making higher highs but RSI making lower highs
        """
        lookback_config = getattr(config, 'DIVERGENCE_LOOKBACK', 20)

        if df is None or df.empty or len(df) < lookback_config + 5:
            return {"detected": False, "type": None, "score_adjustment": 0}

        try:
            close = df['close']

            # Calculate RSI series
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / (avg_loss + 0.0001)
            rsi = 100 - (100 / (1 + rs))

            # PHASE 3: Extended lookback (20 candles)
            lookback = min(lookback_config, len(close) - 1)
            if lookback < 5:
                return {"detected": False, "type": None, "score_adjustment": 0}

            recent_prices = close.tail(lookback)
            recent_rsi = rsi.tail(lookback)

            # Find price lows/highs and corresponding RSI values
            price_min_idx = recent_prices.idxmin()
            price_max_idx = recent_prices.idxmax()

            price_start = recent_prices.iloc[0]
            price_end = recent_prices.iloc[-1]
            rsi_start = recent_rsi.iloc[0]
            rsi_end = recent_rsi.iloc[-1]

            # Handle NaN
            if any(np.isnan([price_start, price_end, rsi_start, rsi_end])):
                return {"detected": False, "type": None, "score_adjustment": 0}

            # Calculate divergence over different windows for strength
            divergence_candles = 0
            is_bullish = False
            is_bearish = False

            # Check multiple windows for strongest divergence
            for window in [20, 15, 10, 7]:
                if len(close) < window + 5:
                    continue

                w_prices = close.tail(window)
                w_rsi = rsi.tail(window)

                w_price_start = w_prices.iloc[0]
                w_price_end = w_prices.iloc[-1]
                w_rsi_start = w_rsi.iloc[0]
                w_rsi_end = w_rsi.iloc[-1]

                if any(np.isnan([w_price_start, w_price_end, w_rsi_start, w_rsi_end])):
                    continue

                # Bullish divergence: price lower, RSI higher
                price_lower = w_price_end < w_price_start * 0.99
                rsi_higher = w_rsi_end > w_rsi_start + 3
                if price_lower and rsi_higher:
                    is_bullish = True
                    divergence_candles = max(divergence_candles, window)

                # Bearish divergence: price higher, RSI lower
                price_higher = w_price_end > w_price_start * 1.01
                rsi_lower = w_rsi_end < w_rsi_start - 3
                if price_higher and rsi_lower:
                    is_bearish = True
                    divergence_candles = max(divergence_candles, window)

            # PHASE 3: Strength-based score adjustment
            strong_candles = getattr(config, 'DIVERGENCE_STRONG_CANDLES', 15)
            moderate_candles = getattr(config, 'DIVERGENCE_MODERATE_CANDLES', 10)

            if divergence_candles >= strong_candles:
                strength = "strong"
                base_adjustment = getattr(config, 'DIVERGENCE_STRONG_BOOST', 0.25)
            elif divergence_candles >= moderate_candles:
                strength = "moderate"
                base_adjustment = getattr(config, 'DIVERGENCE_MODERATE_BOOST', 0.15)
            elif divergence_candles > 0:
                strength = "weak"
                base_adjustment = getattr(config, 'DIVERGENCE_WEAK_BOOST', 0.10)
            else:
                strength = "none"
                base_adjustment = 0

            if is_bullish:
                return {
                    "detected": True,
                    "type": "bullish",
                    "strength": strength,
                    "candles": divergence_candles,
                    "score_adjustment": base_adjustment,
                    "description": f"Bullish divergence ({strength}) - {divergence_candles} candles"
                }
            elif is_bearish:
                return {
                    "detected": True,
                    "type": "bearish",
                    "strength": strength,
                    "candles": divergence_candles,
                    "score_adjustment": -base_adjustment,
                    "description": f"Bearish divergence ({strength}) - {divergence_candles} candles"
                }
            else:
                return {
                    "detected": False,
                    "type": None,
                    "strength": "none",
                    "candles": 0,
                    "score_adjustment": 0,
                    "description": "No divergence"
                }

        except Exception as e:
            return {"detected": False, "type": None, "score_adjustment": 0, "error": str(e)}

    def calculate_score(self, analysis: Dict, df: pd.DataFrame = None) -> Tuple[float, Dict]:
        """
        Calculate weighted AI score with smart enhancements

        PHASE 3: Now uses 5 indicators (Combined Momentum replaces RSI+Stochastic)

        Returns:
            Tuple of (score, enhanced_details)
        """
        if "error" in analysis:
            return 0.0, {"error": analysis["error"]}
        
        # Initialize enhanced_details early so we can add to it throughout the function
        enhanced_details = {}

        # Get regime data if we have price data
        if df is not None and not df.empty:
            self.regime_data = self.regime_detector.detect_regime(df)
            self.current_regime = self.regime_data.get("regime", MarketRegime.UNKNOWN)

        # Add ML prediction to analysis for confluence (before calculate_confluence)
        # Throttle: run ML at most every ML_INTERVAL_SEC to avoid blocking the loop
        ml_score = 0
        ml_result = {}
        if self.ml_predictor and df is not None and not df.empty and len(df) >= 100:
            try:
                now_ts = time.time()
                interval = getattr(config, "ML_INTERVAL_SEC", 300)
                last_ml_time = getattr(self, "_last_ml_time", 0)
                last_ml_result = getattr(self, "_last_ml_result", None)
                if last_ml_result is not None and last_ml_time and (now_ts - last_ml_time) < interval:
                    ml_score, ml_result = last_ml_result
                else:
                    ml_score, ml_result = self.ml_predictor.get_score(df)
                    self._last_ml_time = now_ts
                    self._last_ml_result = (ml_score, ml_result)
                    # Record prediction so we can update outcome when position closes (ML learning)
                    try:
                        self.ml_predictor.record_prediction(ml_result)
                    except Exception:
                        pass
                analysis = dict(analysis)
                analysis["ml_prediction"] = {
                    "score": ml_score,
                    "direction": ml_result.get("direction", "HOLD"),
                    "confidence": ml_result.get("confidence", 0),
                    "probability": ml_result.get("probability", 0.5),
                    "model_votes": ml_result.get("model_votes", {}),
                    "models_loaded": ml_result.get("models_loaded", False),
                    "predictions": ml_result.get("predictions", {}),
                    "consensus": ml_result.get("consensus", ""),
                    "short_term": ml_result.get("short_term", ""),
                    "long_term": ml_result.get("long_term", ""),
                }
            except Exception:
                analysis = dict(analysis)
                analysis["ml_prediction"] = {"score": 0, "direction": "HOLD", "confidence": 0, "models_loaded": False}
        else:
            analysis = dict(analysis)
            analysis["ml_prediction"] = {"score": 0, "direction": "HOLD", "confidence": 0, "models_loaded": False}

        # PHASE 4: Base indicator scores (10 indicators with ML prediction)
        # Get regime context for accurate interpretation
        regime_name = None
        if self.regime_data:
            regime_name = self.regime_data.get("regime_name", "")
        
        base_scores = {
            "momentum": analysis.get("momentum", {}).get("score", 0),
            "macd": analysis.get("macd", {}).get("score", 0),
            "bollinger": analysis.get("bollinger", {}).get("score", 0),
            "ema": analysis.get("ema", {}).get("score", 0),
            "support_resistance": analysis.get("support_resistance", {}).get("score", 0),
            "ml_prediction": ml_score,
            # Phase 4 NEW indicators
            "ichimoku": analysis.get("ichimoku", {}).get("score", 0),
            "mfi": analysis.get("mfi", {}).get("score", 0),
            "williams_r": analysis.get("williams_r", {}).get("score", 0),
            "cci": analysis.get("cci", {}).get("score", 0),
        }
        
        # Regime-aware indicator adjustment: Make indicators interpret signals correctly based on market context
        scores = self._adjust_indicator_scores_for_regime(base_scores, analysis, regime_name)

        # Calculate weighted average (use adaptive weights if enabled and have history)
        # PHASE 7: Pass current regime for regime-specific weights
        current_regime = self.current_regime.value if self.current_regime else None
        weights = self._get_effective_weights(regime=current_regime)
        weighted_score = sum(
            scores.get(indicator, 0) * weights.get(indicator, 0)
            for indicator in weights
        )

        # Calculate confluence
        confluence = self.calculate_confluence(analysis)

        # PHASE 3: Add Fibonacci and Candle pattern boosts from confluence
        fib_boost = confluence.get("fib_boost", 0)
        candle_boost = confluence.get("candle_boost", 0)
        weighted_score += fib_boost + candle_boost

        # Check for divergence (PHASE 3: Extended lookback)
        divergence = self.detect_divergence(df, analysis)
        if divergence.get("detected"):
            weighted_score += divergence.get("score_adjustment", 0)

        # Regime-based boost: If market is trending up, boost score even if indicators are mixed
        # This helps catch uptrends where indicators lag or show overbought conditions
        if self.regime_data:
            regime_name = self.regime_data.get("regime_name", "")
            if regime_name == "trending_up":
                # Boost score by 0.15-0.25 in uptrends to account for indicator lag/overbought conditions
                uptrend_boost = getattr(config, "UPTREND_SCORE_BOOST", 0.15)
                weighted_score += uptrend_boost
                enhanced_details["regime_boost"] = {
                    "regime": "trending_up",
                    "boost": uptrend_boost,
                    "reason": "Uptrend detected - boosting score to account for indicator lag"
                }

        # Volume confirmation adjustment (PHASE 3: Now a modifier, not blocker)
        volume_data = analysis.get("volume", {})
        volume_confirmed = False
        if volume_data.get("confirmation", False):
            weighted_score *= 1.1  # 10% boost for volume confirmation
            volume_confirmed = True
        elif not getattr(config, 'REQUIRE_VOLUME_BLOCKING', False):
            # PHASE 3: Volume no longer blocks, just reduces score
            penalty = getattr(config, 'VOLUME_NO_CONFIRM_PENALTY', 0.15)
            weighted_score *= (1 - penalty)  # Reduce score by penalty (15%)

        # === PHASE 3: 15m Timeframe Refinement ===
        refinement_15m = {"applied": False, "action": None, "boost": 0}
        if getattr(config, 'USE_15M_REFINEMENT', True) and self.last_15m_analysis:
            stoch_15m = self.last_15m_analysis.get("stochastic", {})
            stoch_15m_k = stoch_15m.get("k", 50)

            overbought_15m = getattr(config, 'STOCH_15M_OVERBOUGHT', 75)
            oversold_15m = getattr(config, 'STOCH_15M_OVERSOLD', 25)
            entry_boost = getattr(config, 'ENTRY_REFINEMENT_BOOST', 0.10)

            # When 1h says BUY but 15m overbought - wait for pullback
            if weighted_score > 0 and stoch_15m_k > overbought_15m:
                refinement_15m = {
                    "applied": True,
                    "action": "wait_pullback",
                    "boost": -0.15,  # Reduce score to delay entry
                    "reason": f"15m overbought ({stoch_15m_k:.0f}) - wait for pullback"
                }
                weighted_score += refinement_15m["boost"]

            # When 1h says BUY and 15m oversold - confirm entry
            elif weighted_score > 0 and stoch_15m_k < oversold_15m:
                refinement_15m = {
                    "applied": True,
                    "action": "confirm_entry",
                    "boost": entry_boost,
                    "reason": f"15m oversold ({stoch_15m_k:.0f}) - confirms entry"
                }
                weighted_score += refinement_15m["boost"]

        # Keep legacy stochastic data for dashboard
        stoch_data = analysis.get("stochastic", {})

        # === PHASE 2: Bollinger Squeeze Alert ===
        squeeze_data = analysis.get("squeeze", {})
        squeeze_alert = None
        if squeeze_data.get("is_squeeze"):
            squeeze_alert = {
                "active": True,
                "intensity": squeeze_data.get("intensity", "moderate"),
                "breakout_bias": squeeze_data.get("breakout_bias", "neutral"),
                "alert": squeeze_data.get("alert", "Squeeze detected")
            }
            # In a squeeze, we wait for direction - don't trade yet
            # But if there's a clear bias with other signals, boost slightly
            if squeeze_data.get("breakout_bias") == "bullish" and weighted_score > 0:
                weighted_score *= 1.05
            elif squeeze_data.get("breakout_bias") == "bearish" and weighted_score < 0:
                weighted_score *= 1.05

        # === PHASE 4: External Data Integration ===
        order_book_data = None
        sentiment_data = None
        correlation_data = None
        external_filters = {"applied": False, "adjustments": []}
        
        # Order Book Analysis (adds to confluence, not score)
        if self.order_book_analyzer:
            try:
                order_book_data = self.order_book_analyzer.analyze()
                self._order_book_data = order_book_data
                
                # Use order book as confirmation filter
                ob_signal = order_book_data.get("signal", "neutral")
                ob_score = order_book_data.get("score", 0)
                
                # If order book strongly disagrees with our signal, reduce score
                if weighted_score > 0.3 and ob_score < -0.3:
                    adjustment = -0.1
                    weighted_score += adjustment
                    external_filters["applied"] = True
                    external_filters["adjustments"].append(f"Order book bearish ({ob_signal}): {adjustment:+.2f}")
                elif weighted_score < -0.3 and ob_score > 0.3:
                    adjustment = 0.1
                    weighted_score += adjustment
                    external_filters["applied"] = True
                    external_filters["adjustments"].append(f"Order book bullish ({ob_signal}): {adjustment:+.2f}")
                # If order book confirms our signal, small boost
                elif weighted_score > 0 and ob_score > 0.2:
                    adjustment = 0.05
                    weighted_score += adjustment
                    external_filters["adjustments"].append(f"Order book confirms buy: {adjustment:+.2f}")
            except Exception:
                pass
        
        # Sentiment Analysis (contrarian filter)
        if self.sentiment_analyzer:
            try:
                sentiment_data = self.sentiment_analyzer.analyze()
                self._sentiment_data = sentiment_data
                
                # Avoid buying in extreme greed
                if sentiment_data.get("should_avoid_buy", False) and weighted_score > 0:
                    adjustment = -0.15
                    weighted_score += adjustment
                    external_filters["applied"] = True
                    external_filters["adjustments"].append(f"Extreme greed warning: {adjustment:+.2f}")
                
                # Boost buys in extreme fear (contrarian)
                elif sentiment_data.get("should_buy", False) and weighted_score > 0:
                    adjustment = 0.1
                    weighted_score += adjustment
                    external_filters["adjustments"].append(f"Fear = opportunity: {adjustment:+.2f}")
            except Exception:
                pass
        
        # Correlation Analysis (market context)
        if self.correlation_analyzer:
            try:
                correlation_data = self.correlation_analyzer.analyze()
                self._correlation_data = correlation_data
                
                # If overall market is very bearish, be more cautious on buys
                corr_score = correlation_data.get("score", 0)
                if corr_score < -0.3 and weighted_score > 0:
                    adjustment = -0.05
                    weighted_score += adjustment
                    external_filters["adjustments"].append(f"Market context bearish: {adjustment:+.2f}")
            except Exception:
                pass
        
        # Clamp score to -1 to 1
        weighted_score = max(-1, min(1, weighted_score))

        # Update enhanced_details with all calculated values
        enhanced_details.update({
            "base_score": round(weighted_score, 4),
            "regime": self.regime_data,
            "confluence": confluence,
            "divergence": divergence,
            "volume_confirmed": volume_confirmed,
            "indicator_scores": scores,
            "ml_prediction": analysis.get("ml_prediction", {}),  # ML ensemble result
            "stochastic": stoch_data,
            "momentum": analysis.get("momentum", {}),  # PHASE 3
            "fibonacci": analysis.get("fibonacci", {}),  # PHASE 3
            "candle_patterns": analysis.get("candle_patterns", {}),  # PHASE 3
            "refinement_15m": refinement_15m,  # PHASE 3
            "fib_boost": fib_boost,
            "candle_boost": candle_boost,
            "squeeze": squeeze_data,
            "squeeze_alert": squeeze_alert,
            "streak_info": self.get_streak_info(),
            "cooldown_info": self.get_cooldown_info(),
            # Phase 4 NEW: External data
            "ichimoku": analysis.get("ichimoku", {}),
            "mfi": analysis.get("mfi", {}),
            "williams_r": analysis.get("williams_r", {}),
            "cci": analysis.get("cci", {}),
            "order_book": order_book_data,
            "sentiment": sentiment_data,
            "correlation": correlation_data,
            "external_filters": external_filters
        })

        return round(weighted_score, 4), enhanced_details

    def calculate_confidence(self, score: float, enhanced: Dict) -> Dict:
        """
        Calculate overall confidence level based on multiple factors
        Higher confidence = more reliable signal!

        PHASE 4: Now uses 10 indicators for confluence
        """
        # Base confidence from score magnitude
        score_confidence = abs(score)

        # Confluence factor (0 to 1) - PHASE 4: 10 indicators with ML
        confluence = enhanced.get("confluence", {})
        confluence_factor = confluence.get("count", 0) / 10.0

        # Volume boost
        volume_boost = 1.1 if enhanced.get("volume_confirmed") else 1.0

        # Divergence boost (PHASE 3: Stronger boost for strong divergence)
        divergence = enhanced.get("divergence", {})
        if divergence.get("strength") == "strong":
            divergence_boost = 1.25
        elif divergence.get("detected"):
            divergence_boost = 1.15
        else:
            divergence_boost = 1.0

        # PHASE 3: Fibonacci level boost
        fib_boost = 1.1 if enhanced.get("fibonacci", {}).get("at_fib") else 1.0

        # PHASE 3: Candle pattern boost
        candle_boost = 1.1 if enhanced.get("candle_patterns", {}).get("pattern_count", 0) > 0 else 1.0

        # PHASE 3: 15m refinement boost
        refinement = enhanced.get("refinement_15m", {})
        refinement_boost = 1.1 if refinement.get("action") == "confirm_entry" else 1.0

        # Combined confidence
        combined = (score_confidence * 0.35 + confluence_factor * 0.5) * volume_boost * divergence_boost
        combined *= fib_boost * candle_boost * refinement_boost
        combined = min(1.0, combined)

        # Determine level
        if combined >= 0.75:
            level = "Very High"
        elif combined >= 0.55:
            level = "High"
        elif combined >= 0.35:
            level = "Medium"
        elif combined >= 0.2:
            level = "Low"
        else:
            level = "Very Low"

        return {
            "level": level,
            "value": round(combined, 2),
            "factors": {
                "score_strength": round(score_confidence, 2),
                "confluence": round(confluence_factor, 2),
                "volume_boost": volume_boost > 1.0,
                "divergence_boost": divergence_boost > 1.0,
                "fib_boost": fib_boost > 1.0,
                "candle_boost": candle_boost > 1.0,
                "refinement_15m": refinement.get("action")
            }
        }

    def get_decision(
        self,
        analysis: Dict,
        current_position: Optional[Dict] = None,
        df: pd.DataFrame = None,
        mtf_trend_4h: Optional[str] = None,
        deep_insights: Optional[Dict] = None,
        performance_context: Optional[Dict] = None,
    ) -> Tuple[Decision, Dict]:
        """
        Generate smart trading decision with regime-awareness.

        Now checks:
        - Regime-adjusted thresholds
        - Confluence requirements
        - Divergence signals
        - Volume confirmation
        - Deep insights (weaknesses) to avoid repeating loss patterns
        """
        score, enhanced = self.calculate_score(analysis, df)
        current_price = analysis.get("current_price", 0)

        # Get regime-adjusted thresholds
        buy_threshold = self.buy_threshold
        sell_threshold = self.sell_threshold
        min_confluence = self.min_confluence

        # Use lower threshold for uptrending markets (safer to enter earlier)
        regime_name = None
        if self.regime_data:
            regime_name = self.regime_data.get("regime_name", "")
            if regime_name == "trending_up":
                buy_threshold_uptrend = getattr(config, "BUY_THRESHOLD_UPTREND", 0.20)
                buy_threshold = min(buy_threshold, buy_threshold_uptrend)  # Use the lower of the two

        # Apply deep_insights: if we have identified weaknesses, require stronger signal for BUY
        if deep_insights and deep_insights.get("weaknesses"):
            weakness_types = {w.get("weakness") for w in deep_insights["weaknesses"]}
            if "low_win_rate" in weakness_types or "poor_risk_reward" in weakness_types:
                buy_threshold += getattr(config, "DEEP_INSIGHT_EXTRA_THRESHOLD", 0.05)
                min_confluence = min_confluence + 1  # require one more agreeing indicator

        # Performance-based: if recent win rate is low, require stronger signal
        if performance_context and performance_context.get("recent_trades", 0) >= 5:
            recent_wr = performance_context.get("recent_win_rate")
            if recent_wr is not None and recent_wr < getattr(config, "PERF_BAD_WIN_RATE_THRESHOLD", 40):
                buy_threshold += getattr(config, "PERF_EXTRA_THRESHOLD", 0.05)
                min_confluence = min_confluence + 1

        if self.regime_data:
            params = self.regime_data.get("adjusted_params", {})
            # Don't override uptrend threshold if regime is uptrend
            if regime_name != "trending_up":
                buy_threshold = params.get("buy_threshold", buy_threshold)
            sell_threshold = params.get("sell_threshold", sell_threshold)
            min_confluence = params.get("confluence_required", min_confluence)

        # Calculate confidence
        confidence = self.calculate_confidence(score, enhanced)

        details = {
            "score": score,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "indicators": self._get_indicator_summary(analysis),
            "indicator_scores": enhanced.get("indicator_scores", {}),
            "reason": [],
            "regime": self.regime_data,
            "confluence": enhanced.get("confluence"),
            "divergence": enhanced.get("divergence"),
            "confidence": confidence,
            "volume_confirmed": enhanced.get("volume_confirmed", False),
            "ml_prediction": enhanced.get("ml_prediction", {}),
            # Phase 4: External data
            "order_book": enhanced.get("order_book"),
            "sentiment": enhanced.get("sentiment"),
            "correlation": enhanced.get("correlation"),
            "ichimoku": enhanced.get("ichimoku"),
            "mfi": enhanced.get("mfi"),
            "williams_r": enhanced.get("williams_r"),
            "cci": enhanced.get("cci"),
        }

        # Check confluence
        confluence = enhanced.get("confluence", {})
        has_confluence = confluence.get("count", 0) >= min_confluence
        confluence_direction = confluence.get("direction", "neutral")

        # === POSITION EXIT LOGIC ===
        if current_position:
            entry_price = current_position.get("entry_price", 0)
            if entry_price > 0:
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                details["pnl_percent"] = round(pnl_percent, 2)

                # Get regime-adjusted targets
                profit_mult = 1.0
                if self.regime_data:
                    profit_mult = self.regime_data.get("adjusted_params", {}).get("profit_target_multiplier", 1.0)

                adjusted_profit_target = config.PROFIT_TARGET * 100 * profit_mult
                adjusted_stop_loss = config.STOP_LOSS * 100

                # Check minimum profit - AUTO SELL as soon as we have profit
                min_profit = getattr(config, 'MIN_PROFIT', 0.005) * 100  # Convert to %
                if pnl_percent >= min_profit:
                    details["reason"].append(f"Min profit reached: +{pnl_percent:.1f}% (target: {min_profit:.1f}%) - Taking profit!")
                    details["exit_type"] = "min_profit"
                    return Decision.SELL, details

                # Check stop loss
                if pnl_percent <= -adjusted_stop_loss:
                    details["reason"].append(f"Stop loss triggered: {pnl_percent:.1f}%")
                    details["exit_type"] = "stop_loss"
                    return Decision.SELL, details

                # Check AI bearish signal with confluence
                # When position is in loss, require stronger bearish signal (SELL_THRESHOLD_IN_LOSS)
                sell_threshold_to_use = sell_threshold
                if pnl_percent < 0:
                    sell_threshold_to_use = getattr(config, 'SELL_THRESHOLD_IN_LOSS', -0.35)
                    if score >= sell_threshold_to_use:
                        # Position in loss but score not bearish enough - don't sell
                        details["reason"].append(f"Position in loss ({pnl_percent:.2f}%) but AI score {score:.2f} >= {sell_threshold_to_use} (need stronger bearish)")
                        return Decision.HOLD, details
                
                if score < sell_threshold_to_use and confluence_direction == "bearish" and has_confluence:
                    details["reason"].append(f"AI bearish signal: {score:.2f} with {confluence.get('count')}/10 confluence")
                    details["exit_type"] = "ai_signal"
                    return Decision.SELL, details

                # Check bearish divergence (early warning)
                divergence = enhanced.get("divergence", {})
                if divergence.get("type") == "bearish" and pnl_percent > 3:
                    details["reason"].append("Bearish divergence detected - taking profit early")
                    details["exit_type"] = "divergence_exit"
                    return Decision.SELL, details

                # Hold position
                details["reason"].append("Holding position - no exit signals")
                return Decision.HOLD, details

        # === ENTRY LOGIC ===

        # PHASE 2: Check cooldown first!
        if self.is_in_cooldown():
            remaining = self.get_cooldown_remaining()
            details["reason"].append(f"🛑 Anti-whipsaw cooldown active: {remaining}s remaining")
            details["reason"].append("Waiting after recent stop loss to avoid re-entering bad trade")
            details["cooldown_active"] = True
            return Decision.HOLD, details

        # Check for buy signal with confluence
        if score > buy_threshold:
            # ML confidence gate: if ML is strongly bearish, require higher score to allow BUY
            ml_pred = enhanced.get("ml_prediction") or {}
            ml_direction = (ml_pred.get("direction") or "HOLD").upper()
            ml_conf = float(ml_pred.get("confidence", 0) or 0)
            ml_veto_threshold = getattr(config, "ML_VETO_CONFIDENCE", 0.6)
            ml_extra_required = getattr(config, "ML_VETO_EXTRA_THRESHOLD", 0.1)
            if ml_direction == "DOWN" and ml_conf >= ml_veto_threshold:
                if score <= buy_threshold + ml_extra_required:
                    details["reason"].append(
                        f"ML bearish (conf {ml_conf:.2f}) - need score > {buy_threshold + ml_extra_required:.2f} to allow BUY (have {score:.2f})"
                    )
                    return Decision.HOLD, details
                details["reason"].append(f"ML bearish but score strong enough ({score:.2f} > {buy_threshold + ml_extra_required:.2f}) - allowing BUY")

            # Confidence floor: require Medium+ confidence for BUY (lower for uptrends)
            min_confidence = getattr(config, 'MIN_CONFIDENCE_BUY', 0.4)
            if regime_name == "trending_up":
                min_confidence = getattr(config, 'MIN_CONFIDENCE_BUY_UPTREND', 0.20)
            if confidence["value"] < min_confidence:
                details["reason"].append(f"Confidence too low: {confidence['value']:.2f} < {min_confidence} (need Medium+)")
                return Decision.HOLD, details

            # === PHASE 5: ORDER BOOK FILTER ===
            order_book = enhanced.get("order_book") or {}
            ob_signal = order_book.get("signal", "neutral")
            if ob_signal == "strong_sell_pressure":
                details["reason"].append(f"🚫 Order book shows strong sell pressure - blocking BUY")
                details["reason"].append(f"Order book imbalance: {order_book.get('imbalance_ratio', 0):.3f}")
                return Decision.HOLD, details
            
            # === PHASE 5: REGIME-BASED ENTRY RESTRICTION ===
            regime = self.current_regime or MarketRegime.UNKNOWN
            sentiment = enhanced.get("sentiment") or {}
            fear_greed = sentiment.get("fear_greed", {})
            fg_value = fear_greed.get("value", 50)
            divergence = enhanced.get("divergence", {})
            has_bullish_divergence = divergence.get("type") == "bullish"
            extreme_fear = fg_value < 20
            
            # In trending_down with weak score, require divergence OR extreme fear
            if regime == MarketRegime.TRENDING_DOWN and score < 0.5:
                if not has_bullish_divergence and not extreme_fear:
                    details["reason"].append(f"🚫 Trending down + weak score ({score:.2f}) - need divergence or extreme fear")
                    details["reason"].append(f"Fear/Greed: {fg_value} (need <20 for extreme fear)")
                    return Decision.HOLD, details
                elif extreme_fear:
                    details["reason"].append(f"✅ Extreme fear ({fg_value}) - contrarian buy allowed in downtrend")
                elif has_bullish_divergence:
                    details["reason"].append(f"✅ Bullish divergence - buy allowed in downtrend")
            
            # Regime-based confluence requirements
            min_confluence_buy = getattr(config, 'MIN_CONFLUENCE_BUY', 5)
            if regime == MarketRegime.TRENDING_UP:
                # Use lower confluence for uptrends (safer market conditions)
                uptrend_confluence = getattr(config, 'MIN_CONFLUENCE_BUY_UPTREND', 4)
                min_confluence_buy = uptrend_confluence
                details["reason"].append(f"📈 Uptrend: requiring {min_confluence_buy} confluence")
            elif regime == MarketRegime.TRENDING_DOWN:
                # Use downtrend threshold if configured, otherwise same as normal
                downtrend_confluence = getattr(config, 'MIN_CONFLUENCE_BUY_DOWNTREND', min_confluence_buy)
                min_confluence_buy = downtrend_confluence
                details["reason"].append(f"📉 Downtrend: requiring {min_confluence_buy} confluence")

            # Volume required in ranging/unknown regime (disabled for more trading activity)
            if getattr(config, 'REQUIRE_VOLUME_RANGING', False):
                if regime in (MarketRegime.RANGING, MarketRegime.UNKNOWN):
                    if not enhanced.get("volume_confirmed", False):
                        details["reason"].append("Volume confirmation required in ranging/unknown regime")
                        return Decision.HOLD, details

            # Trend alignment: 4h trend check (disabled for testing)
            # if mtf_trend_4h is not None and mtf_trend_4h == "bearish":
            #     details["reason"].append("4h trend bearish - blocking BUY (trend alignment)")
            #     return Decision.HOLD, details
            if mtf_trend_4h == "bearish":
                details["reason"].append("4h trend bearish - proceeding anyway (testing mode)")

            # Check confluence with regime-adjusted minimum
            confluence_count = confluence.get("count", 0)
            has_enough_confluence = confluence_count >= min_confluence_buy
            
            # Allow BUY if confluence count is met, even if direction is neutral (for more trading activity)
            confluence_direction = confluence.get("direction", "neutral")
            if has_enough_confluence and (confluence_direction == "bullish" or (confluence_direction == "neutral" and confluence_count >= min_confluence_buy)):
                details["reason"].append(f"AI bullish: {score:.2f} (threshold: {buy_threshold})")
                details["reason"].append(f"Confluence: {confluence_count}/{min_confluence_buy} required ({confluence.get('strength')})")

                # Add divergence info
                if has_bullish_divergence:
                    details["reason"].append("Bullish divergence confirms signal!")

                # Add volume info
                if enhanced.get("volume_confirmed"):
                    details["reason"].append("Volume confirms the move")

                # PHASE 2: Add stochastic info
                stoch = enhanced.get("stochastic", {})
                if stoch.get("oversold"):
                    details["reason"].append(f"Stochastic oversold ({stoch.get('k'):.0f}) - strong buy zone!")
                if stoch.get("bullish_crossover"):
                    details["reason"].append("Stochastic bullish crossover!")

                # PHASE 2: Add squeeze info
                squeeze = enhanced.get("squeeze_alert")
                if squeeze and squeeze.get("active"):
                    details["reason"].append(f"⚠️ {squeeze.get('alert')} - bias: {squeeze.get('breakout_bias')}")

                # PHASE 2: Add streak info
                streak = enhanced.get("streak_info", {})
                if streak.get("win_streak", 0) >= 2:
                    details["reason"].append(f"🔥 Win streak: {streak.get('status')} - position +{int((self.streak_multiplier-1)*100)}%")
                elif streak.get("loss_streak", 0) >= 2:
                    details["reason"].append(f"❄️ Loss streak: {streak.get('status')} - position -{int((1-self.streak_multiplier)*100)}%")

                self._add_bullish_reasons(analysis, details)
                details["streak_multiplier"] = self.streak_multiplier
                return Decision.BUY, details
            else:
                # Score OK but not enough confluence
                details["reason"].append(f"Score OK ({score:.2f}) but confluence insufficient")
                details["reason"].append(f"Need {min_confluence_buy}/10, have {confluence_count}/10")
                return Decision.HOLD, details

        # No buy signal
        details["reason"].append(f"Score ({score:.2f}) below threshold ({buy_threshold})")
        min_conf_display = getattr(config, 'MIN_CONFLUENCE_BUY', 5)
        if confluence.get("count", 0) < min_conf_display:
            details["reason"].append(f"Confluence: {confluence.get('count', 0)}/10 (need {min_conf_display})")

        # PHASE 2: Mention if in squeeze (waiting for breakout)
        squeeze = enhanced.get("squeeze_alert")
        if squeeze and squeeze.get("active"):
            details["reason"].append(f"📊 Squeeze active - waiting for breakout direction")

        return Decision.HOLD, details

    def _get_indicator_summary(self, analysis: Dict) -> Dict:
        """Extract indicator summaries from analysis"""
        return {
            "rsi": {
                "value": analysis.get("rsi", {}).get("value", "N/A"),
                "signal": analysis.get("rsi", {}).get("signal", "N/A"),
                "score": analysis.get("rsi", {}).get("score", 0)
            },
            "macd": {
                "signal": analysis.get("macd", {}).get("signal", "N/A"),
                "score": analysis.get("macd", {}).get("score", 0)
            },
            "bollinger": {
                "signal": analysis.get("bollinger", {}).get("signal", "N/A"),
                "percent_b": analysis.get("bollinger", {}).get("percent_b", "N/A"),
                "score": analysis.get("bollinger", {}).get("score", 0)
            },
            "ema": {
                "trend": analysis.get("ema", {}).get("trend", "N/A"),
                "score": analysis.get("ema", {}).get("score", 0)
            },
            "support_resistance": {
                "signal": analysis.get("support_resistance", {}).get("signal", "N/A"),
                "score": analysis.get("support_resistance", {}).get("score", 0)
            },
            "volume": {
                "signal": analysis.get("volume", {}).get("signal", "N/A"),
                "confirmed": analysis.get("volume", {}).get("confirmation", False)
            },
            # PHASE 2: New indicators
            "stochastic": {
                "k": analysis.get("stochastic", {}).get("k", "N/A"),
                "d": analysis.get("stochastic", {}).get("d", "N/A"),
                "signal": analysis.get("stochastic", {}).get("signal", "N/A"),
                "score": analysis.get("stochastic", {}).get("score", 0)
            },
            "squeeze": {
                "active": analysis.get("squeeze", {}).get("is_squeeze", False),
                "intensity": analysis.get("squeeze", {}).get("intensity", "none"),
                "breakout_bias": analysis.get("squeeze", {}).get("breakout_bias", "neutral")
            }
        }

    def _add_bullish_reasons(self, analysis: Dict, details: Dict):
        """Add specific bullish indicator reasons to details"""
        rsi = analysis.get("rsi", {})
        if rsi.get("signal") == "oversold":
            details["reason"].append(f"RSI oversold ({rsi.get('value')})")

        macd = analysis.get("macd", {})
        if macd.get("signal") in ["bullish_crossover", "bullish"]:
            details["reason"].append(f"MACD {macd.get('signal')}")

        bb = analysis.get("bollinger", {})
        if bb.get("signal") in ["below_lower", "near_lower"]:
            details["reason"].append(f"Price {bb.get('signal')} Bollinger")

        ema = analysis.get("ema", {})
        if ema.get("trend") in ["bullish_crossover", "strong_uptrend", "uptrend"]:
            details["reason"].append(f"EMA {ema.get('trend')}")

        sr = analysis.get("support_resistance", {})
        if sr.get("signal") in ["at_support", "near_support"]:
            details["reason"].append(f"Price {sr.get('signal')}")

    def get_confidence_level(self, score: float) -> str:
        """Get simple confidence level from score"""
        abs_score = abs(score)
        if abs_score >= 0.8:
            return "Very High"
        elif abs_score >= 0.6:
            return "High"
        elif abs_score >= 0.4:
            return "Moderate"
        elif abs_score >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def format_decision(self, decision: Decision, details: Dict) -> str:
        """Format decision and details for display"""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  AI DECISION: {decision.value}")
        lines.append(f"{'='*60}")

        score = details.get("score", 0)
        confidence = details.get("confidence", {})
        lines.append(f"  Score: {score:.4f} | Confidence: {confidence.get('level', 'N/A')} ({confidence.get('value', 0)*100:.0f}%)")

        # Regime info
        regime = details.get("regime", {})
        if regime:
            lines.append(f"  Market: {regime.get('regime_name', 'unknown').upper()} - {regime.get('description', '')}")

        # Confluence info (PHASE 4: Now 10 indicators)
        confluence = details.get("confluence", {})
        if confluence:
            lines.append(f"  Confluence: {confluence.get('count', 0)}/10 ({confluence.get('strength', 'none')}) - {', '.join(confluence.get('agreeing_indicators', []))}")

        # Divergence info
        divergence = details.get("divergence", {})
        if divergence and divergence.get("detected"):
            lines.append(f"  Divergence: {divergence.get('type', '').upper()} - {divergence.get('description', '')}")

        if "pnl_percent" in details:
            pnl = details["pnl_percent"]
            pnl_str = f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%"
            lines.append(f"  Current P/L: {pnl_str}")

        lines.append(f"\n  Reasons:")
        for reason in details.get("reason", []):
            lines.append(f"    - {reason}")

        if config.SHOW_DETAILED_ANALYSIS:
            lines.append(f"\n  Indicator Scores:")
            indicators = details.get("indicators", {})
            for name, data in indicators.items():
                if name != "volume":
                    ind_score = data.get("score", 0)
                    signal = data.get("signal", data.get("trend", "N/A"))
                    lines.append(f"    {name.upper():20s}: {ind_score:+.3f} ({signal})")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


# Test the enhanced AI engine
if __name__ == "__main__":
    from binance.client import Client
    from analyzer import TechnicalAnalyzer

    print("Testing ENHANCED AI Decision Engine...")

    # Get real data (public API, no keys needed)
    klines = Client().get_klines(symbol=config.SYMBOL, interval=config.CANDLE_INTERVAL, limit=100)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    if df.empty:
        print("Failed to get price data")
    else:
        # Run technical analysis
        analyzer = TechnicalAnalyzer()
        analysis = analyzer.analyze(df)

        # Create enhanced AI engine
        ai = AIEngine()

        # Test without position
        print("\n--- Testing Smart Entry Logic ---")
        decision, details = ai.get_decision(analysis, current_position=None, df=df)
        print(ai.format_decision(decision, details))

        # Show regime info
        print("\n--- Market Regime Info ---")
        if ai.regime_data:
            print(f"  Regime: {ai.regime_data.get('regime_name', 'unknown').upper()}")
            print(f"  Description: {ai.regime_data.get('description', '')}")
            params = ai.regime_data.get('adjusted_params', {})
            print(f"  Adjusted Buy Threshold: {params.get('buy_threshold', 0.3)}")
            print(f"  Adjusted Confluence: {params.get('confluence_required', 3)}/5")

        # PHASE 3: Show new features
        print(f"\n--- PHASE 3 Features ---")
        print(f"  Learning File: {ai.learning_file}")
        print(f"  Trade History: {len(ai.trade_history)} trades loaded")
        print(f"  Indicator Weights: {ai.weights}")
        if os.path.exists(ai.learning_file):
            print(f"  [OK] Learning state file exists")
            print(f"  Position Size: {params.get('position_size_multiplier', 1)*100:.0f}%")
