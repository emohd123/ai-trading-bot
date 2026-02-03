"""
Main Trading Logic - PHASE 2 ENHANCED VERSION
Smart trading with:
- Trailing stops (lock in profits automatically)
- Volatility-adjusted position sizing
- Daily loss limits
- Consecutive loss protection

PHASE 2 NEW FEATURES:
- Time-Based Exit: Close stale positions automatically
- Scale-Out Profits: Take profits in stages (33% at 5%, 33% at 10%, hold 33%)
- Dynamic Stop Loss: Adjust stop loss based on market regime
- Win Streak Integration: Uses AI engine's streak tracking
"""
import logging
import time
import json
from datetime import datetime, date, timedelta
from typing import Optional, Dict
from colorama import Fore, Style, init

from core.binance_client import BinanceClient

logger = logging.getLogger(__name__)
from ai.analyzer import TechnicalAnalyzer
from ai.ai_engine import AIEngine, Decision
from market.market_regime import RegimeDetector
from market.multi_timeframe import get_mtf_analysis
import config

# Initialize colorama for colored output
init(autoreset=True)


class Trader:
    """Enhanced trading bot with smart risk management"""

    def __init__(self):
        """Initialize the trader with smart features"""
        # Initialize components
        self.client = BinanceClient()
        self.analyzer = TechnicalAnalyzer()
        self.ai_engine = AIEngine()
        self.regime_detector = RegimeDetector()

        # Position tracking
        self.current_position: Optional[Dict] = None
        self.trade_history = []
        self.starting_balance = self.client.get_balance(config.QUOTE_ASSET)

        # === NEW: Smart Risk Management ===
        # Trailing stop tracking
        self.trailing_stop_price = None
        self.highest_price_since_entry = None
        self.trailing_stop_enabled = True

        # Daily loss tracking
        self.daily_losses = 0
        self.max_daily_loss = config.TRADE_AMOUNT_USDT * 0.3  # 30% of trade size
        self.last_trade_date = None

        # Consecutive loss protection
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3

        # Current regime data
        self.current_regime_data = None

        # === PHASE 2: Time-Based Exit Settings ===
        self.max_position_hours = 24  # Exit if held > 24 hours with < 5% profit
        self.stale_position_hours = 6  # Exit if no movement for 6 hours
        self.stale_threshold_percent = 0.5  # < 0.5% movement = stale
        self.price_history_for_stale = []  # Track recent prices for stale detection

        # === PHASE 2: Scale-Out Settings ===
        self.scale_out_enabled = True
        self.scale_out_levels = [
            {"profit_percent": 5, "sell_percent": 33, "triggered": False},
            {"profit_percent": 10, "sell_percent": 33, "triggered": False},
            # Remaining 33% rides with trailing stop
        ]
        self.partial_profits_taken = 0  # Track total partial profits

        # === PHASE 2: Dynamic Stop Loss ===
        self.dynamic_stop_loss_enabled = True
        self.regime_stop_losses = {
            "trending_up": 0.07,      # 7% - give room in uptrend
            "trending_down": 0.03,    # 3% - tight in downtrend
            "high_volatility": 0.10,  # 10% - wide in volatile
            "ranging": 0.05,          # 5% - standard
            "unknown": 0.05           # 5% - default
        }

        # Statistics
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "max_profit": 0.0,
            "max_loss": 0.0,
            "trailing_stop_exits": 0,
            "divergence_exits": 0,
            "time_exits": 0,           # PHASE 2
            "scale_out_exits": 0,      # PHASE 2
            "partial_profits": 0.0     # PHASE 2
        }

        self._print_startup()

    def _print_startup(self):
        """Log startup banner with smart features"""
        logger.info("=" * 60)
        logger.info("SMART AI CRYPTO TRADING BOT")
        logger.info("Mode: LIVE (24/7) | Symbol: %s | Trade: $%s | Target: +%.0f%% | Stop: -%.0f%%",
                    config.SYMBOL, config.TRADE_AMOUNT_USDT, config.PROFIT_TARGET * 100, config.STOP_LOSS * 100)
        logger.info("Starting balance: $%s", f"{self.starting_balance:,.2f}")
        logger.info("Features: Regime Detection, Confluence, Trailing Stop, Volatility Sizing, Daily Loss Protection")

    def run(self):
        """Main trading loop"""
        logger.info("Starting smart trading bot (check every %s seconds)", config.CHECK_INTERVAL)

        try:
            while True:
                self._trading_cycle()
                time.sleep(config.CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self._print_summary()

    def _reset_daily_tracking(self):
        """Reset daily loss tracking at start of new day"""
        today = date.today()
        if self.last_trade_date != today:
            self.daily_losses = 0
            self.last_trade_date = today

    def _can_trade(self) -> tuple:
        """
        Check if trading is allowed based on risk rules

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        self._reset_daily_tracking()

        # Check daily loss limit
        if self.daily_losses >= self.max_daily_loss:
            return False, f"Daily loss limit reached (${self.daily_losses:,.2f})"

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"{self.consecutive_losses} consecutive losses - pausing"

        return True, "OK"

    def _calculate_position_size(self, regime_data: Dict) -> float:
        """
        Calculate position size based on volatility and regime

        In high volatility = smaller positions
        After losses = smaller positions
        In strong trends = slightly larger positions
        """
        base_size = config.TRADE_AMOUNT_USDT

        # Get regime multiplier
        params = regime_data.get("adjusted_params", {}) if regime_data else {}
        regime_multiplier = params.get("position_size_multiplier", 1.0)

        # Volatility adjustment
        atr_data = regime_data.get("atr", {}) if regime_data else {}
        volatility_ratio = atr_data.get("volatility_ratio", 1.0)

        if volatility_ratio > 2.0:
            volatility_mult = 0.5  # Very high volatility - half size
        elif volatility_ratio > 1.5:
            volatility_mult = 0.7
        elif volatility_ratio > 1.2:
            volatility_mult = 0.85
        else:
            volatility_mult = 1.0

        # Consecutive loss adjustment
        if self.consecutive_losses >= 2:
            loss_mult = 0.5
        elif self.consecutive_losses >= 1:
            loss_mult = 0.75
        else:
            loss_mult = 1.0

        # Calculate final size
        position_size = base_size * regime_multiplier * volatility_mult * loss_mult

        # Bounds
        position_size = max(50, min(position_size, base_size * 1.5))

        return round(position_size, 2)

    def _update_trailing_stop(self, current_price: float, regime_data: Dict):
        """
        Update trailing stop based on current price

        Trailing stop moves up as price increases, locking in profits
        Never moves down - only up!
        """
        if not self.current_position or not self.trailing_stop_enabled:
            return

        entry_price = self.current_position["entry_price"]
        profit_percent = ((current_price - entry_price) / entry_price) * 100

        # Track highest price since entry
        if self.highest_price_since_entry is None:
            self.highest_price_since_entry = current_price
        else:
            self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)

        # Only enable trailing stop after minimum profit (2%)
        params = regime_data.get("adjusted_params", {}) if regime_data else {}
        activation_percent = params.get("trailing_stop_activation", 2.0)

        if profit_percent < activation_percent:
            return

        # Calculate trailing stop distance based on ATR
        atr_data = regime_data.get("atr", {}) if regime_data else {}
        atr_value = atr_data.get("value", current_price * 0.02)
        trail_multiplier = params.get("trailing_stop_distance", 1.5)

        trailing_distance = atr_value * trail_multiplier
        new_trailing_stop = self.highest_price_since_entry - trailing_distance

        # Ensure trailing stop is at least at break-even after 5% profit
        if profit_percent > 5:
            min_stop = entry_price * 1.01  # At least 1% profit locked
            new_trailing_stop = max(new_trailing_stop, min_stop)

        # Only move trailing stop UP, never down
        if self.trailing_stop_price is None:
            self.trailing_stop_price = new_trailing_stop
        else:
            self.trailing_stop_price = max(self.trailing_stop_price, new_trailing_stop)

    def _check_trailing_stop(self, current_price: float) -> bool:
        """Check if trailing stop has been hit"""
        if self.trailing_stop_price and current_price <= self.trailing_stop_price:
            return True
        return False

    # === PHASE 2: Time-Based Exit Methods ===

    def _check_time_based_exit(self, current_price: float) -> tuple:
        """
        Check if position should be closed due to time

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        if not self.current_position:
            return False, ""

        entry_time_str = self.current_position.get("entry_time")
        if not entry_time_str:
            return False, ""

        try:
            entry_time = datetime.fromisoformat(entry_time_str)
        except:
            return False, ""

        entry_price = self.current_position["entry_price"]
        profit_percent = ((current_price - entry_price) / entry_price) * 100
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600

        # Check max position duration
        if hours_held > self.max_position_hours and profit_percent < 5:
            return True, f"Position held {hours_held:.1f}h with only {profit_percent:.1f}% profit"

        # Check for stale position (no movement)
        if hours_held > self.stale_position_hours:
            if self._is_position_stale(current_price, profit_percent):
                return True, f"Position stale for {self.stale_position_hours}h - no significant movement"

        return False, ""

    def _is_position_stale(self, current_price: float, current_profit: float) -> bool:
        """Check if price hasn't moved significantly"""
        # Track price history
        self.price_history_for_stale.append({
            "price": current_price,
            "time": datetime.now()
        })

        # Keep only last 2 hours of data (assuming 1-min checks)
        cutoff = datetime.now() - timedelta(hours=2)
        self.price_history_for_stale = [
            p for p in self.price_history_for_stale
            if p["time"] > cutoff
        ]

        if len(self.price_history_for_stale) < 10:
            return False

        # Calculate price range over last 2 hours
        prices = [p["price"] for p in self.price_history_for_stale]
        price_range = (max(prices) - min(prices)) / min(prices) * 100

        # If price moved less than threshold AND profit is minimal
        return price_range < self.stale_threshold_percent and abs(current_profit) < 2

    def get_position_age(self) -> Dict:
        """Get position age info for dashboard"""
        if not self.current_position:
            return {"hours": 0, "status": "No position", "is_old": False}

        entry_time_str = self.current_position.get("entry_time")
        if not entry_time_str:
            return {"hours": 0, "status": "Unknown", "is_old": False}

        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            hours_held = (datetime.now() - entry_time).total_seconds() / 3600

            if hours_held > self.max_position_hours:
                status = "⚠️ Aging out"
            elif hours_held > self.stale_position_hours:
                status = "Getting old"
            else:
                status = "Healthy"

            return {
                "hours": round(hours_held, 1),
                "status": status,
                "is_old": hours_held > self.stale_position_hours,
                "max_hours": self.max_position_hours
            }
        except:
            return {"hours": 0, "status": "Error", "is_old": False}

    # === PHASE 2: Scale-Out Methods ===

    def _check_scale_out(self, current_price: float) -> tuple:
        """
        Check if we should take partial profits

        Returns:
            Tuple of (should_scale_out: bool, sell_percent: float, level_info: str)
        """
        if not self.current_position or not self.scale_out_enabled:
            return False, 0, ""

        entry_price = self.current_position["entry_price"]
        profit_percent = ((current_price - entry_price) / entry_price) * 100

        for level in self.scale_out_levels:
            if not level["triggered"] and profit_percent >= level["profit_percent"]:
                return True, level["sell_percent"], f"+{level['profit_percent']}% target hit"

        return False, 0, ""

    def _execute_partial_sell(self, current_price: float, sell_percent: float, reason: str) -> bool:
        """
        Execute a partial sell (scale-out)

        Returns:
            True if successful
        """
        if not self.current_position:
            return False

        # Calculate quantity to sell
        total_quantity = self.current_position["quantity"]
        sell_quantity = total_quantity * (sell_percent / 100)

        logger.info("SCALE-OUT: Partial profit taking - %s, selling %s%% of position", reason, sell_percent)

        result = self.client.place_market_sell(quantity=sell_quantity)

        if result.get("status") == "filled":
            # Calculate partial profit
            entry_price = self.current_position["entry_price"]
            exit_price = result["price"]
            partial_profit = (exit_price - entry_price) * sell_quantity
            partial_profit_percent = ((exit_price - entry_price) / entry_price) * 100

            # Update position
            self.current_position["quantity"] -= sell_quantity

            # Track which level was triggered
            for level in self.scale_out_levels:
                if level["sell_percent"] == sell_percent and not level["triggered"]:
                    level["triggered"] = True
                    break

            # Update stats
            self.partial_profits_taken += partial_profit
            self.stats["scale_out_exits"] += 1
            self.stats["partial_profits"] += partial_profit

            logger.info("Partial sell FILLED: sold %.8f %s @ $%s, profit $%+.2f (%.2f%%), remaining %.8f %s",
                        sell_quantity, config.BASE_ASSET, f"{exit_price:,.2f}", partial_profit,
                        partial_profit_percent, self.current_position['quantity'], config.BASE_ASSET)

            return True
        else:
            logger.error("Partial sell failed: %s", result.get('message', 'Unknown'))
            return False

    def get_scale_out_info(self) -> Dict:
        """Get scale-out status for dashboard"""
        if not self.current_position:
            return {
                "enabled": self.scale_out_enabled,
                "levels": [],
                "partial_profits": self.partial_profits_taken
            }

        entry_price = self.current_position["entry_price"]

        return {
            "enabled": self.scale_out_enabled,
            "levels": [
                {
                    "target": f"+{level['profit_percent']}%",
                    "sell": f"{level['sell_percent']}%",
                    "triggered": level["triggered"],
                    "trigger_price": round(entry_price * (1 + level['profit_percent']/100), 2)
                }
                for level in self.scale_out_levels
            ],
            "partial_profits": round(self.partial_profits_taken, 2)
        }

    def _reset_scale_out_levels(self):
        """Reset scale-out tracking for new position"""
        for level in self.scale_out_levels:
            level["triggered"] = False
        self.partial_profits_taken = 0

    # === PHASE 2: Dynamic Stop Loss Methods ===

    def _get_dynamic_stop_loss(self) -> float:
        """
        Get stop loss percentage based on current market regime

        Returns:
            Stop loss as decimal (e.g., 0.05 for 5%)
        """
        if not self.dynamic_stop_loss_enabled:
            return config.STOP_LOSS

        regime_name = "unknown"
        if self.current_regime_data:
            regime_name = self.current_regime_data.get("regime_name", "unknown")

        return self.regime_stop_losses.get(regime_name, config.STOP_LOSS)

    def get_dynamic_stop_info(self) -> Dict:
        """Get dynamic stop loss info for dashboard"""
        regime_name = "unknown"
        if self.current_regime_data:
            regime_name = self.current_regime_data.get("regime_name", "unknown")

        current_stop = self._get_dynamic_stop_loss()

        return {
            "enabled": self.dynamic_stop_loss_enabled,
            "current_regime": regime_name,
            "stop_percent": round(current_stop * 100, 1),
            "all_stops": {k: f"{v*100:.0f}%" for k, v in self.regime_stop_losses.items()}
        }

    def _trading_cycle(self):
        """Execute one trading cycle with smart features"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("[%s] Analyzing market...", timestamp)

        # Multi-timeframe analysis (4h, 1h, 15m)
        mtf_data = get_mtf_analysis(self.client)
        if "error" in mtf_data:
            logger.warning("MTF error: %s, skipping cycle", mtf_data['error'])
            return

        mtf_trend_4h = None
        if mtf_data.get("4h"):
            mtf_trend_4h = mtf_data["4h"].get("trend")

        analysis = mtf_data.get("1h")
        df = mtf_data.get("df_1h")
        if analysis is None or df is None or df.empty:
            logger.warning("Failed to get 1h data, skipping cycle")
            return
        if "error" in analysis:
            logger.warning("Analysis error: %s", analysis['error'])
            return

        current_price = analysis["current_price"]

        # Detect market regime
        self.current_regime_data = self.regime_detector.detect_regime(df)
        regime_name = self.current_regime_data.get("regime_name", "unknown")

        logger.info("Price: $%s | Market: %s", f"{current_price:,.2f}", regime_name.upper())

        # Update trailing stop if we have a position
        if self.current_position:
            self._update_trailing_stop(current_price, self.current_regime_data)

            # PHASE 2: Check time-based exit first
            should_time_exit, time_reason = self._check_time_based_exit(current_price)
            if should_time_exit:
                logger.info("Time-based exit: %s", time_reason)
                self._execute_sell(current_price, exit_type="time_exit")
                self.stats["time_exits"] += 1
                return

            # PHASE 2: Check scale-out (partial profit taking)
            should_scale, sell_pct, scale_reason = self._check_scale_out(current_price)
            if should_scale:
                self._execute_partial_sell(current_price, sell_pct, scale_reason)
                # Don't return - continue checking other exits

            # Check trailing stop
            if self._check_trailing_stop(current_price):
                logger.info("Trailing stop triggered at $%s", f"{self.trailing_stop_price:,.2f}")
                self._execute_sell(current_price, exit_type="trailing_stop")
                return

            # PHASE 2: Check dynamic stop loss
            dynamic_stop = self._get_dynamic_stop_loss()
            entry_price = self.current_position["entry_price"]
            current_loss = ((current_price - entry_price) / entry_price) * 100

            if current_loss <= -dynamic_stop * 100:
                regime_name = self.current_regime_data.get("regime_name", "unknown") if self.current_regime_data else "unknown"
                logger.info("Dynamic stop loss (%.0f%% for %s) triggered", dynamic_stop * 100, regime_name)
                self._execute_sell(current_price, exit_type="dynamic_stop")
                return

        # Get AI decision with regime data and 4h trend
        decision, details = self.ai_engine.get_decision(
            analysis,
            current_position=self.current_position,
            df=df,
            mtf_trend_4h=mtf_trend_4h
        )

        # Print smart analysis
        self._print_smart_analysis(analysis, decision, details)

        # Check if we can trade
        can_trade, reason = self._can_trade()

        # Execute decision
        if decision == Decision.BUY:
            if not can_trade:
                logger.warning("Trade blocked: %s", reason)
            elif getattr(config, 'NO_BUY_IN_DOWNTREND', False) and self.current_regime_data and (self.current_regime_data.get("regime_name") or "").lower() == "trending_down":
                logger.info("BUY skipped: Downtrend (NO_BUY_IN_DOWNTREND)")
            else:
                self._execute_buy(current_price, details=details)
        elif decision == Decision.SELL:
            exit_type = details.get("exit_type", "ai_signal")
            self._execute_sell(current_price, exit_type=exit_type)
        else:
            self._print_hold(details)

    def _print_smart_analysis(self, analysis: Dict, decision: Decision, details: Dict):
        """Print smart analysis with new features"""
        score = details["score"]
        confidence = details.get("confidence", {})
        confluence = details.get("confluence", {})
        regime = details.get("regime", {})

        # Score color
        if score > 0.3:
            score_color = Fore.GREEN
        elif score < -0.3:
            score_color = Fore.RED
        else:
            score_color = Fore.YELLOW

        # Decision color
        if decision == Decision.BUY:
            decision_color = Fore.GREEN
        elif decision == Decision.SELL:
            decision_color = Fore.RED
        else:
            decision_color = Fore.YELLOW

        logger.info("AI Score: %+.3f | Confidence: %s | Confluence: %s/6 (%s) | Decision: %s",
                    score, confidence.get('level', 'N/A'), confluence.get('count', 0),
                    confluence.get('strength', 'none'), decision.value)
        if self.trailing_stop_price and self.current_position:
            logger.info("Trailing Stop: $%s", f"{self.trailing_stop_price:,.2f}")
        if details.get("pnl_percent") is not None:
            logger.info("Position P/L: %+.2f%%", details["pnl_percent"])
        divergence = details.get("divergence", {})
        if divergence and divergence.get("detected"):
            logger.info("Divergence: %s detected", divergence.get('type', '').upper())
        if details.get("volume_confirmed"):
            logger.info("Volume: CONFIRMED")

    def _execute_buy(self, price: float, details: Dict = None):
        """Execute a buy order with smart position sizing"""
        if self.current_position:
            logger.warning("Already have a position, cannot buy")
            return

        # Calculate smart position size
        position_size = self._calculate_position_size(self.current_regime_data)

        # PHASE 2: Apply streak multiplier from AI engine
        streak_mult = self.ai_engine.streak_multiplier
        position_size *= streak_mult

        # Ensure bounds
        position_size = max(50, min(position_size, config.TRADE_AMOUNT_USDT * 1.5))
        position_size = round(position_size, 2)

        # PHASE 2: Reset scale-out levels for new position
        self._reset_scale_out_levels()

        logger.info("EXECUTING BUY ORDER - Position size: $%s (risk-adjusted)", f"{position_size:,.2f}")

        # Check balance before attempting buy
        usdt_balance = self.client.get_balance(config.QUOTE_ASSET)
        min_required = position_size * 1.02  # 2% buffer for fees
        if usdt_balance < min_required:
            logger.warning("Insufficient balance: $%s USDT (need $%s) - add USDT to Spot wallet",
                          f"{usdt_balance:,.2f}", f"{min_required:,.2f}")
            return

        # PHASE 2: Show streak info
        if streak_mult != 1.0:
            streak_info = self.ai_engine.get_streak_info()
            streak_adj = f"+{int((streak_mult-1)*100)}%" if streak_mult > 1 else f"{int((streak_mult-1)*100)}%"
            logger.info("Streak adjustment: %s (%s)", streak_adj, streak_info.get('status', 'N/A'))

        result = self.client.place_market_buy(quote_amount=position_size)

        if result.get("status") == "filled":
            # Store position with regime data
            atr_data = self.current_regime_data.get("atr", {}) if self.current_regime_data else {}

            self.current_position = {
                "entry_price": result["price"],
                "quantity": result["quantity"],
                "entry_time": datetime.now().isoformat(),
                "trade_amount": position_size,
                "entry_atr": atr_data.get("value", result["price"] * 0.02),
                "entry_regime": self.current_regime_data.get("regime_name", "unknown") if self.current_regime_data else "unknown",
                "indicator_scores": (details or {}).get("indicator_scores", {}),
            }

            # Reset trailing stop tracking
            self.trailing_stop_price = None
            self.highest_price_since_entry = result["price"]

            # Check if trailing stop should be enabled
            params = self.current_regime_data.get("adjusted_params", {}) if self.current_regime_data else {}
            self.trailing_stop_enabled = params.get("trailing_stop_enabled", True)

            logger.info("BUY FILLED: price $%s, qty %.8f %s, amount $%s, trailing stop %s",
                       f"{result['price']:,.2f}", result['quantity'], config.BASE_ASSET,
                       f"{position_size:,.2f}", 'ENABLED' if self.trailing_stop_enabled else 'DISABLED')

            self._log_trade("BUY", result)
        else:
            logger.error("BUY FAILED: %s", result.get('message', 'Unknown error'))

    def _execute_sell(self, price: float, exit_type: str = "manual"):
        """Execute a sell order"""
        if not self.current_position:
            logger.warning("No position to sell")
            return

        logger.info("EXECUTING SELL ORDER - Exit type: %s", exit_type.upper())

        result = self.client.place_market_sell(quantity=self.current_position["quantity"])

        if result.get("status") == "filled":
            # Calculate profit/loss
            entry_price = self.current_position["entry_price"]
            exit_price = result["price"]
            quantity = result["quantity"]

            profit = (exit_price - entry_price) * quantity
            profit_percent = ((exit_price - entry_price) / entry_price) * 100

            # Update statistics
            self.stats["total_trades"] += 1
            self.stats["total_profit"] += profit

            if profit > 0:
                self.stats["winning_trades"] += 1
                self.stats["max_profit"] = max(self.stats["max_profit"], profit_percent)
                self.consecutive_losses = 0  # Reset on win
            else:
                self.stats["losing_trades"] += 1
                self.stats["max_loss"] = min(self.stats["max_loss"], profit_percent)
                self.consecutive_losses += 1
                self.daily_losses += abs(profit)

            # Track exit types
            if exit_type == "trailing_stop":
                self.stats["trailing_stop_exits"] += 1
            elif exit_type == "divergence_exit":
                self.stats["divergence_exits"] += 1
            elif exit_type == "time_exit":
                self.stats["time_exits"] += 1
            elif exit_type == "dynamic_stop":
                # Count as stop loss
                pass

            # PHASE 2: Record trade result in AI engine for streak tracking and adaptive weights
            indicator_scores = self.current_position.get("indicator_scores", {})
            self.ai_engine.record_trade_result(
                profit_percent, exit_type,
                indicator_scores_at_entry=indicator_scores if indicator_scores else None
            )

            # Log result
            logger.info("SELL FILLED: entry $%s, exit $%s, qty %.8f %s, P/L $%+.2f (%.2f%%)",
                        f"{entry_price:,.2f}", f"{exit_price:,.2f}", quantity, config.BASE_ASSET,
                        profit, profit_percent)
            if self.trailing_stop_price:
                logger.info("Trailing stop was: $%s", f"{self.trailing_stop_price:,.2f}")

            self._log_trade("SELL", result, profit, profit_percent, exit_type)

            # Clear position and trailing stop
            self.current_position = None
            self.trailing_stop_price = None
            self.highest_price_since_entry = None
        else:
            err = result.get('message', 'Unknown error')
            logger.error("SELL FAILED: %s", err)
            # Clear ghost position when nothing to sell (0 or dust)
            if "below minimum" in str(err).lower() or "quantity 0" in str(err).lower() or "no btc to sell" in str(err).lower():
                logger.info("Clearing position (no balance/dust - nothing to sell)")
                self.current_position = None
                self.trailing_stop_price = None
                self.highest_price_since_entry = None

    def _print_hold(self, details: Dict):
        """Log hold decision info"""
        if self.current_position:
            status = "Holding position"
            if self.trailing_stop_price:
                status += f" (trailing stop: ${self.trailing_stop_price:,.2f})"
            logger.info("%s", status)
        else:
            can_trade, reason = self._can_trade()
            if can_trade:
                logger.info("Waiting for entry signal...")
            else:
                logger.info("Paused: %s", reason)

    def _log_trade(self, side: str, result: Dict, profit: float = 0, profit_percent: float = 0, exit_type: str = ""):
        """Log a trade to history"""
        trade = {
            "timestamp": datetime.now().isoformat(),
            "side": side,
            "symbol": config.SYMBOL,
            "price": result["price"],
            "quantity": result["quantity"],
            "profit": profit,
            "profit_percent": profit_percent,
            "exit_type": exit_type,
            "regime": self.current_regime_data.get("regime_name", "unknown") if self.current_regime_data else "unknown"
        }

        self.trade_history.append(trade)

        # Save to file
        try:
            with open("trades.json", "w") as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error("Error saving trades: %s", e)

    def _print_summary(self):
        """Log trading summary with smart stats"""
        current_balance = self.client.get_account_value()
        total_return = ((current_balance - self.starting_balance) / self.starting_balance) * 100

        logger.info("TRADING SESSION SUMMARY")
        logger.info("Starting: $%s | Current: $%s | Return: %+.2f%%",
                    f"{self.starting_balance:,.2f}", f"{current_balance:,.2f}", total_return)
        logger.info("Trades: %d total, %d winning, %d losing",
                    self.stats['total_trades'], self.stats['winning_trades'], self.stats['losing_trades'])
        if self.stats['total_trades'] > 0:
            win_rate = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            logger.info("Win rate: %.1f%% | Total P/L: $%+.2f", win_rate, self.stats['total_profit'])

        if self.stats['max_profit'] > 0:
            logger.info("Best trade: +%.2f%%", self.stats['max_profit'])
        if self.stats['max_loss'] < 0:
            logger.info("Worst trade: %.2f%%", self.stats['max_loss'])
        logger.info("Exits: trailing=%d, divergence=%d, time=%d, scale_out=%d | Partial: $%s",
                    self.stats['trailing_stop_exits'], self.stats['divergence_exits'],
                    self.stats['time_exits'], self.stats['scale_out_exits'],
                    f"{self.stats['partial_profits']:,.2f}")
        streak_info = self.ai_engine.get_streak_info()
        logger.info("Streak: %s | Win rate (last 5): %s%%",
                    streak_info.get('status', 'N/A'), streak_info.get('win_rate', 0))

    # === Properties for dashboard access ===
    def get_state(self) -> Dict:
        """Get current bot state for dashboard"""
        return {
            "running": True,
            "current_position": self.current_position,
            "trailing_stop": self.trailing_stop_price,
            "highest_price": self.highest_price_since_entry,
            "regime_data": self.current_regime_data,
            "stats": self.stats,
            "consecutive_losses": self.consecutive_losses,
            "daily_losses": self.daily_losses,
            "can_trade": self._can_trade()[0],
            # PHASE 2: New state data
            "streak_info": self.ai_engine.get_streak_info(),
            "cooldown_info": self.ai_engine.get_cooldown_info(),
            "position_age": self.get_position_age(),
            "scale_out_info": self.get_scale_out_info(),
            "dynamic_stop_info": self.get_dynamic_stop_info()
        }


# Run trader directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--testnet":
        config.USE_TESTNET = True

    trader = Trader()
    trader.run()
