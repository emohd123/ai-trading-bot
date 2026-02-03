"""
Smart Trading Bot Dashboard - PHASE 3 ENHANCED VERSION
Web interface with:
- Market regime display
- Confluence indicator
- Confidence meter
- Trailing stop tracking
- Volume confirmation
- Smart risk status

PHASE 2 FEATURES:
- Win Streak Display
- Cooldown Timer
- Position Age Tracking
- Scale-Out Progress
- Stochastic Indicator
- Bollinger Squeeze Alert
- Dynamic Stop Loss Info

PHASE 3 NEW FEATURES:
- ATR-based dynamic stops (2x ATR, capped 3-7%)
- 15m timeframe entry refinement
- 5-indicator confluence (Momentum replaces RSI+Stochastic)
- Fibonacci level awareness
- Candlestick pattern detection
- Persistent learning state
"""
import logging
from flask import Flask, render_template, jsonify, request

logger = logging.getLogger(__name__)
from flask_socketio import SocketIO, emit
import threading
import time
import json
import os
from datetime import datetime

# Import bot components
from core.binance_client import BinanceClient
from ai.analyzer import TechnicalAnalyzer
from ai.ai_engine import AIEngine, Decision
from market.market_regime import RegimeDetector
from market.multi_timeframe import get_mtf_analysis
from notifications.telegram_notifier import get_notifier, init_notifier
from core.self_healing import SelfHealer, quick_fix_balance_mismatch
from ai.meta_ai import get_meta_ai, start_autonomous_ai, ai_think, ai_status
from ai.param_optimizer import get_optimizer, record_trade
from ai.deep_analyzer import get_analyzer
import config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading-bot-secret-key'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable static file caching
app.jinja_env.auto_reload = True  # Force Jinja2 to reload templates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global bot state with smart features
bot_state = {
    "running": False,
    "paused": False,           # Pause trading but keep monitoring
    "current_price": 0,
    "last_price": 0,           # Previous recorded price (for comparison)
    "ai_score": 0,
    "decision": "WAITING",
    "activity_status": "Idle",
    "decision_reasons": [],
    "position": None,          # Legacy single position (for backward compat)
    "positions": [],           # NEW: Multiple positions list
    "max_positions": 2,        # Maximum concurrent positions
    "balance_usdt": 0,
    "balance_btc": 0,
    "total_value": 0,
    "pnl_percent": 0,
    "indicators": {},
    "last_update": None,
    "trade_history": [],
    "logs": [],

    # === TOTAL PROFIT TRACKING ===
    "total_profit": 0.0,        # Total $ profit from all trades
    "total_profit_percent": 0.0, # Total % return
    "total_trades": 0,          # Number of completed trades
    "winning_trades": 0,        # Number of profitable trades
    "losing_trades": 0,         # Number of losing trades
    "win_rate": 0.0,            # Win percentage

    # === NEW: Smart Features ===
    "market_regime": "unknown",
    "regime_description": "Analyzing...",
    "regime_recommendation": "",
    "adx": {"value": 0, "trend_direction": "unknown", "trend_strength": "weak"},
    "atr": {"value": 0, "percent": 0, "volatility_level": "normal"},

    "confluence": {
        "count": 0,
        "direction": "neutral",
        "strength": "none",
        "agreeing_indicators": []
    },

    "confidence": {
        "level": "Low",
        "value": 0
    },

    "divergence": {
        "detected": False,
        "type": None,
        "description": "No divergence"
    },

    "volume": {
        "confirmed": False,
        "signal": "neutral",
        "ratio": 1.0,
        "description": ""
    },

    "trailing_stop": None,
    "highest_price": None,
    "trailing_stop_enabled": True,
    "breakeven_active": False,      # Break-even stop activated
    "ai_stop_override": None,       # AI stop override reason

    "risk_status": {
        "can_trade": True,
        "daily_losses": 0,
        "consecutive_losses": 0,
        "position_size_mult": 1.0
    },

    "adjusted_params": {},

    # === PHASE 2: New Smart Features ===
    "streak_info": {
        "win_streak": 0,
        "loss_streak": 0,
        "streak_multiplier": 1.0,
        "status": "➡️ Neutral",
        "win_rate": 0
    },

    "cooldown_info": {
        "active": False,
        "remaining_seconds": 0,
        "remaining_minutes": 0
    },

    "position_age": {
        "hours": 0,
        "status": "No position",
        "is_old": False
    },

    "scale_out_info": {
        "enabled": True,
        "levels": [],
        "partial_profits": 0
    },

    "stochastic": {
        "k": 50,
        "d": 50,
        "signal": "neutral",
        "oversold": False,
        "overbought": False
    },

    "squeeze": {
        "active": False,
        "intensity": "none",
        "breakout_bias": "neutral",
        "alert": ""
    },

    "dynamic_stop": {
        "enabled": True,
        "current_percent": 5,
        "regime": "unknown",
        "atr_based": True,
        "atr_value": 0,
        "calculated_stop": 0.05
    },

    # === PHASE 3: New Smart Features ===
    "mtf_15m": {
        "stoch_k": 50,
        "signal": "neutral",
        "refinement": "none",
        "refinement_reason": ""
    },

    "momentum": {
        "value": 0,
        "signal": "neutral",
        "rsi_component": 50,
        "stoch_component": 50
    },

    "fibonacci": {
        "levels": [],
        "nearest_level": None,
        "distance_percent": 0,
        "signal": "neutral"
    },

    "candlestick": {
        "pattern": None,
        "signal": "neutral",
        "strength": 0
    },

    "ml_prediction": {
        "direction": "HOLD",
        "confidence": 0,
        "probability": 0.5,
        "model_votes": {},
        "models_loaded": False,
        "error": ""
    },
    
    # === PHASE 4: Enhanced Indicators ===
    "ichimoku": {
        "signal": "neutral",
        "above_cloud": False,
        "below_cloud": False,
        "in_cloud": True,
        "tk_cross": None
    },
    
    "mfi": {
        "value": 50,
        "signal": "neutral",
        "oversold": False,
        "overbought": False
    },
    
    "williams_r": {
        "value": -50,
        "signal": "neutral",
        "oversold": False,
        "overbought": False
    },
    
    "cci": {
        "value": 0,
        "signal": "neutral",
        "oversold": False,
        "overbought": False
    },
    
    # === PHASE 4: External Data Sources ===
    "order_book": {
        "imbalance_ratio": 1.0,
        "signal": "neutral",
        "bid_wall": None,
        "ask_wall": None,
        "score": 0
    },
    
    "sentiment": {
        "fear_greed_value": 50,
        "fear_greed_signal": "neutral",
        "funding_rate": 0,
        "combined_signal": "neutral",
        "score": 0
    },
    
    "correlation": {
        "btc_dominance": 50,
        "market_cap_change": 0,
        "signal": "neutral",
        "score": 0
    }
}

# Bot components
client = None
analyzer = None
ai_engine = None
regime_detector = None
bot_thread = None
stop_bot = threading.Event()
self_healer = None  # Self-healing system for auto-fixing errors
meta_ai = None      # Meta AI controller for autonomous operation

# Trailing stop tracking
trailing_stop_price = None
highest_price_since_entry = None
consecutive_losses = 0
daily_losses = 0
daily_trades = 0  # PHASE 5: Track daily trade count

# Smart stop tracking
breakeven_activated = False  # Track if break-even stop is active
previous_rsi = None          # Track RSI for recovery detection
stop_delay_count = 0         # Track AI stop delay cycles

# Health check tracking
last_health_check = None
health_check_interval = 300  # 5 minutes


def _get_activity_status(decision, details, position):
    """Get human-readable status of what the bot is doing"""
    reasons = details.get("reason", [])
    first_reason = reasons[0] if reasons else ""
    if decision.value == "BUY":
        return "Executing BUY - Placing order"
    if decision.value == "SELL":
        exit_type = details.get("exit_type", "signal")
        if exit_type == "profit_target":
            return "Taking profit - Target reached"
        if exit_type == "stop_loss":
            return "Stop loss - Closing position"
        if exit_type == "trailing_stop":
            return "Trailing stop hit - Closing position"
        return "Executing SELL - Closing position"
    if position:
        return f"Monitoring position - {first_reason}" if first_reason else "Holding position - No exit signal"
    return first_reason if first_reason else "Analyzing market - Waiting for signal"


def add_log(message, level="info"):
    """Add a log message"""
    log_entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "level": level
    }
    bot_state["logs"].insert(0, log_entry)
    bot_state["logs"] = bot_state["logs"][:50]
    socketio.emit('log', _to_json_serializable(log_entry))


def update_dashboard():
    """Send updated state to dashboard"""
    try:
        # Convert to JSON-serializable format before emitting
        safe_state = _to_json_serializable(bot_state)
        socketio.emit('update', safe_state)
        # Save full snapshot for status_check.py
        try:
            temp_path = os.path.join(config.DATA_DIR, "temp_state.json")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(safe_state, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Non-critical
    except (TypeError, ValueError) as e:
        add_log(f"Dashboard update error: {e}", "error")


def calculate_position_size(regime_data, confidence=None):
    """Calculate smart position size with confidence-based adjustment"""
    base_size = config.TRADE_AMOUNT_USDT

    params = regime_data.get("adjusted_params", {}) if regime_data else {}
    regime_mult = params.get("position_size_multiplier", 1.0)

    atr_data = regime_data.get("atr", {}) if regime_data else {}
    vol_ratio = atr_data.get("volatility_ratio", 1.0)

    if vol_ratio > 2.0:
        vol_mult = 0.5
    elif vol_ratio > 1.5:
        vol_mult = 0.7
    else:
        vol_mult = 1.0

    global consecutive_losses
    if consecutive_losses >= 2:
        loss_mult = 0.5
    elif consecutive_losses >= 1:
        loss_mult = 0.75
    else:
        loss_mult = 1.0

    # PHASE 5: Confidence-based position sizing
    conf_value = confidence.get("value", 0.5) if confidence else 0.5
    if conf_value >= 0.7:
        conf_mult = 1.2  # High confidence = larger position
    elif conf_value >= 0.5:
        conf_mult = 1.0  # Medium confidence = normal
    else:
        conf_mult = 0.7  # Low confidence = smaller position

    size = base_size * regime_mult * vol_mult * loss_mult * conf_mult
    bot_state["risk_status"]["position_size_mult"] = round(regime_mult * vol_mult * loss_mult * conf_mult, 2)

    # Min $15, max 1.5x base size
    return max(15, min(size, base_size * 1.5))


def calculate_atr_stop(current_price, regime_data):
    """
    PHASE 3: Calculate ATR-based dynamic stop loss
    Base: 2x ATR, capped at 3-7%
    Adjusted by regime: tighter in downtrend, wider in high volatility
    """
    atr_data = regime_data.get("atr", {}) if regime_data else {}
    atr_value = atr_data.get("value", current_price * 0.02)
    atr_percent = atr_data.get("percent", 2.0)
    
    # Base stop: 2x ATR
    base_multiplier = getattr(config, 'ATR_STOP_MULTIPLIER', 2.0)
    
    # Regime adjustments
    regime = regime_data.get("regime_name", "unknown") if regime_data else "unknown"
    if regime == "trending_down":
        multiplier = base_multiplier * getattr(config, 'ATR_TRENDING_DOWN_MULT', 0.8)
    elif regime == "high_volatility":
        multiplier = base_multiplier * getattr(config, 'ATR_HIGH_VOLATILITY_MULT', 1.5)
    else:
        multiplier = base_multiplier
    
    # Calculate stop percent
    stop_percent = (atr_value * multiplier / current_price) * 100
    
    # Cap between 3% and 7%
    min_stop = getattr(config, 'ATR_STOP_MIN', 0.03) * 100
    max_stop = getattr(config, 'ATR_STOP_MAX', 0.07) * 100
    stop_percent = max(min_stop, min(stop_percent, max_stop))
    
    return {
        "stop_percent": round(stop_percent, 2),
        "atr_value": round(atr_value, 2),
        "multiplier": round(multiplier, 2),
        "regime": regime
    }


def check_smart_stop(current_price, entry_price, pnl_percent, position_data=None):
    """
    AI-Assisted Smart Stop Loss System
    AI can override/delay stops if it sees recovery potential
    Returns: (should_stop, reason, adjusted_stop_level)
    """
    global breakeven_activated, previous_rsi, stop_delay_count
    
    # Initialize delay counter if not exists
    if 'stop_delay_count' not in globals():
        global stop_delay_count
        stop_delay_count = 0
    
    # Get current analysis data
    rsi = bot_state.get("rsi", {}).get("value", 50)
    support = bot_state.get("support_resistance", {}).get("support", 0)
    momentum = bot_state.get("combined_momentum", {})
    ai_score = bot_state.get("ai_score", 0)
    decision = bot_state.get("decision", "HOLD")
    confluence = bot_state.get("confluence", {})
    
    # === 0. HARD STOP - AI CANNOT OVERRIDE ===
    hard_stop = getattr(config, 'HARD_STOP_LIMIT', 0.02) * 100  # 2%
    if pnl_percent <= -hard_stop:
        add_log(f"🚨 HARD STOP: {pnl_percent:.2f}% exceeds max {hard_stop:.1f}% - AI cannot override", "error")
        stop_delay_count = 0
        return True, "hard_stop", None
    
    # === 1. BREAK-EVEN STOP ===
    breakeven_pct = getattr(config, 'BREAKEVEN_ACTIVATION', 0.005) * 100
    breakeven_buffer = getattr(config, 'BREAKEVEN_BUFFER', 0.001)
    
    if pnl_percent >= breakeven_pct:
        breakeven_activated = True
        bot_state["breakeven_active"] = True
        breakeven_stop = entry_price * (1 + breakeven_buffer)
        
        if current_price <= breakeven_stop:
            stop_delay_count = 0
            return True, "breakeven_stop", breakeven_stop
        return False, "protected_by_breakeven", breakeven_stop
    
    # === 2. AI-ASSISTED STOP LOGIC ===
    ai_enabled = getattr(config, 'AI_STOP_ENABLED', True)
    max_delays = getattr(config, 'MAX_STOP_DELAY_CYCLES', 3)
    # In downtrend: do NOT delay stops - cut losses fast (AI "bullish" in downtrend often fails)
    regime = (bot_state.get("market_regime") or "").lower()
    if regime == "trending_down":
        max_delays = 0
    
    if ai_enabled and pnl_percent < 0:
        ai_bullish = getattr(config, 'AI_BULLISH_OVERRIDE', 0.15)
        ai_strong = getattr(config, 'AI_STRONG_BULLISH', 0.30)
        
        # === 2A. AI SCORE CHECK ===
        # If AI score is turning bullish, delay the stop
        if ai_score > ai_strong and stop_delay_count < max_delays:
            stop_delay_count += 1
            add_log(f"🤖 AI OVERRIDE: Score {ai_score:.2f} is STRONG BULLISH - Delaying stop ({stop_delay_count}/{max_delays})", "info")
            bot_state["ai_stop_override"] = f"Strong bullish ({ai_score:.2f})"
            return False, "ai_strong_bullish", None
        
        elif ai_score > ai_bullish and stop_delay_count < max_delays:
            stop_delay_count += 1
            add_log(f"🤖 AI OVERRIDE: Score {ai_score:.2f} turning bullish - Delaying stop ({stop_delay_count}/{max_delays})", "info")
            bot_state["ai_stop_override"] = f"Turning bullish ({ai_score:.2f})"
            return False, "ai_turning_bullish", None
        
        # === 2B. AI DECISION CHECK ===
        # If AI says HOLD (not SELL), give more tolerance
        ai_sell_immediate = getattr(config, 'AI_SELL_IMMEDIATE', True)
        
        if decision == "SELL" and ai_sell_immediate:
            # AI says SELL - don't delay, execute stop
            add_log(f"🤖 AI confirms SELL - No delay", "warning")
            stop_delay_count = 0
            return None, "ai_confirms_sell", None  # Let regular stop logic decide
        
        elif decision == "HOLD" and stop_delay_count < max_delays:
            # AI says HOLD - widen stop tolerance
            stop_delay_count += 1
            add_log(f"🤖 AI says HOLD - Giving more time ({stop_delay_count}/{max_delays})", "info")
            bot_state["ai_stop_override"] = "AI says HOLD"
            return False, "ai_hold_override", None
        
        # === 2C. CONFLUENCE CHECK ===
        # Check if indicators are mostly bearish before stopping
        confluence_override = getattr(config, 'CONFLUENCE_OVERRIDE', True)
        min_bearish = getattr(config, 'MIN_BEARISH_CONFLUENCE', 4)
        
        if confluence_override:
            bearish_count = confluence.get("bearish_count", 0)
            bullish_count = confluence.get("bullish_count", 0)
            
            if bearish_count < min_bearish and stop_delay_count < max_delays:
                stop_delay_count += 1
                add_log(f"🤖 Only {bearish_count} bearish signals (need {min_bearish}) - Delaying stop ({stop_delay_count}/{max_delays})", "info")
                bot_state["ai_stop_override"] = f"Weak bearish ({bearish_count}/{min_bearish})"
                return False, "weak_bearish_confluence", None
            
            elif bullish_count > bearish_count and stop_delay_count < max_delays:
                stop_delay_count += 1
                add_log(f"🤖 More bullish ({bullish_count}) than bearish ({bearish_count}) - Delaying stop ({stop_delay_count}/{max_delays})", "info")
                bot_state["ai_stop_override"] = f"Bullish confluence ({bullish_count}>{bearish_count})"
                return False, "bullish_confluence_override", None
    
    # === 3. SUPPORT-BASED STOP ===
    use_support = getattr(config, 'USE_SUPPORT_STOP', True)
    if use_support and support > 0:
        support_buffer = getattr(config, 'SUPPORT_STOP_BUFFER', 0.002)
        support_stop = support * (1 - support_buffer)
        
        if current_price <= support_stop:
            add_log(f"📉 Support broken at ${support:.2f}", "warning")
            stop_delay_count = 0
            return True, "support_broken", support_stop
    
    # === 4. RECOVERY CHECK (Momentum) ===
    recovery_enabled = getattr(config, 'RECOVERY_CHECK_ENABLED', True)
    if recovery_enabled and pnl_percent < 0 and stop_delay_count < max_delays:
        rsi_recovering = previous_rsi is not None and rsi > previous_rsi and rsi < 40
        momentum_signal = momentum.get("signal", "")
        momentum_recovering = "bullish" in momentum_signal.lower()
        
        if rsi_recovering or momentum_recovering:
            stop_delay_count += 1
            add_log(f"🔄 Recovery detected (RSI: {rsi:.1f}) - Delaying stop ({stop_delay_count}/{max_delays})", "info")
            bot_state["ai_stop_override"] = f"Momentum recovering"
            return False, "recovery_in_progress", None
    
    previous_rsi = rsi
    
    # === 5. TIME-BASED EXIT ===
    max_hours = getattr(config, 'MAX_POSITION_AGE_HOURS', 4)
    stale_threshold = getattr(config, 'STALE_LOSS_THRESHOLD', -0.003) * 100
    
    if position_data and 'time' in position_data:
        try:
            entry_time = datetime.strptime(position_data['time'], "%Y-%m-%d %H:%M:%S")
            age_hours = (datetime.now() - entry_time).total_seconds() / 3600
            
            if age_hours > max_hours and pnl_percent < stale_threshold:
                add_log(f"⏰ Position stale ({age_hours:.1f}h) with loss {pnl_percent:.2f}%", "warning")
                stop_delay_count = 0
                return True, "time_exit", None
        except:
            pass
    
    # === 6. MAX DELAYS REACHED ===
    if stop_delay_count >= max_delays:
        add_log(f"⚠️ Max delay cycles ({max_delays}) reached - Allowing stop", "warning")
        bot_state["ai_stop_override"] = None
        stop_delay_count = 0
        return None, "max_delays_reached", None
    
    # Clear override status if no delay
    bot_state["ai_stop_override"] = None
    return None, "check_regular", None


def update_trailing_stop(current_price, regime_data):
    """Update trailing stop with PHASE 5 earlier activation based on win streak"""
    global trailing_stop_price, highest_price_since_entry, breakeven_activated, previous_rsi, stop_delay_count

    if not bot_state["position"]:
        trailing_stop_price = None
        highest_price_since_entry = None
        breakeven_activated = False
        previous_rsi = None
        stop_delay_count = 0
        bot_state["trailing_stop"] = None
        bot_state["highest_price"] = None
        bot_state["ai_stop_override"] = None
        return

    entry_price = bot_state["position"]["entry_price"]
    profit_percent = ((current_price - entry_price) / entry_price) * 100

    if highest_price_since_entry is None:
        highest_price_since_entry = current_price
    else:
        highest_price_since_entry = max(highest_price_since_entry, current_price)

    bot_state["highest_price"] = highest_price_since_entry

    # PHASE 5: Earlier trailing stop activation based on win streak
    params = regime_data.get("adjusted_params", {}) if regime_data else {}
    streak_info = bot_state.get("streak_info", {})
    win_streak = streak_info.get("win_streak", 0)
    
    # Use win streak to determine activation threshold
    if win_streak >= 3:
        # Hot streak - activate very early at +0.5%
        activation = getattr(config, 'TRAILING_ACTIVATION_HOT', 0.5)
    elif win_streak >= 2:
        # Good streak - activate at +0.75%
        activation = 0.75
    else:
        # Normal - activate at +1% (was +2%)
        activation = getattr(config, 'TRAILING_ACTIVATION', 1.0) * 100
        # Fall back to regime-based if not set
        if activation == 100:
            activation = params.get("trailing_stop_activation", 1.0)

    if profit_percent < activation:
        return

    # PHASE 3: Use ADX-based trailing distance
    atr_data = regime_data.get("atr", {}) if regime_data else {}
    adx_data = regime_data.get("adx", {}) if regime_data else {}
    atr_value = atr_data.get("value", current_price * 0.02)
    adx_value = adx_data.get("value", 25)
    
    # Trailing distance based on trend strength (ADX)
    if adx_value > getattr(config, 'TRAIL_ADX_STRONG', 40):
        trail_mult = getattr(config, 'TRAIL_ATR_STRONG', 2.5)
    elif adx_value > getattr(config, 'TRAIL_ADX_MODERATE', 30):
        trail_mult = getattr(config, 'TRAIL_ATR_MODERATE', 2.0)
    else:
        trail_mult = params.get("trailing_stop_distance", 1.5)
    
    # Wider in high volatility
    if atr_data.get("volatility_level") == "high":
        trail_mult = getattr(config, 'TRAIL_ATR_HIGH_VOL', 3.0)

    trail_distance = atr_value * trail_mult
    new_stop = highest_price_since_entry - trail_distance

    if profit_percent > 5:
        min_stop = entry_price * 1.01
        new_stop = max(new_stop, min_stop)

    if trailing_stop_price is None:
        trailing_stop_price = new_stop
    else:
        trailing_stop_price = max(trailing_stop_price, new_stop)

    bot_state["trailing_stop"] = trailing_stop_price


_last_state_save = 0
_STATE_SAVE_INTERVAL = 60  # Save state every 60 seconds when running


def bot_loop():
    """Main bot trading loop with smart features"""
    global client, analyzer, ai_engine, regime_detector, self_healer
    global trailing_stop_price, highest_price_since_entry, consecutive_losses, daily_losses, daily_trades
    global last_health_check

    add_log("Smart bot started", "success")
    bot_state["activity_status"] = "Starting - Fetching market data..."
    
    # Initialize self-healing system
    if self_healer is None and client:
        try:
            notifier = get_notifier()
            self_healer = SelfHealer(client, bot_state, notifier)
            add_log("Self-healing system initialized", "success")
        except Exception as e:
            add_log(f"Self-healing init failed: {e}", "warning")

    while not stop_bot.is_set():
        try:
            # === SELF-HEALING: Periodic health check ===
            if self_healer and self_healer.should_run_health_check():
                try:
                    health_status = self_healer.run_health_check()
                    if health_status.get("fixes_applied"):
                        for fix in health_status["fixes_applied"]:
                            add_log(f"🔧 Auto-fix: {fix}", "info")
                except Exception as e:
                    add_log(f"Health check error: {e}", "warning")
            
            bot_state["activity_status"] = "Fetching 4h/1h/15m data..."
            update_dashboard()

            # Multi-timeframe analysis (4h, 1h, 15m)
            mtf_data = get_mtf_analysis(client)
            if "error" in mtf_data:
                bot_state["activity_status"] = f"Error: {mtf_data['error']}"
                add_log(f"MTF error: {mtf_data['error']}", "error")
                time.sleep(10)
                continue

            bot_state["activity_status"] = "Analyzing indicators..."
            update_dashboard()

            analysis = mtf_data.get("1h")
            df = mtf_data.get("df_1h")
            if analysis is None or df is None or df.empty:
                add_log("Failed to get 1h data", "error")
                time.sleep(10)
                continue

            if "error" in analysis:
                add_log(f"Analysis error: {analysis['error']}", "error")
                time.sleep(10)
                continue

            current_price = analysis["current_price"]

            # When in position, use LIVE ticker price for accurate PnL (candle close can be 1h stale)
            if bot_state["position"]:
                try:
                    live_price = client.get_current_price()
                    current_price = live_price
                    analysis["current_price"] = live_price
                except Exception:
                    pass  # Fall back to candle close if API fails

            # Detect market regime (from 1h df)
            regime_data = regime_detector.detect_regime(df)

            # 4h trend for alignment check
            mtf_trend_4h = None
            if mtf_data.get("4h"):
                mtf_trend_4h = mtf_data["4h"].get("trend")

            # PHASE 3: 15m data for entry refinement
            analysis_15m = mtf_data.get("15m")
            if analysis_15m and ai_engine:
                ai_engine.last_15m_analysis = analysis_15m
                
                # Update 15m state for dashboard
                stoch_15m = analysis_15m.get("stochastic", {})
                stoch_15m_k = stoch_15m.get("k", 50)
                
                refinement = "none"
                refinement_reason = ""
                if stoch_15m_k > getattr(config, 'STOCH_15M_OVERBOUGHT', 75):
                    refinement = "wait_pullback"
                    refinement_reason = f"15m overbought ({stoch_15m_k:.0f}) - wait for pullback"
                elif stoch_15m_k < getattr(config, 'STOCH_15M_OVERSOLD', 25):
                    refinement = "confirm_entry"
                    refinement_reason = f"15m oversold ({stoch_15m_k:.0f}) - confirms entry"
                
                bot_state["mtf_15m"] = {
                    "stoch_k": round(stoch_15m_k, 1),
                    "signal": stoch_15m.get("signal", "neutral"),
                    "refinement": refinement,
                    "refinement_reason": refinement_reason
                }

            # Update state
            # Save previous price before updating
            if bot_state.get("current_price", 0) > 0:
                bot_state["last_price"] = bot_state["current_price"]
            bot_state["current_price"] = current_price
            bot_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update regime info
            bot_state["market_regime"] = regime_data.get("regime_name", "unknown")
            bot_state["regime_description"] = regime_data.get("description", "")
            bot_state["regime_recommendation"] = regime_data.get("recommendation", "")
            bot_state["adx"] = _to_json_serializable(regime_data.get("adx", {}))
            bot_state["atr"] = _to_json_serializable(regime_data.get("atr", {}))
            bot_state["adjusted_params"] = _to_json_serializable(regime_data.get("adjusted_params", {}))

            # Get balances
            bot_state["balance_usdt"] = client.get_balance(config.QUOTE_ASSET)
            bot_state["balance_btc"] = client.get_balance(config.BASE_ASSET)
            bot_state["total_value"] = client.get_account_value()

            # === MULTI-POSITION MANAGEMENT ===
            # Check all positions for exits (MIN_PROFIT, STOP_LOSS, TRAILING)
            positions_to_close = []
            for i, pos in enumerate(bot_state.get("positions", [])):
                entry_price = pos["entry_price"]
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                
                # PHASE 6: Track lowest price for max drawdown learning
                lowest_price = pos.get("lowest_price", entry_price)
                if current_price < lowest_price:
                    bot_state["positions"][i]["lowest_price"] = current_price

                # === IMMEDIATE MIN PROFIT CHECK (highest priority) ===
                min_profit_pct = getattr(config, 'MIN_PROFIT', 0.0025) * 100
                if pnl_percent >= min_profit_pct:
                    add_log(f"💰 POSITION #{i+1} MIN PROFIT: +{pnl_percent:.2f}% - SELLING!", "success")
                    positions_to_close.append((i, pos, "min_profit"))
                    continue

                # === SMART STOP LOSS CHECK (second priority) ===
                # Use smart stop system to avoid premature stops
                should_stop, stop_reason, stop_level = check_smart_stop(
                    current_price, entry_price, pnl_percent, pos
                )
                
                if should_stop:
                    add_log(f"🛑 POSITION #{i+1} SMART STOP ({stop_reason}): {pnl_percent:.2f}% - SELLING!", "warning")
                    positions_to_close.append((i, pos, stop_reason))
                    continue
                elif stop_reason == "protected_by_breakeven":
                    # Position is protected - don't check regular stop
                    continue
                elif stop_reason == "recovery_in_progress":
                    # Momentum recovering - skip this cycle
                    continue
                
                # Fallback to regime-based stop if smart stop didn't trigger
                current_regime = bot_state.get("market_regime", "unknown")
                atr_data = bot_state.get("atr", {})
                volatility_high = atr_data.get("volatility_level") == "high"
                
                # Select stop loss based on regime
                if current_regime == "trending_down":
                    stop_loss_pct = getattr(config, 'STOP_LOSS_TRENDING_DOWN', 0.008) * 100
                elif volatility_high:
                    stop_loss_pct = getattr(config, 'STOP_LOSS_HIGH_VOL', 0.012) * 100
                else:
                    stop_loss_pct = getattr(config, 'STOP_LOSS', 0.01) * 100
                    
                if pnl_percent <= -stop_loss_pct:
                    add_log(f"🛑 POSITION #{i+1} STOP LOSS: {pnl_percent:.2f}% (limit: -{stop_loss_pct:.1f}%) - SELLING!", "warning")
                    positions_to_close.append((i, pos, "stop_loss"))
                    continue

            # Close positions that hit targets (in reverse order to maintain indices)
            for idx, pos, exit_type in reversed(positions_to_close):
                execute_sell(current_price, exit_type, position_index=idx)
                update_dashboard()

            # Legacy single position support (backward compatibility)
            if bot_state["position"] and not bot_state.get("positions"):
                entry_price = bot_state["position"]["entry_price"]
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                
                # PHASE 6: Track lowest price for max drawdown learning
                lowest_price = bot_state["position"].get("lowest_price", entry_price)
                if current_price < lowest_price:
                    bot_state["position"]["lowest_price"] = current_price

                min_profit_pct = getattr(config, 'MIN_PROFIT', 0.0025) * 100
                if pnl_percent >= min_profit_pct:
                    add_log(f"💰 MIN PROFIT REACHED: +{pnl_percent:.2f}% - SELLING NOW!", "success")
                    execute_sell(current_price, "min_profit")
                    update_dashboard()
                    continue

                # SMART STOP LOSS (replaces basic regime-aware)
                should_stop, stop_reason, stop_level = check_smart_stop(
                    current_price, entry_price, pnl_percent, bot_state["position"]
                )
                
                if should_stop:
                    add_log(f"🛑 SMART STOP ({stop_reason}): {pnl_percent:.2f}% - SELLING NOW!", "warning")
                    execute_sell(current_price, stop_reason)
                    update_dashboard()
                    continue
                elif stop_reason == "protected_by_breakeven":
                    # Position is protected at breakeven - continue monitoring
                    pass
                elif stop_reason == "recovery_in_progress":
                    # Momentum recovering - skip stop this cycle
                    pass
                else:
                    # Fallback to regime-based stop
                    current_regime = bot_state.get("market_regime", "unknown")
                    atr_data = bot_state.get("atr", {})
                    volatility_high = atr_data.get("volatility_level") == "high"
                    
                    if current_regime == "trending_down":
                        stop_loss_pct = getattr(config, 'STOP_LOSS_TRENDING_DOWN', 0.008) * 100
                    elif volatility_high:
                        stop_loss_pct = getattr(config, 'STOP_LOSS_HIGH_VOL', 0.012) * 100
                    else:
                        stop_loss_pct = getattr(config, 'STOP_LOSS', 0.01) * 100
                        
                    if pnl_percent <= -stop_loss_pct:
                        add_log(f"🛑 STOP LOSS HIT: {pnl_percent:.2f}% (limit: -{stop_loss_pct:.1f}%) - SELLING NOW!", "warning")
                        execute_sell(current_price, "stop_loss")
                        update_dashboard()
                        continue

                update_trailing_stop(current_price, regime_data)

                if trailing_stop_price and current_price <= trailing_stop_price:
                    add_log(f"Trailing stop hit at ${trailing_stop_price:,.2f}!", "warning")
                    execute_sell(current_price, "trailing_stop")
                    update_dashboard()
                    continue

            # Get AI decision with enhanced features
            current_position = None
            if bot_state["position"]:
                current_position = {
                    "entry_price": bot_state["position"]["entry_price"],
                    "quantity": bot_state["position"]["quantity"]
                }

            decision, details = ai_engine.get_decision(
                analysis, current_position, df, mtf_trend_4h=mtf_trend_4h
            )

            # Update smart features in state
            bot_state["ai_score"] = details["score"]
            bot_state["decision"] = decision.value
            bot_state["decision_reasons"] = details.get("reason", [])
            bot_state["activity_status"] = _get_activity_status(decision, details, bot_state["position"])

            # Confluence
            confluence = details.get("confluence", {})
            bot_state["confluence"] = {
                "count": confluence.get("count", 0),
                "direction": confluence.get("direction", "neutral"),
                "strength": confluence.get("strength", "none"),
                "agreeing_indicators": confluence.get("agreeing_indicators", []),
                "bullish_count": confluence.get("bullish_count", 0),
                "bearish_count": confluence.get("bearish_count", 0),
            }

            # Confidence
            confidence = details.get("confidence", {})
            bot_state["confidence"] = {
                "level": confidence.get("level", "Low"),
                "value": confidence.get("value", 0)
            }

            # Divergence
            divergence = details.get("divergence", {})
            bot_state["divergence"] = {
                "detected": divergence.get("detected", False),
                "type": divergence.get("type"),
                "description": divergence.get("description", "No divergence")
            }

            # ML Prediction
            ml_pred = details.get("ml_prediction", {})
            bot_state["ml_prediction"] = {
                "direction": ml_pred.get("direction", "HOLD"),
                "confidence": ml_pred.get("confidence", 0),
                "probability": ml_pred.get("probability", 0.5),
                "model_votes": ml_pred.get("model_votes", {}),
                "models_loaded": ml_pred.get("models_loaded", False),
                "error": ml_pred.get("error", "")
            }

            # Volume
            volume = analysis.get("volume", {})
            bot_state["volume"] = {
                "confirmed": volume.get("confirmation", False),
                "signal": volume.get("signal", "neutral"),
                "ratio": volume.get("ratio", 1.0),
                "description": volume.get("description", ""),
                "trend": "rising" if volume.get("trending_up", False) else "falling" if volume.get("ratio", 1) < 0.9 else "flat",
            }

            # Indicators
            bot_state["indicators"] = {
                "rsi": analysis["rsi"],
                "macd": analysis["macd"],
                "bollinger": analysis["bollinger"],
                "ema": analysis["ema"],
                "sr": analysis["support_resistance"],
                "volume": analysis.get("volume", {}),
                "stochastic": analysis.get("stochastic", {}),  # PHASE 2
                "squeeze": analysis.get("squeeze", {}),         # PHASE 2
                # Phase 4 NEW indicators
                "ichimoku": analysis.get("ichimoku", {}),
                "mfi": analysis.get("mfi", {}),
                "williams_r": analysis.get("williams_r", {}),
                "cci": analysis.get("cci", {})
            }

            # PHASE 2: Stochastic
            stoch = analysis.get("stochastic", {})
            bot_state["stochastic"] = {
                "k": stoch.get("k", 50),
                "d": stoch.get("d", 50),
                "signal": stoch.get("signal", "neutral"),
                "oversold": stoch.get("oversold", False),
                "overbought": stoch.get("overbought", False)
            }

            # PHASE 2: Bollinger Squeeze
            squeeze = analysis.get("squeeze", {})
            bot_state["squeeze"] = {
                "active": squeeze.get("is_squeeze", False),
                "intensity": squeeze.get("intensity", "none"),
                "breakout_bias": squeeze.get("breakout_bias", "neutral"),
                "alert": squeeze.get("alert", ""),
                "bandwidth": squeeze.get("bandwidth", 0)
            }

            # PHASE 3: ATR-based dynamic stop
            atr_stop = calculate_atr_stop(current_price, regime_data)
            bot_state["dynamic_stop"] = {
                "enabled": True,
                "current_percent": atr_stop["stop_percent"],
                "regime": atr_stop["regime"],
                "atr_based": True,
                "atr_value": atr_stop["atr_value"],
                "calculated_stop": round(current_price * (1 - atr_stop["stop_percent"]/100), 2),
                "multiplier": atr_stop["multiplier"]
            }

            # PHASE 3: Combined Momentum (RSI + Stochastic)
            momentum_data = analysis.get("momentum", {})
            rsi_data = analysis.get("rsi", {})
            stoch_data = analysis.get("stochastic", {})
            bot_state["momentum"] = {
                "value": momentum_data.get("score", 0),
                "signal": momentum_data.get("signal", "neutral"),
                "rsi_component": rsi_data.get("value", 50),
                "stoch_component": stoch_data.get("k", 50),
                "combined_oversold": momentum_data.get("both_oversold", False),
                "combined_overbought": momentum_data.get("both_overbought", False)
            }

            # PHASE 3: Fibonacci Levels
            fib_data = analysis.get("fibonacci", {})
            bot_state["fibonacci"] = {
                "levels": fib_data.get("levels", []),
                "nearest_level": fib_data.get("nearest_level"),
                "distance_percent": fib_data.get("distance_percent", 0),
                "signal": fib_data.get("signal", "neutral"),
                "at_key_level": fib_data.get("at_key_level", False)
            }

            # PHASE 3: Candlestick Patterns
            candle_data = analysis.get("candlestick", {})
            bot_state["candlestick"] = {
                "pattern": candle_data.get("pattern"),
                "signal": candle_data.get("signal", "neutral"),
                "strength": candle_data.get("score", 0),
                "description": candle_data.get("description", "")
            }

            # === PHASE 4: New Indicators ===
            ichimoku_data = analysis.get("ichimoku", {})
            bot_state["ichimoku"] = {
                "signal": ichimoku_data.get("signal", "neutral"),
                "above_cloud": ichimoku_data.get("above_cloud", False),
                "below_cloud": ichimoku_data.get("below_cloud", False),
                "in_cloud": ichimoku_data.get("in_cloud", True),
                "tk_cross": "bullish" if ichimoku_data.get("tk_cross_bullish") else ("bearish" if ichimoku_data.get("tk_cross_bearish") else None),
                "tenkan": ichimoku_data.get("tenkan", 0),
                "kijun": ichimoku_data.get("kijun", 0),
                "score": ichimoku_data.get("score", 0)
            }

            mfi_data = analysis.get("mfi", {})
            bot_state["mfi"] = {
                "value": mfi_data.get("value", 50),
                "signal": mfi_data.get("signal", "neutral"),
                "oversold": mfi_data.get("oversold", False),
                "overbought": mfi_data.get("overbought", False),
                "score": mfi_data.get("score", 0)
            }

            williams_data = analysis.get("williams_r", {})
            bot_state["williams_r"] = {
                "value": williams_data.get("value", -50),
                "signal": williams_data.get("signal", "neutral"),
                "oversold": williams_data.get("oversold", False),
                "overbought": williams_data.get("overbought", False),
                "score": williams_data.get("score", 0)
            }

            cci_data = analysis.get("cci", {})
            bot_state["cci"] = {
                "value": cci_data.get("value", 0),
                "signal": cci_data.get("signal", "neutral"),
                "oversold": cci_data.get("oversold", False),
                "overbought": cci_data.get("overbought", False),
                "score": cci_data.get("score", 0)
            }

            # === PHASE 4: External Data from AI Engine ===
            order_book_data = details.get("order_book", {})
            if order_book_data:
                bot_state["order_book"] = {
                    "imbalance_ratio": order_book_data.get("imbalance_ratio", 1.0),
                    "signal": order_book_data.get("signal", "neutral"),
                    "bid_wall": order_book_data.get("bid_wall"),
                    "ask_wall": order_book_data.get("ask_wall"),
                    "depth_score": order_book_data.get("depth_score", 0),
                    "spread_bps": order_book_data.get("spread_bps", 0),
                    "score": order_book_data.get("score", 0)
                }

            sentiment_data = details.get("sentiment", {})
            if sentiment_data:
                fg = sentiment_data.get("fear_greed", {})
                fr = sentiment_data.get("funding_rate", {})
                bot_state["sentiment"] = {
                    "fear_greed_value": fg.get("value", 50),
                    "fear_greed_signal": fg.get("signal", "neutral"),
                    "fear_greed_class": fg.get("classification", "Neutral"),
                    "funding_rate": fr.get("rate_pct", 0),
                    "funding_signal": fr.get("signal", "neutral"),
                    "combined_signal": sentiment_data.get("combined_signal", "neutral"),
                    "risk_level": sentiment_data.get("risk_level", "moderate"),
                    "score": sentiment_data.get("combined_score", 0)
                }

            correlation_data = details.get("correlation", {})
            if correlation_data:
                bot_state["correlation"] = {
                    "btc_dominance": correlation_data.get("btc_dominance", 50),
                    "market_cap_change": correlation_data.get("market_cap_change_24h", 0),
                    "signal": correlation_data.get("signal", "neutral"),
                    "score": correlation_data.get("score", 0)
                }

            # PHASE 2: Streak info
            streak = ai_engine.get_streak_info()
            bot_state["streak_info"] = streak
            
            # PHASE 7: Learning summary for dashboard
            try:
                learning_summary = ai_engine.get_learning_summary()
                bot_state["learning_summary"] = learning_summary
            except Exception as e:
                logger.warning("Could not get learning summary: %s", e)
                bot_state["learning_summary"] = {}

            # PHASE 2: Cooldown info
            cooldown = ai_engine.get_cooldown_info()
            bot_state["cooldown_info"] = cooldown

            # PHASE 5: Enhanced risk status with daily limits
            total_value = bot_state.get("total_value", 100)
            max_daily_loss = total_value * getattr(config, 'MAX_DAILY_LOSS_PCT', 0.03)
            max_daily_trades = getattr(config, 'MAX_DAILY_TRADES', 20)
            daily_loss_pct = (daily_losses / total_value * 100) if total_value > 0 else 0
            
            can_trade = (
                consecutive_losses < 3 and 
                daily_losses < max_daily_loss and
                daily_trades < max_daily_trades
            )
            
            bot_state["risk_status"] = {
                "can_trade": can_trade,
                "daily_losses": daily_losses,
                "daily_loss_pct": round(daily_loss_pct, 2),
                "max_daily_loss": round(max_daily_loss, 2),
                "daily_trades": daily_trades,
                "max_daily_trades": max_daily_trades,
                "consecutive_losses": consecutive_losses,
                "position_size_mult": bot_state["risk_status"].get("position_size_mult", 1.0)
            }

            # Calculate P/L for all positions (multi-position support)
            positions = bot_state.get("positions", [])
            if positions:
                # Calculate combined P/L for all positions
                total_pnl_pct = 0
                positions_info = []
                for i, pos in enumerate(positions):
                    entry = pos["entry_price"]
                    pos_pnl_pct = ((current_price - entry) / entry) * 100
                    total_pnl_pct += pos_pnl_pct
                    positions_info.append({
                        "id": i + 1,
                        "entry": entry,
                        "pnl_percent": round(pos_pnl_pct, 2),
                        "quantity": pos["quantity"],
                        "amount": pos.get("amount_usdt", 0)
                    })
                bot_state["pnl_percent"] = total_pnl_pct / len(positions)  # Average P/L
                bot_state["positions_info"] = positions_info
                bot_state["active_positions"] = len(positions)
            elif bot_state["position"]:
                # Legacy single position
                entry = bot_state["position"]["entry_price"]
                bot_state["pnl_percent"] = ((current_price - entry) / entry) * 100
                bot_state["active_positions"] = 1

                # PHASE 2: Calculate position age
                entry_time_str = bot_state["position"].get("entry_time")
                if entry_time_str:
                    try:
                        entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                        hours_held = (datetime.now() - entry_time).total_seconds() / 3600

                        if hours_held > 18:
                            age_status = "Consider exiting"
                        elif hours_held > 6:
                            age_status = "Getting old"
                        else:
                            age_status = "Healthy"

                        bot_state["position_age"] = {
                            "hours": round(hours_held, 1),
                            "status": age_status,
                            "is_old": hours_held > 18
                        }
                    except:
                        pass
                # PHASE 2: Update scale-out info
                profit_pct = bot_state["pnl_percent"]
                entry_price = bot_state["position"]["entry_price"]

                scale_levels = [
                    {
                        "target": "+5%",
                        "sell": "33%",
                        "profit_percent": 5,
                        "trigger_price": round(entry_price * 1.05, 2),
                        "triggered": profit_pct >= 5
                    },
                    {
                        "target": "+10%",
                        "sell": "33%",
                        "profit_percent": 10,
                        "trigger_price": round(entry_price * 1.10, 2),
                        "triggered": profit_pct >= 10
                    }
                ]
                bot_state["scale_out_info"] = {
                    "enabled": True,
                    "levels": scale_levels,
                    "partial_profits": sum(1 for l in scale_levels if l["triggered"]) * 33
                }
            else:
                bot_state["pnl_percent"] = 0
                bot_state["position_age"] = {
                    "hours": 0,
                    "status": "No position",
                    "is_old": False
                }
                bot_state["scale_out_info"] = {
                    "enabled": True,
                    "levels": [],
                    "partial_profits": 0
                }

            # Execute trades (MULTI-POSITION: allow up to MAX_POSITIONS concurrent trades)
            max_positions = getattr(config, 'MAX_POSITIONS', 2)
            current_positions = len(bot_state.get("positions", []))
            # Also count legacy position if exists
            if bot_state["position"] and current_positions == 0:
                current_positions = 1

            can_open_new = current_positions < max_positions

            # Check if trading is paused
            if bot_state.get("paused", False):
                bot_state["activity_status"] = "PAUSED - Monitoring only (send /resume to continue)"
                update_dashboard()
                time.sleep(config.CHECK_INTERVAL)
                continue
            
            if decision == Decision.BUY and can_open_new:
                regime = bot_state.get("market_regime", "unknown")
                ai_score = details.get("score", 0) if details else 0
                no_buy_downtrend = getattr(config, 'NO_BUY_IN_DOWNTREND', False)
                downtrend_threshold = getattr(config, 'BUY_THRESHOLD_DOWNTREND', 0.50)
                # Best loss avoidance: no new buys in downtrend (or require very strong signal)
                if regime == "trending_down":
                    if no_buy_downtrend:
                        add_log("BUY skipped: Downtrend - no new buys (NO_BUY_IN_DOWNTREND)", "info")
                        bot_state["activity_status"] = "Downtrend - no new buys (avoiding losses)"
                    elif ai_score < downtrend_threshold:
                        add_log(f"BUY blocked: Downtrend - need score > {downtrend_threshold} (got {ai_score:.2f})", "info")
                        bot_state["activity_status"] = f"Downtrend - waiting for stronger signal (score {ai_score:.2f} < {downtrend_threshold})"
                    elif bot_state["risk_status"]["can_trade"]:
                        bot_state["activity_status"] = f"Executing BUY #{current_positions+1}/{max_positions}..."
                        update_dashboard()
                        execute_buy(current_price, regime_data, details=details)
                    else:
                        add_log("Trade blocked by risk rules", "warning")
                elif bot_state["risk_status"]["can_trade"]:
                    bot_state["activity_status"] = f"Executing BUY #{current_positions+1}/{max_positions}..."
                    update_dashboard()
                    execute_buy(current_price, regime_data, details=details)
                else:
                    add_log("Trade blocked by risk rules", "warning")
            elif decision == Decision.BUY and not can_open_new:
                # At max positions - wait for one to close
                bot_state["activity_status"] = f"Max positions ({current_positions}/{max_positions}) - waiting for exit"
            elif decision == Decision.SELL and (bot_state["position"] or current_positions > 0):
                exit_type = details.get("exit_type", "ai_signal")
                bot_state["activity_status"] = "Executing SELL - Closing position..."
                update_dashboard()
                execute_sell(current_price, exit_type)

            # Update dashboard
            update_dashboard()

            # === TELEGRAM PERIODIC STATUS UPDATES ===
            notifier = get_notifier()
            if notifier.enabled and notifier.should_send_periodic_status():
                # Refresh totals so "Total Profit So Far" is accurate
                calculate_total_profit()
                # Send comprehensive status update every 30 minutes
                notifier.notify_status(
                    current_price=current_price,
                    balance_usdt=bot_state["balance_usdt"],
                    total_value=bot_state["total_value"],
                    position=bot_state["position"],
                    pnl_percent=bot_state["pnl_percent"],
                    regime=bot_state["market_regime"],
                    total_profit=bot_state.get("total_profit", 0),
                    total_trades=bot_state.get("total_trades", 0),
                    win_rate=bot_state.get("win_rate", 0)
                )

            # Periodically save state
            now = time.time()
            if now - _last_state_save > _STATE_SAVE_INTERVAL:
                save_bot_state()
                globals()["_last_state_save"] = now

            # Wait for next cycle
            for _ in range(config.CHECK_INTERVAL):
                if stop_bot.is_set():
                    break
                time.sleep(1)

        except Exception as e:
            add_log(f"Error: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            time.sleep(10)

    bot_state["activity_status"] = "Stopped"
    add_log("Bot stopped", "warning")


def execute_buy(price, regime_data=None, details=None):
    """Execute a buy order with smart sizing - supports multiple positions"""
    global trailing_stop_price, highest_price_since_entry, breakeven_activated, previous_rsi, stop_delay_count

    current_positions = len(bot_state.get("positions", []))
    max_positions = getattr(config, 'MAX_POSITIONS', 2)

    bot_state["activity_status"] = f"Placing BUY order #{current_positions+1}..."
    update_dashboard()

    # Extract confidence from details for position sizing
    confidence = (details or {}).get("confidence")
    position_size = calculate_position_size(regime_data, confidence)
    conf_level = confidence.get("level", "Medium") if confidence else "Medium"
    add_log(f"Smart BUY #{current_positions+1}: ${position_size:.0f} (conf: {conf_level})", "info")

    # Check balance before attempting buy
    usdt_balance = client.get_balance(config.QUOTE_ASSET)
    min_required = position_size * 1.02  # 2% buffer for fees
    if usdt_balance < min_required:
        add_log(f"Insufficient balance: ${usdt_balance:,.2f} USDT (need ${min_required:,.2f})", "error")
        add_log("Add USDT to your Spot wallet to start trading", "warning")
        return

    result = client.place_market_buy(quote_amount=position_size)

    if result.get("status") == "filled":
        # PHASE 6: Capture entry conditions for learning
        entry_conditions = {}
        indicator_combo_key = ""
        if ai_engine:
            # Get current analysis for entry conditions
            try:
                current_analysis = {
                    "rsi": bot_state.get("rsi", {}),
                    "macd": bot_state.get("macd", {}),
                    "bollinger": bot_state.get("bollinger", {}),
                    "ema": bot_state.get("ema", {}),
                    "support_resistance": bot_state.get("support_resistance", {}),
                    "stochastic": bot_state.get("stochastic", {}),
                    "atr": bot_state.get("atr", {}),
                }
                entry_conditions = ai_engine.capture_entry_conditions(current_analysis, bot_state)
                indicator_combo_key = ai_engine.get_indicator_combo_key(current_analysis)
            except Exception as e:
                logger.warning("Could not capture entry conditions: %s", e)
        
        new_position = {
            "entry_price": result["price"],
            "quantity": result["quantity"],
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "amount_usdt": position_size,
            "regime": bot_state["market_regime"],
            "indicator_scores": (details or {}).get("indicator_scores", {}),
            "position_id": current_positions + 1,
            # PHASE 6: Store entry conditions for learning
            "entry_conditions": entry_conditions,
            "indicator_combo_key": indicator_combo_key,
            "volatility": bot_state.get("atr", {}).get("volatility_level", "medium"),
            "lowest_price": result["price"],  # Track lowest price for drawdown
        }

        # Add to positions list
        if "positions" not in bot_state:
            bot_state["positions"] = []
        bot_state["positions"].append(new_position)

        # Also update legacy single position for backward compatibility
        bot_state["position"] = new_position

        # Reset trailing stop and smart stop tracking
        trailing_stop_price = None
        highest_price_since_entry = result["price"]
        breakeven_activated = False  # New position - reset break-even
        previous_rsi = None          # Reset RSI tracking
        stop_delay_count = 0         # Reset AI delay counter
        bot_state["trailing_stop"] = None
        bot_state["highest_price"] = result["price"]
        bot_state["ai_stop_override"] = None

        # Check if trailing stop enabled
        params = regime_data.get("adjusted_params", {}) if regime_data else {}
        bot_state["trailing_stop_enabled"] = params.get("trailing_stop_enabled", True)

        trade = {
            "id": len(bot_state["trade_history"]) + 1,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "BUY",
            "price": result["price"],
            "quantity": result["quantity"],
            "amount": position_size,
            "pnl": 0,
            "pnl_percent": 0,
            "regime": bot_state["market_regime"]
        }
        bot_state["trade_history"].insert(0, trade)
        save_trades()

        add_log(f"BUY filled: {result['quantity']:.6f} BTC @ ${result['price']:,.2f}", "success")
        bot_state["activity_status"] = "Position opened - Monitoring"
        save_bot_state()

        # Send Telegram notification with total profit so far
        calculate_total_profit()  # Refresh totals
        notifier = get_notifier()
        notifier.notify_buy(
            price=result["price"],
            quantity=result["quantity"],
            amount_usdt=position_size,
            regime=bot_state["market_regime"],
            confidence=bot_state.get("confidence", {}).get("level", "Medium"),
            total_profit=bot_state.get("total_profit", 0),
            total_trades=bot_state.get("total_trades", 0),
            win_rate=bot_state.get("win_rate", 0)
        )
    else:
        bot_state["activity_status"] = f"BUY failed: {result.get('message', 'Unknown error')}"
        add_log(f"BUY failed: {result.get('message', 'Unknown error')}", "error")
        # Notify error
        get_notifier().notify_error(f"BUY failed: {result.get('message', 'Unknown error')}")


def execute_sell(price, exit_type="manual", position_index=None):
    """Execute a sell order - supports multiple positions"""
    global trailing_stop_price, highest_price_since_entry, consecutive_losses, daily_losses, daily_trades
    global breakeven_activated, previous_rsi, stop_delay_count

    # Determine which position to sell
    positions = bot_state.get("positions", [])
    if position_index is not None and positions:
        # Sell specific position from multi-position list
        if position_index >= len(positions):
            add_log(f"Invalid position index {position_index}", "error")
            return
        position_to_sell = positions[position_index]
        pos_num = position_index + 1
    elif bot_state["position"]:
        # Legacy single position
        position_to_sell = bot_state["position"]
        pos_num = 1
    else:
        add_log("No position to sell", "error")
        return

    # Proactive check: if actual BTC is dust (below Binance min), clear position and skip sell
    try:
        actual_btc = client.get_balance(config.BASE_ASSET)
        min_qty = 0.00001  # BTCUSDT minimum order size
        if actual_btc < min_qty:
            add_log(f"🔧 Clearing position (BTC {actual_btc:.8f} below min {min_qty} - dust)", "info")
            if position_index is not None and bot_state.get("positions") and 0 <= position_index < len(bot_state["positions"]):
                bot_state["positions"].pop(position_index)
            if not bot_state.get("positions"):
                bot_state["position"] = None
            else:
                bot_state["position"] = bot_state["positions"][0]
            bot_state["activity_status"] = "Position cleared (dust - nothing to sell)"
            save_bot_state()
            return
    except Exception:
        pass

    bot_state["activity_status"] = f"Placing SELL order #{pos_num} ({exit_type})..."
    update_dashboard()

    add_log(f"SELL #{pos_num} ({exit_type}) at ${price:,.2f}", "info")

    result = client.place_market_sell(quantity=position_to_sell["quantity"])

    if result.get("status") == "filled":
        entry_price = position_to_sell["entry_price"]
        exit_price = result["price"]
        quantity = result["quantity"]

        pnl = (exit_price - entry_price) * quantity
        pnl_percent = ((exit_price - entry_price) / entry_price) * 100

        # Update loss tracking
        if pnl > 0:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            daily_losses += abs(pnl)
        
        # PHASE 5: Track daily trade count
        daily_trades += 1

        # PHASE 2 + 6: Record in AI engine for streak tracking, adaptive weights, and advanced learning
        if ai_engine:
            indicator_scores = position_to_sell.get("indicator_scores", {})
            regime = position_to_sell.get("regime", bot_state.get("market_regime"))
            
            # PHASE 6: Get entry conditions and calculate max drawdown
            entry_conditions = position_to_sell.get("entry_conditions", {})
            indicator_combo_key = position_to_sell.get("indicator_combo_key", "")
            volatility = position_to_sell.get("volatility", "medium")
            
            # Calculate max drawdown from lowest price
            lowest_price = position_to_sell.get("lowest_price", entry_price)
            max_drawdown = ((entry_price - lowest_price) / entry_price) * 100 if lowest_price < entry_price else 0
            
            ai_engine.record_trade_result(
                pnl_percent, exit_type,
                indicator_scores_at_entry=indicator_scores if indicator_scores else None,
                regime=regime,
                # PHASE 6: New learning data
                entry_conditions=entry_conditions,
                indicator_combo_key=indicator_combo_key,
                max_drawdown=max_drawdown,
                volatility=volatility
            )
            
            # PHASE 8: Record for parameter optimizer
            try:
                record_trade(
                    is_win=pnl > 0,
                    pnl_percent=pnl_percent,
                    exit_type=exit_type,
                    regime=regime
                )
            except Exception:
                pass
        
        # PHASE 5: Check if ML models need retraining
        try:
            from ai.ml_predictor import MLPredictor
            ml = MLPredictor()
            should_retrain, reason = ml.should_retrain()
            if should_retrain:
                add_log(f"🧠 ML retrain needed: {reason}", "warning")
                # Notify via Telegram
                if notifier:
                    notifier.send_message(f"🧠 ML Retrain Alert: {reason}\nRun: python ml_training.py")
        except Exception:
            pass  # ML predictor not available

        trade = {
            "id": len(bot_state["trade_history"]) + 1,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "SELL",
            "price": result["price"],
            "quantity": result["quantity"],
            "amount": result["quote_amount"],
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "exit_type": exit_type,
            "regime": bot_state.get("market_regime", "unknown"),
            "position_num": pos_num
        }
        bot_state["trade_history"].insert(0, trade)
        save_trades()

        # Remove from positions list
        if position_index is not None and positions:
            bot_state["positions"].pop(position_index)

        # Update legacy position (set to remaining position or None)
        if bot_state.get("positions"):
            bot_state["position"] = bot_state["positions"][0] if bot_state["positions"] else None
        else:
            bot_state["position"] = None

        # Only clear trailing stop if no positions left
        if not bot_state.get("positions") and not bot_state["position"]:
            trailing_stop_price = None
            highest_price_since_entry = None
            breakeven_activated = False  # Reset break-even flag
            previous_rsi = None          # Reset RSI tracking
            stop_delay_count = 0         # Reset AI delay counter
            bot_state["trailing_stop"] = None
            bot_state["highest_price"] = None
            bot_state["ai_stop_override"] = None

        remaining = len(bot_state.get("positions", []))
        level = "success" if pnl >= 0 else "warning"
        add_log(f"SELL #{pos_num}: ${result['quote_amount']:,.2f} | P/L: ${pnl:+,.2f} ({pnl_percent:+.2f}%) | Remaining: {remaining}", level)
        bot_state["activity_status"] = f"Position #{pos_num} closed - {remaining} active"
        save_bot_state()

        # Send Telegram notification with total profit
        notifier = get_notifier()
        notifier.notify_sell(
            price=result["price"],
            quantity=result["quantity"],
            amount_usdt=result["quote_amount"],
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_type=exit_type,
            total_profit=bot_state.get("total_profit", 0),
            total_trades=bot_state.get("total_trades", 0),
            win_rate=bot_state.get("win_rate", 0)
        )
        
        # Auto-send total profit report after each trade
        notifier.notify_total_profit(
            total_profit=bot_state.get("total_profit", 0),
            total_trades=bot_state.get("total_trades", 0),
            winning_trades=bot_state.get("winning_trades", 0),
            losing_trades=bot_state.get("losing_trades", 0),
            win_rate=bot_state.get("win_rate", 0),
            balance=bot_state.get("balance_usdt", 0)
        )
    else:
        error_msg = result.get('message', 'Unknown error')
        
        # === SELF-HEALING: Auto-fix common sell errors ===
        if "below minimum" in error_msg.lower() or "quantity 0" in error_msg.lower() or "no btc to sell" in error_msg.lower():
            # No BTC or dust - clear ghost position so we stop retrying
            add_log("🔧 Clearing position (no balance or dust - nothing to sell)", "info")
            if position_index is not None and bot_state.get("positions") and 0 <= position_index < len(bot_state["positions"]):
                bot_state["positions"].pop(position_index)
            if not bot_state.get("positions"):
                bot_state["position"] = None
            else:
                bot_state["position"] = bot_state["positions"][0]
            bot_state["activity_status"] = "Position cleared (balance too small to sell)"
            save_bot_state()
            return
        if "insufficient balance" in error_msg.lower():
            add_log("🔧 Auto-fixing: Insufficient balance - syncing with Binance...", "info")
            try:
                # Get actual balance and retry
                actual_btc = client.get_balance("BTC")
                if actual_btc > 0.00001:  # More than dust
                    add_log(f"🔧 Retrying with actual balance: {actual_btc:.8f} BTC", "info")
                    
                    # Update position quantity
                    position_to_sell["quantity"] = actual_btc
                    if bot_state["position"]:
                        bot_state["position"]["quantity"] = actual_btc
                    for pos in bot_state.get("positions", []):
                        if pos.get("entry_price") == position_to_sell.get("entry_price"):
                            pos["quantity"] = actual_btc
                    
                    # Retry the sell
                    result = client.place_market_sell(quantity=actual_btc)
                    if result.get("status") == "filled":
                        add_log(f"🔧 Auto-fix successful! SELL completed at ${result['price']:,.2f}", "success")
                        # Process the successful sell...
                        entry_price = position_to_sell["entry_price"]
                        exit_price = result["price"]
                        quantity = result["quantity"]
                        pnl = (exit_price - entry_price) * quantity
                        pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                        
                        # Clear position
                        if position_index is not None and bot_state.get("positions"):
                            bot_state["positions"].pop(position_index)
                        if not bot_state.get("positions"):
                            bot_state["position"] = None
                        else:
                            bot_state["position"] = bot_state["positions"][0]
                        
                        # Notify success
                        get_notifier().notify_sell(
                            price=result["price"],
                            quantity=result["quantity"],
                            amount_usdt=result["quote_amount"],
                            pnl=pnl,
                            pnl_percent=pnl_percent,
                            exit_type=exit_type + "_auto_fixed",
                            total_profit=bot_state.get("total_profit", 0),
                            total_trades=bot_state.get("total_trades", 0),
                            win_rate=bot_state.get("win_rate", 0)
                        )
                        save_bot_state()
                        return
                else:
                    # No BTC - clear ghost position
                    add_log("🔧 Auto-fix: No BTC on exchange - clearing ghost position", "info")
                    bot_state["position"] = None
                    bot_state["positions"] = []
                    save_bot_state()
                    return
            except Exception as e:
                add_log(f"🔧 Auto-fix failed: {e}", "error")
        
        bot_state["activity_status"] = f"SELL failed: {error_msg}"
        add_log(f"SELL failed: {error_msg}", "error")
        # Notify error
        get_notifier().notify_error(f"SELL failed: {error_msg}")


BOT_STATE_FILE = os.path.join(config.DATA_DIR, "bot_state.json")


def save_bot_state():
    """Save bot state to file so it persists across restarts"""
    try:
        state = {
            "running": bot_state["running"],
            "position": bot_state["position"],
            "positions": bot_state.get("positions", []),  # Save multi-positions too
            "logs": bot_state["logs"][:30],
            "decision": bot_state["decision"],
        }
        with open(BOT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(_to_json_serializable(state), f, indent=2)
    except Exception as e:
        logger.error("Error saving bot state: %s", e)


def load_bot_state():
    """Load bot state from file on startup"""
    try:
        if os.path.exists(BOT_STATE_FILE):
            with open(BOT_STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            bot_state["running"] = saved.get("running", False)
            bot_state["position"] = saved.get("position")
            bot_state["positions"] = saved.get("positions", [])
            if saved.get("logs"):
                bot_state["logs"] = saved["logs"]
            bot_state["decision"] = saved.get("decision", "WAITING")
            bot_state["activity_status"] = "Resuming..." if saved.get("running") else "Idle"
    except Exception as e:
        logger.error("Error loading bot state: %s", e)


def recover_position_from_binance():
    """
    Position Recovery: Check Binance balance and trade history
    If we have BTC but no tracked position, recover it from trade history
    """
    global client
    
    try:
        # Initialize client if needed
        if client is None:
            client = BinanceClient()
        
        # Get actual BTC balance from Binance
        btc_balance = client.get_balance(config.BASE_ASSET)
        
        # If we have significant BTC but no position tracked
        min_btc = 0.00001  # Minimum BTC to consider as position
        
        if btc_balance > min_btc and not bot_state.get("position") and not bot_state.get("positions"):
            logger.info("RECOVERY: Found %.8f BTC but no position tracked", btc_balance)
            
            # Look at trade history to find the entry
            trades = bot_state.get("trade_history", [])
            
            # Sort by ID descending to get most recent first
            sorted_trades = sorted(trades, key=lambda x: x.get("id", 0), reverse=True)
            
            # Find the most recent BUY that doesn't have a SELL after it
            # Walk through from newest to oldest
            open_position = None
            for trade in sorted_trades:
                trade_type = trade.get("type", "")
                if trade_type == "SELL":
                    # Found a SELL first = no open position
                    logger.info("RECOVERY: Most recent trade is SELL #%s - no open position", trade.get('id'))
                    break
                elif trade_type == "BUY":
                    # Found a BUY with no SELL after it = open position
                    open_position = trade
                    logger.info("RECOVERY: Found open position: BUY #%s at $%s", trade.get('id'), f"{trade.get('price', 0):,.2f}")
                    break
            
            if open_position:
                # Recover position from the BUY
                recovered_position = {
                    "entry_price": open_position.get("price", 0),
                    "quantity": btc_balance,  # Use actual Binance balance
                    "amount": open_position.get("amount", btc_balance * open_position.get("price", 0)),
                    "time": open_position.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "recovered": True
                }
                
                bot_state["position"] = recovered_position
                bot_state["positions"] = [recovered_position]
                
                logger.info("RECOVERY: Position recovered - Entry $%s, Qty %.8f BTC, Time %s",
                            f"{recovered_position['entry_price']:,.2f}", recovered_position['quantity'], recovered_position['time'])
                
                # Save recovered state
                save_bot_state()
                add_log(f"Position recovered: {btc_balance:.6f} BTC @ ${recovered_position['entry_price']:,.2f}", "success")
                
                return True
            else:
                # Most recent trade was SELL - check if we really have BTC
                # This could be dust or an external deposit
                logger.info("RECOVERY: Have %.8f BTC but last trade was SELL - may be dust or external deposit", btc_balance)
                return False
        
        elif btc_balance <= min_btc:
            # No significant BTC balance
            if bot_state.get("position") or bot_state.get("positions"):
                logger.info("RECOVERY: Position tracked but no BTC balance - clearing stale position")
                bot_state["position"] = None
                bot_state["positions"] = []
                save_bot_state()
            return False
        
        return False
        
    except Exception as e:
        logger.error("RECOVERY: Error: %s", e)
        return False


def save_trades():
    """Save trade history to file and update total profit"""
    try:
        with open(os.path.join(config.DATA_DIR, "trade_history.json"), "w", encoding="utf-8") as f:
            json.dump(bot_state["trade_history"], f, indent=2, ensure_ascii=True)
        calculate_total_profit()  # Update totals after each trade
        save_bot_state()
    except Exception as e:
        logger.error("Error saving trades: %s", e)


def load_trades():
    """Load trade history from file"""
    try:
        if os.path.exists(os.path.join(config.DATA_DIR, "trade_history.json")):
            with open(os.path.join(config.DATA_DIR, "trade_history.json"), "r", encoding="utf-8") as f:
                bot_state["trade_history"] = json.load(f)
        calculate_total_profit()
    except Exception as e:
        logger.error("Error loading trades: %s", e)


def calculate_total_profit():
    """Calculate total profit from all SELL trades"""
    total_profit = 0.0
    total_invested = 0.0
    winning = 0
    losing = 0
    
    for trade in bot_state["trade_history"]:
        if trade.get("type") == "SELL":
            pnl = trade.get("pnl", 0)
            total_profit += pnl
            total_invested += trade.get("amount", 0) - pnl
            
            if pnl > 0:
                winning += 1
            elif pnl < 0:
                losing += 1
    
    total_trades = winning + losing
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    profit_percent = (total_profit / total_invested * 100) if total_invested > 0 else 0
    
    bot_state["total_profit"] = round(total_profit, 2)
    bot_state["total_profit_percent"] = round(profit_percent, 2)
    bot_state["total_trades"] = total_trades
    bot_state["winning_trades"] = winning
    bot_state["losing_trades"] = losing
    bot_state["win_rate"] = round(win_rate, 1)
    
    return total_profit


# Flask Routes
@app.route('/')
def index():
    """Main dashboard page"""
    from flask import make_response
    response = make_response(render_template('dashboard.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/api/state')
def get_state():
    """Get current bot state"""
    try:
        _refresh_data_when_stopped()
        # Ensure JSON-serializable (numpy/pandas types can cause 500)
        state = _to_json_serializable(bot_state)
        return jsonify(state)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return minimal state so frontend doesn't break
        return jsonify({
            "running": bot_state.get("running", False),
            "current_price": 0,
            "last_price": 0,
            "ai_score": 0,
            "decision": "WAITING",
            "activity_status": f"Error: {str(e)}",
            "decision_reasons": [],
            "error": str(e),
        }), 200  # Still 200 so frontend can handle


@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global client, analyzer, ai_engine, regime_detector, bot_thread, stop_bot
    global trailing_stop_price, highest_price_since_entry, consecutive_losses, daily_losses, daily_trades

    if bot_state["running"]:
        return jsonify({"status": "error", "message": "Bot already running"})

    bot_state["running"] = True
    stop_bot.clear()

    # Reset tracking
    trailing_stop_price = None
    highest_price_since_entry = None
    consecutive_losses = 0
    daily_losses = 0
    daily_trades = 0

    # Initialize components
    client = BinanceClient()
    analyzer = TechnicalAnalyzer()
    ai_engine = AIEngine()
    regime_detector = RegimeDetector()

    # Initialize Telegram notifier
    init_notifier()

    # Start bot thread
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

    add_log("Smart bot started in LIVE mode", "success")
    calculate_total_profit()  # Refresh totals before notifying
    get_notifier().notify_start(
        total_profit=bot_state.get("total_profit", 0),
        total_trades=bot_state.get("total_trades", 0),
        win_rate=bot_state.get("win_rate", 0)
    )
    add_log("Features: Regime detection, Confluence, Trailing stops, Smart sizing", "info")
    save_bot_state()

    return jsonify({"status": "ok", "message": "Bot started in LIVE mode"})


@app.route('/api/stop', methods=['POST'])
def stop_bot_route():
    """Stop the trading bot"""
    global stop_bot

    if not bot_state["running"]:
        return jsonify({"status": "error", "message": "Bot not running"})

    stop_bot.set()
    bot_state["running"] = False
    bot_state["decision"] = "STOPPED"
    bot_state["activity_status"] = "Stopped"
    bot_state["decision_reasons"] = []

    add_log("Bot stop requested", "warning")
    save_bot_state()
    update_dashboard()
    calculate_total_profit()  # Refresh totals before notifying
    get_notifier().notify_stop(
        total_profit=bot_state.get("total_profit", 0),
        total_trades=bot_state.get("total_trades", 0),
        win_rate=bot_state.get("win_rate", 0)
    )

    return jsonify({"status": "ok", "message": "Bot stopped"})


@app.route('/api/pause', methods=['POST'])
def pause_bot():
    """Pause trading but keep monitoring"""
    if not bot_state["running"]:
        return jsonify({"status": "error", "message": "Bot not running"})
    
    bot_state["paused"] = True
    bot_state["activity_status"] = "PAUSED - Monitoring only"
    add_log("Trading PAUSED via Telegram", "warning")
    save_bot_state()
    update_dashboard()
    
    return jsonify({"status": "ok", "message": "Trading paused"})


@app.route('/api/resume', methods=['POST'])
def resume_bot():
    """Resume trading after pause"""
    if not bot_state["running"]:
        return jsonify({"status": "error", "message": "Bot not running"})
    
    bot_state["paused"] = False
    bot_state["activity_status"] = "Resumed trading"
    add_log("Trading RESUMED via Telegram", "success")
    save_bot_state()
    update_dashboard()
    
    return jsonify({"status": "ok", "message": "Trading resumed"})


@app.route('/api/health', methods=['GET', 'POST'])
def health_check_endpoint():
    """
    Run health check and return status.
    GET: Get last health status
    POST: Force run health check
    """
    global self_healer
    
    if self_healer is None:
        return jsonify({
            "status": "error", 
            "message": "Self-healing system not initialized"
        })
    
    if request.method == 'POST':
        # Force run health check
        try:
            health_status = self_healer.run_health_check()
            add_log(f"Health check completed: {len(health_status.get('fixes_applied', []))} fixes", "info")
            return jsonify({
                "status": "ok",
                "health": health_status,
                "report": self_healer.get_health_report()
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    else:
        # GET - return last status
        return jsonify({
            "status": "ok",
            "report": self_healer.get_health_report()
        })


@app.route('/api/sync_balance', methods=['POST'])
def sync_balance_endpoint():
    """Force sync position quantity with actual Binance balance"""
    global client
    
    try:
        actual_btc = client.get_balance("BTC")
        position = bot_state.get("position")
        
        if position:
            old_qty = position.get("quantity", 0)
            position["quantity"] = actual_btc
            
            # Also update in positions list
            for pos in bot_state.get("positions", []):
                pos["quantity"] = actual_btc
            
            save_bot_state()
            add_log(f"Balance synced: {old_qty:.8f} -> {actual_btc:.8f} BTC", "success")
            
            return jsonify({
                "status": "ok",
                "old_quantity": old_qty,
                "new_quantity": actual_btc,
                "message": f"Position synced to {actual_btc:.8f} BTC"
            })
        else:
            return jsonify({
                "status": "ok",
                "btc_balance": actual_btc,
                "message": "No position to sync"
            })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# =============================================================================
# META AI AUTONOMOUS SYSTEM ENDPOINTS
# =============================================================================

@app.route('/api/ai/status', methods=['GET'])
def ai_status_endpoint():
    """Get Meta AI status and awareness"""
    try:
        status = ai_status()
        return jsonify({"status": "ok", "ai": status})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/ai/think', methods=['POST'])
def ai_think_endpoint():
    """Trigger AI thinking cycle"""
    try:
        result = ai_think()
        add_log(f"AI Think: {result.get('action_taken', 'unknown')}", "info")
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/ai/start', methods=['POST'])
def ai_start_endpoint():
    """Start autonomous AI operation"""
    global meta_ai
    try:
        meta_ai = start_autonomous_ai()
        add_log("Meta AI autonomous mode started", "success")
        return jsonify({"status": "ok", "message": "Autonomous AI started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/ai/stop', methods=['POST'])
def ai_stop_endpoint():
    """Stop autonomous AI operation"""
    try:
        ai = get_meta_ai()
        ai.stop_autonomous_loop()
        add_log("Meta AI autonomous mode stopped", "info")
        return jsonify({"status": "ok", "message": "Autonomous AI stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/ai/analyze', methods=['POST'])
def ai_analyze_endpoint():
    """Run deep analysis"""
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze_all()
        return jsonify({"status": "ok", "analysis": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/ai/optimize', methods=['POST'])
def ai_optimize_endpoint():
    """Run parameter optimization"""
    try:
        optimizer = get_optimizer()
        result = optimizer.auto_optimize()
        if result.get("adjustments"):
            add_log(f"AI Optimized {len(result['adjustments'])} parameters", "success")
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/ai/evolve', methods=['POST'])
def ai_evolve_endpoint():
    """Run strategy evolution (this may take a while)"""
    try:
        from ai.strategy_evolver import quick_evolve
        add_log("Starting strategy evolution...", "info")
        params = quick_evolve(generations=2, days=7)
        if params:
            add_log("Strategy evolution complete", "success")
            return jsonify({"status": "ok", "evolved_params": params})
        else:
            return jsonify({"status": "ok", "message": "Evolution complete, no improvements found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/ai/goals', methods=['GET'])
def ai_goals_endpoint():
    """Get AI goals and progress"""
    try:
        ai = get_meta_ai()
        goals = [g.to_dict() for g in ai.goals]
        return jsonify({"status": "ok", "goals": goals})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update bot settings and save to config file"""
    data = request.json

    if "trade_amount" in data:
        config.TRADE_AMOUNT_USDT = float(data["trade_amount"])
        add_log(f"Trade amount updated to ${config.TRADE_AMOUNT_USDT}", "info")

    if "profit_target" in data:
        config.PROFIT_TARGET = float(data["profit_target"]) / 100
        add_log(f"Profit target updated to {data['profit_target']}%", "info")

    if "stop_loss" in data:
        config.STOP_LOSS = float(data["stop_loss"]) / 100
        add_log(f"Stop loss updated to {data['stop_loss']}%", "info")

    if "check_interval" in data:
        config.CHECK_INTERVAL = int(data["check_interval"])
        add_log(f"Check interval updated to {config.CHECK_INTERVAL}s", "info")

    # Save settings to config.py file permanently
    save_settings_to_file()

    return jsonify({"status": "ok", "message": "Settings saved!"})


def save_settings_to_file():
    """Save current settings to config.py file"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.py')
        with open(config_path, 'r') as f:
            content = f.read()

        # Update TRADE_AMOUNT_USDT
        import re
        content = re.sub(
            r'TRADE_AMOUNT_USDT = \d+',
            f'TRADE_AMOUNT_USDT = {int(config.TRADE_AMOUNT_USDT)}',
            content
        )

        # Update PROFIT_TARGET
        content = re.sub(
            r'PROFIT_TARGET = [\d.]+',
            f'PROFIT_TARGET = {config.PROFIT_TARGET:.2f}',
            content
        )

        # Update STOP_LOSS
        content = re.sub(
            r'STOP_LOSS = [\d.]+',
            f'STOP_LOSS = {config.STOP_LOSS:.2f}',
            content
        )

        # Update CHECK_INTERVAL
        content = re.sub(
            r'CHECK_INTERVAL = \d+',
            f'CHECK_INTERVAL = {config.CHECK_INTERVAL}',
            content
        )

        with open(config_path, 'w') as f:
            f.write(content)

        logger.info("Settings saved to config.py")
    except Exception as e:
        logger.error("Error saving settings to file: %s", e)


@app.route('/api/get-settings', methods=['GET'])
def get_settings():
    """Get current trading settings"""
    return jsonify({
        "trade_amount": config.TRADE_AMOUNT_USDT,
        "profit_target": config.PROFIT_TARGET * 100,
        "stop_loss": config.STOP_LOSS * 100,
        "check_interval": config.CHECK_INTERVAL
    })


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear trade history"""
    bot_state["trade_history"] = []
    save_trades()
    add_log("Trade history cleared", "info")
    return jsonify({"status": "ok", "message": "History cleared"})


@app.route('/api/save-keys', methods=['POST'])
def save_api_keys():
    """Save API keys to .env file"""
    try:
        data = request.json
        api_key = data.get('api_key', '')
        api_secret = data.get('api_secret', '')
        use_testnet = data.get('use_testnet', False)

        if not api_key or not api_secret:
            return jsonify({"status": "error", "message": "Missing API key or secret"})

        # Write to .env file
        env_content = f"""# Binance API Configuration
# Get your API keys from: https://www.binance.com/en/my/settings/api-management

BINANCE_API_KEY={api_key}
BINANCE_API_SECRET={api_secret}

# Set to True to use Binance testnet (recommended for testing)
USE_TESTNET={str(use_testnet)}
"""
        with open('.env', 'w') as f:
            f.write(env_content)

        # Reload environment variables
        from dotenv import load_dotenv
        load_dotenv(override=True)

        # Reload config
        import importlib
        importlib.reload(config)

        add_log("API keys saved successfully", "info")
        return jsonify({"status": "ok", "message": "API keys saved"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/send-profit-telegram', methods=['POST'])
def send_profit_telegram():
    """Send total profit report to Telegram"""
    try:
        # Calculate totals
        calculate_total_profit()
        
        notifier.notify_total_profit(
            total_profit=bot_state.get("total_profit", 0),
            total_trades=bot_state.get("total_trades", 0),
            winning_trades=bot_state.get("winning_trades", 0),
            losing_trades=bot_state.get("losing_trades", 0),
            win_rate=bot_state.get("win_rate", 0),
            balance=bot_state.get("balance_usdt", 0)
        )
        
        add_log("Total profit report sent to Telegram", "info")
        return jsonify({"status": "ok", "message": "Profit report sent to Telegram"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/check-keys', methods=['GET'])
def check_api_keys():
    """Check if API keys are configured"""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)

        api_key = os.getenv('BINANCE_API_KEY', '').strip()
        use_testnet = os.getenv('USE_TESTNET', 'False').lower() == 'true'

        # Reject common placeholders (case-insensitive)
        placeholders = ('your_api_key_here', 'your_api_key', 'your_secret_here', 
                       'YOUR_API_KEY_HERE', 'xxx')
        is_placeholder = api_key.lower() in [p.lower() for p in placeholders]
        configured = api_key and len(api_key) > 10 and not is_placeholder

        # Mask the API key for display (show first 8 and last 4 chars)
        api_key_masked = ''
        if configured and len(api_key) > 12:
            api_key_masked = api_key[:8] + '••••••••••••' + api_key[-4:]

        return jsonify({
            "configured": configured,
            "testnet": use_testnet,
            "api_key_masked": api_key_masked
        })
    except Exception as e:
        return jsonify({"configured": False, "testnet": False, "api_key_masked": ""})


@app.route('/api/test-connection', methods=['GET'])
def test_api_connection():
    """Test API connection and get balance"""
    try:
        # Reload environment
        from dotenv import load_dotenv
        load_dotenv(override=True)

        # Get fresh API keys
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        use_testnet = os.getenv('USE_TESTNET', 'False').lower() == 'true'

        if not api_key or api_key == 'YOUR_API_KEY_HERE':
            return jsonify({
                "status": "error",
                "message": "API keys not configured"
            })

        # Test real connection directly with Binance API
        from binance.client import Client

        if use_testnet:
            test_client = Client(api_key, api_secret, testnet=True)
        else:
            test_client = Client(api_key, api_secret)

        # Try to get account info - this verifies the API keys work
        account = test_client.get_account()

        # Find USDT balance
        usdt_balance = 0
        for asset in account['balances']:
            if asset['asset'] == 'USDT':
                usdt_balance = float(asset['free'])
                break

        return jsonify({
            "status": "ok",
            "balance": usdt_balance,
            "message": "Connected successfully",
            "testnet": use_testnet
        })

    except Exception as e:
        error_msg = str(e)
        if 'Invalid API-key' in error_msg:
            error_msg = "Invalid API Key - check your key"
        elif 'Signature' in error_msg:
            error_msg = "Invalid API Secret - check your secret"
        elif 'timestamp' in error_msg.lower():
            error_msg = "Time sync error - check your computer time"

        return jsonify({
            "status": "error",
            "message": error_msg
        })


@app.route('/api/telegram/test', methods=['POST'])
def test_telegram():
    """Test Telegram connection and send a test message"""
    try:
        notifier = get_notifier()
        if not notifier.enabled:
            return jsonify({
                "status": "error",
                "message": "Telegram not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env file"
            })

        if notifier.test_connection():
            notifier.send("🧪 Test message from Trading Bot - Connection successful!")
            return jsonify({
                "status": "ok",
                "message": "Test message sent! Check your Telegram."
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Connection failed. Check your bot token."
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/telegram/save', methods=['POST'])
def save_telegram_config():
    """Save Telegram configuration to .env file"""
    try:
        data = request.json
        bot_token = data.get('bot_token', '')
        chat_id = data.get('chat_id', '')

        if not bot_token or not chat_id:
            return jsonify({"status": "error", "message": "Missing bot_token or chat_id"})

        # Read existing .env
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        env_content = ""
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                env_content = f.read()

        # Remove old Telegram settings if present
        lines = env_content.split('\n')
        lines = [l for l in lines if not l.startswith('TELEGRAM_BOT_TOKEN') and not l.startswith('TELEGRAM_CHAT_ID')]
        env_content = '\n'.join(lines).strip()

        # Add new Telegram settings
        env_content += f"\n\n# Telegram Notifications\nTELEGRAM_BOT_TOKEN={bot_token}\nTELEGRAM_CHAT_ID={chat_id}\n"

        with open(env_path, 'w') as f:
            f.write(env_content)

        # Reload environment and reinitialize notifier
        from dotenv import load_dotenv
        load_dotenv(override=True)
        init_notifier(bot_token, chat_id)

        add_log("Telegram configuration saved", "info")
        return jsonify({"status": "ok", "message": "Telegram configured successfully"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/telegram/status', methods=['GET'])
def telegram_status():
    """Check if Telegram is configured"""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)

        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        configured = bool(bot_token and chat_id and len(bot_token) > 10)

        return jsonify({
            "configured": configured,
            "chat_id_masked": chat_id[:4] + "..." if chat_id else ""
        })
    except Exception:
        return jsonify({"configured": False, "chat_id_masked": ""})


@app.route('/api/telegram/send_status', methods=['POST'])
def send_telegram_status():
    """Manually send a full status update to Telegram"""
    try:
        notifier = get_notifier()
        if not notifier.enabled:
            return jsonify({
                "status": "error",
                "message": "Telegram not configured"
            })

        # Send comprehensive status with total profit
        notifier.notify_status(
            current_price=bot_state.get("current_price", 0),
            balance_usdt=bot_state.get("balance_usdt", 0),
            total_value=bot_state.get("total_value", 0),
            position=bot_state.get("position"),
            pnl_percent=bot_state.get("pnl_percent", 0),
            regime=bot_state.get("market_regime", "unknown"),
            total_profit=bot_state.get("total_profit", 0),
            total_trades=bot_state.get("total_trades", 0),
            win_rate=bot_state.get("win_rate", 0)
        )

        return jsonify({
            "status": "ok",
            "message": "Status update sent to Telegram"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/telegram/send_summary', methods=['POST'])
def send_telegram_summary():
    """Send trading summary to Telegram"""
    try:
        notifier = get_notifier()
        if not notifier.enabled:
            return jsonify({
                "status": "error",
                "message": "Telegram not configured"
            })

        # Send daily summary
        notifier.notify_daily_summary(
            total_trades=bot_state.get("total_trades", 0),
            wins=bot_state.get("winning_trades", 0),
            losses=bot_state.get("losing_trades", 0),
            total_profit=bot_state.get("total_profit", 0),
            win_rate=bot_state.get("win_rate", 0),
            balance=bot_state.get("balance_usdt", 0)
        )

        return jsonify({
            "status": "ok",
            "message": "Trading summary sent to Telegram"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


_last_stopped_analysis = 0
_last_stopped_balance = 0
_ANALYSIS_THROTTLE = 10  # Run full analysis every 10 seconds for stable updates
_BALANCE_THROTTLE = 15   # Fetch balance every 15 seconds


def _to_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    import numpy as np
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(i) for i in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Catch-all for numpy scalars (np.bool_, etc. in some numpy versions)
    if hasattr(obj, 'item') and hasattr(obj, 'dtype'):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except ImportError:
        pass
    if hasattr(obj, 'value') and not isinstance(obj, str):  # Enum
        return obj.value
    return obj


def _refresh_data_when_stopped():
    """Fetch price, balance, and run full analysis when bot is stopped (24/7 live updates)"""
    global client
    if bot_state["running"]:
        return
    try:
        from binance.client import Client
        import pandas as pd

        # Show live status
        bot_state["activity_status"] = "Live - Fetching data..."

        # Price (public API - no keys needed) - fast
        pub_client = Client()
        ticker = pub_client.get_symbol_ticker(symbol=config.SYMBOL)
        new_price = float(ticker["price"])
        if bot_state.get("current_price", 0) > 0:
            bot_state["last_price"] = bot_state["current_price"]
        bot_state["current_price"] = new_price
        bot_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Balance (requires API keys - use bot's BinanceClient or create if configured)
        now = time.time()
        if now - _last_stopped_balance > _BALANCE_THROTTLE:
            try:
                _balance_client = client if client else BinanceClient()
                bot_state["balance_usdt"] = _balance_client.get_balance(config.QUOTE_ASSET)
                bot_state["balance_btc"] = _balance_client.get_balance(config.BASE_ASSET)
                bot_state["total_value"] = _balance_client.get_account_value()
                globals()["_last_stopped_balance"] = now
            except Exception:
                pass  # Keep existing values if keys not configured or API fails

        # Run full analysis periodically when stopped (throttled, but always on first load)
        now = time.time()
        first_load = _last_stopped_analysis == 0
        if first_load or (now - _last_stopped_analysis > _ANALYSIS_THROTTLE):
            try:
                bot_state["activity_status"] = "Loading Smart Analysis..."

                klines = pub_client.get_klines(
                    symbol=config.SYMBOL,
                    interval=config.CANDLE_INTERVAL,
                    limit=config.CANDLE_LIMIT
                )
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

                if not df.empty:
                    # Use existing components or create for stopped analysis
                    _analyzer = analyzer or TechnicalAnalyzer()
                    _regime = regime_detector or RegimeDetector()
                    _ai = ai_engine or AIEngine()

                    analysis = _analyzer.analyze(df)
                    regime_data = _regime.detect_regime(df)
                    decision, details = _ai.get_decision(analysis, None, df)

                    bot_state["ai_score"] = details.get("score", 0)
                    bot_state["decision"] = decision.value
                    bot_state["decision_reasons"] = details.get("reason", [])
                    bot_state["activity_status"] = _get_activity_status(decision, details, None)
                    bot_state["market_regime"] = regime_data.get("regime_name", "unknown")
                    bot_state["regime_description"] = regime_data.get("description", "")
                    bot_state["regime_recommendation"] = regime_data.get("recommendation", "")
                    bot_state["adx"] = _to_json_serializable(regime_data.get("adx", {}))
                    bot_state["atr"] = _to_json_serializable(regime_data.get("atr", {}))
                    bot_state["adjusted_params"] = _to_json_serializable(regime_data.get("adjusted_params", {}))

                    conf = details.get("confluence", {})
                    bot_state["confluence"] = {
                        "count": conf.get("count", 0),
                        "direction": conf.get("direction", "neutral"),
                        "strength": conf.get("strength", "none"),
                        "agreeing_indicators": conf.get("agreeing_indicators", []),
                        "bullish_count": conf.get("bullish_count", 0),
                        "bearish_count": conf.get("bearish_count", 0),
                    }

                    conf_details = details.get("confidence", {})
                    bot_state["confidence"] = {
                        "level": conf_details.get("level", "Low"),
                        "value": conf_details.get("value", 0)
                    }

                    div = details.get("divergence", {})
                    bot_state["divergence"] = {
                        "detected": div.get("detected", False),
                        "type": div.get("type"),
                        "description": div.get("description", "No divergence")
                    }

                    ml_pred = details.get("ml_prediction", {})
                    bot_state["ml_prediction"] = {
                        "direction": ml_pred.get("direction", "HOLD"),
                        "confidence": ml_pred.get("confidence", 0),
                        "probability": ml_pred.get("probability", 0.5),
                        "model_votes": ml_pred.get("model_votes", {}),
                        "models_loaded": ml_pred.get("models_loaded", False),
                        "error": ml_pred.get("error", "")
                    }

                    vol = analysis.get("volume", {})
                    bot_state["volume"] = {
                        "confirmed": vol.get("confirmation", False),
                        "signal": vol.get("signal", "neutral"),
                        "ratio": vol.get("ratio", 1.0),
                        "description": vol.get("description", ""),
                        "trend": "rising" if vol.get("trending_up", False) else "falling" if vol.get("ratio", 1) < 0.9 else "flat",
                    }

                    # PHASE 3: ATR-based dynamic stop
                    current_price = bot_state["current_price"]
                    atr_stop = calculate_atr_stop(current_price, regime_data)
                    bot_state["dynamic_stop"] = {
                        "enabled": True,
                        "current_percent": atr_stop["stop_percent"],
                        "regime": atr_stop["regime"],
                        "atr_based": True,
                        "atr_value": atr_stop["atr_value"],
                        "calculated_stop": round(current_price * (1 - atr_stop["stop_percent"]/100), 2),
                        "multiplier": atr_stop["multiplier"]
                    }

                    # PHASE 3: Combined Momentum
                    momentum_data = analysis.get("momentum", {})
                    rsi_data = analysis.get("rsi", {})
                    stoch_data = analysis.get("stochastic", {})
                    bot_state["momentum"] = {
                        "value": momentum_data.get("score", 0),
                        "signal": momentum_data.get("signal", "neutral"),
                        "rsi_component": rsi_data.get("value", 50),
                        "stoch_component": stoch_data.get("k", 50),
                        "combined_oversold": momentum_data.get("both_oversold", False),
                        "combined_overbought": momentum_data.get("both_overbought", False)
                    }

                    # PHASE 3: Fibonacci Levels
                    fib_data = analysis.get("fibonacci", {})
                    bot_state["fibonacci"] = {
                        "levels": fib_data.get("levels", []),
                        "nearest_level": fib_data.get("nearest_level"),
                        "distance_percent": fib_data.get("distance_percent", 0),
                        "signal": fib_data.get("signal", "neutral"),
                        "at_key_level": fib_data.get("at_key_level", False)
                    }

                    # PHASE 3: Candlestick Patterns
                    candle_data = analysis.get("candlestick", {})
                    bot_state["candlestick"] = {
                        "pattern": candle_data.get("pattern"),
                        "signal": candle_data.get("signal", "neutral"),
                        "strength": candle_data.get("score", 0),
                        "description": candle_data.get("description", "")
                    }

                    # PHASE 3: Stochastic (for dashboard)
                    bot_state["stochastic"] = {
                        "k": stoch_data.get("k", 50),
                        "d": stoch_data.get("d", 50),
                        "signal": stoch_data.get("signal", "neutral"),
                        "oversold": stoch_data.get("oversold", False),
                        "overbought": stoch_data.get("overbought", False)
                    }

                    # === PHASE 4: New Technical Indicators ===
                    ichimoku_data = analysis.get("ichimoku", {})
                    bot_state["ichimoku"] = {
                        "signal": ichimoku_data.get("signal", "neutral"),
                        "above_cloud": ichimoku_data.get("above_cloud", False),
                        "below_cloud": ichimoku_data.get("below_cloud", False),
                        "in_cloud": ichimoku_data.get("in_cloud", True),
                        "tk_cross": "bullish" if ichimoku_data.get("tk_cross_bullish") else ("bearish" if ichimoku_data.get("tk_cross_bearish") else None),
                        "score": ichimoku_data.get("score", 0)
                    }

                    mfi_data = analysis.get("mfi", {})
                    bot_state["mfi"] = {
                        "value": mfi_data.get("value", 50),
                        "signal": mfi_data.get("signal", "neutral"),
                        "oversold": mfi_data.get("oversold", False),
                        "overbought": mfi_data.get("overbought", False),
                        "score": mfi_data.get("score", 0)
                    }

                    williams_data = analysis.get("williams_r", {})
                    bot_state["williams_r"] = {
                        "value": williams_data.get("value", -50),
                        "signal": williams_data.get("signal", "neutral"),
                        "oversold": williams_data.get("oversold", False),
                        "overbought": williams_data.get("overbought", False),
                        "score": williams_data.get("score", 0)
                    }

                    cci_data = analysis.get("cci", {})
                    bot_state["cci"] = {
                        "value": cci_data.get("value", 0),
                        "signal": cci_data.get("signal", "neutral"),
                        "oversold": cci_data.get("oversold", False),
                        "overbought": cci_data.get("overbought", False),
                        "score": cci_data.get("score", 0)
                    }

                    # Add to indicators dict for frontend
                    bot_state["indicators"] = {
                        "rsi": analysis.get("rsi", {}),
                        "macd": analysis.get("macd", {}),
                        "bollinger": analysis.get("bollinger", {}),
                        "ema": analysis.get("ema", {}),
                        "sr": analysis.get("support_resistance", {}),
                        "volume": analysis.get("volume", {}),
                        "stochastic": analysis.get("stochastic", {}),
                        "squeeze": analysis.get("squeeze", {}),
                        "ichimoku": ichimoku_data,
                        "mfi": mfi_data,
                        "williams_r": williams_data,
                        "cci": cci_data
                    }

                    # === PHASE 4: External Data Sources (when stopped) ===
                    try:
                        from market.sentiment import SentimentAnalyzer
                        sent_analyzer = SentimentAnalyzer()
                        sent_data = sent_analyzer.analyze()
                        fg = sent_data.get("fear_greed", {})
                        fr = sent_data.get("funding_rate", {})
                        bot_state["sentiment"] = {
                            "fear_greed_value": fg.get("value", 50),
                            "fear_greed_signal": fg.get("signal", "neutral"),
                            "fear_greed_class": fg.get("classification", "Neutral"),
                            "funding_rate": fr.get("rate_pct", 0),
                            "funding_signal": fr.get("signal", "neutral"),
                            "combined_signal": sent_data.get("combined_signal", "neutral"),
                            "risk_level": sent_data.get("risk_level", "moderate"),
                            "score": sent_data.get("combined_score", 0)
                        }
                    except Exception:
                        pass

                    try:
                        from market.order_book import OrderBookAnalyzer
                        ob_analyzer = OrderBookAnalyzer()
                        ob_data = ob_analyzer.analyze()
                        bot_state["order_book"] = {
                            "imbalance_ratio": ob_data.get("imbalance_ratio", 1.0),
                            "signal": ob_data.get("signal", "neutral"),
                            "bid_wall": ob_data.get("bid_wall"),
                            "ask_wall": ob_data.get("ask_wall"),
                            "depth_score": ob_data.get("depth_score", 0),
                            "spread_bps": ob_data.get("spread_bps", 0),
                            "score": ob_data.get("score", 0)
                        }
                    except Exception:
                        pass

                    try:
                        from market.correlation import CorrelationAnalyzer
                        corr_analyzer = CorrelationAnalyzer()
                        corr_data = corr_analyzer.analyze()
                        bot_state["correlation"] = {
                            "btc_dominance": corr_data.get("btc_dominance", 50),
                            "market_cap_change": corr_data.get("market_cap_change_24h", 0),
                            "signal": corr_data.get("signal", "neutral"),
                            "score": corr_data.get("score", 0)
                        }
                    except Exception:
                        pass

                    globals()["_last_stopped_analysis"] = now
            except Exception:
                pass
    except Exception:
        pass


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    _refresh_data_when_stopped()
    safe_state = _to_json_serializable(bot_state)
    emit('update', safe_state)


@socketio.on('request_update')
def handle_request_update():
    """Client requests fresh data"""
    _refresh_data_when_stopped()
    safe_state = _to_json_serializable(bot_state)
    emit('update', safe_state)


def _auto_start_if_was_running():
    """Auto-start the bot if it was running when last saved (24/7 mode)"""
    global client, analyzer, ai_engine, regime_detector, bot_thread, stop_bot
    global trailing_stop_price, highest_price_since_entry, consecutive_losses, daily_losses, daily_trades

    if not bot_state.get("running", False):
        logger.info("24/7: Bot was not running, waiting for manual start or auto-start from frontend")
        return
    try:
        logger.info("24/7: Auto-resuming bot (was running before restart)...")
        stop_bot.clear()
        trailing_stop_price = None
        highest_price_since_entry = None
        consecutive_losses = 0
        daily_losses = 0
        daily_trades = 0
        client = BinanceClient()
        analyzer = TechnicalAnalyzer()
        ai_engine = AIEngine()
        regime_detector = RegimeDetector()
        bot_thread = threading.Thread(target=bot_loop, daemon=True)
        bot_thread.start()
        add_log("Bot auto-resumed for 24/7 live trading", "success")
        logger.info("24/7: Bot auto-started successfully")
    except Exception as e:
        bot_state["running"] = False
        bot_state["decision"] = "STOPPED"
        add_log(f"Auto-resume failed: {e}", "error")
        logger.error("24/7: Auto-resume failed: %s", e)


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    load_trades()
    load_bot_state()
    
    # Position Recovery: Check if we have BTC but lost position tracking
    logger.info("STARTUP: Checking for lost positions...")
    try:
        recovered = recover_position_from_binance()
        if recovered:
            logger.info("STARTUP: Position recovery complete")
        else:
            logger.info("STARTUP: No position recovery needed")
    except Exception as e:
        print(f"  [STARTUP] Position check skipped: {e}")
    
    _auto_start_if_was_running()
    
    # Start Telegram command bot
    try:
        from notifications.telegram_bot import start_bot_thread
        telegram_thread = start_bot_thread()
        if telegram_thread:
            logger.info("Telegram command bot started")
    except Exception as e:
        logger.warning("Telegram bot not started: %s", e)
    
    # Start Meta AI Autonomous System
    try:
        start_autonomous_ai()
        logger.info("Meta AI autonomous system started")
    except Exception as e:
        logger.warning("Meta AI not started: %s", e)

    logger.info("=" * 55)
    logger.info("AI TRADING BOT - FULLY AUTONOMOUS")
    logger.info("Open in browser: http://localhost:5000 | Updates every 3s | Bot checks every 30s")
    logger.info("Features: Self-Healing, Parameter Tuning, Strategy Evolution, Deep Analysis, Goal-Driven Improvement")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
