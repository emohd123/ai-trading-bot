"""
Trading Bot Configuration
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# BINANCE API CONFIGURATION
# =============================================================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Use testnet for safe testing (set to True for paper trading)
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() == "true"

# Testnet URLs
TESTNET_API_URL = "https://testnet.binance.vision"
MAINNET_API_URL = "https://api.binance.com"

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================
# Trading pair (current pair; overwritten each loop when using multi-coin rotation)
SYMBOL = "BTCUSDT"
BASE_ASSET = "BTC"
QUOTE_ASSET = "USDT"

# Multi-coin rotation: list of (symbol, base_asset) to choose from
# Extended list for more trading opportunities across market conditions
SYMBOLS = [
    # Major coins
    ("BTCUSDT", "BTC"),
    ("ETHUSDT", "ETH"),
    ("BNBUSDT", "BNB"),
    ("SOLUSDT", "SOL"),
    ("XRPUSDT", "XRP"),
    # Popular altcoins
    ("DOGEUSDT", "DOGE"),
    ("ADAUSDT", "ADA"),
    ("AVAXUSDT", "AVAX"),
    ("MATICUSDT", "MATIC"),
    ("LINKUSDT", "LINK"),
    ("DOTUSDT", "DOT"),
    ("ATOMUSDT", "ATOM"),
    ("LTCUSDT", "LTC"),
    ("UNIUSDT", "UNI"),
    ("NEARUSDT", "NEAR"),
]
# Symbol selection cache: reuse regime/score for this many minutes to limit API calls
SYMBOL_SCAN_CACHE_MINUTES = 0.5  # Scan every 30 seconds for faster gainer detection
# Re-run full symbol selection at most every N minutes when no position (sticky)
SYMBOL_ROTATION_INTERVAL_MINUTES = 0.5  # Re-evaluate every 30 seconds - faster switching
# Optional: minimum AI score to consider a symbol for rotation (default: use BUY_THRESHOLD)
ROTATION_MIN_SCORE = None  # None = use BUY_THRESHOLD

# Trade amount in quote currency (USDT)
TRADE_AMOUNT_USDT = 40  # Amount to trade per position (use up to this when balance allows)
TRADE_AMOUNT_MIN = 10   # Minimum USDT to open a position; if balance below this, skip and log (throttled)

# Maximum concurrent positions (1 = focus on one position, fewer whipsaw losses)
# IMPORTANT: Set to 1 to trade one position at a time and track profit properly
MAX_POSITIONS = 1  # Single position mode - wait for position to close before opening new one

# Profit and loss targets (as decimals) - 2:1 Risk/Reward Ratio
PROFIT_TARGET = 0.015  # 1.5% profit target (improved from 1% for better risk/reward)
PROFIT_TARGET_HIGH_VOL = 0.003  # 0.3% profit target in high volatility (faster sells for quick profits - optimized)
MIN_PROFIT = 0.006   # 0.6% minimum profit to take (above 0.2% round-trip fees)
MIN_PROFIT_HIGH_VOL = 0.004  # 0.4% minimum profit in high volatility (above fees)

# Quick Profit Mode: Exit quickly on fast gains
QUICK_PROFIT_ENABLED = True  # Enable quick profit mode for fast gainers
QUICK_PROFIT_TARGET = 0.008  # 0.8% profit target for quick exits (meaningful after 0.2% fees)
QUICK_PROFIT_TIME_LIMIT = 30  # Exit at quick profit if reached within 30 minutes
STOP_LOSS = 0.007        # 0.7% stop loss (base) - cut losses faster
STOP_LOSS_TRENDING_DOWN = 0.005  # 0.5% in downtrend - cut losses faster
STOP_LOSS_HIGH_VOL = 0.0075      # 0.75% in high volatility (matches previous)

# =============================================================================
# AI ENGINE CONFIGURATION
# =============================================================================
# AI score thresholds (stricter to reduce weak entries and losses)
BUY_THRESHOLD = 0.28   # Buy when AI score > 0.28 (lowered for more active trading while staying smart)
BUY_THRESHOLD_UPTREND = 0.08   # In uptrend: allow even lower score (uptrends are safer, can enter earlier)
UPTREND_SCORE_BOOST = 0.15     # Boost AI score by this amount when regime is trending_up (accounts for indicator lag/overbought)
BUY_THRESHOLD_DOWNTREND = 0.40  # In downtrend: require stronger score to buy
SELL_THRESHOLD = -0.25 # Sell when AI score < -0.25 (in profit)
# Stricter threshold when position is in loss - avoid locking small losses on weak bearish flicker
SELL_THRESHOLD_IN_LOSS = -0.35  # Require score < -0.35 to sell at a loss on AI signal
# Allow buys in downtrend only with stronger signal (BUY_THRESHOLD_DOWNTREND)
NO_BUY_IN_DOWNTREND = True  # True = block weak buys in downtrend (most losses were from weak downtrend buys)
# HIGH SCORE OVERRIDE: If AI score is VERY high, allow trading even in downtrend (AI is confident)
HIGH_SCORE_DOWNTREND_OVERRIDE = 0.70  # Raised from 0.60 - only buy downtrend when AI very confident (reduces losses)
# When True: AI fully decides - no hard block on downtrend; if AI says BUY we allow it (multi-coin auto)
# DISABLED: All recent losses were from downtrend buys. Let NO_BUY_IN_DOWNTREND block them.
AI_DECIDES_ALL = False  # Respect NO_BUY_IN_DOWNTREND to avoid losses in bear markets

# === LOSS AVOIDANCE (stricter after losses) ===
BUY_THRESHOLD_AFTER_LOSS = 0.32   # After 1+ consecutive loss, require slightly stronger signal (more active recovery)
BUY_THRESHOLD_AFTER_TWO_LOSSES = 0.38  # After 2+ consecutive losses, require stronger signal (balanced recovery)
COOLDOWN_AFTER_LOSS_MINUTES = 5  # Wait 5 min after stop loss before next buy (faster recovery - anti-whipsaw)
MIN_TRADE_INTERVAL_SECONDS = 120  # 2 min minimum between ANY trades (prevents rapid-fire overtrading)
TRADING_FEE_RATE = 0.001          # 0.1% per trade (Binance standard maker/taker fee; 0.2% round-trip)
ONLY_BUY_STRICT_UPTREND = False   # True = only buy when regime is "trending_up" (not "ranging") - very conservative
MIN_CONFLUENCE_AFTER_LOSS = 4     # After a loss, require 4+ indicators (more active recovery, normal is 5)
LOSS_AVOIDANCE_TIMEOUT_MINUTES = 60  # Reset loss avoidance after 60 min if no trades (prevents permanent blocking)

# When buying in downtrend is allowed (or for positions carried into downtrend): use smaller size
POSITION_SIZE_DOWNTREND_MULT = 0.6   # 60% of normal size in downtrend (0.7 = 70%, 1.0 = no reduction)

# Indicator weights for AI scoring (must sum to 1.0) - 10 indicators
# ML prediction reduced to 0.02 until model accuracy improves beyond 55%
# LSTM disabled. Best performers get most weight.
INDICATOR_WEIGHTS = {
    "momentum": 0.25,           # BEST: 76.9% accuracy
    "bollinger": 0.20,          # GOOD: 76.8% accuracy
    "support_resistance": 0.17, # GOOD: 76.8% accuracy
    "mfi": 0.12,                # GOOD: 76.8% accuracy
    "cci": 0.10,                # GOOD: 76.8% accuracy
    "williams_r": 0.05,         # AVG: 48.2% accuracy
    "ml_prediction": 0.02,      # REDUCED: ~50% accuracy (near random) - reduced until retrained
    "macd": 0.04,               # WEAK: 23.2% accuracy
    "ichimoku": 0.03,           # WEAK: 23.2% accuracy
    "ema": 0.02,                # WEAK: 23.2% accuracy
}

# Entry rules (stricter to reduce weak entries and losses)
MIN_CONFIDENCE_BUY = 0.50      # Require higher confidence for BUY (more conservative)
MIN_CONFIDENCE_BUY_UPTREND = 0.20  # Lower confidence required in uptrends (safer market conditions)
MIN_CONFLUENCE_BUY = 5         # At least 5 indicators must agree for BUY (more active while still smart)
MIN_CONFLUENCE_BUY_UPTREND = 2  # Lower confluence required in uptrends (2/10 indicators - very active in uptrends)
MIN_CONFLUENCE_BUY_DOWNTREND = 6  # Even stricter in downtrend if allowed (was 4)
MIN_CONFLUENCE_SELL = 4        # At least 4 indicators must agree for SELL
MIN_CONFLUENCE = 5             # Default confluence requirement
REQUIRE_VOLUME_BLOCKING = False # Volume is now a modifier, not a blocker
REQUIRE_VOLUME_RANGING = False  # Don't require volume in ranging/unknown (was True - blocking trades)
VOLUME_NO_CONFIRM_PENALTY = 0.15  # Reduce score by 15% if no volume confirmation
ADAPTIVE_WEIGHTS_ENABLED = True # Learn indicator weights from trade outcomes

# Trailing stop activation - activate earlier to lock profits
TRAILING_ACTIVATION = 0.005     # Activate at +0.5% (above fees, prevents premature trailing)
TRAILING_ACTIVATION_HOT = 0.003  # +0.3% during win streak (above fees)

# Gainer Detection Boost Settings
GAINER_BOOST_1H_THRESHOLD = 0.02  # +2% gain in 1h triggers boost
GAINER_BOOST_1H_VALUE = 0.15      # +0.15 boost for 1h gainers
GAINER_BOOST_4H_THRESHOLD = 0.05  # +5% gain in 4h triggers boost
GAINER_BOOST_4H_VALUE = 0.12      # +0.12 boost for 4h gainers
GAINER_BOOST_VOLUME_THRESHOLD = 0.5  # 50% volume spike triggers boost
GAINER_BOOST_VOLUME_VALUE = 0.10     # +0.10 boost for volume spikes
GAINER_BOOST_MAX = 0.25              # Maximum total gainer boost (cap to avoid overconfidence)

# Daily limits
MAX_DAILY_LOSS_PCT = 0.03      # 3% of total portfolio max daily loss
MAX_DAILY_TRADES = 12          # Cap round-trips to reduce churn and fee drag

# =============================================================================
# SMART STOP LOSS SYSTEM (Avoid premature stops)
# =============================================================================
# Break-even stop - move stop to entry after reaching this profit %
BREAKEVEN_ACTIVATION = 0.005   # +0.5% profit = move stop to entry (lock in no-loss)
BREAKEVEN_BUFFER = 0.003       # 0.3% buffer above entry to cover round-trip fees (0.2%) + small profit

# Support-based stops - use support level instead of fixed %
USE_SUPPORT_STOP = True        # Enable support-based stop loss
SUPPORT_STOP_BUFFER = 0.002    # 0.2% below support level

# Recovery check - don't stop if momentum recovering
RECOVERY_CHECK_ENABLED = True  # Check momentum before stopping
RSI_RECOVERY_THRESHOLD = 35   # If RSI > 35 and rising, delay stop

# Time-based exit for stale positions
MAX_POSITION_AGE_HOURS = 4     # Exit negative positions after 4 hours
STALE_LOSS_THRESHOLD = -0.003  # Only time-exit if loss > -0.3%

# Minimum hold time before regular stop loss applies (reduces whipsaw; hard stop still applies)
MIN_HOLD_ENABLED = True        # If True, skip regular stop if position age < MIN_HOLD_MINUTES
MIN_HOLD_MINUTES = 45          # Don't apply regular stop in first 45 min (hard stop & breakeven still apply)

# =============================================================================
# AI-ASSISTED STOP LOSS (AI can override/delay stops)
# =============================================================================
AI_STOP_ENABLED = True         # Enable AI involvement in stop decisions

# AI Score Override - if AI is bullish, delay stop
# NOTE: These must be high enough that only genuinely strong signals delay stops.
# Too low (was 0.15/0.30) = bot delays stops endlessly on mediocre scores, bleeding losses.
AI_BULLISH_OVERRIDE = 0.45     # If AI score > 0.45, delay stop (must be genuinely bullish)
AI_STRONG_BULLISH = 0.60       # If AI score > 0.60, give maximum tolerance (strong conviction only)

# AI Decision Override - if AI says HOLD, widen stop
AI_HOLD_STOP_MULTIPLIER = 1.5  # If AI=HOLD, multiply stop % by 1.5x (wider)
AI_SELL_IMMEDIATE = True       # If AI says SELL, don't delay stop

# Confluence Override - need X indicators bearish to stop
MIN_BEARISH_CONFLUENCE = 4     # Need 4+ bearish indicators to confirm stop
CONFLUENCE_OVERRIDE = True     # Enable confluence check before stopping

# Max delay cycles - don't delay forever
MAX_STOP_DELAY_CYCLES = 3      # Max times to delay stop (prevent holding losers)
# Note: In downtrend the dashboard uses 0 delays so we cut losses quickly (no "AI override" delay)
HARD_STOP_LIMIT = 0.02         # 2% = absolute max loss, AI cannot override (backup when smart stop delays)

# =============================================================================
# ATR-BASED DYNAMIC STOPS
# =============================================================================
ATR_STOP_MULTIPLIER = 2.0      # Base: 2x ATR for stop loss
ATR_STOP_MIN = 0.03            # Minimum 3% stop loss
ATR_STOP_MAX = 0.07            # Maximum 7% stop loss
ATR_TRENDING_DOWN_MULT = 0.8   # Tighter stops in downtrend
ATR_HIGH_VOLATILITY_MULT = 1.5 # Wider stops in high volatility

# Trailing stop by trend strength
TRAIL_ADX_STRONG = 40          # ADX > 40 = strong trend
TRAIL_ADX_MODERATE = 30        # ADX > 30 = moderate trend
TRAIL_ATR_STRONG = 2.5         # 2.5x ATR for strong trends
TRAIL_ATR_MODERATE = 2.0       # 2.0x ATR for moderate trends
TRAIL_ATR_HIGH_VOL = 3.0       # 3.0x ATR for high volatility

# =============================================================================
# DIVERGENCE DETECTION
# =============================================================================
DIVERGENCE_LOOKBACK = 20       # Extended from 10 to 20 candles
DIVERGENCE_STRONG_CANDLES = 15 # 15+ candles = strong divergence (+0.25)
DIVERGENCE_MODERATE_CANDLES = 10  # 10-15 candles = moderate (+0.15)
DIVERGENCE_WEAK_BOOST = 0.10   # Weak divergence boost
DIVERGENCE_MODERATE_BOOST = 0.15  # Moderate divergence boost
DIVERGENCE_STRONG_BOOST = 0.25    # Strong divergence boost

# =============================================================================
# 15-MINUTE TIMEFRAME REFINEMENT
# =============================================================================
USE_15M_REFINEMENT = True      # Enable 15m entry refinement
STOCH_15M_OVERBOUGHT = 75      # 15m overbought - wait for pullback
STOCH_15M_OVERSOLD = 25        # 15m oversold - confirm entry (+0.1 boost)
ENTRY_REFINEMENT_BOOST = 0.10  # Boost when 15m confirms entry

# =============================================================================
# DATA PATHS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # Runtime data (state, history, learning)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")  # ML model files
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure data dir exists

# =============================================================================
# LEARNING PERSISTENCE
# =============================================================================
LEARNING_FILE = os.path.join(DATA_DIR, "ai_learning.json")  # File to save learning state
MAX_TRADE_HISTORY = 50         # Keep last 50 trades for learning (increased from 20)

# =============================================================================
# ENHANCED INDICATOR LEARNING (PHASE 7)
# =============================================================================
# Per-indicator agreement thresholds (tuned for each indicator type)
INDICATOR_THRESHOLDS = {
    "momentum": 0.30,           # RSI+Stochastic needs stronger signal
    "macd": 0.20,               # MACD crossovers are clearer
    "bollinger": 0.25,          # Standard threshold
    "ema": 0.15,                # EMA trends are gradual
    "support_resistance": 0.30, # S/R needs clear bounce
    "ml_prediction": 0.35,      # ML needs higher confidence
    "ichimoku": 0.20,           # Cloud signals are binary
    "mfi": 0.25,                # Volume-weighted RSI
    "williams_r": 0.30,         # Momentum oscillator
    "cci": 0.25,                # Mean reversion indicator
}

# Profit-weighted learning - larger wins/losses count more
PROFIT_WEIGHTED_LEARNING = True    # Enable profit-weighted accuracy updates
PROFIT_WEIGHT_CAP = 2.0            # Maximum weight multiplier (cap at 2x)
LOSS_WEIGHT_FACTOR = 0.5           # Losses weighted at 50% of wins (asymmetric)

# Time decay for learning - recent trades matter more
LEARNING_TIME_DECAY = True         # Enable time-based decay
DECAY_RATE_PER_DAY = 0.95          # 5% decay per day (0.95^days)
MIN_DECAY_FACTOR = 0.3             # Minimum decay factor (never go below 30%)

# Statistical significance thresholds
MIN_SAMPLES_FOR_LEARNING = 8       # Minimum trades per indicator before applying
MIN_ACCURACY_DEVIATION = 0.08      # At least 8% deviation from 0.5 to apply

# Directional learning - track bullish vs bearish accuracy separately
DIRECTIONAL_LEARNING = True        # Enable separate bullish/bearish tracking

# Regime-specific base weights (learned weights blend with these)
REGIME_BASE_WEIGHTS = {
    "trending_up": {
        "momentum": 0.20, "macd": 0.15, "bollinger": 0.10, "ema": 0.10,
        "support_resistance": 0.10, "ml_prediction": 0.10, "ichimoku": 0.10,
        "mfi": 0.05, "williams_r": 0.05, "cci": 0.05
    },
    "trending_down": {
        "momentum": 0.15, "macd": 0.12, "bollinger": 0.12, "ema": 0.05,
        "support_resistance": 0.18, "ml_prediction": 0.08, "ichimoku": 0.12,
        "mfi": 0.08, "williams_r": 0.05, "cci": 0.05
    },
    "ranging": {
        "momentum": 0.12, "macd": 0.10, "bollinger": 0.18, "ema": 0.05,
        "support_resistance": 0.20, "ml_prediction": 0.08, "ichimoku": 0.08,
        "mfi": 0.08, "williams_r": 0.06, "cci": 0.05
    },
    "high_volatility": {
        "momentum": 0.15, "macd": 0.10, "bollinger": 0.15, "ema": 0.05,
        "support_resistance": 0.12, "ml_prediction": 0.10, "ichimoku": 0.10,
        "mfi": 0.08, "williams_r": 0.08, "cci": 0.07
    }
}

# Multi-timeframe
MTF_4H_INTERVAL = "4h"
MTF_15M_INTERVAL = "15m"

# =============================================================================
# TECHNICAL ANALYSIS CONFIGURATION
# =============================================================================
# RSI settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30    # Buy signal
RSI_OVERBOUGHT = 70  # Sell signal

# MACD settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands settings
BB_PERIOD = 20
BB_STD_DEV = 2

# EMA settings
EMA_SHORT = 9
EMA_LONG = 21

# Support/Resistance lookback
SR_LOOKBACK = 50

# =============================================================================
# BOT OPERATION SETTINGS - 24/7 LIVE TRADING
# =============================================================================
# How often to check prices and analyze (in seconds)
# 30s = fast reaction | 15s = very fast | 10s = real-time (more API calls)
CHECK_INTERVAL = 15  # Check every 15 seconds for near real-time updates

# ML inference throttle: run ML at most every N seconds (avoids blocking the loop)
ML_INTERVAL_SEC = 300  # 5 minutes; set to 60 for more frequent updates
# Set to False for faster inference (LSTM is slow); weight is redistributed to RF/XGB/LGB
ML_USE_LSTM = False

# =============================================================================
# POSITION SIZE LEARNING CONFIGURATION
# =============================================================================
# ML-based position size learning: learns optimal trade amounts from history
POSITION_SIZE_LEARNING_ENABLED = True  # Enable position size learning
POSITION_SIZE_MIN_SAMPLES = 5  # Minimum trades per condition before using learned size
POSITION_SIZE_BLEND_RATIO = 0.7  # Blend ratio: 70% learned, 30% current system (gradual adoption)

# Number of historical candles to fetch for analysis
CANDLE_LIMIT = 100

# Candle timeframe
CANDLE_INTERVAL = "1h"  # 1 hour candles (primary)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_FILE = os.path.join(DATA_DIR, "trading_bot.log")
LOG_LEVEL = "INFO"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3


def setup_logging():
    """Configure logging for the trading bot. Call once at startup."""
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    # Check if structured logging is enabled
    use_structured = os.getenv("STRUCTURED_LOGGING", "False").lower() == "true"
    
    if use_structured:
        # JSON format for log aggregation
        import json as json_lib
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                return json_lib.dumps(log_data)
        log_format = JSONFormatter()
    else:
        # Standard format
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    if root.handlers:
        return  # Already configured
    root.setLevel(log_level)

    # File handler (rotating)
    try:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
        fh.setLevel(log_level)
        if use_structured:
            fh.setFormatter(log_format)
        else:
            fh.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root.addHandler(fh)
    except OSError:
        pass  # Fallback to console only if file fails

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    if use_structured:
        ch.setFormatter(log_format)
    else:
        ch.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root.addHandler(ch)


# Initialize logging when config is loaded
setup_logging()


def validate_config():
    """
    Validate configuration on startup.
    Returns list of validation errors (empty if all valid).
    """
    errors = []
    warnings = []
    
    # Check API keys
    if not BINANCE_API_KEY or BINANCE_API_KEY.strip() == "":
        errors.append("BINANCE_API_KEY is not set")
    elif BINANCE_API_KEY in ["YOUR_API_KEY_HERE", "xxx", ""]:
        errors.append("BINANCE_API_KEY appears to be a placeholder - please set a real API key")
    
    if not BINANCE_API_SECRET or BINANCE_API_SECRET.strip() == "":
        errors.append("BINANCE_API_SECRET is not set")
    elif BINANCE_API_SECRET in ["YOUR_SECRET_HERE", "xxx", ""]:
        errors.append("BINANCE_API_SECRET appears to be a placeholder - please set a real API secret")
    
    # Check required directories
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            logger.info(f"Created DATA_DIR: {DATA_DIR}")
        except Exception as e:
            errors.append(f"Cannot create DATA_DIR {DATA_DIR}: {e}")
    
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            logger.info(f"Created MODEL_DIR: {MODEL_DIR}")
        except Exception as e:
            errors.append(f"Cannot create MODEL_DIR {MODEL_DIR}: {e}")
    
    # Validate trading parameters
    if TRADE_AMOUNT_USDT <= 0:
        errors.append(f"TRADE_AMOUNT_USDT must be > 0, got {TRADE_AMOUNT_USDT}")
    elif TRADE_AMOUNT_USDT < 10:
        warnings.append(f"TRADE_AMOUNT_USDT is very low ({TRADE_AMOUNT_USDT}), may hit minimum order size limits")
    
    if PROFIT_TARGET <= 0 or PROFIT_TARGET > 1:
        errors.append(f"PROFIT_TARGET must be between 0 and 1, got {PROFIT_TARGET}")
    
    if STOP_LOSS <= 0 or STOP_LOSS > 1:
        errors.append(f"STOP_LOSS must be between 0 and 1, got {STOP_LOSS}")
    
    if STOP_LOSS >= PROFIT_TARGET:
        warnings.append(f"STOP_LOSS ({STOP_LOSS*100}%) >= PROFIT_TARGET ({PROFIT_TARGET*100}%) - risk/reward ratio is poor")
    
    if CHECK_INTERVAL < 5:
        warnings.append(f"CHECK_INTERVAL is very low ({CHECK_INTERVAL}s), may cause rate limiting")
    elif CHECK_INTERVAL > 300:
        warnings.append(f"CHECK_INTERVAL is very high ({CHECK_INTERVAL}s), may miss trading opportunities")
    
    # Check for dangerous configurations
    hard_stop_limit = globals().get('HARD_STOP_LIMIT', 0.02)
    if hard_stop_limit > 0.10:
        warnings.append(f"HARD_STOP_LIMIT is very high ({hard_stop_limit*100}%), consider lower value")
    
    max_daily_loss_pct = globals().get('MAX_DAILY_LOSS_PCT', 0.03)
    if max_daily_loss_pct > 0.20:
        warnings.append(f"MAX_DAILY_LOSS_PCT is very high ({max_daily_loss_pct*100}%), consider lower value")
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"Config warning: {warning}")
    
    return errors

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
SHOW_DETAILED_ANALYSIS = True
USE_COLORS = True

# =============================================================================
# PHASE ENHANCEMENTS - NEW CONFIGURATION
# =============================================================================

# --- PHASE 1: Risk Management ---
# Note: HARD_STOP_LIMIT is defined above in AI-ASSISTED STOP LOSS section (0.02 = 2%)

# Slippage buffer for stop loss execution
STOP_LOSS_SLIPPAGE_BUFFER = 0.002  # 0.2% buffer

# Maximum drawdown limit (triggers recovery mode)
MAX_DRAWDOWN_LIMIT = 0.10  # 10% max drawdown

# Position size reduction in recovery mode
DRAWDOWN_REDUCTION_FACTOR = 0.5  # 50% of normal size in recovery

# Trailing stop percentage
TRAILING_STOP_PCT = 0.003  # 0.3% trail

# --- PHASE 4: Portfolio Risk Management ---
# Maximum portfolio exposure (as percentage of total value)
MAX_PORTFOLIO_EXPOSURE = 0.80  # 80% max in positions

# Maximum single position size (as percentage of total value)
MAX_SINGLE_POSITION = 0.40  # 40% max per position

# --- PHASE 6: Paper Trading ---
# Enable paper trading mode (simulated trades)
PAPER_TRADING_MODE = False  # Set True to enable paper trading

# Paper trading initial balance
PAPER_TRADING_BALANCE = 1000.0  # Starting USDT for paper trading

# --- PHASE 6: REST API ---
# API key for protected endpoints (set to None to disable)
API_KEY = os.getenv("BOT_API_KEY", None)

# --- PHASE 5: Alerts ---
# Enable alert system
ALERTS_ENABLED = True

# Alert thresholds
ALERT_WIN_RATE_THRESHOLD = 45  # Alert if win rate drops below 45%
ALERT_DAILY_LOSS_WARNING = 2.0  # Alert at 2% daily loss
ALERT_DRAWDOWN_WARNING = 7.0  # Alert at 7% drawdown
ALERT_NO_TRADE_HOURS = 8  # Alert if no trades for 8 hours

# --- PHASE 7: Database & Caching ---
# Enable SQLite database for persistence
USE_DATABASE = True

# Enable caching for performance
USE_CACHE = True

# Cache TTL settings (in seconds)
CACHE_TTL_MARKET = 60  # 1 minute for market data
CACHE_TTL_INDICATOR = 300  # 5 minutes for indicator calculations
CACHE_TTL_ML = 300  # 5 minutes for ML predictions
