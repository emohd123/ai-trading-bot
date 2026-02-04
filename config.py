"""
Trading Bot Configuration
"""
import os
import logging
from logging.handlers import RotatingFileHandler
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
# Trading pair
SYMBOL = "BTCUSDT"
BASE_ASSET = "BTC"
QUOTE_ASSET = "USDT"

# Trade amount in quote currency (USDT)
TRADE_AMOUNT_USDT = 40  # Amount to trade per position

# Maximum concurrent positions (2 = faster trading with 2 simultaneous positions)
MAX_POSITIONS = 2  # Allow 2 trades at once - speed up profit generation

# Profit and loss targets (as decimals) - 2:1 Risk/Reward Ratio
PROFIT_TARGET = 0.015  # 1.5% profit target (improved from 1% for better risk/reward)
PROFIT_TARGET_HIGH_VOL = 0.005  # 0.5% profit target in high volatility (faster sells for quick profits)
MIN_PROFIT = 0.0025  # 0.25% minimum profit to take
MIN_PROFIT_HIGH_VOL = 0.0015  # 0.15% minimum profit in high volatility (faster exits)
STOP_LOSS = 0.0075       # 0.75% stop loss (base) - tighter for better risk/reward (2:1 ratio)
STOP_LOSS_TRENDING_DOWN = 0.006  # 0.6% in downtrend - cut losses faster
STOP_LOSS_HIGH_VOL = 0.0075       # 0.75% in high volatility (matches base stop)

# =============================================================================
# AI ENGINE CONFIGURATION
# =============================================================================
# AI score thresholds (very low for maximum trading activity - debug mode)
BUY_THRESHOLD = 0.15   # Buy when AI score > 0.15 (very low - should trade often)
BUY_THRESHOLD_DOWNTREND = 0.25   # In downtrend: require score > 0.25 (lowered for activity)
SELL_THRESHOLD = -0.25 # Sell when AI score < -0.25 (stronger signal needed)
# Allow buys in downtrend with low signal (score > 0.25, confluence >= 3)
NO_BUY_IN_DOWNTREND = False  # False = can buy in downtrend if score >= BUY_THRESHOLD_DOWNTREND

# When buying in downtrend is allowed (or for positions carried into downtrend): use smaller size
POSITION_SIZE_DOWNTREND_MULT = 0.6   # 60% of normal size in downtrend (0.7 = 70%, 1.0 = no reduction)

# Indicator weights for AI scoring (must sum to 1.0) - 10 indicators with ML prediction
# Rebalanced to match your learned accuracy (25 trades): best macd/ml/ichimoku/ema, weak williams_r/sr/mfi
INDICATOR_WEIGHTS = {
    "macd": 0.18,               # Best performer 76.6%
    "ml_prediction": 0.15,      # 74.8%
    "ichimoku": 0.12,           # 70%
    "ema": 0.10,                # 69.4%
    "momentum": 0.10,           # 38%
    "bollinger": 0.10,          # 30.6%
    "support_resistance": 0.08, # 29.1% - needs improvement
    "mfi": 0.07,                # 30%
    "williams_r": 0.05,         # 27.9% - needs improvement
    "cci": 0.05,                # 30%
}

# Entry rules (very low for maximum trading activity - debug mode)
MIN_CONFIDENCE_BUY = 0.20      # Require very low confidence (lowered from 0.30)
MIN_CONFLUENCE_BUY = 3         # At least 3 indicators must agree for BUY (lowered from 4)
MIN_CONFLUENCE_BUY_DOWNTREND = 3  # Same as normal (no extra requirement in downtrend)
MIN_CONFLUENCE_SELL = 4        # At least 4 indicators must agree for SELL
MIN_CONFLUENCE = 5             # Default confluence requirement
REQUIRE_VOLUME_BLOCKING = False # Volume is now a modifier, not a blocker
REQUIRE_VOLUME_RANGING = False  # Don't require volume in ranging/unknown (was True - blocking trades)
VOLUME_NO_CONFIRM_PENALTY = 0.15  # Reduce score by 15% if no volume confirmation
ADAPTIVE_WEIGHTS_ENABLED = True # Learn indicator weights from trade outcomes

# Trailing stop activation - activate earlier to lock profits
TRAILING_ACTIVATION = 0.005     # Activate at +0.5% (lowered from 1% to lock profits earlier)
TRAILING_ACTIVATION_HOT = 0.003  # +0.3% during win streak (even earlier)

# Daily limits
MAX_DAILY_LOSS_PCT = 0.03      # 3% of total portfolio max daily loss
MAX_DAILY_TRADES = 20          # Prevent overtrading

# =============================================================================
# SMART STOP LOSS SYSTEM (Avoid premature stops)
# =============================================================================
# Break-even stop - move stop to entry after reaching this profit %
BREAKEVEN_ACTIVATION = 0.005   # +0.5% profit = move stop to entry (lock in no-loss)
BREAKEVEN_BUFFER = 0.001       # Small buffer above entry (0.1%) to cover fees

# Support-based stops - use support level instead of fixed %
USE_SUPPORT_STOP = True        # Enable support-based stop loss
SUPPORT_STOP_BUFFER = 0.002    # 0.2% below support level

# Recovery check - don't stop if momentum recovering
RECOVERY_CHECK_ENABLED = True  # Check momentum before stopping
RSI_RECOVERY_THRESHOLD = 35   # If RSI > 35 and rising, delay stop

# Time-based exit for stale positions
MAX_POSITION_AGE_HOURS = 4     # Exit negative positions after 4 hours
STALE_LOSS_THRESHOLD = -0.003  # Only time-exit if loss > -0.3%

# =============================================================================
# AI-ASSISTED STOP LOSS (AI can override/delay stops)
# =============================================================================
AI_STOP_ENABLED = True         # Enable AI involvement in stop decisions

# AI Score Override - if AI is bullish, delay stop
AI_BULLISH_OVERRIDE = 0.15     # If AI score > 0.15, delay stop (turning bullish)
AI_STRONG_BULLISH = 0.30       # If AI score > 0.30, give maximum tolerance

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
# 30s = fast reaction | 15s = very fast (more API calls)
CHECK_INTERVAL = 30

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
        fh.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root.addHandler(fh)
    except OSError:
        pass  # Fallback to console only if file fails

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root.addHandler(ch)


# Initialize logging when config is loaded
setup_logging()

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
SHOW_DETAILED_ANALYSIS = True
USE_COLORS = True
