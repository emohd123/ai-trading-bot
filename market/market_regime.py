"""
Market Regime Detection Module - PHASE 3 ENHANCED
Automatically detects market conditions and adjusts strategy parameters
No manual tuning needed - everything auto-adjusts!

PHASE 3 Enhancements:
- ATR-based dynamic stop losses (2x ATR, capped 3-7%)
- Regime-adjusted confluence requirements (5 indicators)
- Improved trailing stops based on trend strength (ADX)
"""
from enum import Enum
from typing import Dict
import pandas as pd
import numpy as np
import config


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


class RegimeDetector:
    """
    Automatically detects market regime using ADX and ATR
    All parameters are pre-tuned - no user configuration needed!
    """

    def __init__(self):
        # ADX parameters (pre-tuned for crypto)
        self.adx_period = 14
        self.adx_trend_threshold = 25  # Above = trending
        self.adx_strong_trend = 40     # Above = strong trend

        # ATR parameters for volatility
        self.atr_period = 14
        self.volatility_lookback = 50  # Compare to historical

    def detect_regime(self, df: pd.DataFrame) -> Dict:
        """
        Detect current market regime and return auto-adjusted parameters

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Dictionary with regime info and adjusted parameters
        """
        if df.empty or len(df) < self.adx_period * 2:
            return self._get_default_result()

        adx_data = self._calculate_adx(df)
        atr_data = self._calculate_atr(df)

        # Determine regime
        regime = self._classify_regime(adx_data, atr_data, df)

        # Get auto-adjusted parameters
        adjusted_params = self._get_regime_parameters(regime, atr_data)

        return {
            "regime": regime,
            "regime_name": regime.value,
            "adx": adx_data,
            "atr": atr_data,
            "adjusted_params": adjusted_params,
            "description": self._get_regime_description(regime),
            "recommendation": self._get_regime_recommendation(regime)
        }

    def _get_default_result(self) -> Dict:
        """Return default result when insufficient data"""
        return {
            "regime": MarketRegime.UNKNOWN,
            "regime_name": "unknown",
            "adx": {"value": 0, "is_trending": False, "trend_direction": "unknown"},
            "atr": {"value": 0, "percent": 0, "volatility_ratio": 1, "is_high_volatility": False},
            "adjusted_params": self._get_regime_parameters(MarketRegime.UNKNOWN, {}),
            "description": "Analyzing market...",
            "recommendation": "Gathering data"
        }

    def _calculate_adx(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Average Directional Index for trend strength

        ADX > 25 = Trending market
        ADX > 40 = Strong trend
        ADX < 20 = Ranging/choppy market
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed averages
        atr = tr.rolling(window=self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)

        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(window=self.adx_period).mean()

        current_adx = adx.iloc[-1] if not adx.empty else 0
        current_plus_di = plus_di.iloc[-1] if not plus_di.empty else 0
        current_minus_di = minus_di.iloc[-1] if not minus_di.empty else 0

        # Handle NaN
        current_adx = 0 if np.isnan(current_adx) else current_adx
        current_plus_di = 0 if np.isnan(current_plus_di) else current_plus_di
        current_minus_di = 0 if np.isnan(current_minus_di) else current_minus_di

        is_trending = current_adx > self.adx_trend_threshold
        is_strong_trend = current_adx > self.adx_strong_trend

        return {
            "value": round(current_adx, 2),
            "plus_di": round(current_plus_di, 2),
            "minus_di": round(current_minus_di, 2),
            "is_trending": is_trending,
            "is_strong_trend": is_strong_trend,
            "trend_direction": "up" if current_plus_di > current_minus_di else "down",
            "trend_strength": "strong" if is_strong_trend else "moderate" if is_trending else "weak"
        }

    def _calculate_atr(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Average True Range for volatility measurement

        ATR tells us how much the price typically moves
        Higher ATR = More volatile = Need wider stops
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=self.atr_period).mean()
        current_atr = atr.iloc[-1] if not atr.empty else 0
        current_price = close.iloc[-1] if not close.empty else 1

        # Handle NaN
        current_atr = 0 if np.isnan(current_atr) else current_atr

        # ATR as percentage of price
        atr_percent = (current_atr / current_price) * 100 if current_price > 0 else 0

        # Compare to historical ATR (is volatility higher than normal?)
        lookback = min(self.volatility_lookback, len(atr))
        historical_atr = atr.tail(lookback).mean() if lookback > 0 else current_atr
        historical_atr = 0 if np.isnan(historical_atr) else historical_atr

        volatility_ratio = current_atr / historical_atr if historical_atr > 0 else 1

        # Determine volatility level
        is_high_volatility = volatility_ratio > 1.5
        is_extreme_volatility = volatility_ratio > 2.0

        return {
            "value": round(current_atr, 2),
            "percent": round(atr_percent, 3),
            "historical_avg": round(historical_atr, 2),
            "volatility_ratio": round(volatility_ratio, 2),
            "is_high_volatility": is_high_volatility,
            "is_extreme_volatility": is_extreme_volatility,
            "volatility_level": "extreme" if is_extreme_volatility else "high" if is_high_volatility else "normal"
        }

    def _classify_regime(self, adx_data: Dict, atr_data: Dict, df: pd.DataFrame) -> MarketRegime:
        """Classify the current market regime"""

        # Extreme volatility takes precedence - be careful!
        if atr_data.get("is_extreme_volatility", False):
            return MarketRegime.HIGH_VOLATILITY

        # Check for trending market
        if adx_data.get("is_trending", False):
            if adx_data.get("trend_direction") == "up":
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN

        # High volatility but not trending = dangerous
        if atr_data.get("is_high_volatility", False):
            return MarketRegime.HIGH_VOLATILITY

        # Default to ranging
        return MarketRegime.RANGING

    def _get_regime_parameters(self, regime: MarketRegime, atr_data: Dict) -> Dict:
        """
        Auto-adjust trading parameters based on regime
        These override the defaults automatically - no user input needed!

        PHASE 3: Now includes ATR-based dynamic stops and 5-indicator confluence
        """
        # PHASE 3: Calculate ATR-based stop loss
        atr_stop = self._calculate_atr_stop(regime, atr_data)

        # Base parameters
        base_params = {
            "buy_threshold": 0.3,
            "sell_threshold": -0.4,
            "confluence_required": 3,        # PHASE 3: 3/5 indicators (60%)
            "stop_loss_multiplier": 1.0,
            "profit_target_multiplier": 1.0,
            "position_size_multiplier": 1.0,
            "trailing_stop_enabled": True,
            "trailing_stop_activation": 2.0,  # Activate after 2% profit
            "trailing_stop_distance": 1.5,    # ATR multiplier for trail distance
            "use_atr_stops": True,
            "atr_stop_percent": atr_stop,     # PHASE 3: Dynamic ATR-based stop
            "trailing_atr_mult": 2.0          # PHASE 3: Trailing stop ATR multiplier
        }

        if regime == MarketRegime.TRENDING_UP:
            # Bullish trend: easier to buy, harder to sell, use trailing stops
            # PHASE 3: 3/5 confluence (60%) - follow momentum
            trailing_mult = self._get_trailing_mult(atr_data)
            return {
                **base_params,
                "buy_threshold": 0.2,           # Lower - easier to enter
                "sell_threshold": -0.5,         # Higher - let profits run
                "confluence_required": 3,        # PHASE 3: 3/5 (60%) in trends
                "position_size_multiplier": 1.2, # Slightly larger positions
                "trailing_stop_enabled": True,
                "trailing_stop_activation": 1.5, # Activate earlier
                "profit_target_multiplier": 1.5, # Bigger targets in trends
                "atr_stop_percent": atr_stop,
                "trailing_atr_mult": trailing_mult  # PHASE 3: Dynamic trailing
            }

        elif regime == MarketRegime.TRENDING_DOWN:
            # Bearish trend: be selective - only strong bounce setups; smaller position size
            down_mult = getattr(config, 'POSITION_SIZE_DOWNTREND_MULT', 0.6)
            return {
                **base_params,
                "buy_threshold": 0.30,          # Higher - need stronger signal
                "sell_threshold": -0.15,        # Exit quickly
                "confluence_required": 3,        # Need 3/6 indicators - avoid weak setups
                "position_size_multiplier": down_mult,
                "trailing_stop_enabled": True,
                "profit_target_multiplier": 0.7,
                "atr_stop_percent": atr_stop,
                "trailing_atr_mult": 1.5
            }

        elif regime == MarketRegime.HIGH_VOLATILITY:
            # High volatility: very cautious, small positions, wide stops
            # PHASE 3: 4/5 confluence (80%) - extra caution
            trailing_mult = getattr(config, 'TRAIL_ATR_HIGH_VOL', 3.0)
            return {
                **base_params,
                "buy_threshold": 0.45,
                "sell_threshold": -0.35,
                "confluence_required": 4,        # PHASE 3: 4/5 (80%) in volatility
                "stop_loss_multiplier": 1.5,     # Wider stops
                "profit_target_multiplier": 1.3, # Bigger targets (more movement)
                "position_size_multiplier": 0.4, # Much smaller positions!
                "trailing_stop_enabled": True,
                "trailing_stop_distance": 2.5,   # Wider trailing stop
                "atr_stop_percent": atr_stop,    # PHASE 3: ATR-based (wider)
                "trailing_atr_mult": trailing_mult  # PHASE 3: 3x ATR in volatility
            }

        else:  # RANGING or UNKNOWN
            # Ranging: mean reversion, tighter targets, no trailing
            # PHASE 3: 4/5 confluence (80%) - be selective in ranges
            return {
                **base_params,
                "buy_threshold": 0.35,
                "sell_threshold": -0.35,
                "confluence_required": 4,        # PHASE 3: 4/5 (80%) in ranging
                "profit_target_multiplier": 0.8, # Take profits quicker
                "trailing_stop_enabled": False,  # No trailing in ranges
                "position_size_multiplier": 0.9,
                "atr_stop_percent": atr_stop,
                "trailing_atr_mult": 2.0
            }

    def _calculate_atr_stop(self, regime: MarketRegime, atr_data: Dict) -> float:
        """
        PHASE 3: Calculate dynamic stop loss based on ATR

        Stop = ATR_MULTIPLIER * ATR_PERCENT, capped between MIN and MAX

        Regime adjustments:
        - Trending Down: tighter (0.8x)
        - High Volatility: wider (1.5x)
        """
        atr_percent = atr_data.get("percent", 2.0)  # ATR as % of price

        # Get config values
        base_mult = getattr(config, 'ATR_STOP_MULTIPLIER', 2.0)
        min_stop = getattr(config, 'ATR_STOP_MIN', 0.03)  # 3%
        max_stop = getattr(config, 'ATR_STOP_MAX', 0.07)  # 7%

        # Regime multipliers
        if regime == MarketRegime.TRENDING_DOWN:
            regime_mult = getattr(config, 'ATR_TRENDING_DOWN_MULT', 0.8)  # Tighter
        elif regime == MarketRegime.HIGH_VOLATILITY:
            regime_mult = getattr(config, 'ATR_HIGH_VOLATILITY_MULT', 1.5)  # Wider
        else:
            regime_mult = 1.0

        # Calculate stop: ATR% * multiplier * regime adjustment
        stop_percent = (atr_percent / 100) * base_mult * regime_mult

        # Clamp between min and max
        stop_percent = max(min_stop, min(max_stop, stop_percent))

        return round(stop_percent, 4)

    def _get_trailing_mult(self, atr_data: Dict) -> float:
        """
        PHASE 3: Get trailing stop ATR multiplier based on trend strength

        Stronger trends get wider trailing stops to capture more profit:
        - ADX > 40 (strong): 2.5x ATR
        - ADX > 30 (moderate): 2.0x ATR
        - ADX < 30 (weak): 1.5x ATR
        """
        adx_value = atr_data.get("value", 25) if "value" not in atr_data else 25

        # Get from config
        adx_strong = getattr(config, 'TRAIL_ADX_STRONG', 40)
        adx_moderate = getattr(config, 'TRAIL_ADX_MODERATE', 30)
        mult_strong = getattr(config, 'TRAIL_ATR_STRONG', 2.5)
        mult_moderate = getattr(config, 'TRAIL_ATR_MODERATE', 2.0)

        if adx_value >= adx_strong:
            return mult_strong
        elif adx_value >= adx_moderate:
            return mult_moderate
        else:
            return 1.5

    def _get_regime_description(self, regime: MarketRegime) -> str:
        """Human-readable regime description for dashboard"""
        descriptions = {
            MarketRegime.TRENDING_UP: "Bullish Trend - Following momentum upward",
            MarketRegime.TRENDING_DOWN: "Bearish Trend - Being cautious, quick exits",
            MarketRegime.RANGING: "Sideways Market - Mean reversion strategy",
            MarketRegime.HIGH_VOLATILITY: "High Volatility - Reduced exposure, careful",
            MarketRegime.UNKNOWN: "Analyzing market conditions..."
        }
        return descriptions.get(regime, "Unknown market condition")

    def _get_regime_recommendation(self, regime: MarketRegime) -> str:
        """Trading recommendation based on regime"""
        recommendations = {
            MarketRegime.TRENDING_UP: "Trade with trend, use trailing stops",
            MarketRegime.TRENDING_DOWN: "Be selective, take quick profits",
            MarketRegime.RANGING: "Buy support, sell resistance",
            MarketRegime.HIGH_VOLATILITY: "Reduce position size, wider stops",
            MarketRegime.UNKNOWN: "Wait for clear signals"
        }
        return recommendations.get(regime, "Proceed with caution")


# Test the module
if __name__ == "__main__":
    from binance.client import Client
    import config

    print("Testing Market Regime Detection...")

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
        # Create detector
        detector = RegimeDetector()

        # Detect regime
        result = detector.detect_regime(df)

        print(f"\n{'='*60}")
        print(f"MARKET REGIME ANALYSIS")
        print(f"{'='*60}")

        print(f"\n--- Regime ---")
        print(f"  Current: {result['regime_name'].upper()}")
        print(f"  Description: {result['description']}")
        print(f"  Recommendation: {result['recommendation']}")

        print(f"\n--- ADX (Trend Strength) ---")
        adx = result['adx']
        print(f"  ADX Value: {adx['value']}")
        print(f"  Trend Direction: {adx['trend_direction']}")
        print(f"  Trend Strength: {adx['trend_strength']}")
        print(f"  Is Trending: {adx['is_trending']}")

        print(f"\n--- ATR (Volatility) ---")
        atr = result['atr']
        print(f"  ATR Value: ${atr['value']:,.2f}")
        print(f"  ATR %: {atr['percent']:.2f}%")
        print(f"  Volatility Ratio: {atr['volatility_ratio']}x")
        print(f"  Volatility Level: {atr['volatility_level']}")

        print(f"\n--- Auto-Adjusted Parameters ---")
        params = result['adjusted_params']
        print(f"  Buy Threshold: {params['buy_threshold']}")
        print(f"  Sell Threshold: {params['sell_threshold']}")
        print(f"  Confluence Required: {params['confluence_required']}/5")
        print(f"  Position Size: {params['position_size_multiplier']*100:.0f}%")
        print(f"  Trailing Stop: {'Enabled' if params['trailing_stop_enabled'] else 'Disabled'}")

        print(f"\n{'='*60}")
