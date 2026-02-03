"""
Technical Analysis Module - PHASE 4 ENHANCED VERSION
Calculates RSI, MACD, Bollinger Bands, EMA, Support/Resistance, Volume Analysis,
Stochastic Oscillator, Bollinger Squeeze Detection, Combined Momentum,
Fibonacci Levels, Candlestick Patterns, Ichimoku Cloud, MFI, Williams %R, and CCI!

Phase 2 additions:
- Stochastic Oscillator: Faster oversold/overbought detection than RSI
- Bollinger Squeeze: Detects when big breakout is imminent

Phase 3 additions:
- Combined Momentum: Merges RSI + Stochastic (removes 90% correlation issue)
- Fibonacci Retracement Levels: 0.382, 0.5, 0.618 for better S/R
- Candlestick Patterns: Engulfing, Hammer, Doji detection

Phase 4 additions:
- Ichimoku Cloud: Complete trend system with cloud support/resistance
- Money Flow Index (MFI): Volume-weighted RSI for money flow analysis
- Williams %R: Fast momentum oscillator for overbought/oversold
- Commodity Channel Index (CCI): Mean deviation for trend strength
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import config


class TechnicalAnalyzer:
    """Performs technical analysis on price data"""

    def __init__(self):
        """Initialize the analyzer with config settings"""
        self.rsi_period = config.RSI_PERIOD
        self.rsi_oversold = config.RSI_OVERSOLD
        self.rsi_overbought = config.RSI_OVERBOUGHT

        self.macd_fast = config.MACD_FAST
        self.macd_slow = config.MACD_SLOW
        self.macd_signal = config.MACD_SIGNAL

        self.bb_period = config.BB_PERIOD
        self.bb_std = config.BB_STD_DEV

        self.ema_short = config.EMA_SHORT
        self.ema_long = config.EMA_LONG

        self.sr_lookback = config.SR_LOOKBACK

        # Stochastic Oscillator settings (optimized for crypto)
        self.stoch_k_period = 14  # %K period
        self.stoch_d_period = 3   # %D smoothing period
        self.stoch_smooth = 3     # %K smoothing
        self.stoch_oversold = 20  # Oversold threshold
        self.stoch_overbought = 80  # Overbought threshold

        # Bollinger Squeeze settings
        self.squeeze_threshold = 0.02  # Bandwidth < 2% = squeeze

        # Phase 4: Ichimoku Cloud settings
        self.ichimoku_tenkan = 9      # Tenkan-sen (Conversion Line)
        self.ichimoku_kijun = 26      # Kijun-sen (Base Line)
        self.ichimoku_senkou_b = 52   # Senkou Span B period

        # Phase 4: Money Flow Index settings
        self.mfi_period = 14
        self.mfi_oversold = 20
        self.mfi_overbought = 80

        # Phase 4: Williams %R settings
        self.williams_period = 14
        self.williams_oversold = -80
        self.williams_overbought = -20

        # Phase 4: CCI settings
        self.cci_period = 20
        self.cci_oversold = -100
        self.cci_overbought = 100

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform full technical analysis on price data

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Dictionary with all indicator values and signals
        """
        if df.empty or len(df) < self.macd_slow:
            return {"error": "Insufficient data for analysis"}

        # Calculate all indicators
        rsi_data = self.calculate_rsi(df)
        macd_data = self.calculate_macd(df)
        bb_data = self.calculate_bollinger_bands(df)
        ema_data = self.calculate_ema(df)
        sr_data = self.calculate_support_resistance(df)
        volume_data = self.calculate_volume_analysis(df)
        stoch_data = self.calculate_stochastic(df)  # Phase 2: Stochastic
        squeeze_data = self.calculate_bollinger_squeeze(df, bb_data)  # Phase 2: Squeeze

        # Phase 3 NEW: Combined Momentum (fixes RSI/Stochastic correlation)
        momentum_data = self.calculate_combined_momentum(rsi_data, stoch_data)

        # Phase 3 NEW: Fibonacci Levels
        fib_data = self.calculate_fibonacci_levels(df)

        # Phase 3 NEW: Candlestick Patterns
        candle_data = self.calculate_candlestick_patterns(df)

        # Phase 4 NEW: Additional indicators
        ichimoku_data = self.calculate_ichimoku(df)
        mfi_data = self.calculate_mfi(df)
        williams_data = self.calculate_williams_r(df)
        cci_data = self.calculate_cci(df)

        current_price = df['close'].iloc[-1]

        return {
            "current_price": current_price,
            "rsi": rsi_data,
            "macd": macd_data,
            "bollinger": bb_data,
            "ema": ema_data,
            "support_resistance": sr_data,
            "volume": volume_data,
            "stochastic": stoch_data,  # Phase 2
            "squeeze": squeeze_data,   # Phase 2
            "momentum": momentum_data, # Phase 3 NEW - Combined RSI+Stochastic
            "fibonacci": fib_data,     # Phase 3 NEW
            "candle_patterns": candle_data,  # Phase 3 NEW
            "ichimoku": ichimoku_data,  # Phase 4 NEW
            "mfi": mfi_data,            # Phase 4 NEW
            "williams_r": williams_data, # Phase 4 NEW
            "cci": cci_data,            # Phase 4 NEW
            "timestamp": df.index[-1] if hasattr(df.index[-1], 'isoformat') else str(df.index[-1])
        }

    def calculate_rsi(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Relative Strength Index

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss

        Args:
            df: DataFrame with 'close' column

        Returns:
            RSI value and signal
        """
        close = df['close']
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # Determine signal
        if current_rsi < self.rsi_oversold:
            signal = "oversold"  # BUY signal
            score = 1.0
        elif current_rsi > self.rsi_overbought:
            signal = "overbought"  # SELL signal
            score = -1.0
        else:
            signal = "neutral"
            # Scale score from -1 to 1 based on RSI position
            # 50 = 0, 30 = 1, 70 = -1
            score = (50 - current_rsi) / 20
            score = max(-1, min(1, score))

        return {
            "value": round(current_rsi, 2),
            "signal": signal,
            "score": round(score, 3),
            "oversold_threshold": self.rsi_oversold,
            "overbought_threshold": self.rsi_overbought
        }

    def calculate_macd(self, df: pd.DataFrame) -> Dict:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line)
        Histogram = MACD Line - Signal Line

        Args:
            df: DataFrame with 'close' column

        Returns:
            MACD values and signal
        """
        close = df['close']

        # Calculate EMAs
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else 0

        # Determine signal
        if current_macd > current_signal and prev_histogram < 0 and current_histogram > 0:
            signal = "bullish_crossover"
            score = 1.0
        elif current_macd < current_signal and prev_histogram > 0 and current_histogram < 0:
            signal = "bearish_crossover"
            score = -1.0
        elif current_macd > current_signal:
            signal = "bullish"
            score = 0.5
        elif current_macd < current_signal:
            signal = "bearish"
            score = -0.5
        else:
            signal = "neutral"
            score = 0.0

        # Adjust score based on histogram momentum
        if current_histogram > prev_histogram:
            score = min(1.0, score + 0.2)
        elif current_histogram < prev_histogram:
            score = max(-1.0, score - 0.2)

        return {
            "macd": round(current_macd, 4),
            "signal_line": round(current_signal, 4),
            "histogram": round(current_histogram, 4),
            "signal": signal,
            "score": round(score, 3)
        }

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Bollinger Bands

        Middle Band = SMA(period)
        Upper Band = Middle Band + (std_dev * std)
        Lower Band = Middle Band - (std_dev * std)

        Args:
            df: DataFrame with 'close' column

        Returns:
            Bollinger Bands values and signal
        """
        close = df['close']

        # Calculate bands
        middle = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()

        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)

        current_price = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_middle = middle.iloc[-1]

        # Calculate %B (position within bands)
        # %B = (Price - Lower) / (Upper - Lower)
        bb_width = current_upper - current_lower
        if bb_width > 0:
            percent_b = (current_price - current_lower) / bb_width
        else:
            percent_b = 0.5

        # Determine signal
        if current_price <= current_lower:
            signal = "below_lower"  # Strong BUY
            score = 1.0
        elif current_price >= current_upper:
            signal = "above_upper"  # Strong SELL
            score = -1.0
        elif percent_b < 0.2:
            signal = "near_lower"  # BUY
            score = 0.7
        elif percent_b > 0.8:
            signal = "near_upper"  # SELL
            score = -0.7
        else:
            signal = "neutral"
            # Scale score based on position (-1 at upper, +1 at lower)
            score = 1 - (2 * percent_b)

        return {
            "upper": round(current_upper, 2),
            "middle": round(current_middle, 2),
            "lower": round(current_lower, 2),
            "percent_b": round(percent_b, 3),
            "signal": signal,
            "score": round(score, 3)
        }

    def calculate_ema(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Exponential Moving Averages

        EMA trend analysis using short and long term EMAs

        Args:
            df: DataFrame with 'close' column

        Returns:
            EMA values and trend signal
        """
        close = df['close']

        ema_short = close.ewm(span=self.ema_short, adjust=False).mean()
        ema_long = close.ewm(span=self.ema_long, adjust=False).mean()

        current_price = close.iloc[-1]
        current_ema_short = ema_short.iloc[-1]
        current_ema_long = ema_long.iloc[-1]
        prev_ema_short = ema_short.iloc[-2] if len(ema_short) > 1 else current_ema_short
        prev_ema_long = ema_long.iloc[-2] if len(ema_long) > 1 else current_ema_long

        # Determine trend and signal
        short_above_long = current_ema_short > current_ema_long
        prev_short_above_long = prev_ema_short > prev_ema_long
        price_above_short = current_price > current_ema_short

        if short_above_long and not prev_short_above_long:
            signal = "bullish_crossover"  # Golden cross
            score = 1.0
        elif not short_above_long and prev_short_above_long:
            signal = "bearish_crossover"  # Death cross
            score = -1.0
        elif short_above_long and price_above_short:
            signal = "strong_uptrend"
            score = 0.7
        elif not short_above_long and not price_above_short:
            signal = "strong_downtrend"
            score = -0.7
        elif short_above_long:
            signal = "uptrend"
            score = 0.4
        elif not short_above_long:
            signal = "downtrend"
            score = -0.4
        else:
            signal = "neutral"
            score = 0.0

        return {
            "ema_short": round(current_ema_short, 2),
            "ema_long": round(current_ema_long, 2),
            "trend": signal,
            "price_vs_ema": "above" if price_above_short else "below",
            "score": round(score, 3)
        }

    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Support and Resistance levels

        Uses pivot points and recent highs/lows

        Args:
            df: DataFrame with OHLC columns

        Returns:
            Support/Resistance levels and signal
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Get recent price action
        lookback = min(self.sr_lookback, len(df))
        recent_high = high.tail(lookback).max()
        recent_low = low.tail(lookback).min()
        current_price = close.iloc[-1]

        # Calculate pivot point (classic)
        last_high = high.iloc[-1]
        last_low = low.iloc[-1]
        last_close = close.iloc[-1]

        pivot = (last_high + last_low + last_close) / 3
        r1 = (2 * pivot) - last_low
        s1 = (2 * pivot) - last_high
        r2 = pivot + (last_high - last_low)
        s2 = pivot - (last_high - last_low)

        # Determine position relative to S/R
        range_size = recent_high - recent_low
        if range_size > 0:
            position = (current_price - recent_low) / range_size
        else:
            position = 0.5

        # Calculate distance to key levels (as percentage)
        dist_to_support = ((current_price - s1) / current_price) * 100
        dist_to_resistance = ((r1 - current_price) / current_price) * 100

        # Determine signal
        if dist_to_support < 1:  # Within 1% of support
            signal = "at_support"
            score = 0.8
        elif dist_to_resistance < 1:  # Within 1% of resistance
            signal = "at_resistance"
            score = -0.8
        elif position < 0.3:
            signal = "near_support"
            score = 0.5
        elif position > 0.7:
            signal = "near_resistance"
            score = -0.5
        else:
            signal = "middle_range"
            score = 0.0

        return {
            "support_1": round(s1, 2),
            "resistance_1": round(r1, 2),
            "support_2": round(s2, 2),
            "resistance_2": round(r2, 2),
            "pivot": round(pivot, 2),
            "recent_high": round(recent_high, 2),
            "recent_low": round(recent_low, 2),
            "position": round(position, 3),
            "signal": signal,
            "score": round(score, 3)
        }

    def calculate_volume_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze trading volume to confirm signals

        Volume confirms price moves:
        - Rising price + rising volume = Strong bullish (confirmed)
        - Rising price + falling volume = Weak rally (caution)
        - Falling price + rising volume = Strong bearish (confirmed)
        - Falling price + falling volume = Weak decline (possible reversal)

        Args:
            df: DataFrame with 'volume' and 'close' columns

        Returns:
            Volume analysis with confirmation signal
        """
        if 'volume' not in df.columns:
            return {
                "signal": "no_data",
                "confirmation": False,
                "score": 0
            }

        volume = df['volume']
        close = df['close']

        # Current and average volume
        current_volume = volume.iloc[-1]
        avg_volume_20 = volume.rolling(window=20).mean().iloc[-1]
        avg_volume_5 = volume.rolling(window=5).mean().iloc[-1]

        # Handle NaN
        if np.isnan(avg_volume_20):
            avg_volume_20 = current_volume
        if np.isnan(avg_volume_5):
            avg_volume_5 = current_volume

        # Volume ratio (current vs average)
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1

        # Volume trend (5-period MA vs 20-period MA)
        volume_trending_up = avg_volume_5 > avg_volume_20

        # Price change over last 5 periods
        price_change_5 = close.iloc[-1] - close.iloc[-5] if len(close) >= 5 else 0
        price_change_percent = (price_change_5 / close.iloc[-5]) * 100 if len(close) >= 5 and close.iloc[-5] > 0 else 0

        # Volume change over last 5 periods
        volume_change = current_volume - volume.iloc[-5] if len(volume) >= 5 else 0

        # Determine signal and confirmation
        if price_change_5 > 0 and volume_change > 0:
            signal = "bullish_confirmed"
            confirmation = True
            score = 0.4
            description = "Price up with volume - strong move"
        elif price_change_5 < 0 and volume_change > 0:
            signal = "bearish_confirmed"
            confirmation = True
            score = -0.4
            description = "Price down with volume - strong move"
        elif price_change_5 > 0 and volume_change <= 0:
            signal = "weak_rally"
            confirmation = False
            score = 0.1
            description = "Price up but low volume - weak"
        elif price_change_5 < 0 and volume_change <= 0:
            signal = "weak_decline"
            confirmation = False
            score = -0.1
            description = "Price down but low volume - possible reversal"
        else:
            signal = "neutral"
            confirmation = False
            score = 0
            description = "No clear volume signal"

        # Unusual volume spike detection
        is_spike = volume_ratio > 2.0
        is_high = volume_ratio > 1.5

        # Volume level description
        if volume_ratio > 2.0:
            volume_level = "very_high"
        elif volume_ratio > 1.5:
            volume_level = "high"
        elif volume_ratio > 0.8:
            volume_level = "normal"
        elif volume_ratio > 0.5:
            volume_level = "low"
        else:
            volume_level = "very_low"

        return {
            "current": round(current_volume, 2),
            "average": round(avg_volume_20, 2),
            "ratio": round(volume_ratio, 2),
            "trending_up": volume_trending_up,
            "signal": signal,
            "confirmation": confirmation,
            "score": round(score, 3),
            "description": description,
            "is_spike": is_spike,
            "is_high": is_high,
            "volume_level": volume_level,
            "price_change_percent": round(price_change_percent, 2)
        }

    def calculate_stochastic(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Stochastic Oscillator - PHASE 2 NEW!

        Stochastic is FASTER than RSI at detecting oversold/overbought.
        Great for catching early reversals!

        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K (smoothing)

        Signals:
        - %K < 20 = Oversold (buy zone)
        - %K > 80 = Overbought (sell zone)
        - %K crosses above %D = Bullish crossover
        - %K crosses below %D = Bearish crossover

        Args:
            df: DataFrame with OHLC columns

        Returns:
            Stochastic values and signals
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate %K (fast stochastic)
        lowest_low = low.rolling(window=self.stoch_k_period).min()
        highest_high = high.rolling(window=self.stoch_k_period).max()

        # Avoid division by zero
        range_hl = highest_high - lowest_low
        range_hl = range_hl.replace(0, np.nan)

        stoch_k_raw = ((close - lowest_low) / range_hl) * 100

        # Smooth %K
        stoch_k = stoch_k_raw.rolling(window=self.stoch_smooth).mean()

        # Calculate %D (signal line - SMA of %K)
        stoch_d = stoch_k.rolling(window=self.stoch_d_period).mean()

        # Get current values
        current_k = stoch_k.iloc[-1] if not stoch_k.empty else 50
        current_d = stoch_d.iloc[-1] if not stoch_d.empty else 50
        prev_k = stoch_k.iloc[-2] if len(stoch_k) > 1 else current_k
        prev_d = stoch_d.iloc[-2] if len(stoch_d) > 1 else current_d

        # Handle NaN
        current_k = 50 if np.isnan(current_k) else current_k
        current_d = 50 if np.isnan(current_d) else current_d
        prev_k = 50 if np.isnan(prev_k) else prev_k
        prev_d = 50 if np.isnan(prev_d) else prev_d

        # Determine signal
        # Check for crossovers
        bullish_crossover = prev_k <= prev_d and current_k > current_d
        bearish_crossover = prev_k >= prev_d and current_k < current_d

        if current_k < self.stoch_oversold:
            if bullish_crossover:
                signal = "oversold_crossover"  # STRONG BUY
                score = 1.0
            else:
                signal = "oversold"  # BUY zone
                score = 0.8
        elif current_k > self.stoch_overbought:
            if bearish_crossover:
                signal = "overbought_crossover"  # STRONG SELL
                score = -1.0
            else:
                signal = "overbought"  # SELL zone
                score = -0.8
        elif bullish_crossover:
            signal = "bullish_crossover"
            score = 0.6
        elif bearish_crossover:
            signal = "bearish_crossover"
            score = -0.6
        elif current_k > current_d:
            signal = "bullish"
            score = 0.3
        elif current_k < current_d:
            signal = "bearish"
            score = -0.3
        else:
            signal = "neutral"
            score = 0.0

        return {
            "k": round(current_k, 2),
            "d": round(current_d, 2),
            "signal": signal,
            "score": round(score, 3),
            "oversold": current_k < self.stoch_oversold,
            "overbought": current_k > self.stoch_overbought,
            "bullish_crossover": bullish_crossover,
            "bearish_crossover": bearish_crossover
        }

    def calculate_combined_momentum(self, rsi_data: Dict, stoch_data: Dict) -> Dict:
        """
        Calculate Combined Momentum - PHASE 3 NEW!

        Combines RSI and Stochastic to create single "Momentum" indicator.
        This fixes the 90% correlation issue where both were double-counting
        the same momentum signal.

        Strategy:
        - Use Stochastic for entry detection (faster, more responsive)
        - Use RSI for exit/confirmation (smoother, less noise)
        - Combined score weights Stochastic more heavily for entries

        Args:
            rsi_data: Pre-calculated RSI data
            stoch_data: Pre-calculated Stochastic data

        Returns:
            Combined momentum signal and score
        """
        rsi_value = rsi_data.get('value', 50)
        rsi_score = rsi_data.get('score', 0)
        rsi_signal = rsi_data.get('signal', 'neutral')

        stoch_k = stoch_data.get('k', 50)
        stoch_score = stoch_data.get('score', 0)
        stoch_signal = stoch_data.get('signal', 'neutral')

        # Weight Stochastic more for entry signals (60%), RSI for confirmation (40%)
        combined_score = (stoch_score * 0.6) + (rsi_score * 0.4)

        # Determine combined signal
        # Strong signals when both agree
        both_oversold = rsi_value < 30 and stoch_k < 20
        both_overbought = rsi_value > 70 and stoch_k > 80

        if both_oversold:
            signal = "strong_oversold"
            combined_score = max(combined_score, 0.9)  # Boost for agreement
        elif both_overbought:
            signal = "strong_overbought"
            combined_score = min(combined_score, -0.9)  # Boost for agreement
        elif stoch_k < 25:  # Stochastic gives early entry signal
            signal = "oversold_entry"
            combined_score = max(combined_score, stoch_score)
        elif stoch_k > 75:  # Stochastic gives early exit signal
            signal = "overbought_exit"
            combined_score = min(combined_score, stoch_score)
        elif rsi_value < 40 and stoch_score > 0:
            signal = "bullish_momentum"
        elif rsi_value > 60 and stoch_score < 0:
            signal = "bearish_momentum"
        elif combined_score > 0.3:
            signal = "bullish"
        elif combined_score < -0.3:
            signal = "bearish"
        else:
            signal = "neutral"

        # Confluence bonus: when RSI and Stoch agree directionally
        rsi_bullish = rsi_score > 0
        stoch_bullish = stoch_score > 0
        confluence = rsi_bullish == stoch_bullish

        if confluence and abs(combined_score) > 0.2:
            combined_score *= 1.15  # 15% boost for confluence
            combined_score = max(-1.0, min(1.0, combined_score))

        return {
            "score": round(combined_score, 3),
            "signal": signal,
            "rsi_value": rsi_value,
            "stoch_k": stoch_k,
            "rsi_signal": rsi_signal,
            "stoch_signal": stoch_signal,
            "confluence": confluence,
            "both_oversold": both_oversold,
            "both_overbought": both_overbought
        }

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Fibonacci Retracement Levels - PHASE 3 NEW!

        Fibonacci levels are powerful support/resistance zones.
        Key levels: 0.236, 0.382, 0.5, 0.618, 0.786

        When price is near a Fib level, it often bounces or breaks through.

        Args:
            df: DataFrame with OHLC columns

        Returns:
            Fibonacci levels and proximity signal
        """
        # Use last 50 candles to find swing high/low
        lookback = min(50, len(df))
        high = df['high'].tail(lookback).max()
        low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]

        # Calculate Fibonacci levels
        diff = high - low

        fib_levels = {
            "0.0": high,                    # 0% retracement (swing high)
            "0.236": high - (diff * 0.236), # 23.6%
            "0.382": high - (diff * 0.382), # 38.2%
            "0.5": high - (diff * 0.5),     # 50%
            "0.618": high - (diff * 0.618), # 61.8% (golden ratio)
            "0.786": high - (diff * 0.786), # 78.6%
            "1.0": low                      # 100% retracement (swing low)
        }

        # Find closest Fib level
        closest_level = None
        closest_distance = float('inf')
        closest_name = None

        for name, level in fib_levels.items():
            distance = abs(current_price - level)
            if distance < closest_distance:
                closest_distance = distance
                closest_level = level
                closest_name = name

        # Distance as percentage
        distance_pct = (closest_distance / current_price) * 100 if current_price > 0 else 0

        # Determine signal based on proximity to Fib level
        # Within 0.5% of a key Fib level = significant
        near_fib = distance_pct < 0.5
        at_fib = distance_pct < 0.2

        # Score based on price position relative to Fib levels
        if current_price <= fib_levels["0.618"]:
            # Below 61.8% = potential support zone (bullish)
            score = 0.5
            signal = "near_fib_support"
        elif current_price >= fib_levels["0.382"]:
            # Above 38.2% = potential resistance zone (bearish)
            score = -0.3
            signal = "near_fib_resistance"
        else:
            score = 0.0
            signal = "mid_fib_range"

        # Boost score if very close to key level
        if at_fib and closest_name in ["0.382", "0.5", "0.618"]:
            score *= 1.5
            signal = f"at_fib_{closest_name}"

        return {
            "levels": {k: round(v, 2) for k, v in fib_levels.items()},
            "closest_level": closest_name,
            "closest_price": round(closest_level, 2) if closest_level else 0,
            "distance_pct": round(distance_pct, 3),
            "near_fib": near_fib,
            "at_fib": at_fib,
            "signal": signal,
            "score": round(score, 3),
            "swing_high": round(high, 2),
            "swing_low": round(low, 2)
        }

    def calculate_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect Candlestick Patterns - PHASE 3 NEW!

        High-probability reversal patterns:
        - Bullish Engulfing: Strong buy signal
        - Bearish Engulfing: Strong sell signal
        - Hammer: Potential bottom reversal
        - Shooting Star: Potential top reversal
        - Doji: Indecision, possible reversal

        Args:
            df: DataFrame with OHLC columns

        Returns:
            Detected patterns and trading signal
        """
        if len(df) < 3:
            return {"patterns": [], "signal": "insufficient_data", "score": 0}

        open_price = df['open'].iloc[-1]
        high_price = df['high'].iloc[-1]
        low_price = df['low'].iloc[-1]
        close_price = df['close'].iloc[-1]

        prev_open = df['open'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]

        patterns = []
        total_score = 0

        # Body and wick calculations
        body = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        total_range = high_price - low_price if high_price > low_price else 0.0001

        prev_body = abs(prev_close - prev_open)
        is_bullish = close_price > open_price
        is_bearish = close_price < open_price
        prev_bullish = prev_close > prev_open
        prev_bearish = prev_close < prev_open

        # ===== BULLISH ENGULFING =====
        # Current bullish candle completely engulfs previous bearish candle
        if (is_bullish and prev_bearish and
            close_price > prev_open and open_price < prev_close and
            body > prev_body * 1.1):  # Current body 10% larger
            patterns.append({
                "name": "bullish_engulfing",
                "type": "bullish",
                "strength": "strong",
                "score": 0.6
            })
            total_score += 0.6

        # ===== BEARISH ENGULFING =====
        # Current bearish candle completely engulfs previous bullish candle
        if (is_bearish and prev_bullish and
            close_price < prev_open and open_price > prev_close and
            body > prev_body * 1.1):
            patterns.append({
                "name": "bearish_engulfing",
                "type": "bearish",
                "strength": "strong",
                "score": -0.6
            })
            total_score -= 0.6

        # ===== HAMMER (Bullish reversal) =====
        # Small body at top, long lower wick (2x+ body), small upper wick
        if (body > 0 and
            lower_wick >= body * 2 and
            upper_wick <= body * 0.5 and
            body / total_range < 0.3):  # Body is less than 30% of range
            patterns.append({
                "name": "hammer",
                "type": "bullish",
                "strength": "moderate",
                "score": 0.5
            })
            total_score += 0.5

        # ===== SHOOTING STAR (Bearish reversal) =====
        # Small body at bottom, long upper wick (2x+ body), small lower wick
        if (body > 0 and
            upper_wick >= body * 2 and
            lower_wick <= body * 0.5 and
            body / total_range < 0.3):
            patterns.append({
                "name": "shooting_star",
                "type": "bearish",
                "strength": "moderate",
                "score": -0.5
            })
            total_score -= 0.5

        # ===== DOJI (Indecision) =====
        # Very small body (open ≈ close), with wicks
        if body / total_range < 0.1 and total_range > 0:  # Body less than 10% of range
            # Dragonfly Doji (bullish) - long lower wick
            if lower_wick > upper_wick * 2:
                patterns.append({
                    "name": "dragonfly_doji",
                    "type": "bullish",
                    "strength": "weak",
                    "score": 0.3
                })
                total_score += 0.3
            # Gravestone Doji (bearish) - long upper wick
            elif upper_wick > lower_wick * 2:
                patterns.append({
                    "name": "gravestone_doji",
                    "type": "bearish",
                    "strength": "weak",
                    "score": -0.3
                })
                total_score -= 0.3
            else:
                patterns.append({
                    "name": "doji",
                    "type": "neutral",
                    "strength": "weak",
                    "score": 0
                })

        # ===== MORNING STAR (3-candle bullish reversal) =====
        if len(df) >= 3:
            candle_3_open = df['open'].iloc[-3]
            candle_3_close = df['close'].iloc[-3]
            candle_3_bearish = candle_3_close < candle_3_open
            candle_2_small = prev_body < abs(candle_3_close - candle_3_open) * 0.3

            if (candle_3_bearish and candle_2_small and is_bullish and
                close_price > (candle_3_open + candle_3_close) / 2):
                patterns.append({
                    "name": "morning_star",
                    "type": "bullish",
                    "strength": "strong",
                    "score": 0.7
                })
                total_score += 0.7

        # Cap score at -1 to 1
        total_score = max(-1.0, min(1.0, total_score))

        # Determine overall signal
        if total_score >= 0.5:
            signal = "strong_bullish_pattern"
        elif total_score >= 0.2:
            signal = "bullish_pattern"
        elif total_score <= -0.5:
            signal = "strong_bearish_pattern"
        elif total_score <= -0.2:
            signal = "bearish_pattern"
        else:
            signal = "no_clear_pattern"

        return {
            "patterns": patterns,
            "pattern_count": len(patterns),
            "signal": signal,
            "score": round(total_score, 3),
            "current_candle": "bullish" if is_bullish else "bearish" if is_bearish else "doji"
        }

    def calculate_bollinger_squeeze(self, df: pd.DataFrame, bb_data: Dict = None) -> Dict:
        """
        Detect Bollinger Band Squeeze - PHASE 2 NEW!

        A squeeze happens when Bollinger Bands get very tight (low volatility).
        This often precedes a BIG price move (breakout)!

        Bandwidth = (Upper - Lower) / Middle
        Squeeze = Bandwidth < threshold (e.g., 2%)

        When squeeze releases, price typically moves 2-5% quickly!

        Args:
            df: DataFrame with 'close' column
            bb_data: Pre-calculated Bollinger Bands data (optional)

        Returns:
            Squeeze detection with breakout readiness
        """
        close = df['close']

        # Calculate Bollinger Bands if not provided
        if bb_data is None:
            middle = close.rolling(window=self.bb_period).mean()
            std = close.rolling(window=self.bb_period).std()
            upper = middle + (std * self.bb_std)
            lower = middle - (std * self.bb_std)
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            current_middle = middle.iloc[-1]
        else:
            current_upper = bb_data.get('upper', 0)
            current_lower = bb_data.get('lower', 0)
            current_middle = bb_data.get('middle', 1)

        # Calculate bandwidth (as percentage)
        if current_middle > 0:
            bandwidth = (current_upper - current_lower) / current_middle
        else:
            bandwidth = 0

        # Historical bandwidth for comparison
        middle = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        upper_series = middle + (std * self.bb_std)
        lower_series = middle - (std * self.bb_std)
        bandwidth_series = (upper_series - lower_series) / middle.replace(0, np.nan)

        # Average bandwidth over last 20 periods
        avg_bandwidth = bandwidth_series.rolling(window=20).mean().iloc[-1]
        if np.isnan(avg_bandwidth):
            avg_bandwidth = bandwidth

        # Bandwidth trend (getting tighter or wider?)
        bandwidth_5_ago = bandwidth_series.iloc[-5] if len(bandwidth_series) >= 5 else bandwidth
        if np.isnan(bandwidth_5_ago):
            bandwidth_5_ago = bandwidth
        bandwidth_trending = "tightening" if bandwidth < bandwidth_5_ago else "expanding"

        # Squeeze detection
        is_squeeze = bandwidth < self.squeeze_threshold
        is_tight = bandwidth < avg_bandwidth * 0.7  # 30% tighter than average

        # Squeeze intensity (how tight?)
        if bandwidth < 0.01:
            squeeze_intensity = "extreme"
        elif bandwidth < 0.015:
            squeeze_intensity = "strong"
        elif bandwidth < 0.02:
            squeeze_intensity = "moderate"
        elif bandwidth < avg_bandwidth * 0.8:
            squeeze_intensity = "mild"
        else:
            squeeze_intensity = "none"

        # Breakout direction hint (based on price position in bands)
        current_price = close.iloc[-1]
        if current_middle > 0:
            position_in_bands = (current_price - current_lower) / (current_upper - current_lower) if (current_upper - current_lower) > 0 else 0.5
        else:
            position_in_bands = 0.5

        if position_in_bands > 0.6:
            breakout_bias = "bullish"
        elif position_in_bands < 0.4:
            breakout_bias = "bearish"
        else:
            breakout_bias = "neutral"

        # Generate alert
        if is_squeeze and bandwidth_trending == "tightening":
            alert = "SQUEEZE ACTIVE - Breakout imminent!"
            readiness = "high"
        elif is_squeeze:
            alert = "Squeeze detected - Watch for breakout"
            readiness = "medium"
        elif is_tight:
            alert = "Volatility contracting"
            readiness = "low"
        else:
            alert = "Normal volatility"
            readiness = "none"

        return {
            "bandwidth": round(bandwidth, 4),
            "avg_bandwidth": round(avg_bandwidth, 4),
            "is_squeeze": is_squeeze,
            "is_tight": is_tight,
            "intensity": squeeze_intensity,
            "trend": bandwidth_trending,
            "breakout_bias": breakout_bias,
            "readiness": readiness,
            "alert": alert,
            "position_in_bands": round(position_in_bands, 3)
        }

    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Ichimoku Cloud - PHASE 4 NEW!

        Ichimoku Kinko Hyo ("one glance equilibrium chart") is a complete
        trading system that shows support/resistance, trend direction, and momentum.

        Components:
        - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
        - Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
        - Chikou Span (Lagging Span): Close plotted 26 periods back

        Signals:
        - Price above cloud = Bullish
        - Price below cloud = Bearish
        - TK Cross (Tenkan crosses Kijun) = Entry signal
        - Cloud color (Span A vs Span B) = Trend strength

        Args:
            df: DataFrame with OHLC columns

        Returns:
            Ichimoku values and signals
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Tenkan-sen (Conversion Line) - 9-period
        tenkan_high = high.rolling(window=self.ichimoku_tenkan).max()
        tenkan_low = low.rolling(window=self.ichimoku_tenkan).min()
        tenkan = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line) - 26-period
        kijun_high = high.rolling(window=self.ichimoku_kijun).max()
        kijun_low = low.rolling(window=self.ichimoku_kijun).min()
        kijun = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A) - shifted forward 26 periods
        senkou_a = ((tenkan + kijun) / 2).shift(self.ichimoku_kijun)

        # Senkou Span B (Leading Span B) - 52-period, shifted forward 26 periods
        senkou_b_high = high.rolling(window=self.ichimoku_senkou_b).max()
        senkou_b_low = low.rolling(window=self.ichimoku_senkou_b).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(self.ichimoku_kijun)

        # Get current values (handle NaN)
        current_price = close.iloc[-1]
        current_tenkan = tenkan.iloc[-1] if not np.isnan(tenkan.iloc[-1]) else current_price
        current_kijun = kijun.iloc[-1] if not np.isnan(kijun.iloc[-1]) else current_price
        
        # For cloud, we need to look at current values (not shifted forward)
        # The cloud at current price is senkou_a and senkou_b from 26 periods ago
        cloud_top = max(senkou_a.iloc[-1] if not np.isnan(senkou_a.iloc[-1]) else current_price,
                       senkou_b.iloc[-1] if not np.isnan(senkou_b.iloc[-1]) else current_price)
        cloud_bottom = min(senkou_a.iloc[-1] if not np.isnan(senkou_a.iloc[-1]) else current_price,
                          senkou_b.iloc[-1] if not np.isnan(senkou_b.iloc[-1]) else current_price)

        # Previous values for crossover detection
        prev_tenkan = tenkan.iloc[-2] if len(tenkan) > 1 and not np.isnan(tenkan.iloc[-2]) else current_tenkan
        prev_kijun = kijun.iloc[-2] if len(kijun) > 1 and not np.isnan(kijun.iloc[-2]) else current_kijun

        # TK Cross detection
        bullish_tk_cross = prev_tenkan <= prev_kijun and current_tenkan > current_kijun
        bearish_tk_cross = prev_tenkan >= prev_kijun and current_tenkan < current_kijun

        # Cloud color (bullish when Span A > Span B)
        span_a_current = senkou_a.iloc[-1] if not np.isnan(senkou_a.iloc[-1]) else current_price
        span_b_current = senkou_b.iloc[-1] if not np.isnan(senkou_b.iloc[-1]) else current_price
        cloud_bullish = span_a_current > span_b_current

        # Price position relative to cloud
        above_cloud = current_price > cloud_top
        below_cloud = current_price < cloud_bottom
        in_cloud = not above_cloud and not below_cloud

        # Calculate score and signal
        score = 0.0

        if above_cloud:
            score += 0.4
            if cloud_bullish:
                score += 0.2  # Extra bullish when cloud is green
        elif below_cloud:
            score -= 0.4
            if not cloud_bullish:
                score -= 0.2  # Extra bearish when cloud is red

        # TK Cross signals
        if bullish_tk_cross:
            score += 0.3
            if above_cloud:
                score += 0.2  # Strong signal when above cloud
        elif bearish_tk_cross:
            score -= 0.3
            if below_cloud:
                score -= 0.2  # Strong signal when below cloud

        # Tenkan vs Kijun (momentum)
        if current_tenkan > current_kijun:
            score += 0.1
        elif current_tenkan < current_kijun:
            score -= 0.1

        # Clamp score
        score = max(-1.0, min(1.0, score))

        # Determine signal
        if bullish_tk_cross and above_cloud:
            signal = "strong_bullish"
        elif bearish_tk_cross and below_cloud:
            signal = "strong_bearish"
        elif bullish_tk_cross:
            signal = "bullish_cross"
        elif bearish_tk_cross:
            signal = "bearish_cross"
        elif above_cloud and cloud_bullish:
            signal = "bullish"
        elif below_cloud and not cloud_bullish:
            signal = "bearish"
        elif above_cloud:
            signal = "above_cloud"
        elif below_cloud:
            signal = "below_cloud"
        elif in_cloud:
            signal = "in_cloud"
        else:
            signal = "neutral"

        return {
            "tenkan": round(current_tenkan, 2),
            "kijun": round(current_kijun, 2),
            "senkou_a": round(span_a_current, 2),
            "senkou_b": round(span_b_current, 2),
            "cloud_top": round(cloud_top, 2),
            "cloud_bottom": round(cloud_bottom, 2),
            "above_cloud": above_cloud,
            "below_cloud": below_cloud,
            "in_cloud": in_cloud,
            "cloud_bullish": cloud_bullish,
            "tk_cross_bullish": bullish_tk_cross,
            "tk_cross_bearish": bearish_tk_cross,
            "signal": signal,
            "score": round(score, 3)
        }

    def calculate_mfi(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Money Flow Index (MFI) - PHASE 4 NEW!

        MFI is a volume-weighted RSI that measures buying/selling pressure.
        It's excellent for confirming trends and spotting divergences.

        Formula:
        1. Typical Price = (High + Low + Close) / 3
        2. Raw Money Flow = Typical Price * Volume
        3. Positive/Negative Money Flow based on TP change
        4. Money Flow Ratio = Positive MF / Negative MF
        5. MFI = 100 - (100 / (1 + Money Flow Ratio))

        Signals:
        - MFI < 20 = Oversold (buy signal)
        - MFI > 80 = Overbought (sell signal)
        - Divergence = Powerful reversal signal

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            MFI value and signal
        """
        if 'volume' not in df.columns:
            return {"value": 50, "signal": "no_volume_data", "score": 0}

        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']

        # Typical Price
        typical_price = (high + low + close) / 3

        # Raw Money Flow
        raw_money_flow = typical_price * volume

        # Positive and Negative Money Flow
        tp_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(tp_diff > 0, 0)
        negative_flow = raw_money_flow.where(tp_diff < 0, 0)

        # Sum over period
        positive_mf = positive_flow.rolling(window=self.mfi_period).sum()
        negative_mf = negative_flow.rolling(window=self.mfi_period).sum()

        # Money Flow Ratio and MFI
        # Avoid division by zero
        mf_ratio = positive_mf / negative_mf.replace(0, 0.0001)
        mfi = 100 - (100 / (1 + mf_ratio))

        current_mfi = mfi.iloc[-1]
        if np.isnan(current_mfi):
            current_mfi = 50

        # Determine signal and score
        if current_mfi < self.mfi_oversold:
            signal = "oversold"
            score = 0.8
        elif current_mfi > self.mfi_overbought:
            signal = "overbought"
            score = -0.8
        elif current_mfi < 40:
            signal = "bullish"
            score = 0.4
        elif current_mfi > 60:
            signal = "bearish"
            score = -0.4
        else:
            signal = "neutral"
            score = (50 - current_mfi) / 50  # Scale -1 to 1

        return {
            "value": round(current_mfi, 2),
            "signal": signal,
            "score": round(score, 3),
            "oversold": current_mfi < self.mfi_oversold,
            "overbought": current_mfi > self.mfi_overbought
        }

    def calculate_williams_r(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Williams %R - PHASE 4 NEW!

        Williams %R is a fast momentum oscillator that shows overbought/oversold levels.
        Similar to Stochastic but inverted scale (-100 to 0).

        Formula:
        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

        Signals:
        - %R > -20 = Overbought (sell signal)
        - %R < -80 = Oversold (buy signal)
        - Faster than RSI - catches reversals early

        Args:
            df: DataFrame with OHLC columns

        Returns:
            Williams %R value and signal
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Highest high and lowest low over period
        highest_high = high.rolling(window=self.williams_period).max()
        lowest_low = low.rolling(window=self.williams_period).min()

        # Williams %R calculation
        range_hl = highest_high - lowest_low
        range_hl = range_hl.replace(0, 0.0001)  # Avoid division by zero

        williams_r = ((highest_high - close) / range_hl) * -100

        current_wr = williams_r.iloc[-1]
        if np.isnan(current_wr):
            current_wr = -50

        prev_wr = williams_r.iloc[-2] if len(williams_r) > 1 else current_wr
        if np.isnan(prev_wr):
            prev_wr = current_wr

        # Detect crossovers
        crossing_up = prev_wr < self.williams_oversold and current_wr >= self.williams_oversold
        crossing_down = prev_wr > self.williams_overbought and current_wr <= self.williams_overbought

        # Determine signal and score
        if current_wr < self.williams_oversold:
            if crossing_up:
                signal = "oversold_reversal"
                score = 1.0
            else:
                signal = "oversold"
                score = 0.7
        elif current_wr > self.williams_overbought:
            if crossing_down:
                signal = "overbought_reversal"
                score = -1.0
            else:
                signal = "overbought"
                score = -0.7
        elif current_wr < -60:
            signal = "bullish"
            score = 0.3
        elif current_wr > -40:
            signal = "bearish"
            score = -0.3
        else:
            signal = "neutral"
            score = (-50 - current_wr) / 50  # Scale -1 to 1

        return {
            "value": round(current_wr, 2),
            "signal": signal,
            "score": round(score, 3),
            "oversold": current_wr < self.williams_oversold,
            "overbought": current_wr > self.williams_overbought,
            "crossing_up": crossing_up,
            "crossing_down": crossing_down
        }

    def calculate_cci(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Commodity Channel Index (CCI) - PHASE 4 NEW!

        CCI measures price deviation from the statistical mean.
        Great for identifying cyclical trends and extreme conditions.

        Formula:
        1. Typical Price = (High + Low + Close) / 3
        2. SMA of TP over period
        3. Mean Deviation = average of absolute differences from SMA
        4. CCI = (TP - SMA) / (0.015 * Mean Deviation)

        Signals:
        - CCI > 100 = Overbought / Strong uptrend
        - CCI < -100 = Oversold / Strong downtrend
        - Zero line crossovers = Trend changes

        Args:
            df: DataFrame with OHLC columns

        Returns:
            CCI value and signal
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Typical Price
        typical_price = (high + low + close) / 3

        # SMA of Typical Price
        sma_tp = typical_price.rolling(window=self.cci_period).mean()

        # Mean Deviation
        mean_deviation = typical_price.rolling(window=self.cci_period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        # CCI calculation (0.015 is the constant factor)
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

        current_cci = cci.iloc[-1]
        if np.isnan(current_cci):
            current_cci = 0

        prev_cci = cci.iloc[-2] if len(cci) > 1 else current_cci
        if np.isnan(prev_cci):
            prev_cci = current_cci

        # Zero line crossovers
        crossing_above_zero = prev_cci < 0 and current_cci >= 0
        crossing_below_zero = prev_cci > 0 and current_cci <= 0

        # Extreme levels
        extreme_overbought = current_cci > 200
        extreme_oversold = current_cci < -200

        # Determine signal and score
        if extreme_oversold:
            signal = "extreme_oversold"
            score = 0.9
        elif extreme_overbought:
            signal = "extreme_overbought"
            score = -0.9
        elif current_cci < self.cci_oversold:
            signal = "oversold"
            score = 0.6
        elif current_cci > self.cci_overbought:
            signal = "overbought"
            score = -0.6
        elif crossing_above_zero:
            signal = "bullish_cross"
            score = 0.5
        elif crossing_below_zero:
            signal = "bearish_cross"
            score = -0.5
        elif current_cci > 0:
            signal = "bullish"
            score = min(0.4, current_cci / 250)
        elif current_cci < 0:
            signal = "bearish"
            score = max(-0.4, current_cci / 250)
        else:
            signal = "neutral"
            score = 0

        return {
            "value": round(current_cci, 2),
            "signal": signal,
            "score": round(score, 3),
            "oversold": current_cci < self.cci_oversold,
            "overbought": current_cci > self.cci_overbought,
            "extreme_oversold": extreme_oversold,
            "extreme_overbought": extreme_overbought,
            "zero_cross_up": crossing_above_zero,
            "zero_cross_down": crossing_below_zero
        }


# Test the analyzer
if __name__ == "__main__":
    from binance.client import Client

    print("Testing Technical Analyzer...")

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
        # Create analyzer
        analyzer = TechnicalAnalyzer()

        # Run analysis
        analysis = analyzer.analyze(df)

        print(f"\n{'='*60}")
        print(f"Technical Analysis for {config.SYMBOL}")
        print(f"{'='*60}")

        print(f"\nCurrent Price: ${analysis['current_price']:,.2f}")

        print(f"\n--- RSI ---")
        rsi = analysis['rsi']
        print(f"  Value: {rsi['value']}")
        print(f"  Signal: {rsi['signal']}")
        print(f"  Score: {rsi['score']}")

        print(f"\n--- MACD ---")
        macd = analysis['macd']
        print(f"  MACD: {macd['macd']}")
        print(f"  Signal Line: {macd['signal_line']}")
        print(f"  Histogram: {macd['histogram']}")
        print(f"  Signal: {macd['signal']}")
        print(f"  Score: {macd['score']}")

        print(f"\n--- Bollinger Bands ---")
        bb = analysis['bollinger']
        print(f"  Upper: ${bb['upper']:,.2f}")
        print(f"  Middle: ${bb['middle']:,.2f}")
        print(f"  Lower: ${bb['lower']:,.2f}")
        print(f"  %B: {bb['percent_b']}")
        print(f"  Signal: {bb['signal']}")
        print(f"  Score: {bb['score']}")

        print(f"\n--- EMA Trend ---")
        ema = analysis['ema']
        print(f"  EMA Short ({config.EMA_SHORT}): ${ema['ema_short']:,.2f}")
        print(f"  EMA Long ({config.EMA_LONG}): ${ema['ema_long']:,.2f}")
        print(f"  Trend: {ema['trend']}")
        print(f"  Score: {ema['score']}")

        print(f"\n--- Support/Resistance ---")
        sr = analysis['support_resistance']
        print(f"  Resistance 1: ${sr['resistance_1']:,.2f}")
        print(f"  Pivot: ${sr['pivot']:,.2f}")
        print(f"  Support 1: ${sr['support_1']:,.2f}")
        print(f"  Signal: {sr['signal']}")
        print(f"  Score: {sr['score']}")

        print(f"\n--- Volume Analysis ---")
        vol = analysis['volume']
        print(f"  Current Volume: {vol['current']:,.0f}")
        print(f"  Average Volume: {vol['average']:,.0f}")
        print(f"  Volume Ratio: {vol['ratio']}x")
        print(f"  Volume Level: {vol['volume_level']}")
        print(f"  Signal: {vol['signal']}")
        print(f"  Confirmation: {'YES' if vol['confirmation'] else 'NO'}")
        print(f"  Description: {vol['description']}")

        print(f"\n--- Stochastic Oscillator (Phase 2 NEW!) ---")
        stoch = analysis['stochastic']
        print(f"  %K: {stoch['k']}")
        print(f"  %D: {stoch['d']}")
        print(f"  Signal: {stoch['signal']}")
        print(f"  Score: {stoch['score']}")
        print(f"  Oversold: {'YES' if stoch['oversold'] else 'NO'}")
        print(f"  Overbought: {'YES' if stoch['overbought'] else 'NO'}")

        print(f"\n--- Bollinger Squeeze (Phase 2 NEW!) ---")
        squeeze = analysis['squeeze']
        print(f"  Bandwidth: {squeeze['bandwidth']*100:.2f}%")
        print(f"  Avg Bandwidth: {squeeze['avg_bandwidth']*100:.2f}%")
        print(f"  Squeeze Active: {'YES!' if squeeze['is_squeeze'] else 'NO'}")
        print(f"  Intensity: {squeeze['intensity']}")
        print(f"  Trend: {squeeze['trend']}")
        print(f"  Breakout Bias: {squeeze['breakout_bias']}")
        print(f"  Alert: {squeeze['alert']}")
