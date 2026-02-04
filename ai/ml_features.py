"""
ML Feature Engineering Module
Creates 40+ engineered features for price prediction from OHLCV data.
PHASE 5: Added market context features (Fear/Greed, funding rate, order book)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import config

# Optional market context imports (market.sentiment, market.order_book)
try:
    from market.sentiment import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    from market.order_book import OrderBookAnalyzer
    ORDER_BOOK_AVAILABLE = True
except ImportError:
    ORDER_BOOK_AVAILABLE = False


def select_features(X: pd.DataFrame, y: np.ndarray, n_features: int = 25) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top N features using mutual information.
    Returns (X with selected features, list of selected feature names).
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
        importance = mutual_info_classif(X.fillna(0), y, random_state=42)
        top_indices = np.argsort(importance)[-n_features:]
        selected_cols = [X.columns[i] for i in top_indices]
        return X[selected_cols], selected_cols
    except Exception:
        return X, list(X.columns)


class MLFeatureEngineer:
    """Feature engineering for ML price prediction"""

    def __init__(self):
        self.rsi_period = config.RSI_PERIOD
        self.macd_fast = config.MACD_FAST
        self.macd_slow = config.MACD_SLOW
        self.bb_period = config.BB_PERIOD
        self.ema_short = config.EMA_SHORT
        self.ema_long = config.EMA_LONG

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from OHLCV DataFrame.
        Returns DataFrame with features for the last row (current state).
        """
        if df.empty or len(df) < 100:
            return pd.DataFrame()

        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        features = {}

        # === PRICE-BASED FEATURES ===
        features['price_change_1h'] = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(df) > 1 else 0
        features['price_change_4h'] = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(df) > 5 else 0
        features['price_change_12h'] = (close.iloc[-1] - close.iloc[-13]) / close.iloc[-13] if len(df) > 13 else 0
        features['price_change_24h'] = (close.iloc[-1] - close.iloc[-25]) / close.iloc[-25] if len(df) > 25 else 0
        features['price_change_7d'] = (close.iloc[-1] - close.iloc[-169]) / close.iloc[-169] if len(df) > 169 else 0

        # Price volatility (rolling std)
        returns = close.pct_change()
        features['volatility_24h'] = returns.tail(24).std() if len(df) > 24 else 0
        features['volatility_7d'] = returns.tail(168).std() if len(df) > 168 else 0

        # Price momentum (rate of change)
        features['momentum_4h'] = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(df) > 5 else 0
        features['momentum_24h'] = (close.iloc[-1] - close.iloc[-25]) / close.iloc[-25] if len(df) > 25 else 0

        # High/Low range
        hl_range = high - low
        features['hl_range_pct'] = (hl_range.iloc[-1] / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 0

        # Distance from 24h high/low
        high_24h = high.tail(24).max() if len(df) >= 24 else high.max()
        low_24h = low.tail(24).min() if len(df) >= 24 else low.min()
        features['dist_from_24h_high'] = (high_24h - close.iloc[-1]) / high_24h if high_24h > 0 else 0
        features['dist_from_24h_low'] = (close.iloc[-1] - low_24h) / low_24h if low_24h > 0 else 0

        # === RSI ===
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features['rsi'] = rsi.iloc[-1] / 100 - 0.5 if not np.isnan(rsi.iloc[-1]) else 0
        features['rsi_change'] = (rsi.iloc[-1] - rsi.iloc[-2]) / 100 if len(df) > 2 and not np.isnan(rsi.iloc[-2]) else 0

        # === MACD ===
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        features['macd_hist'] = macd_hist.iloc[-1] / close.iloc[-1] * 1000 if close.iloc[-1] > 0 else 0
        features['macd_signal'] = (macd_line.iloc[-1] - signal_line.iloc[-1]) / close.iloc[-1] * 1000 if close.iloc[-1] > 0 else 0

        # === BOLLINGER BANDS ===
        bb_mid = close.rolling(window=self.bb_period).mean()
        bb_std = close.rolling(window=self.bb_period).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        percent_b = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        features['bb_percent_b'] = percent_b.iloc[-1] - 0.5 if not np.isnan(percent_b.iloc[-1]) else 0
        features['bb_width'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_mid.iloc[-1] if not np.isnan(bb_mid.iloc[-1]) and bb_mid.iloc[-1] > 0 else 0

        # === EMA ===
        ema_short = close.ewm(span=self.ema_short, adjust=False).mean()
        ema_long = close.ewm(span=self.ema_long, adjust=False).mean()
        features['ema_cross_dist'] = (ema_short.iloc[-1] - ema_long.iloc[-1]) / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0
        features['price_above_ema'] = 1 if close.iloc[-1] > ema_long.iloc[-1] else -1

        # === STOCHASTIC ===
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        stoch_k = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()
        features['stoch_k'] = (stoch_k.iloc[-1] / 100 - 0.5) if not np.isnan(stoch_k.iloc[-1]) else 0
        features['stoch_d'] = (stoch_d.iloc[-1] / 100 - 0.5) if not np.isnan(stoch_d.iloc[-1]) else 0

        # === ADX ===
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(14).mean()
        features['adx'] = adx.iloc[-1] / 50 - 0.5 if not np.isnan(adx.iloc[-1]) else 0
        features['adx_trend_up'] = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else -1

        # === ATR ===
        atr_val = tr.rolling(14).mean().iloc[-1]
        features['atr_percent'] = (atr_val / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 0

        # === VOLUME FEATURES ===
        vol_ma = volume.rolling(20).mean()
        features['volume_ratio'] = (volume.iloc[-1] / vol_ma.iloc[-1] - 1) if vol_ma.iloc[-1] > 0 else 0
        features['volume_change'] = (volume.iloc[-1] - volume.iloc[-2]) / volume.iloc[-2] if len(df) > 2 and volume.iloc[-2] > 0 else 0
        features['volume_trend'] = (volume.iloc[-1] - volume.iloc[-5]) / volume.iloc[-5] if len(df) > 5 and volume.iloc[-5] > 0 else 0

        # Taker buy volume (if available)
        if 'taker_buy_base' in df.columns:
            taker_buy = df['taker_buy_base'].astype(float)
            features['buy_sell_ratio'] = (taker_buy.iloc[-1] / volume.iloc[-1] - 0.5) * 2 if volume.iloc[-1] > 0 else 0
        else:
            features['buy_sell_ratio'] = 0

        # === TIME FEATURES ===
        try:
            ts = df.index[-1]
            hour = getattr(ts, 'hour', 12)
            day = getattr(ts, 'dayofweek', 2)
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * day / 7)
            features['day_cos'] = np.cos(2 * np.pi * day / 7)
            features['is_weekend'] = 1 if day >= 5 else 0
            features['session_asia'] = 1 if 0 <= hour < 8 else 0
            features['session_europe'] = 1 if 8 <= hour < 16 else 0
            features['session_us'] = 1 if 16 <= hour < 24 else 0
        except Exception:
            for k in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'session_asia', 'session_europe', 'session_us']:
                features[k] = 0.0

        # === PHASE 5: MARKET CONTEXT FEATURES ===
        # Fear & Greed Index (0-100, inverted for bullish signal)
        if SENTIMENT_AVAILABLE:
            try:
                sentiment = SentimentAnalyzer()
                sent_data = sentiment.analyze()
                fg = sent_data.get("fear_greed", {})
                features['fear_greed'] = fg.get("value", 50)
                features['fear_greed_normalized'] = (fg.get("value", 50) - 50) / 50  # -1 to 1
                # Funding rate (negative = shorts paying longs = bullish)
                fr = sent_data.get("funding_rate", {})
                features['funding_rate'] = fr.get("rate", 0) * 10000  # Scale up for ML
                features['funding_signal'] = 1 if fr.get("rate", 0) < -0.0001 else (-1 if fr.get("rate", 0) > 0.0003 else 0)
            except Exception:
                features['fear_greed'] = 50
                features['fear_greed_normalized'] = 0
                features['funding_rate'] = 0
                features['funding_signal'] = 0
        else:
            features['fear_greed'] = 50
            features['fear_greed_normalized'] = 0
            features['funding_rate'] = 0
            features['funding_signal'] = 0
        
        # Order Book Imbalance
        if ORDER_BOOK_AVAILABLE:
            try:
                ob = OrderBookAnalyzer()
                ob_data = ob.analyze()
                features['order_book_imbalance'] = ob_data.get("imbalance_ratio", 1.0)
                features['order_book_score'] = ob_data.get("score", 0)
                features['order_book_signal'] = 1 if ob_data.get("signal") in ["buy_pressure", "strong_buy_pressure"] else (-1 if ob_data.get("signal") in ["sell_pressure", "strong_sell_pressure"] else 0)
            except Exception:
                features['order_book_imbalance'] = 1.0
                features['order_book_score'] = 0
                features['order_book_signal'] = 0
        else:
            features['order_book_imbalance'] = 1.0
            features['order_book_score'] = 0
            features['order_book_signal'] = 0

        # === PATTERN FEATURES ===
        # Candlestick body size
        body = abs(close.iloc[-1] - df['open'].iloc[-1])
        features['candle_body_pct'] = (body / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 0
        features['candle_bullish'] = 1 if close.iloc[-1] > df['open'].iloc[-1] else -1

        # Support/Resistance proximity (simplified)
        lookback = min(50, len(df) - 1)
        recent_high = high.iloc[-lookback:-1].max()
        recent_low = low.iloc[-lookback:-1].min()
        features['near_resistance'] = (recent_high - close.iloc[-1]) / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0
        features['near_support'] = (close.iloc[-1] - recent_low) / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0

        # Squeeze (Bollinger width)
        features['bb_squeeze'] = 1 if features.get('bb_width', 1) < 0.02 else 0

        # Fill NaN with 0
        for k, v in features.items():
            if isinstance(v, (np.floating, float)) and (np.isnan(v) or np.isinf(v)):
                features[k] = 0.0

        return pd.DataFrame([features])

    def create_features_sequence(self, df: pd.DataFrame, n_rows: int = 24) -> pd.DataFrame:
        """
        Create features for the last n_rows candles.
        Returns DataFrame with shape (n_rows, n_features) for LSTM sequence input.
        """
        if df.empty or len(df) < 100 + n_rows:
            return pd.DataFrame()

        rows = []
        for i in range(len(df) - n_rows, len(df)):
            df_slice = df.iloc[:i + 1].copy()
            feat = self.create_features(df_slice)
            if not feat.empty:
                rows.append(feat.iloc[0])

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names in order"""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=150, freq='h')
        df_dummy = pd.DataFrame({
            'open': np.random.rand(150) * 50000 + 50000,
            'high': np.random.rand(150) * 50000 + 51000,
            'low': np.random.rand(150) * 50000 + 49000,
            'close': np.random.rand(150) * 50000 + 50000,
            'volume': np.random.rand(150) * 1000
        }, index=dates)
        feats = self.create_features(df_dummy)
        return list(feats.columns) if not feats.empty else []
