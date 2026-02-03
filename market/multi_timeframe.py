"""
Multi-Timeframe Analysis Module
Fetches 4h, 1h, and 15m data for higher-TF trend confirmation and lower-TF entry refinement.

Architecture:
- 4h: Regime and trend (smoother, for alignment check)
- 1h: Primary signal analysis
- 15m: Entry refinement
"""
from typing import Dict, Optional
import pandas as pd
import config

from ai.analyzer import TechnicalAnalyzer
from market.market_regime import RegimeDetector


def get_mtf_analysis(client) -> Dict:
    """
    Fetch and analyze multi-timeframe data.

    Fetches:
    - 4h: 50 bars - regime and trend (for alignment)
    - 1h: 100 bars - primary signal analysis
    - 15m: 96 bars - entry refinement

    Args:
        client: BinanceClient instance (or object with get_historical_klines)

    Returns:
        {
            "4h": { "regime", "regime_name", "trend", "ema_trend", "adx", "atr" },
            "1h": full analysis dict from TechnicalAnalyzer,
            "15m": full analysis dict from TechnicalAnalyzer
        }
    """
    analyzer = TechnicalAnalyzer()
    regime_detector = RegimeDetector()

    # Fetch 4h data (50 bars)
    interval_4h = getattr(config, 'MTF_4H_INTERVAL', '4h')
    df_4h = client.get_historical_klines(interval=interval_4h, limit=50)
    if df_4h.empty or len(df_4h) < 20:
        return {"error": "Insufficient 4h data", "4h": None, "1h": None, "15m": None}

    # 4h regime and trend
    regime_data = regime_detector.detect_regime(df_4h)
    adx = regime_data.get("adx", {})
    trend = adx.get("trend_direction", "unknown")  # "up" or "down"
    trend_bullish = trend == "up"

    # 4h EMA trend
    analysis_4h = analyzer.analyze(df_4h)
    ema_4h = analysis_4h.get("ema", {})
    ema_trend = ema_4h.get("trend", "neutral")
    ema_score = ema_4h.get("score", 0)
    ema_bullish = ema_score > 0.2

    # Combined 4h trend: bullish if ADX says up OR EMA bullish
    mtf_4h_trend = "bullish" if (trend_bullish or ema_bullish) else "bearish" if (not trend_bullish and ema_score < -0.2) else "neutral"

    result_4h = {
        "regime": regime_data.get("regime"),
        "regime_name": regime_data.get("regime_name", "unknown"),
        "trend": mtf_4h_trend,
        "ema_trend": ema_trend,
        "adx": adx,
        "atr": regime_data.get("atr", {}),
    }

    # Fetch 1h data (100 bars)
    df_1h = client.get_historical_klines(interval=config.CANDLE_INTERVAL, limit=100)
    if df_1h.empty or len(df_1h) < 20:
        return {"error": "Insufficient 1h data", "4h": result_4h, "1h": None, "15m": None}

    analysis_1h = analyzer.analyze(df_1h)

    # Fetch 15m data (96 bars)
    interval_15m = getattr(config, 'MTF_15M_INTERVAL', '15m')
    df_15m = client.get_historical_klines(interval=interval_15m, limit=96)
    if df_15m.empty or len(df_15m) < 20:
        return {"error": "Insufficient 15m data", "4h": result_4h, "1h": analysis_1h, "15m": None}

    analysis_15m = analyzer.analyze(df_15m)

    return {
        "4h": result_4h,
        "1h": analysis_1h,
        "15m": analysis_15m,
        "df_1h": df_1h,  # For regime detection in get_decision
    }
