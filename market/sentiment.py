"""
Sentiment Analysis Module - PHASE 4 NEW!

Fetches and analyzes market sentiment data:
- Fear & Greed Index (from Alternative.me API)
- Binance Funding Rate (from Binance Futures API)

Sentiment is a contrarian indicator:
- Extreme fear = potential buy opportunity
- Extreme greed = potential sell signal
- Funding rate extremes = crowded trade reversal risk
"""
import requests
import time
from typing import Dict, Optional, Tuple
from datetime import datetime


class SentimentAnalyzer:
    """
    Analyzes market sentiment from multiple sources.
    
    Key metrics:
    - Fear & Greed Index: 0-100 scale (0 = extreme fear, 100 = extreme greed)
    - Funding Rate: Positive = longs pay shorts, Negative = shorts pay longs
    """
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Initialize sentiment analyzer.
        
        Args:
            symbol: Trading pair symbol for funding rate
        """
        self.symbol = symbol
        
        # API endpoints
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.funding_rate_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        
        # Thresholds
        self.extreme_fear = 25          # Buy signal zone
        self.fear = 40                  # Moderate fear
        self.greed = 60                 # Moderate greed
        self.extreme_greed = 75         # Sell signal zone
        
        self.funding_extreme_positive = 0.001   # 0.1% = very bullish sentiment
        self.funding_extreme_negative = -0.001  # -0.1% = very bearish sentiment
        
        # Cache
        self._fear_greed_cache = None
        self._fear_greed_cache_time = 0
        self._funding_cache = None
        self._funding_cache_time = 0
        self._cache_duration = 300  # 5 minutes cache
        
    def get_fear_greed_index(self) -> Dict:
        """
        Fetch Fear & Greed Index from Alternative.me.
        
        This is a free API that provides daily sentiment data.
        
        Returns:
            Fear & Greed data with value, classification, and timestamp
        """
        now = time.time()
        
        # Check cache
        if self._fear_greed_cache and (now - self._fear_greed_cache_time) < self._cache_duration:
            return self._fear_greed_cache
        
        try:
            response = requests.get(self.fear_greed_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and len(data["data"]) > 0:
                fng_data = data["data"][0]
                value = int(fng_data.get("value", 50))
                classification = fng_data.get("value_classification", "Neutral")
                timestamp = fng_data.get("timestamp", "")
                
                # Calculate score (-1 to +1, contrarian)
                # Low fear = buy signal (positive score)
                # High greed = sell signal (negative score)
                if value <= self.extreme_fear:
                    score = 0.8  # Strong buy signal
                    signal = "extreme_fear_buy"
                elif value <= self.fear:
                    score = 0.4  # Moderate buy signal
                    signal = "fear"
                elif value >= self.extreme_greed:
                    score = -0.8  # Strong sell signal
                    signal = "extreme_greed_sell"
                elif value >= self.greed:
                    score = -0.4  # Moderate sell signal
                    signal = "greed"
                else:
                    score = 0.0
                    signal = "neutral"
                
                result = {
                    "value": value,
                    "classification": classification,
                    "signal": signal,
                    "score": round(score, 3),
                    "timestamp": timestamp,
                    "extreme_fear": value <= self.extreme_fear,
                    "extreme_greed": value >= self.extreme_greed
                }
                
                # Update cache
                self._fear_greed_cache = result
                self._fear_greed_cache_time = now
                
                return result
                
        except Exception as e:
            return {
                "value": 50,
                "classification": "Unavailable",
                "signal": "unavailable",
                "score": 0,
                "error": str(e)
            }
        
        return {
            "value": 50,
            "classification": "Unknown",
            "signal": "neutral",
            "score": 0
        }
    
    def get_funding_rate(self) -> Dict:
        """
        Fetch Binance Futures funding rate.
        
        Funding rate is paid between longs and shorts every 8 hours.
        - Positive = longs pay shorts (market is bullish/overleveraged long)
        - Negative = shorts pay longs (market is bearish/overleveraged short)
        
        Extreme funding rates often precede reversals (crowded trade).
        
        Returns:
            Funding rate data with value and signal
        """
        now = time.time()
        
        # Check cache
        if self._funding_cache and (now - self._funding_cache_time) < self._cache_duration:
            return self._funding_cache
        
        try:
            params = {"symbol": self.symbol, "limit": 1}
            response = requests.get(self.funding_rate_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                funding_data = data[0]
                rate = float(funding_data.get("fundingRate", 0))
                funding_time = funding_data.get("fundingTime", 0)
                
                # Convert to datetime
                if funding_time:
                    funding_datetime = datetime.fromtimestamp(funding_time / 1000)
                    time_str = funding_datetime.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_str = ""
                
                # Calculate score (contrarian)
                # High positive funding = crowded long, potential short (negative score)
                # High negative funding = crowded short, potential long (positive score)
                if rate >= self.funding_extreme_positive:
                    score = -0.5  # Crowded long, bearish signal
                    signal = "extreme_positive"
                    interpretation = "Longs overleveraged - potential reversal down"
                elif rate >= 0.0005:
                    score = -0.2
                    signal = "positive"
                    interpretation = "Moderate bullish sentiment"
                elif rate <= self.funding_extreme_negative:
                    score = 0.5  # Crowded short, bullish signal
                    signal = "extreme_negative"
                    interpretation = "Shorts overleveraged - potential reversal up"
                elif rate <= -0.0005:
                    score = 0.2
                    signal = "negative"
                    interpretation = "Moderate bearish sentiment"
                else:
                    score = 0.0
                    signal = "neutral"
                    interpretation = "Balanced funding"
                
                # Convert to percentage for display
                rate_pct = rate * 100
                
                result = {
                    "rate": rate,
                    "rate_pct": round(rate_pct, 4),
                    "signal": signal,
                    "score": round(score, 3),
                    "interpretation": interpretation,
                    "funding_time": time_str,
                    "extreme_positive": rate >= self.funding_extreme_positive,
                    "extreme_negative": rate <= self.funding_extreme_negative
                }
                
                # Update cache
                self._funding_cache = result
                self._funding_cache_time = now
                
                return result
                
        except Exception as e:
            return {
                "rate": 0,
                "rate_pct": 0,
                "signal": "unavailable",
                "score": 0,
                "error": str(e)
            }
        
        return {
            "rate": 0,
            "rate_pct": 0,
            "signal": "neutral",
            "score": 0
        }
    
    def analyze(self) -> Dict:
        """
        Perform comprehensive sentiment analysis.
        
        Combines Fear & Greed Index and Funding Rate into overall sentiment.
        
        Returns:
            Combined sentiment analysis
        """
        fear_greed = self.get_fear_greed_index()
        funding = self.get_funding_rate()
        
        # Combine scores (weighted average)
        # Fear & Greed is longer-term sentiment (60% weight)
        # Funding rate is shorter-term sentiment (40% weight)
        fg_score = fear_greed.get("score", 0)
        fr_score = funding.get("score", 0)
        
        combined_score = (fg_score * 0.6) + (fr_score * 0.4)
        
        # Generate combined signal
        if combined_score >= 0.5:
            combined_signal = "strong_buy"
            action = "Market sentiment is fearful - potential buying opportunity"
        elif combined_score >= 0.2:
            combined_signal = "buy"
            action = "Sentiment leans fearful - consider buying"
        elif combined_score <= -0.5:
            combined_signal = "strong_sell"
            action = "Market sentiment is greedy - consider taking profits"
        elif combined_score <= -0.2:
            combined_signal = "sell"
            action = "Sentiment leans greedy - be cautious"
        else:
            combined_signal = "neutral"
            action = "Sentiment is balanced - no strong signal"
        
        # Risk assessment
        if fear_greed.get("extreme_fear") or funding.get("extreme_negative"):
            risk_level = "low"  # Good time to buy
        elif fear_greed.get("extreme_greed") or funding.get("extreme_positive"):
            risk_level = "high"  # Risky to buy
        else:
            risk_level = "moderate"
        
        return {
            "fear_greed": fear_greed,
            "funding_rate": funding,
            "combined_score": round(combined_score, 3),
            "combined_signal": combined_signal,
            "action": action,
            "risk_level": risk_level,
            "should_buy": combined_score > 0.2,
            "should_avoid_buy": combined_score < -0.3
        }
    
    def get_summary(self) -> str:
        """
        Get human-readable sentiment summary.
        """
        analysis = self.analyze()
        fg = analysis["fear_greed"]
        fr = analysis["funding_rate"]
        
        lines = []
        lines.append("Market Sentiment Analysis")
        lines.append("-" * 40)
        lines.append(f"Fear & Greed Index: {fg.get('value', 'N/A')} ({fg.get('classification', 'N/A')})")
        lines.append(f"  Signal: {fg.get('signal', 'N/A')} (score: {fg.get('score', 0):.3f})")
        
        lines.append(f"\nFunding Rate: {fr.get('rate_pct', 0):.4f}%")
        lines.append(f"  Signal: {fr.get('signal', 'N/A')} (score: {fr.get('score', 0):.3f})")
        lines.append(f"  {fr.get('interpretation', '')}")
        
        lines.append(f"\nCombined Sentiment:")
        lines.append(f"  Score: {analysis['combined_score']:.3f}")
        lines.append(f"  Signal: {analysis['combined_signal']}")
        lines.append(f"  Action: {analysis['action']}")
        lines.append(f"  Risk Level: {analysis['risk_level']}")
        
        return "\n".join(lines)


# Test the analyzer
if __name__ == "__main__":
    print("Testing Sentiment Analyzer...")
    
    analyzer = SentimentAnalyzer()
    
    print("\n" + "="*60)
    print(analyzer.get_summary())
    print("="*60)
    
    print("\nFull Analysis:")
    analysis = analyzer.analyze()
    print(f"  Combined Score: {analysis['combined_score']}")
    print(f"  Should Buy: {analysis['should_buy']}")
    print(f"  Should Avoid: {analysis['should_avoid_buy']}")
