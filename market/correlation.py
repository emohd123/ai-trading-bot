"""
Correlation Analysis Module - PHASE 4 NEW!

Analyzes broader market context and correlations:
- BTC Dominance (from CoinGecko API)
- Total Market Cap trends
- ETH/BTC ratio for alt season detection
- Stablecoin market cap (USDT/USDC flow)

This provides macro context for trading decisions:
- Rising BTC dominance = altcoins weak, focus on BTC
- Falling dominance = altcoin season
- Stablecoin inflow = potential buying pressure
"""
import requests
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta


class CorrelationAnalyzer:
    """
    Analyzes market-wide correlations and macro trends.
    
    Key metrics:
    - BTC Dominance: BTC market cap as % of total crypto
    - Market Cap Trend: Overall crypto market direction
    - Alt Season Index: Whether altcoins outperforming BTC
    """
    
    def __init__(self):
        """Initialize correlation analyzer."""
        # CoinGecko API (free, no key needed, rate limited)
        self.coingecko_global_url = "https://api.coingecko.com/api/v3/global"
        self.coingecko_simple_url = "https://api.coingecko.com/api/v3/simple/price"
        
        # Thresholds
        self.btc_dominance_high = 55      # BTC strong, altcoins weak
        self.btc_dominance_low = 45       # Potential alt season
        self.dominance_change_significant = 1.0  # 1% change is significant
        
        # Cache (CoinGecko has rate limits)
        self._global_cache = None
        self._global_cache_time = 0
        self._prices_cache = None
        self._prices_cache_time = 0
        self._cache_duration = 60  # 1 minute cache (CoinGecko rate limit friendly)
        
        # Historical data for trend detection
        self._dominance_history: List[Dict] = []
        self._max_history = 24  # Keep 24 data points
        
    def get_global_data(self) -> Dict:
        """
        Fetch global crypto market data from CoinGecko.
        
        Returns:
            Global market data including dominance and market caps
        """
        now = time.time()
        
        # Check cache
        if self._global_cache and (now - self._global_cache_time) < self._cache_duration:
            return self._global_cache
        
        try:
            response = requests.get(self.coingecko_global_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data:
                global_data = data["data"]
                
                result = {
                    "total_market_cap": global_data.get("total_market_cap", {}).get("usd", 0),
                    "total_volume_24h": global_data.get("total_volume", {}).get("usd", 0),
                    "btc_dominance": global_data.get("market_cap_percentage", {}).get("btc", 0),
                    "eth_dominance": global_data.get("market_cap_percentage", {}).get("eth", 0),
                    "market_cap_change_24h": global_data.get("market_cap_change_percentage_24h_usd", 0),
                    "active_cryptocurrencies": global_data.get("active_cryptocurrencies", 0),
                    "updated_at": global_data.get("updated_at", 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in history
                self._add_to_history(result)
                
                # Update cache
                self._global_cache = result
                self._global_cache_time = now
                
                return result
                
        except Exception as e:
            return {
                "error": str(e),
                "btc_dominance": 50,
                "eth_dominance": 15,
                "total_market_cap": 0,
                "market_cap_change_24h": 0
            }
        
        return {"error": "Unknown error", "btc_dominance": 50}
    
    def _add_to_history(self, data: Dict):
        """Add data point to dominance history."""
        self._dominance_history.append({
            "btc_dominance": data.get("btc_dominance", 50),
            "eth_dominance": data.get("eth_dominance", 15),
            "market_cap": data.get("total_market_cap", 0),
            "timestamp": datetime.now()
        })
        
        # Keep only last N entries
        if len(self._dominance_history) > self._max_history:
            self._dominance_history = self._dominance_history[-self._max_history:]
    
    def get_eth_btc_ratio(self) -> Dict:
        """
        Get ETH/BTC ratio for alt season detection.
        
        Rising ETH/BTC = altcoins gaining strength
        Falling ETH/BTC = BTC outperforming
        
        Returns:
            ETH/BTC ratio and trend
        """
        now = time.time()
        
        # Check cache
        if self._prices_cache and (now - self._prices_cache_time) < self._cache_duration:
            return self._prices_cache
        
        try:
            params = {
                "ids": "bitcoin,ethereum",
                "vs_currencies": "usd"
            }
            response = requests.get(self.coingecko_simple_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            btc_price = data.get("bitcoin", {}).get("usd", 0)
            eth_price = data.get("ethereum", {}).get("usd", 0)
            
            if btc_price > 0:
                eth_btc_ratio = eth_price / btc_price
            else:
                eth_btc_ratio = 0
            
            result = {
                "btc_price": btc_price,
                "eth_price": eth_price,
                "eth_btc_ratio": round(eth_btc_ratio, 6),
                "eth_btc_pct": round(eth_btc_ratio * 100, 4)
            }
            
            # Update cache
            self._prices_cache = result
            self._prices_cache_time = now
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "eth_btc_ratio": 0.05,
                "btc_price": 0,
                "eth_price": 0
            }
    
    def analyze_dominance_trend(self) -> Dict:
        """
        Analyze BTC dominance trend over time.
        
        Returns:
            Dominance trend analysis
        """
        global_data = self.get_global_data()
        current_dominance = global_data.get("btc_dominance", 50)
        
        # Calculate trend if we have history
        trend = "unknown"
        change_1h = 0
        
        if len(self._dominance_history) >= 2:
            oldest = self._dominance_history[0]["btc_dominance"]
            latest = self._dominance_history[-1]["btc_dominance"]
            change_1h = latest - oldest
            
            if change_1h > self.dominance_change_significant:
                trend = "rising"
            elif change_1h < -self.dominance_change_significant:
                trend = "falling"
            else:
                trend = "stable"
        
        # Determine market condition
        if current_dominance >= self.btc_dominance_high:
            condition = "btc_season"
            description = "BTC dominance high - altcoins weak"
            alt_signal = "avoid_alts"
        elif current_dominance <= self.btc_dominance_low:
            condition = "alt_season"
            description = "BTC dominance low - altcoins strong"
            alt_signal = "favor_alts"
        else:
            condition = "balanced"
            description = "Mixed market conditions"
            alt_signal = "neutral"
        
        return {
            "current_dominance": round(current_dominance, 2),
            "trend": trend,
            "change_1h": round(change_1h, 2),
            "condition": condition,
            "description": description,
            "alt_signal": alt_signal,
            "history_length": len(self._dominance_history)
        }
    
    def analyze(self) -> Dict:
        """
        Perform comprehensive correlation analysis.
        
        Returns:
            Full market correlation analysis
        """
        global_data = self.get_global_data()
        eth_btc = self.get_eth_btc_ratio()
        dominance_trend = self.analyze_dominance_trend()
        
        # Calculate overall market score
        score = 0.0
        
        # Market cap change contribution
        market_change = global_data.get("market_cap_change_24h", 0)
        if market_change > 3:
            score += 0.3  # Strong bull market
        elif market_change > 0:
            score += 0.1  # Mild bullish
        elif market_change < -3:
            score -= 0.3  # Strong bear market
        elif market_change < 0:
            score -= 0.1  # Mild bearish
        
        # BTC dominance contribution (for BTC trading)
        btc_dom = global_data.get("btc_dominance", 50)
        if btc_dom >= self.btc_dominance_high:
            score += 0.2  # Good for BTC
        elif btc_dom <= self.btc_dominance_low:
            score -= 0.1  # Money flowing to alts
        
        # Dominance trend contribution
        if dominance_trend["trend"] == "rising":
            score += 0.1  # BTC gaining strength
        elif dominance_trend["trend"] == "falling":
            score -= 0.1  # BTC losing to alts
        
        # Clamp score
        score = max(-1.0, min(1.0, score))
        
        # Generate signal
        if score >= 0.3:
            signal = "bullish"
            action = "Market conditions favorable for BTC"
        elif score <= -0.3:
            signal = "bearish"
            action = "Market conditions unfavorable - be cautious"
        else:
            signal = "neutral"
            action = "Mixed market conditions"
        
        return {
            "global": global_data,
            "eth_btc": eth_btc,
            "dominance_trend": dominance_trend,
            "market_cap_usd": global_data.get("total_market_cap", 0),
            "market_cap_change_24h": round(market_change, 2),
            "btc_dominance": round(btc_dom, 2),
            "eth_dominance": round(global_data.get("eth_dominance", 0), 2),
            "score": round(score, 3),
            "signal": signal,
            "action": action
        }
    
    def get_summary(self) -> str:
        """
        Get human-readable correlation summary.
        """
        analysis = self.analyze()
        global_data = analysis["global"]
        dom_trend = analysis["dominance_trend"]
        eth_btc = analysis["eth_btc"]
        
        lines = []
        lines.append("Market Correlation Analysis")
        lines.append("-" * 40)
        
        # Format market cap
        total_mc = global_data.get("total_market_cap", 0)
        if total_mc > 1e12:
            mc_str = f"${total_mc/1e12:.2f}T"
        elif total_mc > 1e9:
            mc_str = f"${total_mc/1e9:.2f}B"
        else:
            mc_str = f"${total_mc:,.0f}"
        
        lines.append(f"Total Market Cap: {mc_str}")
        lines.append(f"24h Change: {analysis['market_cap_change_24h']:+.2f}%")
        
        lines.append(f"\nBTC Dominance: {analysis['btc_dominance']:.2f}%")
        lines.append(f"  Trend: {dom_trend['trend']} ({dom_trend['change_1h']:+.2f}%)")
        lines.append(f"  Condition: {dom_trend['condition']}")
        lines.append(f"  {dom_trend['description']}")
        
        lines.append(f"\nETH Dominance: {analysis['eth_dominance']:.2f}%")
        lines.append(f"ETH/BTC Ratio: {eth_btc.get('eth_btc_pct', 0):.4f}%")
        
        lines.append(f"\nOverall Signal: {analysis['signal']} (score: {analysis['score']:.3f})")
        lines.append(f"Action: {analysis['action']}")
        
        return "\n".join(lines)


# Test the analyzer
if __name__ == "__main__":
    print("Testing Correlation Analyzer...")
    
    analyzer = CorrelationAnalyzer()
    
    print("\n" + "="*60)
    print(analyzer.get_summary())
    print("="*60)
    
    print("\nFull Analysis Keys:")
    analysis = analyzer.analyze()
    for key in analysis.keys():
        if key not in ["global", "eth_btc", "dominance_trend"]:
            print(f"  {key}: {analysis[key]}")
