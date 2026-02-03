"""
Order Book Analysis Module - PHASE 4 NEW!

Analyzes Binance order book depth to detect:
- Bid/Ask imbalance (buying vs selling pressure)
- Large walls (support/resistance from big orders)
- Order flow momentum

This provides real-time market microstructure data that complements
technical indicators for better entry/exit timing.
"""
import numpy as np
from typing import Dict, Optional, Tuple
from binance.client import Client
import config


class OrderBookAnalyzer:
    """
    Analyzes order book depth for trading signals.
    
    Key metrics:
    - Imbalance ratio: Bid volume / Ask volume
    - Wall detection: Large orders that act as S/R
    - Depth score: Overall buying/selling pressure
    """
    
    def __init__(self, symbol: str = None):
        """
        Initialize order book analyzer.
        
        Args:
            symbol: Trading pair symbol (default from config)
        """
        self.symbol = symbol or config.SYMBOL
        self.client = Client()  # Public API - no keys needed
        
        # Configuration
        self.depth_limit = 20           # Top 20 levels each side
        self.wall_threshold = 2.0       # Order > 2x average = wall
        self.strong_wall_threshold = 3.0  # Order > 3x average = strong wall
        self.imbalance_bullish = 1.5    # Bid > 1.5x Ask = bullish
        self.imbalance_bearish = 0.67   # Bid < 0.67x Ask = bearish
        
        # Cache
        self._last_analysis = None
        self._cache_seconds = 5  # Cache for 5 seconds
        self._last_fetch_time = 0
        
    def get_order_book(self) -> Optional[Dict]:
        """
        Fetch order book from Binance.
        
        Returns:
            Order book data with bids and asks
        """
        try:
            depth = self.client.get_order_book(symbol=self.symbol, limit=self.depth_limit)
            return depth
        except Exception as e:
            return {"error": str(e)}
    
    def analyze(self) -> Dict:
        """
        Perform comprehensive order book analysis.
        
        Returns:
            Analysis with imbalance, walls, and signals
        """
        import time
        
        # Check cache
        now = time.time()
        if self._last_analysis and (now - self._last_fetch_time) < self._cache_seconds:
            return self._last_analysis
        
        order_book = self.get_order_book()
        
        if "error" in order_book:
            return {
                "error": order_book["error"],
                "imbalance_ratio": 1.0,
                "signal": "unavailable",
                "score": 0
            }
        
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        
        if not bids or not asks:
            return {
                "error": "Empty order book",
                "imbalance_ratio": 1.0,
                "signal": "unavailable",
                "score": 0
            }
        
        # Convert to numpy arrays for faster processing
        bid_prices = np.array([float(b[0]) for b in bids])
        bid_volumes = np.array([float(b[1]) for b in bids])
        ask_prices = np.array([float(a[0]) for a in asks])
        ask_volumes = np.array([float(a[1]) for a in asks])
        
        # Calculate metrics
        imbalance = self._calculate_imbalance(bid_volumes, ask_volumes)
        walls = self._detect_walls(bid_prices, bid_volumes, ask_prices, ask_volumes)
        depth_score = self._calculate_depth_score(bid_volumes, ask_volumes)
        spread = self._calculate_spread(bid_prices, ask_prices)
        
        # Generate overall signal
        signal, score = self._generate_signal(imbalance, walls, depth_score)
        
        analysis = {
            "imbalance_ratio": round(imbalance["ratio"], 3),
            "imbalance_signal": imbalance["signal"],
            "total_bid_volume": round(imbalance["total_bid"], 4),
            "total_ask_volume": round(imbalance["total_ask"], 4),
            "walls": walls,
            "bid_wall": walls.get("strongest_bid_wall"),
            "ask_wall": walls.get("strongest_ask_wall"),
            "depth_score": round(depth_score, 3),
            "spread_pct": round(spread["percentage"], 4),
            "spread_bps": round(spread["bps"], 2),
            "mid_price": round(spread["mid_price"], 2),
            "signal": signal,
            "score": round(score, 3),
            "best_bid": round(bid_prices[0], 2),
            "best_ask": round(ask_prices[0], 2)
        }
        
        # Update cache
        self._last_analysis = analysis
        self._last_fetch_time = now
        
        return analysis
    
    def _calculate_imbalance(self, bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> Dict:
        """
        Calculate bid/ask volume imbalance.
        
        A ratio > 1 means more buying pressure (bullish)
        A ratio < 1 means more selling pressure (bearish)
        """
        total_bid = np.sum(bid_volumes)
        total_ask = np.sum(ask_volumes)
        
        ratio = total_bid / total_ask if total_ask > 0 else 1.0
        
        if ratio >= self.imbalance_bullish:
            signal = "strong_buy_pressure"
        elif ratio >= 1.2:
            signal = "buy_pressure"
        elif ratio <= self.imbalance_bearish:
            signal = "strong_sell_pressure"
        elif ratio <= 0.8:
            signal = "sell_pressure"
        else:
            signal = "balanced"
        
        return {
            "ratio": ratio,
            "total_bid": total_bid,
            "total_ask": total_ask,
            "signal": signal
        }
    
    def _detect_walls(self, bid_prices: np.ndarray, bid_volumes: np.ndarray,
                     ask_prices: np.ndarray, ask_volumes: np.ndarray) -> Dict:
        """
        Detect large orders (walls) that may act as support/resistance.
        
        A wall is an order significantly larger than the average order size.
        """
        # Average order sizes
        avg_bid = np.mean(bid_volumes) if len(bid_volumes) > 0 else 0
        avg_ask = np.mean(ask_volumes) if len(ask_volumes) > 0 else 0
        
        # Find bid walls (support)
        bid_walls = []
        for i, (price, volume) in enumerate(zip(bid_prices, bid_volumes)):
            if avg_bid > 0 and volume >= avg_bid * self.wall_threshold:
                strength = "strong" if volume >= avg_bid * self.strong_wall_threshold else "moderate"
                bid_walls.append({
                    "price": round(price, 2),
                    "volume": round(volume, 4),
                    "multiple": round(volume / avg_bid, 2),
                    "strength": strength,
                    "type": "support"
                })
        
        # Find ask walls (resistance)
        ask_walls = []
        for i, (price, volume) in enumerate(zip(ask_prices, ask_volumes)):
            if avg_ask > 0 and volume >= avg_ask * self.wall_threshold:
                strength = "strong" if volume >= avg_ask * self.strong_wall_threshold else "moderate"
                ask_walls.append({
                    "price": round(price, 2),
                    "volume": round(volume, 4),
                    "multiple": round(volume / avg_ask, 2),
                    "strength": strength,
                    "type": "resistance"
                })
        
        # Get strongest walls
        strongest_bid = max(bid_walls, key=lambda x: x["volume"]) if bid_walls else None
        strongest_ask = max(ask_walls, key=lambda x: x["volume"]) if ask_walls else None
        
        return {
            "bid_walls": bid_walls[:5],  # Top 5 bid walls
            "ask_walls": ask_walls[:5],  # Top 5 ask walls
            "bid_wall_count": len(bid_walls),
            "ask_wall_count": len(ask_walls),
            "strongest_bid_wall": strongest_bid,
            "strongest_ask_wall": strongest_ask,
            "more_support": len(bid_walls) > len(ask_walls),
            "more_resistance": len(ask_walls) > len(bid_walls)
        }
    
    def _calculate_depth_score(self, bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> float:
        """
        Calculate weighted depth score.
        
        Weights closer price levels more heavily (they matter more).
        Returns score from -1 (heavy selling) to +1 (heavy buying).
        """
        # Weights: closer levels weighted more
        weights = np.exp(-np.arange(len(bid_volumes)) * 0.15)
        weights = weights / np.sum(weights)  # Normalize
        
        weighted_bid = np.sum(bid_volumes * weights[:len(bid_volumes)])
        
        weights_ask = np.exp(-np.arange(len(ask_volumes)) * 0.15)
        weights_ask = weights_ask / np.sum(weights_ask)
        weighted_ask = np.sum(ask_volumes * weights_ask[:len(ask_volumes)])
        
        total = weighted_bid + weighted_ask
        if total == 0:
            return 0.0
        
        # Score from -1 to +1
        score = (weighted_bid - weighted_ask) / total
        return score
    
    def _calculate_spread(self, bid_prices: np.ndarray, ask_prices: np.ndarray) -> Dict:
        """
        Calculate bid-ask spread.
        
        Tight spread = liquid market
        Wide spread = illiquid or volatile
        """
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        mid_price = (best_bid + best_ask) / 2
        
        spread = best_ask - best_bid
        spread_pct = (spread / mid_price) * 100
        spread_bps = spread_pct * 100  # Basis points
        
        return {
            "spread": spread,
            "percentage": spread_pct,
            "bps": spread_bps,
            "mid_price": mid_price
        }
    
    def _generate_signal(self, imbalance: Dict, walls: Dict, depth_score: float) -> Tuple[str, float]:
        """
        Generate overall order book signal.
        
        Combines imbalance, walls, and depth score into a single signal.
        """
        score = 0.0
        
        # Imbalance contribution (40% weight)
        ratio = imbalance["ratio"]
        if ratio >= self.imbalance_bullish:
            score += 0.4
        elif ratio >= 1.2:
            score += 0.2
        elif ratio <= self.imbalance_bearish:
            score -= 0.4
        elif ratio <= 0.8:
            score -= 0.2
        
        # Wall contribution (30% weight)
        if walls["more_support"]:
            score += 0.15
            if walls["strongest_bid_wall"] and walls["strongest_bid_wall"]["strength"] == "strong":
                score += 0.15
        elif walls["more_resistance"]:
            score -= 0.15
            if walls["strongest_ask_wall"] and walls["strongest_ask_wall"]["strength"] == "strong":
                score -= 0.15
        
        # Depth score contribution (30% weight)
        score += depth_score * 0.3
        
        # Clamp to -1 to +1
        score = max(-1.0, min(1.0, score))
        
        # Generate signal string
        if score >= 0.5:
            signal = "strong_buy_pressure"
        elif score >= 0.2:
            signal = "buy_pressure"
        elif score <= -0.5:
            signal = "strong_sell_pressure"
        elif score <= -0.2:
            signal = "sell_pressure"
        else:
            signal = "neutral"
        
        return signal, score
    
    def get_wall_summary(self) -> str:
        """
        Get human-readable wall summary.
        """
        analysis = self.analyze()
        
        if "error" in analysis:
            return f"Order book unavailable: {analysis['error']}"
        
        lines = []
        lines.append(f"Order Book Analysis for {self.symbol}")
        lines.append(f"  Imbalance: {analysis['imbalance_ratio']:.2f}x ({analysis['imbalance_signal']})")
        lines.append(f"  Depth Score: {analysis['depth_score']:.3f}")
        lines.append(f"  Spread: {analysis['spread_bps']:.1f} bps")
        
        if analysis["bid_wall"]:
            bw = analysis["bid_wall"]
            lines.append(f"  Support Wall: ${bw['price']:,.2f} ({bw['volume']:.4f} BTC, {bw['strength']})")
        
        if analysis["ask_wall"]:
            aw = analysis["ask_wall"]
            lines.append(f"  Resistance Wall: ${aw['price']:,.2f} ({aw['volume']:.4f} BTC, {aw['strength']})")
        
        lines.append(f"  Signal: {analysis['signal']} (score: {analysis['score']:.3f})")
        
        return "\n".join(lines)


# Test the analyzer
if __name__ == "__main__":
    print("Testing Order Book Analyzer...")
    
    analyzer = OrderBookAnalyzer()
    analysis = analyzer.analyze()
    
    print(f"\n{'='*60}")
    print(analyzer.get_wall_summary())
    print(f"{'='*60}")
    
    print("\nFull Analysis:")
    for key, value in analysis.items():
        if key != "walls":
            print(f"  {key}: {value}")
