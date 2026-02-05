"""
Binance API Client
Handles all communication with Binance exchange
"""
import logging
import time
from typing import Optional, Dict, List, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import config

logger = logging.getLogger(__name__)


class BinanceClient:
    """Wrapper for Binance API interactions - live trading only"""

    def __init__(self):
        """Initialize Binance client with API keys"""
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Binance client and sync time with Binance server"""
        try:
            if config.USE_TESTNET:
                self.client = Client(
                    config.BINANCE_API_KEY,
                    config.BINANCE_API_SECRET,
                    testnet=True
                )
                print("[TESTNET] Connected to Binance Testnet")
            else:
                self.client = Client(
                    config.BINANCE_API_KEY,
                    config.BINANCE_API_SECRET
                )
                print("[LIVE] Connected to Binance")

            # Sync local time with Binance server to avoid -1021 timestamp errors
            try:
                server_time = self.client.get_server_time()
                local_ms = int(time.time() * 1000)
                offset_ms = server_time["serverTime"] - local_ms
                self.client.timestamp_offset = offset_ms
                if abs(offset_ms) > 500:
                    logger.info("Time sync: offset %+d ms (Binance server vs local)", offset_ms)
            except Exception as e:
                logger.warning("Could not sync time with Binance: %s – check system clock if you see timestamp errors", e)

            self.client.ping()
            print("API connection successful!")

        except BinanceAPIException as e:
            if e.code == -1021:
                logger.warning("Binance timestamp error at init – sync Windows time (Settings > Time & language > Sync now)")
            raise
        except Exception as e:
            logger.error("Connection Error: %s", e)
            raise

    def _sync_time(self):
        """Re-sync local time with Binance server (call on -1021 timestamp errors)."""
        try:
            server_time = self.client.get_server_time()
            local_ms = int(time.time() * 1000)
            offset_ms = server_time["serverTime"] - local_ms
            self.client.timestamp_offset = offset_ms
            logger.info("Time re-synced with Binance: offset %+d ms", offset_ms)
        except Exception as e:
            logger.warning("Could not re-sync time with Binance: %s", e)

    def get_current_price(self, symbol: str = None) -> float:
        """
        Get current price for a symbol

        Args:
            symbol: Trading pair symbol (default: from config)

        Returns:
            Current price as float
        """
        symbol = symbol or config.SYMBOL
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def get_historical_klines(
        self,
        symbol: str = None,
        interval: str = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get historical candlestick data

        Args:
            symbol: Trading pair symbol
            interval: Candle interval (1m, 5m, 1h, 1d, etc.)
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol or config.SYMBOL
        interval = interval or config.CANDLE_INTERVAL
        limit = limit or config.CANDLE_LIMIT

        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df.set_index('timestamp', inplace=True)

            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error("Error fetching klines: %s", e)
            return pd.DataFrame()

    def get_balance(self, asset: str = None) -> Optional[float]:
        """
        Get account balance for an asset.

        Returns:
            Available balance, or None if the API call failed (so callers can avoid
            overwriting good state with zeros).
        """
        asset = asset or config.QUOTE_ASSET
        max_retries = 3
        for attempt in range(max_retries):
            try:
                balance = self.client.get_asset_balance(asset=asset)
                return float(balance['free'])
            except BinanceAPIException as e:
                if e.code == -1021 or "timestamp" in str(e).lower():
                    self._sync_time()
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                logger.error("Error getting balance: %s", e)
                return None
            except Exception as e:
                logger.error("Error getting balance: %s", e)
                return None
        return None

    def place_market_buy(
        self,
        symbol: str = None,
        quote_amount: float = None
    ) -> Dict[str, Any]:
        """
        Place a market buy order

        Args:
            symbol: Trading pair
            quote_amount: Amount in quote currency (USDT) to spend

        Returns:
            Order result dictionary
        """
        symbol = symbol or config.SYMBOL
        quote_amount = quote_amount or config.TRADE_AMOUNT_USDT

        for attempt in range(2):
            try:
                price = self.get_current_price(symbol)
                quantity = quote_amount / price
                info = self.client.get_symbol_info(symbol)
                step_size = float([f['stepSize'] for f in info['filters']
                                 if f['filterType'] == 'LOT_SIZE'][0])
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                quantity = round(quantity, precision)
                order = self.client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
                return {
                    "status": "filled",
                    "symbol": symbol,
                    "side": "BUY",
                    "type": "MARKET",
                    "quantity": float(order['executedQty']),
                    "price": float(order['fills'][0]['price']) if order['fills'] else self.get_current_price(symbol),
                    "quote_amount": quote_amount,
                    "order_id": order['orderId'],
                    "timestamp": time.time(),
                }
            except BinanceAPIException as e:
                if (e.code == -1021 or "timestamp" in str(e).lower()) and attempt == 0:
                    self._sync_time()
                    time.sleep(0.5)
                    continue
                return {"status": "error", "message": str(e)}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "error", "message": "Timestamp error after retry"}

    def place_market_sell(
        self,
        symbol: str = None,
        quantity: float = None
    ) -> Dict[str, Any]:
        """
        Place a market sell order

        Args:
            symbol: Trading pair
            quantity: Amount in base currency (BTC) to sell

        Returns:
            Order result dictionary
        """
        symbol = symbol or config.SYMBOL
        import math

        for attempt in range(2):
            try:
                actual_balance = self.get_balance(config.BASE_ASSET)
                if actual_balance is None:
                    return {"status": "error", "message": "Could not get balance (API error)"}
                if quantity is None or quantity > actual_balance:
                    if quantity is not None:
                        logger.info("SELL: Adjusting quantity from %.8f to actual balance %.8f", quantity, actual_balance)
                    quantity = actual_balance
                if quantity <= 0:
                    return {"status": "error", "message": "No balance to sell - clear position"}
                info = self.client.get_symbol_info(symbol)
                step_size = float([f['stepSize'] for f in info['filters']
                                 if f['filterType'] == 'LOT_SIZE'][0])
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                multiplier = 10 ** precision
                quantity = math.floor(quantity * multiplier) / multiplier
                min_qty = float([f['minQty'] for f in info['filters']
                               if f['filterType'] == 'LOT_SIZE'][0])
                if quantity < min_qty:
                    return {"status": "error", "message": f"Quantity {quantity} below minimum {min_qty}"}
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
                return {
                    "status": "filled",
                    "symbol": symbol,
                    "side": "SELL",
                    "type": "MARKET",
                    "quantity": float(order['executedQty']),
                    "price": float(order['fills'][0]['price']) if order['fills'] else self.get_current_price(symbol),
                    "quote_amount": float(order['cummulativeQuoteQty']),
                    "order_id": order['orderId'],
                    "timestamp": time.time(),
                }
            except BinanceAPIException as e:
                if (e.code == -1021 or "timestamp" in str(e).lower()) and attempt == 0:
                    self._sync_time()
                    time.sleep(0.5)
                    continue
                return {"status": "error", "message": str(e)}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "error", "message": "Timestamp error after retry"}

    def get_trade_history(self) -> List[Dict]:
        """Get trade history from Binance"""
        try:
            trades = self.client.get_my_trades(symbol=config.SYMBOL, limit=50)
            return trades
        except Exception as e:
            logger.error("Error getting trade history: %s", e)
            return []

    def get_account_value(self) -> float:
        """Get total account value in quote currency (USDT) - includes all traded coins"""
        total_value = 0.0
        
        # Add USDT balance
        usdt_balance = self.get_balance(config.QUOTE_ASSET) or 0.0
        total_value += usdt_balance
        
        # Add value of all traded coins (BTC, ETH, BNB, SOL, XRP, DOGE, ADA)
        symbols = getattr(config, 'SYMBOLS', [("BTCUSDT", "BTC")])
        for symbol, base_asset in symbols:
            try:
                balance = self.get_balance(base_asset) or 0.0
                if balance > 0:
                    price = self.get_current_price(symbol)
                    total_value += balance * price
            except Exception as e:
                logger.warning(f"Could not get value for {base_asset}: {e}")
                continue
        
        return total_value


# Test the client (requires API keys in .env)
if __name__ == "__main__":
    import config as _  # Ensure logging is set up
    logger.info("Testing Binance Client (Live)...")

    client = BinanceClient()

    # Get current price
    price = client.get_current_price()
    logger.info("Current BTC/USDT price: $%s", f"{price:,.2f}")

    # Get historical data
    df = client.get_historical_klines(limit=10)
    logger.info("Last 10 candles:\n%s", df.tail())

    usdt_balance = client.get_balance(config.QUOTE_ASSET)
    logger.info("USDT balance: $%s | Total value: $%s", f"{(usdt_balance or 0):,.2f}", f"{client.get_account_value():,.2f}")
