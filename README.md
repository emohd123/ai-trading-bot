# AI Crypto Trading Bot

Automated BTC/USDT trading bot that buys low and sells high with a 10% profit target.

## Features

- **AI-Powered Decisions**: Combines 5 technical indicators (RSI, MACD, Bollinger Bands, EMA, Support/Resistance)
- **10% Profit Target**: Automatically sells when profit reaches 10%
- **5% Stop Loss**: Protects against large losses
- **Binance Integration**: Works with Binance exchange
- **Live Trading Only**: Real market execution

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file:

```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_secret
```

### 3. Run Live

```bash
python main.py
```

Or use the web dashboard:

```bash
python dashboard.py
```

Then open http://localhost:5000

## Project Structure

```
bot/
├── core/           # Core trading: binance_client, trader, self_healing
├── ai/             # AI/ML: ai_engine, analyzer, ml_*, meta_ai, backtester, etc.
├── market/         # Market analysis: market_regime, order_book, sentiment, etc.
├── notifications/  # Telegram bot and notifier
├── tests/          # Tests and status_check utility
├── data/           # Runtime data (state, history, learning)
├── models/         # ML models and performance data
├── scripts/        # Startup scripts (start_bot.bat, stop_bot.bat, etc.)
├── templates/      # Dashboard HTML templates
├── main.py         # CLI entry point
├── dashboard.py    # Web dashboard entry point
└── config.py       # Configuration
```

## How It Works

1. **Fetches** BTC/USDT price data from Binance
2. **Calculates** 5 technical indicators:
   - RSI (oversold/overbought)
   - MACD (momentum)
   - Bollinger Bands (volatility)
   - EMA (trend)
   - Support/Resistance (price levels)
3. **Generates** AI score (-1 to +1)
4. **Buys** when score > 0.3 (bullish)
5. **Sells** at +10% profit, -5% loss, or bearish signal

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| SYMBOL | BTCUSDT | Trading pair |
| TRADE_AMOUNT_USDT | 100 | Amount per trade |
| PROFIT_TARGET | 0.10 | 10% profit target |
| STOP_LOSS | 0.05 | 5% stop loss |
| BUY_THRESHOLD | 0.3 | AI score to trigger buy |
| CHECK_INTERVAL | 60 | Seconds between checks |

## Commands

```bash
python main.py              # Live trading
python main.py --testnet    # Binance testnet (paper trading)
python main.py --live       # Live trading (with confirmation)
python dashboard.py         # Web dashboard
python tests/status_check.py  # Quick status check (or scripts/status_check.bat)
python -m ai.ml_training    # Retrain ML models
```

## Risk Warning

Trading cryptocurrency involves significant risk. Only trade with money you can afford to lose. This bot is for educational purposes - always do your own research.

## License

MIT
