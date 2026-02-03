# Project Rules – File & Folder Guidelines

Use these rules when editing or adding code.

---

## core/

- **binance_client.py** – Only Binance API calls. No trading logic. Use `config` for SYMBOL, API URLs.
- **trader.py** – Main trading loop. Imports from `ai.*`, `market.*`, `core.binance_client`. Uses `config` for thresholds.
- **self_healing.py** – Balance/position recovery only. No buy/sell decisions.

---

## ai/

- **ai_engine.py** – Decision logic only (BUY/SELL/HOLD). No order execution. Uses `config.INDICATOR_WEIGHTS`, `config.BUY_THRESHOLD`, etc.
- **analyzer.py** – Technical indicators. Pure functions, no I/O. Uses `config` for RSI_PERIOD, MACD_*, etc.
- **ml_*.py** – Use `config.MODEL_DIR` for model paths. No hardcoded `"models"`.
- **meta_ai.py** – Coordinates backtester, strategy_evolver, param_optimizer, deep_analyzer. No direct trading.
- **backtester.py** – Historical simulation only. No live orders.

---

## market/

- **market_regime.py** – Regime detection (trending_up/down, ranging, high_volatility). Uses `config` for thresholds.
- **multi_timeframe.py** – Fetches 4h/1h/15m. Needs client with `get_historical_klines`.
- **order_book.py**, **sentiment.py**, **correlation.py** – Optional. Handle missing APIs gracefully.

---

## notifications/

- **telegram_notifier.py** – Sends messages. Uses `config.DATA_DIR` for any file reads.
- **telegram_bot.py** – Command handler. Uses `config.DATA_DIR` for trade_history, bot_state.

---

## Root Files

- **config.py** – Single source for paths (`DATA_DIR`, `MODEL_DIR`) and trading params. No business logic.
- **dashboard.py** – Flask app. Imports from all packages. Uses `config.DATA_DIR` for state files.
- **main.py** – Thin entry point. Validates config, creates Trader, runs loop.

---

## Paths

- Use `config.DATA_DIR` for state, history, learning files.
- Use `config.MODEL_DIR` for ML models (.pkl, .keras).
- Never hardcode `"data"` or `"models"`.

---

## Imports

- Use package paths: `from core.binance_client import ...`, `from ai.ai_engine import ...`, `from market.market_regime import ...`.
