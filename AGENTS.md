# AI Trading Bot – Agent Guide

## Project Overview

AI-powered BTC/USDT trading bot for Binance. Uses technical analysis, ML prediction, and market regime detection to make buy/sell decisions. Includes a web dashboard, Telegram notifications, and autonomous Meta AI for self-improvement.

**Stack:** Python 3, Flask, SocketIO, Binance API, scikit-learn, XGBoost, LightGBM

---

## Entry Points

| Entry | Purpose | Run |
|-------|---------|-----|
| **main.py** | CLI trading loop | `python main.py` or `python main.py --testnet` |
| **dashboard.py** | Web UI + trading | `python dashboard.py` → http://localhost:5000 |
| **tests/status_check.py** | Quick status | `python tests/status_check.py` or `scripts/status_check.bat` |
| **ai/ml_training.py** | Retrain ML models | `python -m ai.ml_training` |

---

## Component Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                     │
│  main.py (CLI)                    dashboard.py (Web + Trading)           │
└────────────────┬──────────────────────────────┬─────────────────────────┘
                 │                              │
                 ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  core/trader.py  – Main trading loop (or dashboard runs its own loop)    │
│  • Fetches data via BinanceClient                                         │
│  • Gets analysis from TechnicalAnalyzer + get_mtf_analysis               │
│  • Gets decision from AIEngine                                            │
│  • Executes BUY/SELL via BinanceClient                                    │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
     ┌───────────┼───────────┬──────────────┬──────────────┐
     ▼           ▼           ▼              ▼              ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ ┌─────────────┐
│ core/   │ │ ai/     │ │ market/ │ │ ai/         │ │ notifications│
│binance_ │ │ai_engine│ │market_  │ │meta_ai      │ │telegram_    │
│client   │ │analyzer │ │regime   │ │param_opt    │ │notifier     │
│         │ │ml_*     │ │multi_tf │ │deep_analyzer│ │telegram_bot │
└─────────┘ └─────────┘ └─────────┘ └─────────────┘ └─────────────┘
```

---

## Package Roles

| Package | Role | Key Files |
|---------|------|------------|
| **core/** | Exchange & trading | `binance_client` – API; `trader` – loop; `self_healing` – balance fixes |
| **ai/** | Decisions & learning | `ai_engine` – BUY/SELL/HOLD; `analyzer` – indicators; `ml_predictor` – ML; `meta_ai` – self-improvement |
| **market/** | Market context | `market_regime` – trend; `multi_timeframe` – 4h/1h/15m; `order_book`, `sentiment`, `correlation` |
| **notifications/** | Alerts | `telegram_notifier` – trade alerts; `telegram_bot` – commands |
| **tests/** | Checks | `status_check` – status; `test_ml_prediction` – ML pipeline |

---

## Data Flow

1. **BinanceClient** → klines (OHLCV)
2. **TechnicalAnalyzer** → indicators (RSI, MACD, Bollinger, etc.)
3. **MarketRegime** → regime (trending_up/down, ranging, high_volatility)
4. **MultiTimeframe** → 4h/1h/15m analysis
5. **AIEngine** → score (-1 to +1), Decision (BUY/SELL/HOLD)
6. **Trader** → places orders via BinanceClient

**Persistence:** `config.DATA_DIR` (data/), `config.MODEL_DIR` (models/)

---

## Restart after changes

After changing `config.py` or any bot code (core/, ai/, dashboard.py, etc.), **restart the bot** so changes take effect: stop Python processes, then start `dashboard.py` (e.g. `scripts\start_bot.bat` or run `dashboard.py` minimized). See `.cursor/rules/restart-bot-after-changes.mdc`.

---

## Config

All settings in `config.py`. Paths: `config.DATA_DIR`, `config.MODEL_DIR`. API keys from `.env`.
