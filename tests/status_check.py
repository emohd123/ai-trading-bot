import json
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Try temp_state.json (full snapshot from dashboard) first, fallback to bot_state.json
temp_path = os.path.join(config.DATA_DIR, 'temp_state.json')
bot_path = os.path.join(config.DATA_DIR, 'bot_state.json')
if os.path.exists(temp_path):
    with open(temp_path, encoding='utf-8-sig') as f:
        data = json.load(f)
elif os.path.exists(bot_path):
    with open(bot_path, encoding='utf-8-sig') as f:
        state = json.load(f)
    data = {**state}
    data.setdefault("current_price", 0)
    data.setdefault("last_update", "--")
    data.setdefault("balance_usdt", 0)
    data.setdefault("balance_btc", 0)
    data.setdefault("ai_score", 0)
    data.setdefault("activity_status", "--")
    data.setdefault("market_regime", "--")
    data.setdefault("indicators", {})
    data.setdefault("stochastic", {})
else:
    data = {"running": False, "current_price": 0, "position": None}

print('=' * 50)
print('       TRADING BOT STATUS')
print('=' * 50)
print()
running = 'YES - LIVE' if data.get('running') else 'STOPPED'
print(f'Running: {running}')
print(f'BTC Price: ${data.get("current_price", 0):,.2f}')
print(f'Last Update: {data.get("last_update", "--")}')
print()
print('--- ACCOUNT ---')
print(f'USDT Balance: ${data.get("balance_usdt", 0):.2f}')
print(f'BTC Balance: {data.get("balance_btc", 0):.8f}')
pos = 'IN TRADE' if data.get('position') else 'No position'
print(f'Position: {pos}')
print()
print('--- AI DECISION ---')
print(f'AI Score: {data.get("ai_score", 0):.3f}')
print(f'Decision: {data.get("decision", "--")}')
print(f'Activity: {data.get("activity_status", "--")}')
print()
print('--- MARKET ---')
print(f'Regime: {data.get("market_regime", "--")}')
ind = data.get('indicators', {})
rsi = ind.get('rsi', {})
macd = ind.get('macd', {})
print()
print('--- INDICATORS ---')
print(f'RSI: {rsi.get("value", "--")} ({rsi.get("signal", "--")})')
print(f'MACD: {macd.get("histogram", "--")} ({macd.get("signal", "--")})')
stoch = data.get('stochastic', {})
print(f'Stochastic: {stoch.get("k", "--")} ({stoch.get("signal", "--")})')
print()
print('--- SETTINGS ---')
print('Trade Amount: $20')
print('Max Positions: 1')
print('Profit Target: 10%')
print('Stop Loss: 5%')
print('=' * 50)
