"""
Telegram Bot Command Handler - Enhanced Version
Responds to your questions with live trading data!

QUERY COMMANDS:
  /status - Current bot status and price
  /profit - Total profit report
  /trades - Recent trade history
  /balance - Account balance
  /position - Current position info
  /stats - Trading statistics

CONTROL COMMANDS:
  /start_trading - Start the trading bot
  /stop_trading - Stop the trading bot
  /pause - Pause trading (keep monitoring)
  /resume - Resume trading after pause

SETTINGS COMMANDS:
  /setprofit <value> - Set profit target (e.g., /setprofit 2)
  /setstop <value> - Set stop loss (e.g., /setstop 1)
  /settrade <value> - Set trade amount (e.g., /settrade 50)
  /settings - Show current settings

MARKET COMMANDS:
  /market - Full market overview
  /fear - Fear & Greed Index
  /dominance - BTC dominance
  /funding - Funding rate
  /orderbook - Order book analysis

  /help - List all commands
"""

import os
import json
import threading
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in path for config
import sys
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
DASHBOARD_URL = "http://localhost:5000"

# Track last update ID to avoid processing same message twice
last_update_id = 0

# Scheduled reports tracking
last_hourly_report = 0
last_daily_report = 0
last_trade_count = 0


def get_trade_history():
    """Load trade history from file"""
    try:
        path = os.path.join(config.DATA_DIR, 'trade_history.json')
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return []


def get_bot_state():
    """Load bot state from file"""
    try:
        path = os.path.join(config.DATA_DIR, 'bot_state.json')
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}


def calculate_stats():
    """Calculate trading statistics"""
    trades = get_trade_history()
    
    total_profit = 0
    winning = 0
    losing = 0
    biggest_win = 0
    biggest_loss = 0
    
    for t in trades:
        if t.get('type') == 'SELL':
            pnl = t.get('pnl', 0)
            total_profit += pnl
            if pnl >= 0:
                winning += 1
                if pnl > biggest_win:
                    biggest_win = pnl
            else:
                losing += 1
                if pnl < biggest_loss:
                    biggest_loss = pnl
    
    total_trades = winning + losing
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_trades': total_trades,
        'winning': winning,
        'losing': losing,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'biggest_win': biggest_win,
        'biggest_loss': biggest_loss
    }


def send_message(chat_id, text):
    """Send a message to Telegram"""
    try:
        url = f"{BASE_URL}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error("Error sending message: %s", e)


def handle_status(chat_id):
    """Handle /status command"""
    state = get_bot_state()
    
    running = "🟢 RUNNING" if state.get('running', False) else "🔴 STOPPED"
    price = state.get('current_price', 0)
    decision = state.get('decision', 'UNKNOWN')
    regime = state.get('market_regime', 'unknown')
    ai_score = state.get('ai_score', 0)
    
    msg = f"""📊 BOT STATUS

{running}

💵 BTC Price: ${price:,.2f}
🤖 AI Score: {ai_score:.3f}
📈 Decision: {decision}
🌊 Market: {regime.upper()}

🕐 {datetime.now().strftime('%H:%M:%S')}"""
    
    send_message(chat_id, msg)


def handle_profit(chat_id):
    """Handle /profit command"""
    stats = calculate_stats()
    state = get_bot_state()
    
    total_profit = stats['total_profit']
    status = "IN PROFIT ✅" if total_profit >= 0 else "IN LOSS ❌"
    
    if stats['win_rate'] >= 70:
        performance = "🔥 Excellent"
    elif stats['win_rate'] >= 55:
        performance = "👍 Good"
    elif stats['win_rate'] >= 45:
        performance = "📊 Average"
    else:
        performance = "📉 Needs Work"
    
    msg = f"""💰 PROFIT REPORT

💎 Total Profit: ${total_profit:+,.2f}
📊 Status: {status}

━━━━━━━━━━━━━━━━━━━━

📈 Statistics:
   🔢 Total Trades: {stats['total_trades']}
   ✅ Winning: {stats['winning']}
   ❌ Losing: {stats['losing']}
   🎯 Win Rate: {stats['win_rate']:.1f}%
   💵 Avg/Trade: ${stats['avg_profit']:+,.2f}

━━━━━━━━━━━━━━━━━━━━

🏆 Best Trade: ${stats['biggest_win']:+,.2f}
💔 Worst Trade: ${stats['biggest_loss']:+,.2f}
📊 Performance: {performance}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    send_message(chat_id, msg)


def handle_trades(chat_id):
    """Handle /trades command"""
    trades = get_trade_history()
    
    if not trades:
        send_message(chat_id, "📭 No trades yet")
        return
    
    # Get last 5 SELL trades
    sell_trades = [t for t in trades if t.get('type') == 'SELL'][-5:]
    
    if not sell_trades:
        send_message(chat_id, "📭 No completed trades yet")
        return
    
    msg = "📜 RECENT TRADES (Last 5)\n\n"
    
    for i, t in enumerate(reversed(sell_trades), 1):
        pnl = t.get('pnl', 0)
        pnl_pct = t.get('pnl_percent', 0)
        emoji = "✅" if pnl >= 0 else "❌"
        time_str = t.get('time', 'Unknown')
        
        msg += f"{emoji} #{i}: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
        msg += f"   📅 {time_str}\n\n"
    
    send_message(chat_id, msg)


def handle_balance(chat_id):
    """Handle /balance command"""
    state = get_bot_state()
    
    usdt = state.get('balance_usdt', 0)
    btc = state.get('balance_btc', 0)
    total = state.get('total_value', 0)
    
    msg = f"""💰 ACCOUNT BALANCE

💵 USDT: ${usdt:,.2f}
🪙 BTC: {btc:.8f}

━━━━━━━━━━━━━━━━━━━━

📊 Total Value: ${total:,.2f}

🕐 {datetime.now().strftime('%H:%M:%S')}"""
    
    send_message(chat_id, msg)


def handle_position(chat_id):
    """Handle /position command"""
    state = get_bot_state()
    position = state.get('position')
    
    if not position:
        msg = """📍 POSITION STATUS

🔵 No active position

Waiting for buy signal..."""
    else:
        entry = position.get('entry_price', 0)
        qty = position.get('quantity', 0)
        current = state.get('current_price', 0)
        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        pnl_usd = (current - entry) * qty
        emoji = "📈" if pnl_pct >= 0 else "📉"
        
        msg = f"""📍 POSITION STATUS

🟢 IN POSITION

💵 Entry Price: ${entry:,.2f}
💰 Current Price: ${current:,.2f}
🪙 Quantity: {qty:.6f} BTC

━━━━━━━━━━━━━━━━━━━━

{emoji} Unrealized P/L:
   ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%)

🕐 {datetime.now().strftime('%H:%M:%S')}"""
    
    send_message(chat_id, msg)


def handle_stats(chat_id):
    """Handle /stats command"""
    stats = calculate_stats()
    state = get_bot_state()
    
    # Calculate additional stats
    trades = get_trade_history()
    sell_trades = [t for t in trades if t.get('type') == 'SELL']
    
    # Streak calculation
    current_streak = 0
    streak_type = ""
    for t in reversed(sell_trades):
        pnl = t.get('pnl', 0)
        if current_streak == 0:
            streak_type = "win" if pnl >= 0 else "loss"
            current_streak = 1
        elif (pnl >= 0 and streak_type == "win") or (pnl < 0 and streak_type == "loss"):
            current_streak += 1
        else:
            break
    
    streak_emoji = "🔥" if streak_type == "win" else "❄️"
    streak_text = f"{streak_emoji} {current_streak} {streak_type}{'s' if current_streak > 1 else ''} in a row"
    
    msg = f"""📊 TRADING STATISTICS

💰 Total Profit: ${stats['total_profit']:+,.2f}
🔢 Total Trades: {stats['total_trades']}

━━━━━━━━━━━━━━━━━━━━

✅ Wins: {stats['winning']}
❌ Losses: {stats['losing']}
🎯 Win Rate: {stats['win_rate']:.1f}%

━━━━━━━━━━━━━━━━━━━━

💵 Avg Profit/Trade: ${stats['avg_profit']:+,.2f}
🏆 Biggest Win: ${stats['biggest_win']:+,.2f}
💔 Biggest Loss: ${stats['biggest_loss']:+,.2f}

━━━━━━━━━━━━━━━━━━━━

📈 Current Streak: {streak_text}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    send_message(chat_id, msg)


# =============================================================================
# CONTROL COMMANDS
# =============================================================================

def handle_start_trading(chat_id):
    """Handle /start_trading command - Start the bot"""
    try:
        r = requests.post(f"{DASHBOARD_URL}/api/start", timeout=10)
        if r.ok:
            data = r.json()
            if data.get('status') == 'success':
                send_message(chat_id, "🚀 Trading bot STARTED!\n\nBot is now actively trading.")
            else:
                send_message(chat_id, f"⚠️ {data.get('message', 'Unknown response')}")
        else:
            send_message(chat_id, f"❌ Failed to start: {r.text}")
    except requests.exceptions.ConnectionError:
        send_message(chat_id, "❌ Could not connect to dashboard.\nMake sure dashboard.py is running.")
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


def handle_stop_trading(chat_id):
    """Handle /stop_trading command - Stop the bot"""
    try:
        r = requests.post(f"{DASHBOARD_URL}/api/stop", timeout=10)
        if r.ok:
            data = r.json()
            send_message(chat_id, "⛔ Trading bot STOPPED!\n\nBot is no longer trading.")
        else:
            send_message(chat_id, f"❌ Failed to stop: {r.text}")
    except requests.exceptions.ConnectionError:
        send_message(chat_id, "❌ Could not connect to dashboard.")
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


def handle_pause(chat_id):
    """Handle /pause command - Pause trading"""
    try:
        r = requests.post(f"{DASHBOARD_URL}/api/pause", timeout=10)
        if r.ok:
            send_message(chat_id, "⏸️ Trading PAUSED!\n\nBot is monitoring but not trading.\nSend /resume to continue.")
        else:
            send_message(chat_id, f"❌ Failed to pause: {r.text}")
    except requests.exceptions.ConnectionError:
        send_message(chat_id, "❌ Could not connect to dashboard.")
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


def handle_resume(chat_id):
    """Handle /resume command - Resume trading"""
    try:
        r = requests.post(f"{DASHBOARD_URL}/api/resume", timeout=10)
        if r.ok:
            send_message(chat_id, "▶️ Trading RESUMED!\n\nBot is now actively trading again.")
        else:
            send_message(chat_id, f"❌ Failed to resume: {r.text}")
    except requests.exceptions.ConnectionError:
        send_message(chat_id, "❌ Could not connect to dashboard.")
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


# =============================================================================
# SETTINGS COMMANDS
# =============================================================================

def handle_settings(chat_id):
    """Handle /settings command - Show current settings"""
    try:
        r = requests.get(f"{DASHBOARD_URL}/api/settings", timeout=10)
        if r.ok:
            s = r.json()
            msg = f"""⚙️ CURRENT SETTINGS

💰 Trade Amount: ${s.get('trade_amount', 0):.0f}
🎯 Profit Target: {s.get('profit_target', 0):.1f}%
🛑 Stop Loss: {s.get('stop_loss', 0):.1f}%
⏱️ Check Interval: {s.get('check_interval', 30)}s

━━━━━━━━━━━━━━━━━━━━

To change settings:
  /setprofit <value>
  /setstop <value>
  /settrade <value>

Example: /setprofit 2"""
            send_message(chat_id, msg)
        else:
            send_message(chat_id, "❌ Could not fetch settings")
    except requests.exceptions.ConnectionError:
        send_message(chat_id, "❌ Could not connect to dashboard.")
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


def handle_setprofit(chat_id, value):
    """Handle /setprofit command - Set profit target"""
    # Validate range (0.5% to 10%)
    if value < 0.5 or value > 10:
        send_message(chat_id, "❌ Profit target must be between 0.5% and 10%")
        return
    
    try:
        r = requests.post(f"{DASHBOARD_URL}/api/settings", 
                         json={"profit_target": value}, timeout=10)
        if r.ok:
            send_message(chat_id, f"✅ Profit target set to {value}%")
        else:
            send_message(chat_id, f"❌ Failed to update: {r.text}")
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


def handle_setstop(chat_id, value):
    """Handle /setstop command - Set stop loss"""
    # Validate range (0.5% to 5%)
    if value < 0.5 or value > 5:
        send_message(chat_id, "❌ Stop loss must be between 0.5% and 5%")
        return
    
    try:
        r = requests.post(f"{DASHBOARD_URL}/api/settings", 
                         json={"stop_loss": value}, timeout=10)
        if r.ok:
            send_message(chat_id, f"✅ Stop loss set to {value}%")
        else:
            send_message(chat_id, f"❌ Failed to update: {r.text}")
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


def handle_settrade(chat_id, value):
    """Handle /settrade command - Set trade amount"""
    # Validate range ($10 to $1000)
    if value < 10 or value > 1000:
        send_message(chat_id, "❌ Trade amount must be between $10 and $1000")
        return
    
    try:
        r = requests.post(f"{DASHBOARD_URL}/api/settings", 
                         json={"trade_amount": value}, timeout=10)
        if r.ok:
            send_message(chat_id, f"✅ Trade amount set to ${value:.0f}")
        else:
            send_message(chat_id, f"❌ Failed to update: {r.text}")
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


# =============================================================================
# MARKET COMMANDS
# =============================================================================

def handle_market(chat_id):
    """Handle /market command - Full market overview"""
    try:
        # Get data from dashboard API
        r = requests.get(f"{DASHBOARD_URL}/api/state", timeout=10)
        if not r.ok:
            send_message(chat_id, "❌ Could not fetch market data")
            return
        
        data = r.json()
        
        # Extract market context
        sentiment = data.get('sentiment', {})
        correlation = data.get('correlation', {})
        order_book = data.get('order_book', {})
        
        fear_greed = sentiment.get('fear_greed_value', 50)
        fg_class = sentiment.get('fear_greed_class', 'Neutral')
        funding = sentiment.get('funding_rate', 0)
        btc_dom = correlation.get('btc_dominance', 50)
        market_24h = correlation.get('market_cap_change', 0)
        ob_signal = order_book.get('signal', 'neutral')
        ob_ratio = order_book.get('imbalance_ratio', 1.0)
        
        # Fear emoji
        if fear_greed <= 20:
            fg_emoji = "😱"
        elif fear_greed <= 40:
            fg_emoji = "😰"
        elif fear_greed <= 60:
            fg_emoji = "😐"
        elif fear_greed <= 80:
            fg_emoji = "😊"
        else:
            fg_emoji = "🤑"
        
        price = data.get('current_price', 0)
        regime = data.get('market_regime', 'unknown')
        
        msg = f"""🌍 MARKET OVERVIEW

💵 BTC Price: ${price:,.2f}
🌊 Regime: {regime.upper()}

━━━━━━━━━━━━━━━━━━━━

{fg_emoji} Fear & Greed: {fear_greed} ({fg_class})
📊 BTC Dominance: {btc_dom:.1f}%
📈 Market 24h: {market_24h:+.2f}%
💹 Funding Rate: {funding:.4f}%

━━━━━━━━━━━━━━━━━━━━

📖 Order Book: {ob_signal.replace('_', ' ').title()}
   Imbalance: {ob_ratio:.2f}x

🕐 {datetime.now().strftime('%H:%M:%S')}"""
        
        send_message(chat_id, msg)
    except Exception as e:
        send_message(chat_id, f"❌ Error fetching market data: {str(e)}")


def handle_fear(chat_id):
    """Handle /fear command - Fear & Greed Index"""
    try:
        from sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer()
        data = sa.analyze()
        
        fg = data.get('fear_greed', {})
        value = fg.get('value', 50)
        classification = fg.get('classification', 'Neutral')
        
        # Emoji based on value
        if value <= 20:
            emoji = "😱"
            advice = "Extreme Fear = Potential buying opportunity"
        elif value <= 40:
            emoji = "😰"
            advice = "Fear in the market - be cautious"
        elif value <= 60:
            emoji = "😐"
            advice = "Neutral sentiment"
        elif value <= 80:
            emoji = "😊"
            advice = "Greed in the market - watch for tops"
        else:
            emoji = "🤑"
            advice = "Extreme Greed = Consider taking profits"
        
        # Create visual bar
        filled = int(value / 10)
        bar = "🟢" * filled + "⚪" * (10 - filled)
        
        msg = f"""{emoji} FEAR & GREED INDEX

📊 Value: {value}/100
📈 Status: {classification}

{bar}
0 ━━━━━━━━━━━━━ 100
Fear              Greed

━━━━━━━━━━━━━━━━━━━━

💡 {advice}

🕐 {datetime.now().strftime('%H:%M:%S')}"""
        
        send_message(chat_id, msg)
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


def handle_dominance(chat_id):
    """Handle /dominance command - BTC dominance"""
    try:
        from correlation import CorrelationAnalyzer
        ca = CorrelationAnalyzer()
        data = ca.analyze()
        
        btc_dom = data.get('btc_dominance', 50)
        eth_dom = data.get('eth_dominance', 0)
        market_cap = data.get('market_cap_usd', 0)
        change_24h = data.get('market_cap_change_24h', 0)
        
        # Interpretation
        if btc_dom > 55:
            trend = "📈 BTC Dominant - Alts underperforming"
        elif btc_dom < 45:
            trend = "📉 Alt Season - Alts outperforming"
        else:
            trend = "📊 Balanced market"
        
        msg = f"""👑 BTC DOMINANCE

📊 BTC Dominance: {btc_dom:.2f}%
📊 ETH Dominance: {eth_dom:.2f}%

━━━━━━━━━━━━━━━━━━━━

💰 Total Market Cap: ${market_cap/1e12:.2f}T
📈 24h Change: {change_24h:+.2f}%

━━━━━━━━━━━━━━━━━━━━

💡 {trend}

🕐 {datetime.now().strftime('%H:%M:%S')}"""
        
        send_message(chat_id, msg)
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


def handle_funding(chat_id):
    """Handle /funding command - Funding rate"""
    try:
        from sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer()
        data = sa.analyze()
        
        fr = data.get('funding_rate', {})
        rate = fr.get('rate', 0)
        rate_pct = fr.get('rate_pct', 0)
        signal = fr.get('signal', 'neutral')
        
        # Interpretation
        if rate < -0.01:
            emoji = "🟢"
            interpretation = "Very negative - Shorts paying longs (Bullish)"
        elif rate < 0:
            emoji = "🟡"
            interpretation = "Slightly negative - Mild bullish"
        elif rate < 0.01:
            emoji = "⚪"
            interpretation = "Neutral funding"
        elif rate < 0.03:
            emoji = "🟡"
            interpretation = "Slightly positive - Mild bearish"
        else:
            emoji = "🔴"
            interpretation = "Very positive - Longs paying shorts (Bearish)"
        
        msg = f"""{emoji} FUNDING RATE

💹 Rate: {rate_pct:.4f}%
📊 Signal: {signal.title()}

━━━━━━━━━━━━━━━━━━━━

💡 {interpretation}

ℹ️ Negative = shorts pay longs (bullish)
ℹ️ Positive = longs pay shorts (bearish)

🕐 {datetime.now().strftime('%H:%M:%S')}"""
        
        send_message(chat_id, msg)
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


def handle_orderbook(chat_id):
    """Handle /orderbook command - Order book analysis"""
    try:
        from order_book import OrderBookAnalyzer
        ob = OrderBookAnalyzer()
        data = ob.analyze()
        
        imbalance = data.get('imbalance_ratio', 1.0)
        signal = data.get('signal', 'neutral')
        score = data.get('score', 0)
        bid_vol = data.get('total_bid_volume', 0)
        ask_vol = data.get('total_ask_volume', 0)
        spread = data.get('spread_bps', 0)
        
        # Get walls
        walls = data.get('walls', {})
        bid_walls = walls.get('bid_wall_count', 0)
        ask_walls = walls.get('ask_wall_count', 0)
        
        # Signal emoji
        if 'buy' in signal:
            emoji = "🟢"
        elif 'sell' in signal:
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        msg = f"""{emoji} ORDER BOOK ANALYSIS

📊 Signal: {signal.replace('_', ' ').title()}
📈 Imbalance: {imbalance:.3f}x
🎯 Score: {score:+.2f}

━━━━━━━━━━━━━━━━━━━━

📗 Bid Volume: {bid_vol:.2f} BTC
📕 Ask Volume: {ask_vol:.2f} BTC
📊 Spread: {spread:.1f} bps

━━━━━━━━━━━━━━━━━━━━

🧱 Support Walls: {bid_walls}
🧱 Resistance Walls: {ask_walls}

🕐 {datetime.now().strftime('%H:%M:%S')}"""
        
        send_message(chat_id, msg)
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


# =============================================================================
# SCHEDULED REPORTS
# =============================================================================

def send_hourly_summary():
    """Send hourly trading summary if trades occurred"""
    global last_trade_count
    
    stats = calculate_stats()
    current_trades = stats['total_trades']
    
    # Only send if new trades occurred
    if current_trades <= last_trade_count:
        return
    
    trades_this_hour = current_trades - last_trade_count
    last_trade_count = current_trades
    
    msg = f"""⏰ HOURLY SUMMARY

📊 Trades this hour: {trades_this_hour}
💰 Total Profit: ${stats['total_profit']:+,.2f}
🎯 Win Rate: {stats['win_rate']:.1f}%

🕐 {datetime.now().strftime('%H:%M')}"""
    
    send_message(TELEGRAM_CHAT_ID, msg)


def send_daily_summary():
    """Send daily trading summary"""
    stats = calculate_stats()
    state = get_bot_state()
    
    balance = state.get('total_value', 0)
    
    if stats['total_profit'] >= 0:
        emoji = "🎉"
    else:
        emoji = "😔"
    
    msg = f"""{emoji} DAILY SUMMARY

📊 Total Trades: {stats['total_trades']}
✅ Wins: {stats['winning']}
❌ Losses: {stats['losing']}
🎯 Win Rate: {stats['win_rate']:.1f}%

━━━━━━━━━━━━━━━━━━━━

💰 Total Profit: ${stats['total_profit']:+,.2f}
💵 Balance: ${balance:,.2f}

📅 {datetime.now().strftime('%Y-%m-%d')}"""
    
    send_message(TELEGRAM_CHAT_ID, msg)


def check_scheduled_reports():
    """Check if scheduled reports should be sent"""
    global last_hourly_report, last_daily_report
    
    now = time.time()
    current_hour = datetime.now().hour
    
    # Hourly report (every hour)
    if now - last_hourly_report > 3600:
        last_hourly_report = now
        send_hourly_summary()
    
    # Daily report at midnight (hour 0)
    if current_hour == 0 and now - last_daily_report > 82800:  # 23 hours
        last_daily_report = now
        send_daily_summary()


# =============================================================================
# MAINTENANCE COMMANDS
# =============================================================================

def handle_health(chat_id):
    """Handle /health command - run health check"""
    try:
        send_message(chat_id, "Running health check...")
        
        response = requests.post(f"{DASHBOARD_URL}/api/health", timeout=30)
        if response.status_code == 200:
            data = response.json()
            report = data.get('report', {})
            health = data.get('health', {})
            
            fixes = health.get('fixes_applied', [])
            issues = report.get('total_issues', 0)
            total_fixes = report.get('total_fixes', 0)
            
            if fixes:
                fix_list = '\n'.join([f"  - {f}" for f in fixes])
                msg = f"""🔧 HEALTH CHECK COMPLETE

Fixes Applied: {len(fixes)}
{fix_list}

Total Issues Found: {issues}
Total Fixes (all time): {total_fixes}"""
            else:
                msg = f"""✅ HEALTH CHECK COMPLETE

All Systems Healthy!

Total Issues Found: {issues}
Total Fixes (all time): {total_fixes}"""
            
            send_message(chat_id, msg)
        else:
            send_message(chat_id, f"Health check failed: {response.status_code}")
    except Exception as e:
        send_message(chat_id, f"Health check error: {e}")


def handle_sync(chat_id):
    """Handle /sync command - sync balance with Binance"""
    try:
        send_message(chat_id, "Syncing balance with Binance...")
        
        response = requests.post(f"{DASHBOARD_URL}/api/sync_balance", timeout=15)
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == 'ok':
                old_qty = data.get('old_quantity', 0)
                new_qty = data.get('new_quantity', 0)
                
                if old_qty != new_qty:
                    msg = f"""✅ BALANCE SYNCED

Old: {old_qty:.8f} BTC
New: {new_qty:.8f} BTC
Diff: {new_qty - old_qty:.8f} BTC

Position updated successfully!"""
                else:
                    msg = f"""✅ BALANCE IN SYNC

Current: {new_qty:.8f} BTC
No changes needed."""
                
                send_message(chat_id, msg)
            else:
                send_message(chat_id, data.get('message', 'Sync failed'))
        else:
            send_message(chat_id, f"Sync failed: {response.status_code}")
    except Exception as e:
        send_message(chat_id, f"Sync error: {e}")


def handle_fix(chat_id):
    """Handle /fix command - auto-fix common issues"""
    try:
        send_message(chat_id, "Running auto-fix routines...")
        
        # Run health check which includes auto-fixes
        response = requests.post(f"{DASHBOARD_URL}/api/health", timeout=30)
        
        # Also run balance sync
        sync_response = requests.post(f"{DASHBOARD_URL}/api/sync_balance", timeout=15)
        
        fixes = []
        
        if response.status_code == 200:
            data = response.json()
            health_fixes = data.get('health', {}).get('fixes_applied', [])
            fixes.extend(health_fixes)
        
        if sync_response.status_code == 200:
            sync_data = sync_response.json()
            if sync_data.get('old_quantity') != sync_data.get('new_quantity'):
                fixes.append(f"Synced balance: {sync_data.get('new_quantity'):.8f} BTC")
        
        if fixes:
            fix_list = '\n'.join([f"  - {f}" for f in fixes])
            msg = f"""🔧 AUTO-FIX COMPLETE

Fixes Applied: {len(fixes)}
{fix_list}

Bot should be healthy now!"""
        else:
            msg = """✅ AUTO-FIX COMPLETE

No issues found - everything looks good!"""
        
        send_message(chat_id, msg)
        
    except Exception as e:
        send_message(chat_id, f"Auto-fix error: {e}")


# =============================================================================
# AUTONOMOUS AI COMMANDS
# =============================================================================

def handle_ai_status(chat_id):
    """Handle /ai command - get AI status"""
    try:
        response = requests.get(f"{DASHBOARD_URL}/api/ai/status", timeout=15)
        if response.status_code == 200:
            data = response.json()
            ai = data.get('ai', {})
            awareness = ai.get('awareness', {})
            
            msg = f"""🤖 META AI STATUS

State: {ai.get('state', 'unknown').upper()}
Version: {ai.get('version', 1)}
Autonomous: {'Running' if ai.get('autonomous_running') else 'Stopped'}

📊 AWARENESS
Win Rate: {awareness.get('win_rate', 0)}%
Total Trades: {awareness.get('total_trades', 0)}
Trades Learned: {awareness.get('trades_learned', 0)}
Current Streak: {awareness.get('current_streak', 0)}

⏰ NEXT ACTIONS
Analysis: {ai.get('schedules', {}).get('analysis', 'now')}
Optimization: {ai.get('schedules', {}).get('optimization', 'now')}
Evolution: {ai.get('schedules', {}).get('evolution', 'now')}

Improvements Made: {ai.get('improvements', 0)}"""
            
            send_message(chat_id, msg)
        else:
            send_message(chat_id, f"Error getting AI status: {response.status_code}")
    except Exception as e:
        send_message(chat_id, f"AI status error: {e}")


def handle_ai_think(chat_id):
    """Handle /think command - trigger AI thinking"""
    try:
        send_message(chat_id, "Triggering AI thinking cycle...")
        
        response = requests.post(f"{DASHBOARD_URL}/api/ai/think", timeout=60)
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            
            msg = f"""🧠 AI THINKING COMPLETE

Action Taken: {result.get('action_taken', 'unknown')}
Status: {result.get('result', {}).get('status', 'unknown')}

📊 Current Awareness:
Win Rate: {result.get('awareness', {}).get('win_rate', 0)}%
Trades Learned: {result.get('awareness', {}).get('trades_learned', 0)}"""
            
            send_message(chat_id, msg)
        else:
            send_message(chat_id, f"Think error: {response.status_code}")
    except Exception as e:
        send_message(chat_id, f"Think error: {e}")


def handle_ai_goals(chat_id):
    """Handle /goals command - view AI goals"""
    try:
        response = requests.get(f"{DASHBOARD_URL}/api/ai/goals", timeout=15)
        if response.status_code == 200:
            data = response.json()
            goals = data.get('goals', [])
            
            msg = "🎯 AI GOALS\n\n"
            for goal in goals:
                status = "✅" if goal.get('achieved') else "⏳"
                progress = goal.get('progress_percent', 0)
                msg += f"{status} {goal.get('name', 'Unknown')}\n"
                msg += f"   Target: {goal.get('target')} | Current: {goal.get('current', 0):.1f}\n"
                msg += f"   Progress: {progress:.0f}%\n\n"
            
            send_message(chat_id, msg)
        else:
            send_message(chat_id, f"Goals error: {response.status_code}")
    except Exception as e:
        send_message(chat_id, f"Goals error: {e}")


def handle_ai_evolve(chat_id):
    """Handle /evolve command - run strategy evolution"""
    try:
        send_message(chat_id, "Starting strategy evolution... This may take a few minutes.")
        
        response = requests.post(f"{DASHBOARD_URL}/api/ai/evolve", timeout=300)
        if response.status_code == 200:
            data = response.json()
            
            if data.get('evolved_params'):
                msg = """🧬 EVOLUTION COMPLETE

New optimized strategy found!
Parameters have been updated.

The AI has evolved a better trading strategy."""
            else:
                msg = """🧬 EVOLUTION COMPLETE

No better strategy found.
Current strategy is performing well."""
            
            send_message(chat_id, msg)
        else:
            send_message(chat_id, f"Evolution error: {response.status_code}")
    except Exception as e:
        send_message(chat_id, f"Evolution error: {e}")


def handle_ai_optimize(chat_id):
    """Handle /optimize command - run parameter optimization"""
    try:
        send_message(chat_id, "Running parameter optimization...")
        
        response = requests.post(f"{DASHBOARD_URL}/api/ai/optimize", timeout=60)
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            adjustments = result.get('adjustments', [])
            
            if adjustments:
                msg = f"""⚙️ OPTIMIZATION COMPLETE

Adjustments Made: {len(adjustments)}

"""
                for adj in adjustments[:5]:
                    msg += f"• {adj.get('param')}: {adj.get('old_value'):.4f} → {adj.get('new_value'):.4f}\n"
            else:
                msg = """⚙️ OPTIMIZATION COMPLETE

No adjustments needed.
Current parameters are optimal."""
            
            send_message(chat_id, msg)
        else:
            send_message(chat_id, f"Optimize error: {response.status_code}")
    except Exception as e:
        send_message(chat_id, f"Optimize error: {e}")


# =============================================================================
# HELP COMMAND
# =============================================================================

def handle_help(chat_id):
    """Handle /help command"""
    msg = """🤖 TRADING BOT COMMANDS

📊 QUERY COMMANDS
━━━━━━━━━━━━━━━━━━━━
/status - Bot status & price
/profit - Total profit report
/trades - Recent trade history
/balance - Account balance
/position - Current position
/stats - Trading statistics

🎮 CONTROL COMMANDS
━━━━━━━━━━━━━━━━━━━━
/start_trading - Start bot
/stop_trading - Stop bot
/pause - Pause trading
/resume - Resume trading

⚙️ SETTINGS COMMANDS
━━━━━━━━━━━━━━━━━━━━
/settings - Show settings
/setprofit X - Set profit %
/setstop X - Set stop loss %
/settrade X - Set trade $

🌍 MARKET COMMANDS
━━━━━━━━━━━━━━━━━━━━
/market - Market overview
/fear - Fear & Greed
/dominance - BTC dominance
/funding - Funding rate
/orderbook - Order book

🔧 MAINTENANCE
━━━━━━━━━━━━━━━━━━━━
/health - Run health check
/sync - Sync balance with Binance
/fix - Auto-fix common issues

🤖 AUTONOMOUS AI
━━━━━━━━━━━━━━━━━━━━
/ai - AI status & awareness
/think - Trigger AI thinking
/goals - View AI goals
/evolve - Run strategy evolution
/optimize - Run optimization

❓ /help - This help"""
    
    send_message(chat_id, msg)


def handle_message(message):
    """Process incoming message"""
    chat_id = message.get('chat', {}).get('id')
    text = message.get('text', '').strip().lower()
    
    # Only respond to our chat
    if str(chat_id) != TELEGRAM_CHAT_ID:
        return
    
    # === QUERY COMMANDS ===
    if text == '/status' or text == 'status':
        handle_status(chat_id)
    elif text == '/profit' or text == 'profit':
        handle_profit(chat_id)
    elif text == '/trades' or text == 'trades':
        handle_trades(chat_id)
    elif text == '/balance' or text == 'balance':
        handle_balance(chat_id)
    elif text == '/position' or text == 'position':
        handle_position(chat_id)
    elif text == '/stats' or text == 'stats':
        handle_stats(chat_id)
    
    # === CONTROL COMMANDS ===
    elif text == '/start_trading' or text == 'start_trading':
        handle_start_trading(chat_id)
    elif text == '/stop_trading' or text == 'stop_trading':
        handle_stop_trading(chat_id)
    elif text == '/pause' or text == 'pause':
        handle_pause(chat_id)
    elif text == '/resume' or text == 'resume':
        handle_resume(chat_id)
    
    # === SETTINGS COMMANDS ===
    elif text == '/settings' or text == 'settings':
        handle_settings(chat_id)
    elif text.startswith('/setprofit') or text.startswith('setprofit'):
        try:
            value = float(text.split()[1])
            handle_setprofit(chat_id, value)
        except (IndexError, ValueError):
            send_message(chat_id, "Usage: /setprofit 2\n(Sets profit target to 2%)")
    elif text.startswith('/setstop') or text.startswith('setstop'):
        try:
            value = float(text.split()[1])
            handle_setstop(chat_id, value)
        except (IndexError, ValueError):
            send_message(chat_id, "Usage: /setstop 1\n(Sets stop loss to 1%)")
    elif text.startswith('/settrade') or text.startswith('settrade'):
        try:
            value = float(text.split()[1])
            handle_settrade(chat_id, value)
        except (IndexError, ValueError):
            send_message(chat_id, "Usage: /settrade 50\n(Sets trade amount to $50)")
    
    # === MARKET COMMANDS ===
    elif text == '/market' or text == 'market':
        handle_market(chat_id)
    elif text == '/fear' or text == 'fear':
        handle_fear(chat_id)
    elif text == '/dominance' or text == 'dominance':
        handle_dominance(chat_id)
    elif text == '/funding' or text == 'funding':
        handle_funding(chat_id)
    elif text == '/orderbook' or text == 'orderbook':
        handle_orderbook(chat_id)
    
    # === MAINTENANCE COMMANDS ===
    elif text == '/health' or text == 'health':
        handle_health(chat_id)
    elif text == '/sync' or text == 'sync':
        handle_sync(chat_id)
    elif text == '/fix' or text == 'fix':
        handle_fix(chat_id)
    
    # === AUTONOMOUS AI COMMANDS ===
    elif text == '/ai' or text == 'ai':
        handle_ai_status(chat_id)
    elif text == '/think' or text == 'think':
        handle_ai_think(chat_id)
    elif text == '/goals' or text == 'goals':
        handle_ai_goals(chat_id)
    elif text == '/evolve' or text == 'evolve':
        handle_ai_evolve(chat_id)
    elif text == '/optimize' or text == 'optimize':
        handle_ai_optimize(chat_id)
    
    # === HELP ===
    elif text == '/help' or text == 'help' or text == '/start':
        handle_help(chat_id)
    elif text.startswith('/'):
        send_message(chat_id, "Unknown command. Send /help for available commands.")


def poll_updates():
    """Poll for new messages"""
    global last_update_id
    
    try:
        url = f"{BASE_URL}/getUpdates"
        params = {"offset": last_update_id + 1, "timeout": 30}
        response = requests.get(url, params=params, timeout=35)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                for update in data.get('result', []):
                    last_update_id = update['update_id']
                    if 'message' in update:
                        handle_message(update['message'])
    except Exception as e:
        logger.error("Polling error: %s", e)
        time.sleep(5)


def run_bot():
    """Run the Telegram bot in background"""
    global last_hourly_report, last_daily_report, last_trade_count
    
    # Initialize timestamps
    last_hourly_report = time.time()
    last_daily_report = time.time()
    last_trade_count = calculate_stats()['total_trades']
    
    logger.info("Telegram bot started - send /help for commands (Query, Control, Settings, Market, Scheduled Reports)")
    
    while True:
        poll_updates()
        check_scheduled_reports()
        time.sleep(1)


def start_bot_thread():
    """Start bot in background thread"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram bot not configured")
        return None
    
    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    logger.info("Starting Telegram command bot...")
    run_bot()
