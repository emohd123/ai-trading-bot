"""
Telegram Notifier for Trading Bot
Sends real-time trade notifications to your Telegram

SETUP INSTRUCTIONS:
1. Open Telegram and search for @BotFather
2. Send /newbot and follow the prompts to create a bot
3. Copy the API token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
4. Search for your new bot and click START
5. Search for @userinfobot and send /start to get your Chat ID
6. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to your .env file
"""

import logging
import requests

logger = logging.getLogger(__name__)
import threading
from datetime import datetime
from typing import Optional
import os
import time


class TelegramNotifier:
    # Track last status update time for periodic updates
    _last_status_time = 0
    _STATUS_INTERVAL = 1800  # Send status every 30 minutes
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize Telegram notifier

        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID from @userinfobot
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID', '')
        self.enabled = bool(self.bot_token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        if self.enabled:
            logger.info("Telegram notifications enabled (Chat ID: %s...)", self.chat_id[:4])
        else:
            logger.info("Telegram notifications disabled (no token/chat_id configured)")

    def _send_async(self, message: str, parse_mode: str = "HTML"):
        """Send message in background thread to not block the bot"""
        def send():
            try:
                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True
                }
                response = requests.post(url, json=payload, timeout=10)
                if not response.ok:
                    logger.warning("Telegram send failed: %s", response.text)
            except Exception as e:
                logger.warning("Telegram error: %s", e)

        if self.enabled:
            threading.Thread(target=send, daemon=True).start()

    def send(self, message: str):
        """Send a plain text message"""
        self._send_async(message)

    def notify_buy(self, price: float, quantity: float, amount_usdt: float,
                   regime: str = "unknown", confidence: str = "Medium",
                   total_profit: float = 0, total_trades: int = 0, win_rate: float = 0):
        """Send notification when bot opens a position"""
        profit_line = f"\n💎 <b>Total Profit So Far:</b> ${total_profit:+,.2f} | Trades: {total_trades} | Win Rate: {win_rate:.1f}%\n"
        message = f"""
🟢 <b>BUY ORDER EXECUTED</b>

💰 <b>Price:</b> ${price:,.2f}
📊 <b>Amount:</b> ${amount_usdt:,.2f} USDT
🪙 <b>Quantity:</b> {quantity:.6f} BTC

📈 <b>Regime:</b> {regime}
🎯 <b>Confidence:</b> {confidence}
{profit_line}
🕐 <b>Time:</b> {datetime.now().strftime("%H:%M:%S")}
"""
        self._send_async(message)

    def notify_sell(self, price: float, quantity: float, amount_usdt: float,
                    pnl: float, pnl_percent: float, exit_type: str = "signal",
                    total_profit: float = 0, total_trades: int = 0, win_rate: float = 0):
        """Send notification when bot closes a position"""
        # Emoji based on profit/loss
        if pnl >= 0:
            emoji = "🟢" if pnl_percent >= 1 else "✅"
            status = "PROFIT"
        else:
            emoji = "🔴" if pnl_percent <= -2 else "⚠️"
            status = "LOSS"

        # Exit type display
        exit_display = {
            "min_profit": "💰 Min Profit Target",
            "profit_target": "🎯 Profit Target",
            "stop_loss": "🛑 Stop Loss",
            "trailing_stop": "📉 Trailing Stop",
            "ai_signal": "🤖 AI Signal",
            "divergence_exit": "📊 Divergence Exit",
            "manual": "👤 Manual"
        }.get(exit_type, exit_type)

        message = f"""
{emoji} <b>SELL ORDER - {status}</b>

💰 <b>Price:</b> ${price:,.2f}
📊 <b>Amount:</b> ${amount_usdt:,.2f} USDT
🪙 <b>Quantity:</b> {quantity:.6f} BTC

{"📈" if pnl >= 0 else "📉"} <b>This Trade:</b> ${pnl:+,.2f} ({pnl_percent:+.2f}%)
🚪 <b>Exit Type:</b> {exit_display}

💎 <b>Total Profit So Far:</b> ${total_profit:+,.2f}
📊 <b>Total Trades:</b> {total_trades} | Win Rate: {win_rate:.1f}%
🕐 <b>Time:</b> {datetime.now().strftime("%H:%M:%S")}
"""
        self._send_async(message)

    def notify_status(self, current_price: float, balance_usdt: float,
                      total_value: float, position: Optional[dict] = None,
                      pnl_percent: float = 0, regime: str = "unknown",
                      total_profit: float = 0, total_trades: int = 0, win_rate: float = 0):
        """Send periodic status update"""
        if position:
            position_text = f"""
📍 <b>IN POSITION</b>
   Entry: ${position.get('entry_price', 0):,.2f}
   Current P/L: {pnl_percent:+.2f}%
"""
        else:
            position_text = "📍 <b>NO POSITION</b> - Waiting for signal"

        message = f"""
📊 <b>BOT STATUS UPDATE</b>

💵 <b>BTC Price:</b> ${current_price:,.2f}
💰 <b>Balance:</b> ${balance_usdt:,.2f} USDT
📈 <b>Total Value:</b> ${total_value:,.2f}
🌊 <b>Regime:</b> {regime}

{position_text}

💎 <b>Total Profit So Far:</b> ${total_profit:+,.2f}
📊 <b>Trades:</b> {total_trades} | Win Rate: {win_rate:.1f}%
🕐 <b>Time:</b> {datetime.now().strftime("%H:%M:%S")}
"""
        self._send_async(message)

    def notify_error(self, error_message: str):
        """Send error notification"""
        message = f"""
🚨 <b>BOT ERROR</b>

❌ {error_message}

🕐 <b>Time:</b> {datetime.now().strftime("%H:%M:%S")}
"""
        self._send_async(message)

    def notify_start(self, total_profit: float = 0, total_trades: int = 0, win_rate: float = 0):
        """Send notification when bot starts"""
        profit_line = f"\n💎 <b>Total Profit So Far:</b> ${total_profit:+,.2f} | Trades: {total_trades} | Win Rate: {win_rate:.1f}%\n"
        message = f"""
🚀 <b>TRADING BOT STARTED</b>

✅ Bot is now running in LIVE mode
📊 Monitoring BTC/USDT
{profit_line}
🕐 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

You will receive notifications for:
• Buy/Sell orders
• Profit/Loss updates
• Errors and warnings
"""
        self._send_async(message)

    def notify_stop(self, total_profit: float = 0, total_trades: int = 0, win_rate: float = 0):
        """Send notification when bot stops"""
        profit_line = f"\n💎 <b>Total Profit So Far:</b> ${total_profit:+,.2f} | Trades: {total_trades} | Win Rate: {win_rate:.1f}%\n"
        message = f"""
⛔ <b>TRADING BOT STOPPED</b>
{profit_line}
🕐 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        self._send_async(message)

    def notify_signal_detected(self, signal_type: str, score: float, regime: str,
                                confluence: int, reasons: list):
        """Send notification when a trading signal is detected"""
        emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⏸️"

        reasons_text = "\n".join([f"  • {r}" for r in reasons[:5]]) if reasons else "  • No specific reasons"

        message = f"""
{emoji} <b>SIGNAL DETECTED: {signal_type}</b>

🎯 <b>AI Score:</b> {score:.2f}
🌊 <b>Regime:</b> {regime}
📊 <b>Confluence:</b> {confluence}/5 indicators

<b>Analysis:</b>
{reasons_text}

🕐 {datetime.now().strftime("%H:%M:%S")}
"""
        self._send_async(message)

    def notify_position_update(self, entry_price: float, current_price: float,
                                pnl_percent: float, highest_price: float = None,
                                trailing_stop: float = None,
                                total_profit: float = 0, total_trades: int = 0, win_rate: float = 0):
        """Send position monitoring update"""
        direction = "📈" if pnl_percent >= 0 else "📉"
        status = "PROFIT" if pnl_percent >= 0 else "LOSS"

        trailing_text = ""
        if trailing_stop:
            trailing_text = f"\n🛡️ <b>Trailing Stop:</b> ${trailing_stop:,.2f}"
        if highest_price:
            trailing_text += f"\n🔝 <b>Highest:</b> ${highest_price:,.2f}"
        profit_line = f"\n💎 <b>Total Profit So Far:</b> ${total_profit:+,.2f} | Trades: {total_trades} | Win Rate: {win_rate:.1f}%"

        message = f"""
{direction} <b>POSITION UPDATE - {status}</b>

💵 <b>Entry:</b> ${entry_price:,.2f}
💰 <b>Current:</b> ${current_price:,.2f}
{direction} <b>P/L:</b> {pnl_percent:+.2f}%{trailing_text}
{profit_line}

🕐 {datetime.now().strftime("%H:%M:%S")}
"""
        self._send_async(message)

    def notify_market_change(self, old_regime: str, new_regime: str,
                              adx: float, atr_percent: float):
        """Send notification when market regime changes"""
        message = f"""
🔄 <b>MARKET REGIME CHANGE</b>

📊 <b>From:</b> {old_regime}
📈 <b>To:</b> {new_regime}

📉 <b>ADX:</b> {adx:.1f}
📊 <b>ATR:</b> {atr_percent:.2f}%

🕐 {datetime.now().strftime("%H:%M:%S")}
"""
        self._send_async(message)

    def notify_total_profit(self, total_profit: float, total_trades: int, 
                             winning_trades: int, losing_trades: int, 
                             win_rate: float, balance: float = 0):
        """Send dedicated total profit message"""
        if total_profit >= 100:
            header_emoji = "🎉"
            status = "EXCELLENT!"
        elif total_profit >= 0:
            header_emoji = "✅"
            status = "IN PROFIT"
        elif total_profit >= -50:
            header_emoji = "⚠️"
            status = "SMALL LOSS"
        else:
            header_emoji = "🔴"
            status = "IN LOSS"

        # Performance rating
        if win_rate >= 70:
            performance = "🔥 Excellent"
        elif win_rate >= 55:
            performance = "👍 Good"
        elif win_rate >= 45:
            performance = "📊 Average"
        else:
            performance = "📉 Needs Work"

        # Profit per trade
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        message = f"""{header_emoji} 𝗧𝗢𝗧𝗔𝗟 𝗣𝗥𝗢𝗙𝗜𝗧 𝗥𝗘𝗣𝗢𝗥𝗧 {header_emoji}

💎 Total Profit: ${total_profit:+,.2f}
📊 Status: {status}

━━━━━━━━━━━━━━━━━━━━

📈 𝗦𝘁𝗮𝘁𝗶𝘀𝘁𝗶𝗰𝘀:
   🔢 Total Trades: {total_trades}
   ✅ Winning: {winning_trades}
   ❌ Losing: {losing_trades}
   🎯 Win Rate: {win_rate:.1f}%
   💵 Avg/Trade: ${avg_profit:+,.2f}

━━━━━━━━━━━━━━━━━━━━

💰 Balance: ${balance:,.2f} USDT
📊 Performance: {performance}

🕐 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""
        # Send directly (not async) to ensure delivery
        if self.enabled:
            try:
                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "disable_web_page_preview": True
                }
                requests.post(url, json=payload, timeout=10)
            except Exception:
                pass

    def notify_daily_summary(self, total_trades: int, wins: int, losses: int,
                              total_profit: float, win_rate: float, balance: float):
        """Send daily trading summary"""
        emoji = "🎉" if total_profit >= 0 else "😔"

        message = f"""
{emoji} <b>DAILY TRADING SUMMARY</b>

📊 <b>Total Trades:</b> {total_trades}
✅ <b>Wins:</b> {wins}
❌ <b>Losses:</b> {losses}
🎯 <b>Win Rate:</b> {win_rate:.1f}%

💰 <b>Total P/L:</b> ${total_profit:+,.2f}
💵 <b>Balance:</b> ${balance:,.2f}

📅 {datetime.now().strftime("%Y-%m-%d")}
"""
        self._send_async(message)

    def notify_learning_update(self, indicator_accuracy: dict, win_streak: int,
                                loss_streak: int, total_trades: int):
        """Send AI learning status update"""
        # Find best and worst indicators
        sorted_indicators = sorted(indicator_accuracy.items(), key=lambda x: x[1], reverse=True)
        best = sorted_indicators[0] if sorted_indicators else ("N/A", 0)
        worst = sorted_indicators[-1] if sorted_indicators else ("N/A", 0)

        streak_text = f"🔥 Win Streak: {win_streak}" if win_streak > 0 else f"❄️ Loss Streak: {loss_streak}"

        message = f"""
🧠 <b>AI LEARNING UPDATE</b>

📚 <b>Total Trades Learned:</b> {total_trades}
{streak_text}

<b>Indicator Accuracy:</b>
✅ Best: {best[0]} ({best[1]*100:.0f}%)
⚠️ Worst: {worst[0]} ({worst[1]*100:.0f}%)

🕐 {datetime.now().strftime("%H:%M:%S")}
"""
        self._send_async(message)

    def should_send_periodic_status(self) -> bool:
        """Check if it's time to send a periodic status update"""
        current_time = time.time()
        if current_time - TelegramNotifier._last_status_time >= TelegramNotifier._STATUS_INTERVAL:
            TelegramNotifier._last_status_time = current_time
            return True
        return False

    def test_connection(self) -> bool:
        """Test if Telegram connection works"""
        if not self.enabled:
            return False
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            if response.ok:
                bot_info = response.json()
                bot_name = bot_info.get('result', {}).get('username', 'Unknown')
                logger.info("Telegram connected: @%s", bot_name)
                return True
            return False
        except Exception as e:
            logger.error("Telegram connection test failed: %s", e)
            return False


# Singleton instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the global notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def init_notifier(bot_token: str = None, chat_id: str = None) -> TelegramNotifier:
    """Initialize the notifier with credentials"""
    global _notifier
    _notifier = TelegramNotifier(bot_token, chat_id)
    return _notifier
