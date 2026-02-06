#!/bin/bash
# Clean update script - stops all bots, updates code, restarts

echo "🛑 Stopping all bot processes..."
pkill -f "dashboard.py"
sleep 2

# Make sure they're stopped
pkill -9 -f "dashboard.py" 2>/dev/null
sleep 1

echo "📥 Updating code from GitHub..."
cd /root/bot

# Reset to match GitHub exactly (discard local changes)
git reset --hard origin/main
git pull origin main

echo "✅ Code updated!"
echo ""
echo "▶️  Starting bot..."
source venv/bin/activate
nohup python dashboard.py > data/bot.log 2>&1 &

sleep 3

echo ""
echo "📊 Bot status:"
if pgrep -f "dashboard.py" > /dev/null; then
    echo "✅ Bot is running!"
    ps aux | grep dashboard | grep -v grep
    echo ""
    echo "📝 View logs: tail -f /root/bot/data/bot.log"
    echo "🌐 Dashboard: http://188.166.184.170:5000"
else
    echo "❌ Bot failed to start. Check logs:"
    tail -20 /root/bot/data/bot.log
fi
