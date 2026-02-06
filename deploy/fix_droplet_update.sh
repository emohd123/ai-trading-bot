#!/bin/bash
# Fix Droplet Update - Resolve conflicts and restart bot

echo "🔧 Fixing Droplet update conflicts..."

cd /root/bot || exit 1

# Option 1: Discard local changes and match GitHub exactly
echo "📥 Resetting to match GitHub (discarding local changes)..."
git reset --hard origin/main
git pull origin main

# Check if systemd service exists
if systemctl list-unit-files | grep -q tradingbot.service; then
    echo "▶️  Restarting systemd service..."
    systemctl restart tradingbot
    systemctl status tradingbot --no-pager -l
else
    echo "⚠️  Systemd service not found. Checking how bot is running..."
    
    # Check if bot is running as a process
    if pgrep -f "dashboard.py" > /dev/null; then
        echo "🔄 Bot is running as process. Stopping..."
        pkill -f "dashboard.py"
        sleep 2
    fi
    
    # Start bot manually
    echo "▶️  Starting bot manually..."
    cd /root/bot
    source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate
    nohup python dashboard.py > data/bot.log 2>&1 &
    
    sleep 3
    if pgrep -f "dashboard.py" > /dev/null; then
        echo "✅ Bot started successfully!"
        echo "📝 View logs: tail -f /root/bot/data/bot.log"
    else
        echo "❌ Bot failed to start. Check logs: tail -f /root/bot/data/bot.log"
    fi
fi

echo ""
echo "✅ Update complete!"
echo "🌐 Dashboard: http://$(hostname -I | awk '{print $1}'):5000"
