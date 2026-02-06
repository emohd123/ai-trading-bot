#!/bin/bash
# Update Trading Bot on DigitalOcean Droplet
# Run this script on your Droplet after pushing changes to GitHub

echo "🔄 Updating Trading Bot from GitHub..."

# Navigate to bot directory (adjust path if different)
cd /root/bot || cd ~/bot || exit 1

# Stop the bot service if running
echo "⏸️  Stopping bot service..."
systemctl stop tradingbot 2>/dev/null || pkill -f "dashboard.py" || pkill -f "python.*dashboard"

# Backup current state (optional - keeps your trade history)
echo "💾 Creating backup..."
mkdir -p data/backups
cp -r data/*.json data/backups/ 2>/dev/null || true

# Pull latest changes from GitHub
echo "📥 Pulling latest code from GitHub..."
git fetch origin
git pull origin main

# If git pull fails, try resetting (WARNING: this will discard local changes)
# git reset --hard origin/main

# Activate virtual environment and update dependencies (if needed)
echo "📦 Checking dependencies..."
source venv/bin/activate
pip install -q --upgrade python-binance pandas numpy ta scikit-learn xgboost lightgbm flask flask-socketio eventlet python-dotenv colorama requests joblib

# Restart the bot service
echo "▶️  Starting bot service..."
systemctl start tradingbot 2>/dev/null || {
    echo "⚠️  Systemd service not found, starting manually..."
    nohup python dashboard.py > data/bot.log 2>&1 &
}

# Wait a moment for bot to start
sleep 3

# Check status
echo "✅ Update complete!"
echo ""
echo "📊 Bot status:"
systemctl status tradingbot --no-pager -l || ps aux | grep -E "dashboard|python.*bot" | grep -v grep

echo ""
echo "📝 View logs with: journalctl -u tradingbot -f"
echo "🌐 Dashboard: http://YOUR_SERVER_IP:5000"
