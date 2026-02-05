#!/bin/bash
# ============================================
# Trading Bot VPS Setup Script
# Run this on your new Ubuntu VPS
# ============================================

set -e  # Exit on error

echo "=========================================="
echo "  Trading Bot VPS Setup"
echo "=========================================="

# Update system
echo "[1/7] Updating system..."
apt update && apt upgrade -y

# Install Python and dependencies
echo "[2/7] Installing Python and dependencies..."
apt install -y python3 python3-pip python3-venv git curl wget unzip

# Create bot directory
echo "[3/7] Setting up bot directory..."
mkdir -p /root/bot
cd /root/bot

# Create virtual environment
echo "[4/7] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "[5/7] Installing Python packages..."
pip install --upgrade pip
pip install python-binance pandas numpy ta scikit-learn xgboost lightgbm flask flask-socketio eventlet python-dotenv colorama requests joblib

# Create systemd service
echo "[6/7] Creating systemd service..."
cat > /etc/systemd/system/tradingbot.service << 'EOF'
[Unit]
Description=AI Trading Bot Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/bot
Environment=PATH=/root/bot/venv/bin:/usr/bin:/bin
ExecStart=/root/bot/venv/bin/python dashboard.py
Restart=always
RestartSec=10
StandardOutput=append:/root/bot/data/bot.log
StandardError=append:/root/bot/data/bot_error.log

[Install]
WantedBy=multi-user.target
EOF

# Create data directory
mkdir -p /root/bot/data

# Setup firewall
echo "[7/7] Configuring firewall..."
ufw allow 22/tcp    # SSH
ufw allow 5000/tcp  # Dashboard
ufw --force enable

echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your bot files to /root/bot/"
echo "2. Create /root/bot/.env with your API keys"
echo "3. Run: systemctl daemon-reload"
echo "4. Run: systemctl enable tradingbot"
echo "5. Run: systemctl start tradingbot"
echo ""
echo "Dashboard will be at: http://YOUR_SERVER_IP:5000"
echo "=========================================="
