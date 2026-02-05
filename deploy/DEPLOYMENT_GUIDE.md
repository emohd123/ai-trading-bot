# Trading Bot Deployment Guide - DigitalOcean

## Step 1: Create DigitalOcean Account

1. Go to: https://www.digitalocean.com
2. Click "Sign Up"
3. Create account with email or GitHub
4. Add payment method (credit card or PayPal)
5. You'll get $200 free credit for 60 days (new users)

## Step 2: Create a Droplet (VPS)

1. Click **"Create"** → **"Droplets"**
2. Choose settings:
   - **Region**: Choose closest to Binance servers (Singapore, London, or New York)
   - **Image**: Ubuntu 22.04 LTS
   - **Size**: Basic → Regular → **$6/month** (1GB RAM, 1 vCPU, 25GB SSD)
   - **Authentication**: Choose **SSH Key** (more secure) or **Password**
   - **Hostname**: `trading-bot`
3. Click **"Create Droplet"**
4. Wait 1-2 minutes for it to be ready
5. Copy the **IP Address** shown

## Step 3: Connect to Your Server

### On Windows (PowerShell):
```powershell
ssh root@YOUR_SERVER_IP
```

### On Windows (if using password):
- Download PuTTY: https://www.putty.org/
- Enter your server IP and connect

## Step 4: Run Setup Script

Once connected to your server, run:

```bash
# Download and run setup script
curl -sSL https://raw.githubusercontent.com/YOUR_REPO/setup_vps.sh | bash

# Or manually copy and run:
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git
mkdir -p /root/bot
cd /root/bot
python3 -m venv venv
source venv/bin/activate
pip install python-binance pandas numpy ta scikit-learn xgboost lightgbm flask flask-socketio eventlet python-dotenv colorama requests joblib
```

## Step 5: Upload Bot Files

### Option A: Using SCP (from your Windows laptop)
Open PowerShell on your laptop:

```powershell
# Upload entire bot folder
scp -r "C:\Users\cactu\OneDrive\Desktop\app\bot\*" root@YOUR_SERVER_IP:/root/bot/
```

### Option B: Using FileZilla (GUI)
1. Download FileZilla: https://filezilla-project.org/
2. Connect: Host=YOUR_SERVER_IP, Username=root, Port=22
3. Drag files from left (local) to right (server /root/bot/)

## Step 6: Create .env File on Server

```bash
cd /root/bot
nano .env
```

Add your API keys:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
USE_TESTNET=False
```

Save: Press `Ctrl+X`, then `Y`, then `Enter`

## Step 7: Start the Bot Service

```bash
# Reload systemd
systemctl daemon-reload

# Enable auto-start on boot
systemctl enable tradingbot

# Start the bot
systemctl start tradingbot

# Check status
systemctl status tradingbot
```

## Step 8: Access Dashboard

Open in your browser:
```
http://YOUR_SERVER_IP:5000
```

## Useful Commands

```bash
# View live logs
journalctl -u tradingbot -f

# Restart bot
systemctl restart tradingbot

# Stop bot
systemctl stop tradingbot

# View bot logs
tail -f /root/bot/data/bot.log

# Check if bot is running
systemctl status tradingbot
```

## Security (Optional but Recommended)

### Add SSL with Nginx (HTTPS)
```bash
apt install nginx certbot python3-certbot-nginx -y

# Configure nginx as reverse proxy
cat > /etc/nginx/sites-available/tradingbot << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
EOF

ln -s /etc/nginx/sites-available/tradingbot /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx

# Get SSL certificate (if you have a domain)
certbot --nginx -d your-domain.com
```

### Restrict Dashboard Access (IP whitelist)
```bash
ufw delete allow 5000
ufw allow from YOUR_HOME_IP to any port 5000
```

## Troubleshooting

### Bot won't start
```bash
# Check logs for errors
journalctl -u tradingbot -n 50

# Try running manually to see errors
cd /root/bot
source venv/bin/activate
python dashboard.py
```

### Can't connect to dashboard
```bash
# Check if port 5000 is open
ufw status

# Check if bot is listening
netstat -tlnp | grep 5000
```

### API connection issues
- Make sure .env file has correct keys
- Check if Binance API is accessible from server region
- Verify API key permissions (enable Spot trading)
