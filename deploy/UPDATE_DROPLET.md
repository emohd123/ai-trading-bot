# Update Droplet from GitHub

## Quick Update (Recommended)

### Step 1: SSH into your Droplet

Open PowerShell on your Windows machine and run:

```powershell
ssh root@YOUR_DROPLET_IP
```

Replace `YOUR_DROPLET_IP` with your actual Droplet IP address.

### Step 2: Navigate to bot directory and pull updates

Once connected to your Droplet, run:

```bash
cd /root/bot
git pull origin main
```

### Step 3: Restart the bot

```bash
systemctl restart tradingbot
```

### Step 4: Verify it's running

```bash
systemctl status tradingbot
```

---

## Alternative: Using the Update Script

If you've uploaded the `update_droplet.sh` script to your Droplet:

```bash
cd /root/bot
chmod +x deploy/update_droplet.sh
./deploy/update_droplet.sh
```

---

## Manual Update Steps (Detailed)

### 1. Connect to Droplet
```bash
ssh root@YOUR_DROPLET_IP
```

### 2. Navigate to bot directory
```bash
cd /root/bot
# or if your bot is in a different location:
# cd ~/bot
```

### 3. Check current status
```bash
systemctl status tradingbot
# or check if process is running:
ps aux | grep dashboard
```

### 4. Stop the bot
```bash
systemctl stop tradingbot
# or if not using systemd:
pkill -f dashboard.py
```

### 5. Backup current state (optional but recommended)
```bash
mkdir -p data/backups
cp data/*.json data/backups/ 2>/dev/null || true
```

### 6. Pull latest code from GitHub
```bash
git fetch origin
git pull origin main
```

**If you have local changes that conflict:**
```bash
# Option 1: Stash local changes
git stash
git pull origin main
git stash pop

# Option 2: Discard local changes (WARNING: loses local modifications)
git reset --hard origin/main
git pull origin main
```

### 7. Update Python dependencies (if needed)
```bash
source venv/bin/activate
pip install --upgrade python-binance pandas numpy ta scikit-learn xgboost lightgbm flask flask-socketio eventlet python-dotenv colorama requests joblib
```

### 8. Restart the bot
```bash
systemctl start tradingbot
# or if not using systemd:
cd /root/bot
source venv/bin/activate
nohup python dashboard.py > data/bot.log 2>&1 &
```

### 9. Verify it's running
```bash
systemctl status tradingbot
# Check logs:
tail -f data/bot.log
# or:
journalctl -u tradingbot -f
```

---

## Troubleshooting

### Git pull fails with "permission denied"
```bash
# Check git remote
git remote -v

# If using HTTPS, you may need to set up credentials
# Or switch to SSH:
git remote set-url origin git@github.com:emohd123/ai-trading-bot.git
```

### Bot won't start after update
```bash
# Check for errors:
journalctl -u tradingbot -n 50

# Try running manually to see errors:
cd /root/bot
source venv/bin/activate
python dashboard.py
```

### Missing dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt  # if you have one
# or install individually:
pip install python-binance pandas numpy ta scikit-learn xgboost lightgbm flask flask-socketio eventlet python-dotenv
```

### Port 5000 already in use
```bash
# Find what's using port 5000:
lsof -i :5000
# Kill it:
kill -9 PID_NUMBER
# Then restart bot
```

---

## Quick Commands Reference

```bash
# View live logs
journalctl -u tradingbot -f

# Restart bot
systemctl restart tradingbot

# Stop bot
systemctl stop tradingbot

# Start bot
systemctl start tradingbot

# Check status
systemctl status tradingbot

# View bot logs file
tail -f /root/bot/data/bot.log

# Check if bot is running
ps aux | grep dashboard
```

---

## One-Liner Update (Copy & Paste)

```bash
cd /root/bot && git pull origin main && systemctl restart tradingbot && systemctl status tradingbot
```
