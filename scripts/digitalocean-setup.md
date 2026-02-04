# Run the Bot 24/7 on DigitalOcean

Use a **Droplet** (Ubuntu 22.04). About **$6/mo** for 1 GB RAM (enough for the bot).

---

## 1. Create a Droplet

1. Log in to [DigitalOcean](https://cloud.digitalocean.com).
2. **Create** → **Droplets**.
3. Choose **Ubuntu 22.04 (LTS)**.
4. Plan: **Basic**, **$6/mo** (1 GB RAM / 1 CPU) or **$12/mo** (2 GB if you want more headroom).
5. Datacenter: pick one close to you (e.g. NYC, London, Singapore).
6. Authentication: **SSH key** (recommended) or **Password**.
7. Create Droplet. Note the **IP address** (e.g. `164.92.xxx.xxx`).

---

## 2. Connect and install

SSH in (replace with your Droplet IP):

```bash
ssh root@YOUR_DROPLET_IP
```

Run these commands **one block at a time**:

```bash
# Update system
apt update && apt upgrade -y

# Install Python 3.10+ and pip
apt install -y python3 python3-pip python3-venv git

# Create app directory and clone your repo
cd /root
git clone https://github.com/emohd123/ai-trading-bot.git bot
cd bot

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env from example (you will edit this next)
cp .env.example .env
```

---

## 3. Add your API keys

Edit `.env` on the server (use nano or paste from your local machine):

```bash
nano .env
```

Set at least:

- `BINANCE_API_KEY=your_key`
- `BINANCE_API_SECRET=your_secret`
- `USE_TESTNET=False` (or `True` for testnet)

Save: **Ctrl+O**, Enter, then **Ctrl+X**.

(Or from your PC: `scp .env root@YOUR_DROPLET_IP:/root/bot/.env` — only if your local `.env` has the same keys you want on the server.)

---

## 4. Install the 24/7 service

Still on the Droplet, in `/root/bot`:

```bash
# Copy service file and set paths for root user
sudo sed 's|YOUR_USER|root|g; s|/home/root/bot|/root/bot|g' scripts/bot.service > /tmp/bot.service
sudo mv /tmp/bot.service /etc/systemd/system/bot.service

# Use the venv Python
sudo sed -i 's|ExecStart=/usr/bin/python3 dashboard.py|ExecStart=/root/bot/venv/bin/python dashboard.py|' /etc/systemd/system/bot.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable bot
sudo systemctl start bot
sudo systemctl status bot
```

You should see **active (running)** in green.

---

## 5. Useful commands

| Task | Command |
|------|--------|
| Check status | `sudo systemctl status bot` |
| View live logs | `journalctl -u bot -f` |
| Stop bot | `sudo systemctl stop bot` |
| Start bot | `sudo systemctl start bot` |
| Restart bot | `sudo systemctl restart bot` |

---

## 6. Access the dashboard (optional)

The bot runs the Flask dashboard on **port 5000**. To open it in a browser:

1. **Option A – SSH tunnel (no open port):**  
   On your PC: `ssh -L 5000:localhost:5000 root@YOUR_DROPLET_IP`  
   Then open **http://localhost:5000** in your browser.

2. **Option B – Open port on DigitalOcean:**  
   In the DO control panel: **Networking** → **Firewall** → Add rule: **Inbound** TCP **5000**.  
   Then open **http://YOUR_DROPLET_IP:5000** (only if you’re comfortable exposing the dashboard; use a strong password or restrict IP if the dashboard has auth).

---

## 7. Updating the bot later

```bash
cd /root/bot
git pull origin main
sudo systemctl restart bot
```

---

## Summary

- Droplet: Ubuntu 22.04, $6/mo.
- Bot in `/root/bot`, runs as systemd service `bot`, restarts on crash and on reboot.
- API keys in `/root/bot/.env`; never commit `.env` to git.

**If you use a non-root user** (e.g. `ubuntu`): replace `root` with that username and use `/home/ubuntu/bot` instead of `/root/bot` in the commands above.
