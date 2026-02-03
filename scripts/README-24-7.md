# Running the Bot 24/7

Three practical options, from best to simplest.

---

## Option 1: VPS / Cloud (recommended for real 24/7)

Run the bot on a **Linux server** (e.g. $5–6/mo: DigitalOcean, Linode, Vultr, AWS Lightsail). Your PC can be off; the bot keeps running.

### Steps

1. Copy the project to the VPS (git clone or upload).
2. On the server, install Python 3.10+, then:
   ```bash
   cd /path/to/bot
   pip install -r requirements.txt
   cp .env.example .env   # edit .env with your API keys
   ```
3. Install and enable the systemd service:
   ```bash
   sudo cp scripts/bot.service /etc/systemd/system/
   # Edit the service file: set your actual path in WorkingDirectory= and ExecStart=
   sudo systemctl daemon-reload
   sudo systemctl enable bot    # start on boot
   sudo systemctl start bot     # start now
   sudo systemctl status bot    # check status
   ```
4. View logs: `journalctl -u bot -f`

**Pros:** Survives reboots, runs when your PC is off, stable.  
**Cons:** Small monthly cost, need to maintain a server.

---

## Option 2: This Windows PC (start on login + auto-restart)

Run the bot on your machine with **auto-restart if it crashes** and **start when you log in**.

### A. Start when Windows starts (Task Scheduler)

1. Open **Task Scheduler** (search “Task Scheduler” in Start).
2. **Create Task** (not “Basic”):
   - **General:** Name e.g. `AI Trading Bot`, “Run whether user is logged on or not” or “Run only when user is logged on” (your choice).
   - **Triggers:** New → “At log on” (or “At startup” if you use “Run whether user is logged on”).
   - **Actions:** New → Program: `C:\Users\cactu\OneDrive\Desktop\app\bot\scripts\start_bot_24_7.bat` (use your real path). Start in: `C:\Users\cactu\OneDrive\Desktop\app\bot`.
   - **Settings:** Optional: “Run task as soon as possible after a scheduled start is missed”.
3. Save. Test by running the task manually.

### B. Run with auto-restart (no scheduler)

Double-click **`start_bot_24_7.bat`** (or run it from a terminal). It starts the dashboard and **restarts it if it exits** (crash or update). Leave this window open (or minimize). For true 24/7 on this PC, also add the Task Scheduler step above so it starts after reboot.

---

## Option 3: Run hidden on this PC (current style)

Use your existing **`start_bot_hidden.vbs`**: no window, starts the bot once. If the process crashes or you reboot, you must start it again manually. Good for “run while I’m at the PC”, not for unattended 24/7 unless combined with Option 2 (scheduler + restart script).

---

## Summary

| Goal                         | Use this                          |
|-----------------------------|------------------------------------|
| Real 24/7, PC can be off    | **Option 1** (VPS + systemd)       |
| 24/7 on this PC, auto-start| **Option 2** (Task Scheduler + `start_bot_24_7.bat`) |
| Just restart on crash       | **Option 2B** (`start_bot_24_7.bat` only) |
| Run hidden, manual start   | **Option 3** (`start_bot_hidden.vbs`)     |

---

## Security notes (especially for VPS)

- Keep **.env** (API keys) out of git and backups you share.
- Prefer **testnet** until you’re happy with behavior: `USE_TESTNET=True` in .env.
- Restrict Binance API key to “Spot” and “Enable Reading” + “Enable Spot & Margin Trading” only; use IP whitelist if the exchange allows it.
