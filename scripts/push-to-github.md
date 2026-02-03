# Push this project to GitHub

The repo is committed locally. To push:

1. **Create a new repository on GitHub**
   - Go to https://github.com/new
   - Name it (e.g. `trading-bot` or `ai-trading-bot`)
   - Leave "Add a README" **unchecked** (we already have one)
   - Create repository

2. **Add your repo as remote and push** (replace `YOUR_USERNAME` and `YOUR_REPO`):

   ```bash
   cd "c:\Users\cactu\OneDrive\Desktop\app\bot"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

   If you use SSH:
   ```bash
   git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

3. **Optional – set your Git identity** (so future commits show your name):
   ```bash
   git config --global user.email "your@email.com"
   git config --global user.name "Your Name"
   ```

**Note:** `.env` is not in the repo (in `.gitignore`). On a new machine or VPS, copy `.env.example` to `.env` and fill in your API keys.
