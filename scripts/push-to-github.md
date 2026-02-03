# Push this project to GitHub

The repo is committed locally. To push:

1. **Create a new repository on GitHub**
   - Go to https://github.com/new
   - Name it (e.g. `ai-trading-bot`)
   - Leave "Add a README" **unchecked**
   - Create repository

2. **Run the push script** (it will ask for your username and repo name, then push):

   ```powershell
   cd "c:\Users\cactu\OneDrive\Desktop\app\bot"
   .\scripts\push-to-github.ps1
   ```

   Or do it manually:
   ```powershell
   git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

3. **Optional – set your Git identity:**
   ```powershell
   git config --global user.email "your@email.com"
   git config --global user.name "Your Name"
   ```

**Note:** `.env` is not in the repo. On a new machine or VPS, copy `.env.example` to `.env` and add your API keys.
