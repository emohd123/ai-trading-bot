@echo off
:: Start Trading Bot in Minimized Window
cd /d "%~dp0.."
start /min "Trading Bot" python dashboard.py
echo Bot started in minimized window!
echo Dashboard: http://localhost:5000
