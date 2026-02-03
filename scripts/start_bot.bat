@echo off
echo Starting Trading Bot...
cd /d "%~dp0.."
start /min "AI Trading Bot" python dashboard.py
echo.
echo ========================================
echo   Bot started in MINIMIZED window!
echo   Dashboard: http://localhost:5000
echo ========================================
echo.
echo The bot is running in taskbar (minimized).
echo To stop: run stop_bot.bat or close the minimized window.
echo.
pause
