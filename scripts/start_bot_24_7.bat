@echo off
title AI Trading Bot 24/7
cd /d "%~dp0.."

:loop
echo [%date% %time%] Starting Trading Bot (dashboard)...
python dashboard.py
set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] Bot exited with code %EXITCODE%. Restarting in 10 seconds...
timeout /t 10 /nobreak >nul
goto loop
