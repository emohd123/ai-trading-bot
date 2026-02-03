@echo off
echo Stopping Trading Bot...
taskkill /F /IM pythonw.exe 2>nul
taskkill /F /IM python.exe 2>nul
echo Bot stopped!
pause
