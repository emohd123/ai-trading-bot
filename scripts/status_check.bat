@echo off
cd /d "%~dp0.."
python tests\status_check.py
pause
