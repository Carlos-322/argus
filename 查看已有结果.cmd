@echo off
wsl.exe -d Ubuntu-22.04 --cd "%~dp0." -- bash ./local-dev/view-results.sh
if errorlevel 1 pause
