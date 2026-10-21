@echo off
wsl.exe -d Ubuntu-22.04 --cd "%~dp0." -- bash ./local-dev/argus-menu.sh
if errorlevel 1 pause
