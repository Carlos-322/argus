@echo off
wsl.exe -d Ubuntu-22.04 --cd "%~dp0." -- bash --rcfile ./local-dev/argus-env.sh -i
if errorlevel 1 pause
