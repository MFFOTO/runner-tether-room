@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Tether Room

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Not set up yet -- run setup.bat first.
    pause
    exit /b 1
)
if not exist "runner_suite_core.py" (
    echo [ERROR] The crop engine is missing -- run setup.bat, or:
    echo         .venv\Scripts\python.exe fetch_deps.py
    pause
    exit /b 1
)
if not exist "settings_merged.json" (
    copy "settings_merged.example.json" "settings_merged.json" >nul
    echo Created settings_merged.json from the example.
)

set "PYTHONNOUSERSITE=1"
".venv\Scripts\python.exe" merged_ui.py settings_merged.json

echo.
echo Stopped.
pause
