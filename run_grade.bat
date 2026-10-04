@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Grade Room

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Not set up yet -- run setup.bat first.
    pause
    exit /b 1
)
if not exist "settings_grade.json" (
    copy "settings_grade.example.json" "settings_grade.json" >nul
    echo Created settings_grade.json from the example.
)

set "PYTHONNOUSERSITE=1"
".venv\Scripts\python.exe" grade_ui.py settings_grade.json

echo.
echo Stopped.
pause
