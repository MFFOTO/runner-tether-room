@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Tether Room - setup

echo ==========================================================
echo   Tether Room - setup
echo ==========================================================
echo.

rem ---- 1. Python -------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo         Install Python 3.10 or newer from python.org and re-run this.
    pause
    exit /b 1
)

rem ---- 2. virtual environment -----------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
set "PY=.venv\Scripts\python.exe"
set "PYTHONNOUSERSITE=1"

rem ---- 3. dependencies -------------------------------------------------
echo.
echo Installing dependencies ...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    pause
    exit /b 1
)

echo.
echo For an NVIDIA GPU, install the CUDA build of torch as well:
echo    .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
echo (CPU-only works too, just slower.)

rem ---- 4. crop engine + LUTs ------------------------------------------
echo.
echo Fetching the crop engine and LUTs ...
"%PY%" fetch_deps.py
if errorlevel 1 (
    echo [ERROR] Could not fetch runner_suite_core.py -- check your internet connection.
    pause
    exit /b 1
)

rem ---- 5. config -------------------------------------------------------
if not exist "settings_merged.json" (
    echo Creating settings_merged.json from the example ...
    copy "settings_merged.example.json" "settings_merged.json" >nul
)

echo.
echo ==========================================================
echo   Done. Edit settings_merged.json if you need to, then
echo   run:  run.bat
echo ==========================================================
pause
