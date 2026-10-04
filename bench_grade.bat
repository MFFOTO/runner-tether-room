@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Grade Room speed test

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Not set up yet -- run setup.bat first.
    pause
    exit /b 1
)

set "IN=%~1"
if "%IN%"=="" set /p "IN=Folder with photos to test with: "
set "OUT=%~2"

set "PYTHONNOUSERSITE=1"
if "%OUT%"=="" (
    ".venv\Scripts\python.exe" grade_suite.py bench "%IN%"
) else (
    ".venv\Scripts\python.exe" grade_suite.py bench "%IN%" --out "%OUT%"
)

echo.
echo The result was also saved as bench_grade_result.txt in this folder.
pause
