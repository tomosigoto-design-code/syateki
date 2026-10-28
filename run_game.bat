@echo off
title Nerf Target Game Launcher
echo ====================================================
echo   Nerf Target Game Launcher
echo ====================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found on Windows!
    pause
    exit /b 1
)

echo [2/3] Installing opencv-python and numpy...
python -m pip install opencv-python numpy

echo [3/3] Starting Game...
echo.
echo --------------------------------------------
echo  Controls:
echo    [B]   : Reset Background
echo    [R]   : Reset Score
echo    [ESC] : Exit
echo --------------------------------------------
echo.
python nerf_target.py

if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong.
    pause
)
