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

echo [2/3] Installing requirements (opencv-python, numpy)...
python -m pip install -r requirements.txt

echo [3/3] Starting Game...
echo.
echo --------------------------------------------
echo  Controls:
echo    [C]          : Auto align corners (flash)
echo    [M]          : Click 4 corners manually
echo    [TAB]        : Select corner
echo    Arrows/WASD  : Move selected corner (2px)
echo    [R]          : Reset score and background
echo    [F]          : Fullscreen projector view
echo    [ESC]/[Q]    : Exit
echo --------------------------------------------
echo.
python nerf_target.py

if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong.
    pause
)
