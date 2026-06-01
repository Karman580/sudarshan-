@echo off
title SUDARSHAN AI GUI Launcher
cd /d "%~dp0"

if not exist ".venv" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo It seems the software dependencies have not been installed yet.
    echo Please double-click the setup file to install everything first:
    echo   --> Install_SUDARSHAN.bat
    echo.
    pause
    exit /b 1
)

echo ======================================================================
echo                    Launching SUDARSHAN AI...                         
echo ======================================================================
echo Starting Tkinter Desktop UI inside the virtual environment.
echo Keep this command prompt window open while using the application.
echo.

.venv\Scripts\python.exe gui\sudarshn_ui.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed or exited with an error code.
    pause
)
exit /b 0
