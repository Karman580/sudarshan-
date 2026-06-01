@echo off
title SUDARSHAN AI - Windows Setup Wizard
cd /d "%~dp0"

cls
echo ======================================================================
echo                  SUDARSHAN AI - Windows Setup Wizard                  
echo ======================================================================
echo Starting system checks and installer...
echo.

:: ------------------------------------------------------------------------------
:: STEP 1: Verify Python 3
:: ------------------------------------------------------------------------------
echo [1/4] Checking Python Installation...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system!
    echo.
    echo SUDARSHAN AI requires Python 3.10 or higher.
    echo Opening Python Downloads page in your browser...
    start https://www.python.org/downloads/
    echo.
    echo IMPORTANT: In the installer, make sure to check "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

:: Validate Python Version >= 3.10
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if %errorlevel% neq 0 (
    echo [ERROR] Python version too old! Need Python >= 3.10.
    echo Opening Python Downloads page in your browser...
    start https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo Python version check passed!
echo.

:: ------------------------------------------------------------------------------
:: STEP 2: Setup Virtual Environment
:: ------------------------------------------------------------------------------
echo [2/4] Setting up isolated Virtual Environment (.venv)...
if not exist ".venv" (
    echo Creating virtual environment. This keeps your system clean...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
)
echo Virtual environment prepared!
echo.

:: ------------------------------------------------------------------------------
:: STEP 3: Install Dependencies
:: ------------------------------------------------------------------------------
echo [3/4] Upgrading Pip & Installing Python packages (requirements.txt)...
echo This may take a few minutes as deep learning libraries are installed...
echo.

.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed!
    echo Please check your internet connection and try running setup again.
    pause
    exit /b 1
)
echo All package dependencies successfully installed!
echo.

:: ------------------------------------------------------------------------------
:: STEP 4: Setup Weights and Verify Compilation
:: ------------------------------------------------------------------------------
echo [4/4] Performing system compilation and loading checks...

:: Ensure models directory exists
if not exist "models" (
    mkdir "models"
)

:: Copy integrated weights to central models directory if missing
if not exist "models\best_vit_model.pth" (
    if exist "v2_balanced\checkpoints\best_vit_model.pth" (
        echo Copying integrated weights to central models directory...
        copy "v2_balanced\checkpoints\best_vit_model.pth" "models\best_vit_model.pth" >nul
    )
)

.venv\Scripts\python.exe verify_timm.py
if %errorlevel% neq 0 (
    echo [ERROR] System verification check failed!
    echo There is a compilation or runtime configuration issue.
    pause
    exit /b 1
)
echo System verification complete! System is fully functional!
echo.

:: Create Launcher if missing
if not exist "Launch_SUDARSHAN.bat" (
    echo @echo off > Launch_SUDARSHAN.bat
    echo title SUDARSHAN AI GUI >> Launch_SUDARSHAN.bat
    echo cd /d "%%~dp0" >> Launch_SUDARSHAN.bat
    echo echo Launching SUDARSHAN AI Tkinter Desktop UI... >> Launch_SUDARSHAN.bat
    echo .venv\Scripts\python.exe gui\sudarshn_ui.py >> Launch_SUDARSHAN.bat
)

echo ======================================================================
echo              SUDARSHAN AI INSTALLATION COMPLETED!                    
echo ======================================================================
echo You can now run SUDARSHAN AI anytime by double-clicking the launcher:
echo   --^> Launch_SUDARSHAN.bat (located in this folder)
echo.
echo Thank you for installing SUDARSHAN AI!
echo ======================================================================
echo.
pause
exit /b 0
