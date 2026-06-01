@echo off
title SUDARSHAN AI - Windows Setup Wizard
cd /d "%~dp0"

:: Initialize clean install.log
echo ====================================================================== > install.log
echo                  SUDARSHAN AI - Windows Setup Log                      >> install.log
echo ====================================================================== >> install.log
echo Date: %date% %time% >> install.log
echo. >> install.log

cls
echo ======================================================================
echo                  SUDARSHAN AI - Windows Setup Wizard                  
echo ======================================================================
echo Starting system checks and installer... Logging to install.log
echo.

:: ------------------------------------------------------------------------------
:: STEP 1: Verify Python 3
:: ------------------------------------------------------------------------------
echo [1/4] Checking Python Installation...
echo [1/4] Checking Python Installation... >> install.log
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system!
    echo [ERROR] Python was not found on your system! >> install.log
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
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >> install.log 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python version too old! Need Python >= 3.10.
    echo [ERROR] Python version too old! Need Python >= 3.10. >> install.log
    echo Opening Python Downloads page in your browser...
    start https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo Python version check passed!
echo Python version check passed! >> install.log
echo.

:: ------------------------------------------------------------------------------
:: STEP 2: Setup Virtual Environment
:: ------------------------------------------------------------------------------
echo [2/4] Setting up isolated Virtual Environment (.venv)...
echo [2/4] Setting up isolated Virtual Environment (.venv)... >> install.log
if not exist ".venv" (
    echo Creating virtual environment. This keeps your system clean...
    python -m venv .venv >> install.log 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        echo [ERROR] Failed to create virtual environment! >> install.log
        pause
        exit /b 1
    )
)
echo Virtual environment prepared!
echo Virtual environment prepared! >> install.log
echo.

:: ------------------------------------------------------------------------------
:: STEP 3: Install Dependencies
:: ------------------------------------------------------------------------------
echo [3/4] Upgrading Pip & Installing Python packages (requirements.txt)...
echo [3/4] Upgrading Pip & Installing Python packages (requirements.txt)... >> install.log
echo This may take a few minutes as deep learning libraries are installed...
echo.

echo Upgrading pip inside virtual environment... >> install.log
.venv\Scripts\python.exe -m pip install --upgrade pip >> install.log 2>&1

echo Installing packages from requirements.txt... >> install.log
.venv\Scripts\pip.exe install -r requirements.txt >> install.log 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed! Check install.log for details.
    echo [ERROR] Dependency installation failed! >> install.log
    echo Please check your internet connection and try running setup again.
    pause
    exit /b 1
)
echo All package dependencies successfully installed!
echo All package dependencies successfully installed! >> install.log
echo.

:: ------------------------------------------------------------------------------
:: STEP 4: Setup Weights and Verify Compilation
:: ------------------------------------------------------------------------------
echo [4/4] Performing system compilation and loading checks...
echo [4/4] Performing system compilation and loading checks... >> install.log

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

echo Running verify_timm.py system checks... >> install.log
.venv\Scripts\python.exe verify_timm.py >> install.log 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] System verification check failed! Check install.log for details.
    echo [ERROR] System verification check failed! >> install.log
    pause
    exit /b 1
)
echo System verification complete! System is fully functional!
echo System verification complete! System is fully functional! >> install.log
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
