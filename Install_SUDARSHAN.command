#!/bin/bash

# ==============================================================================
# SUDARSHAN AI - macOS Setup & Installer Wizard
# ==============================================================================
# Double-click this file to automatically check dependencies, create a Python
# virtual environment, install requirements, and set up your launchers.
# ==============================================================================

# Ensure working directory is always the script directory (handles double-clicks)
cd "$(dirname "$0")"

# Text Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

clear
echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN}                   SUDARSHAN AI - macOS Setup Wizard                  ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Starting system checks and installer..."
echo ""

# ------------------------------------------------------------------------------
# STEP 1: Verify Python 3
# ------------------------------------------------------------------------------
echo -e "${BLUE}[1/5] Checking Python 3 Installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 was not found on your system!${NC}"
    echo -e "SUDARSHAN AI requires Python 3.10 or higher."
    echo -e "Please install it using one of the following methods:"
    echo -e "  1. Download and run the installer from: https://www.python.org/downloads/macos/"
    echo -e "  2. If you use Homebrew, run: brew install python"
    echo ""
    read -p "Press [Enter] to exit..."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

echo -e "Found Python version: ${GREEN}${PYTHON_VERSION}${NC}"

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    echo -e "${RED}[ERROR] Python version too old! Found ${PYTHON_VERSION}, but need >= 3.10.${NC}"
    echo -e "Please upgrade Python at: https://www.python.org/downloads/macos/"
    echo ""
    read -p "Press [Enter] to exit..."
    exit 1
fi
echo -e "${GREEN}✓ Python version check passed!${NC}"
echo ""

# ------------------------------------------------------------------------------
# STEP 2: Verify Tkinter
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/5] Checking Tkinter GUI support...${NC}"
if ! python3 -c "import tkinter" &> /dev/null; then
    echo -e "${YELLOW}[WARN] Tkinter module is not available in your Python distribution!${NC}"
    echo -e "Tkinter is standard in Python, but some custom builds or homebrew installations omit it."
    echo -e "  To fix this, please run: ${CYAN}brew install python-tk${NC}"
    echo -e "  Or reinstall Python from: https://www.python.org/downloads/macos/"
    echo ""
    read -p "Press [Enter] to exit..."
    exit 1
fi
echo -e "${GREEN}✓ Tkinter GUI support check passed!${NC}"
echo ""

# ------------------------------------------------------------------------------
# STEP 3: Setup Virtual Environment
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/5] Setting up clean Virtual Environment (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "Creating new isolated virtual environment. This keeps your system clean..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to create virtual environment!${NC}"
        echo -e "Please ensure python3-venv is available or recreate Python setup."
        read -p "Press [Enter] to exit..."
        exit 1
    fi
fi
echo -e "${GREEN}✓ Virtual environment prepared!${NC}"
echo ""

# ------------------------------------------------------------------------------
# STEP 4: Install Dependencies
# ------------------------------------------------------------------------------
echo -e "${BLUE}[4/5] Upgrading Pip & Installing Python packages (requirements.txt)...${NC}"
echo -e "This may take a few minutes as deep learning libraries are installed..."
echo ""

.venv/bin/python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}[WARN] Failed to upgrade pip. Proceeding with dependency installation...${NC}"
fi

.venv/bin/python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Installation of dependencies failed!${NC}"
    echo -e "Please check your internet connection and try running setup again."
    read -p "Press [Enter] to exit..."
    exit 1
fi
echo -e "${GREEN}✓ All package dependencies successfully installed!${NC}"
echo ""

# ------------------------------------------------------------------------------
# STEP 5: Verify Model Setup & Weights
# ------------------------------------------------------------------------------
echo -e "${BLUE}[5/5] Performing system compilation and loading checks...${NC}"

# Check for weight file in models directory, copy if missing from v2_balanced
if [ ! -f "models/best_vit_model.pth" ]; then
    if [ -f "v2_balanced/checkpoints/best_vit_model.pth" ]; then
        echo -e "Copying integrated weights to central models directory..."
        mkdir -p models
        cp v2_balanced/checkpoints/best_vit_model.pth models/best_vit_model.pth
    fi
fi

.venv/bin/python verify_timm.py
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] System verification check failed!${NC}"
    echo -e "There is a compile or missing dependency issue."
    read -p "Press [Enter] to exit..."
    exit 1
fi

echo -e "${GREEN}✓ System verification complete! System is fully functional!${NC}"
echo ""

# ------------------------------------------------------------------------------
# FINALIZELAUNCHERS
# ------------------------------------------------------------------------------
chmod +x Launch_SUDARSHAN.command 2>/dev/null

echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}              SUDARSHAN AI INSTALLATION COMPLETED!                    ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "You can now run SUDARSHAN AI anytime by double-clicking the launcher:"
echo -e "  --> ${GREEN}Launch_SUDARSHAN.command${NC} (located in this folder)"
echo ""
echo -e "We have granted execution permissions to the launcher."
echo -e "Thank you for installing SUDARSHAN AI!"
echo -e "${CYAN}======================================================================${NC}"
echo ""
read -p "Press [Enter] to exit Setup..."
exit 0
