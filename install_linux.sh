#!/bin/bash

# ==============================================================================
# SUDARSHAN AI - Linux Setup & Installer Wizard
# ==============================================================================
# Run this script to check package dependencies, configure a clean virtual
# environment, install requirements, and prepare launching scripts.
# ==============================================================================

# Ensure working directory is always the script directory
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
echo -e "${CYAN}                   SUDARSHAN AI - Linux Setup Wizard                  ${NC}"
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
    echo -e "Please install it using your system package manager (e.g. sudo apt install python3)"
    echo ""
    read -p "Press [Enter] to exit..."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

echo -e "Found Python version: ${GREEN}${PYTHON_VERSION}${NC}"

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    echo -e "${RED}[ERROR] Python version too old! Need >= 3.10.${NC}"
    echo -e "Please upgrade Python via your system package manager."
    echo ""
    read -p "Press [Enter] to exit..."
    exit 1
fi
echo -e "${GREEN}✓ Python version check passed!${NC}"
echo ""

# ------------------------------------------------------------------------------
# STEP 2: Verify python3-venv & python3-tk
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/5] Checking Linux System Packages...${NC}"

# Check for python3-venv module capability
python3 -c "import venv" &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] The 'venv' module is missing!${NC}"
    echo -e "Debian/Ubuntu-based systems separate this from the main Python bundle."
    echo -e "Please run the following command to install it:"
    echo -e "  --> ${CYAN}sudo apt-get install python3-venv${NC}"
    echo ""
    read -p "Press [Enter] to exit..."
    exit 1
fi

# Check for python3-tk (Tkinter on Linux is NOT bundled!)
python3 -c "import tkinter" &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Tkinter GUI module is not installed!${NC}"
    echo -e "SUDARSHAN AI requires Tkinter for the desktop graphical user interface."
    echo -e "Please install it using one of these commands depending on your distro:"
    echo -e "  Ubuntu/Debian: ${CYAN}sudo apt-get install python3-tk${NC}"
    echo -e "  Fedora:        ${CYAN}sudo dnf install python3-tkinter${NC}"
    echo -e "  Arch Linux:    ${CYAN}sudo pacman -S tk${NC}"
    echo ""
    read -p "Press [Enter] to exit..."
    exit 1
fi

echo -e "${GREEN}✓ All necessary Linux system packages found!${NC}"
echo ""

# ------------------------------------------------------------------------------
# STEP 3: Setup Virtual Environment
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/5] Setting up isolated Virtual Environment (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "Creating virtual environment inside this folder..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to create virtual environment!${NC}"
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
echo -e "This may take a few minutes as deep learning libraries compile/install..."
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
# STEP 5: Verify Weights & System Functionality
# ------------------------------------------------------------------------------
echo -e "${BLUE}[5/5] Performing system compilation and loading checks...${NC}"

# Check for weights file, copy if missing from v2_balanced
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
    echo -e "There is a compilation or missing dependency issue."
    read -p "Press [Enter] to exit..."
    exit 1
fi

echo -e "${GREEN}✓ System verification complete! System is fully functional!${NC}"
echo ""

# ------------------------------------------------------------------------------
# CREATE LAUNCHER
# ------------------------------------------------------------------------------
if [ ! -f "launch_linux.sh" ]; then
    echo "#!/bin/bash" > launch_linux.sh
    echo 'cd "$(dirname "$0")"' >> launch_linux.sh
    echo "echo 'Launching SUDARSHAN AI...'" >> launch_linux.sh
    echo ".venv/bin/python gui/sudarshn_ui.py" >> launch_linux.sh
fi

chmod +x launch_linux.sh install_linux.sh 2>/dev/null

echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}              SUDARSHAN AI INSTALLATION COMPLETED!                    ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "You can now run SUDARSHAN AI anytime by running the launcher script:"
echo -e "  --> ${GREEN}./launch_linux.sh${NC} (located in this folder)"
echo ""
echo -e "We have granted execution permissions to the launcher."
echo -e "Thank you for installing SUDARSHAN AI!"
echo -e "${CYAN}======================================================================${NC}"
echo ""
exit 0
