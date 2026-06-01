#!/bin/bash

# ==============================================================================
# SUDARSHAN AI - macOS GUI Launcher
# ==============================================================================
# Double-click this file to run the Tkinter desktop deepfake detection app.
# Make sure to run Install_SUDARSHAN.command first.
# ==============================================================================

# Ensure working directory is always the script directory (handles double-clicks)
cd "$(dirname "$0")"

# Text Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

if [ ! -d ".venv" ]; then
    clear
    echo -e "${RED}[ERROR] Virtual environment (.venv) not found!${NC}"
    echo -e "It seems the software dependencies have not been installed yet."
    echo -e "Please double-click the setup file to install everything first:"
    echo -e "  --> ${GREEN}Install_SUDARSHAN.command${NC}"
    echo ""
    read -p "Press [Enter] to exit..."
    exit 1
fi

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN}                    Launching SUDARSHAN AI...                         ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Activating virtual environment and starting Tkinter Desktop UI."
echo -e "Keep this Terminal window open while using the application."
echo ""

.venv/bin/python gui/sudarshn_ui.py

exit 0
