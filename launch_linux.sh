#!/bin/bash

# ==============================================================================
# SUDARSHAN AI - Linux GUI Launcher
# ==============================================================================
# Run this shell script to start the Tkinter desktop GUI app.
# Make sure to run ./install_linux.sh first.
# ==============================================================================

# Ensure working directory is always the script directory
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
    echo -e "Please run the setup script to install everything first:"
    echo -e "  --> ${GREEN}./install_linux.sh${NC}"
    echo ""
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
