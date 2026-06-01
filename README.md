# SUDARSHAN AI
### Explainable Multi-Modal Deepfake & Video Manipulation Detection
*A project by Thapar Institute of Engineering & Technology*

---

SUDARSHAN AI is a state-of-the-art, explainable, cross-platform desktop application designed to detect deepfakes and facial manipulations in digital video, static images, and live video streams. 

By leveraging **three independent analysis streams** (Frequency pyramid phases, Temporal physiological heartbeat/blinking signals, and Spatial Vision Transformer attention CLS), SUDARSHAN AI provides high-precision binary classifications (REAL vs. FAKE) combined with plain-English explanation reports for non-technical users and Z-score diagnostic logs for developers.

---

## 🏗️ Repository Architecture

To maintain a clean repository structure while providing seamless distribution packages, this repository utilizes a **single master codebase** where OS-specific installers/launchers are housed inside the `installers/` directory:

```
SUDARSHAN-AI/
├── src/                      # Core deepfake models and pipelines
├── gui/                      # Desktop Tkinter GUI application
├── models/                   # Production weights (best_vit_model.pth)
├── assets/                   # Graphics, logo resources, and badges
├── docs/                     # Release checklists and deployment workflows
├── installers/               # Platform-specific installer & launcher scripts
│   ├── windows/              # Batch setup for Windows
│   ├── macos/                # Terminal double-clickable shell setup for Mac
│   └── linux/                # Interactive shell setup for Linux
├── USER_MANUAL.md            # Extensive end-user manual
├── README.md                 # Project developer guide (this file)
├── requirements.txt          # Shared python package dependencies
└── package_releases.py       # Automated build & release pipeline script
```

---

## 💻 Supported Operating Systems & Requirements

* **Windows**: Windows 10 / 11 (64-bit). Requires standard Python 3.10+ installation.
* **macOS**: Catalina or higher (Intel / Apple Silicon M1/M2/M3).
* **Linux**: Ubuntu/Debian, Fedora, Arch Linux. Requires `python3-tk` and `python3-venv` system modules.

---

## 🚀 End-User Installation Flow (Non-Technical)

Users do not need to clone the repository or use terminal commands. They simply download the pre-packaged ZIP bundle for their operating system:

### 🪟 Windows Setup
1. Download **`SUDARSHAN_AI_Windows_v1.0.zip`** and extract it.
2. Double-click **`Install_SUDARSHAN.bat`** (the setup wizard automatically checks requirements, configures a virtual environment, installs dependencies, and registers launchers).
3. Double-click **`Launch_SUDARSHAN.bat`** to start the app.

### 🍎 macOS Setup
1. Download **`SUDARSHAN_AI_Mac_v1.0.zip`** and extract it.
2. Double-click **`Install_SUDARSHAN.command`** to run the setup Terminal wizard.
3. Double-click **`Launch_SUDARSHAN.command`** to start the app.

### 🐧 Linux Setup
1. Download **`SUDARSHAN_AI_Linux_v1.0.zip`** and extract it.
2. Open terminal in the folder and run:
   ```bash
   ./install_linux.sh
   ./launch_linux.sh
   ```

---

## 🛠️ Automated Packaging Pipeline (For Developers)

SUDARSHAN AI contains a robust, cross-platform Python script to package and bundle OS-specific releases automatically from the single codebase.

### Run the Packaging Build:
To package the project, execute the packaging script at the root:
```bash
python3 package_releases.py --version v1.0
```

### What this script does:
1. **Validates files**: Verifies all source files, GUI modules, production weights (`models/best_vit_model.pth`), and installer scripts are present.
2. **Stages packages**: Creates isolated staging folders for Windows, macOS, and Linux, mapping installer/launcher files to the root level.
3. **Compresses release ZIPs**: Automatically generates versioned ZIP artifacts in the `releases/` folder:
   * `releases/SUDARSHAN_AI_Windows_v1.0.zip`
   * `releases/SUDARSHAN_AI_Mac_v1.0.zip`
   * `releases/SUDARSHAN_AI_Linux_v1.0.zip`
4. **Calculates Checksums**: Computes **SHA-256 hashes** for secure verification.
5. **Release Notes**: Automatically generates `releases/RELEASE_NOTES_v1.0.md` detailing changes and checksum matrices.

---

## 📈 Release & Versioning Strategy

SUDARSHAN AI follows strict Semantic Versioning (`vMAJOR.MINOR`). 

* **Stable Releases**: Labeled `v1.0`, `v2.0` - fully validated on all operating systems and published to production branches.
* **Development Builds**: Active branches for feature integrations.
* **Beta Builds**: Release candidates packaged using `package_releases.py` and distributed to select groups for cross-hardware evaluation before production rollouts.
