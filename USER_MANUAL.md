# SUDARSHAN AI — User & Setup Manual
### Explainable Deepfake & Video Manipulation Detection
*Supported Platforms: Windows, macOS, and Linux*

---

## Welcome to SUDARSHAN AI
SUDARSHAN AI is a state-of-the-art, explainable, multi-modal deepfake detection software. By fusing three independent analysis streams—**Frequency (Complex Steerable Pyramid)**, **Temporal (rPPG/heartbeat, blinking, lip movements)**, and **Spatial (Vision Transformer CLS & Attention)**—SUDARSHAN AI delivers highly robust detection of video manipulations alongside detailed user and developer explainability reports.

---

## 1. System Requirements & Prerequisites

SUDARSHAN AI runs on standard desktop systems, but requires Python 3 and system-level graphics/GUI toolkits.

### A. Python 3 (Required)
Ensure you have **Python >= 3.10** installed. 

* **Windows**: Download from [python.org/downloads/windows](https://www.python.org/downloads/windows/). 
  > [!IMPORTANT]
  > During installation, you **MUST check the box** that says **"Add python.exe to PATH"** at the bottom of the first setup window.
* **macOS**: Download from [python.org/downloads/macos](https://www.python.org/downloads/macos/) or install via Homebrew: `brew install python`.
* **Linux**: Standard package repositories (e.g. `sudo apt install python3`).

### B. OS-Specific Prerequisites
* **Windows**: No additional libraries required.
* **macOS**: If you install Python via Homebrew, you may need to install the Tkinter graphics package:
  ```bash
  brew install python-tk
  ```
* **Linux (Debian/Ubuntu-based)**: You must manually install standard Python virtual environment and Tkinter packages, as Debian/Ubuntu does not bundle them with the base Python install:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-tk python3-venv
  ```

---

## 2. Step-by-Step Installation

SUDARSHAN AI features fully automated, single-click setup scripts for all platforms.

### 🪟 Windows Setup
1. Unzip the downloaded `SUDARSHAN AI` folder.
2. Double-click the **`Install_SUDARSHAN.bat`** file.
3. A command prompt window will open. It will automatically check your Python version, create a clean, isolated virtual environment (`.venv`), install all required deep learning libraries (PyTorch, OpenCV, Mediapipe, timm, etc.), set up the weights, and run a validation check.
4. Once completed, press any key to close the window.

### 🍎 macOS Setup
1. Unzip the downloaded `SUDARSHAN AI` folder.
2. Double-click the **`Install_SUDARSHAN.command`** file.
3. A Terminal window will open automatically, verify your dependencies, build a secure Python virtual environment (`.venv`), install the packages, and copy the model weights.
4. Once completed, close the Terminal.

### 🐧 Linux Setup
1. Open a Terminal inside the unzipped `SUDARSHAN AI` folder.
2. Grant execution permissions on the installer (if needed) and run it:
   ```bash
   chmod +x install_linux.sh launch_linux.sh
   ./install_linux.sh
   ```
3. The wizard will check your system packages, alert you if `python3-tk` or `python3-venv` is missing with detailed commands, build the environment, and verify installation.

---

## 3. How to Launch the Application

Once installation is complete, running the GUI requires only a simple double-click:

* **Windows**: Double-click **`Launch_SUDARSHAN.bat`** in the application folder.
* **macOS**: Double-click **`Launch_SUDARSHAN.command`** in the application folder.
* **Linux**: Run **`./launch_linux.sh`** from your Terminal.

*Note: Keep the terminal window that opens in the background alive; it shows runtime logging and debug info while you use the app.*

---

## 4. Comprehensive Feature Guide

SUDARSHAN AI provides four specialized operating modes.

### 📁 1. Video Upload Analysis (Offline Mode)
Analyze pre-recorded videos to generate detailed spatial heatmaps and structured, explainable reports.
1. Click **"Upload Video"** in the UI and select any `.mp4`, `.avi`, or `.mov` file.
2. The progress indicator will display the current active analysis stage:
   * **Stage 1**: Extracting frames
   * **Stage 2**: Analyzing temporal dynamics (blinks, lip movements, rPPG heartbeat signals)
   * **Stage 3**: Extracting spatial features (ViT embeddings)
   * **Stage 4**: Computing frequency analysis (pyramid CSP phase)
   * **Stage 5**: Running F-Net model inference
   * **Stage 6**: Generating explanations
3. Read the reports in the explanation panels:
   * **User Explanation**: Plain-English physiological summaries (e.g. heartbeat patterns, irregular eye blinks).
   * **Developer Explanation**: Detailed algorithmic descriptions, modal stream importances, and anomaly Z-scores.
4. Observe the **Output Frame Display** to see spatial attention heatmaps overlaid on the face.

### 🖼️ 2. Static Image Analysis
Run inference on a static photograph.
1. Click **"Upload Image"** and choose any `.jpg`, `.png`, or `.webp` file.
2. The model will run a high-speed forward pass.
3. *Note: Since static images do not have a timeline, temporal and physiological indicators will be flagged as inactive in the report.*

### 📹 3. Live Webcam Analysis (Real-time Mode)
Real-time deepfake monitoring using your computer's built-in webcam.
1. Click **"Start Live Stream"** to open your default webcam.
2. The UI will show a mirror feed of your camera.
3. Every 32 frames (~1.5 seconds), the system takes a temporal snapshot buffer and runs high-speed inference in the background to update the real-time classification (REAL vs. FAKE) and confidence gauge.
4. Click **"Stop Stream"** to end live monitoring.

### 🖥️ 4. Screen Capture / Interview Monitor (Professional Mode)
Window-locked monitoring designed for HR interviews or live video calls (e.g., Google Meet, Zoom, Teams).
1. **Join the Video Call**: Open your web browser and join your meeting.
2. **Lock the Window**: In SUDARSHAN AI, click **"Detect Meet Window"**. The software will search for active windows with "Meet" in the title (you can also type custom window titles).
3. **Capture a Preview**: Once locked, a preview frame of the video call appears in the application's input screen.
4. **Identify the Candidate**: Click **"Select Target Face"**. The software will detect all faces on the screen and draw numbered boxes over them.
5. **Select Face**: Enter the number corresponding to the face of the candidate you want to monitor. SUDARSHAN AI extracts a high-fidelity **FaceNet identity embedding** of this person.
6. **Active Monitoring**: The system captures and tracks *only* the selected candidate's face. If another person jumps into the screen, or if the candidate moves, the system matches their facial embedding in real-time, locking onto the correct face.
7. If the candidate turns their camera off or leaves, the status updates to **"SEARCHING"** or **"TARGET LOST"**.
8. If the software detects manipulation, a visual **"SUSPICION ALERT"** is triggered with red highlight indicators.

---

## 5. Troubleshooting & FAQ

#### Q: The installer opens and immediately closes or prints an error on Windows.
**A:** This is usually because Python is either not installed, or you forgot to check the "Add python.exe to PATH" option during Python installation. Re-run the Python setup from [python.org](https://www.python.org), choose "Modify", and make sure the PATH checkmark is checked.

#### Q: macOS blocks "Launch_SUDARSHAN.command" stating it is from an unidentified developer.
**A:** This is macOS Gatekeeper security behavior for downloaded scripts. To allow it:
1. Open your Mac's **System Settings > Privacy & Security**.
2. Scroll down to find the section about "Launch_SUDARSHAN.command was blocked".
3. Click **"Open Anyway"** and enter your password. You will only need to do this once.
*(Alternatively, open Terminal in the folder and run: `chmod +x Launch_SUDARSHAN.command`)*

#### Q: Live webcam displays a "Could not open webcam" error.
**A:** Make sure no other application (like Zoom, MS Teams, or Skype) is currently using your webcam, as operating systems only allow one program to capture the camera feed at a time.

#### Q: The program crashes on Linux with "ImportError: No module named tkinter".
**A:** Unlike Windows and macOS, Linux does not bundle Tkinter with the standard Python installer. Run the following command in Terminal to install it, then re-run the setup script:
`sudo apt-get install python3-tk`
