#!/usr/bin/env python3
"""
SUDARSHAN AI - macOS Environment Bootstrapper & First-Launch Wizard
==============================================================================
This script is executed natively from SUDARSHAN AI.app. It verifies if the
isolated environment is prepared inside Library/Application Support.
If setup is required, it boots a professional dark-themed Tkinter installer window
which builds a virtual environment, installs dependencies, verifies models,
and launches the main application automatically.
==============================================================================
"""

import os
import sys
import subprocess
import threading
import time

# ------------------------------------------------------------------------------
# STEP 1: Verify Tkinter (fallback to AppleScript native modal dialogs if missing)
# ------------------------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import scrolledtext
    from tkinter import messagebox
except ImportError:
    applescript = (
        'display alert "SUDARSHAN AI - System Check" '
        'message "Tkinter (Python GUI Toolkit) is missing from your system Python interpreter.\\n\\n'
        'SUDARSHAN AI requires Tkinter to render the installation and desktop interfaces. '
        'To resolve this, please install a standard Python distribution or run:\\n\\n'
        '  brew install python-tk\\n\\n'
        'Would you like to visit the official Python downloads page?" '
        'buttons {"Download Python", "Cancel"} default button "Download Python"'
    )
    proc = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True)
    if "Download Python" in proc.stdout:
        subprocess.run(['open', 'https://www.python.org/downloads/macos/'])
    sys.exit(1)

# ------------------------------------------------------------------------------
# ENVIRONMENT RESOLUTION Paths
# ------------------------------------------------------------------------------
APP_SUPPORT_DIR = os.path.expanduser("~/Library/Application Support/SUDARSHAN_AI")
VENV_DIR = os.path.join(APP_SUPPORT_DIR, ".venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python")
VENV_PIP = os.path.join(VENV_DIR, "bin", "pip")
LOG_FILE = os.path.join(APP_SUPPORT_DIR, "install.log")

# ------------------------------------------------------------------------------
# HIGH-SPEED HEALTH CHECK
# ------------------------------------------------------------------------------
def check_venv_healthy():
    """Verify if the virtual environment exists and has all required packages."""
    if not os.path.exists(VENV_PYTHON):
        return False
    
    # Fast import check to verify dependency health
    check_code = "import torch, cv2, timm, mediapipe; print('HEALTHY')"
    try:
        res = subprocess.run(
            [VENV_PYTHON, "-c", check_code],
            capture_output=True,
            text=True,
            timeout=4
        )
        return res.returncode == 0 and "HEALTHY" in res.stdout
    except Exception:
        return False

# ------------------------------------------------------------------------------
# SETUP BACKGROUND THREAD WORKER
# ------------------------------------------------------------------------------
class InstallWorker(threading.Thread):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.success = False

    def log(self, text, is_step=False):
        """Append logs to the UI and the log file."""
        clean_text = f"{text}\n"
        self.ui.append_log(clean_text, is_step)
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
        except Exception:
            pass

    def run(self):
        try:
            # Step 1: Initialize folders
            self.ui.update_status("Initializing directories...", 10)
            self.log("Step 1/5: Initializing App Support folders...", is_step=True)
            os.makedirs(APP_SUPPORT_DIR, exist_ok=True)
            
            # Write a clean header in install.log
            with open(LOG_FILE, "w") as f:
                f.write("======================================================================\n")
                f.write("                  SUDARSHAN AI - macOS Installation Log               \n")
                f.write("======================================================================\n")
                f.write(f"Venv target: {VENV_DIR}\n\n")

            self.log(f"Logs redirected to: {LOG_FILE}")
            
            # Step 2: Build isolated virtual environment
            self.ui.update_status("Creating virtual environment (this keeps your system clean)...", 25)
            self.log("Step 2/5: Creating isolated Python virtual environment...", is_step=True)
            
            # Use sys.executable to build venv
            cmd_venv = [sys.executable, "-m", "venv", VENV_DIR]
            self.log(f"Running venv creation command: {' '.join(cmd_venv)}")
            proc_venv = subprocess.Popen(cmd_venv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                line = proc_venv.stdout.readline()
                if not line:
                    break
                self.log(f"  [venv] {line.strip()}")
            proc_venv.wait()
            
            if proc_venv.returncode != 0:
                raise RuntimeError(f"Venv creation failed with code {proc_venv.returncode}")
                
            self.log("✓ Virtual environment successfully built!")
            
            # Step 3: Upgrade pip
            self.ui.update_status("Upgrading package installer (pip)...", 45)
            self.log("Step 3/5: Upgrading pip environment packages...", is_step=True)
            cmd_pip_up = [VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"]
            proc_pip_up = subprocess.Popen(cmd_pip_up, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                line = proc_pip_up.stdout.readline()
                if not line:
                    break
                self.log(f"  [pip] {line.strip()}")
            proc_pip_up.wait()
            
            # Step 4: Install packages from requirements.txt
            self.ui.update_status("Installing deep learning libraries (PyTorch, cv2, timm)... This may take a minute...", 60)
            self.log("Step 4/5: Installing deepfake detection packages (requirements.txt)...", is_step=True)
            
            # Resolve requirements path relative to active directory
            req_path = "requirements.txt"
            if not os.path.exists(req_path):
                raise FileNotFoundError("requirements.txt not found in bundle Resources!")
                
            cmd_install = [VENV_PIP, "install", "-r", req_path]
            self.log(f"Running dependency installation: {' '.join(cmd_install)}")
            proc_install = subprocess.Popen(cmd_install, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                line = proc_install.stdout.readline()
                if not line:
                    break
                self.log(f"  [pip-install] {line.strip()}")
            proc_install.wait()
            
            if proc_install.returncode != 0:
                raise RuntimeError(f"Package installation failed with code {proc_install.returncode}")
                
            self.log("✓ Packages successfully compiled and installed!")

            # Step 5: System Verification Tests
            self.ui.update_status("Performing model checks and verification...", 85)
            self.log("Step 5/5: Executing F-Net weights and neural stream verification...", is_step=True)
            
            verify_script = "verify_timm.py"
            if not os.path.exists(verify_script):
                raise FileNotFoundError("verify_timm.py check script is missing from resources!")
                
            cmd_verify = [VENV_PYTHON, verify_script]
            proc_verify = subprocess.Popen(cmd_verify, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                line = proc_verify.stdout.readline()
                if not line:
                    break
                self.log(f"  [verify] {line.strip()}")
            proc_verify.wait()
            
            if proc_verify.returncode != 0:
                raise RuntimeError(f"System verification checks failed! Please review logs.")
                
            self.log("✓ System verification check passed!")
            self.ui.update_status("Setup completed successfully! Starting application...", 100)
            self.success = True
            
            # Launch App after 1.5 seconds
            time.sleep(1.5)
            self.ui.launch_app_and_exit()
            
        except Exception as e:
            self.log(f"\n[CRITICAL ERROR] Setup failed: {str(e)}")
            self.ui.show_error(str(e))

# ------------------------------------------------------------------------------
# SETUP WIZARD TKINTER GUI CLASS
# ------------------------------------------------------------------------------
class SetupWizardUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SUDARSHAN AI - macOS Setup Wizard")
        self.root.geometry("680x480")
        self.root.resizable(False, False)
        
        # Center Window on Screen
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")
        
        # Color System: Slate Dark Theme
        self.bg_color = "#0F172A"       # Deep dark slate
        self.card_color = "#1E293B"     # Darker slate card
        self.accent_color = "#06B6D4"   # Vibrant Cyan
        self.text_color = "#F8FAFC"     # Off-white
        self.sub_text_color = "#94A3B8" # Muted slate gray
        self.console_color = "#020617"  # Deep rich black
        
        self.root.configure(bg=self.bg_color)
        
        # Custom ttk styles
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TProgressbar", thickness=12, troughcolor=self.card_color, background=self.accent_color)
        
        self.build_ui()
        
    def build_ui(self):
        # 1. Header Layout
        header_frame = tk.Frame(self.root, bg=self.bg_color, pady=15)
        header_frame.pack(fill=tk.X)
        
        title_lbl = tk.Label(
            header_frame, 
            text="SUDARSHAN AI - Environment Setup", 
            font=("Helvetica", 18, "bold"), 
            bg=self.bg_color, 
            fg=self.text_color
        )
        title_lbl.pack(anchor="w", padx=25)
        
        desc_lbl = tk.Label(
            header_frame, 
            text="Setting up an isolated virtual environment and downloading deep learning features...", 
            font=("Helvetica", 11), 
            bg=self.bg_color, 
            fg=self.sub_text_color
        )
        desc_lbl.pack(anchor="w", padx=25, pady=3)
        
        # 2. Progress panel
        self.progress_frame = tk.Frame(self.root, bg=self.card_color, padx=20, pady=15, bd=0)
        self.progress_frame.pack(fill=tk.X, padx=25, pady=5)
        
        self.status_lbl = tk.Label(
            self.progress_frame,
            text="Starting setup pipeline...",
            font=("Helvetica", 11, "bold"),
            bg=self.card_color,
            fg=self.accent_color
        )
        self.status_lbl.pack(anchor="w", pady=2)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate', value=0, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=8)
        
        # 3. Log Console Window
        console_frame = tk.Frame(self.root, bg=self.bg_color)
        console_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=10)
        
        console_lbl = tk.Label(
            console_frame,
            text="INSTALLATION LOGGER CONSOLE",
            font=("Helvetica", 9, "bold"),
            bg=self.bg_color,
            fg=self.sub_text_color
        )
        console_lbl.pack(anchor="w", pady=2)
        
        self.log_txt = scrolledtext.ScrolledText(
            console_frame,
            bg=self.console_color,
            fg="#22D3EE", # Light Neon Cyan
            insertbackground="#22D3EE",
            font=("Courier", 10),
            bd=1,
            highlightthickness=1,
            highlightcolor=self.accent_color,
            highlightbackground=self.card_color,
            state=tk.DISABLED
        )
        self.log_txt.pack(fill=tk.BOTH, expand=True)
        
        # 4. Footer controls
        self.footer_frame = tk.Frame(self.root, bg=self.bg_color, pady=12)
        self.footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Log view trigger button (hidden by default, enabled on failure)
        self.btn_logs = tk.Button(
            self.footer_frame,
            text="Open install.log",
            command=self.open_log_file,
            font=("Helvetica", 11, "bold"),
            bg=self.card_color,
            fg=self.text_color,
            activebackground=self.accent_color,
            activeforeground=self.bg_color,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=6,
            state=tk.DISABLED
        )
        self.btn_logs.pack(side=tk.LEFT, padx=25)
        
        self.btn_close = tk.Button(
            self.footer_frame,
            text="Close Wizard",
            command=self.root.destroy,
            font=("Helvetica", 11, "bold"),
            bg=self.card_color,
            fg=self.sub_text_color,
            activebackground=self.accent_color,
            activeforeground=self.bg_color,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=6
        )
        self.btn_close.pack(side=tk.RIGHT, padx=25)

    def start_install(self):
        """Kick off installation worker thread."""
        self.worker = InstallWorker(self)
        self.worker.start()

    def update_status(self, text, progress_val):
        """Update progress bar and status text safely from helper threads."""
        self.root.after(0, lambda: self._update_status_main(text, progress_val))

    def _update_status_main(self, text, progress_val):
        self.status_lbl.configure(text=text)
        self.progress_bar['value'] = progress_val

    def append_log(self, text, is_step=False):
        """Append trace log cleanly into console widget."""
        self.root.after(0, lambda: self._append_log_main(text, is_step))

    def _append_log_main(self, text, is_step):
        self.log_txt.configure(state=tk.NORMAL)
        if is_step:
            # Highlight step titles in a distinct format
            self.log_txt.insert(tk.END, f"\n>>> {text}", 'step')
            self.log_txt.tag_config('step', foreground="#F43F5E", font=("Courier", 10, "bold")) # Rose/red highlights
        else:
            self.log_txt.insert(tk.END, text)
        self.log_txt.see(tk.END)
        self.log_txt.configure(state=tk.DISABLED)

    def show_error(self, error_msg):
        """Display installation failure state."""
        self.root.after(0, lambda: self._show_error_main(error_msg))

    def _show_error_main(self, error_msg):
        self.status_lbl.configure(text="Installation aborted with critical errors!", fg="#EF4444") # Red error
        self.progress_bar['value'] = 100
        # Change progress trough color or highlight border to signify failure
        self.log_txt.configure(highlightbackground="#EF4444", highlightcolor="#EF4444")
        self.btn_logs.configure(state=tk.NORMAL, bg="#EF4444", fg=self.text_color)
        messagebox.showerror(
            "SUDARSHAN AI - Setup Failed", 
            f"An error occurred during environment configuration:\n\n{error_msg}\n\n"
            f"Please click 'Open install.log' to troubleshoot, or file a bug report."
        )

    def open_log_file(self):
        """Open local troubleshooting install.log in the default text editor."""
        if os.path.exists(LOG_FILE):
            subprocess.run(["open", LOG_FILE])

    def launch_app_and_exit(self):
        """Launch the main GUI using our newly built virtual environment python and exit."""
        self.root.after(0, self._launch_app_and_exit_main)

    def _launch_app_and_exit_main(self):
        try:
            # Launch App in background without hanging the parent process
            subprocess.Popen([VENV_PYTHON, "gui/sudarshn_ui.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start SUDARSHAN AI: {str(e)}")
        finally:
            self.root.destroy()
            sys.exit(0)

# ------------------------------------------------------------------------------
# CORE APPLICATION ENTRY ROUTINE
# ------------------------------------------------------------------------------
def main():
    # If environment is healthy, skip installation and launch immediately!
    if check_venv_healthy():
        try:
            subprocess.Popen([VENV_PYTHON, "gui/sudarshn_ui.py"])
            sys.exit(0)
        except Exception:
            # Fall back to wizard if direct launch fails
            pass
            
    # Otherwise, boot the installer UI wizard!
    app = SetupWizardUI()
    app.start_install()
    app.root.mainloop()

if __name__ == "__main__":
    main()
