# ==============================================================================
# sudarshn_ui.py - Explainable Deepfake Detection GUI
# ==============================================================================
# Tkinter-based desktop application for deepfake detection with:
# - Dual video display (raw input / heatmap output)
# - Stage-based progress indicators
# - Natural language explanation panel
# - Thread-safe, Windows-compatible design
#
# Developed @ Thapar Institute of Engineering & Technology, Patiala
# ==============================================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import subprocess
import threading
import queue
from typing import Dict, List, Optional

# Backend import with updated interface
from src.inference import run_video_pipeline, run_image_pipeline, run_webcam_pipeline
from src.aadhaar_verification.integration.pipeline import run_aadhaar_verification
from gui.language_manager import load_config, save_config, load_language_json, get_available_languages, smart_translate
from tkinter import font as tkfont

# ==============================================================================
# Interview Monitor Mode imports
# ==============================================================================
import webbrowser
import time
import statistics
from src.inference import run_screen_buffer_pipeline
from src.screen_capture import find_meet_window, capture_window
from src.face_identity import detect_and_embed, cosine_similarity


class SudarshnApp:
    """
    Main application class for SUDARSHN-0.0 Explainable Deepfake Detector.
    
    Features:
    - Input panel: Shows raw video playback
    - Output panel: Shows heatmap-overlaid frames after inference
    - Explanation panel: Natural language interpretation
    - Confidence score: Video-level prediction confidence
    - Stage indicator: Real-time progress during inference
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title(
            "SUDARSHN-0.0 | Thapar Institute of Engineering & Technology, Patiala"
        )
        self.root.geometry("1200x800")
        self.root.configure(bg="#f4f4f4")
        
        # Video playback state
        self.cap = None
        self.running = False
        self.current_video_path = None
        
        # Heatmap playback state
        self.heatmap_frames: Dict[int, str] = {}
        self.all_frame_paths: List[str] = []
        self.heatmap_running = False
        self.heatmap_frame_idx = 0
        
        # Thread communication
        self.progress_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Inference state
        self.inference_running = False
        
        # Inference state
        self.inference_running = False
        
        # Live Webcam State
        self.webcam_thread = None
        self.webcam_inference_thread = None
        self.webcam_lock = threading.Lock()
        self.frame_buffer = []
        self.BUFFER_SIZE = 32
        self.frame_counter = 0
        
        # ==================================================================
        # Interview Monitor Mode State (Window-Locked + Identity)
        # ==================================================================
        self.monitoring_active = False
        self.monitor_capture_thread = None
        self.monitor_inference_thread = None
        self.monitor_buffer = []
        self.monitor_lock = threading.Lock()
        self.monitor_queue = queue.Queue()
        self.smoothed_confidence = 0.0
        self.monitor_start_time = None
        # Window-locked capture
        self.meet_window_info = None
        self.meet_custom_title = None
        # Identity matching
        self.target_embedding = None
        self.target_lost_count = 0
        self.MAX_TARGET_LOST = 3
        self.SIMILARITY_THRESHOLD = 0.5
        self.MIN_FACE_SIZE = 50
        # Temporal stability
        self.confidence_history = []
        self.consecutive_high_confidence = 0
        self.SUSPICION_THRESHOLD = 0.85
        self.SUSPICION_COUNT = 3
        # Capture failure tracking
        self.monitor_capture_failures = 0
        self.MAX_CAPTURE_FAILURES = 10
        
        self.config = load_config()
        self.current_lang = self.config.get("language", "eng_Latn")
        self.lang_data = load_language_json(self.current_lang)
        self.ui_elements = {}
        
        self.build_ui()
        self.update_ui_language(self.current_lang)
        
        # Start queue polling
        self.poll_queues()
    
    # ==========================================================================
    # UI CONSTRUCTION
    # ==========================================================================
    
    def build_ui(self):
        """Build the main application UI — redesigned for professional research-grade look."""
        
        # =====================================================================
        # STYLE CONSTANTS
        # =====================================================================
        BG_MAIN = "#f4f4f4"        # Keep original main background
        BG_HEADER = "#1a1a1a"      # Keep original header dark
        BG_FOOTER = "#d0d0d0"      # Keep original footer
        BG_CONTROLS = "#e2e2e2"    # Slightly lighter controls for cleaner look
        BG_PANEL = "#fafafa"       # Subtle off-white for panels
        BG_ANALYSIS = "#f0f0f0"    # Analysis section
        
        FG_TITLE = "#ffffff"
        FG_SUBTITLE = "#e0e0e0"
        FG_ACCENT = "#FFD700"      # Gold accent for PI labels
        FG_BODY = "#2a2a2a"        # Strong body text
        FG_SECONDARY = "#555555"
        FG_PATENT = "#8B0000"      # Dark red for patent notice
        
        ff = "Noto Sans" if "Noto Sans" in tkfont.families() else "Arial Unicode MS"
        FONT_TITLE = (ff, 16, "bold")
        FONT_PROJECT = (ff, 11)
        FONT_PI = (ff, 10, "bold")
        FONT_PI_NAME = (ff, 10)
        FONT_SECTION = (ff, 11, "bold")
        FONT_BTN = (ff, 10)
        FONT_BTN_BOLD = (ff, 10, "bold")
        FONT_BODY = (ff, 10)
        FONT_LABEL = (ff, 10, "bold")
        FONT_CONFIDENCE = (ff, 18, "bold")
        FONT_RESULT = (ff, 16, "bold")
        FONT_FOOTER = (ff, 11, "bold")
        FONT_FOOTER_PATENT = (ff, 10, "bold")
        FONT_STATUS = (ff, 9)
        FONT_MONITOR = (ff, 9, "bold")
        
        # Common button styling
        BTN_PAD_Y = 4
        BTN_HEIGHT = 2
        BTN_WIDTH = 24
        
        # Helper for hover effects
        def _on_enter(e, bg_hover):
            e.widget.config(bg=bg_hover)
        def _on_leave(e, bg_normal):
            e.widget.config(bg=bg_normal)
        
        def make_button(parent, text, command, bg="#d0d0d0", fg="#1a1a1a",
                        font=FONT_BTN, width=BTN_WIDTH, bold=False,
                        hover_bg=None):
            """Create a styled button with hover effect."""
            if bold:
                font = FONT_BTN_BOLD
            if hover_bg is None:
                # Lighten bg by ~20% for hover
                try:
                    r = int(bg[1:3], 16)
                    g = int(bg[3:5], 16)
                    b = int(bg[5:7], 16)
                    hover_bg = "#{:02x}{:02x}{:02x}".format(
                        min(255, r + 30), min(255, g + 30), min(255, b + 30)
                    )
                except Exception:
                    hover_bg = bg
            
            btn = tk.Button(
                parent, text=text, command=command,
                bg=bg, fg=fg, font=font,
                width=width, height=BTN_HEIGHT,
                bd=0, relief=tk.FLAT,
                activebackground=hover_bg, activeforeground=fg,
                cursor="hand2",
                padx=8, pady=BTN_PAD_Y
            )
            btn.bind("<Enter>", lambda e: _on_enter(e, hover_bg))
            btn.bind("<Leave>", lambda e: _on_leave(e, bg))
            return btn
        
        # =====================================================================
        # HEADER (3-column: Logo | Text | Logo)
        # =====================================================================
        header_frame = tk.Frame(self.root, bg=BG_HEADER, pady=8)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        
        header_frame.columnconfigure(0, weight=0)
        header_frame.columnconfigure(1, weight=1)
        header_frame.columnconfigure(2, weight=0)
        
        # --- LEFT LOGO (25-35% larger: 60 → 80px) ---
        logo_dir = os.path.join(os.path.dirname(__file__), "assets")
        left_logo_path = os.path.join(logo_dir, "logo_left.png")
        try:
            left_img = Image.open(left_logo_path)
            left_img = left_img.resize((80, 80), Image.LANCZOS)
            self._left_logo_photo = ImageTk.PhotoImage(left_img)
            tk.Label(
                header_frame, image=self._left_logo_photo, bg=BG_HEADER
            ).grid(row=0, column=0, padx=(12, 8), pady=4, sticky="w")
        except Exception as e:
            print(f"[WARN] Could not load left logo: {e}")
            tk.Label(header_frame, text="", bg=BG_HEADER, width=10).grid(
                row=0, column=0, padx=(12, 8), pady=4, sticky="w"
            )
        
        # --- CENTER TEXT ---
        center_frame = tk.Frame(header_frame, bg=BG_HEADER)
        center_frame.grid(row=0, column=1, sticky="nsew", pady=2)
        
        lbl_title = tk.Label(
            center_frame,
            text=self.lang_data.get("ui_title", "SUDARSHAN-0.0 : A Tool to Reveal the Truth"),
            font=FONT_TITLE, bg=BG_HEADER, fg=FG_TITLE
        )
        lbl_title.pack(pady=(4, 2))
        if lbl_title is not None: self.ui_elements["ui_title"] = lbl_title
        
        lbl_sub = tk.Label(
            center_frame,
            text=self.lang_data.get("ui_subtitle", "Deepfake Detection and Explanation for Social Media Video Content  (DSAI2025-CS-1012)"),
            font=FONT_PROJECT, bg=BG_HEADER, fg=FG_SUBTITLE
        )
        lbl_sub.pack(pady=(0, 5))
        if lbl_sub is not None: self.ui_elements["ui_subtitle"] = lbl_sub
        
        # PI line
        pi_frame = tk.Frame(center_frame, bg=BG_HEADER)
        pi_frame.pack(pady=(0, 1))
        tk.Label(
            pi_frame, text="Principal Investigator: ",
            font=FONT_PI, bg=BG_HEADER, fg=FG_ACCENT
        ).pack(side=tk.LEFT)
        tk.Label(
            pi_frame,
            text="Dr. Mashhuda Glencross, The University Of Queensland (UoQ), Brisbane, Australia",
            font=FONT_PI_NAME, bg=BG_HEADER, fg=FG_TITLE
        ).pack(side=tk.LEFT)
        
        # Co-PI line
        copi_frame = tk.Frame(center_frame, bg=BG_HEADER)
        copi_frame.pack(pady=(0, 4))
        tk.Label(
            copi_frame, text="Co-PIs: ",
            font=FONT_PI, bg=BG_HEADER, fg=FG_ACCENT
        ).pack(side=tk.LEFT)
        tk.Label(
            copi_frame,
            text="Dr. Suresh Raikwar (TIET)  |  Dr. Kapil Rana (TIET)  |  Dr. Priyanka Singh (UoQ)",
            font=FONT_PI_NAME, bg=BG_HEADER, fg=FG_SUBTITLE
        ).pack(side=tk.LEFT)
        
        # --- RIGHT LOGO (25-35% larger: 60 → 80px) ---
        right_logo_path = os.path.join(logo_dir, "logo_right.png")
        try:
            right_img = Image.open(right_logo_path)
            right_img = right_img.resize((80, 80), Image.LANCZOS)
            self._right_logo_photo = ImageTk.PhotoImage(right_img)
            tk.Label(
                header_frame, image=self._right_logo_photo, bg=BG_HEADER
            ).grid(row=0, column=2, padx=(8, 12), pady=4, sticky="e")
        except Exception as e:
            print(f"[WARN] Could not load right logo: {e}")
            tk.Label(header_frame, text="", bg=BG_HEADER, width=10).grid(
                row=0, column=2, padx=(8, 12), pady=4, sticky="e"
            )
            
        # ==============================================================
        # Language Selector (Moved to Header)
        # ==============================================================
        header_frame.columnconfigure(3, weight=0)
        lang_frame = tk.Frame(header_frame, bg=BG_HEADER)
        lang_frame.grid(row=0, column=3, sticky="e", padx=(0, 20))
        
        lbl_lang = tk.Label(
            lang_frame, text="Language / भाषा",
            font=FONT_STATUS, bg=BG_HEADER, fg=FG_SUBTITLE
        )
        lbl_lang.pack(anchor="e")
        self.ui_elements["language_label"] = lbl_lang
        
        self.lang_combo = ttk.Combobox(lang_frame, values=get_available_languages(), state="readonly", width=12)
        self.lang_combo.set(self.current_lang)
        self.lang_combo.pack(anchor="e", pady=(2, 0))
        
        def on_lang_change(event):
            selected = self.lang_combo.get()
            self.config["language"] = selected
            save_config(self.config)
            self.current_lang = selected
            self.update_ui_language(selected)
            
        self.lang_combo.bind("<<ComboboxSelected>>", on_lang_change)

        
        # =====================================================================
        # FOOTER (larger text, tighter spacing, patent notice)
        # =====================================================================
        footer_frame = tk.Frame(self.root, bg=BG_FOOTER, pady=6)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        footer_text = (
            "Development Team: Karman Singh Talwar (UG)  |  Nikhil Sahotra (JRF)  |  "
            "Funded by CODSAI, TIET  |  Developed @ TIET © 2026 TIET & UoQ"
        )
        tk.Label(
            footer_frame, text=footer_text,
            font=FONT_FOOTER, bg=BG_FOOTER, fg="#1a1a1a",
            anchor="center", justify="center"
        ).pack(fill=tk.X, padx=8, pady=(2, 0))
        
        tk.Label(
            footer_frame,
            text="⚖  SUDARSHAN has been patented as per Indian Patent Law",
            font=FONT_FOOTER_PATENT, bg=BG_FOOTER, fg=FG_PATENT,
            anchor="center", justify="center"
        ).pack(fill=tk.X, padx=8, pady=(0, 2))
        
        # =====================================================================
        # MAIN CONTAINER (fill between header and footer)
        # =====================================================================
        main_frame = tk.Frame(self.root, bg=BG_MAIN)
        main_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=6, pady=4)
        
        # =====================================================================
        # LEFT PANEL — CONTROLS (compact, styled)
        # =====================================================================
        control_frame = tk.Frame(main_frame, width=230, bg=BG_CONTROLS, bd=0)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        control_frame.pack_propagate(False)
        
        # Section header
        lbl_ctrl = tk.Label(
            control_frame, text=self.lang_data.get("controls", "CONTROLS"),
            font=FONT_SECTION, bg=BG_CONTROLS, fg=FG_BODY,
            anchor="w"
        )
        lbl_ctrl.pack(fill=tk.X, padx=12, pady=(10, 6))
        if lbl_ctrl is not None: self.ui_elements["controls"] = lbl_ctrl
        
        # --- Primary Actions ---
        btn_upload_video = make_button(
            control_frame, self.lang_data.get("upload_video", "UPLOAD VIDEO"),
            self.upload_video, bold=True
        )
        btn_upload_video.pack(padx=10, pady=3, fill=tk.X)
        self.ui_elements["upload_video"] = btn_upload_video
        
        btn_upload_image = make_button(
            control_frame, self.lang_data.get("upload_image", "UPLOAD IMAGE"),
            self.upload_image, bold=True
        )
        btn_upload_image.pack(padx=10, pady=3, fill=tk.X)
        self.ui_elements["upload_image"] = btn_upload_image
        
        btn_verify_aadhaar = make_button(
            control_frame, self.lang_data.get("verify_aadhaar", "VERIFY AADHAAR"),
            self.verify_aadhaar, bold=True, bg="#ffcc00", fg="#1a1a1a"
        )
        btn_verify_aadhaar.pack(padx=10, pady=3, fill=tk.X)
        self.ui_elements["verify_aadhaar"] = btn_verify_aadhaar
        
        # --- Top / Live Separator ---
        tk.Frame(control_frame, bg="#b0b0b0", height=1).pack(
            fill=tk.X, padx=14, pady=6
        )
        
        # --- Live / Stream ---
        btn_webcam = make_button(
            control_frame, self.lang_data.get("live_webcam", "LIVE WEBCAM"),
            self.start_live
        )
        btn_webcam.pack(padx=10, pady=3, fill=tk.X)
        self.ui_elements["live_webcam"] = btn_webcam
        
        lbl_stream = tk.Label(
            control_frame, text=self.lang_data.get("live_stream_url", "Live Stream URL"),
            font=FONT_STATUS, bg=BG_CONTROLS, fg=FG_SECONDARY
        )
        lbl_stream.pack(padx=12, pady=(6, 2), anchor="w")
        if lbl_stream is not None: self.ui_elements["live_stream_url"] = lbl_stream
        
        self.url_entry = tk.Entry(
            control_frame, width=26, font=FONT_STATUS,
            bd=1, relief=tk.SOLID
        )
        self.url_entry.pack(padx=12, pady=(0, 3), fill=tk.X)
        
        btn_url = make_button(
            control_frame, self.lang_data.get("start_url_stream", "START URL STREAM"),
            self.start_url_stream
        )
        btn_url.pack(padx=10, pady=3, fill=tk.X)
        self.ui_elements["start_url_stream"] = btn_url
        
        # ==============================================================
        # Interview Monitor Mode Controls
        # ==============================================================
        tk.Frame(control_frame, bg="#b0b0b0", height=1).pack(
            fill=tk.X, padx=14, pady=6
        )
        
        lbl_monitor = tk.Label(
            control_frame, text=self.lang_data.get("interview_monitor", "Interview Monitor"),
            font=FONT_MONITOR, bg=BG_CONTROLS, fg=FG_BODY
        )
        lbl_monitor.pack(padx=12, pady=(2, 3), anchor="w")
        if lbl_monitor is not None: self.ui_elements["interview_monitor"] = lbl_monitor
        
        self.monitor_url_entry = tk.Entry(
            control_frame, width=26, font=FONT_STATUS,
            bd=1, relief=tk.SOLID
        )
        self.monitor_url_entry.pack(padx=12, pady=2, fill=tk.X)
        self.monitor_url_entry.insert(0, "Paste meeting URL here")
        
        btn_meet = make_button(
            control_frame, self.lang_data.get("open_meet", "OPEN MEET"),
            self._open_meet_url
        )
        btn_meet.pack(padx=10, pady=2, fill=tk.X)
        self.ui_elements["open_meet"] = btn_meet
        
        btn_detect = make_button(
            control_frame, self.lang_data.get("detect_meet_window", "DETECT MEET WINDOW"),
            self._detect_meet_window
        )
        btn_detect.pack(padx=10, pady=2, fill=tk.X)
        self.ui_elements["detect_meet_window"] = btn_detect
        
        btn_face = make_button(
            control_frame, self.lang_data.get("select_target_face", "SELECT TARGET FACE"),
            self._select_target_face
        )
        btn_face.pack(padx=10, pady=2, fill=tk.X)
        self.ui_elements["select_target_face"] = btn_face
        
        # Monitoring indicator
        self.monitor_indicator = tk.Label(
            control_frame, text="● Monitoring Stopped",
            font=FONT_MONITOR, bg=BG_CONTROLS, fg="#cc0000"
        )
        self.monitor_indicator.pack(pady=2)
        
        # Duration timer
        self.monitor_timer_label = tk.Label(
            control_frame, text="Duration: --:--:--",
            font=FONT_STATUS, bg=BG_CONTROLS, fg=FG_SECONDARY
        )
        self.monitor_timer_label.pack(pady=1)
        # ==============================================================
        
        # --- STOP ---
        tk.Frame(control_frame, bg="#b0b0b0", height=1).pack(
            fill=tk.X, padx=14, pady=5
        )
        
        btn_stop = make_button(
            control_frame, self.lang_data.get("stop_stream", "STOP STREAM"),
            self.stop_stream, bg="#cc3333", fg="white", bold=True,
            hover_bg="#e74c3c"
        )
        btn_stop.pack(padx=10, pady=3, fill=tk.X)
        self.ui_elements["stop_stream"] = btn_stop
        
        # --- Status / Progress ---
        tk.Frame(control_frame, bg="#b0b0b0", height=1).pack(
            fill=tk.X, padx=14, pady=5
        )
        
        lbl_status = tk.Label(
            control_frame, text=self.lang_data.get("status", "Status"),
            font=FONT_LABEL, bg=BG_CONTROLS, fg=FG_BODY
        )
        lbl_status.pack(padx=12, anchor="w", pady=(2, 0))
        if lbl_status is not None: self.ui_elements["status"] = lbl_status
        
        self.ui_elements["ready"] = tk.Label(
            control_frame, text=self.lang_data.get("ready", "Ready"),

            font=FONT_STATUS, bg=BG_CONTROLS, fg=FG_SECONDARY,
            wraplength=200, justify="left", anchor="w"
        )
        self.status_label = self.ui_elements["ready"]
        self.status_label.pack(padx=12, anchor="w", pady=(1, 3))
        
        # Styled progress bar
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor='#c8c8c8', background='#3d5a80',
            thickness=10
        )
        self.progress_bar = ttk.Progressbar(
            control_frame, mode="determinate", length=200,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(padx=12, pady=(0, 8))
        
        # =====================================================================
        # RIGHT PANEL — VIDEO + ANALYSIS
        # =====================================================================
        right_panel = tk.Frame(main_frame, bg=BG_MAIN)
        right_panel.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        # --- VIDEO SECTION (expanded, subtle borders) ---
        video_section = tk.Frame(right_panel, bg=BG_MAIN)
        video_section.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 4))
        video_section.columnconfigure(0, weight=1, uniform="group1")
        video_section.columnconfigure(1, weight=1, uniform="group1")
        video_section.rowconfigure(0, weight=0)
        video_section.rowconfigure(1, weight=1)
        
        # Input Video Panel
        input_header = tk.Label(
            video_section, text=self.lang_data.get("input_video_display", "Input Video Display"),
            font=FONT_SECTION, bg=BG_MAIN, fg=FG_BODY
        )
        input_header.grid(row=0, column=0, sticky="w", padx=8, pady=(2, 2))
        self.ui_elements["input_video_display"] = input_header
        
        input_frame = tk.Frame(
            video_section, bg=BG_PANEL,
            bd=1, relief=tk.SOLID,
            highlightbackground="#c0c0c0", highlightthickness=1
        )
        input_frame.grid(row=1, column=0, sticky="nsew", padx=(4, 2), pady=(0, 2))
        
        self.input_label = tk.Label(input_frame, bg="#111111")
        self.input_label.pack(expand=True, fill=tk.BOTH, padx=3, pady=3)
        
        # Output / Heatmap Panel
        output_header = tk.Label(
            video_section, text=self.lang_data.get("explainability_output", "Explainability Output (Heatmaps)"),
            font=FONT_SECTION, bg=BG_MAIN, fg=FG_BODY
        )
        output_header.grid(row=0, column=1, sticky="w", padx=8, pady=(2, 2))
        self.ui_elements["explainability_output"] = output_header
        
        output_frame = tk.Frame(
            video_section, bg=BG_PANEL,
            bd=1, relief=tk.SOLID,
            highlightbackground="#c0c0c0", highlightthickness=1
        )
        output_frame.grid(row=1, column=1, sticky="nsew", padx=(2, 4), pady=(0, 2))
        
        self.output_label = tk.Label(output_frame, bg="#111111")
        self.output_label.pack(expand=True, fill=tk.BOTH, padx=3, pady=3)
        
        # =====================================================================
        # ANALYSIS SECTION (expanded, better hierarchy)
        # =====================================================================
        analysis_section = tk.Frame(
            right_panel, bg=BG_ANALYSIS,
            bd=1, relief=tk.SOLID,
            highlightbackground="#c0c0c0", highlightthickness=1,
            height=200
        )
        analysis_section.pack_propagate(False)
        analysis_section.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(2, 0))
        
        analysis_section.columnconfigure(0, weight=4)
        analysis_section.columnconfigure(1, weight=1)
        analysis_section.rowconfigure(1, weight=1)
        
        # --- Explanation (column 0) ---
        lbl_exp = tk.Label(
            analysis_section, text=self.lang_data.get("explanation", "Explanation"),
            font=FONT_SECTION, bg=BG_ANALYSIS, fg=FG_BODY
        )
        lbl_exp.grid(row=0, column=0, sticky="w", padx=10, pady=(6, 2))
        if lbl_exp is not None: self.ui_elements["explanation"] = lbl_exp
        
        self.explanation = tk.Text(
            analysis_section, height=8, width=60,
            font=FONT_BODY, wrap=tk.WORD,
            bd=1, relief=tk.SOLID,
            bg="#ffffff", fg=FG_BODY,
            padx=6, pady=4
        )
        self.explanation.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        
        # --- Confidence (column 1) ---
        confidence_frame = tk.Frame(analysis_section, bg=BG_ANALYSIS)
        confidence_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(4, 10), pady=6)
        
        lbl_conf = tk.Label(
            confidence_frame, text=self.lang_data.get("confidence", "Confidence"),
            font=FONT_LABEL, bg=BG_ANALYSIS, fg=FG_SECONDARY
        )
        lbl_conf.pack(anchor="center", pady=(8, 4))
        if lbl_conf is not None: self.ui_elements["confidence"] = lbl_conf
        
        self.confidence = tk.Entry(
            confidence_frame, width=10,
            font=FONT_CONFIDENCE, justify="center",
            bd=1, relief=tk.SOLID,
            fg="#1a1a1a", bg="#ffffff"
        )
        self.confidence.pack(anchor="center", pady=4)
        
        # Result label (FAKE/REAL)
        self.result_label = tk.Label(
            confidence_frame, text="—",
            font=FONT_RESULT, bg=BG_ANALYSIS, fg=FG_SECONDARY
        )
        self.result_label.pack(anchor="center", pady=(6, 4))


    def update_ui_language(self, lang_code):
        """Dynamically swap UI strings across all packed/gridded widgets."""
        self.lang_data = load_language_json(lang_code)
        
        debug_mode = False
        
        for key, widget in self.ui_elements.items():
            if widget is None:
                continue
            
            if not hasattr(widget, "config"):
                continue
            
            if key not in self.lang_data:
                continue
                
            if debug_mode:
                print(f"Updating: {key}, Widget: {type(widget)}")
                
            try:
                widget.config(text=self.lang_data[key])
            except Exception:
                continue
        
        # Manually update explanation placeholder if empty
        current_explanation = self.explanation.get("1.0", tk.END).strip()
        if not current_explanation or "Analyzing" in current_explanation:
            self.explanation.delete("1.0", tk.END)
            self.explanation.insert(tk.END, self.lang_data.get("ready", "Ready") + "\n")

    
    # ==========================================================================
    # QUEUE POLLING (THREAD-SAFE GUI UPDATES)
    # ==========================================================================
    
    def poll_queues(self):
        """
        Poll message queues for thread-safe GUI updates.
        
        This runs every 100ms to check for progress updates and results
        from the background inference thread.
        """
        # Process progress updates
        try:
            while True:
                stage, current, total = self.progress_queue.get_nowait()
                self.update_progress(stage, current, total)
        except queue.Empty:
            pass
        
        # Process inference results
        try:
            while True:
                result = self.result_queue.get_nowait()
                self.handle_inference_result(result)
        except queue.Empty:
            pass
        
        # ==============================================================
        # NEW: Process Interview Monitor messages
        # ==============================================================
        try:
            while True:
                msg = self.monitor_queue.get_nowait()
                if isinstance(msg, tuple) and len(msg) == 2:
                    msg_type, data = msg
                    if msg_type == "frame":
                        self._handle_monitor_frame(data)
                    elif msg_type == "result":
                        self._handle_monitor_result(data)
        except queue.Empty:
            pass
        
        # Update monitor duration timer
        if self.monitoring_active and self.monitor_start_time:
            elapsed = time.time() - self.monitor_start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.monitor_timer_label.config(
                text=f"Duration: {h:02d}:{m:02d}:{s:02d}"
            )
        # ==============================================================
        
        # Schedule next poll
        self.root.after(100, self.poll_queues)
    
    def update_progress(self, stage: str, current: int, total: int):
        """Update progress bar and status label."""
        self.status_label.config(text=stage)
        progress_pct = (current / total) * 100 if total > 0 else 0
        self.progress_bar["value"] = progress_pct
    
    # ==========================================================================
    # IMAGE UPLOAD (STATIC)
    # ==========================================================================
    
    # ==========================================================================
    # IMAGE UPLOAD (STATIC)
    # ==========================================================================
    
    def upload_image(self):
        """Handle static image upload."""
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        if not path:
            return
        
        # Stop any running streams
        self.stop_stream()
        
        # Display image with aspect ratio preserved
        image = Image.open(path)
        image.thumbnail((400, 300)) # Resize to fit within bounds
        photo = ImageTk.PhotoImage(image)
        
        self.input_label.config(image=photo)
        self.input_label.image = photo
        self.output_label.config(image=photo) # Mirror to output for static
        self.output_label.image = photo
        
        self.explanation.delete("1.0", tk.END)
        self.explanation.insert(tk.END, "Analyzing image...\n")
        self.confidence.delete(0, tk.END)
        self.result_label.config(text="...", fg="#333")
        
        # Run inference in background
        threading.Thread(target=self._run_image_thread, args=(path,), daemon=True).start()

    def _run_image_thread(self, path):
        """Background thread for image inference."""
        try:
            result = run_image_pipeline(path)
            self.result_queue.put(result)
        except Exception as e:
            self.result_queue.put({"label": "ERROR", "explanation": str(e)})

    # ==========================================================================
    # AADHAAR VERIFICATION (SINGLE-SHOT)
    # ==========================================================================
    
    def verify_aadhaar(self):
        """Handle offline Aadhaar verification input."""
        if self.inference_running:
            messagebox.showwarning(
                "Busy",
                "Inference is already running. Please wait."
            )
            return
            
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.png *.jpeg *.webp")]
        )
        if not path:
            return
            
        self.stop_stream()
        
        # Display image identically mapped
        image = Image.open(path)
        image.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(image)
        self.input_label.config(image=photo)
        self.input_label.image = photo
        
        # Reset output 
        self.output_label.config(image="")
        self.explanation.delete("1.0", tk.END)
        self.explanation.insert(tk.END, "Verifying Aadhaar (Offline)...\n")
        self.confidence.delete(0, tk.END)
        self.result_label.config(text="...", fg="#333")
        
        self.inference_running = True
        threading.Thread(target=self._run_aadhaar_thread, args=(path,), daemon=True).start()

    def _run_aadhaar_thread(self, path):
        """Background thread for Aadhaar inference."""
        try:
            self.progress_queue.put(("Running Checksum & Graph Analyzers...", 1, 2))
            result = run_aadhaar_verification(path, lang=self.current_lang)
            result["is_aadhaar"] = True
            self.progress_queue.put(("Analysis Complete", 2, 2))
            self.result_queue.put(result)
        except Exception as e:
            self.result_queue.put({"label": "ERROR", "explanation": str(e), "is_aadhaar": True})
        finally:
            self.inference_running = False
    
    # ==========================================================================
    # VIDEO UPLOAD + INFERENCE
    # ==========================================================================
    
    def upload_video(self):
        """
        Handle video upload and start inference.
        
        - Plays raw video in Input panel
        - Runs inference in background thread
        - Updates Output panel with heatmaps after inference
        """
        if self.inference_running:
            messagebox.showwarning(
                "Busy",
                "Inference is already running. Please wait."
            )
            return
        
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return
        
        # Convert to absolute path for Windows safety
        path = os.path.abspath(path)
        
        self.current_video_path = path
        self.stop_stream()
        
        # Reset heatmap state
        self.heatmap_frames = {}
        self.all_frame_paths = []
        self.heatmap_running = False
        
        # Start raw video playback in input panel
        self.cap = cv2.VideoCapture(path)
        self.running = True
        self.play_input_video()
        
        # Clear output panel
        self.output_label.config(image="")
        self.output_label.config(text="Waiting for analysis...", fg="#666")
        
        # Reset UI
        self.explanation.delete("1.0", tk.END)
        self.explanation.insert(tk.END, "Starting analysis...\n")
        self.confidence.delete(0, tk.END)
        self.confidence.insert(0, "--")
        self.result_label.config(text="...", fg="#333")
        self.progress_bar["value"] = 0
        
        # Start inference in background thread
        self.inference_running = True
        threading.Thread(
            target=self.run_inference_thread,
            daemon=True
        ).start()
    
    def run_inference_thread(self):
        """
        Background thread for inference with explainability.
        
        Uses queues for thread-safe GUI updates.
        """
        try:
            def progress_callback(stage, current, total):
                """Send progress updates to main thread via queue."""
                self.progress_queue.put((stage, current, total))
            
            # Run full pipeline with explainability
            result = run_video_pipeline(
                video_path=self.current_video_path,
                generate_explanations=True,
                progress_callback=progress_callback
            )
            
            # Send result to main thread
            self.result_queue.put(result)
            
        except Exception as e:
            # Send error result
            self.result_queue.put({
                "label": "ERROR",
                "confidence": 0.0,
                "explanation": f"Pipeline error: {str(e)}",
                "heatmap_frames": {},
                "all_frame_paths": []
            })
        
        finally:
            self.inference_running = False
    
    def handle_inference_result(self, result: Dict):
        """
        Handle inference result on main thread.
        
        Updates GUI with results and starts heatmap playback.
        """
        label = result.get("label", "ERROR")
        confidence = result.get("confidence", 0.0)
        explanation = result.get("explanation", "")
        heatmap_frames = result.get("heatmap_frames", {})
        all_frame_paths = result.get("all_frame_paths", [])
        is_aadhaar = result.get("is_aadhaar", False)
        
        # Anti-spam guard
        if not hasattr(self, "_last_raw_explanation"):
            self._last_raw_explanation = ""
            self._last_translation_lang = ""
            
        same_text = (explanation == self._last_raw_explanation)
        same_lang = (self._last_translation_lang == self.current_lang)
        
        if same_text and same_lang and not is_aadhaar:
            # Skip updating text repeatedly unless Aadhaar pipeline triggered anew
            pass
        else:
            self._last_raw_explanation = explanation
            self._last_translation_lang = self.current_lang
            
            # Threaded logic for UI AI Output Translation
            def _apply_trans():
                trans_exp = smart_translate(explanation, self.current_lang)
                # Ensure Tkinter thread safety for UI update
                def _update():
                    self.explanation.delete("1.0", tk.END)
                    self.explanation.insert(tk.END, trans_exp)
                self.root.after(0, _update)

            import threading
            if explanation and self.current_lang != "eng_Latn":
                self.explanation.delete("1.0", tk.END)
                self.explanation.insert(tk.END, "Translating... \n")
                threading.Thread(target=_apply_trans, daemon=True).start()
            else:
                self.explanation.delete("1.0", tk.END)
                self.explanation.insert(tk.END, explanation)

        
        # Update confidence
        self.confidence.delete(0, tk.END)
        self.confidence.insert(0, f"{confidence:.2%}")
        
        # Update result label with color coding
        if label == "FAKE":
            self.result_label.config(text="\u26a0\ufe0f FAKE", fg="#cc0000")
        elif label == "REAL":
            self.result_label.config(text="\u2713 REAL", fg="#00aa00")
        elif label == "NO FACE":
            self.result_label.config(text="NO FACE", fg="#ff9900")
        else:
            self.result_label.config(text="ERROR", fg="#666666")
        
        # Update progress
        self.progress_bar["value"] = 100
        self.status_label.config(text="Analysis complete")
        
        # Store heatmap data for playback
        self.heatmap_frames = heatmap_frames
        self.all_frame_paths = all_frame_paths
        
        # Start heatmap playback if available
        if heatmap_frames:
            if is_aadhaar:
                self.heatmap_running = False
                out_path = heatmap_frames[0]
                try:
                    img = Image.open(out_path)
                    img.thumbnail((400, 300))
                    photo = ImageTk.PhotoImage(img)
                    self.output_label.config(image=photo)
                    self.output_label.image = photo
                except Exception:
                    self.output_label.config(text="Failed to load output image", fg="#cc0000", image="")
            else:
                self.start_heatmap_playback()
        else:
            # No heatmaps - show message in output panel unless there is already a mirrored image
            if not getattr(self.output_label, "image", None):
                self.output_label.config(text="No output visuals", fg="#666", image="")
            else:
                self.output_label.config(text="")
    
    # ==========================================================================
    # HEATMAP PLAYBACK
    # ==========================================================================
    
    def start_heatmap_playback(self):
        """Start playing heatmap frames in output panel."""
        if not self.heatmap_frames:
            return
        
        self.heatmap_frame_idx = 0
        self.heatmap_running = True
        self.play_heatmap_frame()
    
    def play_heatmap_frame(self):
        """
        Play next heatmap frame in output panel.
        
        Loops through representative frames with temporal alignment.
        """
        if not self.heatmap_running or not self.heatmap_frames:
            return
        
        # Get sorted frame indices for temporal order
        frame_indices = sorted(self.heatmap_frames.keys())
        
        if self.heatmap_frame_idx >= len(frame_indices):
            self.heatmap_frame_idx = 0  # Loop playback
        
        frame_idx = frame_indices[self.heatmap_frame_idx]
        heatmap_path = self.heatmap_frames[frame_idx]
        
        try:
            if os.path.exists(heatmap_path):
                frame = cv2.imread(heatmap_path)
                if frame is not None:
                    # Resize to fit fixed window
                    frame = cv2.resize(frame, (400, 300))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = ImageTk.PhotoImage(Image.fromarray(rgb))
                    
                    self.output_label.config(image=img, text="")
                    self.output_label.image = img
        except Exception as e:
            print(f"[WARN] Failed to display heatmap: {e}")
        
        self.heatmap_frame_idx += 1
        
        # Schedule next frame (slower playback for analysis)
        self.root.after(500, self.play_heatmap_frame)
    
    # ==========================================================================
    # INPUT VIDEO PLAYBACK
    # ==========================================================================
    
    def play_input_video(self):
        """Play raw video in input panel."""
        if not self.running or not self.cap or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Resize logic consistent with fixed frame
            frame = cv2.resize(frame, (400, 300))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            
            self.input_label.config(image=img)
            self.input_label.image = img
        else:
            # Video ended - loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.root.after(30, self.play_input_video)
    
    # ==========================================================================
    # LIVE STREAM
    # ==========================================================================
    
    # ==========================================================================
    # LIVE STREAM
    # ==========================================================================
    
    def start_live(self):
        """Start live webcam stream with background inference."""
        self.stop_stream()
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return
                
            self.running = True
            self.frame_buffer = []
            self.frame_counter = 0
            
            # Start capture thread
            self.webcam_thread = threading.Thread(target=self._webcam_capture_loop, daemon=True)
            self.webcam_thread.start()
            
            self.explanation.delete("1.0", tk.END)
            self.explanation.insert(
                tk.END,
                "Live Analysis Active.\n\n"
                "Inference runs every ~32 frames (approx 1 sec).\n"
                "Explainability is disabled for real-time performance."
            )
            self.status_label.config(text="Live analysis active")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _webcam_capture_loop(self):
        """
        Continuous webcam capture loop.
        - Updates UI with live video
        - Buffers frames
        - Triggers inference periodically
        """
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Update UI (Mirror to input)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(rgb, (400, 300)) # Keep consistent
            img = ImageTk.PhotoImage(Image.fromarray(small_frame))
            
            # Schedule UI update on main thread
            self.root.after(0, lambda i=img: self.input_label.config(image=i))
            self.root.after(0, lambda: setattr(self.input_label, 'image', img)) # Keep ref
            
            # Buffer frame for inference
            with self.webcam_lock:
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > self.BUFFER_SIZE:
                    self.frame_buffer.pop(0)
            
            self.frame_counter += 1
            
            # Trigger inference every X frames (e.g. 30)
            if self.frame_counter % 30 == 0:
                # Copy buffer snapshot to avoid threading issues
                with self.webcam_lock:
                    snapshot = list(self.frame_buffer)
                
                if len(snapshot) >= 16: # Min frames needed
                    threading.Thread(
                        target=self._run_webcam_inference,
                        args=(snapshot,),
                        daemon=True
                    ).start()
            
            # Cap frame rate slightly to save CPU
            cv2.waitKey(30)

    def _run_webcam_inference(self, frames):
        """Background inference task."""
        if self.inference_running:
            return # Skip if previous still running
            
        self.inference_running = True
        try:
            result = run_webcam_pipeline(frames)
            self.result_queue.put(result)
        finally:
            self.inference_running = False

    def play_live_mirror(self):
        """Deprecated: Replaced by threaded loop"""
        pass
    
    def play_live_mirror(self):
        """Mirror live feed to output panel."""
        if not self.running or not self.cap or not self.cap.isOpened():
            return
        
        # Get current position
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (400, 300))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            
            self.output_label.config(image=img)
            self.output_label.image = img
        
        self.root.after(30, self.play_live_mirror)
    
    def start_url_stream(self):
        """Start YouTube/URL stream."""
        url = self.url_entry.get().strip()
        if not url:
            return
        
        try:
            self.stop_stream()
            result = subprocess.run(
                ["yt-dlp", "-f", "best", "-g", url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            stream_url = result.stdout.strip()
            
            if not stream_url:
                raise ValueError("Could not extract stream URL")
            
            self.cap = cv2.VideoCapture(stream_url)
            self.running = True
            self.play_input_video()
            
            self.explanation.delete("1.0", tk.END)
            self.explanation.insert(
                tk.END,
                "URL stream active.\n\n"
                "Note: Real-time inference is not supported. "
                "Download and upload a video for deepfake analysis."
            )
            self.confidence.delete(0, tk.END)
            self.confidence.insert(0, "--")
            self.result_label.config(text="STREAM", fg="#0066cc")
            self.status_label.config(text="URL stream active")
            
        except subprocess.TimeoutExpired:
            messagebox.showerror("Stream Error", "Timeout while extracting stream URL")
        except Exception as e:
            messagebox.showerror("Stream Error", str(e))
    
    # ==========================================================================
    # INTERVIEW MONITOR MODE (Window-Locked + Identity-Based)
    # ==========================================================================
    
    def _open_meet_url(self):
        """
        Open meeting URL in default browser.
        User must join the meeting before detecting the window.
        """
        url = self.monitor_url_entry.get().strip()
        if not url or url == "Paste meeting URL here":
            messagebox.showwarning(
                "Missing URL",
                "Please paste a meeting URL (e.g. Google Meet link)."
            )
            return
        
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Browser Error", f"Could not open URL: {e}")
            return
        
        self.status_label.config(
            text="Join the meeting, then click 'Detect Meet Window'."
        )
    
    def _detect_meet_window(self):
        """
        Detect Google Meet window and capture a preview frame.
        
        Searches for windows with 'Meet' in the title. If found,
        stores the window bounds and captures a preview frame for
        the Input Video panel. If not found, shows error.
        """
        if self.monitoring_active:
            messagebox.showwarning("Busy", "Monitor is already active.")
            return
        
        self.status_label.config(text="Searching for Meet window...")
        self.root.update()
        
        window_info = find_meet_window(self.meet_custom_title)
        
        if window_info is None:
            # Offer manual title input
            custom = messagebox.askretrycancel(
                "Window Not Found",
                "No Google Meet window detected.\n\n"
                "Make sure the meeting is open and NOT minimized.\n\n"
                "Click Retry to search again, or Cancel to enter a "
                "custom window title."
            )
            if custom:
                # Retry
                window_info = find_meet_window(self.meet_custom_title)
                if window_info is None:
                    # Ask for custom title
                    import tkinter.simpledialog as simpledialog
                    title = simpledialog.askstring(
                        "Custom Window Title",
                        "Enter part of the meeting window title:",
                        parent=self.root
                    )
                    if title:
                        self.meet_custom_title = title
                        window_info = find_meet_window(title)
            
            if window_info is None:
                self.status_label.config(text="Meet window not found.")
                return
        
        self.meet_window_info = window_info
        self.status_label.config(
            text=f"Window: {window_info['title'][:40]}..."
        )
        
        # Capture preview frame and display it
        preview = capture_window(window_info)
        if preview is not None:
            self._handle_monitor_frame(preview)
            self.explanation.delete("1.0", tk.END)
            self.explanation.insert(
                tk.END,
                f"Meet Window Detected\n\n"
                f"Title: {window_info['title']}\n"
                f"Size: {window_info['width']}x{window_info['height']}\n"
                f"Position: ({window_info['left']}, {window_info['top']})\n\n"
                f"Click 'Select Target Face' to choose the person to monitor."
            )
        
        messagebox.showinfo(
            "Window Detected",
            f"Found: {window_info['title'][:50]}\n\n"
            "Click 'Select Target Face' to choose the interviewee."
        )
    
    def _select_target_face(self):
        """
        Detect all faces in the Meet window, let user select the target,
        extract FaceNet embedding, and start monitoring.
        
        Flow:
        1. Verify Meet window is detected
        2. Capture fresh frame from window
        3. MTCNN: detect all faces
        4. Show numbered face overlay in input panel
        5. User enters target face number
        6. FaceNet: extract target embedding
        7. Start dual-thread monitoring
        """
        if self.monitoring_active:
            messagebox.showwarning("Busy", "Monitor is already active.")
            return
        
        if self.meet_window_info is None:
            messagebox.showwarning(
                "No Window",
                "Click 'Detect Meet Window' first."
            )
            return
        
        # Re-read window bounds (may have moved)
        window_info = find_meet_window(self.meet_custom_title)
        if window_info is None:
            messagebox.showerror(
                "Window Lost",
                "The Meet window is no longer visible. "
                "Make sure it is open and not minimized."
            )
            self.meet_window_info = None
            return
        self.meet_window_info = window_info
        
        self.status_label.config(text="Detecting faces...")
        self.root.update()
        
        # Capture fresh frame
        frame_bgr = capture_window(window_info)
        if frame_bgr is None:
            messagebox.showerror("Capture Error", "Failed to capture window.")
            return
        
        # Detect all faces via identity MTCNN
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(frame_rgb)
        
        faces = detect_and_embed(pil_image)
        
        if not faces:
            messagebox.showwarning(
                "No Faces",
                "No faces detected in the Meet window.\n"
                "Make sure the participant's camera is on and face is visible."
            )
            return
        
        # Draw numbered bounding boxes on the frame
        display_frame = frame_bgr.copy()
        for i, face_info in enumerate(faces):
            x1, y1, x2, y2 = face_info["box"]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame, str(i + 1),
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
        
        # Show numbered faces in input panel
        self._handle_monitor_frame(display_frame)
        
        # Ask user to select face number
        import tkinter.simpledialog as simpledialog
        face_num = simpledialog.askinteger(
            "Select Target Face",
            f"{len(faces)} face(s) detected.\n"
            f"Enter the number (1-{len(faces)}) of the person to monitor:",
            parent=self.root,
            minvalue=1,
            maxvalue=len(faces)
        )
        
        if face_num is None:
            self.status_label.config(text="Face selection cancelled.")
            return
        
        # Store target embedding
        selected_face = faces[face_num - 1]
        self.target_embedding = selected_face["embedding"]
        
        # Stop any existing streams
        self.stop_stream()
        
        # Reset state and start monitoring
        self.monitoring_active = True
        self.monitor_buffer = []
        self.smoothed_confidence = 0.0
        self.confidence_history = []
        self.consecutive_high_confidence = 0
        self.target_lost_count = 0
        self.monitor_capture_failures = 0
        self.monitor_start_time = time.time()
        
        # Update UI
        self.monitor_indicator.config(
            text="\u25cf Monitoring Active", fg="#00aa00"
        )
        self.status_label.config(text="Monitoring Interview...")
        self.explanation.delete("1.0", tk.END)
        self.explanation.insert(
            tk.END,
            f"Interview Monitor Active\n\n"
            f"Window: {window_info['title'][:40]}\n"
            f"Target: Face #{face_num}\n"
            f"Identity Threshold: {self.SIMILARITY_THRESHOLD}\n\n"
            "Capturing window and running identity-matched inference every ~2s.\n"
            "Click STOP STREAM to end monitoring."
        )
        self.confidence.delete(0, tk.END)
        self.confidence.insert(0, "--")
        self.result_label.config(text="MONITORING", fg="#2266aa")
        
        # Start dual threads
        self.monitor_capture_thread = threading.Thread(
            target=self._monitor_capture_loop, daemon=True
        )
        self.monitor_inference_thread = threading.Thread(
            target=self._monitor_inference_loop, daemon=True
        )
        self.monitor_capture_thread.start()
        self.monitor_inference_thread.start()
    
    # --------------------------------------------------------------------------
    # Capture Thread (lightweight â€” NO neural networks)
    # --------------------------------------------------------------------------
    
    def _monitor_capture_loop(self):
        """
        Window-locked capture thread (Thread 1).
        
        Re-reads window bounds every cycle via find_meet_window().
        Captures the window via mss -> appends raw frame to buffer.
        Puts ("frame", frame) into monitor_queue for live display.
        
        NO MTCNN. NO FaceNet. Just pixel capture.
        
        Stops on: window lost, or 10 consecutive capture failures.
        """
        self.monitor_capture_failures = 0
        
        while self.monitoring_active:
            try:
                # Re-read window bounds every cycle (handles move/resize)
                bounds = find_meet_window(self.meet_custom_title)
                if bounds is None:
                    # Window lost â€” stop monitoring
                    print("[WARN] Meet window lost. Stopping monitor.")
                    self.monitoring_active = False
                    self.monitor_queue.put(("result", {
                        "label": "ERROR",
                        "confidence": 0.0,
                        "smoothed_confidence": 0.0,
                        "explanation": (
                            "Meet window lost.\n"
                            "The window may have been closed or minimized.\n\n"
                            "Monitoring has been stopped automatically."
                        ),
                        "stability": None,
                        "suspicion_alert": False,
                        "quality_warning": None
                    }))
                    break
                
                # Update stored window info
                self.meet_window_info = bounds
                
                frame = capture_window(bounds)
                if frame is not None:
                    # [FORENSIC DEBUG] Capture Validation
                    # Save 1 frame every ~2s (assuming 30fps) to verify window content
                    if not hasattr(self, "_debug_frame_count"):
                        self._debug_frame_count = 0
                    self._debug_frame_count += 1
                    
                    if self._debug_frame_count % 60 == 0:
                        try:
                            ts = int(time.time())
                            filename = f"debug_capture_{ts}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"[DEBUG] Saved capture dump: {filename} (check for black frames/wrong window)")
                        except Exception as e:
                            print(f"[DEBUG] Save failed: {e}")

                    with self.monitor_lock:
                        self.monitor_buffer.append(frame)
                        if len(self.monitor_buffer) > 32:
                            self.monitor_buffer.pop(0)
                    
                    # Live display
                    self.monitor_queue.put(("frame", frame))
                    
                    self.monitor_capture_failures = 0
                else:
                    self.monitor_capture_failures += 1
            except Exception as e:
                print(f"[ERROR] Monitor capture error: {e}")
                self.monitor_capture_failures += 1
            
            if self.monitor_capture_failures >= self.MAX_CAPTURE_FAILURES:
                print(f"[WARN] {self.MAX_CAPTURE_FAILURES} capture failures.")
                self.monitoring_active = False
                self.monitor_queue.put(("result", {
                    "label": "ERROR",
                    "confidence": 0.0,
                    "smoothed_confidence": 0.0,
                    "explanation": (
                        "Screen capture failed repeatedly.\n"
                        "Monitoring has been stopped automatically."
                    ),
                    "stability": None,
                    "suspicion_alert": False,
                    "quality_warning": None
                }))
                break
            
            time.sleep(0.03)
    
    # --------------------------------------------------------------------------
    # Inference Thread (identity matching + ViT â€” every 2 seconds)
    # --------------------------------------------------------------------------
    
    def _monitor_inference_loop(self):
        """
        Periodic inference thread (Thread 2).
        
        Every ~2 seconds:
        1. Snapshot buffer (requires >= 16 frames)
        2. run_screen_buffer_pipeline(snapshot, target_embedding)
           - MTCNN + FaceNet identity matching runs here, not in capture
        3. Handle NO FACE -> target_lost_count, 3 retries
        4. EMA: 0.6 * new + 0.4 * old
        5. Temporal stability + suspicion alerts
        6. Queue result for GUI
        """
        while self.monitoring_active:
            time.sleep(2.0)
            
            if not self.monitoring_active:
                break
            
            with self.monitor_lock:
                snapshot = list(self.monitor_buffer)
            
            if len(snapshot) < 16:
                continue
            
            try:
                result = run_screen_buffer_pipeline(
                    snapshot,
                    target_embedding=self.target_embedding,
                    similarity_threshold=self.SIMILARITY_THRESHOLD,
                    min_face_size=self.MIN_FACE_SIZE
                )
                
                label = result.get("label", "ERROR")
                raw_confidence = result.get("confidence", 0.0)
                quality_warning = result.get("quality_warning", None)
                
                # Target lost handling (3-cycle retry)
                if label == "NO FACE":
                    self.target_lost_count += 1
                    if self.target_lost_count >= self.MAX_TARGET_LOST:
                        self.monitor_queue.put(("result", {
                            "label": "TARGET LOST",
                            "confidence": 0.0,
                            "smoothed_confidence": self.smoothed_confidence,
                            "explanation": (
                                "Target face not matched for "
                                f"{self.MAX_TARGET_LOST} consecutive cycles.\n"
                                "The person may have left or turned away."
                            ),
                            "stability": None,
                            "suspicion_alert": False,
                            "quality_warning": None
                        }))
                    else:
                        self.monitor_queue.put(("result", {
                            "label": "SEARCHING",
                            "confidence": 0.0,
                            "smoothed_confidence": self.smoothed_confidence,
                            "explanation": (
                                f"Searching for target face... "
                                f"(attempt {self.target_lost_count}/{self.MAX_TARGET_LOST})"
                            ),
                            "stability": None,
                            "suspicion_alert": False,
                            "quality_warning": None
                        }))
                    continue
                
                # Target found â€” reset lost counter
                self.target_lost_count = 0
                
                # EMA smoothing: 0.6 * new + 0.4 * old
                if self.smoothed_confidence == 0.0:
                    self.smoothed_confidence = raw_confidence
                else:
                    self.smoothed_confidence = (
                        0.6 * raw_confidence + 0.4 * self.smoothed_confidence
                    )
                
                # Temporal stability tracking (max 20 values)
                self.confidence_history.append(raw_confidence)
                if len(self.confidence_history) > 20:
                    self.confidence_history.pop(0)
                
                stability = None
                if len(self.confidence_history) >= 3:
                    mean_conf = statistics.mean(self.confidence_history)
                    std_conf = statistics.stdev(self.confidence_history)
                    stability = {
                        "mean": mean_conf,
                        "std": std_conf,
                        "stable": std_conf < 0.15
                    }
                
                # Suspicion alert
                suspicion_alert = False
                if label == "FAKE" and raw_confidence > self.SUSPICION_THRESHOLD:
                    self.consecutive_high_confidence += 1
                else:
                    self.consecutive_high_confidence = 0
                
                if self.consecutive_high_confidence >= self.SUSPICION_COUNT:
                    suspicion_alert = True
                
                self.monitor_queue.put(("result", {
                    "label": label,
                    "confidence": raw_confidence,
                    "smoothed_confidence": self.smoothed_confidence,
                    "explanation": "",
                    "stability": stability,
                    "suspicion_alert": suspicion_alert,
                    "quality_warning": quality_warning
                }))
                
            except Exception as e:
                print(f"[ERROR] Monitor inference error: {e}")
    
    # --------------------------------------------------------------------------
    # GUI Handlers (main thread only)
    # --------------------------------------------------------------------------
    
    def _handle_monitor_frame(self, frame_bgr):
        """
        Display a captured frame in the Input Video Display panel.
        Converts BGR numpy array to PhotoImage, resized to fit.
        """
        try:
            from PIL import Image as PILImage, ImageTk
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(frame_rgb)
            
            label_w = self.input_label.winfo_width()
            label_h = self.input_label.winfo_height()
            if label_w > 1 and label_h > 1:
                img_w, img_h = pil_img.size
                scale = min(label_w / img_w, label_h / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_img)
            self.input_label.config(image=photo)
            self.input_label._photo = photo
        except Exception as e:
            print(f"[ERROR] Frame display error: {e}")
    
    def _handle_monitor_result(self, result: Dict):
        """
        Handle Interview Monitor result on main thread.
        Updates confidence, result label, explanation, and alerts.
        """
        label = result.get("label", "ERROR")
        smoothed = result.get("smoothed_confidence", 0.0)
        raw_conf = result.get("confidence", 0.0)
        stability = result.get("stability")
        suspicion_alert = result.get("suspicion_alert", False)
        explanation = result.get("explanation", "")
        quality_warning = result.get("quality_warning", None)
        
        self.confidence.delete(0, tk.END)
        self.confidence.insert(0, f"{smoothed:.2%}")
        
        if label == "ERROR":
            self.result_label.config(text="ERROR", fg="#666666")
            self.monitor_indicator.config(
                text="\u25cf Monitoring Stopped", fg="#cc0000"
            )
        elif label in ("NO FACE", "SEARCHING"):
            self.result_label.config(text="SEARCHING", fg="#ff9900")
        elif label == "TARGET LOST":
            self.result_label.config(text="TARGET LOST", fg="#cc6600")
        elif label == "FAKE":
            self.result_label.config(text="\u26a0\ufe0f FAKE", fg="#cc0000")
        elif label == "REAL":
            self.result_label.config(text="\u2713 REAL", fg="#00aa00")
        
        lines = []
        lines.append("Interview Monitor Active\n")
        
        if label == "ERROR":
            if explanation:
                lines.append(explanation)
        elif label in ("SEARCHING", "TARGET LOST"):
            lines.append(f"\u26a0 {label}")
            if explanation:
                lines.append(explanation)
        else:
            lines.append(f"Latest: {label} (Raw: {raw_conf:.2%})")
            lines.append(f"Smoothed Confidence: {smoothed:.2%}\n")
        
        if quality_warning:
            lines.append(f"\n\u26a0 {quality_warning}")
        
        if stability:
            status = "Stable" if stability["stable"] else "Unstable"
            lines.append(
                f"Temporal Stability: {status} "
                f"(mean={stability['mean']:.2%}, std={stability['std']:.3f})"
            )
            if not stability["stable"]:
                lines.append(
                    "\u26a0 High variance \u2014 detection may be unreliable.\n"
                )
        
        if suspicion_alert:
            lines.append(
                "\n\u26a0\ufe0f SUSPICION ALERT: High fake confidence "
                f"detected for {self.SUSPICION_COUNT}+ consecutive runs!"
            )
        
        if explanation and label not in ("ERROR", "SEARCHING", "TARGET LOST"):
            lines.append(f"\n{explanation}")
        
        self.explanation.delete("1.0", tk.END)
        self.explanation.insert(tk.END, "\n".join(lines))
    
    # --------------------------------------------------------------------------
    # Stop
    # --------------------------------------------------------------------------
    
    def stop_interview_monitor(self):
        """
        Stop Interview Monitor Mode cleanly.
        Joins threads, clears buffer and state, resets UI.
        Safe to call multiple times.
        """
        if not self.monitoring_active and self.monitor_capture_thread is None:
            return
        
        self.monitoring_active = False
        
        if self.monitor_capture_thread is not None:
            self.monitor_capture_thread.join(timeout=3.0)
            self.monitor_capture_thread = None
        if self.monitor_inference_thread is not None:
            self.monitor_inference_thread.join(timeout=5.0)
            self.monitor_inference_thread = None
        
        with self.monitor_lock:
            self.monitor_buffer = []
        
        self.target_embedding = None
        self.meet_window_info = None
        self.target_lost_count = 0
        self.smoothed_confidence = 0.0
        self.confidence_history = []
        self.consecutive_high_confidence = 0
        self.monitor_capture_failures = 0
        self.monitor_start_time = None
        
        self.monitor_indicator.config(
            text="\u25cf Monitoring Stopped", fg="#cc0000"
        )
        self.monitor_timer_label.config(text="Duration: --:--:--")
    # ==========================================================================
    # STOP
    # ==========================================================================
    
    def stop_stream(self):
        """Stop all video playback."""
        self.running = False
        self.heatmap_running = False
        
        # NEW: Stop interview monitor if active
        self.stop_interview_monitor()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.input_label.config(image="")
        self.output_label.config(image="")
        
        self.explanation.delete("1.0", tk.END)
        self.explanation.insert(tk.END, "Playback stopped.")
        self.confidence.delete(0, tk.END)
        self.confidence.insert(0, "--")
        self.result_label.config(text="--", fg="#333")
        self.status_label.config(text="Ready")
        self.progress_bar["value"] = 0


# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================

def main():
    """Entry point for the ``sudarshn`` / ``sudarshn-gui`` console scripts."""
    root = tk.Tk()
    app = SudarshnApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
