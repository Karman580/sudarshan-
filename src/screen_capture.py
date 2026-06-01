# ==============================================================================
# screen_capture.py - Screen Capture for Interview Monitor Mode
# ==============================================================================
# Provides:
#   select_region()             - Interactive overlay for ROI selection (legacy)
#   capture_region()            - Fast screen grab using mss (legacy)
#   find_meet_window()          - Detect Google Meet window via pygetwindow
#   capture_window()            - TRUE window-content capture via PrintWindow
#   get_hwnd_from_title()       - Win32 HWND lookup by title substring
#   capture_window_by_hwnd()    - Win32 PrintWindow API capture
#
# Capture method: Win32 PrintWindow API (PW_RENDERFULLCONTENT)
# - Captures the window's own rendered content from its back buffer
# - Works even if the window is partially or fully overlapped by other windows
# - Does NOT capture screen pixels — captures the window's internal surface
#
# Dependencies: pygetwindow, pywin32 (win32gui, win32ui, win32con)
# Thread-safe: Yes (all Win32 resources created and destroyed per call)
# Windows-only: Yes
# ==============================================================================

import numpy as np
import tkinter as tk
import ctypes
from typing import Optional, Dict, List

try:
    import mss
except ImportError:
    mss = None  # Only needed for legacy capture_region

try:
    import pygetwindow as gw
except ImportError:
    gw = None

try:
    import win32gui
    import win32ui
    import win32con
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False


# PrintWindow flag: PW_RENDERFULLCONTENT = 0x00000002
# This tells PrintWindow to render the FULL window content including
# DirectComposition and DWM-composed content (Chrome, Edge, etc.)
PW_RENDERFULLCONTENT = 0x00000002


# ==============================================================================
# WIN32 HWND LOOKUP
# ==============================================================================

def get_hwnd_from_title(title_substring: str) -> Optional[int]:
    """
    Find a window handle (HWND) by searching for a title substring.

    Uses win32gui.EnumWindows to search all top-level windows.
    Returns the first visible, non-minimized window whose title
    contains title_substring (case-insensitive).

    Args:
        title_substring: Substring to search for in window titles.

    Returns:
        HWND (int) or None if not found.
    """
    if not _WIN32_AVAILABLE:
        print("[ERROR] pywin32 not installed. Run: pip install pywin32")
        return None

    result = [None]
    search_lower = title_substring.lower()

    def _enum_callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        if win32gui.IsIconic(hwnd):  # Minimized
            return True
        title = win32gui.GetWindowText(hwnd)
        if title and search_lower in title.lower():
            result[0] = hwnd
            return False  # Stop enumeration
        return True

    try:
        win32gui.EnumWindows(_enum_callback, None)
    except Exception:
        pass  # EnumWindows raises when callback returns False — expected

    return result[0]


# ==============================================================================
# WIN32 PRINTWINDOW CAPTURE
# ==============================================================================

def capture_window_by_hwnd(hwnd: int) -> Optional[np.ndarray]:
    """
    Capture a window's rendered content using the Win32 PrintWindow API.

    This captures the window's OWN content from its internal buffer,
    NOT screen pixels. The captured image is correct even if:
    - Another window overlaps it
    - The SUDARSHN GUI covers part of it
    - Other dialogs are on top

    Does NOT work if the window is minimized (returns None).
    Does NOT write to disk. All operations are in-memory.

    Args:
        hwnd: Win32 window handle (HWND).

    Returns:
        BGR numpy array (H, W, 3) or None on failure.
    """
    if not _WIN32_AVAILABLE:
        print("[ERROR] pywin32 not installed. Run: pip install pywin32")
        return None

    # ------------------------------------------------------------------
    # 1. Validate HWND
    # ------------------------------------------------------------------
    if not win32gui.IsWindow(hwnd):
        return None

    if win32gui.IsIconic(hwnd):
        # Window is minimized — cannot capture content
        return None

    # ------------------------------------------------------------------
    # 2. Get client rect dimensions
    #    GetClientRect gives the inner content area (excludes title bar,
    #    borders). This is what we want — just the page content.
    # ------------------------------------------------------------------
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width = right - left
        height = bottom - top
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None

    # ------------------------------------------------------------------
    # 3. Create Win32 device contexts and bitmap
    # ------------------------------------------------------------------
    hwnd_dc = None
    mem_dc = None
    bitmap = None

    try:
        # Get window's device context
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mem_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mem_dc.CreateCompatibleDC()

        # Create bitmap to receive the rendered content
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mem_dc, width, height)
        save_dc.SelectObject(bitmap)

        # ------------------------------------------------------------------
        # 4. PrintWindow — captures the window's own rendered content
        #
        #    PW_RENDERFULLCONTENT (0x2) is critical for modern Chrome/Edge
        #    windows that use DirectComposition for rendering.
        #
        #    We use ctypes to call PrintWindow directly because pywin32's
        #    wrapper doesn't support the PW_RENDERFULLCONTENT flag.
        # ------------------------------------------------------------------

        # Offset the source origin to skip the non-client area (title bar)
        # GetWindowRect gives full window; GetClientRect gives content area
        # The difference is the non-client border
        win_rect = win32gui.GetWindowRect(hwnd)
        client_origin = win32gui.ClientToScreen(hwnd, (0, 0))
        x_offset = client_origin[0] - win_rect[0]
        y_offset = client_origin[1] - win_rect[1]

        # Capture the FULL window (including non-client area)
        ctypes.windll.user32.PrintWindow(
            hwnd,
            save_dc.GetSafeHdc(),
            PW_RENDERFULLCONTENT
        )

        # BitBlt from the captured full window to get only the client area
        # We already have the full window in save_dc, so we extract the
        # client portion by reading the bitmap at the right offset
        #
        # Actually, since we need client-area content and PrintWindow
        # captures the full window, we need a second bitmap for extraction
        # OR we can just grab the full window and crop.

        # ------------------------------------------------------------------
        # 5. Extract pixel data from bitmap
        # ------------------------------------------------------------------
        bmp_info = bitmap.GetInfo()
        bmp_bits = bitmap.GetBitmapBits(True)

        # GetBitmapBits returns BGRA data (32-bit) or BGR (24-bit)
        # depending on the display mode. We handle both.
        bpp = bmp_info['bmBitsPixel']

        if bpp == 32:
            img = np.frombuffer(bmp_bits, dtype=np.uint8)
            expected_size = bmp_info['bmHeight'] * bmp_info['bmWidth'] * 4
            if img.size < expected_size:
                return None
            img = img[:expected_size].reshape(
                bmp_info['bmHeight'], bmp_info['bmWidth'], 4
            )
            # BGRA → BGR
            img = img[:, :, :3]
        elif bpp == 24:
            img = np.frombuffer(bmp_bits, dtype=np.uint8)
            # 24-bit rows are padded to 4-byte boundaries
            stride = ((bmp_info['bmWidth'] * 3 + 3) // 4) * 4
            expected_size = bmp_info['bmHeight'] * stride
            if img.size < expected_size:
                return None
            img = img[:expected_size].reshape(
                bmp_info['bmHeight'], stride
            )
            img = img[:, :bmp_info['bmWidth'] * 3].reshape(
                bmp_info['bmHeight'], bmp_info['bmWidth'], 3
            )
        else:
            # Unsupported bit depth
            return None

        # ------------------------------------------------------------------
        # 6. Crop to client area
        #    PrintWindow captures the full window including title bar.
        #    We crop using the offset between window rect and client origin.
        # ------------------------------------------------------------------
        if y_offset > 0 or x_offset > 0:
            max_y = min(y_offset + height, img.shape[0])
            max_x = min(x_offset + width, img.shape[1])
            if y_offset < img.shape[0] and x_offset < img.shape[1]:
                img = img[y_offset:max_y, x_offset:max_x]

        # Validate output
        if img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
            return None

        # Return a contiguous copy (required for downstream numpy/cv2)
        return np.ascontiguousarray(img)

    except Exception as e:
        print(f"[ERROR] PrintWindow capture failed: {e}")
        return None

    finally:
        # ------------------------------------------------------------------
        # 7. Clean up ALL Win32 resources (critical to prevent GDI leaks)
        # ------------------------------------------------------------------
        try:
            if bitmap is not None:
                win32gui.DeleteObject(bitmap.GetHandle())
        except Exception:
            pass
        try:
            if save_dc is not None:
                save_dc.DeleteDC()
        except Exception:
            pass
        try:
            if mem_dc is not None:
                mem_dc.DeleteDC()
        except Exception:
            pass
        try:
            if hwnd_dc is not None:
                win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass


# ==============================================================================
# WINDOW-LOCKED CAPTURE (Interview Monitor v2 API)
# ==============================================================================

def find_meet_window(custom_title: Optional[str] = None) -> Optional[Dict]:
    """
    Search for a Google Meet window by title.

    Returns window info dict including the Win32 HWND for PrintWindow capture.
    Falls back to pygetwindow if win32gui is unavailable.

    Args:
        custom_title: Optional custom window title substring to search for.

    Returns:
        dict with keys {"title", "left", "top", "width", "height", "hwnd"}
        or None if not found.
    """
    search_terms = [custom_title] if custom_title else [
        "Meet", "Google Meet", "meet.google.com"
    ]

    # ----- Primary: Win32 EnumWindows (gets HWND for PrintWindow) -----
    if _WIN32_AVAILABLE:
        for term in search_terms:
            results = []  # type: List[Dict]
            search_lower = term.lower()

            def _enum_callback(hwnd, _):
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                if win32gui.IsIconic(hwnd):
                    return True
                title = win32gui.GetWindowText(hwnd)
                if not title or search_lower not in title.lower():
                    return True
                rect = win32gui.GetWindowRect(hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w > 100 and h > 100:
                    results.append({
                        "title": title,
                        "left": rect[0],
                        "top": rect[1],
                        "width": w,
                        "height": h,
                        "hwnd": hwnd
                    })
                return True

            try:
                win32gui.EnumWindows(_enum_callback, None)
            except Exception:
                pass

            if results:
                return results[0]

        return None

    # ----- Fallback: pygetwindow (no HWND, capture will fail) -----
    if gw is not None:
        try:
            all_windows = gw.getAllWindows()
            for term in search_terms:
                for win in all_windows:
                    if (win.title
                            and term.lower() in win.title.lower()
                            and win.width > 100
                            and win.height > 100
                            and not win.isMinimized):
                        return {
                            "title": win.title,
                            "left": win.left,
                            "top": win.top,
                            "width": win.width,
                            "height": win.height,
                            "hwnd": None
                        }
        except Exception as e:
            print(f"[ERROR] Window search failed: {e}")

    return None


def capture_window(bounds: Dict) -> Optional[np.ndarray]:
    """
    Capture a window's content using Win32 PrintWindow API.

    Uses the HWND from find_meet_window() to perform TRUE window-content
    capture that is immune to GUI overlap contamination.

    Falls back to mss screen grab ONLY if HWND is unavailable (should
    not happen on Windows with pywin32 installed).

    Args:
        bounds: dict from find_meet_window() containing "hwnd" key.

    Returns:
        BGR numpy array (H, W, 3) or None on failure.
    """
    hwnd = bounds.get("hwnd")

    # ----- Primary: PrintWindow (TRUE window capture) -----
    if hwnd is not None and _WIN32_AVAILABLE:
        # Verify window still exists
        if not win32gui.IsWindow(hwnd):
            return None
        return capture_window_by_hwnd(hwnd)

    # ----- Last-resort fallback: mss screen grab -----
    if mss is not None:
        try:
            region = {
                "left": bounds["left"],
                "top": bounds["top"],
                "width": bounds["width"],
                "height": bounds["height"]
            }
            with mss.mss() as sct:
                screenshot = sct.grab(region)
                frame = np.array(screenshot, dtype=np.uint8)
                return frame[:, :, :3]
        except Exception as e:
            print(f"[ERROR] Fallback screen capture failed: {e}")
            return None

    print("[ERROR] No capture method available.")
    return None


# ==============================================================================
# REGION SELECTION (INTERACTIVE OVERLAY) — kept for backward compat
# ==============================================================================

def select_region(parent_window=None) -> Optional[Dict[str, int]]:
    """
    Opens a fullscreen semi-transparent overlay for the user to drag-select
    a screen region of interest.

    Args:
        parent_window: Optional tk.Tk root window to hide during selection.

    Returns:
        dict with keys {"top", "left", "width", "height"} or None if cancelled.
    """
    region_result = {"value": None}

    if parent_window:
        parent_window.withdraw()
        import time
        time.sleep(0.5)

    overlay = tk.Toplevel()
    overlay.attributes("-fullscreen", True)
    overlay.attributes("-topmost", True)
    overlay.attributes("-alpha", 0.3)
    overlay.configure(bg="black")
    overlay.title("Select Region - Drag to select, Escape to cancel")

    canvas = tk.Canvas(overlay, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    canvas.create_text(
        overlay.winfo_screenwidth() // 2, 50,
        text="Drag to select the interview region. Press ESC to cancel.",
        fill="white", font=("Arial", 16, "bold")
    )

    start_x = [0]
    start_y = [0]
    rect_id = [None]

    def _finish():
        overlay.destroy()
        if parent_window:
            parent_window.deiconify()

    def on_press(event):
        start_x[0] = event.x
        start_y[0] = event.y
        if rect_id[0]:
            canvas.delete(rect_id[0])
        rect_id[0] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="lime", width=2
        )

    def on_drag(event):
        if rect_id[0]:
            canvas.coords(rect_id[0], start_x[0], start_y[0], event.x, event.y)

    def on_release(event):
        x1 = min(start_x[0], event.x)
        y1 = min(start_y[0], event.y)
        x2 = max(start_x[0], event.x)
        y2 = max(start_y[0], event.y)
        w = x2 - x1
        h = y2 - y1
        if w > 20 and h > 20:
            region_result["value"] = {
                "top": y1, "left": x1, "width": w, "height": h
            }
        _finish()

    def on_escape(event):
        _finish()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    overlay.bind("<Escape>", on_escape)

    overlay.grab_set()
    overlay.wait_window()

    return region_result["value"]


# ==============================================================================
# SCREEN REGION CAPTURE — kept for backward compat
# ==============================================================================

def capture_region(region: Dict[str, int]) -> Optional[np.ndarray]:
    """
    Capture a screen region and return it as a BGR numpy array.

    Args:
        region: dict with keys {"top", "left", "width", "height"}

    Returns:
        BGR numpy array (H, W, 3) or None on failure
    """
    if mss is None:
        print("[ERROR] mss not installed for legacy capture_region.")
        return None
    try:
        with mss.mss() as sct:
            screenshot = sct.grab(region)
            frame = np.array(screenshot, dtype=np.uint8)
            return frame[:, :, :3]  # BGRA → BGR
    except Exception as e:
        print(f"[ERROR] Screen capture failed: {e}")
        return None
