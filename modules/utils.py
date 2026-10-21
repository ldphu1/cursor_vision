import ctypes
import os
import sys
import urllib.request
from typing import Tuple
from .config import MODEL_PATH, MODEL_URL

# Bật DPI Awareness trên Windows để tránh lệch tọa độ khi màn hình để Scale 125% / 150%
if sys.platform.startswith("win"):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def get_screen_resolution() -> Tuple[int, int]:
    """Lấy độ phân giải thực tế của màn hình (Windows) hoặc mặc định 1920x1080."""
    try:
        user32 = ctypes.windll.user32
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080


def ensure_model_exists(model_path: str = MODEL_PATH, model_url: str = MODEL_URL) -> str:
    """Tải file mô hình MediaPipe Face Landmarker nếu chưa có."""
    if not os.path.exists(model_path):
        print(f"[*] Downloading MediaPipe model to: {model_path} ...")
        urllib.request.urlretrieve(model_url, model_path)
        print("[+] Model downloaded successfully!")
    return model_path


def set_mouse_position(x: int, y: int) -> bool:
    """Điều khiển con trỏ chuột hệ điều hành đến tọa độ (x, y)."""
    try:
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
        return True
    except Exception:
        return False