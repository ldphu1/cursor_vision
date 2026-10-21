import ctypes
import os
import urllib.request
from typing import Tuple
from .config import MODEL_PATH, MODEL_URL


def get_screen_resolution() -> Tuple[int, int]:
    """Lấy độ phân giải màn hình hiện tại (Windows) hoặc mặc định 1920x1080."""
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
