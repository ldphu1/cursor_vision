import os
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
GAZE_MODEL_PATH = os.path.join(BASE_DIR, "gaze_ridge_model.pkl")

# LANDMARK INDICES (MediaPipe Face Mesh)
RIGHT_EYE_OUTER_IDX = 33
RIGHT_EYE_INNER_IDX = 133
RIGHT_IRIS_CENTER_IDX = 468
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]  #mống mắt phải
RIGHT_EYE_TOP_IDX = 159
RIGHT_EYE_BOTTOM_IDX = 145

LEFT_EYE_INNER_IDX = 362
LEFT_EYE_OUTER_IDX = 263
LEFT_IRIS_CENTER_IDX = 473
LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]   #mống mắt trái
LEFT_EYE_TOP_IDX = 386
LEFT_EYE_BOTTOM_IDX = 374

NOSE_TIP_IDX = 1

LEFT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]

# THÔNG SỐ HIỆU CHUẨN & LÀM MƯỢT
CALIB_MARGIN_X = 0.06
CALIB_MARGIN_Y = 0.06
CALIB_SETTLE_TIME = 0.85
CALIB_COLLECT_TIME = 1.15
CALIB_TIME_PER_POINT = CALIB_SETTLE_TIME + CALIB_COLLECT_TIME

BLINK_EAR_THRESHOLD = 0.16

# Bộ lọc 1€ Filter (One Euro Filter)
ONE_EURO_MIN_CUTOFF = 0.5  # Giữ con trỏ đứng im tuyệt đối khi nhìn cố định
ONE_EURO_BETA = 0.015      # Bám sát tức thì khi liếc mắt nhanh
ONE_EURO_D_CUTOFF = 1.0

DEFAULT_ALPHA_SMOOTH = 0.35

# Cấu hình bù trôi (Re-centering) & Chống rung vi nhãn cầu (Dead-zone)
DEAD_ZONE_RADIUS = 6.0          # Bán kính vi mô (pixel) để loại bỏ rung nhãn cầu khi nhìn cố định
RECENTER_SMOOTH_FACTOR = 0.5    # Hệ số làm mượt khi bù trôi
