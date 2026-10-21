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

# ==========================================================
# LANDMARK INDICES (MediaPipe Face Mesh)
# ==========================================================
RIGHT_EYE_OUTER_IDX = 33
RIGHT_EYE_INNER_IDX = 133
RIGHT_IRIS_CENTER_IDX = 468
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_EYE_TOP_IDX = 159
RIGHT_EYE_BOTTOM_IDX = 145

LEFT_EYE_INNER_IDX = 362
LEFT_EYE_OUTER_IDX = 263
LEFT_IRIS_CENTER_IDX = 473
LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]
LEFT_EYE_TOP_IDX = 386
LEFT_EYE_BOTTOM_IDX = 374

# Cụm điểm neo xương khóe mắt và sống mũi (không bị ảnh hưởng khi nheo/mở mí mắt)
RIGHT_OUTER_ANCHORS = [33, 246, 7]
RIGHT_INNER_ANCHORS = [133, 173, 155]
LEFT_INNER_ANCHORS = [362, 398, 382]
LEFT_OUTER_ANCHORS = [263, 466, 249]
NASION_ANCHORS = [168, 6]  # Gốc xương sống mũi cố định giữa 2 mắt

NOSE_TIP_IDX = 1

LEFT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]

# ==========================================================
# THÔNG SỐ HIỆU CHUẨN & KHÓA ĐIỂM NHÌN (FIXATION LOCK)
# ==========================================================
CALIB_MARGIN_X = 0.06
CALIB_MARGIN_Y = 0.06
CALIB_SETTLE_TIME = 0.75   # Thời gian chờ mắt định vị vào điểm (giây)
CALIB_COLLECT_TIME = 0.95  # Thời gian thu mẫu khi mắt đã hội tụ (giây)
CALIB_TIME_PER_POINT = CALIB_SETTLE_TIME + CALIB_COLLECT_TIME

BLINK_EAR_THRESHOLD = 0.16

# Bộ lọc 1€ Filter
ONE_EURO_MIN_CUTOFF = 0.35
ONE_EURO_BETA = 0.018
ONE_EURO_D_CUTOFF = 1.0

# Thông số Fixation Stabilizer (Ghim con trỏ khi tập trung nhìn vào 1 điểm)
FIXATION_WINDOW_SIZE = 10      # Số frame cửa sổ trượt kiểm tra độ hội tụ (~0.3s)
FIXATION_DISPERSION_PX = 36.0  # Bán kính phân tán tối đa để kích hoạt khóa điểm (pixel)
FIXATION_RELEASE_PX = 52.0     # Ngưỡng dịch chuyển mắt để thoát khóa điểm tức thì (pixel)
FIXATION_MAGNET_STRENGTH = 0.82 # Lực hút ghim vào tâm điểm khi đang tập trung [0..1]

DEFAULT_ALPHA_SMOOTH = 0.35