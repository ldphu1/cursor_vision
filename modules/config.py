import os
import sys

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ==========================================================
# ĐƯỜNG DẪN & CẤU HÌNH TÀI NGUYÊN
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
GAZE_MODEL_PATH = os.path.join(BASE_DIR, "gaze_ridge_model.pkl")

# ==========================================================
# LANDMARK INDICES (Theo chuẩn MediaPipe Face Mesh)
# ==========================================================
# - Mắt phải (giải phẫu người): Khóe ngoài = 33, Khóe trong = 133, Tâm mống mắt = 468
# - Mắt trái (giải phẫu người): Khóe trong = 362, Khóe ngoài = 263, Tâm mống mắt = 473
RIGHT_EYE_OUTER_IDX = 33
RIGHT_EYE_INNER_IDX = 133
RIGHT_IRIS_CENTER_IDX = 468
<<<<<<< HEAD
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]
=======
>>>>>>> parent of b942f83 (update...)
RIGHT_EYE_TOP_IDX = 159
RIGHT_EYE_BOTTOM_IDX = 145

LEFT_EYE_INNER_IDX = 362
LEFT_EYE_OUTER_IDX = 263
LEFT_IRIS_CENTER_IDX = 473
<<<<<<< HEAD
LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]
=======
>>>>>>> parent of b942f83 (update...)
LEFT_EYE_TOP_IDX = 386
LEFT_EYE_BOTTOM_IDX = 374

NOSE_TIP_IDX = 1

# Viền hốc mắt
LEFT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]

<<<<<<< HEAD
# THÔNG SỐ HIỆU CHUẨN & LÀM MƯỢT (Tính trên đơn vị Pixel)
CALIB_MARGIN_X = 0.07
CALIB_MARGIN_Y = 0.07
CALIB_SETTLE_TIME = 0.80
CALIB_COLLECT_TIME = 1.05
=======
# ==========================================================
# THÔNG SỐ HIỆU CHUẨN & LÀM MƯỢT
# ==========================================================
CALIB_MARGIN_X = 0.10
CALIB_MARGIN_Y = 0.10
CALIB_SETTLE_TIME = 0.8    # Thời gian chờ mắt định vị và ổn định (giây)
CALIB_COLLECT_TIME = 1.1   # Thời gian thu thập mẫu khi mắt đã nhìn cố định (giây)
>>>>>>> parent of b942f83 (update...)
CALIB_TIME_PER_POINT = CALIB_SETTLE_TIME + CALIB_COLLECT_TIME

# Ngưỡng phát hiện chớp mắt (Eye Aspect Ratio - EAR)
BLINK_EAR_THRESHOLD = 0.16

<<<<<<< HEAD
# Thông số OneEuroFilter chuẩn hóa cho không gian PIXEL (0..1920)
ONE_EURO_MIN_CUTOFF = 0.45  # Chống rung khi mắt tập trung nhìn cố định vào 1 điểm
ONE_EURO_BETA = 0.005       # Tăng tốc mượt mà theo vận tốc pixel/giây để bắt trúng đích ngay
=======
# Bộ lọc 1€ Filter (One Euro Filter) chống rung lắc con trỏ
ONE_EURO_MIN_CUTOFF = 0.8  # Giảm rung tối đa khi mắt dừng nhìn (fixation)
ONE_EURO_BETA = 0.02       # Phản hồi tức thì không trễ khi mắt di chuyển nhanh (saccade)
>>>>>>> parent of b942f83 (update...)
ONE_EURO_D_CUTOFF = 1.0

# Hệ số EMA phụ trợ (dùng khi fallback)
DEFAULT_ALPHA_SMOOTH = 0.35
