import math
from typing import Tuple, Optional
import cv2
import numpy as np


class OneEuroFilter:
<<<<<<< HEAD
    """Bộ lọc 1€ Filter hoạt động trên không gian tọa độ Pixel màn hình."""

    def __init__(self, min_cutoff: float = 0.45, beta: float = 0.005, d_cutoff: float = 1.0):
=======
    """
    Bộ lọc 1€ Filter (One Euro Filter) - Casiez et al., CHI 2012.
    Tự động điều chỉnh tần số cắt dựa trên vận tốc:
    - Vận tốc thấp (mắt đứng yên / fixation): Giảm jitter, con trỏ cực kỳ ổn định.
    - Vận tốc cao (mắt liếc nhanh / saccade): Loại bỏ hoàn toàn độ trễ (latency).
    """

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.02, d_cutoff: float = 1.0):
>>>>>>> parent of b942f83 (update...)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t)
            return float(x)

        t_e = max(t - self.t_prev, 1e-5)
        self.t_prev = float(t)

        # Ước lượng đạo hàm dx/dt với lọc thông thấp d_cutoff
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Tần số cắt động tỷ lệ thuận với tốc độ di chuyển
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return float(x_hat)

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


def compute_eye_metrics(
    p_left_bound,
    p_right_bound,
    p_top,
    p_bot,
    p_iris,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, float, float]:
    """
<<<<<<< HEAD
    Chuẩn hóa tọa độ mống mắt dựa trên đường cơ sở nối 2 khóe mắt (chiều dài L):
    - Tránh hoàn toàn phép chia cho H (độ mở mắt) gây vọt giá trị khi tập trung nhìn.
=======
    Chuẩn hóa tọa độ mắt theo hướng thống nhất (Màn hình Phải & Màn hình Dưới):
    - p_left_bound: điểm mốc phía bên trái trên ảnh (với mắt trái là khóe ngoài 263, mắt phải là khóe trong 133).
    - p_right_bound: điểm mốc phía bên phải trên ảnh (với mắt trái là khóe trong 362, mắt phải là khóe ngoài 33).
    - p_top: mí mắt trên.
    - p_bot: mí mắt dưới.
    - p_iris: tâm mống mắt.

    Trả về:
        (rx, ry, ear, ex, ey, L, H)
        - rx: tỷ lệ ngang [0: trái, 0.5: giữa, 1: phải]
        - ry: tỷ lệ dọc [-: nhìn lên, 0: giữa, +: nhìn xuống]
        - ear: Eye Aspect Ratio (độ mở của mắt)
>>>>>>> parent of b942f83 (update...)
    """
    p_l = np.array(p_left_bound, dtype=np.float64)
    p_r = np.array(p_right_bound, dtype=np.float64)
    p_t = np.array(p_top, dtype=np.float64)
    p_b = np.array(p_bot, dtype=np.float64)
    p_i = np.array(p_iris, dtype=np.float64)

<<<<<<< HEAD
=======
    # Trục ngang u hướng từ trái sang phải trên ảnh (Screen-Right)
>>>>>>> parent of b942f83 (update...)
    u = p_r - p_l
    L = float(np.linalg.norm(u))
    if L < 1e-6:
        return 0.5, 0.0, 0.25, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, 1.0

    ex = u / L
<<<<<<< HEAD
=======
    # Trục dọc ey vuông góc hướng xuống dưới trên ảnh (Screen-Down)
>>>>>>> parent of b942f83 (update...)
    ey = np.array([-ex[1], ex[0]], dtype=np.float64)
    if ey[1] < 0:
        ey = -ey

<<<<<<< HEAD
    v = p_i - p_l
    rx = float(np.dot(v, ex) / L)

    # Trung điểm 2 khóe mắt làm gốc chiếu dọc ổn định
    p_corner_mid = 0.5 * (p_l + p_r)
    p_lid_mid = 0.5 * (p_t + p_b)

    ry_iris_corner = float(np.dot(p_i - p_corner_mid, ey) / L)
    ry_lid_corner = float(np.dot(p_lid_mid - p_corner_mid, ey) / L)

    # Công thức kết hợp đã kiểm chứng độ nhạy tốt ở Bản 1
    ry = float(0.72 * ry_iris_corner + 0.28 * ry_lid_corner)

=======
    # Vector từ mốc trái đến tâm mống mắt
    v = p_i - p_l
    rx = float(np.dot(v, ex) / L)

    # Độ mở mắt H và tỷ lệ dọc theo mí mắt
>>>>>>> parent of b942f83 (update...)
    H = float(np.linalg.norm(p_b - p_t))
    ymid = (p_t[1] + p_b[1]) / 2.0
    ry_lid = float((p_i[1] - ymid) / max(H, 1.0))
    ry_proj = float(np.dot(v, ey) / L)

    # Kết hợp độ nhạy dọc từ cả mí mắt và hình chiếu
    ry = float(0.75 * ry_lid + 0.25 * (ry_proj * 2.5))

    # Tỷ lệ co/mở mắt EAR
    ear = float(H / max(L, 1e-4))

    return rx, ry, ear, ex, ey, L, H


def compute_normalized_eye_coords(p_outer, p_inner, p_iris):
    """Hàm tương thích ngược với phiên bản cũ."""
    p_outer = np.array(p_outer, dtype=np.float64)
    p_inner = np.array(p_inner, dtype=np.float64)
    p_iris = np.array(p_iris, dtype=np.float64)

    u = p_inner - p_outer
    L = float(np.linalg.norm(u))
    if L < 1e-6:
        return 0.5, 0.0, u, np.array([1.0, 0.0]), np.array([0.0, 1.0]), float(L)

    e_x = u / L
    e_y = np.array([-e_x[1], e_x[0]], dtype=np.float64)

    v = p_iris - p_outer
    r_x = float(np.dot(v, e_x) / L)
    r_y = float(np.dot(v, e_y) / L)

    return r_x, r_y, u, e_x, e_y, float(L)


def get_head_pose_from_matrix(transform_matrix: np.ndarray):
<<<<<<< HEAD
    """Trích xuất góc Pitch, Yaw, Roll liên tục bằng atan2."""
    rmat = transform_matrix[:3, :3]
=======
    """
    Trích xuất góc Pitch, Yaw, Roll từ ma trận xoay 4x4 của MediaPipe Face Landmarker.
>>>>>>> parent of b942f83 (update...)

    Trả về:
        (pitch, yaw, roll, rotation_matrix, direction_text)
    """
    rotation_matrix = transform_matrix[:3, :3]
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])

    direction_h = "Right" if yaw > 10.0 else ("Left" if yaw < -10.0 else "Center")
    direction_v = "Up" if pitch > 10.0 else ("Down" if pitch < -10.0 else "Level")

    if direction_h == "Center" and direction_v == "Level":
        direction_text = "Looking Forward"
    elif direction_v == "Level":
        direction_text = f"Looking {direction_h}"
    elif direction_h == "Center":
        direction_text = f"Looking {direction_v}"
    else:
        direction_text = f"Looking {direction_v}-{direction_h}"

    return pitch, yaw, roll, rotation_matrix, direction_text
