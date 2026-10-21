import math
from typing import Tuple, Optional
import numpy as np


class OneEuroFilter:
    """Bộ lọc 1€ Filter hoạt động trên không gian tọa độ Pixel màn hình."""

    def __init__(self, min_cutoff: float = 0.45, beta: float = 0.005, d_cutoff: float = 1.0):
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

        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

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
    Chuẩn hóa tọa độ mống mắt dựa trên đường cơ sở nối 2 khóe mắt (chiều dài L):
    - Tránh hoàn toàn phép chia cho H (độ mở mắt) gây vọt giá trị khi tập trung nhìn.
    """
    p_l = np.array(p_left_bound, dtype=np.float64)
    p_r = np.array(p_right_bound, dtype=np.float64)
    p_t = np.array(p_top, dtype=np.float64)
    p_b = np.array(p_bot, dtype=np.float64)
    p_i = np.array(p_iris, dtype=np.float64)

    u = p_r - p_l
    L = float(np.linalg.norm(u))
    if L < 1e-6:
        return 0.5, 0.0, 0.25, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, 1.0

    ex = u / L
    ey = np.array([-ex[1], ex[0]], dtype=np.float64)
    if ey[1] < 0:
        ey = -ey

    v = p_i - p_l
    rx = float(np.dot(v, ex) / L)

    # Trung điểm 2 khóe mắt làm gốc chiếu dọc ổn định
    p_corner_mid = 0.5 * (p_l + p_r)
    p_lid_mid = 0.5 * (p_t + p_b)

    ry_iris_corner = float(np.dot(p_i - p_corner_mid, ey) / L)
    ry_lid_corner = float(np.dot(p_lid_mid - p_corner_mid, ey) / L)

    # Công thức kết hợp đã kiểm chứng độ nhạy tốt ở Bản 1
    ry = float(0.72 * ry_iris_corner + 0.28 * ry_lid_corner)

    H = float(np.linalg.norm(p_b - p_t))
    ear = float(H / max(L, 1e-4))

    return rx, ry, ear, ex, ey, L, H


def compute_normalized_eye_coords(p_outer, p_inner, p_iris):
    """Hàm tương thích ngược."""
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
    """Trích xuất góc Pitch, Yaw, Roll liên tục bằng atan2."""
    rmat = transform_matrix[:3, :3]

    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[2, 2]))
        yaw = math.degrees(math.atan2(rmat[0, 2], sy))
        roll = math.degrees(math.atan2(-rmat[0, 1], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[1, 1]))
        yaw = math.degrees(math.atan2(rmat[0, 2], sy))
        roll = 0.0

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

    return float(pitch), float(yaw), float(roll), rmat, direction_text