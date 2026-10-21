import math
from collections import deque
from typing import Tuple, Optional
import numpy as np


class OneEuroFilter:
    """Bộ lọc 1€ Filter (Casiez et al., CHI 2012)."""

    def __init__(self, min_cutoff: float = 0.35, beta: float = 0.018, d_cutoff: float = 1.0):
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


class FixationStabilizer:
    """
    Bộ phát hiện và ghim điểm nhìn (Fixation Dwell Lock / Magnet):
    - Khi mắt tập trung vào 1 điểm (độ phân tán trong cửa sổ ngắn < dispersion_px),
      con trỏ được hút về tâm hội tụ (median anchor) giúp đứng yên tại điểm đích.
    - Khi mắt liếc sang điểm khác (khoảng cách > release_px), khóa được mở ngay lập tức.
    """

    def __init__(
        self,
        window_size: int = 10,
        dispersion_px: float = 36.0,
        release_px: float = 52.0,
        magnet_strength: float = 0.82,
    ):
        self.window_size = window_size
        self.dispersion_px = dispersion_px
        self.release_px = release_px
        self.magnet_strength = magnet_strength
        self.history: deque = deque(maxlen=window_size)
        self.anchor: Optional[np.ndarray] = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        pt = np.array([x, y], dtype=np.float64)
        self.history.append(pt)

        if len(self.history) < 4:
            return float(x), float(y)

        pts = np.array(self.history, dtype=np.float64)
        med = np.median(pts, axis=0)
        dispersion = float(np.median(np.linalg.norm(pts - med, axis=1)))

        if self.anchor is not None:
            dist_from_anchor = float(np.linalg.norm(pt - self.anchor))
            if dist_from_anchor > self.release_px:
                # Mắt bắt đầu liếc sang vùng mới -> Mở khóa tức thì
                self.anchor = None
                self.history.clear()
                self.history.append(pt)
                return float(x), float(y)
            else:
                # Cập nhật nhẹ tâm neo nếu mắt dịch chuyển chậm có chủ đích
                drift_weight = 0.08 if dist_from_anchor < self.dispersion_px * 0.6 else 0.22
                self.anchor = (1.0 - drift_weight) * self.anchor + drift_weight * med
                locked = self.magnet_strength * self.anchor + (1.0 - self.magnet_strength) * pt
                return float(locked[0]), float(locked[1])
        else:
            if dispersion <= self.dispersion_px:
                # Bắt đầu trạng thái tập trung nhìn vào 1 điểm -> Đặt tâm neo
                self.anchor = med.copy()
                locked = self.magnet_strength * self.anchor + (1.0 - self.magnet_strength) * pt
                return float(locked[0]), float(locked[1])

        return float(x), float(y)

    def reset(self):
        self.history.clear()
        self.anchor = None


def compute_eye_metrics(
    p_left_bound,
    p_right_bound,
    p_top,
    p_bot,
    p_iris,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, float, float]:
    """
    Tính tọa độ mống mắt cục bộ hoàn toàn độc lập với chuyển động mí mắt (Zero Eyelid Bias):
    - p_top, p_bot CHỈ dùng để tính EAR phát hiện chớp mắt, tuyệt đối không cộng vào ry.
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

    # Tỷ lệ ngang và dọc thuần túy dựa trên trục 2 khóe mắt
    v = p_i - p_l
    rx = float(np.dot(v, ex) / L)

    p_corner_mid = 0.5 * (p_l + p_r)
    ry = float(np.dot(p_i - p_corner_mid, ey) / L)

    H = float(np.linalg.norm(p_b - p_t))
    ear = float(H / max(L, 1e-4))

    return rx, ry, ear, ex, ey, L, H


def compute_binocular_skull_metrics(
    p_l_out, p_l_in, p_l_iris,
    p_r_in, p_r_out, p_r_iris,
    p_nasion,
) -> Tuple[float, float, float, float]:
    """
    Tính tọa độ mống mắt trong Hệ quy chiếu Liên nhãn cầu (Binocular IOD Skull Frame):
    - Đường cơ sở nối tâm 2 mắt dài gấp ~4 lần chiều rộng 1 mắt, giúp góc nghiêng (roll)
      và khoảng cách chuẩn hóa (IOD) ổn định gấp 4 lần so với tính trên từng mắt đơn lẻ.
    - Bổ sung độ lệch dọc từ tâm mống mắt tới gốc xương sống mũi (p_nasion).
    """
    c_l = 0.5 * (np.array(p_l_out, dtype=np.float64) + np.array(p_l_in, dtype=np.float64))
    c_r = 0.5 * (np.array(p_r_in, dtype=np.float64) + np.array(p_r_out, dtype=np.float64))
    iris_l = np.array(p_l_iris, dtype=np.float64)
    iris_r = np.array(p_r_iris, dtype=np.float64)
    nasion = np.array(p_nasion, dtype=np.float64)

    # Đảm bảo vector u_iod luôn hướng từ trái sang phải trên ảnh (Screen-Right)
    if c_l[0] <= c_r[0]:
        u_iod = c_r - c_l
    else:
        u_iod = c_l - c_r

    iod = float(np.linalg.norm(u_iod))
    if iod < 1e-5:
        return 0.0, 0.0, 0.0, 1.0

    ex_bio = u_iod / iod
    ey_bio = np.array([-ex_bio[1], ex_bio[0]], dtype=np.float64)
    if ey_bio[1] < 0:
        ey_bio = -ey_bio

    # Dịch chuyển của từng mống mắt so với tâm khóe mắt tương ứng, chiếu lên trục liên nhãn cầu
    dl = iris_l - c_l
    dr = iris_r - c_r
    gx_avg = float(0.5 * (np.dot(dl, ex_bio) + np.dot(dr, ex_bio)) / iod)
    gy_avg = float(0.5 * (np.dot(dl, ey_bio) + np.dot(dr, ey_bio)) / iod)

    # Khoảng cách dọc từ trung điểm 2 mống mắt tới gốc xương sống mũi (Nasion)
    iris_mid = 0.5 * (iris_l + iris_r)
    gy_nasion = float(np.dot(iris_mid - nasion, ey_bio) / iod)

    return gx_avg, gy_avg, gy_nasion, iod


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