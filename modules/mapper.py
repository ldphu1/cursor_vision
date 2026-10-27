import math
import os
import time
from typing import Optional, Tuple, List, Dict, Any
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.linear_model import RidgeCV

from .config import (
    GAZE_MODEL_PATH,
    CALIB_MARGIN_X,
    CALIB_MARGIN_Y,
    CALIB_SETTLE_TIME,
    CALIB_COLLECT_TIME,
    CALIB_TIME_PER_POINT,
    ONE_EURO_MIN_CUTOFF,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    DEAD_ZONE_RADIUS,
    RECENTER_SMOOTH_FACTOR,
)
from .geometry import OneEuroFilter

FEATURE_DIM = 12
EXPANDED_FEATURE_DIM = 19
MODEL_VERSION = "dual_poly_ridge_v3.0"


def expand_features(X: np.ndarray) -> np.ndarray:
    """
    Biến đổi mở rộng đặc trưng có định hướng vật lý (Domain-specific Physical Interaction):
    Thay thế PolynomialFeatures bậc 2 thuần túy (90 chiều) bằng 19 chiều tinh gọn:
    - 6 đặc trưng mắt cốt lõi: [rx_l, ry_l, rx_r, ry_r, rx_avg, ry_avg]
    - 3 đặc trưng phi tuyến nhãn cầu: [rx_avg^2, ry_avg^2, rx_avg * ry_avg]
    - 5 đặc trưng tư thế đầu: [head_x, head_y, pitch_norm, yaw_norm, roll_norm]
    - 2 đặc trưng phi tuyến góc quay đầu: [pitch_norm^2, yaw_norm^2]
    - 3 tương tác chéo mắt - đầu:
        + rx_avg * yaw_norm (mắt liếc ngang bù trừ xoay đầu ngang)
        + ry_avg * pitch_norm (mắt liếc dọc bù trừ gật đầu)
        + rx_avg * pitch_norm (hiệu ứng phối cảnh khi đầu gật)
    Tổng cộng: 6 + 3 + 5 + 2 + 3 = 19 chiều.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    is_1d = (X_arr.ndim == 1)
    if is_1d:
        X_arr = np.expand_dims(X_arr, axis=0)

    rx_l = X_arr[:, 0]
    ry_l = X_arr[:, 1]
    rx_r = X_arr[:, 2]
    ry_r = X_arr[:, 3]
    rx_avg = X_arr[:, 4]
    ry_avg = X_arr[:, 5]
    ear_avg = X_arr[:, 6]
    head_x = X_arr[:, 7]
    head_y = X_arr[:, 8]
    pitch_n = X_arr[:, 9]
    yaw_n = X_arr[:, 10]
    roll_n = X_arr[:, 11]

    f_eye = [rx_l, ry_l, rx_r, ry_r, rx_avg, ry_avg]
    f_eye_poly = [rx_avg**2, ry_avg**2, rx_avg * ry_avg]
    f_head = [head_x, head_y, pitch_n, yaw_n, roll_n]
    f_head_poly = [pitch_n**2, yaw_n**2]
    f_cross = [rx_avg * yaw_n, ry_avg * pitch_n, rx_avg * pitch_n]

    out = np.column_stack(f_eye + f_eye_poly + f_head + f_head_poly + f_cross)
    return out[0] if is_1d else out


def _build_axis_pipeline(alphas: np.ndarray) -> Pipeline:
    """
    Pipeline cho từng trục (X hoặc Y):
    - FunctionTransformer: Mở rộng đặc trưng 19 chiều có ý nghĩa vật lý.
    - RobustScaler: Chống ngoại lai, chuẩn hóa dữ liệu bền vững.
    - RidgeCV: Chọn hệ số điều chuẩn alpha tối ưu độc lập cho trục tương ứng.
    """
    return Pipeline([
        ("expand", FunctionTransformer(expand_features, validate=False)),
        ("scaler", RobustScaler()),
        ("ridge", RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")),
    ])


class GazeScreenMapper:
    """
    Quản lý quy trình hiệu chuẩn 9 điểm và ánh xạ ánh mắt lên màn hình máy tính
    sử dụng kiến trúc Dual-Axis RidgeCV với đặc trưng 19D và bộ bù trôi nhanh.
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        model_path: str = GAZE_MODEL_PATH,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.model_path = model_path
        self.model_x: Optional[Pipeline] = None
        self.model_y: Optional[Pipeline] = None

        self.filter_x = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.filter_y = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)

        # Bù trôi nhanh (Quick Re-centering) & Chống rung vi nhãn cầu
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.last_raw_x: Optional[float] = None
        self.last_raw_y: Optional[float] = None
        self.last_smooth_x: Optional[float] = None
        self.last_smooth_y: Optional[float] = None

        mx, my = CALIB_MARGIN_X, CALIB_MARGIN_Y
        self.norm_calib_points: List[Tuple[float, float]] = [\
            (0.5,       0.5      ),  # 1. Tâm màn hình
            (mx,        my       ),  # 2. Góc trên trái
            (0.5,       my       ),  # 3. Giữa trên
            (1.0 - mx,  my       ),  # 4. Góc trên phải
            (1.0 - mx,  0.5      ),  # 5. Giữa phải
            (1.0 - mx,  1.0 - my ),  # 6. Góc dưới phải
            (0.5,       1.0 - my ),  # 7. Giữa dưới
            (mx,        1.0 - my ),  # 8. Góc dưới trái
            (mx,        0.5      ),  # 9. Giữa trái
        ]
        self.calib_points: List[Tuple[int, int]] = [\
            (int(px * screen_w), int(py * screen_h))
            for px, py in self.norm_calib_points
        ]

        self.is_calibrating = False
        self.calib_step = 0
        self.calib_start_time = 0.0
        self.settle_time = CALIB_SETTLE_TIME
        self.collect_time = CALIB_COLLECT_TIME
        self.time_per_point = CALIB_TIME_PER_POINT

        self.current_point_samples: List[List[float]] = []
        self.X_data: List[List[float]] = []
        self.Y_data: List[List[float]] = []

        self.load_model()

    @property
    def model(self):
        """Thuộc tính tương thích ngược: trả về tuple (model_x, model_y) nếu đã có mô hình."""
        if self.model_x is not None and self.model_y is not None:
            return (self.model_x, self.model_y)
        return None

    def start_calibration(self):
        """Bắt đầu phiên hiệu chuẩn 9 điểm toàn màn hình."""
        self.is_calibrating = True
        self.calib_step = 0
        self.calib_start_time = time.time()
        self.current_point_samples = []
        self.X_data = []
        self.Y_data = []
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.last_smooth_x = None
        self.last_smooth_y = None
        self.filter_x.reset()
        self.filter_y.reset()
        print("\n[+] Bắt đầu hiệu chuẩn 9 điểm (Dual-Axis RidgeCV 19D)!")
        print("[+] Hãy giữ đầu tự nhiên và nhìn thẳng vào tâm chấm tròn...")

    def update_calibration(
        self,
        feature_vector: Optional[List[float]],
        is_blinking: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        if not self.is_calibrating:
            return None, False

        elapsed = time.time() - self.calib_start_time

        if self.calib_step >= len(self.calib_points):
            self.train_model()
            self.is_calibrating = False
            return None, True

        target_pt = self.calib_points[self.calib_step]
        target_norm = self.norm_calib_points[self.calib_step]

        if elapsed < self.settle_time:
            phase = "settle"
            progress = min(1.0, elapsed / self.settle_time)
        else:
            phase = "collect"
            progress = min(1.0, (elapsed - self.settle_time) / self.collect_time)
            if not is_blinking and feature_vector is not None:
                self.current_point_samples.append(list(feature_vector))

        if elapsed >= self.time_per_point:
            self._finalize_current_point(target_norm)
            self.calib_step += 1
            self.calib_start_time = time.time()
            self.current_point_samples = []

            if self.calib_step >= len(self.calib_points):
                self.train_model()
                self.is_calibrating = False
                return None, True

        return {
            "point": target_pt,
            "step": self.calib_step,
            "total_steps": len(self.calib_points),
            "phase": phase,
            "progress": progress,
            "sample_count": len(self.current_point_samples),
            "is_blinking": is_blinking,
        }, False

    def _finalize_current_point(self, target_norm: Tuple[float, float]):
        """
        Lọc mẫu ngoại lai bằng khoảng cách MAD trên 6 đặc trưng mắt chính.
        Giữ lại các mẫu sạch tự nhiên và thêm trọng số cho tâm cụm (centroid) làm điểm neo.
        """
        n_raw = len(self.current_point_samples)
        if n_raw < 3:
            print(f"[!] Điểm {self.calib_step + 1}: chỉ có {n_raw} mẫu — bỏ qua.")
            return

        samples = np.array(self.current_point_samples, dtype=np.float64)
        eye_feats = samples[:, :6]
        median_eye = np.median(eye_feats, axis=0)
        dists = np.linalg.norm(eye_feats - median_eye, axis=1)
        mad = max(np.median(dists), 1e-6)

        threshold = max(2.0 * mad, np.percentile(dists, 80))
        mask = dists <= threshold
        clean = samples[mask] if mask.sum() >= 3 else samples

        # Tính tâm cụm sạch (trimmed mean + median) làm điểm neo
        centroid = 0.5 * (np.median(clean, axis=0) + np.mean(clean, axis=0))

        tx, ty = target_norm
        # Thêm các mẫu sạch tự nhiên để giữ phương sai học máy
        for s in clean:
            self.X_data.append(s.tolist())
            self.Y_data.append([tx, ty])

        # Thêm trọng số cho tâm cụm (3 lần)
        for _ in range(3):
            self.X_data.append(centroid.tolist())
            self.Y_data.append([tx, ty])

        print(f"[+] Điểm {self.calib_step + 1}/{len(self.calib_points)}: "
              f"lọc {len(clean)}/{n_raw} mẫu sạch (rx={centroid[4]:.3f}, ry={centroid[5]:.3f}).")

    def train_model(self):
        """Huấn luyện mô hình Dual-Axis RidgeCV với alpha tối ưu riêng biệt cho X và Y."""
        if len(self.X_data) < 15:
            print("[!] Không đủ dữ liệu huấn luyện.")
            return

        X = np.array(self.X_data, dtype=np.float64)
        Y = np.array(self.Y_data, dtype=np.float64)

        print(f"\n[+] Huấn luyện Dual-Axis RidgeCV trên {len(X)} mẫu (Vector 19 chiều vật lý)...")
        t0 = time.time()

        alphas_x = np.logspace(-3, 3, 35)
        alphas_y = np.logspace(-2, 4, 35)

        self.model_x = _build_axis_pipeline(alphas_x)
        self.model_y = _build_axis_pipeline(alphas_y)

        self.model_x.fit(X, Y[:, 0])
        self.model_y.fit(X, Y[:, 1])
        dt = (time.time() - t0) * 1000

        pred_x = self.model_x.predict(X)
        pred_y = self.model_y.predict(X)

        rmse_x = float(np.sqrt(np.mean((pred_x * self.screen_w - Y[:, 0] * self.screen_w) ** 2)))
        rmse_y = float(np.sqrt(np.mean((pred_y * self.screen_h - Y[:, 1] * self.screen_h) ** 2)))
        rmse = float(np.sqrt(rmse_x**2 + rmse_y**2))

        best_alpha_x = self.model_x.named_steps["ridge"].alpha_
        best_alpha_y = self.model_y.named_steps["ridge"].alpha_
        print(f"[+] Huấn luyện hoàn tất trong {dt:.1f}ms | Alpha X: {best_alpha_x:.3f}, Alpha Y: {best_alpha_y:.3f} | RMSE: {rmse:.1f}px (X:{rmse_x:.1f}px, Y:{rmse_y:.1f}px)")

        self.offset_x = 0.0
        self.offset_y = 0.0

        try:
            payload = {
                "version": MODEL_VERSION,
                "model_x": self.model_x,
                "model_y": self.model_y,
            }
            joblib.dump(payload, self.model_path)
            print(f"[+] Đã lưu mô hình vào: {self.model_path}")
        except Exception as e:
            print(f"[!] Lưu model thất bại: {e}")

    def load_model(self):
        """Tải mô hình từ đĩa và kiểm tra phiên bản tương thích."""
        if not os.path.exists(self.model_path):
            return
        try:
            loaded = joblib.load(self.model_path)
            if isinstance(loaded, dict) and loaded.get("version") == MODEL_VERSION:
                self.model_x = loaded["model_x"]
                self.model_y = loaded["model_y"]
                print(f"[+] Đã tải mô hình {MODEL_VERSION} từ: {self.model_path}")
            else:
                old_ver = loaded.get("version") if isinstance(loaded, dict) else "unknown"
                print(f"[!] Phát hiện file model phiên bản cũ ({old_ver}).")
                print("[*] Hãy nhấn phím 'C' để hiệu chuẩn lại với thuật toán Dual-Axis v3.0.")
                self.model_x = None
                self.model_y = None
        except Exception as e:
            print(f"[!] Lỗi khi tải model: {e}")
            self.model_x = None
            self.model_y = None

    def recenter(
        self,
        feature_vector: Optional[List[float]] = None,
        smooth: bool = False,
    ) -> Tuple[float, float]:
        """
        Bù trôi nhanh từ feature_vector hoặc vị trí thô gần nhất.
        """
        if feature_vector is not None and self.model_x is not None and self.model_y is not None:
            feat_arr = np.array([feature_vector], dtype=np.float64)
            pred_norm_x = float(self.model_x.predict(feat_arr)[0])
            pred_norm_y = float(self.model_y.predict(feat_arr)[0])
            raw_x = pred_norm_x * self.screen_w
            raw_y = pred_norm_y * self.screen_h
            return self.apply_recenter(pred_raw_x=raw_x, pred_raw_y=raw_y, smooth=smooth)
        return self.apply_recenter(smooth=smooth)

    def apply_recenter(
        self,
        pred_raw_x: Optional[float] = None,
        pred_raw_y: Optional[float] = None,
        smooth: bool = False,
    ) -> Tuple[float, float]:
        """
        Bù lệch nhanh: Giả định mắt đang nhìn vào tâm màn hình (screen_w / 2, screen_h / 2).
        Tính delta offset = target - pred_raw.
        """
        target_x = self.screen_w / 2.0
        target_y = self.screen_h / 2.0

        curr_x = pred_raw_x if pred_raw_x is not None else self.last_raw_x
        curr_y = pred_raw_y if pred_raw_y is not None else self.last_raw_y

        if curr_x is None or curr_y is None:
            return self.offset_x, self.offset_y

        diff_x = target_x - curr_x
        diff_y = target_y - curr_y

        if smooth:
            self.offset_x = float(self.offset_x * (1.0 - RECENTER_SMOOTH_FACTOR) + diff_x * RECENTER_SMOOTH_FACTOR)
            self.offset_y = float(self.offset_y * (1.0 - RECENTER_SMOOTH_FACTOR) + diff_y * RECENTER_SMOOTH_FACTOR)
        else:
            self.offset_x = float(diff_x)
            self.offset_y = float(diff_y)

        print(f"[+] Re-center thành công! Offset mới: dx={self.offset_x:+.1f}px, dy={self.offset_y:+.1f}px")
        return self.offset_x, self.offset_y

    def reset_offset(self):
        """Đặt lại offset bù trôi về 0."""
        self.offset_x = 0.0
        self.offset_y = 0.0
        print("[*] Đã reset offset về (0, 0)")

    def predict(
        self,
        feature_vector: List[float],
        timestamp: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """Dự đoán tọa độ màn hình (Sx, Sy), bù trôi, lọc 1€ Filter và áp dụng dead-zone vi mô."""
        if self.model_x is None or self.model_y is None or feature_vector is None:
            return None

        t = timestamp if timestamp is not None else time.time()
        feat_arr = np.array([feature_vector], dtype=np.float64)
        pred_norm_x = float(self.model_x.predict(feat_arr)[0])
        pred_norm_y = float(self.model_y.predict(feat_arr)[0])

        raw_x = pred_norm_x * self.screen_w
        raw_y = pred_norm_y * self.screen_h
        self.last_raw_x = raw_x
        self.last_raw_y = raw_y

        # Áp dụng offset bù trôi
        adjusted_x = raw_x + self.offset_x
        adjusted_y = raw_y + self.offset_y

        norm_adj_x = adjusted_x / self.screen_w
        norm_adj_y = adjusted_y / self.screen_h

        smooth_norm_x = self.filter_x.filter(norm_adj_x, t)
        smooth_norm_y = self.filter_y.filter(norm_adj_y, t)

        target_sx = smooth_norm_x * self.screen_w
        target_sy = smooth_norm_y * self.screen_h

        # Vùng chết vi mô (Dead-zone) chống rung vi nhãn cầu khi nhìn cố định
        if self.last_smooth_x is not None and self.last_smooth_y is not None:
            dist_shift = math.hypot(target_sx - self.last_smooth_x, target_sy - self.last_smooth_y)
            if dist_shift < DEAD_ZONE_RADIUS:
                target_sx = self.last_smooth_x
                target_sy = self.last_smooth_y
            else:
                self.last_smooth_x = target_sx
                self.last_smooth_y = target_sy
        else:
            self.last_smooth_x = target_sx
            self.last_smooth_y = target_sy

        sx = int(np.clip(target_sx, 0, self.screen_w - 1))
        sy = int(np.clip(target_sy, 0, self.screen_h - 1))

        return sx, sy
