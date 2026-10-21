import os
import time
from typing import Optional, Tuple, List, Dict, Any
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
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
)
from .geometry import OneEuroFilter

FEATURE_DIM = 12
MODEL_VERSION = "poly_ridge_v2.3"


def _build_pipeline() -> Pipeline:
    """
    Pipeline Hồi quy Đa thức bậc 2 + RidgeCV:
    - RobustScaler: chống lại các giá trị ngoại lai (outliers) tốt hơn StandardScaler.
    - PolynomialFeatures(degree=2): mô hình hóa chính xác hình học cầu của nhãn cầu
      và sự tương tác chéo giữa góc nhìn mắt (rx, ry) với vị trí/góc đầu (head pose).
    - RidgeCV: tự động chọn hệ số điều chuẩn alpha tối ưu bằng Leave-One-Out CV,
      đảm bảo nội suy mượt và ngoại suy chính xác ra sát 4 cạnh màn hình.
    """
    alphas = np.logspace(-2, 3, 30)
    return Pipeline([
        ("scaler", RobustScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")),
    ])


class GazeScreenMapper:
    """
    Quản lý quy trình hiệu chuẩn 9 điểm và ánh xạ ánh mắt lên màn hình máy tính.
    Vector đầu vào 12 chiều:
        [rx_l, ry_l, rx_r, ry_r, rx_avg, ry_avg, ear_avg, head_x, head_y, pitch_norm, yaw_norm, roll_norm]
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
        self.model: Optional[Pipeline] = None

        self.filter_x = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.filter_y = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)

        mx, my = CALIB_MARGIN_X, CALIB_MARGIN_Y
        self.norm_calib_points: List[Tuple[float, float]] = [
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
        self.calib_points: List[Tuple[int, int]] = [
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

    def start_calibration(self):
        """Bắt đầu phiên hiệu chuẩn 9 điểm toàn màn hình."""
        self.is_calibrating = True
        self.calib_step = 0
        self.calib_start_time = time.time()
        self.current_point_samples = []
        self.X_data = []
        self.Y_data = []
        self.filter_x.reset()
        self.filter_y.reset()
        print("\n[+] Bắt đầu hiệu chuẩn 9 điểm (Poly-RidgeCV 12D)!")
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
        Lọc mẫu ngoại lai tập trung vào các đặc trưng mắt cốt lõi (rx_l..ry_avg)
        và kết hợp tâm cụm (Median Centroid) để triệt tiêu nhiễu vi mô tại chỗ.
        """
        n_raw = len(self.current_point_samples)
        if n_raw < 3:
            print(f"[!] Điểm {self.calib_step + 1}: chỉ có {n_raw} mẫu — bỏ qua.")
            return

        samples = np.array(self.current_point_samples, dtype=np.float64)
        # Tính khoảng cách MAD trên 6 đặc trưng mắt chính [rx_l, ry_l, rx_r, ry_r, rx_avg, ry_avg]
        eye_feats = samples[:, :6]
        median_eye = np.median(eye_feats, axis=0)
        dists = np.linalg.norm(eye_feats - median_eye, axis=1)
        mad = max(np.median(dists), 1e-6)

        # Giữ lại 70% mẫu gần tâm nhất hoặc trong ngưỡng 1.8 * MAD
        threshold = max(1.8 * mad, np.percentile(dists, 70))
        mask = dists <= threshold
        clean = samples[mask] if mask.sum() >= 3 else samples

        # Tính tâm cụm sạch (trimmed mean + median) làm điểm neo chuẩn xác
        centroid = 0.5 * (np.median(clean, axis=0) + np.mean(clean, axis=0))

        tx, ty = target_norm
        # Đưa các mẫu đã làm mịn quanh tâm cụm vào tập huấn luyện (giảm 70% phương sai nhiễu nội cụm)
        for s in clean:
            stabilized_sample = 0.65 * centroid + 0.35 * s
            self.X_data.append(stabilized_sample.tolist())
            self.Y_data.append([tx, ty])

        # Thêm trọng số cho chính tâm cụm
        for _ in range(5):
            self.X_data.append(centroid.tolist())
            self.Y_data.append([tx, ty])

        print(f"[+] Điểm {self.calib_step + 1}/{len(self.calib_points)}: "
              f"lọc {len(clean)}/{n_raw} mẫu sạch (rx={centroid[4]:.3f}, ry={centroid[5]:.3f}).")

    def train_model(self):
        """Huấn luyện mô hình Đa thức bậc 2 + RidgeCV."""
        if len(self.X_data) < 15:
            print("[!] Không đủ dữ liệu huấn luyện.")
            return

        X = np.array(self.X_data, dtype=np.float64)
        Y = np.array(self.Y_data, dtype=np.float64)

        print(f"\n[+] Huấn luyện Poly-RidgeCV trên {len(X)} mẫu (Vector 12 chiều)...")
        t0 = time.time()
        self.model = _build_pipeline()
        self.model.fit(X, Y)
        dt = (time.time() - t0) * 1000

        pred = self.model.predict(X)
        rmse = float(np.sqrt(np.mean(
            (pred[:, 0] * self.screen_w - Y[:, 0] * self.screen_w) ** 2 +
            (pred[:, 1] * self.screen_h - Y[:, 1] * self.screen_h) ** 2
        )))
        best_alpha = self.model.named_steps["ridge"].alpha_
        print(f"[+] Huấn luyện hoàn tất trong {dt:.1f}ms | Alpha tối ưu: {best_alpha:.3f} | Sai số RMSE: {rmse:.1f}px")

        try:
            payload = {"version": MODEL_VERSION, "model": self.model}
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
                self.model = loaded["model"]
                print(f"[+] Đã tải mô hình {MODEL_VERSION} từ: {self.model_path}")
            else:
                print("[!] Phát hiện file model phiên bản cũ (SVR/khác hệ đặc trưng).")
                print("[*] Hãy nhấn phím 'C' để hiệu chuẩn lại với thuật toán mới.")
                self.model = None
        except Exception as e:
            print(f"[!] Lỗi khi tải model: {e}")
            self.model = None

    def predict(
        self,
        feature_vector: List[float],
        timestamp: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """Dự đoán tọa độ màn hình (Sx, Sy) và lọc mượt bằng One Euro Filter."""
        if self.model is None or feature_vector is None:
            return None

        t = timestamp if timestamp is not None else time.time()
        pred_norm = self.model.predict(np.array([feature_vector], dtype=np.float64))[0]

        # Áp dụng lọc 1€ trước khi kẹp biên để tránh hiệu ứng dính viền
        smooth_norm_x = self.filter_x.filter(float(pred_norm[0]), t)
        smooth_norm_y = self.filter_y.filter(float(pred_norm[1]), t)

        sx = int(np.clip(smooth_norm_x * self.screen_w, 0, self.screen_w - 1))
        sy = int(np.clip(smooth_norm_y * self.screen_h, 0, self.screen_h - 1))

        return sx, sy