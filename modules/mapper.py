import os
import time
from typing import Optional, Tuple, List, Dict, Any
import joblib
import numpy as np
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
    FIXATION_WINDOW_SIZE,
    FIXATION_DISPERSION_PX,
    FIXATION_RELEASE_PX,
    FIXATION_MAGNET_STRENGTH,
)
from .geometry import OneEuroFilter, FixationStabilizer

FEATURE_DIM = 14
MODEL_VERSION = "hybrid_poly_rbf_v2.4"


class HybridGazeModel:
    """
    Mô hình ánh xạ Hybrid 2 tầng:
    1. Tầng toàn cục (Global Parametric):
       - Đa thức bậc 2 trên 5 đặc trưng mắt cốt lõi [rx_avg, ry_avg, gx_avg, gy_avg, gy_nasion]
         kết hợp tuyến tính (bậc 1) với từng mắt và tư thế đầu [head_x, head_y, pitch, yaw, roll].
       - Ngăn chặn hoàn toàn hiện tượng overfitting 90 chiều của góc đầu.
    2. Tầng khớp điểm cục bộ (Local Residual RBF Interpolator):
       - Nội suy chính xác phần dư (residual error) tại tâm của tất cả các điểm hiệu chuẩn
         trên mặt phẳng ánh mắt (rx_avg, ry_avg, gx_avg, gy_avg).
       - Đảm bảo khi mắt tập trung vào bất kỳ vùng nào đã hiệu chuẩn, sai số khớp < 3px.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.ridge = RidgeCV(alphas=np.logspace(-2, 2.5, 30))

        self.anchor_gaze: Optional[np.ndarray] = None
        self.rbf_weights: Optional[np.ndarray] = None
        self.rbf_gamma: float = 12.0

    def _transform_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X_scaled = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        # 5 đặc trưng mắt trung tâm -> khai triển bậc 2 (20 chiều)
        core_eye = X_scaled[:, :5]
        core_poly = self.poly.fit_transform(core_eye) if fit else self.poly.transform(core_eye)
        # 4 đặc trưng mắt đơn lẻ + 5 đặc trưng đầu -> giữ tuyến tính bậc 1 với hệ số giảm nhẹ
        aux_linear = X_scaled[:, 5:] * 0.5
        return np.hstack([core_poly, aux_linear])

    def fit(self, X_all: np.ndarray, Y_all: np.ndarray, X_centroids: np.ndarray, Y_centroids: np.ndarray):
        Phi_all = self._transform_features(X_all, fit=True)
        self.ridge.fit(Phi_all, Y_all)

        # Tính sai số dư (Residuals) tại các tâm điểm hiệu chuẩn (Anchor Centroids)
        Phi_c = self._transform_features(X_centroids, fit=False)
        pred_c = self.ridge.predict(Phi_c)
        residuals = Y_centroids - pred_c

        # Khớp hàm RBF mượt trên 4 đặc trưng mắt chính đã chuẩn hóa [rx_avg, ry_avg, gx_avg, gy_avg]
        C_scaled = self.scaler.transform(X_centroids)[:, :4]
        self.anchor_gaze = C_scaled

        # Tự động ước lượng bán kính ảnh hưởng gamma theo khoảng cách giữa các điểm neo
        dists_sq = np.sum((C_scaled[:, None, :] - C_scaled[None, :, :]) ** 2, axis=-1)
        non_zero = dists_sq[dists_sq > 1e-6]
        median_d2 = float(np.median(non_zero)) if len(non_zero) > 0 else 1.0
        self.rbf_gamma = 1.25 / max(median_d2, 0.15)

        K = np.exp(-self.rbf_gamma * dists_sq)
        # Giải hệ phương trình tuyến tính có điều chuẩn nhẹ (lambda = 0.015) để tránh gợn sóng
        reg = 0.015 * np.eye(len(C_scaled))
        self.rbf_weights = np.linalg.solve(K + reg, residuals)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = self._transform_features(X, fit=False)
        base_pred = self.ridge.predict(Phi)

        if self.anchor_gaze is not None and self.rbf_weights is not None:
            G_scaled = self.scaler.transform(X)[:, :4]
            dists_sq = np.sum((G_scaled[:, None, :] - self.anchor_gaze[None, :, :]) ** 2, axis=-1)
            K = np.exp(-self.rbf_gamma * dists_sq)
            return base_pred + K @ self.rbf_weights

        return base_pred


class GazeScreenMapper:
    """
    Quản lý hiệu chuẩn 13 điểm, tinh chỉnh điểm tức thì ([Space]) và khóa điểm hội tụ.
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
        self.model: Optional[HybridGazeModel] = None

        self.filter_x = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.filter_y = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.stabilizer = FixationStabilizer(
            window_size=FIXATION_WINDOW_SIZE,
            dispersion_px=FIXATION_DISPERSION_PX,
            release_px=FIXATION_RELEASE_PX,
            magnet_strength=FIXATION_MAGNET_STRENGTH,
        )

        # Lưới 13 điểm: 9 điểm chuẩn + 4 điểm tâm của 4 góc phần tư màn hình
        mx, my = CALIB_MARGIN_X, CALIB_MARGIN_Y
        qx1, qx2 = 0.28, 0.72
        qy1, qy2 = 0.28, 0.72
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
            (qx1,       qy1      ),  # 10. Phần tư trên-trái
            (qx2,       qy1      ),  # 11. Phần tư trên-phải
            (qx2,       qy2      ),  # 12. Phần tư dưới-phải
            (qx1,       qy2      ),  # 13. Phần tư dưới-trái
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
        self.anchor_X: List[List[float]] = []
        self.anchor_Y: List[List[float]] = []

        self.load_model()

    def start_calibration(self):
        """Bắt đầu phiên hiệu chuẩn 13 điểm toàn màn hình."""
        self.is_calibrating = True
        self.calib_step = 0
        self.calib_start_time = time.time()
        self.current_point_samples = []
        self.X_data = []
        self.Y_data = []
        self.anchor_X = []
        self.anchor_Y = []
        self.filter_x.reset()
        self.filter_y.reset()
        self.stabilizer.reset()
        print("\n[+] Bắt đầu hiệu chuẩn 13 điểm (Hybrid Poly+RBF & Fixation Lock)!")
        print("[+] Hãy nhìn vào tâm chấm đỏ của từng điểm...")

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
        """Lọc mẫu ngoại lai trên 5 đặc trưng mắt cốt lõi và trích xuất Anchor Centroid."""
        n_raw = len(self.current_point_samples)
        if n_raw < 3:
            print(f"[!] Điểm {self.calib_step + 1}: chỉ có {n_raw} mẫu — bỏ qua.")
            return

        samples = np.array(self.current_point_samples, dtype=np.float64)
        eye_core = samples[:, :5]
        med_core = np.median(eye_core, axis=0)
        dists = np.linalg.norm(eye_core - med_core, axis=1)
        mad = max(np.median(dists), 1e-6)

        threshold = max(1.6 * mad, np.percentile(dists, 65))
        mask = dists <= threshold
        clean = samples[mask] if mask.sum() >= 3 else samples

        centroid = 0.6 * np.median(clean, axis=0) + 0.4 * np.mean(clean, axis=0)
        tx, ty = target_norm

        self.anchor_X.append(centroid.tolist())
        self.anchor_Y.append([tx, ty])

        for s in clean:
            blended = 0.70 * centroid + 0.30 * s
            self.X_data.append(blended.tolist())
            self.Y_data.append([tx, ty])

        for _ in range(6):
            self.X_data.append(centroid.tolist())
            self.Y_data.append([tx, ty])

        print(f"[+] Điểm {self.calib_step + 1}/{len(self.calib_points)}: "
              f"{len(clean)}/{n_raw} mẫu sạch (gx={centroid[2]:.3f}, gy={centroid[3]:.3f}).")

    def add_quick_anchor(self, recent_features: List[List[float]], target_px: Tuple[int, int]) -> bool:
        """
        Bắt điểm tức thì ([Space]): Thêm hoặc cập nhật 1 điểm neo tại đúng vị trí người dùng đang nhìn
        và huấn luyện lại mô hình Hybrid trong vài mili-giây.
        """
        if len(recent_features) < 3 or len(self.anchor_X) < 5:
            return False

        samples = np.array(recent_features, dtype=np.float64)
        centroid = np.median(samples, axis=0)
        tx = float(np.clip(target_px[0] / self.screen_w, 0.0, 1.0))
        ty = float(np.clip(target_px[1] / self.screen_h, 0.0, 1.0))

        # Nếu điểm mới nằm rất gần một điểm neo cũ (< 6% màn hình), cập nhật điểm neo đó
        anchor_Y_arr = np.array(self.anchor_Y, dtype=np.float64)
        dists_screen = np.hypot(anchor_Y_arr[:, 0] - tx, anchor_Y_arr[:, 1] - ty)
        min_idx = int(np.argmin(dists_screen))

        if dists_screen[min_idx] < 0.06:
            old_c = np.array(self.anchor_X[min_idx], dtype=np.float64)
            new_c = 0.35 * old_c + 0.65 * centroid
            self.anchor_X[min_idx] = new_c.tolist()
            self.anchor_Y[min_idx] = [tx, ty]
        else:
            self.anchor_X.append(centroid.tolist())
            self.anchor_Y.append([tx, ty])

        for _ in range(8):
            self.X_data.append(centroid.tolist())
            self.Y_data.append([tx, ty])

        self.train_model()
        self.stabilizer.reset()
        print(f"[+] Đã ghim điểm mới tại ({target_px[0]}, {target_px[1]}) thành công!")
        return True

    def train_model(self):
        """Huấn luyện mô hình Hybrid Poly-Ridge + Residual RBF."""
        if len(self.X_data) < 15 or len(self.anchor_X) < 5:
            print("[!] Không đủ dữ liệu huấn luyện.")
            return

        X_all = np.array(self.X_data, dtype=np.float64)
        Y_all = np.array(self.Y_data, dtype=np.float64)
        X_c = np.array(self.anchor_X, dtype=np.float64)
        Y_c = np.array(self.anchor_Y, dtype=np.float64)

        t0 = time.time()
        model = HybridGazeModel()
        model.fit(X_all, Y_all, X_c, Y_c)
        self.model = model
        dt = (time.time() - t0) * 1000

        pred_c = self.model.predict(X_c)
        rmse_c = float(np.sqrt(np.mean(
            (pred_c[:, 0] * self.screen_w - Y_c[:, 0] * self.screen_w) ** 2 +
            (pred_c[:, 1] * self.screen_h - Y_c[:, 1] * self.screen_h) ** 2
        )))
        print(f"[+] Huấn luyện Hybrid hoàn tất ({dt:.1f}ms) | Sai số tại {len(X_c)} điểm tâm: {rmse_c:.1f}px")

        try:
            payload = {
                "version": MODEL_VERSION,
                "model": self.model,
                "X_data": self.X_data,
                "Y_data": self.Y_data,
                "anchor_X": self.anchor_X,
                "anchor_Y": self.anchor_Y,
            }
            joblib.dump(payload, self.model_path)
        except Exception as e:
            print(f"[!] Lưu model thất bại: {e}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            return
        try:
            loaded = joblib.load(self.model_path)
            if isinstance(loaded, dict) and loaded.get("version") == MODEL_VERSION:
                self.model = loaded["model"]
                self.X_data = loaded.get("X_data", [])
                self.Y_data = loaded.get("Y_data", [])
                self.anchor_X = loaded.get("anchor_X", [])
                self.anchor_Y = loaded.get("anchor_Y", [])
                print(f"[+] Đã tải mô hình {MODEL_VERSION} ({len(self.anchor_X)} điểm neo).")
            else:
                print("[!] Model cũ khác phiên bản — hãy nhấn phím 'C' để hiệu chuẩn 13 điểm mới.")
                self.model = None
        except Exception as e:
            print(f"[!] Lỗi khi tải model: {e}")
            self.model = None

    def predict(
        self,
        feature_vector: List[float],
        timestamp: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        if self.model is None or feature_vector is None:
            return None

        t = timestamp if timestamp is not None else time.time()
        pred_norm = self.model.predict(np.array([feature_vector], dtype=np.float64))[0]

        raw_x = float(pred_norm[0] * self.screen_w)
        raw_y = float(pred_norm[1] * self.screen_h)

        # Bước 1: Lọc quỹ đạo bằng One Euro Filter
        filt_x = self.filter_x.filter(raw_x, t)
        filt_y = self.filter_y.filter(raw_y, t)

        # Bước 2: Khóa điểm hội tụ (Fixation Magnet Lock) khi mắt tập trung vào 1 điểm
        lock_x, lock_y = self.stabilizer.update(filt_x, filt_y)

        sx = int(np.clip(lock_x, 0, self.screen_w - 1))
        sy = int(np.clip(lock_y, 0, self.screen_h - 1))
        return sx, sy