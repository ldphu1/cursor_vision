import os
import time
from typing import Optional, Tuple, List, Dict, Any
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, WhiteKernel, ConstantKernel

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

FEATURE_DIM = 8
MODEL_VERSION = "gpr_grid_ensemble_v3.0"


class GPRGridEnsemble:
    """
    Hệ thống ánh xạ không dùng Ridge:
    1. Gaussian Process Regression (GPR) với nhân hỗn hợp:
       - DotProduct(): nắm bắt xu hướng mặt phẳng toàn cục (ngoại suy mượt ra sát viền).
       - Matern(nu=1.5): nội suy phi tuyến cục bộ chính xác tại từng vùng mắt.
       - WhiteKernel(): tự động học phương sai nhiễu của webcam.
    2. Piecewise Quadrant Grid Mapping (Nội suy hình học trực tiếp từ 9 điểm tâm):
       - Chia 9 điểm thành 4 góc phần tư quanh tâm màn hình (Index 0) và nội suy tỷ lệ trực tiếp,
         đảm bảo không bao giờ bị lệch tâm hay overfit.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2))
            + ConstantKernel(0.5, (1e-3, 1e2)) * Matern(length_scale=1.5, length_scale_bounds=(0.2, 10.0), nu=1.5)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=42,
        )
        self.anchors_gaze: Optional[np.ndarray] = None  # Shape (9, 2)
        self.anchors_screen: Optional[np.ndarray] = None  # Shape (9, 2)

    def fit(self, X_sub: np.ndarray, Y_sub: np.ndarray, X_centroids: np.ndarray, Y_centroids: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_sub)
        self.gpr.fit(X_scaled, Y_sub)

        if len(X_centroids) == 9:
            # Lưu 2 đặc trưng chính (gx_comp, gy_comp) của 9 điểm tâm để nội suy hình học trực tiếp
            self.anchors_gaze = X_centroids[:, :2].copy()
            self.anchors_screen = Y_centroids.copy()
        return self

    def _predict_grid(self, gx: float, gy: float) -> Optional[np.ndarray]:
        """
        Nội suy hình học tách trục (Piecewise Axis Interpolation) dựa trên 9 điểm chuẩn:
        0: Tâm, 1: Trên-Trái, 2: Giữa-Trên, 3: Trên-Phải, 4: Giữa-Phải,
        5: Dưới-Phải, 6: Giữa-Dưới, 7: Dưới-Trái, 8: Giữa-Trái.
        """
        if self.anchors_gaze is None or self.anchors_screen is None:
            return None

        G = self.anchors_gaze
        S = self.anchors_screen

        # Tâm mắt tại điểm giữa màn hình
        cx_g, cy_g = G[0, 0], G[0, 1]
        cx_s, cy_s = S[0, 0], S[0, 1]

        # Biên trái (trung bình cột trái: điểm 1, 8, 7) và biên phải (điểm 3, 4, 5)
        left_g  = float(0.25 * G[1, 0] + 0.50 * G[8, 0] + 0.25 * G[7, 0])
        right_g = float(0.25 * G[3, 0] + 0.50 * G[4, 0] + 0.25 * G[5, 0])

        # Biên trên (điểm 1, 2, 3) và biên dưới (điểm 7, 6, 5)
        top_g = float(0.25 * G[1, 1] + 0.50 * G[2, 1] + 0.25 * G[3, 1])
        bot_g = float(0.25 * G[7, 1] + 0.50 * G[6, 1] + 0.25 * G[5, 1])

        left_s, right_s = S[8, 0], S[4, 0]
        top_s, bot_s = S[2, 1], S[6, 1]

        # Nội suy từng nửa trục X (Trái / Phải so với tâm)
        if gx <= cx_g:
            denom = max( abs(cx_g - left_g), 1e-4 )
            t_x = (cx_g - gx) / denom
            sx = cx_s - t_x * (cx_s - left_s)
        else:
            denom = max( abs(right_g - cx_g), 1e-4 )
            t_x = (gx - cx_g) / denom
            sx = cx_s + t_x * (right_s - cx_s)

        # Nội suy từng nửa trục Y (Trên / Dưới so với tâm)
        if gy <= cy_g:
            denom = max( abs(cy_g - top_g), 1e-4 )
            t_y = (cy_g - gy) / denom
            sy = cy_s - t_y * (cy_s - top_s)
        else:
            denom = max( abs(bot_g - cy_g), 1e-4 )
            t_y = (gy - cy_g) / denom
            sy = cy_s + t_y * (bot_s - cy_s)

        return np.array([sx, sy], dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        gpr_pred = self.gpr.predict(X_scaled)[0]

        grid_pred = self._predict_grid(float(X[0, 0]), float(X[0, 1]))
        if grid_pred is not None:
            # Kết hợp 70% GPR (8D phi tuyến) + 30% Khung lưới hình học 2D để chống trôi tuyệt đối
            return np.array([0.70 * gpr_pred + 0.30 * grid_pred])
        return np.array([gpr_pred])


class GazeScreenMapper:
    """Quản lý quy trình hiệu chuẩn 9 điểm và ánh xạ GPR + Grid Ensemble."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        model_path: str = GAZE_MODEL_PATH,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.model_path = model_path
        self.model: Optional[GPRGridEnsemble] = None

        self.filter_x = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.filter_y = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)

        mx, my = CALIB_MARGIN_X, CALIB_MARGIN_Y
        self.norm_calib_points: List[Tuple[float, float]] = [
            (0.5,       0.5      ),  # 0. Tâm màn hình
            (mx,        my       ),  # 1. Góc trên trái
            (0.5,       my       ),  # 2. Giữa trên
            (1.0 - mx,  my       ),  # 3. Góc trên phải
            (1.0 - mx,  0.5      ),  # 4. Giữa phải
            (1.0 - mx,  1.0 - my ),  # 5. Góc dưới phải
            (0.5,       1.0 - my ),  # 6. Giữa dưới
            (mx,        1.0 - my ),  # 7. Góc dưới trái
            (mx,        0.5      ),  # 8. Giữa trái
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
        self.X_sub: List[List[float]] = []
        self.Y_sub: List[List[float]] = []
        self.X_centroids: List[List[float]] = []
        self.Y_centroids: List[List[float]] = []

        self.load_model()

    def start_calibration(self):
        self.is_calibrating = True
        self.calib_step = 0
        self.calib_start_time = time.time()
        self.current_point_samples = []
        self.X_sub = []
        self.Y_sub = []
        self.X_centroids = []
        self.Y_centroids = []
        self.filter_x.reset()
        self.filter_y.reset()
        print("\n[+] Bắt đầu hiệu chuẩn 9 điểm (MediaPipe Blendshapes + Gaussian Process)!")

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
        n_raw = len(self.current_point_samples)
        if n_raw < 3:
            print(f"[!] Điểm {self.calib_step + 1}: chỉ có {n_raw} mẫu — bỏ qua.")
            return

        samples = np.array(self.current_point_samples, dtype=np.float64)
        med = np.median(samples, axis=0)
        dists = np.linalg.norm(samples - med, axis=1)
        mad = max(np.median(dists), 1e-6)
        mask = dists <= 1.8 * mad
        clean = samples[mask] if mask.sum() >= 3 else samples

        centroid = 0.5 * (np.median(clean, axis=0) + np.mean(clean, axis=0))
        tx, ty = target_norm

        self.X_centroids.append(centroid.tolist())
        self.Y_centroids.append([tx, ty])

        # Chọn tối đa 5 mẫu đại diện gần tâm nhất cho mỗi điểm để GPR huấn luyện nhanh và sạch
        sorted_idx = np.argsort(np.linalg.norm(clean - centroid, axis=1))[:5]
        for idx in sorted_idx:
            s = 0.5 * centroid + 0.5 * clean[idx]
            self.X_sub.append(s.tolist())
            self.Y_sub.append([tx, ty])

        self.X_sub.append(centroid.tolist())
        self.Y_sub.append([tx, ty])

        print(f"[+] Điểm {self.calib_step + 1}/9: {len(clean)}/{n_raw} mẫu "
              f"(BlendGaze: gx={centroid[0]:+.3f}, gy={centroid[1]:+.3f})")

    def train_model(self):
        if len(self.X_centroids) < 5:
            print("[!] Không đủ dữ liệu huấn luyện.")
            return

        X_sub = np.array(self.X_sub, dtype=np.float64)
        Y_sub = np.array(self.Y_sub, dtype=np.float64)
        X_c = np.array(self.X_centroids, dtype=np.float64)
        Y_c = np.array(self.Y_centroids, dtype=np.float64)

        print(f"\n[+] Huấn luyện Gaussian Process + Grid trên {len(X_sub)} mẫu...")
        t0 = time.time()
        model = GPRGridEnsemble()
        model.fit(X_sub, Y_sub, X_c, Y_c)
        self.model = model
        dt = (time.time() - t0) * 1000

        preds = np.vstack([self.model.predict(X_c[i:i+1])[0] for i in range(len(X_c))])
        rmse = float(np.sqrt(np.mean(
            (preds[:, 0] * self.screen_w - Y_c[:, 0] * self.screen_w) ** 2 +
            (preds[:, 1] * self.screen_h - Y_c[:, 1] * self.screen_h) ** 2
        )))
        print(f"[+] Hoàn tất trong {dt:.0f}ms! Sai số RMSE trên 9 điểm tâm: {rmse:.1f}px")

        try:
            joblib.dump({"version": MODEL_VERSION, "model": self.model}, self.model_path)
            print(f"[+] Đã lưu mô hình vào: {self.model_path}")
        except Exception as e:
            print(f"[!] Lưu model thất bại: {e}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            return
        try:
            loaded = joblib.load(self.model_path)
            if isinstance(loaded, dict) and loaded.get("version") == MODEL_VERSION:
                self.model = loaded["model"]
                print(f"[+] Đã tải mô hình {MODEL_VERSION} từ: {self.model_path}")
            else:
                print("[!] Phát hiện model cũ — hãy nhấn phím 'C' để hiệu chuẩn với GPR + Blendshapes.")
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

        sx = int(np.clip(self.filter_x.filter(raw_x, t), 0, self.screen_w - 1))
        sy = int(np.clip(self.filter_y.filter(raw_y, t), 0, self.screen_h - 1))
        return sx, sy