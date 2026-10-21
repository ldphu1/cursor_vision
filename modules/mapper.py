from collections import deque
import os
import time
from typing import Optional, Tuple, List, Dict, Any
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

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

FEATURE_DIM = 6
MODEL_VERSION = "tps_svr_ensemble_v4.0"


class ThinPlateSplineWarper:
    """
    Nội suy mặt phẳng đàn hồi Thin-Plate Spline (TPS - Bookstein 1989):
    - Mô hình hóa phép biến đổi từ tọa độ mắt sang màn hình bằng:
      Mặt phẳng Affine toàn cục (a0 + ax*rx + ay*ry) + Độ uốn cong cục bộ tối thiểu năng lượng (r^2 * ln(r)).
    - Khắc phục nhược điểm của Ridge (không uốn theo từng điểm được) và RBF/GPR (bị sụp ở rìa màn hình).
    """

    def __init__(self, smoothing: float = 0.008):
        self.smoothing = smoothing
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.control_pts: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.affine: Optional[np.ndarray] = None

    @staticmethod
    def _kernel(r: np.ndarray) -> np.ndarray:
        return (r ** 2) * np.log(r + 1e-8)

    def fit(self, G: np.ndarray, S: np.ndarray):
        self.mean = np.mean(G, axis=0)
        self.scale = np.maximum(np.std(G, axis=0), 1e-4)
        P = (G - self.mean) / self.scale
        self.control_pts = P

        n, d = P.shape
        dists = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        K = self._kernel(dists) + self.smoothing * np.eye(n)
        Q = np.hstack([np.ones((n, 1)), P])

        # Hệ phương trình TPS: [K, Q; Q^T, 0] * [W; A] = [S; 0]
        L = np.zeros((n + d + 1, n + d + 1), dtype=np.float64)
        L[:n, :n] = K
        L[:n, n:] = Q
        L[n:, :n] = Q.T
        L[n:, n:] = 1e-6 * np.eye(d + 1)

        rhs = np.zeros((n + d + 1, 2), dtype=np.float64)
        rhs[:n] = S

        params = np.linalg.solve(L, rhs)
        self.weights = params[:n]
        self.affine = params[n:]
        return self

    def predict(self, G: np.ndarray) -> np.ndarray:
        P = (G - self.mean) / self.scale
        dists = np.linalg.norm(P[:, None, :] - self.control_pts[None, :, :], axis=-1)
        U = self._kernel(dists)
        Q = np.hstack([np.ones((len(P), 1)), P])
        return Q @ self.affine + U @ self.weights


class TPSSVREnsemble:
    """
    Kết hợp Thin-Plate Spline (TPS) và Support Vector Regression (SVR):
    - TPS (60%): Nội suy hình học chính xác qua 13 tâm điểm trên cả mắt trung bình, mắt trái và mắt phải.
    - SVR (40%): Học tương quan phi tuyến 6 chiều giữa 2 mắt với điều chuẩn chống nhiễu (C=2.0, epsilon=0.012).
    """

    def __init__(self):
        self.tps_avg = ThinPlateSplineWarper(smoothing=0.006)
        self.tps_full = ThinPlateSplineWarper(smoothing=0.015)
        self.scaler = RobustScaler()
        self.svr = MultiOutputRegressor(
            SVR(kernel="rbf", C=2.0, epsilon=0.012, gamma="scale"),
            n_jobs=-1,
        )

    def fit(self, X_all: np.ndarray, Y_all: np.ndarray, X_centroids: np.ndarray, Y_centroids: np.ndarray):
        self.tps_avg.fit(X_centroids[:, :2], Y_centroids)
        self.tps_full.fit(X_centroids, Y_centroids)

        X_scaled = self.scaler.fit_transform(X_all)
        self.svr.fit(X_scaled, Y_all)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_tps_avg = self.tps_avg.predict(X[:, :2])
        pred_tps_full = self.tps_full.predict(X)
        pred_svr = self.svr.predict(self.scaler.transform(X))
        return 0.40 * pred_tps_avg + 0.25 * pred_tps_full + 0.35 * pred_svr


class GazeScreenMapper:
    """Quản lý hiệu chuẩn 13 điểm (9 điểm biên/tâm + 4 điểm phần tư trong) và ánh xạ TPS + SVR."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        model_path: str = GAZE_MODEL_PATH,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.model_path = model_path
        self.model: Optional[TPSSVREnsemble] = None

        self.raw_history: deque = deque(maxlen=3)
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
            (0.28,      0.28     ),  # 10. Tâm phần tư trên-trái
            (0.72,      0.28     ),  # 11. Tâm phần tư trên-phải
            (0.72,      0.72     ),  # 12. Tâm phần tư dưới-phải
            (0.28,      0.72     ),  # 13. Tâm phần tư dưới-trái
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
        self.X_centroids: List[List[float]] = []
        self.Y_centroids: List[List[float]] = []

        self.load_model()

    def start_calibration(self):
        self.is_calibrating = True
        self.calib_step = 0
        self.calib_start_time = time.time()
        self.current_point_samples = []
        self.X_data = []
        self.Y_data = []
        self.X_centroids = []
        self.Y_centroids = []
        self.raw_history.clear()
        self.filter_x.reset()
        self.filter_y.reset()
        print("\n[+] Bắt đầu hiệu chuẩn 13 điểm (Thin-Plate Spline + SVR Ensemble)!")
        print("[+] Hãy tập trung nhìn vào chấm đỏ nhỏ ở tâm vòng tròn...")

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

        threshold = max(1.6 * mad, np.percentile(dists, 65))
        mask = dists <= threshold
        clean = samples[mask] if mask.sum() >= 3 else samples

        centroid = 0.55 * np.median(clean, axis=0) + 0.45 * np.mean(clean, axis=0)
        tx, ty = target_norm

        self.X_centroids.append(centroid.tolist())
        self.Y_centroids.append([tx, ty])

        for s in clean:
            stabilized = 0.65 * centroid + 0.35 * s
            self.X_data.append(stabilized.tolist())
            self.Y_data.append([tx, ty])

        for _ in range(5):
            self.X_data.append(centroid.tolist())
            self.Y_data.append([tx, ty])

        print(f"[+] Điểm {self.calib_step + 1}/{len(self.calib_points)}: "
              f"{len(clean)}/{n_raw} mẫu sạch (rx={centroid[0]:.3f}, ry={centroid[1]:.3f}).")

    def train_model(self):
        if len(self.X_centroids) < 6:
            print("[!] Không đủ dữ liệu huấn luyện.")
            return

        X_all = np.array(self.X_data, dtype=np.float64)
        Y_all = np.array(self.Y_data, dtype=np.float64)
        X_c = np.array(self.X_centroids, dtype=np.float64)
        Y_c = np.array(self.Y_centroids, dtype=np.float64)

        print(f"\n[+] Huấn luyện Thin-Plate Spline (TPS) + SVR trên {len(X_c)} điểm neo...")
        t0 = time.time()
        model = TPSSVREnsemble()
        model.fit(X_all, Y_all, X_c, Y_c)
        self.model = model
        dt = (time.time() - t0) * 1000

        pred_c = self.model.predict(X_c)
        rmse = float(np.sqrt(np.mean(
            (pred_c[:, 0] * self.screen_w - Y_c[:, 0] * self.screen_w) ** 2 +
            (pred_c[:, 1] * self.screen_h - Y_c[:, 1] * self.screen_h) ** 2
        )))
        print(f"[+] Hoàn tất trong {dt:.1f}ms! Sai số RMSE tại các điểm hiệu chuẩn: {rmse:.1f}px")

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
                print("[!] Phát hiện model phiên bản cũ — hãy nhấn phím 'C' để hiệu chuẩn lại.")
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

        # Lọc trung vị 3 frame liên tiếp trên đặc trưng mắt để cắt các gai nhiễu đột biến
        self.raw_history.append(feature_vector)
        if len(self.raw_history) >= 3:
            clean_feat = np.median(np.array(self.raw_history, dtype=np.float64), axis=0)
        else:
            clean_feat = np.array(feature_vector, dtype=np.float64)

        pred_norm = self.model.predict(clean_feat.reshape(1, -1))[0]

        # Chuyển sang đơn vị PIXEL trước khi đưa vào OneEuroFilter (Sửa lỗi trễ của Bản 1)
        raw_x = float(pred_norm[0] * self.screen_w)
        raw_y = float(pred_norm[1] * self.screen_h)

        filt_x = self.filter_x.filter(raw_x, t)
        filt_y = self.filter_y.filter(raw_y, t)

        sx = int(np.clip(filt_x, 0, self.screen_w - 1))
        sy = int(np.clip(filt_y, 0, self.screen_h - 1))
        return sx, sy