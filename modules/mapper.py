import os
import time
from typing import Optional, Tuple, List, Dict, Any
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

# Số chiều của Rich Feature Vector
FEATURE_DIM = 12


def _build_pipeline() -> Pipeline:
    """
    SVR-RBF pipeline kết hợp StandardScaler:
    - Kernel RBF nội suy phi tuyến mượt mà cho cả 12 đặc trưng.
    - C=100.0: khớp chính xác dữ liệu hiệu chuẩn của từng người dùng.
    - epsilon=0.001: miễn nhiễm với vi rung giật cảm biến nhỏ.
    - Tốc độ: fit ~1.5s, predict ~10ms (đáp ứng 80-100 FPS).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(
            SVR(kernel="rbf", C=100.0, epsilon=0.001, gamma="scale"),
            n_jobs=-1,
        )),
    ])


class GazeScreenMapper:
    """
    Quản lý quy trình hiệu chuẩn 9 điểm và ánh xạ ánh mắt lên màn hình máy tính.

    Rich Feature Vector 12 chiều:
        [rx_l, ry_l, rx_r, ry_r, rx_avg, ry_avg, vergence, ear_avg, cross, pitch_norm, yaw_norm, roll_norm]
        - rx_l, ry_l, rx_r, ry_r: Tọa độ mống mắt 2 bên (Screen-Right & Screen-Down).
        - rx_avg, ry_avg: Trung bình cộng 2 mắt (triệt tiêu 50% vi rung giật webcam).
        - vergence: Độ hội tụ mống mắt = rx_l - rx_r (xác định cự ly & góc xiên).
        - ear_avg: Độ mở mí mắt trung bình (tín hiệu nhìn lên/xuống cực mạnh cho trục Y).
        - cross: rx_avg * ry_avg (nội suy phi tuyến cho 4 góc chéo màn hình).
        - pitch_norm, yaw_norm, roll_norm: Góc xoay đầu 3D (bù trừ cử động đầu nhẹ).

    Pipeline: StandardScaler → SVR-RBF.
    Smoothing: One Euro Filter (Casiez et al., CHI 2012) chống rung khi dừng nhìn và không trễ khi liếc nhanh.
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

        # 9 điểm hiệu chuẩn: Bắt đầu từ tâm, sau đó theo kim đồng hồ quanh 8 hướng
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

        self.is_calibrating     = False
        self.calib_step         = 0
        self.calib_start_time   = 0.0
        self.settle_time        = CALIB_SETTLE_TIME
        self.collect_time       = CALIB_COLLECT_TIME
        self.time_per_point     = CALIB_TIME_PER_POINT

        self.current_point_samples: List[List[float]] = []
        self.X_data: List[List[float]] = []
        self.Y_data: List[List[float]] = []   # Tọa độ chuẩn hóa [0.0, 1.0]

        self.load_model()

    # ------------------------------------------------------------------ #
    # CALIBRATION                                                          #
    # ------------------------------------------------------------------ #

    def start_calibration(self):
        """Bắt đầu phiên hiệu chuẩn 9 điểm toàn màn hình (Fullscreen)."""
        self.is_calibrating         = True
        self.calib_step             = 0
        self.calib_start_time       = time.time()
        self.current_point_samples  = []
        self.X_data                 = []
        self.Y_data                 = []
        self.filter_x.reset()
        self.filter_y.reset()
        print("\n[+] Bắt đầu hiệu chuẩn 9 điểm độ chính xác cao (Rich 12D Vector)!")
        print("[+] Hãy nhìn thẳng vào tâm chấm tròn mỗi khi điểm xuất hiện...")

    def update_calibration(
        self,
        feature_vector: Optional[List[float]],
        is_blinking: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Cập nhật tiến trình hiệu chuẩn.
        - Settle phase (0.8s): Chờ mắt định vị vào điểm tròn.
        - Collect phase (1.1s): Thu thập mẫu, bỏ qua frame chớp mắt.
        """
        if not self.is_calibrating:
            return None, False

        elapsed = time.time() - self.calib_start_time

        if self.calib_step >= len(self.calib_points):
            self.train_model()
            self.is_calibrating = False
            return None, True

        target_pt   = self.calib_points[self.calib_step]
        target_norm = self.norm_calib_points[self.calib_step]

        if elapsed < self.settle_time:
            phase    = "settle"
            progress = min(1.0, elapsed / self.settle_time)
        else:
            phase    = "collect"
            progress = min(1.0, (elapsed - self.settle_time) / self.collect_time)
            if not is_blinking and feature_vector is not None:
                self.current_point_samples.append(list(feature_vector))

        if elapsed >= self.time_per_point:
            self._finalize_current_point(target_norm)
            self.calib_step         += 1
            self.calib_start_time   = time.time()
            self.current_point_samples = []

            if self.calib_step >= len(self.calib_points):
                self.train_model()
                self.is_calibrating = False
                return None, True

        return {
            "point":        target_pt,
            "step":         self.calib_step,
            "total_steps":  len(self.calib_points),
            "phase":        phase,
            "progress":     progress,
            "sample_count": len(self.current_point_samples),
            "is_blinking":  is_blinking,
        }, False

    def _finalize_current_point(self, target_norm: Tuple[float, float]):
        """Lọc bỏ outlier bằng MAD (Median Absolute Deviation) và lưu mẫu sạch."""
        n_raw = len(self.current_point_samples)
        if n_raw < 3:
            print(f"[!] Điểm {self.calib_step + 1}: chỉ có {n_raw} mẫu — bỏ qua.")
            return

        samples  = np.array(self.current_point_samples, dtype=np.float64)
        median_s = np.median(samples, axis=0)
        dists    = np.linalg.norm(samples - median_s, axis=1)
        mad      = max(np.median(dists), 1e-6)
        mask     = dists <= 2.5 * mad
        clean    = samples[mask] if mask.sum() >= 3 else samples

        tx, ty = target_norm
        for s in clean:
            self.X_data.append(s.tolist())
            self.Y_data.append([tx, ty])

        print(f"[+] Điểm {self.calib_step + 1}/{len(self.calib_points)}: "
              f"{len(clean)}/{n_raw} mẫu sạch.")

    # ------------------------------------------------------------------ #
    # TRAINING                                                             #
    # ------------------------------------------------------------------ #

    def train_model(self):
        """Huấn luyện SVR-RBF trên tập mẫu 12 chiều."""
        if len(self.X_data) < 20:
            print("[!] Không đủ dữ liệu huấn luyện (cần ≥20 mẫu).")
            return

        X = np.array(self.X_data, dtype=np.float64)
        Y = np.array(self.Y_data, dtype=np.float64)

        print(f"\n[+] Huấn luyện SVR-RBF trên {len(X)} mẫu (Vector 12 chiều)...")
        t0 = time.time()
        self.model = _build_pipeline()
        self.model.fit(X, Y)
        dt = (time.time() - t0) * 1000

        # Ước lượng sai số RMSE trên tập huấn luyện
        pred = self.model.predict(X)
        rmse = float(np.sqrt(np.mean(
            (pred[:, 0] * self.screen_w  - Y[:, 0] * self.screen_w )**2 +
            (pred[:, 1] * self.screen_h - Y[:, 1] * self.screen_h)**2
        )))
        print(f"[+] Huấn luyện thành công trong {dt:.0f}ms! RMSE tập mẫu: {rmse:.1f}px")

        try:
            joblib.dump(self.model, self.model_path)
            print(f"[+] Đã lưu trọng số mô hình vào: {self.model_path}")
        except Exception as e:
            print(f"[!] Lưu model thất bại: {e}")

    # ------------------------------------------------------------------ #
    # LOAD / SAVE                                                          #
    # ------------------------------------------------------------------ #

    def load_model(self):
        """Tải mô hình từ đĩa và kiểm tra số chiều tương thích."""
        if not os.path.exists(self.model_path):
            return
        try:
            loaded = joblib.load(self.model_path)
            step_names = {name for name, _ in loaded.steps}
            if "svr" in step_names:
                # Kiểm tra số chiều feature đầu vào của model đã lưu
                scaler = loaded.named_steps.get("scaler")
                if scaler is not None and hasattr(scaler, "n_features_in_"):
                    if scaler.n_features_in_ != FEATURE_DIM:
                        print(f"[!] Model cũ có {scaler.n_features_in_} features (yêu cầu {FEATURE_DIM}D).")
                        print("[*] Hãy nhấn phím 'C' trên cửa sổ camera để hiệu chuẩn lại với 12D.")
                        self.model = None
                        return

                self.model = loaded
                print(f"[+] Đã tải mô hình SVR 12D từ: {self.model_path}")
            else:
                print("[!] Model cũ không tương thích — hãy nhấn phím 'C' để hiệu chuẩn lại.")
                self.model = None
        except Exception as e:
            print(f"[!] Lỗi khi tải model: {e}")
            self.model = None

    # ------------------------------------------------------------------ #
    # PREDICT                                                              #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        feature_vector: List[float],
        timestamp: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Dự đoán tọa độ pixel màn hình (Sx, Sy) từ Rich Feature Vector 12 chiều.
        Áp dụng One Euro Filter để triệt tiêu rung giật khi dừng nhìn và không trễ khi liếc nhanh.
        """
        if self.model is None or feature_vector is None:
            return None

        t = timestamp if timestamp is not None else time.time()
        pred_norm = self.model.predict(np.array([feature_vector], dtype=np.float64))[0]

        raw_x = float(np.clip(pred_norm[0] * self.screen_w, 0, self.screen_w  - 1))
        raw_y = float(np.clip(pred_norm[1] * self.screen_h, 0, self.screen_h - 1))

        return int(self.filter_x.filter(raw_x, t)), int(self.filter_y.filter(raw_y, t))
