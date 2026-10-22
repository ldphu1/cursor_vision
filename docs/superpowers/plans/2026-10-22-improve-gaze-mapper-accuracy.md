# Kế hoạch Triển khai: Cải thiện Độ chính xác và Độ ổn định của Gaze Screen Mapper

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cải thiện độ chính xác dự đoán tọa độ màn hình của `GazeScreenMapper`, giảm rung giật trục Y, triệt tiêu hiện tượng văng trỏ chuột do bùng nổ chiều đặc trưng, và hỗ trợ bù trôi nhanh (Quick Re-centering) khi đổi tư thế ngồi.

**Architecture:** 
1. Tách mô hình hồi quy thành 2 pipeline độc lập `model_x` và `model_y`, cho phép trục Y có regularization alpha riêng biệt để chống nhiễu mí mắt.
2. Xây dựng bộ trích xuất đặc trưng có định hướng vật lý (Physical Feature Interaction) thay thế `PolynomialFeatures(degree=2)` thuần túy (giảm từ 90 chiều xuống ~20 chiều tương tác có chọn lọc giữa mắt và đầu).
3. Cải tiến bộ lọc mẫu hiệu chuẩn: giữ lại phương sai tự nhiên của các mẫu sạch thay vì ép cứng về centroid nhân tạo, sử dụng `sample_weight`.
4. Bổ sung cơ chế bù trôi nhanh 1 điểm (Quick Re-centering - Phím `R`) và vùng chết chống rung vi nhãn cầu (micro dead-zone).

**Tech Stack:** Python 3.11, scikit-learn, numpy, OpenCV, MediaPipe, unittest.

**Spec:** Đề xuất cải tiến độ chính xác `GazeScreenMapper` trong dự án `nepro`.

## Global Constraints
- Duy trì tính tương thích ngược với API hiện tại: `predict(feature_vector, timestamp) -> Optional[Tuple[int, int]]`.
- Không làm tăng độ trễ suy luận mỗi khung hình (tổng thời gian xử lý mapper < 2ms/frame).
- Tương thích tốt với mô hình lưu trữ `gaze_ridge_model.pkl` (tự động phát hiện phiên bản mới `poly_ridge_v3.0`).

## Review Focus
1. Người dùng đổi tư thế ngồi hoặc nghiêng đầu nhẹ: con trỏ không bị phóng đại quá mức hoặc bay khỏi màn hình.
2. Chớp mắt trong khi hiệu chuẩn: mẫu chớp mắt bị loại bỏ hoàn toàn, không gây lệch mô hình.
3. Khi mắt đứng yên nhìn vào 1 điểm (fixation): con trỏ không bị rung giật vi mô (jitter < 5px).
4. Phím tắt `R` (Re-center): bù ngay lập tức độ lệch tâm mà không cần chạy lại toàn bộ quy trình hiệu chuẩn 9 điểm.
5. Khi chưa có mô hình hoặc mất khuôn mặt: không sinh lỗi ngoại lệ `NoneType` hoặc crash chương trình.

---

### Task 1: Xây dựng Feature Engineering định hướng vật lý (Physical Interaction)

**Files:**
- Create: `tests/test_mapper.py`
- Modify: `modules/mapper.py`

**Interfaces:**
- Consumes: `feature_vector` (12D: `[rx_l, ry_l, rx_r, ry_r, rx_avg, ry_avg, ear_avg, head_x, head_y, pitch_norm, yaw_norm, roll_norm]`)
- Produces: `expand_features(X: np.ndarray) -> np.ndarray` (kích thước ~19 chiều có ý nghĩa vật lý)

- [x] **Step 1: Viết test kiểm tra hàm `expand_features`**

```python
# tests/test_mapper.py
import unittest
import numpy as np

class TestMapperFeatures(unittest.TestCase):
    def test_expand_features_shape(self):
        from modules.mapper import expand_features
        X_raw = np.random.randn(10, 12)
        X_exp = expand_features(X_raw)
        self.assertEqual(X_exp.shape[0], 10)
        self.assertEqual(X_exp.shape[1], 19)
```

- [x] **Step 2: Chạy test để xác nhận test thất bại**

Run: `.venv\Scripts\python.exe -m unittest tests/test_mapper.py`
Expected: FAIL (ImportError hoặc expand_features chưa tồn tại).

- [x] **Step 3: Triển khai hàm `expand_features(X: np.ndarray) -> np.ndarray` trong `modules/mapper.py`**

Trích xuất:
- Mắt cơ bản: `rx_l, ry_l, rx_r, ry_r, rx_avg, ry_avg` (6)
- Bậc 2 mắt: `rx_avg**2, ry_avg**2, rx_avg * ry_avg` (3)
- Đầu: `head_x, head_y, pitch_norm, yaw_norm, roll_norm` (5)
- Bậc 2 đầu: `pitch_norm**2, yaw_norm**2` (2)
- Tương tác mắt-đầu (Head-Eye interactions): `rx_avg * yaw_norm`, `ry_avg * pitch_norm`, `rx_avg * pitch_norm` (3)
Tổng: 19 đặc trưng không đa cộng tuyến.

- [x] **Step 4: Chạy lại test để xác nhận test vượt qua**

Run: `.venv\Scripts\python.exe -m unittest tests/test_mapper.py`
Expected: PASS.

---

### Task 2: Tách biệt Dual-Axis RidgeCV (`model_x`, `model_y`) & Cải tiến bộ lọc mẫu hiệu chuẩn

**Files:**
- Modify: `modules/mapper.py`
- Modify: `tests/test_mapper.py`

**Interfaces:**
- Consumes: `X_data`, `Y_data`, `sample_weights`
- Produces: `self.model_x: Pipeline`, `self.model_y: Pipeline`, `train_model()`, `predict()`

- [x] **Step 1: Viết test cho Dual-Axis training và prediction**

```python
# Thêm vào tests/test_mapper.py
def test_dual_axis_train_and_predict(self):
    from modules.mapper import GazeScreenMapper
    mapper = GazeScreenMapper(screen_w=1920, screen_h=1080, model_path="test_gaze_model.pkl")
    # Giả lập dữ liệu hiệu chuẩn
    X = np.random.randn(30, 12)
    Y = np.random.rand(30, 2)
    mapper.X_data = X.tolist()
    mapper.Y_data = Y.tolist()
    mapper.train_model()
    self.assertIsNotNone(mapper.model_x)
    self.assertIsNotNone(mapper.model_y)
    pred = mapper.predict(X[0].tolist())
    self.assertIsNotNone(pred)
    self.assertTrue(0 <= pred[0] < 1920)
    self.assertTrue(0 <= pred[1] < 1080)
```

- [x] **Step 2: Chạy test để xác nhận test thất bại**

Run: `.venv\Scripts\python.exe -m unittest tests/test_mapper.py`
Expected: FAIL (chưa có model_x/model_y).

- [x] **Step 3: Triển khai Dual-Axis Ridge Pipelines và cập nhật `_finalize_current_point`**

Trong `modules/mapper.py`:
- `_build_axis_pipeline(alphas)`: `Pipeline([("scaler", RobustScaler()), ("ridge", RidgeCV(alphas=alphas, scoring="neg_mean_squared_error"))])`
- `model_x` dùng `alphas_x = np.logspace(-3, 3, 35)`
- `model_y` dùng `alphas_y = np.logspace(-2, 4, 35)` (phạt mạnh hơn cho trục Y để chống nhiễu mí mắt)
- Trong `_finalize_current_point`: Không nén toàn bộ về `0.65*centroid + 0.35*s`. Lưu các mẫu sạch thực tế `s` và thêm `sample_weights` (hoặc nhân bản nhẹ centroid 3 lần làm điểm neo).
- Cập nhật `MODEL_VERSION = "dual_poly_ridge_v3.0"`.

- [x] **Step 4: Chạy test để xác nhận test vượt qua**

Run: `.venv\Scripts\python.exe -m unittest tests/test_mapper.py`
Expected: PASS.

---

### Task 3: Triển khai Bù trôi nhanh (Quick Re-centering) và Vùng chết vi nhãn cầu (Micro Dead-Zone)

**Files:**
- Modify: `modules/config.py`
- Modify: `modules/mapper.py`
- Modify: `tests/test_mapper.py`

**Interfaces:**
- Consumes: `DEAD_ZONE_PIXELS`, phím bấm từ người dùng
- Produces: `recenter(feature_vector)`, `reset_offset()`, `offset_x`, `offset_y`

- [x] **Step 1: Viết test kiểm tra tính năng Re-center và Dead-zone**

```python
# Thêm vào tests/test_mapper.py
def test_recenter_offset(self):
    from modules.mapper import GazeScreenMapper
    mapper = GazeScreenMapper(screen_w=1920, screen_h=1080)
    # Giả lập dự đoán thô là (900, 500), tâm màn hình là (960, 540)
    mapper.apply_recenter(pred_raw_x=900, pred_raw_y=500)
    self.assertEqual(mapper.offset_x, 60)
    self.assertEqual(mapper.offset_y, 40)
```

- [x] **Step 2: Chạy test để xác nhận test thất bại**

Run: `.venv\Scripts\python.exe -m unittest tests/test_mapper.py`
Expected: FAIL.

- [x] **Step 3: Triển khai Re-center và Dead-zone trong `modules/mapper.py` và `modules/config.py`**

- Thêm cấu hình trong `config.py`: `DEAD_ZONE_RADIUS = 6` (pixels), `RECENTER_SMOOTH_FACTOR = 0.5`.
- Trong `GazeScreenMapper`:
  - Thêm `self.offset_x = 0.0`, `self.offset_y = 0.0`.
  - Hàm `recenter()`: tính sai lệch giữa vị trí dự đoán hiện tại và tâm màn hình `(screen_w / 2, screen_h / 2)` và cập nhật `offset_x, offset_y`.
  - Trong `predict()`: cộng offset `(offset_x, offset_y)` trước khi clip vào màn hình. Áp dụng dead-zone nếu độ dời giữa 2 frame liên tiếp < `DEAD_ZONE_RADIUS`.

- [x] **Step 4: Chạy test để xác nhận test vượt qua**

Run: `.venv\Scripts\python.exe -m unittest tests/test_mapper.py`
Expected: PASS.

---

### Task 4: Tích hợp Phím tắt `R` và Cập nhật Giao diện HUD / Canvas

**Files:**
- Modify: `modules/tracker.py`
- Modify: `modules/visualizer.py`

**Interfaces:**
- Consumes: Phím `R`/`r` trong vòng lặp `cv2.waitKey`
- Produces: Cập nhật HUD hiển thị trạng thái Re-center và hướng dẫn phím bấm `[R] Can giua`

- [x] **Step 1: Thêm xử lý phím `R` / `r` trong `modules/tracker.py`**
  - Khi nhấn `r` hoặc `R`: gọi `self.mapper.recenter_current()` hoặc cập nhật offset từ điểm nhìn hiện tại.
  - In thông báo: `[+] Đã cân giữa màn hình (Offset: dx, dy)`.
  - Nhấn `Shift + R` hoặc phím giữ để reset offset về 0.

- [x] **Step 2: Cập nhật `draw_hud` và `render_screen_canvas` trong `modules/visualizer.py`**
  - Hiển thị thông số Offset `(dx, dy)` trên HUD nếu có bù trôi.
  - Hiển thị hướng dẫn: `[C] Hieu chuan | [R] Can giua | [M] Chuot | [Q] Thoat`.

- [x] **Step 3: Kiểm tra chạy thử syntax và logic qua script kiểm thử**

Run: `.venv\Scripts\python.exe -c "import modules.tracker, modules.visualizer, modules.mapper; print('All modules imported successfully!')"`
Expected: PASS.

---

### Task 5: Kiểm tra Toàn diện và Xác minh

**Files:**
- Modify: `tests/test_mapper.py`
- Test: chạy toàn bộ `unittest`

- [x] **Step 1: Chạy toàn bộ test suite**

Run: `.venv\Scripts\python.exe -m unittest discover tests`
Expected: Tất cả các test đều PASS.

- [x] **Step 2: Dọn dẹp các file tạm test**
  - Xóa `test_gaze_model.pkl` nếu có sinh ra trong quá trình test.
