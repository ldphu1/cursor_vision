import time
from typing import Union, Optional, Tuple
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import MODEL_PATH
from .mapper import GazeScreenMapper
from .utils import get_screen_resolution, ensure_model_exists, set_mouse_position
from .visualizer import draw_visuals, draw_hud, render_screen_canvas


class GazeTracker:
    """
    Trình điều phối theo dõi ánh mắt và điều khiển con trỏ:
    - Nhận diện khuôn mặt & mống mắt 3D qua MediaPipe.
    - Hiệu chuẩn 9 điểm toàn màn hình với bộ lọc loại bỏ chớp mắt & nhiễu.
    - Ánh xạ RidgeCV + Polynomial bậc 2 kèm bộ lọc 1€ Filter triệt tiêu rung giật.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        control_mouse: bool = False,
        model_path: str = MODEL_PATH
    ):
        self.source = source
        self.control_mouse = control_mouse
        self.model_path = model_path

        self.screen_w, self.screen_h = get_screen_resolution()
        self.mapper = GazeScreenMapper(self.screen_w, self.screen_h)
        self.last_gaze_pt: Optional[Tuple[int, int]] = None
        self.is_fullscreen = False

    def run(self):
        """Khởi chạy vòng lặp xử lý video."""
        print(f"[+] Độ phân giải màn hình phát hiện: {self.screen_w} x {self.screen_h}")

        model_path = ensure_model_exists(self.model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[!] Lỗi: Không thể mở webcam/video nguồn: {self.source}")
            return

        # Cửa sổ hiển thị mô phỏng màn hình (Screen Canvas)
        screen_win_name = "Screen Gaze Mapping"
        cv2.namedWindow(screen_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(screen_win_name, min(1280, self.screen_w), min(720, self.screen_h))

        camera_win_name = "MediaPipe - Gaze & Eye Landmarks"
        cv2.namedWindow(camera_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(camera_win_name, 640, 480)

        prev_time = time.time()
        frame_timestamp_ms = 0
        move_mouse_enabled = self.control_mouse

        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("[-] Hết video hoặc không thể đọc frame.")
                    break

                if isinstance(self.source, int):
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                fps_cap = cap.get(cv2.CAP_PROP_FPS)
                step_ms = int(1000 / fps_cap) if fps_cap > 0 else 33
                frame_timestamp_ms += step_ms

                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                # Xử lý hình ảnh và lấy vector đặc trưng 12 chiều
                annotated_frame, feat_vec, eye_info, head_info, is_blinking = draw_visuals(
                    frame, detection_result
                )

                # Tính FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                calib_info = None
                gaze_screen_pt = None

                # Xử lý logic Hiệu chuẩn / Dự đoán
                if self.mapper.is_calibrating:
                    calib_info, finished = self.mapper.update_calibration(
                        feat_vec, is_blinking=is_blinking
                    )
                else:
                    # Trạng thái theo dõi ánh mắt (Tracking)
                    if feat_vec is not None and self.mapper.model is not None:
                        if not is_blinking:
                            gaze_screen_pt = self.mapper.predict(feat_vec, timestamp=curr_time)
                            if gaze_screen_pt is not None:
                                self.last_gaze_pt = gaze_screen_pt
                                if move_mouse_enabled:
                                    set_mouse_position(gaze_screen_pt[0], gaze_screen_pt[1])
                        else:
                            # Khi chớp mắt: Giữ nguyên vị trí con trỏ cuối cùng, không nhảy giật
                            gaze_screen_pt = self.last_gaze_pt

                # Vẽ Canvas màn hình
                screen_canvas = render_screen_canvas(
                    screen_w=self.screen_w,
                    screen_h=self.screen_h,
                    calib_info=calib_info,
                    gaze_screen_pt=gaze_screen_pt,
                    has_model=(self.mapper.model is not None),
                )

                # Vẽ HUD lên cửa sổ Camera
                draw_hud(
                    frame=annotated_frame,
                    eye_info=eye_info,
                    head_info=head_info,
                    gaze_pt=gaze_screen_pt,
                    is_calibrating=self.mapper.is_calibrating,
                    has_model=(self.mapper.model is not None),
                    fps=fps,
                    mouse_enabled=move_mouse_enabled,
                )

                # Hiển thị cả 2 cửa sổ
                cv2.imshow(camera_win_name, annotated_frame)
                cv2.imshow(screen_win_name, screen_canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("c") or key == ord("C"):
                    # Tự động chuyển toàn màn hình khi hiệu chuẩn để tọa độ đạt độ chính xác 1:1
                    cv2.setWindowProperty(screen_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    self.is_fullscreen = True
                    self.mapper.start_calibration()
                elif key == ord("f") or key == ord("F"):
                    # Phím tắt chuyển đổi Toàn màn hình / Cửa sổ
                    self.is_fullscreen = not self.is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if self.is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(screen_win_name, cv2.WND_PROP_FULLSCREEN, prop)
                    if not self.is_fullscreen:
                        cv2.resizeWindow(screen_win_name, min(1280, self.screen_w), min(720, self.screen_h))
                elif key == ord("m") or key == ord("M"):
                    move_mouse_enabled = not move_mouse_enabled
                    print(f"[*] Chế độ điều khiển chuột: {'BẬT' if move_mouse_enabled else 'TẮT'}")

        cap.release()
        cv2.destroyAllWindows()


def run_tracker(source: Union[int, str] = 0, control_mouse: bool = False):
    """Hàm tiện ích chạy Gaze Tracker nhanh chóng."""
    tracker = GazeTracker(source=source, control_mouse=control_mouse)
    tracker.run()
