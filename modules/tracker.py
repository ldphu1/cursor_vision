from collections import deque
import ctypes
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


def _get_cursor_pos() -> Tuple[int, int]:
    """Lấy tọa độ con trỏ chuột hiện tại trên hệ điều hành Windows."""
    try:
        pt = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)
    except Exception:
        return 0, 0


class GazeTracker:
    """
    Trình điều phối theo dõi ánh mắt và điều khiển con trỏ:
    - Hiệu chuẩn 13 điểm toàn màn hình.
    - Hỗ trợ Click chuột chọn điểm trên màn hình + nhấn [Space] để bắt dính điểm bất kỳ.
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
        self.recent_features: deque = deque(maxlen=18)
        self.clicked_target_pt: Optional[Tuple[int, int]] = None

    def _on_screen_mouse(self, event, x, y, flags, param):
        """Sự kiện click chuột trái lên cửa sổ Screen Gaze Mapping để đặt điểm đích cần tinh chỉnh."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_target_pt = (int(x), int(y))
            print(f"[*] Đã chọn điểm đích ({x}, {y}) — Hãy nhìn vào điểm màu tím và nhấn [Space]!")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clicked_target_pt = None

    def run(self):
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

        screen_win_name = "Screen Gaze Mapping"
        cv2.namedWindow(screen_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(screen_win_name, min(1280, self.screen_w), min(720, self.screen_h))
        cv2.setMouseCallback(screen_win_name, self._on_screen_mouse)

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

                annotated_frame, feat_vec, eye_info, head_info, is_blinking = draw_visuals(
                    frame, detection_result
                )

                if feat_vec is not None and not is_blinking:
                    self.recent_features.append(feat_vec)

                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                calib_info = None
                gaze_screen_pt = None

                if self.mapper.is_calibrating:
                    calib_info, finished = self.mapper.update_calibration(
                        feat_vec, is_blinking=is_blinking
                    )
                else:
                    if feat_vec is not None and self.mapper.model is not None:
                        if not is_blinking:
                            gaze_screen_pt = self.mapper.predict(feat_vec, timestamp=curr_time)
                            if gaze_screen_pt is not None:
                                self.last_gaze_pt = gaze_screen_pt
                                if move_mouse_enabled:
                                    set_mouse_position(gaze_screen_pt[0], gaze_screen_pt[1])
                        else:
                            gaze_screen_pt = self.last_gaze_pt

                screen_canvas = render_screen_canvas(
                    screen_w=self.screen_w,
                    screen_h=self.screen_h,
                    calib_info=calib_info,
                    gaze_screen_pt=gaze_screen_pt,
                    has_model=(self.mapper.model is not None),
                    target_marker_pt=self.clicked_target_pt,
                )

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

                cv2.imshow(camera_win_name, annotated_frame)
                cv2.imshow(screen_win_name, screen_canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("c") or key == ord("C"):
                    cv2.setWindowProperty(screen_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    self.is_fullscreen = True
                    self.clicked_target_pt = None
                    self.mapper.start_calibration()
                elif key == ord(" "):
                    # Phím [Space]: Bắt dính ngay điểm đang nhìn (dựa trên điểm đã click hoặc vị trí chuột)
                    if not self.mapper.is_calibrating and len(self.recent_features) >= 5:
                        target_pt = self.clicked_target_pt if self.clicked_target_pt is not None else _get_cursor_pos()
                        self.mapper.add_quick_anchor(list(self.recent_features), target_pt)
                elif key == ord("f") or key == ord("F"):
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
    tracker = GazeTracker(source=source, control_mouse=control_mouse)
    tracker.run()