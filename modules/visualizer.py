from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

from .config import (
    RIGHT_IRIS_INDICES,
    RIGHT_EYE_TOP_IDX,
    RIGHT_EYE_BOTTOM_IDX,
    LEFT_IRIS_INDICES,
    LEFT_EYE_TOP_IDX,
    LEFT_EYE_BOTTOM_IDX,
    RIGHT_OUTER_ANCHORS,
    RIGHT_INNER_ANCHORS,
    LEFT_INNER_ANCHORS,
    LEFT_OUTER_ANCHORS,
    NASION_ANCHORS,
    NOSE_TIP_IDX,
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
    BLINK_EAR_THRESHOLD,
)
from .geometry import (
    compute_eye_metrics,
    compute_binocular_skull_metrics,
    get_head_pose_from_matrix,
)


def _mean_landmark_pt(landmarks, indices: List[int], w: int, h: int) -> Tuple[float, float]:
    """Lấy tọa độ trung bình của một cụm điểm mốc để giảm nhiễu vi mô."""
    return (
        float(np.mean([landmarks[i].x for i in indices]) * w),
        float(np.mean([landmarks[i].y for i in indices]) * h),
    )


def draw_visuals(
    frame: np.ndarray,
    detection_result,
    draw_bbox: bool = True
) -> Tuple[np.ndarray, Optional[List[float]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    """
    Trích xuất Feature Vector 14 chiều (Tách biệt đặc trưng ánh mắt & tư thế đầu):
      - 9 đặc trưng mắt thuần túy (0..8):
        [rx_avg, ry_avg, gx_avg, gy_avg, gy_nasion, rx_l, ry_l, rx_r, ry_r]
      - 5 đặc trưng tư thế/vị trí đầu (9..13):
        [head_x, head_y, pitch_norm, yaw_norm, roll_norm]
    """
    h, w, _ = frame.shape
    annotated = frame.copy()

    if not detection_result.face_landmarks:
        return annotated, None, None, None, False

    landmarks = detection_result.face_landmarks[0]
    transform_matrices = detection_result.facial_transformation_matrixes

    # Tâm mống mắt (trung bình 5 điểm) và cụm điểm neo góc mắt cố định
    p_l_iris = _mean_landmark_pt(landmarks, LEFT_IRIS_INDICES, w, h)
    p_r_iris = _mean_landmark_pt(landmarks, RIGHT_IRIS_INDICES, w, h)

    p_l_out = _mean_landmark_pt(landmarks, LEFT_OUTER_ANCHORS, w, h)
    p_l_in  = _mean_landmark_pt(landmarks, LEFT_INNER_ANCHORS, w, h)
    p_r_in  = _mean_landmark_pt(landmarks, RIGHT_INNER_ANCHORS, w, h)
    p_r_out = _mean_landmark_pt(landmarks, RIGHT_OUTER_ANCHORS, w, h)
    p_nasion = _mean_landmark_pt(landmarks, NASION_ANCHORS, w, h)

    # Điểm mí mắt (chỉ dùng để đo EAR chớp mắt)
    p_l_top = (landmarks[LEFT_EYE_TOP_IDX].x * w, landmarks[LEFT_EYE_TOP_IDX].y * h)
    p_l_bot = (landmarks[LEFT_EYE_BOTTOM_IDX].x * w, landmarks[LEFT_EYE_BOTTOM_IDX].y * h)
    p_r_top = (landmarks[RIGHT_EYE_TOP_IDX].x * w, landmarks[RIGHT_EYE_TOP_IDX].y * h)
    p_r_bot = (landmarks[RIGHT_EYE_BOTTOM_IDX].x * w, landmarks[RIGHT_EYE_BOTTOM_IDX].y * h)

    rx_l, ry_l, ear_l, ex_l, ey_l, L_l, H_l = compute_eye_metrics(
        p_l_out, p_l_in, p_l_top, p_l_bot, p_l_iris
    )
    rx_r, ry_r, ear_r, ex_r, ey_r, L_r, H_r = compute_eye_metrics(
        p_r_in, p_r_out, p_r_top, p_r_bot, p_r_iris
    )

    gx_avg, gy_avg, gy_nasion, iod = compute_binocular_skull_metrics(
        p_l_out, p_l_in, p_l_iris,
        p_r_in, p_r_out, p_r_iris,
        p_nasion,
    )

    is_blinking = (ear_l < BLINK_EAR_THRESHOLD or ear_r < BLINK_EAR_THRESHOLD)

    pitch, yaw, roll = 0.0, 0.0, 0.0
    dir_txt = "Level"
    rmat = np.eye(3)
    if transform_matrices and len(transform_matrices) > 0:
        pitch, yaw, roll, rmat, dir_txt = get_head_pose_from_matrix(transform_matrices[0])

    head_x = float(p_nasion[0] / w - 0.5)
    head_y = float(p_nasion[1] / h - 0.5)

    rx_avg  = float(0.5 * (rx_l + rx_r))
    ry_avg  = float(0.5 * (ry_l + ry_r))
    ear_avg = float(0.5 * (ear_l + ear_r))
    p_norm  = float(pitch / 30.0)
    y_norm  = float(yaw / 30.0)
    r_norm  = float(roll / 30.0)

    # 9 đặc trưng mắt + 5 đặc trưng đầu = 14 chiều (Không chứa ear_avg để tránh trôi khi tập trung)
    feature_vector: List[float] = [
        rx_avg, ry_avg,
        gx_avg, gy_avg,
        gy_nasion,
        rx_l, ry_l,
        rx_r, ry_r,
        head_x, head_y,
        p_norm, y_norm, r_norm,
    ]

    eye_info: Dict[str, Any] = {
        "rx_l": rx_l, "ry_l": ry_l, "ear_l": ear_l, "L_l": L_l,
        "rx_r": rx_r, "ry_r": ry_r, "ear_r": ear_r, "L_r": L_r,
        "rx_avg": rx_avg, "ry_avg": ry_avg, "ear_avg": ear_avg,
        "gx_avg": gx_avg, "gy_avg": gy_avg,
        "iris_l": (int(p_l_iris[0]), int(p_l_iris[1])),
        "iris_r": (int(p_r_iris[0]), int(p_r_iris[1])),
        "is_blinking": is_blinking,
    }
    head_info: Dict[str, Any] = {
        "pitch": pitch, "yaw": yaw, "roll": roll, "direction": dir_txt,
    }

    # Vẽ viền hốc mắt và bounding box
    l_pts = np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_CONTOUR],
        dtype=np.int32,
    )
    r_pts = np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_CONTOUR],
        dtype=np.int32,
    )
    cv2.polylines(annotated, [l_pts], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(annotated, [r_pts], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    if draw_bbox:
        for pts in [l_pts, r_pts]:
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            pad_x = int((x_max - x_min) * 0.15)
            pad_y = int((y_max - y_min) * 0.25)
            box_col = (0, 0, 255) if is_blinking else (0, 255, 128)
            cv2.rectangle(
                annotated,
                (max(0, x_min - pad_x), max(0, y_min - pad_y)),
                (min(w - 1, x_max + pad_x), min(h - 1, y_max + pad_y)),
                box_col, 1, cv2.LINE_AA,
            )

    for (p_origin, p_iris, ex, ey, L) in [
        (p_l_out, p_l_iris, ex_l, ey_l, L_l),
        (p_r_in,  p_r_iris, ex_r, ey_r, L_r),
    ]:
        ox, oy = int(p_origin[0]), int(p_origin[1])
        al = int(L * 0.6)
        cv2.line(annotated, (ox, oy), (int(ox + ex[0]*al), int(oy + ex[1]*al)), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, (ox, oy), (int(ox + ey[0]*al), int(oy + ey[1]*al)), (0, 255, 0), 1, cv2.LINE_AA)
        ix, iy = int(p_iris[0]), int(p_iris[1])
        cv2.circle(annotated, (ix, iy), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, (ix, iy), 6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, (ix-4, iy), (ix+4, iy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, (ix, iy-4), (ix, iy+4), (0, 255, 255), 1, cv2.LINE_AA)

    npx = int(landmarks[NOSE_TIP_IDX].x * w)
    npy = int(landmarks[NOSE_TIP_IDX].y * h)
    al = int(min(w, h) * 0.16)
    cv2.arrowedLine(
        annotated, (npx, npy),
        (npx + int(rmat[0, 2]*al), npy + int(-rmat[1, 2]*al)),
        (0, 215, 255), 2, cv2.LINE_AA, tipLength=0.25,
    )
    cv2.circle(annotated, (npx, npy), 3, (0, 215, 255), -1, cv2.LINE_AA)

    return annotated, feature_vector, eye_info, head_info, is_blinking


def draw_hud(
    frame: np.ndarray,
    eye_info: Optional[Dict[str, Any]],
    head_info: Optional[Dict[str, Any]],
    gaze_pt: Optional[Tuple[int, int]],
    is_calibrating: bool,
    has_model: bool,
    fps: float,
    mouse_enabled: bool = False,
):
    """Vẽ bảng thông tin trực quan HUD lên frame camera."""
    panel_w, panel_h = 410, 235

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1)

    cv2.putText(frame, f"Gaze Tracker v2.4 | FPS: {int(fps)}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA)

    if is_calibrating:
        status_txt   = "DANG HIEU CHUAN 13 DIEM..."
        status_color = (0, 255, 255)
    elif has_model:
        status_txt   = "DA KET NOI (Hybrid Poly+RBF Lock)"
        status_color = (0, 255, 0)
    else:
        status_txt   = "CHUA HIEU CHUAN  [C] de bat dau"
        status_color = (0, 165, 255)

    cv2.putText(frame, f"Trang thai: {status_txt}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1, cv2.LINE_AA)
    cv2.line(frame, (15, 63), (panel_w + 5, 63), (60, 60, 60), 1)

    if eye_info:
        blink = eye_info.get("is_blinking", False)
        blink_tag = "  [CHOP MAT]" if blink else ""
        col       = (0, 80, 255) if blink else (255, 220, 100)
        cv2.putText(frame, f"Mat (L/R): rx={eye_info['rx_avg']:.3f}  ry={eye_info['ry_avg']:.3f}{blink_tag}",
                    (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.41, col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"IOD Gaze: gx={eye_info['gx_avg']:.3f}  gy={eye_info['gy_avg']:.3f}",
                    (20, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 220, 100), 1, cv2.LINE_AA)

    if head_info:
        cv2.putText(frame, f"Dau: {head_info['direction']}", (20, 123),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"P:{head_info['pitch']:+.1f}*  Y:{head_info['yaw']:+.1f}*  R:{head_info['roll']:+.1f}*",
                    (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    if gaze_pt:
        cv2.putText(frame, f"Toa do man hinh: X={gaze_pt[0]}  Y={gaze_pt[1]}",
                    (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Toa do man hinh: --",
                    (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1, cv2.LINE_AA)

    mcol = (0, 255, 0) if mouse_enabled else (160, 160, 160)
    cv2.putText(frame, f"Chuot [M]: {'BAT' if mouse_enabled else 'TAT'} | [Space]: Bat diem",
                (20, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.41, mcol, 1, cv2.LINE_AA)
    cv2.putText(frame, "[C] Hieu chuan | [F] Fullscreen | [Q] Thoat",
                (20, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def render_screen_canvas(
    screen_w: int,
    screen_h: int,
    calib_info: Optional[Dict[str, Any]],
    gaze_screen_pt: Optional[Tuple[int, int]],
    has_model: bool,
    target_marker_pt: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Tạo màn hình mô phỏng (Screen Canvas) hỗ trợ ghim điểm mục tiêu tùy chọn."""
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    for gx in range(0, screen_w, max(1, screen_w // 8)):
        cv2.line(canvas, (gx, 0), (gx, screen_h), (20, 20, 20), 1)
    for gy in range(0, screen_h, max(1, screen_h // 6)):
        cv2.line(canvas, (0, gy), (screen_w, gy), (20, 20, 20), 1)

    if target_marker_pt is not None and calib_info is None:
        tx, ty = target_marker_pt
        cv2.circle(canvas, (tx, ty), 14, (255, 100, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 4, (255, 100, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "Nhin vao day & nhan [SPACE] de bat diem",
                    (max(10, min(screen_w - 420, tx - 180)), max(35, ty - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 140, 255), 1, cv2.LINE_AA)

    if calib_info is not None:
        pt       = calib_info["point"]
        step     = calib_info["step"]
        total    = calib_info["total_steps"]
        phase    = calib_info["phase"]
        progress = calib_info["progress"]
        blinking = calib_info.get("is_blinking", False)
        lx = max(10, min(screen_w - 340, pt[0] - 170))
        ly = max(50, pt[1] - 55) if pt[1] > 100 else pt[1] + 65

        if phase == "settle":
            r = max(12, int(45 * (1.0 - progress * 0.65)))
            cv2.circle(canvas, pt, r, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"Diem {step+1}/{total}: Nhin vao tam...",
                        (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            col = (0, 0, 255) if blinking else (0, 255, 0)
            msg = "CHOP MAT — Tam dung" if blinking else f"Giu anh mat ({step+1}/{total})"
            cv2.circle(canvas, pt, 18, col, 2, cv2.LINE_AA)
            cv2.circle(canvas, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, pt, (28, 28), 0, -90, -90 + int(progress * 360), col, 3, cv2.LINE_AA)
            cv2.putText(canvas, msg, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.72, col, 2, cv2.LINE_AA)

    elif gaze_screen_pt is not None:
        gx, gy = gaze_screen_pt
        cv2.circle(canvas, (gx, gy), 20, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (gx, gy), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, (gx-30, gy), (gx+30, gy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (gx, gy-30), (gx, gy+30), (0, 255, 255), 1, cv2.LINE_AA)
        label_x = min(screen_w - 180, gx + 26)
        label_y = min(screen_h - 20,  gy + 26)
        cv2.putText(canvas, f"({gx}, {gy})", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

    elif not has_model:
        cx, cy = screen_w // 2, screen_h // 2
        cv2.putText(canvas, "CHUA CO MO HINH HIEU CHUAN",
                    (cx - 280, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Nhan phim [C] de bat dau hieu chuan 13 diem (Fullscreen)",
                    (cx - 390, cy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (190, 190, 190), 1, cv2.LINE_AA)

    return canvas