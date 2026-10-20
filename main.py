import argparse
import sys

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from modules.tracker import GazeTracker, run_tracker
from modules.mapper import GazeScreenMapper
from modules.geometry import compute_normalized_eye_coords, get_head_pose_from_matrix
from modules.visualizer import draw_visuals, draw_hud, render_screen_canvas
from modules.utils import get_screen_resolution, ensure_model_exists


def run(source=0, control_mouse=False):
    """Hàm chạy tương thích ngược."""
    run_tracker(source=source, control_mouse=control_mouse)


def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe Eye Gaze Tracking: Chuẩn hóa 2 mắt & Hồi quy Ridge lên màn hình"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Nguồn video: '0' (webcam mặc định) hoặc đường dẫn file",
    )
    parser.add_argument(
        "--control-mouse",
        action="store_true",
        help="Kích hoạt điều khiển con trỏ chuột theo ánh mắt",
    )
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_tracker(source=src, control_mouse=args.control_mouse)


if __name__ == "__main__":
    main()
