"""Live keyboard calibration and fingertip-to-key mapping test.

Run ``uv run python test_key_mapping.py``. The default camera is camera 1.
Automatic detection starts immediately. Press Enter to confirm the outlined
keyboard, or C to set the four corners manually in this order: top-left,
top-right, bottom-right, bottom-left. The app then maps index fingertips.

Controls: Enter confirm, A rescan, C manual corner selection, R reset,
S save snapshot, Q or ESC quit.
"""

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from camera_utils import open_camera
from hand_pose_capture import HAND_CONNECTIONS, _draw_chain, _ensure_model
from keyboard_tracker import (
    KeyboardDetection,
    detect_keyboard,
    draw_keyboard_view,
    finger_pose_to_key,
)

OUTPUT_SIZE = (1000, 284)


def _make_detection(frame: np.ndarray, corners: np.ndarray) -> KeyboardDetection:
    """Build a rectification transform from manually selected keyboard corners."""
    width, height = OUTPUT_SIZE
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    transform = cv2.getPerspectiveTransform(corners, destination)
    inverse = cv2.getPerspectiveTransform(destination, corners)
    warped = cv2.warpPerspective(frame, transform, OUTPUT_SIZE)
    return KeyboardDetection(corners, warped, transform, inverse, 1.0)


def run(camera_index: int = 1, confidence: float = 0.55) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe missing. Add it with: uv add mediapipe") from exc

    model_path = _ensure_model()
    vision = mp.tasks.vision
    options = vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=confidence,
        min_hand_landmarks_confidence=confidence,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    capture = None
    try:
        capture, info = open_camera(camera_index, fourcc=None)
        print(f"Camera {camera_index}: {info['width']}x{info['height']} [{info['backend']}]")
        print("Automatic search is active. Enter confirms the outlined keyboard; C starts manual corner selection.")
        print("Keep the complete keyboard in view. A rescans; S saves a snapshot; Q quits.")
        camera_window = "Key mapping - camera view"
        keyboard_window = "Key mapping - rectified keyboard"
        cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(keyboard_window, cv2.WINDOW_NORMAL)

        state = {
            "frame": None,
            "points": [],
            "calibration": None,
            "selecting": False,
            "automatic": True,
            "auto_detection": None,
        }

        def on_mouse(event, x, y, _flags, _userdata):
            if event != cv2.EVENT_LBUTTONDOWN or not state["selecting"]:
                return
            frame = state["frame"]
            if frame is None:
                return
            state["points"].append((x, y))
            if len(state["points"]) == 4:
                state["calibration"] = np.asarray(state["points"], dtype=np.float32)
                state["selecting"] = False
                print("Keyboard corners selected; mapping is calibrated.")

        cv2.setMouseCallback(camera_window, on_mouse)
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.")
                break
            # Mirror at the input boundary so detection, manual corner clicks,
            # hand landmarks, fingertip mapping, and previews share coordinates.
            frame = cv2.flip(frame, 1)
            state["frame"] = frame.copy()
            frame_number += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, time.monotonic_ns() // 1_000_000)

            if state["automatic"] and (frame_number % 5 == 1):
                state["auto_detection"] = detect_keyboard(frame)
            if state["calibration"] is not None:
                detection = _make_detection(frame, state["calibration"])
            else:
                detection = None
            candidate = state.get("auto_detection") if state["automatic"] else None
            preview_detection = detection
            if preview_detection is None and candidate is not None:
                preview_detection = _make_detection(frame, candidate.corners)

            view = frame.copy()
            if result.pose_landmarks:
                _draw_chain(view, result.pose_landmarks,
                            ((11, 13), (13, 15), (12, 14), (14, 16)), (0, 220, 0))
            mapped = []
            for side in ("left", "right"):
                hand = getattr(result, f"{side}_hand_landmarks")
                if not hand:
                    continue
                _draw_chain(view, hand, HAND_CONNECTIONS, (0, 180, 255))
                tip = hand[8]  # index fingertip
                x, y = tip.x * frame.shape[1], tip.y * frame.shape[0]
                label = finger_pose_to_key((x, y), detection) if detection else None
                mapped.append((side, (x, y), label))

            status = ""
            if preview_detection is not None:
                outline_color = (40, 220, 40) if detection is not None else (0, 190, 255)
                cv2.polylines(view, [preview_detection.corners.astype(np.int32).reshape(-1, 1, 2)],
                              True, outline_color, 2)
                for i, (x, y) in enumerate(preview_detection.corners.astype(int)):
                    cv2.circle(view, (x, y), 5, outline_color, -1)
                    cv2.putText(view, str(i + 1), (x + 6, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, outline_color, 2)
            if detection is not None:
                for side, (x, y), label in mapped:
                    point = (round(x), round(y))
                    cv2.circle(view, point, 8, (0, 80, 255), -1)
                    text = f"{side} index: {label}" if label else f"{side} index: no mapped key"
                    cv2.putText(view, text, (point[0] + 9, point[1] - 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 80, 255), 2)
                selected = next((label for _side, _xy, label in mapped if label), None)
                cv2.imshow(keyboard_window, draw_keyboard_view(detection, selected))
                status = " | ".join(f"{side} index -> {label or 'no key'}"
                                    for side, _xy, label in mapped) or "No hand detected"
            elif candidate is not None:
                cv2.imshow(keyboard_window, preview_detection.warped)
                status = "Keyboard candidate: press Enter to confirm, C to adjust corners, A to rescan"
            else:
                status = ("Searching for keyboard... keep all four corners visible; "
                          "press C for manual corner selection")

            for i, (x, y) in enumerate(state["points"]):
                cv2.circle(view, (x, y), 7, (255, 0, 255), -1)
                cv2.putText(view, str(i + 1), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(view, status, (10, view.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 40), 2)
            if state["selecting"]:
                cv2.putText(view, f"Click corner {len(state['points']) + 1}/4",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.imshow(camera_window, view)

            pressed = cv2.waitKey(1) & 0xFF
            if pressed == ord("c"):
                state["points"] = []
                state["calibration"] = None
                state["selecting"] = True
                state["automatic"] = False
                print("Click the keyboard corners in order: TL, TR, BR, BL.")
            elif pressed in (13, 10, ord(" ")) and candidate is not None:
                state["calibration"] = candidate.corners.copy()
                state["automatic"] = False
                state["selecting"] = False
                print("Keyboard candidate confirmed; key mapping is active.")
            elif pressed == ord("r"):
                state["points"] = []
                state["calibration"] = None
                state["selecting"] = False
                state["auto_detection"] = None
                state["automatic"] = True
                print("Calibration reset; automatic keyboard search restarted.")
            elif pressed == ord("a"):
                state["automatic"] = True
                state["calibration"] = None
                state["selecting"] = False
                state["auto_detection"] = None
                print("Rescanning for a keyboard candidate.")
            elif pressed == ord("s") and detection is not None:
                snapshot_dir = Path(__file__).with_name("captures")
                snapshot_dir.mkdir(exist_ok=True)
                snapshot_path = snapshot_dir / (
                    "key_mapping_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
                )
                cv2.imwrite(str(snapshot_path), view)
                print(f"Saved annotated snapshot: {snapshot_path}")
            elif pressed in (ord("q"), 27):
                break
    finally:
        landmarker.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1,
                        help="camera index that sees keyboard and hands (default: 1)")
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="MediaPipe detection confidence")
    args = parser.parse_args()
    run(args.camera, args.confidence)


if __name__ == "__main__":
    main()
