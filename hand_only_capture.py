"""Track and display hands and finger landmarks only (no face or arms).

Run with ``uv run python hand_only_capture.py``. Press Q or ESC to quit.
The first run downloads MediaPipe's Hand Landmarker model into models/.
"""

# Author: Victor Lancieux
# Date: 2026-11-22 (hand-only capture file created)
# Purpose: Capture and visualize two hands and their 21 landmarks.

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List
from urllib.request import urlretrieve

import cv2

from camera_utils import open_camera

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def ensure_hand_model() -> Path:
    model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        print("Downloading the MediaPipe Hand Landmarker model (one-time setup)...")
        try:
            urlretrieve(url, model_path)
        except Exception as exc:
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download hand_landmarker.task. Download it from "
                f"{url} and place it at models/hand_landmarker.task."
            ) from exc
    return model_path


def extract_hands(result: Any, width: int, height: int) -> List[Dict[str, Any]]:
    """Return each detected hand's label, confidence and 21 pixel landmarks."""
    hands = []
    for index, landmarks in enumerate(result.hand_landmarks):
        category = result.handedness[index][0] if index < len(result.handedness) else None
        label = category.category_name if category else f"Hand {index + 1}"
        confidence = float(category.score) if category else 0.0
        points = [
            {
                "x": round(float(point.x) * width),
                "y": round(float(point.y) * height),
                "z": float(point.z),
            }
            for point in landmarks
        ]
        hands.append({"side": label, "confidence": confidence, "landmarks": points})
    return hands


def draw_hand(frame: Any, landmarks: Any, color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]

    def pixel(index: int) -> tuple[int, int]:
        point = landmarks[index]
        return round(point.x * width), round(point.y * height)

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pixel(start), pixel(end), color, 2, cv2.LINE_AA)
    for index in range(len(landmarks)):
        cv2.circle(frame, pixel(index), 3, color, -1, cv2.LINE_AA)
    for finger, tip_index in FINGER_TIPS.items():
        x, y = pixel(tip_index)
        cv2.circle(frame, (x, y), 6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, finger, (x + 6, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 255, 255), 1, cv2.LINE_AA)


def run(camera_index: int = 0, confidence: float = 0.55) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe missing. Add it with: uv add mediapipe") from exc

    model_path = ensure_hand_model()
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    capture = None
    try:
        capture, info = open_camera(camera_index, fourcc=None)
        print(f"Camera {camera_index}: {info['width']}x{info['height']} [{info['backend']}]")
        print("Hand-only tracking: show one or both hands. Press Q or ESC to quit.")
        window_name = "Hand-only tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.")
                break
            # Mirror before inference so labels and displayed coordinates match the preview.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = landmarker.detect_for_video(image, timestamp_ms)
            hands = extract_hands(result, frame.shape[1], frame.shape[0])

            for index, (hand_result, hand_data) in enumerate(zip(result.hand_landmarks, hands)):
                color = (0, 220, 0) if hand_data["side"] == "Left" else (0, 165, 255)
                draw_hand(frame, hand_result, color)
                wrist = hand_data["landmarks"][0]
                cv2.putText(frame,
                            f"{hand_data['side']} hand {hand_data['confidence']:.2f} "
                            f"wrist=({wrist['x']},{wrist['y']})",
                            (12, 28 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, color, 2, cv2.LINE_AA)
            if not hands:
                cv2.putText(frame, "No hands detected", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        landmarker.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="minimum hand detection/tracking confidence")
    args = parser.parse_args()
    run(args.camera, args.confidence)


if __name__ == "__main__":
    main()
