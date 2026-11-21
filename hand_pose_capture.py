"""Capture arm, hand, and finger landmarks from camera 0 with MediaPipe Tasks.

Run with ``uv run python hand_pose_capture.py``. The first run downloads the
official Holistic Landmarker model into this folder. Press q or ESC to quit.
"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, Tuple
from urllib.request import urlretrieve

import cv2

from camera_utils import open_camera

FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
POSE_IDS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
}
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def _point(landmark: Any, width: int, height: int) -> Dict[str, Any]:
    x, y = float(landmark.x), float(landmark.y)
    point = {
        "normalized": {"x": x, "y": y, "z": float(landmark.z)},
        "pixel": {"x": round(x * width), "y": round(y * height)},
    }
    visibility = getattr(landmark, "visibility", None)
    if visibility is not None:
        point["visibility"] = float(visibility)
    return point


def extract_body_points(results: Any, frame_size: Tuple[int, int]) -> Dict[str, Any]:
    """Return detected shoulder/elbow/wrist joints and all finger landmarks."""
    width, height = frame_size
    pose = {}
    pose_landmarks = results.pose_landmarks
    if pose_landmarks:
        for name, idx in POSE_IDS.items():
            landmark = pose_landmarks[idx]
            visibility = getattr(landmark, "visibility", None)
            if visibility is None or visibility >= 0.35:
                pose[name] = _point(landmark, width, height)

    hands = {}
    for side in ("left", "right"):
        landmarks = getattr(results, f"{side}_hand_landmarks")
        if not landmarks:
            hands[side] = None
            continue
        fingers = {}
        for finger, ids in FINGER_LANDMARKS.items():
            joints = [_point(landmarks[idx], width, height) for idx in ids]
            fingers[finger] = {"joints": joints, "tip": joints[-1]}
        hands[side] = {
            "wrist": _point(landmarks[0], width, height),
            "fingers": fingers,
            "landmarks": [_point(p, width, height) for p in landmarks],
        }
    return {"arms": pose, "hands": hands}


def _draw_chain(frame: Any, landmarks: Any, connections: Any, color: Tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    for start, end in connections:
        a, b = landmarks[start], landmarks[end]
        p1 = (round(a.x * width), round(a.y * height))
        p2 = (round(b.x * width), round(b.y * height))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for landmark in landmarks:
        p = (round(landmark.x * width), round(landmark.y * height))
        cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)


def _ensure_model() -> Path:
    model_path = Path(__file__).with_name("holistic_landmarker.task")
    if not model_path.exists():
        url = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
               "holistic_landmarker/float16/latest/holistic_landmarker.task")
        print("Downloading the MediaPipe Holistic model (one-time setup)...")
        try:
            urlretrieve(url, model_path)
        except Exception as exc:
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download holistic_landmarker.task. Download it from "
                f"{url} and place it beside hand_pose_capture.py."
            ) from exc
    return model_path


def run(camera_index: int = 0, min_detection_confidence: float = 0.55) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe missing. Add it with: uv add mediapipe") from exc

    model_path = _ensure_model()
    vision = mp.tasks.vision
    options = vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=min_detection_confidence,
        min_hand_landmarks_confidence=min_detection_confidence,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    capture = None
    try:
        capture, info = open_camera(camera_index, fourcc=None)
        print(f"Camera {camera_index}: {info['width']}x{info['height']} [{info['backend']}]")
        print("Show your upper body and hands to the camera. Press q or ESC to quit.")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.")
                break
            # Mirror horizontally (reflect across the image's vertical axis)
            # before inference so displayed pixels and landmark coordinates match.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = time.monotonic_ns() // 1_000_000
            results = landmarker.detect_for_video(image, timestamp_ms)
            points = extract_body_points(results, (frame.shape[1], frame.shape[0]))

            if results.pose_landmarks:
                pose_edges = ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12))
                _draw_chain(frame, results.pose_landmarks, pose_edges, (0, 220, 0))
            for side in ("left", "right"):
                hand = getattr(results, f"{side}_hand_landmarks")
                if hand:
                    _draw_chain(frame, hand, HAND_CONNECTIONS, (0, 180, 255))

            left, right = points["hands"]["left"], points["hands"]["right"]
            cv2.putText(frame, f"Hands: L={left is not None} R={right is not None} | "
                        f"arm joints: {len(points['arms'])}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 230, 40), 2)
            if left:
                tip = left["fingers"]["index"]["tip"]["pixel"]
                cv2.putText(frame, f"Left index tip: ({tip['x']}, {tip['y']})",
                            (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 230, 40), 2)
            cv2.imshow("Camera 0 - hands, fingers and arms", frame)
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
                        help="minimum detection confidence")
    args = parser.parse_args()
    run(args.camera, args.confidence)


if __name__ == "__main__":
    main()


