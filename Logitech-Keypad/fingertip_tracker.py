"""MediaPipe Tasks hand tracking in the coordinates of the displayed RGB frame."""

import shutil
import time
import urllib.request
from pathlib import Path

import cv2

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def ensure_model():
    """Cache the versioned official model; never leave a partial final file."""
    if not DEFAULT_MODEL.is_file():
        DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        partial = DEFAULT_MODEL.with_suffix(".download")
        print("Downloading the hand model from Google (first run only) ...", flush=True)
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=30) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            partial.replace(DEFAULT_MODEL)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError("Hand model download failed. Check your connection or supply --model PATH.") from exc
    return DEFAULT_MODEL


def extract_hands(result, width, height, timestamp_ms):
    """JSON-ready landmarks for this frame; hand order is not a persistent ID.

    x/y are normalized image coordinates; z is model-estimated relative depth,
    not distance to the desk. Pixel coordinates share the preview's orientation.
    """
    hands = []
    for index, landmarks in enumerate(result.hand_landmarks):
        category = result.handedness[index][0]
        points = [{"x": float(p.x), "y": float(p.y), "z": float(p.z),
                   "pixel": [round(p.x * width), round(p.y * height)]} for p in landmarks]
        hands.append({
            "handedness": category.category_name,
            "score": float(category.score),
            "landmarks": points,
            "fingertips": {name: points[landmark_id] for name, landmark_id in FINGERTIPS.items()},
        })
    return {"timestamp_ms": timestamp_ms, "frame_size": [width, height], "hands": hands}


class FingertipTracker:
    def __init__(self, model_path, max_hands=2):
        import mediapipe as mp

        self.mp = mp
        self.last_timestamp = -1
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        # Strictly increasing host timestamps, independent of the camera's AI clock.
        timestamp = max(self.last_timestamp + 1, time.monotonic_ns() // 1_000_000)
        self.last_timestamp = timestamp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, timestamp)
        height, width = frame.shape[:2]
        return extract_hands(result, width, height, timestamp)

    def close(self):
        self.detector.close()


def draw_hands(frame, data, *, show_status=True, show_labels=True):
    """Draw skeletons and named fingertips; do not retain lost detections."""
    for hand in data["hands"]:
        points = hand["landmarks"]
        for start, end in CONNECTIONS:
            cv2.line(frame, tuple(points[start]["pixel"]), tuple(points[end]["pixel"]),
                     (80, 200, 80), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, tuple(point["pixel"]), 3, (255, 255, 255), -1)
        for name, point in hand["fingertips"].items():
            x, y = point["pixel"]
            color = (0, 255, 255) if name == "index" else (255, 120, 255)
            cv2.circle(frame, (x, y), 9 if name == "index" else 6, color, -1)
            if show_labels:
                cv2.putText(frame, name, (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1, cv2.LINE_AA)
    if not show_status:
        return
    cv2.putText(frame, f"Hands: {len(data['hands'])} - show your fingers", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    for index, hand in enumerate(data["hands"]):
        x, y = hand["fingertips"]["index"]["pixel"]
        text = f"{hand['handedness']} index: ({x}, {y})"
        cv2.putText(frame, text, (12, frame.shape[0] - 16 - index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
