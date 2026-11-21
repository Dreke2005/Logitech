"""Capture arm, hand, and finger landmarks from camera 0.

Run with ``python hand_pose_capture.py``. The live preview draws the detected
body pose and hand skeletons. Press q or ESC to quit. Install MediaPipe first
if needed: ``pip install mediapipe``.

``extract_body_points`` returns pixel and normalized coordinates so the index
finger tip (or another landmark) can be passed to ``finger_pose_to_key`` in
keyboard_tracker.py after keyboard detection has been added to the frame loop.
"""

import argparse
from typing import Any, Dict, Tuple

import cv2

from camera_utils import open_camera


# MediaPipe Hands landmark ids, grouped by finger from palm to tip.
FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

POSE_IDS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


def _point(landmark: Any, width: int, height: int) -> Dict[str, Any]:
    """Represent a MediaPipe landmark in normalized and image pixel space."""
    x, y = float(landmark.x), float(landmark.y)
    point = {
        "normalized": {"x": x, "y": y, "z": float(landmark.z)},
        "pixel": {"x": round(x * width), "y": round(y * height)},
    }
    if hasattr(landmark, "visibility"):
        point["visibility"] = float(landmark.visibility)
    return point


def extract_body_points(results: Any, frame_size: Tuple[int, int]) -> Dict[str, Any]:
    """Return arm joints and all 21 landmarks per detected hand.

    ``frame_size`` is ``(width, height)``. Each finger has its base-to-tip
    joints, plus a direct ``tip`` field. ``None`` means that side was not
    detected in this frame.
    """
    width, height = frame_size
    pose = {}
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for name, idx in POSE_IDS.items():
            joint = landmarks[idx]
            if joint.visibility >= 0.35:
                pose[name] = _point(joint, width, height)

    hands = {}
    for side in ("left", "right"):
        hand_result = getattr(results, f"{side}_hand_landmarks")
        if hand_result is None:
            hands[side] = None
            continue
        landmarks = hand_result.landmark
        fingers = {}
        for finger, ids in FINGER_LANDMARKS.items():
            joints = [_point(landmarks[idx], width, height) for idx in ids]
            fingers[finger] = {"joints": joints, "tip": joints[-1]}
        hands[side] = {
            "wrist": _point(landmarks[0], width, height),
            "fingers": fingers,
            "landmarks": [_point(landmark, width, height) for landmark in landmarks],
        }
    return {"arms": pose, "hands": hands}


def _draw_results(frame: Any, results: Any, mp: Any) -> None:
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    if results.pose_landmarks:
        drawing.draw_landmarks(
            frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
        )
    for hand_landmarks in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand_landmarks:
            drawing.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                styles.get_default_hand_landmarks_style(),
                styles.get_default_hand_connections_style(),
            )


def run(camera_index: int = 0, min_detection_confidence: float = 0.55) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for hand and body tracking. Install it with: "
            "pip install mediapipe"
        ) from exc

    capture, info = open_camera(camera_index, fourcc=None)
    print(f"Camera {camera_index}: {info['width']}x{info['height']} [{info['backend']}]")
    print("Show your upper body and hands to the camera. Press q or ESC to quit.")
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            _draw_results(frame, results, mp)
            points = extract_body_points(results, (frame.shape[1], frame.shape[0]))
            left = points["hands"]["left"] is not None
            right = points["hands"]["right"] is not None
            arm_count = len(points["arms"])
            cv2.putText(frame, f"Hands: L={left} R={right} | arm joints: {arm_count}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 230, 40), 2)

            # The returned pixel position is the input expected by
            # finger_pose_to_key((x, y), keyboard_detection) in the keyboard module.
            left_hand = points["hands"]["left"]
            if left_hand:
                tip = left_hand["fingers"]["index"]["tip"]["pixel"]
                cv2.putText(frame, f"Left index tip: ({tip['x']}, {tip['y']})",
                            (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 230, 40), 2)

            cv2.imshow("Camera 0 - hands, fingers and arms", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        holistic.close()
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="minimum MediaPipe detection confidence")
    args = parser.parse_args()
    run(args.camera, args.confidence)


if __name__ == "__main__":
    main()
