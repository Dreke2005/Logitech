"""Worker for the visual-only Auto-AVSR inference tutorial."""

import json
import math
import socket
import struct
import sys
from argparse import Namespace
from pathlib import Path

TARGET_FPS = 25
MAX_SECONDS = 15
MAX_EDGE = 720
MIN_FRAMES = math.ceil(TARGET_FPS / 2)
MAX_LIVE_FRAMES = 10 * TARGET_FPS
FRAME_HEADER = struct.Struct("!III")


def read_rgb_clip(path):
    """Decode only RGB frames and resample a constant-rate clip to 25 fps."""
    import cv2

    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot read this video: {path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or not 1 <= source_fps <= 240:
            raise ValueError("No valid frame rate. Record or convert the clip to a constant frame rate.")
        frame_index = 0
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index / source_fps >= MAX_SECONDS:
                raise ValueError(f"Clip is longer than {MAX_SECONDS} seconds. Trim it to one phrase.")
            expected_index = math.floor(len(frames) * source_fps / TARGET_FPS + 1e-7)
            if expected_index == frame_index:
                scale = min(1, MAX_EDGE / max(frame.shape[:2]))
                if scale < 1:
                    size = (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale)))
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                while math.floor(len(frames) * source_fps / TARGET_FPS + 1e-7) == frame_index:
                    frames.append(rgb)
            frame_index += 1
        if len(frames) < MIN_FRAMES:
            raise ValueError("The clip needs at least half a second of readable video.")
        return frames
    finally:
        capture.release()


class VisualRecognizer:
    def __init__(self, repo, checkpoint, device):
        sys.path.insert(0, str(repo))
        import torch
        from datamodule.transforms import VideoTransform
        from lightning import ModelModule
        from preparation.detectors.mediapipe.detector import LandmarksDetector
        from preparation.detectors.mediapipe.video_process import VideoProcess

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use --device cpu or install CUDA PyTorch.")
        if device == "cpu":
            # Leave CPU capacity for the preview while the model is decoding a phrase.
            torch.set_num_threads(min(4, torch.get_num_threads()))
        model = ModelModule(Namespace(modality="video"))
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.model.load_state_dict(weights)
        self.model = model.eval().to(device)
        self.torch = torch
        self.device = device
        self.transform = VideoTransform(subset="test")
        self.crop_mouth = VideoProcess(convert_gray=False)
        self.detector = LandmarksDetector()

    def read(self, frames):
        try:
            landmarks = self.detector(frames)
        except AssertionError as error:
            raise ValueError("Could not track a face. Keep your face and mouth visible.") from error
        if sum(points is not None for points in landmarks) < len(frames) / 2:
            raise ValueError("Could not track a face in enough frames. Keep your face and mouth visible.")
        patches = self.crop_mouth(frames, landmarks)
        video = self.transform(self.torch.from_numpy(patches).permute(0, 3, 1, 2))
        with self.torch.inference_mode():
            return self.model(video.to(self.device))

    def close(self):
        self.detector.short_range_detector.close()
        self.detector.full_range_detector.close()


def recognize(repo, clip, checkpoint, device):
    frames = read_rgb_clip(clip)
    recognizer = VisualRecognizer(repo, checkpoint, device)
    try:
        return recognizer.read(frames)
    finally:
        recognizer.close()


def _read_live_frames(stream, width, height, count):
    import cv2
    import numpy as np

    if not (1 <= width <= MAX_EDGE and 1 <= height <= MAX_EDGE and MIN_FRAMES <= count <= MAX_LIVE_FRAMES):
        raise ValueError("Invalid live phrase size.")
    frame_size = width * height * 3
    frames = []
    for _ in range(count):
        data = stream.read(frame_size)
        if len(data) != frame_size:
            raise EOFError("Incomplete live phrase.")
        bgr = np.frombuffer(data, np.uint8).reshape(height, width, 3)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def _send_event(connection, event, text=""):
    message = json.dumps({"event": event, "text": text}, ensure_ascii=False).encode("utf-8")
    connection.sendall(message + b"\n")


def _serve(recognizer, connection):
    _send_event(connection, "ready")
    with connection.makefile("rb") as stream:
        while header := stream.read(FRAME_HEADER.size):
            frames = _read_live_frames(stream, *FRAME_HEADER.unpack(header))
            try:
                text = (recognizer.read(frames) or "").strip()
                if text:
                    _send_event(connection, "transcript", text)
                else:
                    _send_event(connection, "retry", "No words recognized. Try another short phrase.")
            except (AssertionError, OverflowError, ValueError) as error:
                _send_event(connection, "retry", str(error))
            finally:
                del frames


def serve_live(repo, checkpoint, device, port, token):
    with socket.create_connection(("127.0.0.1", port)) as connection:
        connection.sendall(token)
        try:
            recognizer = VisualRecognizer(repo, checkpoint, device)
            try:
                _serve(recognizer, connection)
            finally:
                recognizer.close()
        except Exception as error:
            _send_event(connection, "error", str(error)[:2000] or type(error).__name__)


if __name__ == "__main__":
    if sys.argv[1] == "--serve":
        _, repo_arg, checkpoint_arg, device_arg, port_arg, token_arg = sys.argv[1:]
        serve_live(Path(repo_arg), Path(checkpoint_arg), device_arg, int(port_arg), bytes.fromhex(token_arg))
    else:
        repo_arg, clip_arg, checkpoint_arg, device_arg, result_arg = sys.argv[1:]
        text = recognize(Path(repo_arg), Path(clip_arg), Path(checkpoint_arg), device_arg)
        Path(result_arg).write_text(text or "", encoding="utf-8")
