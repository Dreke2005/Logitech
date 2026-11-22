"""Record a local 25 FPS MP4 from RGB frames, with no audio stream."""

import cv2

FPS = 25


class VideoClip:
    """Write one clip per preview session, after the user starts recording."""

    def __init__(self, path):
        self.path = path
        self.recording = False
        self.frames = 0
        self._writer = None
        self._started = self._last_time = self._last_frame = None

    @property
    def status(self):
        if self.recording:
            return f"Recording video only | {self.frames / FPS:.1f}s | Space to stop"
        if self.frames:
            return "Clip saved. Press Q or Esc to close."
        return "Press Space to start recording video only."

    def toggle(self):
        if self.recording:
            self.finish()
        elif not self.frames:
            self.recording = True

    def add_frame(self, frame, timestamp):
        if not self.recording:
            return
        if self._writer is None:
            height, width = frame.shape[:2]
            # Reserve the name so we only ever replace our own empty file.
            with self.path.open("xb"):
                pass
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     FPS, (width, height))
            if not writer.isOpened():
                writer.release()
                self.path.unlink(missing_ok=True)
                raise OSError(f"Could not encode a video clip at {self.path}.")
            self._writer = writer
            self._started = timestamp

        if self._last_time is not None and timestamp - self._last_time > 0.25:
            # Resume at the next output frame instead of encoding a long freeze.
            self._started = timestamp - self.frames / FPS
        index = round((timestamp - self._started) * FPS)
        if index >= self.frames:
            while self.frames < index:
                self._writer.write(self._last_frame)
                self.frames += 1
            self._writer.write(frame)
            self.frames += 1
        self._last_frame = frame
        self._last_time = timestamp

    def finish(self):
        self.recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"Saved video-only clip to {self.path}", flush=True)
