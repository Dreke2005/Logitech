"""Use eye aspect ratio (EAR) to detect unilateral winks for mouse clicks."""

# Author: Victor Lancieux
# Date: 2026-11-22 (EAR-based wink clicks added)
# Purpose: Left/right eye clicks on camera 0; ignore bilateral eye closure.


class WinkClickController:
    """Map a left wink to hold/release and a right wink to one right click."""

    def __init__(self, output, closed_threshold=0.16, open_threshold=0.20,
                 hold_frames=3):
        self.output = output
        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold
        self.hold_frames = hold_frames
        self.reset(armed=False)

    def reset(self, armed=False):
        if getattr(self, "held_eye", None) == "left":
            self.output.button_up("left")
        self.held_eye = None
        self.armed = armed
        self.candidate = None
        self.closed_frames = 0
        self.both_closed = False

    def update(self, sample, enabled=True):
        """Click left/right for a wink; require both eyes to reopen before rearming."""
        ear = sample.get("ear") if sample else None
        if not enabled or not ear or len(ear) != 2:
            self.reset(armed=False)
            return None

        left_ear, right_ear = map(float, ear)
        left_closed = left_ear < self.closed_threshold
        right_closed = right_ear < self.closed_threshold
        released_left = False

        # Keep a left click held only for the duration of the left-eye close.
        if self.held_eye == "left" and left_ear > self.open_threshold:
            self.output.button_up("left")
            self.held_eye = None
            released_left = True

        if left_closed and right_closed:
            first_frame = not self.both_closed
            self.both_closed = True
            self.armed = False
            self.candidate = None
            self.closed_frames = 0
            return "Both eyes closed; ignored" if first_frame else None

        if left_ear > self.open_threshold and right_ear > self.open_threshold:
            self.armed = True
            self.both_closed = False
            self.candidate = None
            self.closed_frames = 0
            return "Left eye open: left button released" if released_left else None

        if not self.armed:
            return None

        side = None
        if left_closed and right_ear > self.open_threshold:
            side = "left"
        elif right_closed and left_ear > self.open_threshold:
            side = "right"

        if side is None:
            self.candidate = None
            self.closed_frames = 0
            return "Left eye open: left button released" if released_left else None

        if side == self.candidate:
            self.closed_frames += 1
        else:
            self.candidate = side
            self.closed_frames = 1
        if self.closed_frames < self.hold_frames:
            return "Left eye open: left button released" if released_left else None

        if side == "left":
            self.output.button_down("left")
            self.held_eye = "left"
        else:
            self.output.click(button="right")
        self.armed = False
        self.candidate = None
        self.closed_frames = 0
        return ("Left eye closed: left button held" if side == "left"
                else "Right eye wink: right click")
