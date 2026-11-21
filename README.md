# LogiInsight RGB preview

## RGB preview with a tap keyboard (Command Prompt)

From `C:\Users\andre\git\Logitech`:

```cmd
uv sync
uv run rgb_preview.py
```

The preview automatically selects `LVNext Camera` and runs MediaPipe Hand
Landmarker on the PC. Up to two hands are drawn with 21 landmarks each and five
named fingertips; the index fingertip is highlighted in yellow. Show your hands
in the camera's view. Press **q**, **Escape**, or close the window to stop.
`uv run test_cam2.py` opens the same preview. No SSH credentials are needed.

Keep both hands visible, bring your thumbs close together, and relax your
fingers as if you were about to type. You can choose where to hold your hands
and rotate them within the image. Hold still for about two-thirds of a second.
The keyboard uses the midpoint between your thumbs as its reference, follows
the angle of your hands, and fits the AZERTY home row to your resting fingers.
Space is centered at the thumb midpoint. The layout stays fixed when your
thumbs separate or your hands shift slightly. Press **R** in the preview to
choose a new position.

Move a fingertip over a key, dip it down toward the desk, then lift it while
keeping the palm steady. A completed tap enters one key in the text area below
the video. Thumbs can tap Space. Brief tracking gaps pause input and cancel
unfinished taps. Return your hands to the same area to resume on the existing
layout. After a sustained departure, bring your thumbs together again to set
a new position. Use `--no-keyboard` for the fingertip overlay.

The preview uses Logitech's `camera_utils.open_camera()` to open the RGB stream,
then reads frames with OpenCV. MediaPipe extracts hand landmarks on the PC. The
Logitech SFTP JSON provides face results, so this interaction uses the RGB path
and requires no camera network credentials. Tap detection estimates movement
from a single camera view, using finger flexion and model-estimated depth
relative to the knuckle. Keep both thumbs and fingertips visible.

On first run, the versioned model downloads from Google's MediaPipe model
storage into `models/hand_landmarker.task` (about 8 MB, excluded from Git).
Later runs work offline. Use `--model PATH` for an existing local model.
MediaPipe's model documentation: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python

```cmd
uv run rgb_preview.py --list
uv run rgb_preview.py --mirror
uv run rgb_preview.py --no-hands
uv run rgb_preview.py --landmarks-jsonl fingertips.jsonl
```

`--mirror` flips the frame before inference so the drawn and exported coordinates
match the displayed image. The JSONL option records metadata only when requested
and overwrites the named output file. Each line contains a host monotonic
`timestamp_ms`, `frame_size`, and `hands`. Each hand has 21 `landmarks` and five
`fingertips`, including `hands[i]["fingertips"]["index"]["pixel"]` for keyboard
hit testing. `x` and `y` are normalized image coordinates; `pixel` contains
`[x, y]` in displayed frame pixels. `z` is estimated relative depth, not desk
contact. Empty `hands` clears detections; list order is not a stable hand ID.
Handedness is the model's prediction on the input image and can change with mirroring.

Tracking runs inside the isolated capture process so the UI can stop it if a
camera driver stalls. The root environment uses `opencv-contrib-python`, the
GUI OpenCV build required by MediaPipe; avoid installing another OpenCV build
alongside it.

Checks:

```cmd
uv run python -m unittest discover -s tests -v
uv run rgb_preview.py --frames 30 --no-display
```

## Windows touchpad mode

```cmd
uv run rgb_preview.py --mode touchpad --mirror
```

Keep one hand in view. Press **Ctrl+Alt+F8** to enable desktop input, then focus
the app you want to control. Use the same shortcut to pause from any app. Lower
your hand when you want to reposition it without moving the pointer.

| Gesture | Action |
| --- | --- |
| Raise just the index finger and move it | Move the pointer |
| Briefly pinch thumb and index, then release | Left click. Repeat quickly for a double-click |
| Hold the thumb–index pinch, move, then release | Drag and drop |
| With ring and pinky folded, touch thumb to middle finger and release | Right click |
| Move index and middle fingers together with ring and pinky folded | Scroll vertically or horizontally |
| Extend middle, ring, and pinky. Pinch or spread thumb and index | Zoom out or in in apps that support Ctrl+wheel |
| Extend index, middle, and ring with pinky folded. Swipe left or right | Previous or next tab in apps that support Ctrl+Tab |
| Swipe up or down with the same three fingers | Windows Task View or Show Desktop |

Pause, pose changes, and lost tracking release an active drag. Press **Q** or
**Esc** in the preview to close it.

## Windows gaze pointer and wink click

```cmd
uv run rgb_preview.py --mode eye
```

The first run downloads Google's Face Landmarker model. Press **Ctrl+Alt+F8**
to enable it, then **C** with the preview focused. Look at each of the five
cursor positions on your primary monitor and press **Space** when your gaze is
steady. Once calibrated, look at a target and hold a deliberate one-eye wink
or strong one-eye squint for roughly 0.3 to 1.2 seconds. Reopen the eye to click.
Keep your head close to its calibration position. Press **C** to recalibrate.

The gaze estimate is experimental and depends on camera placement, lighting,
and how clearly both eyes are visible. Both desktop modes share the same pause
shortcut and use the Logitech RGB stream with local MediaPipe landmarks.

## SDK face AI preview

Live OpenCV RGB video with the camera's face boxes and optional 478-point
landmarks, fetched from `/files/logi_conf/logiwebcam_ai_output.json` over SFTP.
SFTP runs on a background thread so slow AI reads do not block the video loop.
The preview hides overlays after two seconds without a new camera timestamp
and shows waiting, inactive, or stale status in the window.

## Run on Windows (PowerShell)

From this repository:

```powershell
py -3 -m venv logiinsight_api/.venv
& ./logiinsight_api/.venv/Scripts/python.exe -m pip install opencv-python numpy paramiko

$env:LOGIWEBCAM_USER = Read-Host 'User name from kit card'
$env:LOGIWEBCAM_PASSWORD = Read-Host 'Password from kit card' -MaskInput

& ./logiinsight_api/.venv/Scripts/python.exe ./logiinsight_api/check_connection.py
& ./logiinsight_api/.venv/Scripts/python.exe ./logiinsight_api/rgb_ai_live.py --face-mesh
```

`-MaskInput` requires PowerShell 7.1+. Keep credentials in environment variables;
do not save them in source files. The connection checker may report missing AI
data until the RGB stream is open.

The camera's USB network adapter needs the PC address `192.168.1.20`, subnet mask
`255.255.255.0`, and empty gateway/DNS. The camera is at `192.168.1.10:22`.
See the [SDK setup guide](logiinsight_api/README.md) for adapter setup.

Choose the landscape RGB camera when prompted. Use `--device 1` (or its actual
index) to skip selection. Press **q**, **Escape**, or **Ctrl+C** to quit.
Omit `--face-mesh` for face boxes only; `--no-overlay` shows video with AI status.

Video and JSON are independent: overlays may lag a moving face. Coordinates
are absolute RGB stream pixels, and mesh points are sometimes absent. Camera
timestamps can wrap or jump; freshness uses host monotonic time and checks
whether the device timestamp changes. No host/device clock subtraction is used.
Keep experiments short: the SDK documents possible overheating/freezing after
about ten minutes of continuous AI. After unplugging, restart the preview.

For exact image/result pairs, the SDK also supports `--device-snapshot` (much
slower, about two seconds per 1080p snapshot).

Reference: [Logitech API documentation](https://raw.githubusercontent.com/Logitech/hackathons/master/docs/logiinsight/API.md).

## Verify without hardware

```powershell
py -3 -m unittest discover -s logiinsight_api -p 'test_*.py' -v
```
