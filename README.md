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
Space is centered at the thumb midpoint. The layout stays fixed through small
hand shifts and thumb movements. To dismiss it, keep both thumbs visible and
move them clearly apart for about 0.4 seconds. Bring them together again to
build a new layout. Press **R** in the preview to reset it immediately.

Move a fingertip over a key, dip it down toward the desk, then lift it while
keeping the palm steady. A completed tap enters one key in the text area below
the video. Thumbs can tap Space. Brief tracking gaps pause input and cancel
unfinished taps. Return your hands to the same area to resume on the existing
layout. Use `--no-keyboard` for the fingertip overlay.

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

## Desktop control with eyes and hands

```cmd
uv run rgb_preview.py --mode desktop --mirror
```

Press **Ctrl+Alt+F8** to enable desktop input. With the preview focused, press
**C** and follow the full-screen dot through 17 positions on your primary
monitor. Keep your head steady and spend about two seconds looking at each dot.
Then keep looking at the **center dot** while following four prompts to turn
your head slightly left and right and tilt it up and down. The last four holds
teach the pointer how your eye measurements change when you move your head
but keep looking at the same spot. Allow about 45 seconds for the full pass.
Use **Esc** to cancel or **C** to restart from the preview. The first run downloads
Google's Face Landmarker model.

Once calibrated, look where you want the pointer to go. Blink one eye
deliberately and reopen it for a left-click. An ordinary blink with both eyes
doesn't click. The pointer holds through small fluctuations and responds faster
to a sustained change in where you look. Head correction works within the range
covered by the calibration. Press **C** to recalibrate if the camera moves.
Hand navigation works before and after gaze calibration. It pauses
while the full-screen targets are showing. To test hands alone, run
`uv run rgb_preview.py --mode touchpad --mirror`.

Keep the wrist and fingertips of your gesture hand visible. Only one hand
should make a navigation pose at a time.

| Gesture | Action |
| --- | --- |
| Open your hand with your fingers together and move it down, up, or sideways | Scroll in the direction of movement |
| Tuck middle, ring, and pinky. Spread thumb and index | Zoom in (Ctrl+wheel) |
| Tuck middle, ring, and pinky. Pinch thumb and index | Zoom out (Ctrl+wheel) |
| Bring all five fingertips of one hand together in front of the knuckles and hold briefly | Open Chrome's tab search and overview (Ctrl+Shift+A) |
| Extend index, middle, and ring with pinky folded. Swipe left or right | Previous or next tab in apps that support Ctrl+Tab |
| Make a fist **or** hold a flat hand with your fingers spread | Stop and reset any gesture |

Each zoom gesture locks to its first direction. Return your thumb and index
to the starting position and repeat the stroke to keep zooming in that
direction. Hold either reset pose briefly to finish. You can move your hand
while holding that pose without sending commands. Bring your fingers back
together to start scrolling from the new position, or start a fresh pinch.
Open your hand or make a fist between tab-overview gestures. Lower a finger
between tab swipes. Keep the browser window focused for tab commands. The
preview labels each navigation command it sends to Windows. Use **Ctrl+Alt+F8**
to pause from any app.
Press **Q** or **Esc** in the preview to close it. The gaze estimate is
experimental and depends on camera placement, lighting, and eye visibility.
Use `--mode eye` for gaze alone or `--mode touchpad` for hand navigation with a
physical mouse. The tracking runs locally on the Logitech RGB stream.

## Live lip reading (experimental)

After completing the [Auto-AVSR model setup](silent_transcription_setup.md),
start the RGB preview in lip-reading mode.

```cmd
uv run rgb_preview.py --mode lip
```

Once the model is ready, focus the preview and press **Space**. With your full
face visible, silently mouth a short English phrase and press **Space** again.
The draft appears below the RGB feed. The model runs locally, keeps the last
phrase in memory during processing, and uses RGB video alone. Each phrase
submits automatically at 10 seconds. Review the draft before using it.

The [model setup](silent_transcription_setup.md) also covers transcription of
existing video files.

## Handwriting to LaTeX, code, and KiCad

Run **uv run paper_workbench.py** to capture a page or open a photo. Choose a
format, check the transcription, and export the editable draft. The MX Keypad
can select the output and display a second menu for VS Code languages.
See the [Paper Workbench guide](paper_workbench.md).

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
uv run --locked python -m unittest discover -s tests
```
