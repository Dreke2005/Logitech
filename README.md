# Logitech camera and hand-control project

These instructions are for Windows PowerShell. Run commands from this folder:
`C:\Users\victo\Documents\projet\Logitech`.

## Setup

Install the main project dependencies and the separate hand-control environment:

```powershell
uv sync
uv sync --project .\Logitech-Keypad
```

The dual-camera launcher uses the nested environment for camera 0 hand
controls and the main environment for camera 1 typing or paper reading.

List the available RGB cameras if you need to check their indices:

```powershell
uv run .\rgb_preview.py --list
```

The examples below use camera 0 for the Logitech RGB hand-control camera and
camera 1 for the keyboard or paper view. Change the indices if your camera list
shows a different order. Press **Q** or **Esc** to close a camera preview.

## Run everything together

Run the hand mouse controls and wink clicks on camera 0 with physical keyboard
typing on camera 1:

```powershell
uv run .\run_dual_camera_controls.py
```

The camera 0 controls start active. **Ctrl+Alt+F8** pauses or resumes them.
**Ctrl+Alt+F9** switches camera 1 between keyboard typing and Paper Workbench
in the LaTeX/formula menu. The launcher closes the current camera 1 app before
starting the other.

To start directly in paper/formula mode:

```powershell
uv run .\run_dual_camera_controls.py --paper
```

The optional MX Keypad can be enabled for the paper session with:

```powershell
uv run .\run_dual_camera_controls.py --paper --keypad
```

The hand controls stay on camera 0 while camera 1 runs one of the two other
workflows. Keyboard typing and live paper reading share camera 1, so they are
switched between instead of running simultaneously.

## Run each feature separately

### Live virtual keyboard and fingertips

Show both hands. Hold the thumbs together briefly to place the on-screen
keyboard; spread the thumbs to dismiss it. Press **R** to reset its layout.

```powershell
uv run .\rgb_preview.py --mode keyboard --device 0 --mirror
```

For fingertip tracking without the virtual keyboard:

```powershell
uv run .\rgb_preview.py --mode keyboard --device 0 --mirror --no-keyboard
```

### Hand-only or upper-body pose capture

Hand-only capture tracks the two hands and their finger landmarks:

```powershell
uv run .\hand_only_capture.py --camera 0
```

This separate view includes arm/body pose landmarks:

```powershell
uv run .\hand_pose_capture.py --camera 0
```

### Hand mouse, gestures, and wink clicks

The standalone hand-control view starts active. Point the left index left to
catch/drag the pointer; use the right hand for navigation gestures. A left-eye
wink holds the left button until the eye opens; a right-eye wink clicks the
right button. Both eyes closed are ignored.

```powershell
uv run .\rgb_preview.py --mode touchpad --device 0 --mirror `
  --right-hand-only --hand-pointer --wink-clicks
```

### Physical keyboard mapping and typing

Check or calibrate the keyboard mapping with the live preview:

```powershell
uv run .\test_key_mapping.py --camera 1
```

It searches for the keyboard automatically. Press **Enter** to confirm a
candidate, **A** to rescan, or **C** to click the four corners in order:
top-left, top-right, bottom-right, bottom-left. Press **S** to save a snapshot.

Run the typing detector:

```powershell
uv run .\keyboard_typing_detector.py --camera 1 --finger indexes-only
```

Press **C** and click the four keyboard corners if automatic mapping is not
right; drag numbered corners to adjust them. Click a text field before typing.
To keep output to text keys, add `--text-only`. To preview without sending
keystrokes to other apps, add `--no-system-output`.

### Paper reading and formulas

Run Paper Workbench with a live camera:

```powershell
uv run .\paper_workbench.py --camera 1
```

Or open an existing image:

```powershell
uv run .\paper_workbench.py --image .\page.jpg
```

Use the on-screen buttons to choose an app and action, then review the reading
and generated draft. The app needs its local Ollama vision model; setup and
writing guidance are in [Logitech-Keypad/paper_workbench.md](Logitech-Keypad/paper_workbench.md).
The physical keypad is optional; the on-screen controls work without it.

To install the MX Keypad integration, install Logi Options+ and its plugin
service, then run the setup script from the project folder:

```powershell
.\Logitech-Keypad\keypad-plugin\setup.ps1
```

Then launch Paper Workbench with `--keypad`, or run the combined launcher with
`--paper --keypad`. The keypad plugin and the app communicate locally.

### Raw camera diagnostics

List DirectShow devices with `rgb_preview.py --list`. The following scripts
perform basic camera probes:

```powershell
uv run .\test_cam.py
uv run .\test_cam2.py
uv run .\test_cam3.py
```

### LogiInsight AI over the camera network

The SDK's AI data and capture examples use SSH/SFTP credentials from the card
in the kit. Set them in the current PowerShell session, then check the link:

```powershell
$env:LOGIWEBCAM_USER = Read-Host 'Device user name'
$env:LOGIWEBCAM_PASSWORD = Read-Host 'Device password' -MaskInput
uv run .\check_connection.py
```

Live RGB with AI overlays:

```powershell
uv run .\rgb_ai_live.py
```

Capture RGB frames and AI results:

```powershell
uv run .\rgb_ai_capture.py --frames 10 --output .\captures
```

These network/AI examples are separate from local RGB hand tracking. For
network troubleshooting, see [docs/API.md](docs/API.md) and run
`uv run .\check_connection.py`.

### Near-infrared camera and microphone

```powershell
uv run .\example_IR.py
uv run .\example_audio.py --list
uv run .\example_audio.py --duration 10
```

The IR preview can save a snapshot with **S**. The audio example saves a WAV
and spectrum plot under `audio_captures`.

## Excluded experimental features

Eye-gaze pointer control and live lip reading are experimental and are not part
of the recommended launch commands. Wink clicks remain available through the
camera 0 hand-control workflow.

## Tests

Run the project tests with:

```powershell
uv run python -m unittest discover -s Logitech-Keypad\tests -v
```
