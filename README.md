# Human-Detection-and-Motion-Analysis-System
Final Project for CSC 231 - System programming

Real-time human pose estimation desktop application using OpenCV + MediaPipe, designed for direct laptop webcam input (built-in or USB camera).

## Recommended Python and dependency versions
To avoid version conflicts (especially with `mediapipe`), use **Python 3.10**.

- Best choice: **Python 3.10.x**
- Usually OK: Python 3.9 or 3.11
- Avoid for this project: Python 3.12+ (MediaPipe compatibility issues are common)

## Installation (recommended)
```bash
# 1) Create and activate a fresh Python 3.10 virtual environment
python3.10 -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 2) Upgrade pip tools
python -m pip install --upgrade pip setuptools wheel

# 3) Install project as an application
python -m pip install .
```

After installation, you can launch it as:
```bash
pose-app
```
(Or `pose-app-gui` on systems that support GUI script launchers.)

## Run directly (without install)
```bash
# Built-in laptop camera (default)
python droidcam_pose_app.py --source 0

# External USB camera (try index 1 or 2)
python droidcam_pose_app.py --source 1
```

## Build a standalone executable (for running on other laptops)
```bash
python -m pip install pyinstaller
python build_executable.py
```
This creates a **portable folder build** in `dist/LaptopPoseEstimationApp/`.

Run it as:
- **Windows:** `dist\LaptopPoseEstimationApp\LaptopPoseEstimationApp.exe`
- **Linux/macOS:** `./dist/LaptopPoseEstimationApp/LaptopPoseEstimationApp`

Important portability notes:
- Copy the **entire** `LaptopPoseEstimationApp` folder to the other device (not just the `.exe`).
- Build on the same OS you want to run on (Windows build for Windows, etc.).
- This build now bundles MediaPipe data/model files to avoid missing `pose_landmark_lite.tflite` errors.

## Runtime controls and output
- Press **`f`** to toggle fullscreen/windowed mode.
- You can resize the window in normal mode.
- Press **`q`** to exit.
- On-screen overlay shows:
  - FPS
  - People detected count (`0` or `1` in single-person mode)
  - Optional motion vectors count (`--optical-flow`)

## Optional Features
```bash
python droidcam_pose_app.py \
  --source 0 \
  --save-output processed_output.mp4 \
  --print-joints \
  --optical-flow
```

## MediaPipe troubleshooting
If you get an error like `module 'mediapipe' has no attribute 'solutions'`:

```bash
python -c "import sys; print(sys.version)"
python -m pip show mediapipe opencv-python numpy protobuf
python -m pip install --upgrade --force-reinstall -r requirements.txt
```

Also ensure there is no local file/folder named `mediapipe.py` or `mediapipe` in your project directory.
