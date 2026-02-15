"""Build a standalone desktop executable with PyInstaller.

This script explicitly bundles MediaPipe model/data files so the app works on
other laptops without missing `*.tflite` runtime errors.

Usage:
    python -m pip install pyinstaller
    python build_executable.py
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",  # more reliable for large ML/data dependencies than --onefile
        "--windowed",
        "--name",
        "LaptopPoseEstimationApp",
        "--collect-data",
        "mediapipe",
        "--collect-submodules",
        "mediapipe",
        "--hidden-import",
        "mediapipe.python.solutions.pose",
        "--hidden-import",
        "mediapipe.python.solutions.holistic",
        "droidcam_pose_app.py",
    ]

    subprocess.run(cmd, check=True)
    print("Build complete. Run: ./dist/LaptopPoseEstimationApp/LaptopPoseEstimationApp")


if __name__ == "__main__":
    main()
