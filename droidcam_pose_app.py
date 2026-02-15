"""
Installation:
    python -m pip install -r requirements.txt

Recommended Python:
    Python 3.10.x

Run examples:
    python droidcam_pose_app.py --source 0
    python droidcam_pose_app.py --source "http://192.168.x.x:4747/video"
    python droidcam_pose_app.py --source 0 --save-output processed_output.mp4 --print-joints --optical-flow
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np


def _get_pose_api():
    """Resolve MediaPipe Pose API across packaging variants."""
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return mp.solutions.pose

    try:
        from mediapipe.python.solutions import pose as mp_pose  # type: ignore

        return mp_pose
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "MediaPipe Pose API could not be resolved. "
            "Try: python -m pip show mediapipe\n"
            "Then reinstall: python -m pip install --upgrade --force-reinstall mediapipe"
        ) from exc


MP_POSE = _get_pose_api()

JOINT_MAP = {
    "head": MP_POSE.PoseLandmark.NOSE,
    "left_shoulder": MP_POSE.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": MP_POSE.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": MP_POSE.PoseLandmark.LEFT_ELBOW,
    "right_elbow": MP_POSE.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": MP_POSE.PoseLandmark.LEFT_WRIST,
    "right_wrist": MP_POSE.PoseLandmark.RIGHT_WRIST,
    "left_hip": MP_POSE.PoseLandmark.LEFT_HIP,
    "right_hip": MP_POSE.PoseLandmark.RIGHT_HIP,
    "left_knee": MP_POSE.PoseLandmark.LEFT_KNEE,
    "right_knee": MP_POSE.PoseLandmark.RIGHT_KNEE,
    "left_ankle": MP_POSE.PoseLandmark.LEFT_ANKLE,
    "right_ankle": MP_POSE.PoseLandmark.RIGHT_ANKLE,
}

SKELETON_EDGES = [
    ("head", "left_shoulder"),
    ("head", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def load_model(
    model_complexity: int = 0,
    min_detection_confidence: float = 0.55,
    min_tracking_confidence: float = 0.55,
):
    """Load and return single-person MediaPipe Pose model."""
    return MP_POSE.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def preprocess_frame(frame: np.ndarray, blur_kernel: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """Apply Gaussian filtering to reduce frame noise with low overhead."""
    return cv2.GaussianBlur(frame, blur_kernel, 0)


def detect_pose(
    model,
    frame: np.ndarray,
    visibility_threshold: float = 0.55,
) -> Tuple[Optional[object], Dict[str, Tuple[int, int]]]:
    """Run single-person pose estimation and extract selected keypoint pixel coordinates."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(rgb_frame)

    keypoints: Dict[str, Tuple[int, int]] = {}
    if not results.pose_landmarks:
        return None, keypoints

    h, w = frame.shape[:2]
    for name, landmark_id in JOINT_MAP.items():
        landmark = results.pose_landmarks.landmark[landmark_id]
        if landmark.visibility >= visibility_threshold:
            keypoints[name] = (int(landmark.x * w), int(landmark.y * h))

    return results, keypoints


def draw_skeleton(frame: np.ndarray, keypoints: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Draw keypoints and skeleton edges on the frame."""
    for start_joint, end_joint in SKELETON_EDGES:
        if start_joint in keypoints and end_joint in keypoints:
            cv2.line(frame, keypoints[start_joint], keypoints[end_joint], (255, 180, 0), 2)

    for _, (x, y) in keypoints.items():
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    return frame


def _compute_motion_overlay(
    prev_gray: Optional[np.ndarray],
    curr_bgr: np.ndarray,
    max_corners: int = 120,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Estimate simple motion vectors using sparse optical flow."""
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return curr_bgr, curr_gray, 0

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_corners,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )
    if prev_pts is None:
        return curr_bgr, curr_gray, 0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if curr_pts is None or status is None:
        return curr_bgr, curr_gray, 0

    good_new = curr_pts[status.flatten() == 1]
    good_old = prev_pts[status.flatten() == 1]

    motion_vectors = 0
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.arrowedLine(curr_bgr, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 1, tipLength=0.3)
        motion_vectors += 1

    return curr_bgr, curr_gray, motion_vectors


def _parse_source(source: str) -> Union[int, str]:
    return int(source) if source.isdigit() else source


def _toggle_fullscreen(window_name: str, is_fullscreen: bool) -> bool:
    new_state = not is_fullscreen
    if new_state:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    return new_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time single-person pose estimation from DroidCam.")
    parser.add_argument("--source", default="0", help="Webcam index or stream URL.")
    parser.add_argument("--save-output", default=None, help="Optional output video path (e.g., out.mp4).")
    parser.add_argument("--print-joints", action="store_true", help="Print joint coordinates to console.")
    parser.add_argument("--optical-flow", action="store_true", help="Overlay basic optical-flow motion vectors.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(_parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    pose_model = load_model()

    window_name = "DroidCam Pose Estimation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = False

    writer = None
    if args.save_output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        fps_out = fps_out if fps_out and fps_out > 1 else 20.0
        writer = cv2.VideoWriter(args.save_output, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (width, height))

    prev_gray = None
    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed. Exiting...")
                break

            processed = preprocess_frame(frame)
            _, keypoints = detect_pose(pose_model, processed)
            output = draw_skeleton(processed, keypoints)

            if args.print_joints and keypoints:
                print(keypoints)

            if args.optical_flow:
                output, prev_gray, vector_count = _compute_motion_overlay(prev_gray, output)
                cv2.putText(output, f"Motion vectors: {vector_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            people_count = 1 if keypoints else 0
            cv2.putText(output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(output, f"People detected: {people_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(output, "Press 'f' for fullscreen, 'q' to quit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow(window_name, output)
            if writer is not None:
                writer.write(output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                is_fullscreen = _toggle_fullscreen(window_name, is_fullscreen)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        pose_model.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
