"""Real-time hand landmark recognition with finger state output.

Requirements:
    pip install opencv-python mediapipe

Usage:
  python motion_recognition.py --camera 0
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="USB camera hand tracking and finger state detection."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="USB camera index (default: 0).",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="Maximum number of hands to track (default: 1).",
    )
    parser.add_argument(
        "--min-detection",
        type=float,
        default=0.6,
        help="Minimum detection confidence (default: 0.6).",
    )
    parser.add_argument(
        "--min-tracking",
        type=float,
        default=0.6,
        help="Minimum tracking confidence (default: 0.6).",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=10,
        help="Max camera index to scan (default: 10).",
    )
    parser.add_argument(
        "--allow-virtual",
        action="store_true",
        help="Allow virtual cameras like OBS Virtual Camera.",
    )
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=5,
        help="Frames to smooth finger state (default: 5).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture height (default: 720).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested capture FPS (default: 30).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do not force MJPG (use the camera's default format).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to hand_landmarker.task (defaults to script folder).",
    )
    return parser.parse_args()


def finger_states(hand_landmarks, handedness_label: str) -> dict[str, bool]:
    """Return finger states, True=extended, False=retracted."""
    lm = hand_landmarks

    # Landmarks used for fingers: tip, pip (or ip for thumb)
    # Thumb: CMC(1), MCP(2), IP(3), TIP(4)
    # Index: MCP(5), PIP(6), DIP(7), TIP(8)
    # Middle: MCP(9), PIP(10), DIP(11), TIP(12)
    # Ring: MCP(13), PIP(14), DIP(15), TIP(16)
    # Pinky: MCP(17), PIP(18), DIP(19), TIP(20)

    states: dict[str, bool] = {}

    # Thumb logic: compare x positions based on handedness.
    # For right hand, thumb tip is right of IP when extended; for left hand, opposite.
    if handedness_label.lower() == "right":
        states["thumb"] = lm[4].x > lm[3].x
    else:
        states["thumb"] = lm[4].x < lm[3].x

    # Other fingers: tip above PIP in image means extended (y is smaller).
    states["index"] = lm[8].y < lm[6].y
    states["middle"] = lm[12].y < lm[10].y
    states["ring"] = lm[16].y < lm[14].y
    states["pinky"] = lm[20].y < lm[18].y

    return states


def overlay_states(frame, states: dict[str, bool]) -> None:
    """Draw finger states on the frame."""
    y0 = 30
    dy = 28
    for i, name in enumerate(FINGER_NAMES):
        state_text = "extended" if states.get(name, False) else "retracted"
        text = f"{name}: {state_text}"
        cv2.putText(
            frame,
            text,
            (10, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if state_text == "extended" else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if all(states.get(name, False) for name in FINGER_NAMES):
        cv2.putText(
            frame,
            "state: open hand",
            (10, y0 + len(FINGER_NAMES) * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )


def emit_commands(states: dict[str, bool], last_states: dict[str, bool]) -> dict[str, bool]:
    """Print commands when a finger changes state."""
    for name in FINGER_NAMES:
        current = states.get(name, False)
        previous = last_states.get(name)
        if previous is None:
            continue
        if current != previous:
            action = "extend" if current else "retract"
            print(f"{action} {name}")
    return states.copy()


def ensure_model_path(model_arg: str) -> Path:
    """Download the MediaPipe hand landmarker model if missing."""
    if model_arg:
        model_path = Path(model_arg)
    else:
        model_path = Path(__file__).resolve().parent / "hand_landmarker.task"

    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model to {model_path}...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def draw_landmarks(frame, landmarks) -> None:
    """Draw landmarks and connections onto the frame."""
    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 0, 0), 2)


def discover_cameras(max_index: int, allow_virtual: bool) -> list[dict[str, str | int]]:
    """Probe camera indices and optionally skip virtual cameras like OBS."""
    device_names: list[str] = []
    if not allow_virtual:
        try:
            from pygrabber.dshow_graph import FilterGraph

            device_names = FilterGraph().get_input_devices()
        except Exception:
            device_names = []

    cameras: list[dict[str, str | int]] = []
    for index in range(max_index + 1):
        name = device_names[index] if index < len(device_names) else ""
        if not allow_virtual and name and "obs" in name.lower():
            continue

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = cap.read()
        cap.release()
        if not ok:
            continue

        cameras.append({"index": index, "name": name})

    return cameras


def open_camera(
    index: int,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    force_mjpg: bool = True,
) -> cv2.VideoCapture:
    """Open a camera index using DirectShow, tuned for USB webcams.

    Integrated USB 2.0 webcams default to raw (YUY2) streaming, which cannot
    fit a high resolution into USB 2.0 bandwidth -- the result is black frames
    and ~1 FPS. Forcing MJPG (compressed) keeps it within bandwidth so the feed
    runs at full speed.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return cap

    if force_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Keep only the freshest frame so we don't drift behind real time.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Warm up: the first frames after init are often empty/black.
    for _ in range(5):
        cap.read()
    return cap


def smooth_states(history: deque[dict[str, bool]]) -> dict[str, bool]:
    """Smooth states by majority vote per finger."""
    if not history:
        return {}

    totals: dict[str, int] = {name: 0 for name in FINGER_NAMES}
    for states in history:
        for name in FINGER_NAMES:
            if states.get(name, False):
                totals[name] += 1

    threshold = max(len(history) // 2 + 1, 1)
    return {name: totals[name] >= threshold for name in FINGER_NAMES}


def main() -> None:
    args = parse_args()

    cameras = discover_cameras(args.max_camera_index, args.allow_virtual)
    if not cameras:
        cameras = [{"index": args.camera, "name": ""}]
    else:
        # Honor --camera: move the requested index to the front of the list.
        requested = [c for c in cameras if int(c["index"]) == args.camera]
        if requested:
            others = [c for c in cameras if int(c["index"]) != args.camera]
            cameras = requested + others

    print("Available cameras:")
    for cam in cameras:
        print(f"  index {cam['index']}: {cam['name'] or '<unnamed>'}")

    camera_pos = 0
    cap = open_camera(
        int(cameras[camera_pos]["index"]),
        args.width,
        args.height,
        args.fps,
        force_mjpg=not args.raw,
    )
    if not cap.isOpened():
        raise SystemExit("Failed to open USB camera.")

    model_path = ensure_model_path(args.model)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        min_hand_presence_confidence=args.min_detection,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    last_states: dict[str, bool] = {}
    state_history: deque[dict[str, bool]] = deque(maxlen=max(args.smooth_frames, 1))
    last_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        current_states: dict[str, bool] | None = None
        if results.hand_landmarks and results.handedness:
            hand_landmarks = results.hand_landmarks[0]
            handedness_label = "Right"
            try:
                handedness_label = results.handedness[0][0].category_name
            except (AttributeError, IndexError):
                pass

            draw_landmarks(frame, hand_landmarks)
            current_states = finger_states(hand_landmarks, handedness_label)
            state_history.append(current_states)
            smoothed_states = smooth_states(state_history)
            overlay_states(frame, smoothed_states)

            if last_states:
                last_states = emit_commands(smoothed_states, last_states)
            else:
                last_states = smoothed_states.copy()
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Optional FPS display
        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Motion Recognition Workshop", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("x"), ord("X")) and len(cameras) > 1:
            cap.release()
            camera_pos = (camera_pos + 1) % len(cameras)
            cap = open_camera(
                int(cameras[camera_pos]["index"]),
                args.width,
                args.height,
                args.fps,
                force_mjpg=not args.raw,
            )
            state_history.clear()
            last_states = {}
            continue
        if key == ord("q"):
            break

    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
