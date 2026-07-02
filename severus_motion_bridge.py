"""Bridge MediaPipe hand recognition to a Severus Arduino prosthetic hand over serial.

This script reuses the proven camera/MediaPipe/finger-state logic from
``motion_recognition.py`` and adds a serial bridge plus an on-screen HUD that
shows every command and the calibration/runtime state machine.

Requirements:
    pip install opencv-python mediapipe pyserial

Usage:
  python severus_motion_bridge.py --port COM3 --baud 115200 --camera 0
  python severus_motion_bridge.py --no-serial            # test without hardware

Workflow:
  1) CALIBRATION phase: keys typed in the camera window are forwarded raw to the
     Arduino so you can calibrate (W, space/Enter, Backspace, z x c v n, e, k).
     When the Arduino prints "Calibration done" the script switches to RUNTIME.
  2) RUNTIME phase: press 'u' to arm. While armed, finger-state changes are
     mimicked on the hand automatically. 'o'/'p' open/close, 'w' recalibrates.
"""

from __future__ import annotations

import argparse
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover - allow no-serial use without pyserial
    serial = None


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Order the Arduino expects for the full-gesture 'g' command and for mimic.
GESTURE_ORDER = ["index", "middle", "ring", "pinky", "thumb"]

# Python finger name -> Arduino finger digit for the 'f' individual command.
FINGER_DIGIT = {
    "index": "1",
    "middle": "2",
    "ring": "3",
    "pinky": "4",
    "thumb": "5",
}

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge MediaPipe hand tracking to a Severus Arduino prosthetic hand."
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
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="Serial port for the Arduino (default: COM3).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate (default: 115200).",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Run without hardware; commands are only printed/HUD-shown.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MediaPipe / camera helpers (adapted from motion_recognition.py)
# ---------------------------------------------------------------------------
def finger_states(hand_landmarks, handedness_label: str) -> dict[str, bool]:
    """Return finger states, True=extended, False=retracted."""
    lm = hand_landmarks
    states: dict[str, bool] = {}

    # Thumb logic: compare x positions based on handedness.
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

    Forcing MJPG (compressed) keeps the high-resolution stream within USB 2.0
    bandwidth so the feed runs at full speed instead of dropping to ~1 FPS.
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


# ---------------------------------------------------------------------------
# Serial bridge
# ---------------------------------------------------------------------------
class SerialBridge:
    """Wrap a pyserial connection with a background reader thread + HUD state."""

    def __init__(self, port: str, baud: int, no_serial: bool) -> None:
        self.port = port
        self.baud = baud
        self.no_serial = no_serial
        self.ser = None

        self.last_tx: str = ""
        self.rx_lines: deque[str] = deque(maxlen=8)
        self.calibration_done = False

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None

        if not no_serial:
            self._open()

    def _open(self) -> None:
        if serial is None:
            print("WARNING: pyserial not installed; running in no-serial mode.")
            self.no_serial = True
            return
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            # Arduino auto-resets on serial open; wait for the bootloader/setup.
            time.sleep(2.0)
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            print(f"Serial open on {self.port} @ {self.baud}.")
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(f"WARNING: could not open serial port {self.port}: {exc}")
            print("Continuing in no-serial mode (commands printed only).")
            self.ser = None
            self.no_serial = True

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            if self.ser is None:
                break
            try:
                raw = self.ser.readline()
            except Exception:
                break
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            with self._lock:
                self.rx_lines.append(line)
                if "calibration done" in line.lower():
                    self.calibration_done = True

    def send(self, data: bytes) -> None:
        """Write bytes to the Arduino (if connected) and update the HUD."""
        text = data.decode("ascii", errors="replace")
        if self.ser is not None and not self.no_serial:
            try:
                self.ser.write(data)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: serial write failed: {exc}")
        with self._lock:
            self.last_tx = text
        print(f"TX: {text!r}")

    def get_last_tx(self) -> str:
        with self._lock:
            return self.last_tx

    def get_rx_lines(self) -> list[str]:
        with self._lock:
            return list(self.rx_lines)

    def consume_calibration_done(self) -> bool:
        """Return True once when calibration completion was detected."""
        with self._lock:
            if self.calibration_done:
                self.calibration_done = False
                return True
            return False

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def draw_hud(
    frame,
    states: dict[str, bool],
    calibrated: bool,
    armed: bool,
    bridge: SerialBridge,
) -> None:
    """Draw finger states, phase/control banners, legend, and TX/RX panels."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Top-left: per-finger states ---
    y0 = 30
    dy = 28
    for i, name in enumerate(FINGER_NAMES):
        extended = states.get(name, False)
        text = f"{name}: {'extended' if extended else 'retracted'}"
        cv2.putText(
            frame,
            text,
            (10, y0 + i * dy),
            font,
            0.7,
            (0, 255, 0) if extended else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # --- Top-right: phase + control banners ---
    phase_text = "PHASE: CALIBRATION" if not calibrated else "PHASE: READY"
    phase_color = (0, 200, 255) if not calibrated else (0, 255, 0)
    cv2.putText(frame, phase_text, (w - 470, 30), font, 0.8, phase_color, 2, cv2.LINE_AA)

    if not calibrated:
        control_text = "CONTROL: (calibrate first)"
        control_color = (0, 200, 255)
    elif armed:
        control_text = "CONTROL: ARMED"
        control_color = (0, 255, 0)
    else:
        control_text = "CONTROL: DISARMED"
        control_color = (0, 0, 255)
    cv2.putText(frame, control_text, (w - 470, 62), font, 0.8, control_color, 2, cv2.LINE_AA)

    # --- Bottom: legend for the current phase ---
    if not calibrated:
        legend = (
            "W=start  Enter/Backspace=set close dir  "
            "z x c v n=extend F1-F5  e=record  k=finish  q=quit"
        )
    else:
        legend = "u=arm/disarm  o=open  p=close  w=recalibrate  x=switch cam  q=quit"
    cv2.putText(
        frame, legend, (10, h - 70), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    # --- Bottom: last TX command ---
    last_tx = bridge.get_last_tx()
    cv2.putText(
        frame,
        f"TX: {last_tx}" if last_tx else "TX: -",
        (10, h - 45),
        font,
        0.6,
        (255, 200, 0),
        2,
        cv2.LINE_AA,
    )

    # --- Bottom-right: last 3 RX lines ---
    rx = bridge.get_rx_lines()[-3:]
    for i, line in enumerate(rx):
        cv2.putText(
            frame,
            f"RX: {line}",
            (w - 470, h - 70 + i * 22),
            font,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------
def build_gesture(states: dict[str, bool]) -> bytes:
    """Build a full 'g' gesture command from current states (order GESTURE_ORDER)."""
    chars = "".join("o" if states.get(name, False) else "c" for name in GESTURE_ORDER)
    return b"g" + chars.encode("ascii")


def build_finger(name: str, extended: bool) -> bytes:
    """Build an individual 'f' finger command, e.g. b'f1o'."""
    digit = FINGER_DIGIT[name]
    direction = "o" if extended else "c"
    return ("f" + digit + direction).encode("ascii")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    bridge = SerialBridge(args.port, args.baud, args.no_serial)

    cameras = discover_cameras(args.max_camera_index, args.allow_virtual)
    if not cameras:
        cameras = [{"index": args.camera, "name": ""}]
    else:
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
        bridge.close()
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

    # State machine flags.
    calibrated = False
    armed = False

    smoothed_states: dict[str, bool] = {name: False for name in FINGER_NAMES}
    last_sent_states: dict[str, bool] = {}
    state_history: deque[dict[str, bool]] = deque(maxlen=max(args.smooth_frames, 1))
    last_time = time.time()

    print("Bridge running. Calibrate via the camera window, then press 'u' to arm.")

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
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30 + len(FINGER_NAMES) * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Background reader may have detected calibration completion.
        if not calibrated and bridge.consume_calibration_done():
            calibrated = True
            armed = False
            print("Calibration done detected -> RUNTIME phase. Press 'u' to arm.")

        # --- Automatic mimic (only when calibrated AND armed) ---
        if calibrated and armed and smoothed_states:
            for name in GESTURE_ORDER:
                now_ext = smoothed_states.get(name, False)
                prev = last_sent_states.get(name)
                if prev is None or now_ext != prev:
                    bridge.send(build_finger(name, now_ext))
                    last_sent_states[name] = now_ext

        # --- HUD ---
        draw_hud(frame, smoothed_states, calibrated, armed, bridge)

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

        cv2.imshow("Severus Motion Bridge", frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit (works in any phase).
        if key in (ord("q"), 27):
            break

        if not calibrated:
            # ---------------- CALIBRATION phase: forward keys raw ----------------
            if key in (ord("w"), ord("W")):
                bridge.send(b"W")
                calibrated = False
                armed = False
            elif key in (13, 32):  # Enter or Space -> close dir +1
                bridge.send(b" ")
            elif key in (8, 127):  # Backspace / Delete -> close dir -1
                bridge.send(bytes([8]))
            elif key in (ord("z"), ord("Z")):
                bridge.send(b"z")
            elif key in (ord("x"), ord("X")):
                bridge.send(b"x")
            elif key in (ord("c"), ord("C")):
                bridge.send(b"c")
            elif key in (ord("v"), ord("V")):
                bridge.send(b"v")
            elif key in (ord("n"), ord("N")):
                bridge.send(b"n")
            elif key in (ord("e"), ord("E")):
                bridge.send(b"e")
            elif key in (ord("k"), ord("K")):
                bridge.send(b"k")
        else:
            # ---------------- RUNTIME phase ----------------
            if key in (ord("u"), ord("U")):
                armed = not armed
                if armed:
                    # Full sync so the hand matches the live pose immediately.
                    bridge.send(build_gesture(smoothed_states))
                    last_sent_states = {
                        name: smoothed_states.get(name, False) for name in GESTURE_ORDER
                    }
                    print("ARMED: hand will mimic finger movements.")
                else:
                    print("DISARMED: auto-mimic stopped (hand holds position).")
            elif key in (ord("o"), ord("O")):
                bridge.send(b"o")  # open whole hand
            elif key in (ord("p"), ord("P")):
                bridge.send(b"p")  # close whole hand
            elif key in (ord("w"), ord("W")):
                bridge.send(b"W")  # re-enter calibration
                calibrated = False
                armed = False
            elif key in (ord("x"), ord("X")) and len(cameras) > 1:
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
                continue

    hand_landmarker.close()
    cap.release()
    bridge.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
