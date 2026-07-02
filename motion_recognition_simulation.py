"""Real-time hand-gesture recognition with a simulated (or real) Severus prosthetic hand.

This is a simulation-friendly companion to ``motion_recognition.py``.

Behaviour
---------
1. On start it scans the serial ports for a connected Arduino (the Severus
   5-finger controller running ``260612_Severus_5FingerControl_v3.ino``).
     * If an Arduino is found, it connects and **sends the real finger commands**
       (``f<digit><o|c>``) over serial as fingers extend / retract -- exactly the
       protocol the .ino expects (F1=index .. F5=thumb, o=extend, c=retract).
     * If no Arduino is found (or ``--force-sim`` / no pyserial), it drops into
       **Simulation Mode**: a pop-up announces it and the OpenCV window is widened
       to show two panels side-by-side -- the camera/gesture view on the left and a
       drawn robotic hand on the right that mirrors every finger in real time.

The camera gesture-recognition pipeline (MediaPipe hand landmarks -> per-finger
extended/retracted states) is reused verbatim from ``motion_recognition.py``.

Requirements:
    pip install opencv-python mediapipe numpy pyserial

Usage:
    python motion_recognition_simulation.py --camera 0
    python motion_recognition_simulation.py --force-sim      # always simulate
    python motion_recognition_simulation.py --port COM5      # force a serial port
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Reuse the shared pipeline pieces from the sibling module (same folder).
from motion_recognition import (
    FINGER_NAMES,
    discover_cameras,
    draw_landmarks,
    ensure_model_path,
    finger_states,
    open_camera,
    overlay_states,
    smooth_states,
)


SIM_WINDOW_TITLE = "Simulation Mode - Severus Prosthetic Control"
REAL_WINDOW_TITLE = "Severus Prosthetic Control - Arduino Connected"

# Arduino per-finger protocol: f<digit><o|c>. F1=index .. F5=thumb (see .ino).
FINGER_TO_DIGIT = {
    "index": "1",
    "middle": "2",
    "ring": "3",
    "pinky": "4",
    "thumb": "5",
}

# Substrings commonly found in the description/hwid of USB-serial / Arduino boards.
ARDUINO_HINTS = (
    "arduino",
    "ch340",
    "ch341",
    "usb serial",
    "usb-serial",
    "usb2.0-serial",
    "wchusbserial",
    "cp210",
    "ftdi",
    "silicon labs",
)


# --------------------------------------------------------------------------- #
# Serial / Arduino detection
# --------------------------------------------------------------------------- #
def find_arduino(preferred_port: str = "", baud: int = 115200):
    """Try to find and open a connected Arduino.

    Returns ``(serial_handle, device_name)`` on success or ``(None, None)`` if no
    board could be opened (which triggers Simulation Mode).
    """
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        print("[serial] pyserial not installed -> Simulation Mode. "
              "(pip install pyserial to enable real hardware control)")
        return None, None

    # Build an ordered list of candidate devices to try.
    ports = list(list_ports.comports())
    candidates: list[str] = []

    if preferred_port:
        candidates.append(preferred_port)

    for p in ports:
        if preferred_port and p.device.lower() == preferred_port.lower():
            continue
        blob = f"{p.description} {getattr(p, 'manufacturer', '') or ''} {p.hwid}".lower()
        if any(hint in blob for hint in ARDUINO_HINTS):
            candidates.append(p.device)

    # Fallback: if nothing matched the hints but exactly one port exists, try it.
    if not candidates and len(ports) == 1:
        candidates.append(ports[0].device)

    if ports:
        print("[serial] Ports seen: "
              + ", ".join(f"{p.device} ({p.description})" for p in ports))
    else:
        print("[serial] No serial ports detected.")

    for device in candidates:
        try:
            handle = serial.Serial(device, baud, timeout=1)
            time.sleep(2.0)  # let the board reset after the port opens
            print(f"[serial] Connected to Arduino on {device} @ {baud} baud.")
            print("[serial] NOTE: the controller must be calibrated (W/space/z..n/K) "
                  "for finger commands to take effect -- see the .ino header.")
            return handle, device
        except Exception as exc:  # noqa: BLE001 - any failure -> next candidate
            print(f"[serial] Could not open {device}: {exc}")

    return None, None


def send_finger_command(handle, finger: str, extended: bool) -> None:
    """Send a single per-finger command to the Arduino (f<digit><o|c>)."""
    if handle is None:
        return
    digit = FINGER_TO_DIGIT.get(finger)
    if digit is None:
        return
    cmd = f"f{digit}{'o' if extended else 'c'}"
    try:
        handle.write(cmd.encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        print(f"[serial] write failed ({cmd}): {exc}")


# --------------------------------------------------------------------------- #
# Simulated anthropomorphic hand renderer
# --------------------------------------------------------------------------- #
# The hand is drawn from the PALMAR side (palm facing the viewer). Fingers flex
# by rotating about a horizontal axis, so on retraction they curl forward (toward
# the screen) and down toward the palm -- not sideways. The thumb sits on the
# radial side and opposes up across the palm. Geometry is normalized (0..1) and
# scaled to the panel.
# Finger layout in design units (x right, y down; same scale on both axes). The
# whole hand is auto-fit into the panel every render, so these are relative.
FINGER_GEOM = {
    "index":  dict(knuckle=(0.420, 0.470), splay=-15, lengths=(0.150, 0.098, 0.064)),
    "middle": dict(knuckle=(0.530, 0.450), splay=-4,  lengths=(0.166, 0.106, 0.070)),
    "ring":   dict(knuckle=(0.632, 0.460), splay=+6,  lengths=(0.152, 0.098, 0.066)),
    "pinky":  dict(knuckle=(0.726, 0.500), splay=+17, lengths=(0.118, 0.080, 0.056)),
}
THUMB_GEOM = dict(knuckle=(0.360, 0.620), lengths=(0.140, 0.105, 0.078))

# Per-joint flexion at full retraction (MCP, PIP, DIP), in degrees.
FINGER_JOINT_BEND = (40.0, 104.0, 78.0)
THUMB_JOINT_BEND = (0.0, 44.0, 40.0)
THUMB_OPEN_ANGLE = -168.0    # extended: points to the radial (left) side, nearly flat
THUMB_CLOSE_SWING = 64.0     # degrees it swings up across the palm when closing

# Panel sizing (auto-fit keeps the whole hand visible at any camera resolution).
PANEL_ASPECT = 0.86          # hand-panel width / height
MIN_PANEL_HEIGHT = 700       # keep the simulated hand comfortably large

# --- Severus-inspired colours (BGR) ---
COL_BG_TOP = (40, 36, 34)
COL_BG_BOT = (16, 14, 13)
COL_PALM = (52, 50, 54)          # matte dark prosthetic body
COL_PALM_DARK = (30, 29, 32)
COL_PALM_EDGE = (96, 94, 100)
COL_RIDGE = (74, 72, 78)
COL_WRIST = (44, 42, 46)
COL_BASE_RING = (232, 232, 236)  # white mounting disc
COL_BASE_RIM = (150, 150, 156)

# Per-state finger colours: segments are lighter than the joints/landmarks so the
# articulation reads clearly. extended = green, retracted = red.
STATE_SEG = {True: (120, 232, 140), False: (96, 110, 240)}
STATE_JOINT = {True: (40, 120, 60), False: (36, 42, 150)}
STATE_OUTLINE = {True: (20, 70, 34), False: (18, 22, 88)}

FINGER_WIDTH_SCALE = {"thumb": 1.30, "index": 1.0, "middle": 1.06, "ring": 0.98, "pinky": 0.82}


def _rot_x(vec, theta):
    """Rotate a 3D vector about the x-axis so 'up' curls toward the viewer then down."""
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = vec
    return np.array([x, y * c + z * s, -y * s + z * c])


def _finger_points(knuckle_px, splay_deg, seg_px, extension):
    """3D joint positions for a finger; flexion curls it forward (+z) and down."""
    ext = float(np.clip(extension, 0.0, 1.0))
    curl = 1.0 - ext
    # Fingers fan out when open and converge toward the palm as they close.
    splay = math.radians(splay_deg) * (0.35 + 0.65 * ext)
    direction = np.array([math.sin(splay), -math.cos(splay), 0.0])
    p = np.array([knuckle_px[0], knuckle_px[1], 0.0], dtype=float)
    pts = [p.copy()]
    for length, bend in zip(seg_px, FINGER_JOINT_BEND):
        direction = _rot_x(direction, math.radians(bend * curl))
        p = p + length * direction
        pts.append(p.copy())
    return pts


def _thumb_points(knuckle_px, seg_px, extension):
    """3D joint positions for the thumb; it opposes up across the palm when closing."""
    ext = float(np.clip(extension, 0.0, 1.0))
    curl = 1.0 - ext
    # Extended: up and out to the radial (left) side. Closed: swings up across the
    # palm and flexes its tip forward (a little +z for depth).
    ang = math.radians(THUMB_OPEN_ANGLE) + math.radians(THUMB_CLOSE_SWING) * curl
    p = np.array([knuckle_px[0], knuckle_px[1], 0.0], dtype=float)
    pts = [p.copy()]
    for length, bend in zip(seg_px, THUMB_JOINT_BEND):
        ang += math.radians(bend * curl)
        forward = math.sin(math.radians(bend * curl))
        direction = np.array([math.cos(ang), math.sin(ang), forward])
        norm = np.linalg.norm(direction) or 1.0
        p = p + length * direction / norm
        pts.append(p.copy())
    return pts


def _gradient_bg(width, height):
    """Vertical gradient backdrop."""
    top = np.array(COL_BG_TOP, dtype=float)
    bot = np.array(COL_BG_BOT, dtype=float)
    t = np.linspace(0.0, 1.0, height)[:, None]
    column = top[None, :] * (1.0 - t) + bot[None, :] * t
    canvas = np.repeat(column[:, None, :], width, axis=1)
    return np.ascontiguousarray(canvas.astype(np.uint8))


def _widths_for(name, scale):
    """Per-segment widths (px) for a limb in fitted pixel units."""
    s = FINGER_WIDTH_SCALE[name]
    base = 0.060 * scale * s
    return [max(3, int(base)), max(3, int(base * 0.86)), max(2, int(base * 0.72))]


def _compute_limbs(display_ext):
    """All finger/thumb joint chains for a pose, in design space."""
    limbs = []
    for name, geom in FINGER_GEOM.items():
        ext = float(display_ext.get(name, 1.0))
        pts = _finger_points(geom["knuckle"], geom["splay"], geom["lengths"], ext)
        limbs.append((name, ext, pts))
    t_ext = float(display_ext.get("thumb", 1.0))
    t_pts = _thumb_points(THUMB_GEOM["knuckle"], THUMB_GEOM["lengths"], t_ext)
    limbs.append(("thumb", t_ext, t_pts))
    return limbs


# Palm / wrist silhouette in design space (x, y).
PALM_OUTLINE = [
    (0.395, 0.500), (0.430, 0.460), (0.530, 0.442), (0.632, 0.452), (0.726, 0.495),
    (0.770, 0.585), (0.762, 0.700), (0.715, 0.778), (0.660, 0.806),
    (0.452, 0.806), (0.398, 0.778), (0.360, 0.690), (0.352, 0.560),
]
WRIST_OUTLINE = [(0.452, 0.788), (0.660, 0.788), (0.638, 0.928), (0.474, 0.928)]
THENAR = ((0.372, 0.640), 0.080)
HYPOTHENAR = ((0.726, 0.640), 0.058)
BASE_RING = ((0.556, 0.952), 0.150, 0.050)  # center, half-width, half-height

_OPEN_BBOX = None


def _open_bbox():
    """Design-space bounding box of the fully-open hand (cached) -- the fit target."""
    global _OPEN_BBOX
    if _OPEN_BBOX is not None:
        return _OPEN_BBOX
    xs, ys = [], []
    for _, _, pts in _compute_limbs({n: 1.0 for n in FINGER_NAMES}):
        for p in pts:
            xs.append(float(p[0])); ys.append(float(p[1]))
    for (x, y) in PALM_OUTLINE + WRIST_OUTLINE:
        xs.append(x); ys.append(y)
    for (cx, cy), r in (THENAR, HYPOTHENAR):
        xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
    (bx, by), bw, bh = BASE_RING
    xs += [bx - bw, bx + bw]; ys += [by - bh, by + bh]
    pad = 0.06  # room for finger thickness
    _OPEN_BBOX = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    return _OPEN_BBOX


def _fit_transform(panel_w, panel_h, top_px, bottom_px):
    """Scale + offset mapping the open-hand bbox into the panel (between header/legend)."""
    xmin, ymin, xmax, ymax = _open_bbox()
    bw, bh = xmax - xmin, ymax - ymin
    side = 0.04 * panel_w
    avail_w = max(1.0, panel_w - 2 * side)
    avail_h = max(1.0, panel_h - top_px - bottom_px)
    scale = min(avail_w / bw, avail_h / bh)
    ox = (panel_w - scale * bw) / 2.0 - scale * xmin
    oy = top_px + (avail_h - scale * bh) / 2.0 - scale * ymin
    return scale, ox, oy


def _draw_palm(canvas, scale, ox, oy):
    """Draw a Severus-style matte palm + wrist + white mounting disc."""
    def D(p):
        return (int(round(ox + scale * p[0])), int(round(oy + scale * p[1])))

    def Dr(r):
        return max(1, int(round(scale * r)))

    # White mounting base disc, behind the wrist.
    (bx, by), bw, bh = BASE_RING
    cv2.ellipse(canvas, D((bx, by)), (Dr(bw), Dr(bh)), 0, 0, 360, COL_BASE_RING, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, D((bx, by)), (Dr(bw), Dr(bh)), 0, 0, 360, COL_BASE_RIM, 2, cv2.LINE_AA)

    # Wrist block.
    wrist = np.array([D(p) for p in WRIST_OUTLINE], np.int32)
    cv2.fillPoly(canvas, [wrist], COL_WRIST, cv2.LINE_AA)
    cv2.polylines(canvas, [wrist], True, COL_PALM_EDGE, 2, cv2.LINE_AA)

    # Rounded body (thenar / hypothenar mounds) before the palm fill.
    for (cx, cy), r in (THENAR, HYPOTHENAR):
        cv2.circle(canvas, D((cx, cy)), Dr(r), COL_PALM, -1, cv2.LINE_AA)

    palm = np.array([D(p) for p in PALM_OUTLINE], np.int32)
    cv2.fillPoly(canvas, [palm], COL_PALM, cv2.LINE_AA)
    cv2.polylines(canvas, [palm], True, COL_PALM_EDGE, 2, cv2.LINE_AA)

    # Grooved finger columns + dark knuckle blocks (the Severus palm look).
    for geom in FINGER_GEOM.values():
        kx, ky = geom["knuckle"]
        cv2.line(canvas, D((kx, ky + 0.02)), D((kx, ky + 0.21)), COL_RIDGE,
                 max(1, Dr(0.012)), cv2.LINE_AA)
    for geom in FINGER_GEOM.values():
        cv2.circle(canvas, D(geom["knuckle"]), Dr(0.042), COL_PALM_DARK, -1, cv2.LINE_AA)


def _draw_limb(canvas, pts2d, pts3d, widths, extended):
    """Draw a limb whose whole length is tinted by state; joints darker than segments."""
    seg = STATE_SEG[extended]
    joint = STATE_JOINT[extended]
    outline = STATE_OUTLINE[extended]
    # Outline pass for a clean silhouette.
    for i in range(len(pts2d) - 1):
        cv2.line(canvas, pts2d[i], pts2d[i + 1], outline, widths[i] + 5, cv2.LINE_AA)
    # Segment fill (lighter), with subtle forward-depth brightening.
    for i in range(len(pts2d) - 1):
        zmid = 0.5 * (pts3d[i][2] + pts3d[i + 1][2])
        shade = float(np.clip(1.0 + 0.5 * zmid, 0.82, 1.18))
        col = tuple(int(np.clip(c * shade, 0, 255)) for c in seg)
        cv2.line(canvas, pts2d[i], pts2d[i + 1], col, widths[i], cv2.LINE_AA)
    # Joints / landmarks (darker) so they read distinctly against the segments.
    for i, pt in enumerate(pts2d):
        r = max(3, int(widths[min(i, len(widths) - 1)] * 0.62))
        cv2.circle(canvas, pt, r + 2, outline, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, r, joint, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, max(1, r // 3), seg, -1, cv2.LINE_AA)


def render_hand(width: int, height: int, display_ext: dict[str, float]) -> np.ndarray:
    """Render the simulated palmar-side Severus hand, auto-fit to the panel."""
    canvas = _gradient_bg(width, height)

    header_h = max(34, int(0.052 * height))
    line_h = max(20, int(0.030 * height))
    legend_h = line_h * len(FINGER_NAMES) + 24
    scale, ox, oy = _fit_transform(width, height, header_h + 6, legend_h)

    def T(p):
        return (int(round(ox + scale * p[0])), int(round(oy + scale * p[1])))

    _draw_palm(canvas, scale, ox, oy)

    limbs = _compute_limbs(display_ext)
    # Fingers first, thumb last so it sits in front of the palm when closing.
    for name, ext, pts in limbs:
        if name == "thumb":
            continue
        _draw_limb(canvas, [T(p) for p in pts], pts, _widths_for(name, scale), ext >= 0.5)
    for name, ext, pts in limbs:
        if name == "thumb":
            _draw_limb(canvas, [T(p) for p in pts], pts, _widths_for(name, scale), ext >= 0.5)

    # Header.
    cv2.rectangle(canvas, (0, 0), (width, header_h), (30, 27, 26), -1)
    cv2.putText(canvas, "SIMULATED HAND - SEVERUS", (14, int(header_h * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (232, 232, 236), 2, cv2.LINE_AA)

    # Per-finger state legend.
    y0 = height - 14 - (len(FINGER_NAMES) - 1) * line_h
    for i, name in enumerate(FINGER_NAMES):
        is_ext = float(display_ext.get(name, 1.0)) >= 0.5
        label = f"{name}: {'EXTEND' if is_ext else 'RETRACT'}"
        cv2.putText(canvas, label, (14, y0 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, STATE_SEG[is_ext], 2, cv2.LINE_AA)

    return canvas


# --------------------------------------------------------------------------- #
# Pop-up announcement
# --------------------------------------------------------------------------- #
def announce_mode(simulation: bool, device: str | None) -> None:
    """Pop up a small GUI announcing the active mode (falls back to console)."""
    if simulation:
        title = SIM_WINDOW_TITLE
        msg = ("No Arduino detected on any serial port.\n\n"
               "Running in SIMULATION MODE.\n"
               "The camera gesture view (left) drives a drawn robotic hand (right).\n\n"
               "Press 'q' in the window to quit.")
    else:
        title = "Severus Prosthetic Control - Connected"
        msg = (f"Arduino detected on {device}.\n\n"
               "Running in HARDWARE MODE.\n"
               "Finger states are sent to Severus over serial.\n\n"
               "Press 'q' in the window to quit.")

    print(f"\n=== {title} ===\n{msg}\n")
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo(title, msg)
        root.destroy()
    except Exception:  # noqa: BLE001 - headless / no Tk: console message is enough
        pass


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Severus prosthetic control with simulated/real hand output."
    )
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (default: 0).")
    parser.add_argument("--max-hands", type=int, default=1, help="Max hands to track (default: 1).")
    parser.add_argument("--min-detection", type=float, default=0.6, help="Min detection conf.")
    parser.add_argument("--min-tracking", type=float, default=0.6, help="Min tracking conf.")
    parser.add_argument("--max-camera-index", type=int, default=10, help="Max camera index to scan.")
    parser.add_argument("--allow-virtual", action="store_true", help="Allow virtual cameras (OBS).")
    parser.add_argument("--smooth-frames", type=int, default=5, help="Frames to smooth (default: 5).")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720).")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture FPS (default: 30).")
    parser.add_argument("--raw", action="store_true", help="Do not force MJPG.")
    parser.add_argument("--model", type=str, default="", help="Path to hand_landmarker.task.")
    # Simulation / serial specific.
    parser.add_argument("--port", type=str, default="", help="Force a serial port (e.g. COM5).")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud (default: 115200).")
    parser.add_argument("--force-sim", action="store_true",
                        help="Force Simulation Mode even if an Arduino is present.")
    parser.add_argument("--no-serial", action="store_true",
                        help="Never touch the serial port (always simulate).")
    parser.add_argument("--ease", type=float, default=0.35,
                        help="Finger animation easing 0..1 (default: 0.35).")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    # --- decide mode: look for an Arduino unless told not to ---
    handle, device = (None, None)
    if not (args.force_sim or args.no_serial):
        handle, device = find_arduino(args.port, args.baud)
    simulation = handle is None
    window_title = SIM_WINDOW_TITLE if simulation else REAL_WINDOW_TITLE

    announce_mode(simulation, device)

    # --- camera setup (shared with motion_recognition.py) ---
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
    cap = open_camera(int(cameras[camera_pos]["index"]), args.width, args.height,
                      args.fps, force_mjpg=not args.raw)
    if not cap.isOpened():
        raise SystemExit("Failed to open USB camera.")

    # --- MediaPipe hand landmarker ---
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
    # Eased display values for the simulated hand (start open/extended).
    display_ext: dict[str, float] = {name: 1.0 for name in FINGER_NAMES}
    target_ext: dict[str, float] = {name: 1.0 for name in FINGER_NAMES}
    ease = float(np.clip(args.ease, 0.05, 1.0))
    last_time = time.time()

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    window_sized = False

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
            smoothed = smooth_states(state_history)
            overlay_states(frame, smoothed)

            # Update simulation targets + emit/send commands on change.
            for name in FINGER_NAMES:
                is_ext = bool(smoothed.get(name, False))
                target_ext[name] = 1.0 if is_ext else 0.0
                prev = last_states.get(name)
                if prev is not None and is_ext != prev:
                    action = "extend" if is_ext else "retract"
                    print(f"{action} {name}")
                    send_finger_command(handle, name, is_ext)
            last_states = {name: bool(smoothed.get(name, False)) for name in FINGER_NAMES}
        else:
            cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Mode banner on the camera frame.
        banner = "SIMULATION MODE" if simulation else f"ARDUINO: {device}"
        cv2.putText(frame, banner, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255) if simulation else (0, 255, 0), 2, cv2.LINE_AA)

        # FPS.
        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Ease the simulated fingers toward their targets every frame.
        for name in FINGER_NAMES:
            display_ext[name] += (target_ext[name] - display_ext[name]) * ease

        if simulation:
            # Auto-size both panels to the detected camera resolution (with a floor
            # so the simulated hand stays comfortably large for low-res cameras).
            cam_h, cam_w = frame.shape[:2]
            panel_h = max(cam_h, MIN_PANEL_HEIGHT)
            panel_w = int(panel_h * PANEL_ASPECT)
            hand_panel = render_hand(panel_w, panel_h, display_ext)
            if cam_h != panel_h:
                frame_disp = cv2.resize(frame, (int(round(cam_w * panel_h / cam_h)), panel_h))
            else:
                frame_disp = frame
            display = cv2.hconcat([frame_disp, hand_panel])
        else:
            display = frame

        # Give the window a sensible initial size on the first frame.
        if not window_sized:
            dh, dw = display.shape[:2]
            target_w = min(1700, dw)
            cv2.resizeWindow(window_title, target_w, max(1, int(dh * target_w / dw)))
            window_sized = True
        cv2.imshow(window_title, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("x"), ord("X")) and len(cameras) > 1:
            cap.release()
            camera_pos = (camera_pos + 1) % len(cameras)
            cap = open_camera(int(cameras[camera_pos]["index"]), args.width, args.height,
                              args.fps, force_mjpg=not args.raw)
            state_history.clear()
            last_states = {}
            continue
        if key == ord("q"):
            break

    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    if handle is not None:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
