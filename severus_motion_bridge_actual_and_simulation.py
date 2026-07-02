"""Auto-detecting Severus bridge: real hardware control OR drawn-hand simulation.

This is a single entry point that picks its behaviour automatically at start-up
based on whether a Severus Arduino is plugged in:

* **An Arduino is found on a COM port** -> run the *exact* hardware bridge from
  ``severus_motion_bridge.py``: the CALIBRATION -> RUNTIME state machine, the
  on-screen HUD, ``u`` to arm, automatic finger mimic, ``o``/``p`` open/close,
  ``w`` to recalibrate, etc. Commands go to the Arduino over serial, driving
  ``260612_Severus_5FingerControl_v3.ino``.

* **No Arduino is found** (or ``--force-sim``) -> run the *exact* simulation from
  ``motion_recognition_simulation.py``: the camera/gesture view on the left and a
  drawn anthropomorphic Severus hand on the right that mirrors every finger in
  real time, with eased animation.

Neither behaviour is re-implemented here -- the unique pieces are imported from
the two sibling modules so they stay byte-for-byte identical to those scripts
(and stay in sync if you edit them). This file only adds the auto-detect step
and dispatches to the matching run loop.

Requirements:
    pip install opencv-python mediapipe numpy pyserial

Usage:
    python severus_motion_bridge_actual_and_simulation.py            # auto-detect
    python severus_motion_bridge_actual_and_simulation.py --force-sim  # always simulate
    python severus_motion_bridge_actual_and_simulation.py --port COM5  # bias detection
"""

from __future__ import annotations

# --- Prerequisite preamble -------------------------------------------------- #
# Runs BEFORE any third-party import so a missing package can be auto-installed
# (into a .venv, built from Python 3.12 if needed) instead of crashing the
# program. May re-launch this script inside that venv and exit the current
# process; control only returns here once every dependency below is importable.
# See severus_software_installation.py for details.
from severus_software_installation import ensure_prerequisites

ensure_prerequisites()

import argparse
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Shared camera / MediaPipe / finger-state pipeline (same folder) ---------
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

# --- Simulation visuals + serial probe (drawn-hand mode) ---------------------
from motion_recognition_simulation import (
    MIN_PANEL_HEIGHT,
    PANEL_ASPECT,
    SIM_WINDOW_TITLE,
    announce_mode,
    find_arduino,
    render_hand,
    send_finger_command,
)

# --- Hardware bridge state machine + HUD (real-control mode) -----------------
from severus_motion_bridge import (
    GESTURE_ORDER,
    SerialBridge,
    build_finger,
    build_gesture,
    draw_hud,
)


HARDWARE_WINDOW_TITLE = "Severus Motion Bridge"


# --------------------------------------------------------------------------- #
# Args (superset of both sibling scripts)
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-detecting Severus bridge: real Arduino control or drawn-hand simulation."
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
    # Serial / mode selection.
    parser.add_argument("--port", type=str, default="", help="Bias/force a serial port (e.g. COM5).")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud (default: 115200).")
    parser.add_argument("--force-sim", action="store_true",
                        help="Force drawn-hand Simulation Mode even if an Arduino is present.")
    parser.add_argument("--no-serial", action="store_true",
                        help="Never touch the serial port (alias for --force-sim).")
    # Simulation only.
    parser.add_argument("--ease", type=float, default=0.35,
                        help="Finger animation easing 0..1 for the simulated hand (default: 0.35).")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Shared camera + landmarker setup
# --------------------------------------------------------------------------- #
def _setup_cameras(args: argparse.Namespace):
    """Discover cameras (honoring --camera) and print them; returns the ordered list."""
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
    return cameras


def _make_landmarker(args: argparse.Namespace):
    """Create the MediaPipe hand landmarker (identical config across both modes)."""
    model_path = ensure_model_path(args.model)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        min_hand_presence_confidence=args.min_detection,
    )
    return vision.HandLandmarker.create_from_options(options)


# --------------------------------------------------------------------------- #
# SIMULATION MODE -- mirrors motion_recognition_simulation.main()
# --------------------------------------------------------------------------- #
def run_simulation(args: argparse.Namespace) -> None:
    """Drawn-hand simulation: camera/gesture view (left) drives a robotic hand (right)."""
    window_title = SIM_WINDOW_TITLE

    cameras = _setup_cameras(args)
    camera_pos = 0
    cap = open_camera(int(cameras[camera_pos]["index"]), args.width, args.height,
                      args.fps, force_mjpg=not args.raw)
    if not cap.isOpened():
        raise SystemExit("Failed to open USB camera.")

    hand_landmarker = _make_landmarker(args)

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

            # Update simulation targets + emit on change (no hardware in this mode).
            for name in FINGER_NAMES:
                is_ext = bool(smoothed.get(name, False))
                target_ext[name] = 1.0 if is_ext else 0.0
                prev = last_states.get(name)
                if prev is not None and is_ext != prev:
                    action = "extend" if is_ext else "retract"
                    print(f"{action} {name}")
                    send_finger_command(None, name, is_ext)
            last_states = {name: bool(smoothed.get(name, False)) for name in FINGER_NAMES}
        else:
            cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Mode banner on the camera frame.
        cv2.putText(frame, "SIMULATION MODE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # FPS.
        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Ease the simulated fingers toward their targets every frame.
        for name in FINGER_NAMES:
            display_ext[name] += (target_ext[name] - display_ext[name]) * ease

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


# --------------------------------------------------------------------------- #
# HARDWARE MODE -- mirrors severus_motion_bridge.main()
# --------------------------------------------------------------------------- #
def run_hardware(args: argparse.Namespace, port: str) -> None:
    """Real Severus control: calibration -> arm -> automatic finger mimic, with HUD."""
    bridge = SerialBridge(port, args.baud, no_serial=False)

    cameras = _setup_cameras(args)
    camera_pos = 0
    cap = open_camera(int(cameras[camera_pos]["index"]), args.width, args.height,
                      args.fps, force_mjpg=not args.raw)
    if not cap.isOpened():
        bridge.close()
        raise SystemExit("Failed to open USB camera.")

    hand_landmarker = _make_landmarker(args)

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

        cv2.imshow(HARDWARE_WINDOW_TITLE, frame)
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


# --------------------------------------------------------------------------- #
# Main -- auto-detect and dispatch
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    # --- Probe for an Arduino on any COM port unless told to simulate ---
    device: str | None = None
    if not (args.force_sim or args.no_serial):
        handle, device = find_arduino(args.port, args.baud)
        # We only needed the probe to learn whether hardware exists + which port.
        # Release it so the hardware bridge can open the port cleanly.
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)  # let the OS release the port before reopening

    simulation = device is None
    announce_mode(simulation, device)

    if simulation:
        run_simulation(args)
    else:
        run_hardware(args, device)


if __name__ == "__main__":
    main()
