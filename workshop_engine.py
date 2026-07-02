"""Run loop shared by both Severus motion workshops.

The participant edits only the ``ANSWER_x`` values in their workshop file. This
engine takes those answers plus the per-task ``status`` (from the harness) and
runs the live camera loop, switching each feature on only when its task is
solved. Everything degrades gracefully: with nothing solved you still get a
camera window and a drawn hand; as answers become correct, landmarks appear,
then finger states, then the simulated (and optionally real) hand moves.

Participants do not need to read this file -- but it is intentionally short and
commented so a curious one can.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from motion_recognition import (
    FINGER_NAMES,
    discover_cameras,
    ensure_model_path,
    open_camera,
)
from motion_recognition_simulation import (
    MIN_PANEL_HEIGHT,
    PANEL_ASPECT,
    find_arduino,
    render_hand,
    send_finger_command,
)

WINDOW_TITLE = "Severus Motion Workshop"

# MediaPipe hand landmark indices: (fingertip, lower knuckle/PIP) per finger.
FINGER_LM = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}
THUMB_LM = (4, 3)  # (tip, IP joint)


# --------------------------------------------------------------------------- #
# Args (kept small; sensible defaults so beginners can just run it)
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Severus motion workshop.")
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (default: 0).")
    parser.add_argument("--max-hands", type=int, default=1, help="Max hands to track.")
    parser.add_argument("--min-detection", type=float, default=0.6, help="Min detection conf.")
    parser.add_argument("--min-tracking", type=float, default=0.6, help="Min tracking conf.")
    parser.add_argument("--max-camera-index", type=int, default=10, help="Max camera index to scan.")
    parser.add_argument("--allow-virtual", action="store_true", help="Allow virtual cameras (OBS).")
    parser.add_argument("--smooth-frames", type=int, default=5, help="Frames to smooth (default: 5).")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture FPS.")
    parser.add_argument("--raw", action="store_true", help="Do not force MJPG.")
    parser.add_argument("--model", type=str, default="", help="Path to hand_landmarker.task.")
    parser.add_argument("--port", type=str, default="", help="Force a serial port (e.g. COM5).")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud (default: 115200).")
    parser.add_argument("--force-sim", action="store_true", help="Never use the real hand.")
    parser.add_argument("--ease", type=float, default=0.35, help="Simulated-finger easing 0..1.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Finger-state maths driven by the participant's operators
# --------------------------------------------------------------------------- #
def _apply_y_op(tip_y: float, pip_y: float, op: Any) -> bool | None:
    """Is the finger extended? op '<' means tip is ABOVE the knuckle (smaller y)."""
    if op == "<":
        return tip_y < pip_y
    if op == ">":
        return tip_y > pip_y
    return None


def _thumb_extended(lm, handedness: str, op: Any) -> bool | None:
    """Thumb uses x (it moves sideways). op describes the RIGHT hand; mirror for left."""
    if op not in ("<", ">"):
        return None
    tip, ip = THUMB_LM
    right = handedness.lower() == "right"
    greater = lm[tip].x > lm[ip].x
    if op == ">":
        return greater if right else (not greater)
    return (not greater) if right else greater


def _compute_states(hand_landmarks, handedness: str, status: dict[int, bool],
                    answers: dict[int, Any]) -> dict[str, bool]:
    """Per-finger extended/retracted using only the tasks that are solved."""
    states: dict[str, bool] = {}
    if status[3]:  # the four fingers
        for name, (tip, pip) in FINGER_LM.items():
            val = _apply_y_op(hand_landmarks[tip].y, hand_landmarks[pip].y, answers[3])
            if val is not None:
                states[name] = val
    if status[4]:  # the thumb
        val = _thumb_extended(hand_landmarks, handedness, answers[4])
        if val is not None:
            states["thumb"] = val
    return states


def _smooth(history: deque, known: set[str]) -> dict[str, bool]:
    """Majority-vote smoothing per known finger (reduces flicker)."""
    if not history:
        return {}
    out: dict[str, bool] = {}
    for name in known:
        votes = [h[name] for h in history if name in h]
        if votes:
            out[name] = sum(votes) >= (len(votes) // 2 + 1)
    return out


def _draw_finger_text(frame, states: dict[str, bool], known: set[str]) -> None:
    """Top-left per-finger readout; '???' for fingers whose task isn't solved yet."""
    y0, dy = 30, 28
    for i, name in enumerate(FINGER_NAMES):
        if name in known and name in states:
            label = f"{name}: {'extended' if states[name] else 'retracted'}"
            color = (0, 255, 0) if states[name] else (0, 0, 255)
        else:
            label = f"{name}: ??? (not detected yet)"
            color = (170, 170, 170)
        cv2.putText(frame, label, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run_workshop(args, harness, status: dict[int, bool], answers: dict[int, Any]) -> None:
    """Run the live workshop loop with whatever features are currently solved."""
    # Optional real hand: only used once finger states work (Task 5). The board
    # must already be flashed + calibrated via severus_motion_bridge for commands
    # to take physical effect; otherwise this is harmless.
    handle, device = (None, None)
    if not args.force_sim:
        handle, device = find_arduino(args.port, args.baud)

    cameras = discover_cameras(args.max_camera_index, args.allow_virtual)
    if not cameras:
        cameras = [{"index": args.camera, "name": ""}]
    else:
        requested = [c for c in cameras if int(c["index"]) == args.camera]
        if requested:
            cameras = requested + [c for c in cameras if int(c["index"]) != args.camera]
    print("Available cameras:")
    for cam in cameras:
        print(f"  index {cam['index']}: {cam['name'] or '<unnamed>'}")

    camera_pos = 0
    cap = open_camera(int(cameras[camera_pos]["index"]), args.width, args.height,
                      args.fps, force_mjpg=not args.raw)
    if not cap.isOpened():
        raise SystemExit("Failed to open USB camera.")

    # Task 1: build the hand landmarker only if the correct class was chosen.
    hand_landmarker = None
    if status[1]:
        model_path = ensure_model_path(args.model)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=args.max_hands,
            min_hand_detection_confidence=args.min_detection,
            min_tracking_confidence=args.min_tracking,
            min_hand_presence_confidence=args.min_detection,
        )
        hand_landmarker = answers[1].create_from_options(options)

    display_ext = {name: 1.0 for name in FINGER_NAMES}
    target_ext = {name: 1.0 for name in FINGER_NAMES}
    last_states: dict[str, bool] = {}
    state_history: deque = deque(maxlen=max(args.smooth_frames, 1))
    ease = float(np.clip(args.ease, 0.05, 1.0))
    last_time = time.time()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    window_sized = False

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame.")
            break
        frame = cv2.flip(frame, 1)

        smoothed: dict[str, bool] = {}
        known: set[str] = set()

        results = None
        if hand_landmarker is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = hand_landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if results and results.hand_landmarks and results.handedness:
            hand_landmarks = results.hand_landmarks[0]
            handedness_label = "Right"
            try:
                handedness_label = results.handedness[0][0].category_name
            except (AttributeError, IndexError):
                pass

            # Task 2: draw the bony landmarks (the chosen draw function).
            if status[2]:
                answers[2](frame, hand_landmarks)

            # Tasks 3 & 4: per-finger extend/retract from the participant's rules.
            raw = _compute_states(hand_landmarks, handedness_label, status, answers)
            known = set(raw.keys())
            if raw:
                state_history.append(raw)
                smoothed = _smooth(state_history, known)
        elif hand_landmarker is None:
            cv2.putText(frame, "Hand detection OFF - solve Task 1", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        _draw_finger_text(frame, smoothed, known)

        # Task 5: connect finger states to the simulated (and real) hand.
        if status[5] and smoothed:
            ext_val = float(answers[5])
            for name in FINGER_NAMES:
                if name in smoothed:
                    is_ext = smoothed[name]
                    target_ext[name] = ext_val if is_ext else (1.0 - ext_val)
                    prev = last_states.get(name)
                    if handle is not None and prev is not None and is_ext != prev:
                        send_finger_command(handle, name, is_ext)
            last_states = dict(smoothed)

        # Ease the drawn fingers toward their targets every frame.
        for name in FINGER_NAMES:
            display_ext[name] += (target_ext[name] - display_ext[name]) * ease

        # Banners + workshop overlays (drawn on the camera frame).
        banner = f"REAL HAND: {device}" if (handle is not None) else "SIMULATION MODE"
        cv2.putText(frame, banner, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0) if handle is not None else (0, 255, 255), 2, cv2.LINE_AA)
        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        harness.draw_status_panel(frame, status)
        harness.draw_todo_banner(frame, status)

        # Right-hand panel: the simulated Severus hand.
        cam_h, cam_w = frame.shape[:2]
        panel_h = max(cam_h, MIN_PANEL_HEIGHT)
        panel_w = int(panel_h * PANEL_ASPECT)
        hand_panel = render_hand(panel_w, panel_h, display_ext)
        if cam_h != panel_h:
            frame_disp = cv2.resize(frame, (int(round(cam_w * panel_h / cam_h)), panel_h))
        else:
            frame_disp = frame
        display = cv2.hconcat([frame_disp, hand_panel])

        if not window_sized:
            dh, dw = display.shape[:2]
            target_w = min(1700, dw)
            cv2.resizeWindow(WINDOW_TITLE, target_w, max(1, int(dh * target_w / dw)))
            window_sized = True
        cv2.imshow(WINDOW_TITLE, display)

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

    if hand_landmarker is not None:
        hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    if handle is not None:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass
