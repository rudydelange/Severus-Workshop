"""INTERMEDIATE Severus Motion Workshop  --  *** ANSWERED / SOLUTION version ***

This is the COMPLETED version of ``intermediate_severus_motion_workshop.py``.
Each ``ANSWER_x = None`` placeholder has been replaced with the correct value,
with a short note on what the value does. Use it to compare against the blank
version (a diff will show only the five changed ANSWER lines) or to unblock a
stuck participant.

There is nothing to fill in here -- this version runs fully working.
"""

from __future__ import annotations

# --- Path bootstrap (do not edit) ------------------------------------------- #
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# --- Environment bootstrap (do not edit) ------------------------------------ #
from severus_software_installation import ensure_prerequisites

ensure_prerequisites()

# --- Imports available to your answers (do not edit) ------------------------ #
from mediapipe.tasks.python import vision
from motion_recognition import draw_landmarks, overlay_states
from workshop_engine import parse_args, run_workshop
from workshop_harness import Task, WorkshopHarness


# =========================================================================== #
#                       ANSWERS (all filled in correctly)                      #
# =========================================================================== #

# TASK 1 -- enable hand-landmark detection.
# The vision task CLASS (no parentheses) that returns 21 hand keypoints in VIDEO
# mode. The engine calls ANSWER_1.create_from_options(HandLandmarkerOptions(...)).
ANSWER_1 = vision.HandLandmarker

# TASK 2 -- draw the bony landmarks.
# Function reference (no parentheses) drawing 21 dots + skeleton on a BGR frame
# with signature (frame, hand_landmarks). overlay_states would only write text.
ANSWER_2 = draw_landmarks

# TASK 3 -- classify the four fingers.
# Operator string for `tip_y <op> pip_y`. Image y grows downward, so an extended
# (raised) finger has the SMALLER tip y -> use "<".
ANSWER_3 = "<"

# TASK 4 -- classify the thumb.
# Operator string for the right-hand case `tip_x <op> ip_x` (mirrored for left).
# An abducted/extended right thumb has tip x GREATER than ip x -> use ">".
ANSWER_4 = ">"

# TASK 5 -- map states onto the simulated hand.
# Openness float in [0.0, 1.0]; the engine uses ANSWER_5 for an EXTENDED finger
# and (1.0 - ANSWER_5) for a retracted one. Fully extended/open = 1.0.
ANSWER_5 = 1.0


# =========================================================================== #
#                   Engine wiring (identical to the blank file)                #
# =========================================================================== #
TASKS = [
    Task(
        key=1, name="Hand detection", var_name="ANSWER_1",
        check=lambda v: v is vision.HandLandmarker,
        hint="The vision task class that returns 21 HAND landmarks (Hand+Landmarker).",
        answer_text="ANSWER_1 = vision.HandLandmarker",
    ),
    Task(
        key=2, name="Draw landmarks", var_name="ANSWER_2",
        check=lambda v: v is draw_landmarks,
        hint="A function reference (no parens) that draws dots+bones: draw_landmarks vs overlay_states.",
        answer_text="ANSWER_2 = draw_landmarks",
    ),
    Task(
        key=3, name="Finger extend/retract", var_name="ANSWER_3",
        check=lambda v: v == "<",
        hint="Image y grows downward; an extended fingertip has the smaller y.",
        answer_text='ANSWER_3 = "<"',
    ),
    Task(
        key=4, name="Thumb extend/retract", var_name="ANSWER_4",
        check=lambda v: v == ">",
        hint="Right-hand extended thumb: tip_x is greater than ip_x.",
        answer_text='ANSWER_4 = ">"',
    ),
    Task(
        key=5, name="Drive the simulated hand", var_name="ANSWER_5",
        check=lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) == 1.0,
        hint="The openness float for a fully EXTENDED finger, in [0.0, 1.0].",
        answer_text="ANSWER_5 = 1.0",
    ),
]


def main() -> None:
    args = parse_args()
    answers = {1: ANSWER_1, 2: ANSWER_2, 3: ANSWER_3, 4: ANSWER_4, 5: ANSWER_5}
    harness = WorkshopHarness("Intermediate Severus Workshop (ANSWERED)", __file__, TASKS)
    status = harness.review(answers)
    run_workshop(args, harness, status, answers)


if __name__ == "__main__":
    main()
