"""BEGINNER Severus Motion Workshop  --  *** ANSWERED / SOLUTION version ***

This is the COMPLETED version of ``beginner_severus_motion_workshop.py``.
Every ANSWER line is filled in correctly so you (the workshop leader) or a stuck
participant can compare it against the blank version and see exactly which line
was changed and WHY.

How to compare:
  * Open this file next to ``beginner_severus_motion_workshop.py``.
  * In the blank file every task starts with ``ANSWER_x = None`` and the correct
    choice is COMMENTED OUT below it.
  * In this file the correct choice has been UN-commented (the leading '#' was
    removed). Because Python uses the LAST assignment, that uncommented line is
    the one that takes effect -- the earlier ``= None`` placeholder is ignored.

There is nothing to do in this file -- it already runs fully working.
"""

from __future__ import annotations

# --- STEP 0a: find the shared workshop code (do not edit) ------------------- #
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# --- STEP 0b: make sure everything is installed (do not edit) --------------- #
from severus_software_installation import ensure_prerequisites

ensure_prerequisites()

# --- Tools we will use (do not edit) ---------------------------------------- #
from mediapipe.tasks.python import vision
from motion_recognition import draw_landmarks, overlay_states
from workshop_engine import parse_args, run_workshop
from workshop_harness import Task, WorkshopHarness


# =========================================================================== #
#                    ANSWERS (all solved) -- see notes per task                #
# =========================================================================== #

# === TASK 1: Turn ON hand detection =======================================
# ACTIVATED: HandLandmarker -- the MediaPipe tool that returns the 21 hand
# keypoints. (PoseLandmarker = whole body, GestureRecognizer = named gestures;
# both would leave hand detection switched off.)
ANSWER_1 = None
# ANSWER_1 = vision.PoseLandmarker
# ANSWER_1 = vision.GestureRecognizer
ANSWER_1 = vision.HandLandmarker      # <-- uncommented: this line now takes effect

# === TASK 2: Draw the bony landmarks on screen ============================
# ACTIVATED: draw_landmarks -- paints the green dots + blue "bone" lines onto the
# hand. (overlay_states would only write the finger words as text.)
ANSWER_2 = None
# ANSWER_2 = overlay_states
ANSWER_2 = draw_landmarks              # <-- uncommented

# === TASK 3: Decide when a FINGER is extended =============================
# ACTIVATED: "<" -- the image y-axis points DOWN, so "higher up" is a SMALLER y.
# An extended finger's tip is above its knuckle, hence  tip_y < knuckle_y.
ANSWER_3 = None
# ANSWER_3 = ">"
ANSWER_3 = "<"                          # <-- uncommented

# === TASK 4: Decide when the THUMB is extended ============================
# ACTIVATED: ">" -- the thumb moves sideways (x). For a right hand an extended
# thumb's tip is further RIGHT (bigger x) than its lower joint. The program
# mirrors this automatically for a left hand.
ANSWER_4 = None
# ANSWER_4 = "<"
ANSWER_4 = ">"                          # <-- uncommented

# === TASK 5: Make the simulated hand copy you ============================
# ACTIVATED: 1.0 -- the simulated finger value is its "openness": 1.0 = fully
# open (extended), 0.0 = fully closed. An extended finger maps to 1.0.
ANSWER_5 = None
# ANSWER_5 = 0.0
ANSWER_5 = 1.0                          # <-- uncommented


# =========================================================================== #
#                   Engine wiring (identical to the blank file)                #
# =========================================================================== #
TASKS = [
    Task(
        key=1, name="Hand detection", var_name="ANSWER_1",
        check=lambda v: v is vision.HandLandmarker,
        hint="We want to find HAND points, not the body or a named gesture.",
        answer_text="ANSWER_1 = vision.HandLandmarker",
    ),
    Task(
        key=2, name="Draw landmarks", var_name="ANSWER_2",
        check=lambda v: v is draw_landmarks,
        hint="One tool writes text, the other draws dots & bone-lines on the hand.",
        answer_text="ANSWER_2 = draw_landmarks",
    ),
    Task(
        key=3, name="Finger extend/retract", var_name="ANSWER_3",
        check=lambda v: v == "<",
        hint="Higher on screen = SMALLER y. An extended finger's tip is higher.",
        answer_text='ANSWER_3 = "<"',
    ),
    Task(
        key=4, name="Thumb extend/retract", var_name="ANSWER_4",
        check=lambda v: v == ">",
        hint="Right-hand thumb sticking out: its tip is further RIGHT = bigger x.",
        answer_text='ANSWER_4 = ">"',
    ),
    Task(
        key=5, name="Drive the simulated hand", var_name="ANSWER_5",
        check=lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) == 1.0,
        hint="Which number did the comments say means fully OPEN?",
        answer_text="ANSWER_5 = 1.0",
    ),
]


def main() -> None:
    args = parse_args()
    answers = {1: ANSWER_1, 2: ANSWER_2, 3: ANSWER_3, 4: ANSWER_4, 5: ANSWER_5}
    harness = WorkshopHarness("Beginner Severus Workshop (ANSWERED)", __file__, TASKS)
    status = harness.review(answers)
    run_workshop(args, harness, status, answers)


if __name__ == "__main__":
    main()
