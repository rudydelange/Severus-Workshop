"""INTERMEDIATE Severus Motion Workshop  --  computer vision + prosthetic control.

For participants who can already read/write a little Python. You will wire up a
full pipeline: MediaPipe hand-landmark detection -> per-finger extend/retract
classification -> a simulated (and optionally real) Severus prosthetic hand.

WHAT IS DIFFERENT FROM THE BEGINNER VERSION
  * You WRITE each answer yourself instead of uncommenting a choice. Replace the
    `None` placeholder on each `ANSWER_x = None` line with the correct value.
  * Each task has a TIP with enough detail that you should not need to search the
    web -- but you may, of course.
  * There is no printed answer sheet. The on-screen pop-up gives a hint every
    time, and reveals the exact answer after you get the SAME line wrong 3 times.

HOW IT RUNS
  The camera window opens immediately and degrades gracefully: unsolved features
  are simply skipped and labelled "not working yet" on screen, so you always have
  a running program to iterate against. Follow the yellow NEXT banner / progress
  checklist. Press 'q' in the window to quit.

Run:  python intermediate_severus_motion_workshop.py
"""

from __future__ import annotations

# --- Path bootstrap (do not edit) ------------------------------------------- #
# Shared workshop modules live in the parent "Motion Recognition Workshop"
# folder; add it to sys.path so the imports resolve from this subfolder.
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# --- Environment bootstrap (do not edit) ------------------------------------ #
from severus_software_installation import ensure_prerequisites

ensure_prerequisites()

# --- Imports available to your answers (do not edit) ------------------------ #
from mediapipe.tasks.python import vision      # MediaPipe vision tasks
from motion_recognition import draw_landmarks, overlay_states
from workshop_engine import parse_args, run_workshop
from workshop_harness import Task, WorkshopHarness


# =========================================================================== #
#                      FILL IN EACH ANSWER (replace None)                      #
#   Order matters: the program guides you through them one at a time.          #
# =========================================================================== #

# --- TASK 1: enable hand-landmark detection -------------------------------- #
# TIP: MediaPipe exposes several vision "task" classes on the `vision` module
#      you imported. You want the one that returns the 21 hand keypoints in VIDEO
#      mode (not body pose, not a gesture *classifier*). The engine will call
#      `ANSWER_1.create_from_options(HandLandmarkerOptions(...))`, so ANSWER_1
#      must be the matching task CLASS itself (no parentheses). Think: Hand +
#      Landmarker.
ANSWER_1 = None

# --- TASK 2: draw the bony landmarks --------------------------------------- #
# TIP: Assign the already-imported FUNCTION (a reference, no parentheses) that
#      renders the 21 dots and their connecting "bones" onto a BGR frame with the
#      signature (frame, hand_landmarks). One of your two imports does exactly
#      that; the other only writes the finger-state TEXT.
ANSWER_2 = None

# --- TASK 3: classify the four fingers ------------------------------------- #
# TIP: For index/middle/ring/pinky the engine evaluates `tip_y  <op>  pip_y`,
#      where <op> is the STRING you provide here ("<" or ">"). Remember image
#      coordinates: the y-axis points DOWN (top = 0), so a finger that is
#      extended (pointing up) has its TIP higher than its PIP knuckle, i.e. a
#      smaller y. Choose the operator that makes "extended" come out True.
ANSWER_3 = None

# --- TASK 4: classify the thumb -------------------------------------------- #
# TIP: The thumb abducts SIDEWAYS, so it is classified on x, not y. The engine
#      evaluates the right-hand case as `tip_x <op> ip_x` and mirrors it for a
#      left hand automatically. For a right hand (palm to camera), an extended
#      thumb's tip sits further toward +x than its IP joint. Provide "<" or ">".
ANSWER_4 = None

# --- TASK 5: map states onto the simulated hand ---------------------------- #
# TIP: The renderer takes a per-finger float in [0.0, 1.0], where the value is
#      the "openness": the engine sets target = ANSWER_5 when your finger is
#      EXTENDED and (1.0 - ANSWER_5) when it is retracted. Provide the float that
#      represents a fully EXTENDED/open finger.
ANSWER_5 = None


# =========================================================================== #
#            Engine wiring below -- you do not need to edit this.              #
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
    harness = WorkshopHarness("Intermediate Severus Workshop", __file__, TASKS)
    status = harness.review(answers)
    run_workshop(args, harness, status, answers)


if __name__ == "__main__":
    main()
