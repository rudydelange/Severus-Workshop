"""BEGINNER Severus Motion Workshop  --  computer vision + prosthetic control.

Welcome! You do NOT need any programming experience to do this workshop.

WHAT YOU WILL BUILD (in about 1 hour)
  1. A camera that recognises your hand and draws its "bones" on screen.
  2. Detection of whether each finger is EXTENDED (open) or RETRACTED (curled).
  3. A simulated robotic Severus hand that copies your real hand. (If a real
     Severus hand is plugged in and calibrated, it copies you too!)

HOW THE WORKSHOP WORKS
  * Run this file. A window opens straight away -- even before you finish! At the
    start most things say "not working yet". That is normal.
  * Look at the yellow "NEXT" banner and the "WORKSHOP PROGRESS" checklist on the
    screen. They tell you which task to do next.
  * Each task is a single line below that starts with  ANSWER_1 , ANSWER_2 , ...
    You fix a task by REMOVING the '#' in front of the correct option (this is
    called "uncommenting" a line).
  * Save the file and run it again. A pop-up tells you if a line is wrong and
    gives a hint. Get the SAME line wrong 3 times and the pop-up reveals the
    answer.
  * Stuck at the very end? An "Answer Sheet" is written (commented out) at the
    bottom of this file.

To run:  press the Run button in VS Code, or in a terminal:
    python beginner_severus_motion_workshop.py
Press the 'q' key inside the camera window to quit.
"""

from __future__ import annotations

# --- STEP 0a: find the shared workshop code (do not edit) ------------------- #
# The shared tools (camera engine, checking harness, installer) live in the
# parent "Motion Recognition Workshop" folder. This adds that folder to Python's
# search path so the imports below work from inside "Beginner Workshop".
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# --- STEP 0b: make sure everything is installed (do not edit) --------------- #
# This checks that Python and all needed packages are present and, if not,
# installs them automatically. It must run before anything else.
from severus_software_installation import ensure_prerequisites

ensure_prerequisites()

# --- Tools we will use (do not edit) ---------------------------------------- #
from mediapipe.tasks.python import vision      # the computer-vision toolbox
from motion_recognition import draw_landmarks, overlay_states
from workshop_engine import parse_args, run_workshop
from workshop_harness import Task, WorkshopHarness


# =========================================================================== #
#                        YOUR TASKS START HERE                                 #
#   Fix them top to bottom. The screen always tells you which one is next.     #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# === TASK 1: Turn ON hand detection ======================================== #
# --------------------------------------------------------------------------- #
# MediaPipe is a toolbox that can detect different things in a camera image.
# We need the tool that finds the 21 "bony landmarks" (knuckles & fingertips)
# of a HAND. Choose the right tool by deleting the '#' in front of ONE line.
#
#   vision.PoseLandmarker    -> finds your WHOLE BODY (shoulders, hips...). No good for fingers.
#   vision.GestureRecognizer -> guesses named gestures (thumbs-up). Not the raw points we need.
#   vision.HandLandmarker    -> finds the 21 points of a HAND.   <-- this is the one we want!
#
# Leave the "= None" line as-is; just uncomment your choice BELOW it (the lower
# line wins). If you pick the wrong one, nothing breaks -- you just get a hint.
ANSWER_1 = None
# ANSWER_1 = vision.PoseLandmarker
# ANSWER_1 = vision.GestureRecognizer
# ANSWER_1 = vision.HandLandmarker


# --------------------------------------------------------------------------- #
# === TASK 2: Draw the bony landmarks on screen ============================= #
# --------------------------------------------------------------------------- #
# Now that the hand is detected, we want to SEE it: green dots on every joint and
# blue lines for the "bones". There are two ready-made drawing tools:
#
#   overlay_states  -> writes the finger words ("extended"/"retracted") as text.
#   draw_landmarks  -> draws the dots and bone-lines ON your hand.  <-- we want this one
#
# Uncomment the tool that draws the dots and bones.
ANSWER_2 = None
# ANSWER_2 = overlay_states
# ANSWER_2 = draw_landmarks


# --------------------------------------------------------------------------- #
# === TASK 3: Decide when a FINGER is extended ============================== #
# --------------------------------------------------------------------------- #
# For the four fingers (index, middle, ring, pinky) we look at two points:
#     the FINGERTIP   and   the KNUCKLE below it.
# The camera image measures position from the TOP: y = 0 is the top of the
# picture and y gets BIGGER as you go DOWN. So a point that is HIGHER on screen
# has a SMALLER y.
#
# When a finger is EXTENDED (pointing up), its fingertip is ABOVE the knuckle,
# so the fingertip's y is SMALLER than the knuckle's y.
#
# Fill in the comparison: "fingertip is extended when  tip_y  ???  knuckle_y".
# Uncomment "<" (smaller / higher up) or ">" (bigger / lower down).
ANSWER_3 = None
# ANSWER_3 = ">"
# ANSWER_3 = "<"


# --------------------------------------------------------------------------- #
# === TASK 4: Decide when the THUMB is extended ============================= #
# --------------------------------------------------------------------------- #
# The thumb does not point up/down like the fingers -- it moves SIDEWAYS. So for
# the thumb we compare LEFT/RIGHT position (the x value) instead of up/down.
# x = 0 is the LEFT edge and x gets BIGGER toward the RIGHT.
#
# For a RIGHT hand held up palm-towards-you, when the thumb is EXTENDED (sticking
# out to the side) its tip is further to the RIGHT than its lower joint, so the
# tip's x is BIGGER. (The program flips this automatically for a left hand.)
#
# Uncomment ">" (tip further right) or "<" (tip further left).
ANSWER_4 = None
# ANSWER_4 = "<"
# ANSWER_4 = ">"


# --------------------------------------------------------------------------- #
# === TASK 5: Make the simulated hand copy you ============================== #
# --------------------------------------------------------------------------- #
# The drawn robotic hand uses a number for each finger:
#     1.0 = fully OPEN (extended)        0.0 = fully CLOSED (retracted)
# We need to tell it: "when MY finger is EXTENDED, set the robot finger to ___".
#
# Uncomment the value that means fully OPEN.
ANSWER_5 = None
# ANSWER_5 = 0.0
# ANSWER_5 = 1.0


# =========================================================================== #
#                         YOUR TASKS END HERE                                  #
#        Everything below is the "engine". You do not need to edit it.         #
# =========================================================================== #

# Each Task tells the workshop how to check your answer and what hint to show.
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
    harness = WorkshopHarness("Beginner Severus Workshop", __file__, TASKS)
    status = harness.review(answers)   # pops up guidance for the next task
    run_workshop(args, harness, status, answers)


if __name__ == "__main__":
    main()


# =========================================================================== #
# Answer Sheet - First try yourself - If it does not work, use this as a reference!
# =========================================================================== #
# Below are the correct lines for every task. Try to solve them on your own
# first -- you learn far more that way! If you get truly stuck, copy the matching
# line up into its task above (and add a '#' in front of the "= None" line, or
# just leave it: the lower, uncommented line always wins).
#
#   TASK 1:  ANSWER_1 = vision.HandLandmarker
#       Why: HandLandmarker is the tool that returns the 21 hand points.
#
#   TASK 2:  ANSWER_2 = draw_landmarks
#       Why: draw_landmarks paints the green dots and blue bone-lines on the hand
#            (overlay_states would only write text).
#
#   TASK 3:  ANSWER_3 = "<"
#       Why: the image y-axis points DOWN, so "higher up" means a SMALLER y. An
#            extended finger has its tip above the knuckle, so tip_y < knuckle_y.
#
#   TASK 4:  ANSWER_4 = ">"
#       Why: the thumb moves sideways. For a right hand, an extended thumb's tip
#            is further to the RIGHT (bigger x) than its lower joint. The program
#            mirrors this automatically for a left hand.
#
#   TASK 5:  ANSWER_5 = 1.0
#       Why: the simulated finger uses 1.0 = fully open (extended) and
#            0.0 = fully closed. Your extended finger should map to 1.0.
# =========================================================================== #
