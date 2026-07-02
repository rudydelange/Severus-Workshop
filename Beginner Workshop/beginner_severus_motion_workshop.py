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
from motion_recognition import (
    draw_landmarks,      # draws the dots + bone-lines (Task 2 answer)
    emit_commands,       # PRINTS "extend/retract" text (a Task 2 distractor)
    finger_states,       # COMPUTES extended/retracted (a Task 2 distractor)
    overlay_states,      # writes the finger words as text (a Task 2 distractor)
)
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
#   vision.FaceLandmarker    -> finds points on your FACE. Wrong body part.
#   vision.ObjectDetector    -> draws boxes around objects (cup, phone...). Not a hand skeleton.
#   vision.HandLandmarker    -> finds the 21 points of a HAND.   <-- this is the one we want!
#
# Leave the "= None" line as-is; just uncomment your choice BELOW it (the lower
# line wins). If you pick the wrong one, nothing breaks -- you just get a hint.
#
# WHERE YOUR ANSWER PLUGS IN (for the curious):
#   * The engine turns your choice into the real detector in
#     workshop_engine.py -> run_workshop(), around line 184:
#         hand_landmarker = answers[1].create_from_options(options)
#     (the options right above it, lines ~176-183, ask for the 21 hand points).
#   * The finished ("grown-up") version of the same line lives in the full
#     pipeline file severus_motion_bridge_actual_and_simulation.py ->
#     _make_landmarker() (around line 141), where vision.HandLandmarker is just
#     written in directly.
ANSWER_1 = None
# ANSWER_1 = vision.PoseLandmarker
# ANSWER_1 = vision.GestureRecognizer
# ANSWER_1 = vision.FaceLandmarker
# ANSWER_1 = vision.ObjectDetector
# ANSWER_1 = vision.HandLandmarker


# --------------------------------------------------------------------------- #
# === TASK 2: Draw the bony landmarks on screen ============================= #
# --------------------------------------------------------------------------- #
# Now that the hand is detected, we want to SEE it: green dots on every joint and
# blue lines for the "bones". Below are four ready-made tools. Only ONE actually
# draws the dots and bone-lines; the others compute or print things instead:
#
#   overlay_states  -> writes the finger words ("extended"/"retracted") as text.
#   emit_commands   -> PRINTS "extend/retract" in the terminal. Draws nothing.
#   finger_states   -> WORKS OUT which fingers are open. Returns numbers, no drawing.
#   draw_landmarks  -> draws the dots and bone-lines ON your hand.  <-- we want this one
#
# Uncomment the tool that draws the dots and bones.
#
# WHERE YOUR ANSWER PLUGS IN (for the curious):
#   * The engine calls whatever you pick, once per frame, in
#     workshop_engine.py -> run_workshop(), around line 222:
#         answers[2](frame, hand_landmarks)
#   * The tool itself lives in motion_recognition.py -> draw_landmarks()
#     (around line 222).
#   * In the full pipeline severus_motion_bridge_actual_and_simulation.py ->
#     run_simulation() calls exactly this, around line 202:
#         draw_landmarks(frame, hand_landmarks)
ANSWER_2 = None
# ANSWER_2 = overlay_states
# ANSWER_2 = emit_commands
# ANSWER_2 = finger_states
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
# Uncomment the sign that means "tip is HIGHER UP (smaller y) than the knuckle".
#
#   ">"   tip_y bigger  (tip LOWER down than the knuckle)
#   ">="  bigger or equal
#   "<="  smaller or equal
#   "<"   tip_y smaller (tip HIGHER UP than the knuckle)   <-- the one we want
#
# WHERE YOUR ANSWER PLUGS IN (for the curious):
#   * Your sign is dropped straight into the comparison in
#     workshop_engine.py -> _apply_y_op() (around lines 80-86):
#         if op == "<":  return tip_y < pip_y
#     which is called for each finger by _compute_states() (around line 107).
#   * The finished version of this rule is hard-wired in
#     motion_recognition.py -> finger_states() (around lines 154-157), e.g.
#         states["index"] = lm[8].y < lm[6].y
#     (landmark 8 = index tip, 6 = its knuckle). That is the function the full
#     pipeline severus_motion_bridge_actual_and_simulation.py uses (line ~203).
ANSWER_3 = None
# ANSWER_3 = ">"
# ANSWER_3 = ">="
# ANSWER_3 = "<="
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
# Uncomment the sign that means "tip further RIGHT (bigger x) than the joint".
#
#   "<"   tip_x smaller (tip further LEFT)
#   "<="  smaller or equal
#   ">="  bigger or equal
#   ">"   tip_x bigger  (tip further RIGHT)   <-- the one we want
#
# WHERE YOUR ANSWER PLUGS IN (for the curious):
#   * Your sign is used in workshop_engine.py -> _thumb_extended()
#     (around lines 89-98), which compares the thumb tip's x to its joint's x and
#     flips the result automatically for a left hand. It is called by
#     _compute_states() (around line 111).
#   * The finished version is hard-wired in motion_recognition.py ->
#     finger_states() (around lines 148-151):
#         states["thumb"] = lm[4].x > lm[3].x     # for a right hand
#     (landmark 4 = thumb tip, 3 = its IP joint) -- the same function the full
#     pipeline severus_motion_bridge_actual_and_simulation.py runs.
ANSWER_4 = None
# ANSWER_4 = "<"
# ANSWER_4 = "<="
# ANSWER_4 = ">="
# ANSWER_4 = ">"


# --------------------------------------------------------------------------- #
# === TASK 5: Make the simulated hand copy you ============================== #
# --------------------------------------------------------------------------- #
# The drawn robotic hand uses a number for each finger:
#     1.0 = fully OPEN (extended)        0.0 = fully CLOSED (retracted)
# We need to tell it: "when MY finger is EXTENDED, set the robot finger to ___".
#
# Uncomment the value that means fully OPEN.
#
#   0.0   fully closed (wrong -- that is the retracted end)
#   0.5   half-open (wrong -- only bends the finger halfway)
#   2.0   past fully open (wrong -- there is no such thing; the range is 0.0..1.0)
#   1.0   fully open   <-- the one we want
#
# WHERE YOUR ANSWER PLUGS IN (for the curious):
#   * Your number becomes the finger target in workshop_engine.py ->
#     run_workshop(), around lines 237-246:
#         ext_val = float(answers[5])
#         target_ext[name] = ext_val if is_ext else (1.0 - ext_val)
#   * That target drives TWO things:
#       - the DRAWN hand: motion_recognition_simulation.py -> render_hand()
#         (around line 375), and
#       - the REAL Severus hand: motion_recognition_simulation.py ->
#         send_finger_command() (around line 137), which sends "f<n>o"/"f<n>c"
#         over the serial cable.
#   * In the full pipeline severus_motion_bridge_actual_and_simulation.py the
#     same mapping lives in run_simulation() (lines ~208-217) for the drawing and
#     in run_hardware() for the physical hand.
ANSWER_5 = None
# ANSWER_5 = 0.0
# ANSWER_5 = 0.5
# ANSWER_5 = 2.0
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
#       Why:   HandLandmarker is the tool that returns the 21 hand points.
#       Used:  workshop_engine.py -> run_workshop() (line ~184).
#       Grown-up version: severus_motion_bridge_actual_and_simulation.py ->
#                         _make_landmarker() (line ~141).
#
#   TASK 2:  ANSWER_2 = draw_landmarks
#       Why:   draw_landmarks paints the green dots and blue bone-lines on the
#              hand (overlay_states/emit_commands/finger_states do not draw it).
#       Used:  workshop_engine.py -> run_workshop() (line ~222).
#       The function itself: motion_recognition.py -> draw_landmarks() (line ~222).
#       Grown-up version: severus_motion_bridge_actual_and_simulation.py ->
#                         run_simulation() (line ~202).
#
#   TASK 3:  ANSWER_3 = "<"
#       Why:   the image y-axis points DOWN, so "higher up" means a SMALLER y. An
#              extended finger has its tip above the knuckle, so tip_y < knuckle_y.
#       Used:  workshop_engine.py -> _apply_y_op() (lines ~80-86), called from
#              _compute_states() (line ~107).
#       Grown-up version: motion_recognition.py -> finger_states() (lines ~154-157),
#                         which the full pipeline runs at line ~203.
#
#   TASK 4:  ANSWER_4 = ">"
#       Why:   the thumb moves sideways. For a right hand, an extended thumb's tip
#              is further to the RIGHT (bigger x) than its lower joint. The program
#              mirrors this automatically for a left hand.
#       Used:  workshop_engine.py -> _thumb_extended() (lines ~89-98), called from
#              _compute_states() (line ~111).
#       Grown-up version: motion_recognition.py -> finger_states() (lines ~148-151).
#
#   TASK 5:  ANSWER_5 = 1.0
#       Why:   the simulated finger uses 1.0 = fully open (extended) and
#              0.0 = fully closed. Your extended finger should map to 1.0.
#       Used:  workshop_engine.py -> run_workshop() (lines ~237-246).
#       Drives: motion_recognition_simulation.py -> render_hand() (line ~375, the
#               drawn hand) and send_finger_command() (line ~137, the real hand).
#       Grown-up version: severus_motion_bridge_actual_and_simulation.py ->
#                         run_simulation() (lines ~208-217) and run_hardware().
# =========================================================================== #
