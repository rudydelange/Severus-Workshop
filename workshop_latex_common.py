"""Shared content + builder for the Severus workshop LaTeX handouts.

Produces IEEE-conference-style (``IEEEtran``) documents for three variants:
    * "beginner"     -- gentle, shows the exact answer line.
    * "intermediate" -- technical tips, leaves the value for the participant.
    * "leader"       -- full answer key + talking points + marking checklist.

The thin per-folder generators (``generate_beginner_latex.py`` etc.) just import
``build_latex`` from here and write the resulting ``.tex`` file. Compile the
``.tex`` with the VS Code "LaTeX Workshop" extension, ``pdflatex``, or Overleaf
(IEEEtran ships with TeX Live / MiKTeX).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Exercise content (single source of truth for all three documents)
# --------------------------------------------------------------------------- #
EXERCISES = [
    {
        "feature": "Hand-landmark detection",
        "var": "ANSWER_1",
        "blank": "ANSWER_1 = None        # choose the correct vision task class",
        "answer": "ANSWER_1 = vision.HandLandmarker",
        "does": (
            "Selects the MediaPipe vision task that, for every frame, returns the "
            "21 hand keypoints (knuckles and fingertips) as normalised "
            "coordinates. The program then builds its detector from this class."
        ),
        "control": (
            "This is the SENSING front end of the prosthesis: it converts a raw "
            "camera image into structured numbers. Without a reliable hand "
            "estimate there is no signal to map onto the fingers, so every later "
            "step depends on this one."
        ),
        "options": (
            "Alternatives are deliberately offered: PoseLandmarker tracks the "
            "whole body and GestureRecognizer only emits named gestures such as "
            "thumbs-up. Neither exposes the per-finger geometry we need."
        ),
        "hint": "The vision task class returning 21 HAND landmarks (Hand + Landmarker).",
        "talking": (
            "Ask why body-pose or gesture-classification is the wrong tool: we "
            "need continuous per-joint geometry, not a discrete label."
        ),
        "mistakes": "Picking GestureRecognizer; adding parentheses (we need the class, not an instance).",
    },
    {
        "feature": "Drawing the landmarks",
        "var": "ANSWER_2",
        "blank": "ANSWER_2 = None        # choose the function that draws dots + bones",
        "answer": "ANSWER_2 = draw_landmarks",
        "does": (
            "Picks the drawing function that overlays the 21 keypoints and the "
            "skeleton connections onto the camera frame so the tracked hand is "
            "visible."
        ),
        "control": (
            "Visual feedback is how an engineer validates a sensor before trusting "
            "it for control. Seeing the skeleton lock onto the hand confirms the "
            "estimate is stable enough to drive an actuator."
        ),
        "options": (
            "The other option, overlay_states, writes the finger words as text. "
            "Useful, but it does not show the geometry that proves tracking works."
        ),
        "hint": "A function reference (no parentheses) drawing dots+bones, not text.",
        "talking": "Have them wiggle fingers and watch the skeleton track; this builds trust in the sensor.",
        "mistakes": "Choosing overlay_states; calling the function (adding parentheses) instead of passing it.",
    },
    {
        "feature": "Finger extend / retract rule",
        "var": "ANSWER_3",
        "blank": 'ANSWER_3 = None        # comparison operator as a string: "<" or ">"',
        "answer": 'ANSWER_3 = "<"',
        "does": (
            "Defines, for the four fingers, the rule that turns two landmark "
            "heights into a binary state. The engine evaluates tip-y (operator) "
            "pip-y; you supply the operator. Image coordinates put y = 0 at the "
            "TOP and increase downward, so a raised (extended) fingertip has the "
            "SMALLER y."
        ),
        "control": (
            "This is the heart of joint-space mapping: a continuous measurement is "
            "thresholded into an open/close command per finger. The same idea "
            "scales up to proportional control, but binary is the clearest start."
        ),
        "options": (
            'Choosing ">" simply inverts the logic (extended reads as retracted), '
            "which is an instructive bug to see on screen."
        ),
        "hint": "Image y grows downward; an extended fingertip has the smaller y.",
        "talking": "Connect screen coordinates to the maths; let them try the wrong operator and watch it invert.",
        "mistakes": 'Using ">"; passing a bare < instead of the string "<".',
    },
    {
        "feature": "Thumb extend / retract rule",
        "var": "ANSWER_4",
        "blank": 'ANSWER_4 = None        # comparison operator as a string: "<" or ">"',
        "answer": 'ANSWER_4 = ">"',
        "does": (
            "Defines the thumb rule. Because the thumb abducts SIDEWAYS rather than "
            "up and down, it is classified on the x axis: the engine evaluates "
            "tip-x (operator) ip-x for a right hand and mirrors it automatically "
            "for a left hand."
        ),
        "control": (
            "Real hands are not uniform; the thumb has a different kinematic axis. "
            "Treating it separately mirrors how prosthetic controllers special-case "
            "the thumb, which dominates grasp quality."
        ),
        "options": (
            "x = 0 is the LEFT edge and increases to the right, so for a right hand "
            "an abducted thumb tip sits at a larger x than its lower joint."
        ),
        "hint": "Right-hand extended thumb: tip x is greater than ip x.",
        "talking": "Discuss why the thumb needs its own axis and how much it matters for grasping.",
        "mistakes": 'Reusing the finger answer "<"; forgetting the thumb uses x, not y.',
    },
    {
        "feature": "Driving the simulated hand",
        "var": "ANSWER_5",
        "blank": "ANSWER_5 = None        # openness float for an EXTENDED finger (0.0..1.0)",
        "answer": "ANSWER_5 = 1.0",
        "does": (
            "Connects the detected states to the actuator model. Each simulated "
            "finger takes an openness value in the range 0.0 to 1.0; the engine "
            "uses ANSWER_5 for an extended finger and one-minus-ANSWER_5 for a "
            "retracted one, then eases smoothly toward that target."
        ),
        "control": (
            "This is the ACTUATOR command mapping. The very same per-finger command "
            "stream drives the on-screen hand and, once an Arduino is connected, "
            "the physical Severus prosthesis -- simulation and hardware share one "
            "interface."
        ),
        "options": (
            "Setting it to 0.0 inverts the hand (your open hand would close it), a "
            "vivid demonstration of why the command convention must be defined."
        ),
        "hint": "The openness value for a fully EXTENDED/open finger, in [0.0, 1.0].",
        "talking": "Emphasise the shared command interface: solve it once, drive both sim and real hand.",
        "mistakes": "Using 0.0 (inverted); using an integer outside the 0..1 range.",
    },
]

KEYWORDS = ("computer vision, MediaPipe, hand tracking, prosthetics, "
            "human-machine interface, gesture recognition, education")


# --------------------------------------------------------------------------- #
# LaTeX assembly helpers
# --------------------------------------------------------------------------- #
PREAMBLE = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{textcomp}
\usepackage{hyperref}
\lstset{
  basicstyle=\ttfamily\footnotesize,
  breaklines=true,
  frame=single,
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  literate={"}{\textquotedbl}1
}
\begin{document}
"""


def _esc(text: str) -> str:
    """Escape LaTeX special characters in PLAIN prose (not code, which is verbatim)."""
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _code(block: str) -> str:
    return "\\begin{lstlisting}\n" + block + "\n\\end{lstlisting}\n"


def _title_block(variant: str) -> str:
    sub = {
        "beginner": "A Beginner Workshop",
        "intermediate": "An Intermediate Workshop",
        "leader": "Workshop Leader Answer Key",
    }[variant]
    return (
        "\\title{Hands-On Computer Vision and Prosthetic Control:\\\\ " + sub + "}\n"
        "\\author{\\IEEEauthorblockN{Severus Prosthetics Workshop}\n"
        "\\IEEEauthorblockA{Motion Recognition Workshop Series}}\n"
        "\\maketitle\n"
    )


def _abstract(variant: str) -> str:
    common = (
        "This workshop teaches two linked skills in roughly one hour: real-time "
        "computer-vision hand tracking, and translating that tracking into "
        "control of a prosthetic hand. Participants extend a working program by "
        "completing five short edits, each of which switches on one stage of the "
        "pipeline -- detection, visualisation, per-finger state estimation, and "
        "actuator command mapping. The program runs at every stage, degrading "
        "gracefully so unfinished features are simply labelled on screen."
    )
    extra = {
        "beginner": (" This beginner edition requires no prior programming "
                     "experience: each edit is an uncomment-one-line choice with "
                     "the answer explained in full."),
        "intermediate": (" This intermediate edition asks participants to write "
                         "each value themselves, supported by technical tips that "
                         "make external references unnecessary."),
        "leader": (" This edition is the facilitator answer key: it lists every "
                   "correct edit, the reasoning to convey, common mistakes, and a "
                   "marking checklist."),
    }[variant]
    return "\\begin{abstract}\n" + common + extra + "\n\\end{abstract}\n\n" \
           "\\begin{IEEEkeywords}\n" + KEYWORDS + "\n\\end{IEEEkeywords}\n"


def _intro(variant: str) -> str:
    body = (
        "Modern prosthetic hands increasingly take their commands from sensors "
        "that estimate user intent. A camera plus a hand-tracking model is one of "
        "the most accessible such sensors: it needs no electrodes and runs on a "
        "laptop. This workshop builds that path end to end, from pixels to finger "
        "motion, using the Severus hand and its simulator.\n\n"
        "The exercise file opens a live window immediately. Five marked lines "
        "(named \\lstinline!ANSWER_1! through \\lstinline!ANSWER_5!) each enable "
        "one stage. As you complete them, the program progresses from doing very "
        "little, to drawing the hand, to recognising finger states, to moving a "
        "simulated (and optionally physical) prosthesis."
    )
    if variant == "leader":
        body += ("\n\nThe physical hand is intentionally out of scope for the "
                 "participants: the facilitator pre-flashes and calibrates the "
                 "Arduino, so that a correctly completed file simply drives the "
                 "real hand when it is plugged in.")
    return "\\section{Introduction}\n" + body + "\n"


def _background() -> str:
    return (
        "\\section{Background}\n"
        "MediaPipe estimates 21 landmarks per hand. Each landmark has normalised "
        "image coordinates, where x increases to the right and y increases "
        "DOWNWARD from the top of the frame. A finger is treated as extended when "
        "its fingertip is higher than the knuckle below it -- a smaller y. The "
        "thumb is handled on the x axis instead, because it moves sideways.\n\n"
        "These per-finger booleans become commands. In simulation each finger has "
        "an openness between 0.0 (closed) and 1.0 (open); the identical command "
        "stream drives the physical Severus hand once it is connected. The "
        "workshop therefore mirrors a real human-machine interface: sense, "
        "interpret, actuate.\n"
    )


def _exercises(variant: str) -> str:
    out = ["\\section{Workshop Exercises}"]
    for i, ex in enumerate(EXERCISES, start=1):
        out.append("\\subsection{Exercise " + str(i) + ": " + _esc(ex["feature"]) + "}")
        out.append("\\textbf{Line to edit:} \\lstinline!" + ex["var"] + "!.")

        if variant == "beginner":
            out.append("Uncomment the correct option so the line reads:")
            out.append(_code(ex["answer"]))
            out.append("\\textbf{What this line does.} " + _esc(ex["does"]))
            out.append("\\textbf{The choices.} " + _esc(ex["options"]))
            out.append("\\textbf{Why it matters for control.} " + _esc(ex["control"]))
        elif variant == "intermediate":
            out.append("Replace the placeholder with your own value:")
            out.append(_code(ex["blank"]))
            out.append("\\textbf{What this line does.} " + _esc(ex["does"]))
            out.append("\\textbf{Tip.} " + _esc(ex["hint"]))
            out.append("\\textbf{Why it matters for control.} " + _esc(ex["control"]))
        else:  # leader
            out.append("\\textbf{Answer:}")
            out.append(_code(ex["answer"]))
            out.append("\\textbf{What it does.} " + _esc(ex["does"]))
            out.append("\\textbf{Control relevance.} " + _esc(ex["control"]))
            out.append("\\textbf{Talk through.} " + _esc(ex["talking"]))
            out.append("\\textbf{Common mistakes.} " + _esc(ex["mistakes"]))
            out.append("\\textbf{Mark:} $\\square$ solved \\quad $\\square$ needs help")
    return "\n\n".join(out) + "\n"


def _conclusion(variant: str) -> str:
    body = (
        "Completing the five edits yields a full sense-interpret-actuate loop: a "
        "camera tracks the hand, per-finger states are inferred, and a prosthetic "
        "hand mirrors the motion. The same structure underlies production "
        "myoelectric and vision-based controllers; only the sophistication of each "
        "stage grows."
    )
    if variant != "leader":
        body += (" If a finished file is connected to a prepared Severus hand, it "
                 "will move the real fingers exactly as it moves the simulated "
                 "ones.")
    return "\\section{Conclusion}\n" + body + "\n"


def build_latex(variant: str) -> str:
    """Return the full ``.tex`` source for ``variant`` in {beginner, intermediate, leader}."""
    if variant not in ("beginner", "intermediate", "leader"):
        raise ValueError(f"unknown variant: {variant!r}")
    parts = [
        PREAMBLE,
        _title_block(variant),
        _abstract(variant),
        _intro(variant),
        _background(),
        _exercises(variant),
        _conclusion(variant),
        "\\end{document}\n",
    ]
    return "\n".join(parts)
