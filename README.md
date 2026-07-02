# Severus Workshop

A workshop on computer vision and prosthetic control. You use a webcam to track your hand in real time (with Google's MediaPipe) and turn that tracking into commands that move the Severus prosthetic hand.

You start from a program that already runs and finish five small edits. Each edit switches on one more part of the pipeline: finding the hand, drawing it, working out which fingers are open, and moving a hand. Everything works without any hardware, because an on-screen hand copies your movements. The same commands drive the real Severus hand when its Arduino is plugged in.

There are two tracks, each about an hour long:

| Track | Folder | For |
|-------|--------|-----|
| Beginner | `Beginner Workshop/` | No programming needed; you uncomment the correct line. |
| Intermediate | `Intermediate Workshop/` | Some Python; you write the answer yourself. |

Each track has a workshop script, an `_ANSWERED` file with the solution, and a PDF handout.

## What you need

- A computer (Windows, macOS or Linux) with a webcam.
- Python 3.10 or newer, from <https://www.python.org/downloads/>. On Windows, tick "Add Python to PATH" while installing.
- VS Code, from <https://code.visualstudio.com/>.
- Optional: the Severus hand and its Arduino, for real-hand control. Without it, everything runs in simulation.

You don't have to install the Python packages yourself. The first time you run a workshop script it checks what is missing and installs it.

## Getting the workshop onto your computer

If you have never used Git before, here is the whole process.

1. Install Git from <https://git-scm.com/downloads> and accept the default options.
2. Open VS Code, click the Accounts icon at the bottom left, and sign in with GitHub.
3. Press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS), type `Git: Clone`, and press Enter.
4. Paste the repository link:
   ```
   https://github.com/rudydelange/Severus-Workshop.git
   ```
5. Pick a folder to save it in, then click Open when VS Code offers to open the clone.

If you prefer the terminal, the same thing is one command:

```bash
git clone https://github.com/rudydelange/Severus-Workshop.git
```

To pick up later changes, open the folder in VS Code and click Sync Changes in the Source Control panel, or run `git pull`.

## Running the workshop

1. Open your track's file:
   - Beginner: `Beginner Workshop/beginner_severus_motion_workshop.py`
   - Intermediate: `Intermediate Workshop/intermediate_severus_motion_workshop.py`
2. Press the Run button, or run it from a terminal:
   ```bash
   python "Beginner Workshop/beginner_severus_motion_workshop.py"
   ```
3. The first launch takes a moment while it checks and installs the packages it needs (OpenCV, MediaPipe, NumPy, pyserial). A pop-up says when it is ready.
4. A window opens with your camera on the left and a drawn hand on the right. Follow the yellow banner and the progress checklist.
5. Edit one marked `ANSWER_x` line, save, and run again. A pop-up confirms progress or gives a hint. Miss the same line three times and it shows you the answer.
6. Press `q` in the window to quit.

Once the five edits are done, your hand drives the on-screen hand live.

### Simulation or the real hand

By default it runs in simulation, so any computer with a webcam works. Add `--force-sim` to stay in simulation even when a hand is plugged in.

For the real Severus hand, plug in its Arduino (already flashed and calibrated) before you run. The script finds it and mirrors your hand with the same commands the simulation uses. You can point it at a specific port with `--port COM5` on Windows or `--port /dev/ttyUSB0` on Linux.

The Arduino firmware is in `260612_Severus_5FingerControl_v3/`.

## What each file does

The top folder holds the shared code, which you do not need to touch:

| File | What it does |
|------|--------------|
| `severus_software_installation.py` | Checks and installs the Python packages. |
| `motion_recognition.py` | Camera and MediaPipe hand-tracking helpers. |
| `motion_recognition_simulation.py` | Draws the simulated hand and talks to the Arduino. |
| `workshop_engine.py` | The main camera, tracking and hand loop both tracks use. |
| `workshop_harness.py` | Checks answers and shows the checklist and hints. |
| `severus_motion_bridge.py` | Standalone link from hand tracking to the real hand. |
| `severus_motion_bridge_actual_and_simulation.py` | Runs in simulation and on the real hand. |
| `workshop_latex_common.py` | Shared text for the PDF handouts. |
| `hand_landmarker.task` | The MediaPipe hand-tracking model, used offline. |

Each workshop folder holds the workshop script, its `_ANSWERED` solution, a `generate_*_latex.py` handout generator, and the `.tex` and `.pdf` handout.

To rebuild the PDF handouts:

```bash
python "Beginner Workshop/generate_beginner_latex.py"
python "Intermediate Workshop/generate_intermediate_latex.py"
```

Open the `.tex` with the VS Code LaTeX Workshop extension, compile it with `pdflatex`, or upload it to Overleaf. The handouts use the IEEEtran class, which comes with TeX Live and MiKTeX.
