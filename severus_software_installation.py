"""Prerequisite checker / auto-installer for the Severus motion bridge.

Imported and called at the very top of
``severus_motion_bridge_actual_and_simulation.py`` -- BEFORE any third-party
import -- so it can guarantee the Python environment is ready (or build it)
without the program crashing on a missing package.

What it checks / fixes
----------------------
  * A compatible CPython (3.9-3.12; MediaPipe has no wheels outside that range).
    If the interpreter running this file is too new/old, it looks for a
    compatible one (preferring **Python 3.12** via the ``py -3.12`` launcher or a
    ``python3.12`` on PATH) and builds the virtual environment from that.
  * The pip packages needed for both modes:
        - opencv-python, mediapipe, numpy  -> camera + gesture + simulated hand
        - pyserial                         -> the COM connection to the Arduino
  * tkinter (used for the pop-ups; ships with the stdlib on Windows).

Behaviour (the requested flow)
------------------------------
  * If every prerequisite is already importable on a compatible interpreter -> a
    pop-up says ``"Program Installation Pre-requisites met"`` and the program runs
    in the current interpreter.
  * Otherwise -> a pop-up says ``"Program installation files required,
    auto-installing required packages in a virtual machine"`` (a Python *virtual
    environment*, ``.venv``, next to this file), the packages are installed into
    it, and the program is **re-launched inside that venv** so it continues
    automatically with everything available.

Scope notes
-----------
  * "Virtual machine" here means a Python ``venv`` -- the correct, lightweight
    tool for this. A full OS-level VM would be overkill and is not used.
  * Nothing here touches the Arduino: the board is assumed already flashed with
    ``260612_Severus_5FingerControl_v3.ino``. Connecting over the COM port and
    sending finger extend/retract commands is done at runtime by the main script
    (``find_arduino`` / ``SerialBridge``); installing ``pyserial`` is all that is
    needed here for that to work.
  * Python itself cannot be installed from within Python; if no compatible
    interpreter exists at all, the pop-up explains what to install.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# import-name -> pip-name for the runtime dependencies (both modes).
REQUIRED_PACKAGES: dict[str, str] = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "serial": "pyserial",
}

MIN_PY = (3, 9)
MAX_PY_EXCLUSIVE = (3, 13)        # MediaPipe wheels currently stop before 3.13
PREFERRED_MINORS = (12, 11, 10, 9)  # 3.12 first -- best supported for MediaPipe

VENV_DIRNAME = ".venv"
GUARD_ENV = "SEVERUS_ENV_BOOTSTRAPPED"  # set on the re-exec'd child to avoid loops


def _here() -> Path:
    return Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Pop-up helper (tkinter, with console fallback)
# --------------------------------------------------------------------------- #
def popup(title: str, message: str, kind: str = "info") -> None:
    """Show a small GUI message box; always also print to the console."""
    banner = {"info": "===", "warning": "!!!", "error": "XXX"}.get(kind, "===")
    print(f"\n{banner} {title} {banner}\n{message}\n")
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if kind == "error":
            messagebox.showerror(title, message)
        elif kind == "warning":
            messagebox.showwarning(title, message)
        else:
            messagebox.showinfo(title, message)
        root.destroy()
    except Exception:  # noqa: BLE001 - headless / no Tk: the console line is enough
        pass


# --------------------------------------------------------------------------- #
# Python interpreter checks / discovery
# --------------------------------------------------------------------------- #
def _missing_packages() -> list[str]:
    """Return the pip names of required packages that are not importable."""
    return [
        pip_name
        for import_name, pip_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]


def _version_ok(version: tuple[int, int]) -> bool:
    return MIN_PY <= version < MAX_PY_EXCLUSIVE


def _query_python(cmd: list[str]) -> tuple[tuple[int, int], str] | None:
    """Run ``cmd`` and read its (major, minor) version and executable path."""
    probe = "import sys;print('%d %d %s' % (sys.version_info[0], sys.version_info[1], sys.executable))"
    try:
        out = subprocess.run(cmd + ["-c", probe], capture_output=True, text=True, timeout=15)
    except Exception:  # noqa: BLE001 - candidate not present / not runnable
        return None
    lines = (out.stdout or "").strip().splitlines()
    if not lines:
        return None
    try:
        major_s, minor_s, exe = lines[-1].split(maxsplit=2)
        return (int(major_s), int(minor_s)), exe.strip()
    except (ValueError, IndexError):
        return None


def _find_compatible_python() -> str | None:
    """Locate a compatible interpreter, preferring Python 3.12.

    Tries the Windows ``py -3.<minor>`` launcher and ``python3.<minor>`` /
    ``python`` on PATH. Returns the resolved executable path, or ``None``.
    """
    candidates: list[list[str]] = []
    for minor in PREFERRED_MINORS:
        candidates.append(["py", f"-3.{minor}"])              # Windows py launcher
        which = shutil.which(f"python3.{minor}")
        if which:
            candidates.append([which])
    for name in ("python3", "python"):                        # generic fallbacks
        which = shutil.which(name)
        if which:
            candidates.append([which])

    for cmd in candidates:
        result = _query_python(cmd)
        if result is None:
            continue
        version, exe = result
        if _version_ok(version) and exe and os.path.exists(exe):
            print(f"[setup] Using Python {version[0]}.{version[1]} at {exe}")
            return exe
    return None


# --------------------------------------------------------------------------- #
# Virtual environment
# --------------------------------------------------------------------------- #
def _venv_python(venv_dir: Path) -> Path:
    """Path to the venv's interpreter (Windows layout, with POSIX fallback)."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _create_venv_and_install(base_python: str, missing: list[str]) -> Path:
    """Create ``.venv`` from ``base_python`` (if needed) and pip-install the deps."""
    venv_dir = _here() / VENV_DIRNAME
    py = _venv_python(venv_dir)

    # If an existing venv was built with an incompatible Python, rebuild it.
    if py.exists():
        existing = _query_python([str(py)])
        if not (existing and _version_ok(existing[0])):
            print(f"[setup] Existing {venv_dir} is incompatible; rebuilding ...")
            shutil.rmtree(venv_dir, ignore_errors=True)

    if not py.exists():
        print(f"[setup] Creating virtual environment at {venv_dir} (base: {base_python}) ...")
        subprocess.check_call([base_python, "-m", "venv", str(venv_dir)])

    print("[setup] Upgrading pip ...")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    # Install the full known set so a fresh venv ends up complete (re-installs of
    # already-present packages are cheap no-ops).
    packages = sorted(set(REQUIRED_PACKAGES.values()) | set(missing))
    print(f"[setup] Installing: {', '.join(packages)} ...")
    subprocess.check_call([str(py), "-m", "pip", "install", *packages])
    return py


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def ensure_prerequisites() -> None:
    """Verify (or build) the environment; may re-exec the program inside a venv."""
    # --- Re-exec'd child: the venv was just built; sanity-check and return. ---
    if os.environ.get(GUARD_ENV) == "1":
        missing = _missing_packages()
        if missing:
            popup("Setup incomplete",
                  "The virtual environment is still missing:\n  " + "\n  ".join(missing)
                  + "\n\nTry deleting the .venv folder and re-running.", "error")
            raise SystemExit(1)
        return

    # --- First (parent) run: assess the current interpreter. ---
    current_ok = _version_ok(sys.version_info[:2])
    missing = _missing_packages()

    # Fast path: compatible Python with everything already importable.
    if current_ok and not missing:
        popup("Prerequisites",
              "Program Installation Pre-requisites met", "info")
        return

    # We need a venv -- either packages are missing or this Python is incompatible.
    if current_ok:
        base_python = sys.executable
    else:
        base_python = _find_compatible_python()
        if base_python is None:
            popup("Incompatible Python",
                  f"This program is running on Python {sys.version.split()[0]}, and no\n"
                  f"compatible interpreter (3.{MIN_PY[1]}-3.{MAX_PY_EXCLUSIVE[1] - 1}, "
                  "ideally 3.12) was found.\n\n"
                  "Please install Python 3.12 (python.org or the Microsoft Store) and\n"
                  "re-run -- the installer will then build the virtual environment\n"
                  "from it automatically.", "error")
            raise SystemExit(1)

    popup("Prerequisites",
          "Program installation files required, auto-installing required "
          "packages in a virtual machine", "warning")

    venv_py = _create_venv_and_install(base_python, missing)

    # Re-launch the program inside the venv and exit with its return code.
    script = os.path.abspath(sys.argv[0])
    env = dict(os.environ, **{GUARD_ENV: "1"})
    print(f"[setup] Relaunching inside the virtual environment:\n  {venv_py} {script}")
    proc = subprocess.run([str(venv_py), script, *sys.argv[1:]], env=env)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    ensure_prerequisites()
    print("Environment ready.")
