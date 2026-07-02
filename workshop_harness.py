"""Shared machinery for the Severus motion workshops (beginner + intermediate).

The workshop scripts let a participant fill in a handful of answers
(``ANSWER_1`` .. ``ANSWER_5``). This module does the *boring but important*
plumbing so the workshop files can stay focused on the teaching:

  * Validates the answers **in sequence** and figures out which features are
    therefore working.
  * Shows a pop-up for the FIRST unsolved task: which ``ANSWER_x`` line to fix
    (with its real line number), a hint, and -- after 3 failed attempts on that
    same task -- the actual answer. Failure counts persist between runs in a
    small JSON file next to the script.
  * Draws an on-screen status panel ("[OK] / [TODO]") and a banner for the next
    thing to fix, so the program runs from not-working -> partly -> fully working
    and always shows *something* useful.

None of this needs editing by participants; they only ever touch the
``ANSWER_x`` lines in their workshop file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2


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
# Task description
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """One thing the participant must get right to enable a feature."""
    key: int                       # 1..N, also the sequence order
    name: str                      # short feature label for the on-screen panel
    var_name: str                  # e.g. "ANSWER_1" (used for line lookup + pop-up)
    check: Callable[[Any], bool]   # returns True when the answer is correct
    hint: str                      # nudge shown on every failed attempt
    answer_text: str               # the real answer, revealed after 3 failures


# --------------------------------------------------------------------------- #
# Source line lookup -- find the line the participant actually needs to edit
# --------------------------------------------------------------------------- #
def find_active_assignment_line(script_path: str, var_name: str) -> int | None:
    """Return the line number of the *active* ``var_name = ...`` assignment.

    Skips commented-out candidate lines so the pop-up points at the line that is
    really in effect. Returns the last matching uncommented assignment.
    """
    try:
        text = Path(script_path).read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return None
    pattern = re.compile(rf"^\s*{re.escape(var_name)}\s*=")
    found: int | None = None
    for i, line in enumerate(text, start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if pattern.match(line):
            found = i
    return found


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
@dataclass
class WorkshopHarness:
    title: str
    script_path: str
    tasks: list[Task]
    progress_path: str = ""
    _counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.progress_path:
            stem = Path(self.script_path).stem
            self.progress_path = str(Path(self.script_path).with_name(f".progress_{stem}.json"))
        self._counts = self._load_counts()

    # --- persistence ---
    def _load_counts(self) -> dict[str, int]:
        try:
            return json.loads(Path(self.progress_path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - first run / unreadable
            return {}

    def _save_counts(self) -> None:
        try:
            Path(self.progress_path).write_text(json.dumps(self._counts), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # --- evaluation ---
    def _status_for(self, task: Task, answers: dict[int, Any]) -> bool:
        try:
            return bool(task.check(answers.get(task.key)))
        except Exception:  # noqa: BLE001 - a broken answer just means "not done yet"
            return False

    def review(self, answers: dict[int, Any]) -> dict[int, bool]:
        """Validate answers, pop up guidance for the next unsolved task, return status."""
        status = {t.key: self._status_for(t, answers) for t in self.tasks}

        first_todo = next((t for t in self.tasks if not status[t.key]), None)
        if first_todo is None:
            popup(f"{self.title}", "All tasks complete -- every feature is enabled.\n"
                                   "Great work! Move your hand in front of the camera.", "info")
            return status

        # Count this attempt against the blocking task and decide whether to reveal.
        ckey = str(first_todo.key)
        self._counts[ckey] = self._counts.get(ckey, 0) + 1
        self._save_counts()
        attempts = self._counts[ckey]
        reveal = attempts >= 3

        line = find_active_assignment_line(self.script_path, first_todo.var_name)
        where = f"line {line}" if line else f"the '{first_todo.var_name}' line"
        current = answers.get(first_todo.key)

        msg = (
            f"Task {first_todo.key} -- {first_todo.name} -- is not working yet.\n\n"
            f"Fix:  {first_todo.var_name}  ({where})\n"
            f"Current value: {current!r}\n\n"
            f"Hint: {first_todo.hint}\n"
            f"(attempt {attempts})"
        )
        if reveal:
            msg += f"\n\nANSWER: {first_todo.answer_text}"
        popup(f"{self.title} - Task {first_todo.key}", msg, "warning")
        return status

    # --- on-screen overlays ---
    def first_todo(self, status: dict[int, bool]) -> Task | None:
        return next((t for t in self.tasks if not status[t.key]), None)

    def draw_status_panel(self, frame, status: dict[int, bool]) -> None:
        """Top-right checklist of every task: green [OK] or red [TODO]."""
        h, w = frame.shape[:2]
        x = w - 360
        y = 30
        cv2.putText(frame, "WORKSHOP PROGRESS", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for t in self.tasks:
            y += 28
            ok = status[t.key]
            tag = "[OK]  " if ok else "[TODO]"
            color = (120, 220, 120) if ok else (90, 90, 235)
            cv2.putText(frame, f"{tag} {t.key}. {t.name}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def draw_todo_banner(self, frame, status: dict[int, bool]) -> None:
        """Centered banner pointing at the next thing to fix (if any)."""
        todo = self.first_todo(status)
        if todo is None:
            return
        h, w = frame.shape[:2]
        text = f"NEXT: Task {todo.key} - {todo.name} (edit {todo.var_name})"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cx = max(10, (w - tw) // 2)
        cv2.rectangle(frame, (cx - 10, 88), (cx + tw + 10, 118), (0, 0, 0), -1)
        cv2.putText(frame, text, (cx, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 215, 255), 2, cv2.LINE_AA)
