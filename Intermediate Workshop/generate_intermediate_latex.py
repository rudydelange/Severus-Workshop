"""Generate the INTERMEDIATE workshop handout as an IEEE-style LaTeX file.

Run it (Run button in VS Code, or ``python generate_intermediate_latex.py``) to
(re)create ``Intermediate_Severus_Workshop.tex`` next to this script. Open that
.tex with the VS Code "LaTeX Workshop" extension (or compile with pdflatex /
Overleaf) to produce the PDF. IEEEtran ships with TeX Live and MiKTeX.
"""

from __future__ import annotations

import os
import sys

# Shared LaTeX content lives in the parent "Motion Recognition Workshop" folder.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from workshop_latex_common import build_latex

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "Intermediate_Severus_Workshop.tex")


def main() -> None:
    tex = build_latex("intermediate")
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"Wrote {OUTPUT} ({len(tex)} chars).")


if __name__ == "__main__":
    main()
