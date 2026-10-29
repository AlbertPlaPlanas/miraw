"""Execute a notebook in its own folder and tidy the stored outputs (progress bars: keep the final state).

Usage: python scripts/run_notebook.py notebooks/pymiraw_tutorial.ipynb     (needs nbclient and ipykernel: pymiraw[dev])
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def final_line_state(text: str) -> str:
    """Apply carriage returns the way a terminal does: keep the text after the last '\\r' on each line."""
    return "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))


def tidy(nb) -> None:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        merged = []
        for out in cell.outputs:
            if (
                out.output_type == "stream"
                and merged
                and merged[-1].output_type == "stream"
                and merged[-1].name == out.name
            ):
                merged[-1].text += out.text
            else:
                merged.append(out)
        for out in merged:
            if out.output_type == "stream":
                out.text = final_line_state(out.text)
                out.text = "\n".join(ln for ln in out.text.split("\n") if "Kernel is running over TCP" not in ln)
        cell.outputs = [o for o in merged if not (o.output_type == "stream" and not o.text.strip())]


path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
t = time.time()
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
tidy(nb)
nb.metadata.pop("widgets", None)
nbformat.write(nb, path)
print(f"executed {path.name} in {time.time() - t:.0f} s")
