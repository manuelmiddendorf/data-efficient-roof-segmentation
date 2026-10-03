"""Execute the Stage 1 notebook from a fresh kernel and export HTML."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/01_data_exploration.ipynb"

document = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(
    document,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute(cwd=str(ROOT))
nbformat.write(document, NOTEBOOK)

exporter = HTMLExporter()
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
html, _ = exporter.from_notebook_node(document)
(ROOT / "reports/01_data_exploration.html").write_text(html, encoding="utf-8")
print(f"Executed {NOTEBOOK.name} and exported reports/01_data_exploration.html")
