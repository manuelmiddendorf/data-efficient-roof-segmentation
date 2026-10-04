"""Execute scientific report notebooks from fresh kernels and export HTML."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks/01_data_exploration.ipynb",
    ROOT / "notebooks/02_training_pilot.ipynb",
]

for notebook in NOTEBOOKS:
    if not notebook.exists():
        continue
    document = nbformat.read(notebook, as_version=4)
    client = NotebookClient(
        document,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute(cwd=str(ROOT))
    nbformat.write(document, notebook)

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(document)
    report = ROOT / "reports" / f"{notebook.stem}.html"
    report.write_text(html, encoding="utf-8")
    print(f"Executed {notebook.name} and exported {report.relative_to(ROOT)}")
