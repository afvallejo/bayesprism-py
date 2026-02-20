from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

MARKER = "VALIDATION_REPORT_JSON="
REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "tutorial_deconvolution_validation.ipynb"


def _collect_text_outputs(notebook: nbformat.NotebookNode) -> str:
    parts: list[str] = []
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                parts.append(str(output.get("text", "")))
            elif output_type in {"display_data", "execute_result"}:
                text_plain = output.get("data", {}).get("text/plain")
                if text_plain is not None:
                    if isinstance(text_plain, list):
                        parts.append("".join(str(x) for x in text_plain))
                    else:
                        parts.append(str(text_plain))
    return "\n".join(parts)


def test_tutorial_deconvolution_notebook_smoke(monkeypatch) -> None:
    assert NOTEBOOK_PATH.exists()

    monkeypatch.setenv("BAYESPRISM_NOTEBOOK_MODE", "synthetic")
    monkeypatch.setenv("BAYESPRISM_NOTEBOOK_RUN_PLOTS", "0")
    monkeypatch.setenv("BAYESPRISM_NOTEBOOK_RUN_DE", "0")
    monkeypatch.setenv("BAYESPRISM_NOTEBOOK_RUN_HEAVY", "0")

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    executed = client.execute()

    output_text = _collect_text_outputs(executed)
    marker_line = next((line for line in output_text.splitlines() if line.startswith(MARKER)), None)
    assert marker_line is not None, "Notebook did not emit validation report marker"

    report = json.loads(marker_line[len(MARKER) :])
    assert report["status"] == "ok"
    assert report["mode"] == "synthetic"
    assert report["invariants"]["theta_non_negative"] is True
    assert report["invariants"]["theta_rowsum_close"] is True
