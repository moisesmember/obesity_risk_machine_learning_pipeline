"""Contratos estáticos dos notebooks versionados no repositório."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"


def test_versioned_notebooks_are_clean_valid_python_notebooks() -> None:
    """Evita notebooks inválidos, com saída persistida ou código Python quebrado."""

    notebook_paths = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    assert notebook_paths, "Nenhum notebook versionado foi encontrado."

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        assert notebook.get("nbformat") == 4, (
            f"{notebook_path.name}: somente o formato nbformat 4 é suportado."
        )

        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue

            assert cell.get("execution_count") is None, (
                f"{notebook_path.name}, célula {index}: remova o contador de execução."
            )
            assert cell.get("outputs", []) == [], (
                f"{notebook_path.name}, célula {index}: remova as saídas antes do commit."
            )

            source = "".join(cell.get("source", []))
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")
