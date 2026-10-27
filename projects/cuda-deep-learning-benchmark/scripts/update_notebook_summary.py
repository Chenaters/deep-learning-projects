from __future__ import annotations

import nbformat as nbf

from make_notebook import NOTEBOOK_PATH, load_summary_markdown


def update_summary() -> None:
    notebook = nbf.read(NOTEBOOK_PATH, as_version=4)
    notebook.cells[0].source = load_summary_markdown()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Updated summary in {NOTEBOOK_PATH}")


if __name__ == "__main__":
    update_summary()
