"""Export a Jupyter notebook into a data file the portfolio site can render.

Usage (from the repo root):
    python3 tools/export_notebook.py EvoCompPlayground.ipynb evocomp-playground

Writes:
    docs/assets/data/<slug>.js           cell sources + text outputs
    docs/assets/img/<slug>/cell-XX-Y.png plot images from the notebook outputs
    docs/assets/notebooks/<notebook>     a downloadable copy of the notebook

The data file is plain JavaScript (not JSON) so the site also works when
opened straight from disk, where fetch() of local files is blocked.
"""

import base64
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def text_of(value):
    return "".join(value) if isinstance(value, list) else (value or "")


def export(notebook_path, slug):
    notebook_path = Path(notebook_path)
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    img_dir = DOCS / "assets" / "img" / slug
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)

    cells = []
    for index, cell in enumerate(nb.get("cells", [])):
        source = text_of(cell.get("source")).rstrip()
        if not source:
            continue

        entry = {"type": cell["cell_type"], "source": source}

        if cell["cell_type"] == "code":
            outputs = []
            for out_index, out in enumerate(cell.get("outputs", [])):
                kind = out.get("output_type")
                data = out.get("data", {})
                if kind == "stream":
                    outputs.append({"kind": "text", "text": text_of(out.get("text"))})
                elif "image/png" in data:
                    name = f"cell-{index:02d}-{out_index}.png"
                    (img_dir / name).write_bytes(base64.b64decode(data["image/png"]))
                    outputs.append({"kind": "image", "src": f"assets/img/{slug}/{name}"})
                elif "text/plain" in data and kind == "execute_result":
                    outputs.append({"kind": "text", "text": text_of(data["text/plain"])})
                elif kind == "error":
                    outputs.append({"kind": "text", "text": f"{out.get('ename')}: {out.get('evalue')}"})
            # Merge consecutive text chunks so printed output reads as one block.
            merged = []
            for out in outputs:
                if merged and out["kind"] == "text" and merged[-1]["kind"] == "text":
                    merged[-1]["text"] += out["text"]
                else:
                    merged.append(out)
            entry["outputs"] = merged

        cells.append(entry)

    copy_dir = DOCS / "assets" / "notebooks"
    copy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, copy_dir / notebook_path.name)

    payload = {
        "name": notebook_path.name,
        "download": f"assets/notebooks/{notebook_path.name}",
        "cells": cells,
    }
    out_file = DOCS / "assets" / "data" / f"{slug}.js"
    out_file.write_text(
        "window.NOTEBOOKS = window.NOTEBOOKS || {};\n"
        f"window.NOTEBOOKS[{json.dumps(slug)}] = {json.dumps(payload, indent=1)};\n",
        encoding="utf-8",
    )
    print(f"Exported {len(cells)} cells -> {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    export(sys.argv[1], sys.argv[2])
