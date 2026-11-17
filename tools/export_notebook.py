"""Export a Jupyter notebook or Python script into a data file the portfolio
site can render.

Usage (from the repo root):
    python3 tools/export_notebook.py EvoCompPlayground.ipynb evocomp-playground
    python3 tools/export_notebook.py path/to/script.py some-slug

Writes:
    docs/assets/data/<slug>.js           cell sources + text outputs
    docs/assets/img/<slug>/cell-XX-Y.png plot images from the notebook outputs
    docs/assets/html/<slug>/cell-XX-Y.html HTML outputs, e.g. matplotlib animations
    docs/assets/notebooks/<file>         a downloadable copy of the source

A .py script becomes one code cell per section, split at comment banners:
    # =====
    # SECTION TITLE
    # =====

The data file is plain JavaScript (not JSON) so the site also works when
opened straight from disk, where fetch() of local files is blocked.
"""

import base64
import json
import re
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def text_of(value):
    return "".join(value) if isinstance(value, list) else (value or "")


BANNER = re.compile(r"^# ?={10,}\n# ?(.+?)\n# ?={10,}\n", re.M)


def script_cells(text):
    """Split a script at its section banners: a heading, then that section's code."""
    cells = []
    pos = 0
    title = None
    for m in BANNER.finditer(text):
        chunk = text[pos:m.start()].strip("\n")
        if chunk:
            if title:
                cells.append({"type": "markdown", "source": f"## {title}"})
            cells.append({"type": "code", "source": chunk, "outputs": []})
        # Titles read like "CONFIG  ——  notes"; keep the part before the dash.
        title = re.split(r"\s+[—–-]{1,2}\s+|\s{2,}", m.group(1).strip())[0].title()
        pos = m.end()
    chunk = text[pos:].strip("\n")
    if chunk:
        if title:
            cells.append({"type": "markdown", "source": f"## {title}"})
        cells.append({"type": "code", "source": chunk, "outputs": []})
    return cells


def export(notebook_path, slug):
    notebook_path = Path(notebook_path)
    if notebook_path.suffix == ".py":
        return export_script(notebook_path, slug)
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    img_dir = DOCS / "assets" / "img" / slug
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    html_dir = DOCS / "assets" / "html" / slug
    if html_dir.exists():
        shutil.rmtree(html_dir)

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
                    png = base64.b64decode(data["image/png"])
                    (img_dir / name).write_bytes(png)
                    # PNG IHDR holds the pixel size; the page reserves that space
                    # up front so lazy-loaded plots never shift the layout.
                    width, height = struct.unpack(">II", png[16:24])
                    outputs.append({"kind": "image", "src": f"assets/img/{slug}/{name}",
                                    "width": width, "height": height})
                elif "text/html" in data and "Animation(" in text_of(data["text/html"]):
                    # A matplotlib to_jshtml() animation: a self-contained page
                    # the site shows in a lazy-loaded frame.
                    html_dir.mkdir(parents=True, exist_ok=True)
                    html = text_of(data["text/html"]).replace(
                        "https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css",
                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.4.0/css/font-awesome.min.css")
                    name = f"cell-{index:02d}-{out_index}.html"
                    (html_dir / name).write_text(
                        '<!doctype html><meta charset="utf-8">'
                        '<style>body{margin:0;background:#fff;font-family:sans-serif}</style>' + html,
                        encoding="utf-8")
                    m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=\\n]{40,})", html)
                    width, height = 500, 500
                    if m:
                        png = base64.b64decode(m.group(1).replace("\\n", "")[:64] + "==")
                        width, height = struct.unpack(">II", png[16:24])
                    outputs.append({"kind": "html", "src": f"assets/html/{slug}/{name}",
                                    "width": width, "height": height + 90})
                elif "text/plain" in data and kind == "execute_result" and not text_of(data["text/plain"]).startswith("<IPython.core.display"):
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

    write_payload(notebook_path, slug, cells, "notebook")


def export_script(script_path, slug):
    cells = script_cells(script_path.read_text(encoding="utf-8"))
    write_payload(script_path, slug, cells, "script")


def write_payload(source_path, slug, cells, kind):
    copy_dir = DOCS / "assets" / "notebooks"
    copy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, copy_dir / source_path.name)

    payload = {
        "name": source_path.name,
        "kind": kind,
        "download": f"assets/notebooks/{source_path.name}",
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
