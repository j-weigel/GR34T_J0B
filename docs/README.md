# Portfolio site

This is a static site with no build step. Open `index.html` in a browser, or serve the folder:

```sh
python3 -m http.server --directory docs 8000   # then visit http://localhost:8000
```

## Layout

| Path | What it is |
| --- | --- |
| `index.html` | Home page: the neural void |
| `about.html`, `past-projects.html`, `experience.html`, `contact.html` | Module pages (placeholder content) |
| `current-projects.html` | EvoComp Playground (DCS 340): explanation plus the notebook's code |
| `assets/css/cosmos.css` | Shared theme: colors, fonts, panels, code blocks |
| `assets/js/cosmos.js` | Background on every page: Milky Way, stars, quantum particles, page fades |
| `assets/js/neurons.js` | Home-page neurons, axons, and signal firing. The `MODULES` list at the top sets labels, links, and positions |
| `assets/js/notebook.js` | Renders an exported notebook into `<div data-notebook="…">` |

## Updating the code on the project page

The code shown on the site is exported from the notebook. After editing `EvoCompPlayground.ipynb`, run this from the repo root:

```sh
python3 tools/export_notebook.py EvoCompPlayground.ipynb evocomp-playground
```

To add another project page, export its notebook under a new slug. Then copy `current-projects.html`, change the text, and point `data-notebook` and the data `<script>` at the new slug.

## Publishing on GitHub Pages

Push the repo, then go to **Settings → Pages → Deploy from a branch** and choose `main` and `/docs`.
