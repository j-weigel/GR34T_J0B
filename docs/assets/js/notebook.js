/* ==========================================================================
   Renders an exported notebook (see tools/export_notebook.py) into a page.
   Usage: <div data-notebook="evocomp-playground"></div>
   ========================================================================== */
(function () {
  "use strict";

  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  }

  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

  // Minimal markdown: headings and paragraphs are all these notebooks use.
  function markdown(src, toc, used, prefix) {
    const out = [];
    let para = [];
    const flush = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
    for (const line of src.split("\n")) {
      // Colab-style headings may omit the space and wrap the title in emphasis: #***Title***
      const h = /^(#{1,6})\s*(.+)$/.exec(line);
      if (h) {
        flush();
        const text = h[2].replace(/^\*+|\*+$/g, "").replace(/:\s*$/, "").trim();
        if (h[1].length <= 2) {
          let id = prefix + "-" + (slug(text) || "section");
          while (used.has(id)) id += "-";
          used.add(id);
          toc.push({ id, text });
          out.push(`<h3 id="${id}">${inline(text)}</h3>`);
        } else {
          out.push(`<h4>${inline(text)}</h4>`);
        }
      } else if (/^\s*(-{3,}|\*{3,})\s*$/.test(line) || !line.trim()) {
        flush();
      } else {
        para.push(line.trim());
      }
    }
    flush();
    return out.join("");
  }

  function render(host) {
    const nb = (window.NOTEBOOKS || {})[host.dataset.notebook];
    if (!nb) {
      host.innerHTML = '<p class="sub">The notebook could not be loaded.</p>';
      return;
    }
    const toc = [];
    const used = new Set();
    let codeIndex = 0;
    const html = nb.cells.map((cell) => {
      if (cell.type === "markdown") return `<div class="nb-md">${markdown(cell.source, toc, used, host.dataset.notebook)}</div>`;
      if (cell.type !== "code") return "";
      codeIndex++;
      const outs = (cell.outputs || []).map((o) =>
        o.kind === "image"
          ? `<img src="${esc(o.src)}" alt="Plot output of cell ${codeIndex}" loading="lazy">`
          : `<pre>${esc(o.text.replace(/\n+$/, ""))}</pre>`
      ).join("");
      return `<div class="nb-cell">
        <div class="bar"><span>${nb.kind === "script" ? esc(nb.name) : `In [${codeIndex}]`}</span><button class="copy" type="button">copy</button></div>
        <pre><code class="language-python">${esc(cell.source)}</code></pre>
        ${outs ? `<div class="nb-out"><div class="tag">OUT</div>${outs}</div>` : ""}
      </div>`;
    }).join("");

    const tocHtml = toc.map((t) => `<a href="#${t.id}">${esc(t.text)}</a>`).join("");
    host.innerHTML = (tocHtml ? `<nav class="toc" aria-label="Notebook sections">${tocHtml}</nav>` : "") + html;

    host.querySelectorAll(".nb-cell").forEach((cell) => {
      const btn = cell.querySelector(".copy");
      const code = cell.querySelector("pre code");
      btn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(code.textContent);
          btn.textContent = "copied ✦";
        } catch (e) {
          btn.textContent = "select & copy";
        }
        setTimeout(() => { btn.textContent = "copy"; }, 1600);
      });
    });

    if (window.Prism) window.Prism.highlightAllUnder(host);
  }

  document.querySelectorAll("[data-notebook]").forEach(render);
})();
