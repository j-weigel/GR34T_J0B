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

  // Minimal markdown: headings, paragraphs, lists, pipe tables and blockquotes.
  function markdown(src, toc, used, prefix) {
    const out = [];
    let para = [], list = null, table = [], quote = [];
    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
    const flushList = () => { if (list) { out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; } };
    const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    const flushTable = () => {
      if (!table.length) return;
      const rows = table.filter((r) => !/^\s*\|?[\s:|-]+\|?\s*$/.test(r));
      const [head, ...body] = rows;
      out.push(`<div class="nb-table"><table><thead><tr>${cells(head).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${
        body.map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      table = [];
    };
    const flushQuote = () => { if (quote.length) { out.push(`<blockquote>${inline(quote.join(" "))}</blockquote>`); quote = []; } };
    const flush = () => { flushPara(); flushList(); flushTable(); flushQuote(); };
    for (const line of src.split("\n")) {
      // Colab-style headings may omit the space and wrap the title in emphasis: #***Title***
      const h = /^(#{1,6})\s*(.+)$/.exec(line);
      const li = /^\s*(?:([-*+])|(\d+)[.)])\s+(.+)$/.exec(line);
      if (h) {
        flush();
        const text = h[2].replace(/^(\*+)(.+)\1$/, "$2").replace(/:\s*$/, "").trim();
        if (h[1].length <= 2) {
          let id = prefix + "-" + (slug(text) || "section");
          while (used.has(id)) id += "-";
          used.add(id);
          toc.push({ id, text });
          out.push(`<h3 id="${id}">${inline(text)}</h3>`);
        } else {
          out.push(`<h4>${inline(text)}</h4>`);
        }
      } else if (/^\s*\|.*\|\s*$/.test(line)) {
        flushPara(); flushList(); flushQuote(); table.push(line);
      } else if (/^\s*>/.test(line)) {
        flushPara(); flushList(); flushTable(); quote.push(line.replace(/^\s*>\s?/, ""));
      } else if (li && !/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
        flushPara(); flushTable(); flushQuote();
        const tag = li[2] ? "ol" : "ul";
        if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
        list.items.push(li[3]);
      } else if (/^\s*(-{3,}|\*{3,})\s*$/.test(line) || !line.trim()) {
        flush();
      } else if (list && /^\s{2,}\S/.test(line)) {
        list.items[list.items.length - 1] += " " + line.trim();
      } else {
        flushList(); flushTable(); flushQuote();
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
          ? `<img src="${esc(o.src)}" alt="Plot output of cell ${codeIndex}" loading="lazy"${o.width ? ` width="${o.width}" height="${o.height}"` : ""}>`
          : o.kind === "html"
          ? `<iframe class="nb-anim" src="${esc(o.src)}" title="Animation output of cell ${codeIndex}" loading="lazy" style="max-width:${o.width + 20}px;aspect-ratio:${o.width + 20} / ${o.height}"></iframe>`
          : `<pre>${esc(o.text.replace(/\n+$/, ""))}</pre>`
      ).join("");
      return `<div class="nb-cell">
        <div class="bar"><span>${nb.kind === "script" ? esc(nb.name) : `In [${codeIndex}]`}</span><button class="copy" type="button">copy</button></div>
        <pre><code class="language-python">${esc(cell.source)}</code></pre>
        ${outs ? `<div class="nb-out"><div class="tag">OUT</div>${outs}</div>` : ""}
      </div>`;
    }).join("");

    const tocHtml = toc.map((t) => `<a href="#${t.id}">${inline(t.text)}</a>`).join("");
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
