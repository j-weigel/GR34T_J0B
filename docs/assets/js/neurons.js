/* ==========================================================================
   Neural void — the home page network.

   Each neuron is grown procedurally from a seed: an irregular soma that pulls
   out into tapering primary dendrites, recursive bifurcating branches studded
   with dendritic spines, a nucleus with nucleolus, Nissl-body texture, and an
   axon hillock. Neurons are joined by myelinated axons (internodes with gaps
   at the nodes of Ranvier) ending in terminal boutons.

   Every neuron is pre-rendered twice — a dim resting state and a luminous
   firing state — and cross-faded by its activation, so the frame loop only
   composites images and draws the travelling action potentials.
   ========================================================================== */
(function () {
  "use strict";

  const MODULES = [
    { id: "about", label: "About Me", href: "about.html", angle: -90 },
    { id: "past", label: "Past Projects", href: "past-projects.html", angle: -18 },
    { id: "current", label: "Current Projects", href: "current-projects.html", angle: 54 },
    { id: "experience", label: "Experience", href: "experience.html", angle: 126 },
    { id: "contact", label: "Contact", href: "contact.html", angle: 198 },
  ];

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const canvas = document.getElementById("neural");
  const ctx = canvas.getContext("2d");
  const DPR = Math.min(window.devicePixelRatio || 1, 2);
  const TAU = Math.PI * 2;

  let W = 0, H = 0, S = 1;
  let WH = 0; // world height: the field continues below the first screen
  let neurons = [];
  let edges = [];
  let pulses = [];
  let rings = [];
  let network = null; // static canvas of all axons
  let center = null;
  let pointer = { x: -1e4, y: -1e4 };
  let hovered = null, focused = null, navHovered = null;
  let nextFire = 0;
  let last = performance.now();
  let departing = false;

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // ---- Geometry -----------------------------------------------------------

  function growBranch(out, rng, x, y, ang, w, len, depth, maxDepth, bend, scale) {
    const steps = 4 + ((rng() * 4) | 0);
    const seg = len / steps;
    const pts = [[x, y, w]];
    let a = ang;
    for (let i = 0; i < steps; i++) {
      a += (rng() - 0.5) * 0.55 + bend;
      x += Math.cos(a) * seg;
      y += Math.sin(a) * seg;
      w = Math.max(0.35 * scale, w * (depth === 0 ? 0.86 : 0.92));
      pts.push([x, y, w]);
      // Dendritic spines sprout from all but the primary shafts.
      if (depth >= 1) {
        const n = rng() < 0.6 ? 1 + ((rng() * 2) | 0) : 0;
        for (let k = 0; k < n; k++) {
          const back = seg * rng();
          const side = rng() < 0.5 ? 1 : -1;
          out.spines.push({
            x: x - Math.cos(a) * back,
            y: y - Math.sin(a) * back,
            a: a + side * (1.05 + rng() * 0.7),
            l: (1.6 + rng() * 2.8) * scale,
            r: (0.5 + rng() * 0.6) * scale,
          });
        }
      }
    }
    out.branches.push({ pts, depth });
    if (depth < maxDepth && w > 0.45 * scale) {
      const k = rng() < 0.8 ? 2 : 1;
      const spread = 0.3 + rng() * 0.45;
      for (let j = 0; j < k; j++) {
        const na = k === 1 ? a + (rng() - 0.5) * 0.4 : a + (j ? 1 : -1) * spread * (0.6 + rng() * 0.6);
        growBranch(out, rng, x, y, na, w * 0.8, len * (0.6 + rng() * 0.25), depth + 1, maxDepth,
          bend * 0.5 + (rng() - 0.5) * 0.05, scale);
      }
    }
  }

  function buildGeometry(n) {
    const rng = mulberry32(n.seed);
    const R = n.R, reach = n.reach, sc = n.scale;
    const geo = { branches: [], spines: [], soma: [], nissl: [], nucleus: null };

    // Primary dendrite directions.
    const prim = [];
    let apical = null;
    if (n.kind === "pyramidal") {
      apical = -Math.PI / 2 + (rng() - 0.5) * 0.6;
      prim.push({ a: apical, w: R * 0.5, len: reach * 0.5, depth: 4 });
      const basal = 4 + ((rng() * 2) | 0);
      for (let i = 0; i < basal; i++) {
        const a = apical + Math.PI + (i / (basal - 1) - 0.5) * 3.2 + (rng() - 0.5) * 0.3;
        prim.push({ a, w: R * (0.28 + rng() * 0.1), len: (reach / 2.9) * (0.7 + rng() * 0.4), depth: 3 });
      }
    } else {
      const count = n.kind === "grand" ? 9 : 6 + ((rng() * 3) | 0);
      const off = rng() * TAU;
      for (let i = 0; i < count; i++) {
        const a = off + (i / count) * TAU + (rng() - 0.5) * 0.5;
        prim.push({ a, w: R * (0.26 + rng() * 0.16), len: (reach / 2.6) * (0.7 + rng() * 0.5), depth: 4 });
      }
    }

    // Axon hillock leaves through the widest gap between dendrites.
    const sortedA = prim.map((p) => ((p.a % TAU) + TAU) % TAU).sort((a, b) => a - b);
    let gap = 0, axonA = 0;
    for (let i = 0; i < sortedA.length; i++) {
      const a0 = sortedA[i], a1 = i + 1 < sortedA.length ? sortedA[i + 1] : sortedA[0] + TAU;
      if (a1 - a0 > gap) { gap = a1 - a0; axonA = (a0 + a1) / 2; }
    }
    n.axonAngle = axonA;

    // Soma outline: low harmonics for irregularity, bulges where dendrites
    // and the hillock pull the membrane outward, a teardrop for pyramidal cells.
    const harm = [2, 3, 4, 5].map((k) => ({ k, a: (0.02 + rng() * 0.07) / (k * 0.5), p: rng() * TAU }));
    const angDist = (a, b) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
    const somaR = (th) => {
      let r = 1;
      for (const h of harm) r += h.a * Math.sin(h.k * th + h.p);
      for (const p of prim) r += (p.w / R) * 0.55 * Math.exp(-Math.pow(angDist(th, p.a) / 0.28, 2));
      r += 0.25 * Math.exp(-Math.pow(angDist(th, axonA) / 0.3, 2));
      if (apical != null) r *= 1 + 0.35 * Math.pow(Math.max(0, Math.cos(th - apical)), 3);
      return R * r;
    };
    const SN = 72;
    for (let i = 0; i < SN; i++) {
      const th = (i / SN) * TAU;
      const r = somaR(th);
      geo.soma.push([Math.cos(th) * r, Math.sin(th) * r]);
    }

    for (const p of prim) {
      const r0 = somaR(p.a) * 0.82;
      growBranch(geo, rng, Math.cos(p.a) * r0, Math.sin(p.a) * r0, p.a, p.w, p.len, 0,
        p.depth, (rng() - 0.5) * 0.08, sc);
    }

    // Hillock tapering into the initial segment of the axon.
    const hill = [];
    let hx = Math.cos(axonA) * somaR(axonA) * 0.8, hy = Math.sin(axonA) * somaR(axonA) * 0.8;
    let ha = axonA, hw = R * 0.42;
    for (let i = 0; i < 14; i++) {
      const seg = i < 4 ? R * 0.22 : R * 0.3;
      ha += (rng() - 0.5) * 0.18;
      hx += Math.cos(ha) * seg; hy += Math.sin(ha) * seg;
      hw = i < 4 ? hw * 0.62 : Math.max(0.5 * sc, hw * 0.93);
      hill.push([hx, hy, hw]);
    }
    geo.branches.push({ pts: [[Math.cos(axonA) * R * 0.5, Math.sin(axonA) * R * 0.5, R * 0.5], ...hill], depth: 0, axon: true });

    // Nucleus sits off-centre, away from the hillock.
    const nOff = R * 0.12;
    geo.nucleus = {
      x: -Math.cos(axonA) * nOff, y: -Math.sin(axonA) * nOff,
      rx: R * (0.4 + rng() * 0.06), ry: R * (0.34 + rng() * 0.06), rot: rng() * Math.PI,
      nx: (rng() - 0.5) * R * 0.14, ny: (rng() - 0.5) * R * 0.14,
    };
    // Nissl substance — granular texture in the cytoplasm.
    const nisslCount = Math.round(R * 1.4);
    for (let i = 0; i < nisslCount; i++) {
      const th = rng() * TAU, rr = Math.sqrt(rng()) * 0.85;
      const r = somaR(th) * rr;
      const x = Math.cos(th) * r, y = Math.sin(th) * r;
      const dx = (x - geo.nucleus.x) / geo.nucleus.rx, dy = (y - geo.nucleus.y) / geo.nucleus.ry;
      if (dx * dx + dy * dy < 1.2) continue;
      geo.nissl.push([x, y, (0.35 + rng() * 0.8) * sc]);
    }

    // Bounds for the offscreen sprite.
    let minX = 0, minY = 0, maxX = 0, maxY = 0;
    const grow = (x, y, pad) => {
      minX = Math.min(minX, x - pad); maxX = Math.max(maxX, x + pad);
      minY = Math.min(minY, y - pad); maxY = Math.max(maxY, y + pad);
    };
    for (const b of geo.branches) for (const [x, y, w] of b.pts) grow(x, y, w + 4);
    for (const [x, y] of geo.soma) grow(x, y, 4);
    const pad = 22 * sc;
    geo.bounds = { x: minX - pad, y: minY - pad, w: maxX - minX + pad * 2, h: maxY - minY + pad * 2 };
    return geo;
  }

  // ---- Rendering neurons to sprites ----------------------------------------

  const DARK = {
    body: (d) => `rgba(50,40,104,${0.92 - d * 0.12})`,
    hl: (d) => `rgba(128,110,205,${0.42 - d * 0.07})`,
    spine: "rgba(104,90,182,0.38)",
    soma: [[0, "#3b3176"], [0.5, "#211a4b"], [1, "#0e0a23"]],
    rim: "rgba(150,130,235,0.38)",
    nissl: "rgba(96,80,170,0.4)",
    nuc: [[0, "#312868"], [1, "#171233"]],
    nucRim: "rgba(125,105,210,0.3)",
    nucleolus: "rgba(160,140,235,0.45)",
  };
  const LIT = {
    body: (d) => `rgba(112,140,255,${0.95 - d * 0.1})`,
    hl: (d) => `rgba(215,238,255,${0.95 - d * 0.12})`,
    spine: "rgba(190,222,255,0.85)",
    soma: [[0, "#fffaf0"], [0.28, "#e4dbff"], [0.62, "#9b7bff"], [1, "#4b2fa8"]],
    rim: "rgba(205,235,255,0.95)",
    nissl: "rgba(255,255,255,0.4)",
    nuc: [[0, "rgba(255,240,200,1)"], [1, "rgba(255,196,120,0.55)"]],
    nucRim: "rgba(255,230,180,0.8)",
    nucleolus: "#ffffff",
  };

  function paintNeuron(g, geo, pal) {
    g.lineCap = "round";
    g.lineJoin = "round";
    const branches = geo.branches.slice().sort((a, b) => b.depth - a.depth);

    // Body pass — the membrane silhouette.
    for (const b of branches) {
      g.strokeStyle = pal.body(b.depth);
      for (let i = 1; i < b.pts.length; i++) {
        const [x0, y0, w0] = b.pts[i - 1], [x1, y1, w1] = b.pts[i];
        g.lineWidth = (w0 + w1) * 0.62 + 0.5;
        g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      }
    }
    // Spines: thin necks with bulbous heads.
    g.strokeStyle = pal.spine;
    g.fillStyle = pal.spine;
    g.lineWidth = 0.55;
    for (const s of geo.spines) {
      const ex = s.x + Math.cos(s.a) * s.l, ey = s.y + Math.sin(s.a) * s.l;
      g.beginPath(); g.moveTo(s.x, s.y); g.lineTo(ex, ey); g.stroke();
      g.beginPath(); g.arc(ex, ey, s.r, 0, TAU); g.fill();
    }
    // Highlight pass — light from the upper left gives the processes volume.
    for (const b of branches) {
      g.strokeStyle = pal.hl(b.depth);
      for (let i = 1; i < b.pts.length; i++) {
        const [x0, y0, w0] = b.pts[i - 1], [x1, y1, w1] = b.pts[i];
        const w = (w0 + w1) * 0.5;
        const o = w * 0.22;
        g.lineWidth = Math.max(0.3, w * 0.4);
        g.beginPath(); g.moveTo(x0 - o, y0 - o); g.lineTo(x1 - o, y1 - o); g.stroke();
      }
    }

    // Soma.
    const pts = geo.soma;
    const R = Math.hypot(pts[0][0], pts[0][1]) || 1;
    g.beginPath();
    for (let i = 0; i <= pts.length; i++) {
      const p = pts[i % pts.length], q = pts[(i + 1) % pts.length];
      const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
      if (i === 0) g.moveTo(mx, my); else g.quadraticCurveTo(p[0], p[1], mx, my);
    }
    g.closePath();
    const maxR = pts.reduce((m, p) => Math.max(m, Math.hypot(p[0], p[1])), R);
    const grad = g.createRadialGradient(-maxR * 0.3, -maxR * 0.35, 0, 0, 0, maxR * 1.15);
    for (const [o, c] of pal.soma) grad.addColorStop(o, c);
    g.fillStyle = grad;
    g.fill();
    g.strokeStyle = pal.rim;
    g.lineWidth = 1;
    g.stroke();

    g.save();
    g.clip();
    g.fillStyle = pal.nissl;
    for (const [x, y, r] of geo.nissl) { g.beginPath(); g.arc(x, y, r, 0, TAU); g.fill(); }
    g.restore();

    const nu = geo.nucleus;
    g.save();
    g.translate(nu.x, nu.y);
    g.rotate(nu.rot);
    const ng = g.createRadialGradient(-nu.rx * 0.3, -nu.ry * 0.3, 0, 0, 0, nu.rx);
    for (const [o, c] of pal.nuc) ng.addColorStop(o, c);
    g.fillStyle = ng;
    g.beginPath(); g.ellipse(0, 0, nu.rx, nu.ry, 0, 0, TAU); g.fill();
    g.strokeStyle = pal.nucRim;
    g.lineWidth = 0.8;
    g.stroke();
    g.fillStyle = pal.nucleolus;
    g.beginPath(); g.arc(nu.nx, nu.ny, nu.rx * 0.22, 0, TAU); g.fill();
    g.restore();
  }

  function sprite(geo, pal, dpr, glow) {
    const b = geo.bounds;
    const c = document.createElement("canvas");
    c.width = Math.ceil(b.w * dpr);
    c.height = Math.ceil(b.h * dpr);
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, -b.x * dpr, -b.y * dpr);
    paintNeuron(g, geo, pal);
    if (!glow) return c;
    // Bloom: composite the sprite over a shadowed copy of itself.
    const out = document.createElement("canvas");
    out.width = c.width; out.height = c.height;
    const o = out.getContext("2d");
    o.shadowColor = "rgba(127,216,255,0.95)";
    o.shadowBlur = glow * dpr;
    o.drawImage(c, 0, 0);
    o.shadowColor = "rgba(155,123,255,0.8)";
    o.shadowBlur = glow * 2.2 * dpr;
    o.drawImage(c, 0, 0);
    o.shadowBlur = 0;
    o.drawImage(c, 0, 0);
    return out;
  }

  // ---- Axons between neurons ---------------------------------------------

  function buildEdge(a, b, rng) {
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
    const ux = dx / d, uy = dy / d, nx = -uy, ny = ux;
    const sx = a.x + ux * a.R * 0.95, sy = a.y + uy * a.R * 0.95;
    const stop = Math.min(b.reach * 0.55, d * 0.3);
    const ex = b.x - ux * stop, ey = b.y - uy * stop;
    const bow1 = (rng() - 0.5) * 0.5 * d, bow2 = (rng() - 0.5) * 0.5 * d;
    const c1x = sx + dx * 0.3 + nx * bow1, c1y = sy + dy * 0.3 + ny * bow1;
    const c2x = sx + dx * 0.7 + nx * bow2, c2y = sy + dy * 0.7 + ny * bow2;
    const N = Math.max(40, Math.round(d / 5));
    const freq = 3 + rng() * 5, ph = rng() * TAU, amp = (2 + rng() * 4) * S;
    const pts = [];
    for (let i = 0; i <= N; i++) {
      const t = i / N, mt = 1 - t;
      let x = mt * mt * mt * sx + 3 * mt * mt * t * c1x + 3 * mt * t * t * c2x + t * t * t * ex;
      let y = mt * mt * mt * sy + 3 * mt * mt * t * c1y + 3 * mt * t * t * c2y + t * t * t * ey;
      const wig = Math.sin(t * freq * TAU / 2 + ph) * amp * Math.sin(Math.PI * t);
      x += nx * wig; y += ny * wig;
      pts.push([x, y]);
    }
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const total = cum[cum.length - 1];

    // Myelin internodes separated by nodes of Ranvier.
    const internode = 20 * S, gapLen = 3.2 * S;
    const nodes = [];
    const sheaths = [];
    for (let p = total * 0.1; p < total * 0.86; p += internode + gapLen) {
      sheaths.push([p, Math.min(p + internode, total * 0.86)]);
      nodes.push(p + internode + gapLen / 2);
    }

    // Terminal arbor fanning into the target's dendritic field.
    const end = pts[pts.length - 1], prev = pts[pts.length - 4];
    const ea = Math.atan2(end[1] - prev[1], end[0] - prev[0]);
    const terminals = [];
    const tn = 3 + ((rng() * 3) | 0);
    for (let i = 0; i < tn; i++) {
      const a = ea + (i / (tn - 1) - 0.5) * 1.5 + (rng() - 0.5) * 0.3;
      const l = (10 + rng() * 16) * S;
      const mx = end[0] + Math.cos(a - 0.2) * l * 0.5, my = end[1] + Math.sin(a - 0.2) * l * 0.5;
      terminals.push({ x0: end[0], y0: end[1], mx, my, x1: end[0] + Math.cos(a) * l, y1: end[1] + Math.sin(a) * l });
    }

    const alpha = Math.min(a.depth, b.depth);
    return { a, b, pts, cum, total, sheaths, nodes, terminals, alpha, termGlow: 0 };
  }

  function pointAt(e, d) {
    d = clamp(d, 0, e.total);
    let lo = 0, hi = e.cum.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (e.cum[mid] < d) lo = mid; else hi = mid;
    }
    const span = e.cum[hi] - e.cum[lo] || 1;
    const t = (d - e.cum[lo]) / span;
    return [e.pts[lo][0] + (e.pts[hi][0] - e.pts[lo][0]) * t, e.pts[lo][1] + (e.pts[hi][1] - e.pts[lo][1]) * t];
  }

  function strokeRange(g, e, d0, d1) {
    g.beginPath();
    const p0 = pointAt(e, d0);
    g.moveTo(p0[0], p0[1]);
    for (let i = 0; i < e.pts.length; i++) {
      if (e.cum[i] > d0 && e.cum[i] < d1) g.lineTo(e.pts[i][0], e.pts[i][1]);
    }
    const p1 = pointAt(e, d1);
    g.lineTo(p1[0], p1[1]);
    g.stroke();
  }

  function drawTerminals(g, e, color, bouton) {
    g.strokeStyle = color;
    g.fillStyle = color;
    g.lineWidth = 0.8 * S;
    for (const t of e.terminals) {
      g.beginPath(); g.moveTo(t.x0, t.y0); g.quadraticCurveTo(t.mx, t.my, t.x1, t.y1); g.stroke();
      g.beginPath(); g.arc(t.x1, t.y1, bouton * S, 0, TAU); g.fill();
    }
  }

  function paintNetwork() {
    const c = document.createElement("canvas");
    c.width = Math.round(W * DPR); c.height = Math.round(WH * DPR);
    const g = c.getContext("2d");
    g.setTransform(DPR, 0, 0, DPR, 0, 0);
    g.lineCap = "round";
    g.lineJoin = "round";
    for (const e of edges) {
      g.globalAlpha = e.alpha;
      g.strokeStyle = "rgba(96,86,172,0.4)";
      g.lineWidth = 0.9 * S;
      strokeRange(g, e, 0, e.total);
      for (const [p0, p1] of e.sheaths) {
        g.strokeStyle = "rgba(52,42,112,0.75)";
        g.lineWidth = 3 * S;
        strokeRange(g, e, p0, p1);
        g.strokeStyle = "rgba(140,124,222,0.24)";
        g.lineWidth = 1 * S;
        strokeRange(g, e, p0, p1);
      }
      drawTerminals(g, e, "rgba(118,104,200,0.55)", 1.7);
    }
    g.globalAlpha = 1;
    return c;
  }

  // ---- Layout -------------------------------------------------------------

  function navHeight() {
    const bar = document.querySelector(".crossbar");
    return bar ? bar.offsetHeight : 64;
  }

  function layout() {
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = Math.round(W * DPR);
    canvas.height = Math.round(H * DPR);
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    S = clamp(Math.min(W, H * 1.3) / 1000, 0.55, 1.2);
    WH = Math.round(H * 1.55);
    document.body.style.height = WH + "px";
    const hint = document.querySelector(".hint");
    if (hint) hint.style.top = H - 64 + "px";

    const top = navHeight();
    const cx = W / 2, cy = top + (H - top) / 2;
    const modR = 30 * S, modReach = 96 * S;
    const portrait = W < H * 0.9;
    const ry = (H - top) / 2 - modR * 2.2 - 34;
    // Leave room for the widest label at the screen edge.
    const rx = portrait ? W / 2 - Math.max(modR * 2.4, 84) : Math.min(W * 0.36, ry * 1.9);

    neurons = [];
    center = makeNeuron({
      x: cx, y: cy, R: 64 * S, reach: 170 * S, kind: "grand", seed: 7, depth: 1, scale: S,
      role: "center", base: 0.22,
    });
    neurons.push(center);

    MODULES.forEach((m, i) => {
      const a = (m.angle * Math.PI) / 180;
      neurons.push(makeNeuron({
        x: cx + Math.cos(a) * rx, y: cy + Math.sin(a) * ry,
        R: modR, reach: modReach, kind: i % 2 ? "stellate" : "pyramidal",
        seed: 101 + i * 37, depth: 1, scale: S, role: "module", module: m, base: 0.04,
      }));
    });

    // Background neurons fill the void at lower depth.
    const rng = mulberry32(4242);
    const target = clamp(Math.round((W * H) / 70000), 6, 20);
    let tries = 0;
    while (neurons.length < 6 + target && tries++ < 900) {
      const depth = 0.42 + rng() * 0.38;
      const R = (14 + rng() * 10) * S * depth * 1.4;
      const reach = (60 + rng() * 40) * S * depth * 1.3;
      const x = -0.04 * W + rng() * W * 1.08;
      const y = top * 0.6 + rng() * (H - top * 0.6) * 1.04;
      const ok = neurons.every((n) => Math.hypot(n.x - x, n.y - y) > (n.reach + reach) * (n.role ? 0.62 : 0.8));
      if (!ok) continue;
      neurons.push(makeNeuron({
        x, y, R, reach, kind: rng() < 0.45 ? "pyramidal" : "stellate",
        seed: 9000 + neurons.length * 131, depth, scale: S * depth * 1.3, base: 0.03,
      }));
    }

    // A few more drift in the deeper void below the fold.
    const deep = clamp(Math.round((W * (WH - H)) / 80000), 4, 10);
    const before = neurons.length;
    tries = 0;
    while (neurons.length < before + deep && tries++ < 900) {
      const depth = 0.4 + rng() * 0.45;
      const R = (14 + rng() * 12) * S * depth * 1.4;
      const reach = (60 + rng() * 50) * S * depth * 1.3;
      const x = 0.02 * W + rng() * W * 0.96;
      const y = H * 0.98 + rng() * (WH - H * 0.98 - reach * 0.5);
      const ok = neurons.every((n) => Math.hypot(n.x - x, n.y - y) > (n.reach + reach) * (n.role ? 0.62 : 0.85));
      if (!ok) continue;
      neurons.push(makeNeuron({
        x, y, R, reach, kind: rng() < 0.45 ? "pyramidal" : "stellate",
        seed: 17000 + neurons.length * 131, depth, scale: S * depth * 1.3, base: 0.03,
      }));
    }

    // Wire it: the title neuron to every module, modules in a ring,
    // background neurons to their nearest neighbours.
    edges = [];
    const erng = mulberry32(77);
    const link = (a, b) => {
      if (a === b || a.links.has(b)) return;
      const e = buildEdge(a, b, erng);
      a.links.add(b); b.links.add(a);
      a.edges.push(e); b.edges.push(e);
      edges.push(e);
    };
    const mods = neurons.filter((n) => n.role === "module");
    mods.forEach((m, i) => {
      link(center, m);
      link(m, mods[(i + 1) % mods.length]);
    });
    for (const n of neurons.filter((q) => !q.role)) {
      const near = neurons.filter((q) => q !== n)
        .sort((p, q) => Math.hypot(p.x - n.x, p.y - n.y) - Math.hypot(q.x - n.x, q.y - n.y));
      link(n, near[0]);
      if (rng() < 0.6) link(near[1], n);
    }

    network = paintNetwork();
    placeOverlays();
  }

  function makeNeuron(o) {
    const n = Object.assign({ hover: 0, flash: 0, act: 0, phase: Math.random() * TAU, links: new Set(), edges: [], lastPing: 0 }, o);
    n.geo = buildGeometry(n);
    const dpr = n.role ? DPR : 1;
    n.dark = sprite(n.geo, DARK, dpr, 0);
    n.lit = sprite(n.geo, LIT, dpr, 10 * n.scale);
    return n;
  }

  // ---- HTML overlays (real links for labels, keyboard, and screen readers)

  const nodeEls = new Map();

  function placeOverlays() {
    for (const n of neurons) {
      if (n.role === "module") {
        let el = nodeEls.get(n.module.id);
        if (!el) {
          el = document.createElement("a");
          el.className = "neuron-node";
          el.href = n.module.href;
          el.dataset.module = n.module.id;
          el.innerHTML = `<span class="label">${n.module.label}</span>`;
          el.addEventListener("focus", () => { focused = n; });
          el.addEventListener("blur", () => { if (focused === n) focused = null; });
          el.addEventListener("click", (e) => {
            if (e.metaKey || e.ctrlKey || e.shiftKey) return;
            e.preventDefault();
            activate(n);
          });
          document.body.appendChild(el);
          nodeEls.set(n.module.id, el);
        }
        el._neuron = n;
        const size = n.R * 3.1;
        Object.assign(el.style, { left: n.x + "px", top: n.y + "px", width: size + "px", height: size + "px" });
      }
    }
    const core = document.querySelector(".neuron-core");
    if (core) {
      const size = center.R * 2.4;
      Object.assign(core.style, { left: center.x + "px", top: center.y + "px", width: size + "px", height: size + "px" });
      if (!core._wired) {
        core._wired = true;
        core.addEventListener("focus", () => { focused = center; });
        core.addEventListener("blur", () => { if (focused === center) focused = null; });
        core.addEventListener("click", (e) => {
          e.preventDefault();
          if (window.Mind) window.Mind.click();
          fire(center, 3, null, 1, true);
        });
      }
    }
    // Crossbar links echo onto their neurons.
    document.querySelectorAll(".crossbar a.mod[data-module]").forEach((a) => {
      if (a._wired) return;
      a._wired = true;
      const find = () => neurons.find((n) => n.module && n.module.id === a.dataset.module);
      a.addEventListener("mouseenter", () => { navHovered = find(); });
      a.addEventListener("mouseleave", () => { navHovered = null; });
      a.addEventListener("focus", () => { navHovered = find(); });
      a.addEventListener("blur", () => { navHovered = null; });
      a.addEventListener("click", (e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey) return;
        e.preventDefault();
        const n = find();
        if (n) activate(n);
      });
    });
  }

  // ---- Signalling ---------------------------------------------------------

  function fire(n, hops, fromEdge, strength, all) {
    n.flash = Math.max(n.flash, strength);
    if (!reduceMotion) rings.push({ x: n.x, y: n.y, R: n.R, age: 0, s: strength });
    if (hops <= 0 || pulses.length > 30) return;
    let out = n.edges.filter((e) => e !== fromEdge);
    out.sort(() => Math.random() - 0.5);
    if (!all) out = out.slice(0, Math.random() < 0.55 ? 1 : 2);
    for (const e of out) {
      const forward = e.a === n;
      pulses.push({
        e, forward, d: 0, speed: rand(260, 360) * S,
        trail: 70 * S,
        arrive: () => {
          if (forward) e.termGlow = 1;
          if (window.Mind) window.Mind.exchange();
          fire(forward ? e.b : e.a, hops - 1, e, strength * 0.92, false);
        },
      });
    }
  }

  function activate(n) {
    if (departing) return;
    departing = true;
    n.flash = 1;
    fire(n, 2, null, 1, true);
    const el = nodeEls.get(n.module.id);
    if (el) el.classList.add("lit");
    setTimeout(() => window.Cosmos.depart(n.module.href, n.x, n.y - window.scrollY), reduceMotion ? 0 : 420);
  }

  function hitTest(x, y) {
    let best = null, bestD = Infinity;
    for (const n of neurons) {
      const d = Math.hypot(n.x - x, n.y - y);
      const r = Math.max(n.R * (n.role ? 1.35 : 1.7), 16);
      const score = d / r;
      if (score < 1 && score < bestD) { best = n; bestD = score; }
    }
    return best;
  }

  // ---- Frame --------------------------------------------------------------

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const t = now / 1000;

    // A spontaneous volley roughly every ten seconds.
    if (!reduceMotion && now > nextFire) {
      const pool = neurons.filter((n) => n.edges.length);
      fire(pool[(Math.random() * pool.length) | 0], 3 + ((Math.random() * 2) | 0), null, 0.9, false);
      nextFire = now + rand(8000, 12000);
    }

    // Clamped so iOS overscroll bounce never reads outside the network canvas.
    const sy = clamp(window.scrollY, 0, Math.max(0, WH - H));
    const over = hitTest(pointer.x, pointer.y + sy);
    if (over !== hovered) {
      hovered = over;
      // Brushing a neuron sends a small volley to its neighbours.
      if (over && now - over.lastPing > 1400 && !reduceMotion) {
        over.lastPing = now;
        fire(over, 1, null, 0.6, false);
      }
    }
    document.body.style.cursor = hovered && !hovered.role ? "crosshair" : "";

    for (const n of neurons) {
      const want = n === hovered || n === focused || n === navHovered ? 1 : 0;
      n.hover += (want - n.hover) * Math.min(1, dt * 7);
      n.flash *= Math.exp(-dt * 1.35);
      const breath = n.base * (0.7 + 0.3 * Math.sin(t * 0.7 + n.phase));
      n.act = clamp(Math.max(n.hover, n.flash) + breath, 0, 1);
    }

    ctx.setTransform(DPR, 0, 0, DPR, 0, -sy * DPR);
    ctx.clearRect(0, sy, W, H);

    const drawN = (n) => {
      const b = n.geo.bounds;
      if (n.y + b.y + b.h < sy || n.y + b.y > sy + H) return;
      ctx.globalAlpha = n.depth;
      ctx.drawImage(n.dark, n.x + b.x, n.y + b.y, b.w, b.h);
      if (n.act > 0.005) {
        ctx.globalAlpha = n.act * n.depth;
        ctx.drawImage(n.lit, n.x + b.x, n.y + b.y, b.w, b.h);
      }
    };

    // Background neurons, then the axon web, then the labelled neurons.
    for (const n of neurons) if (!n.role) drawN(n);
    ctx.globalAlpha = 1;
    ctx.drawImage(network, 0, sy * DPR, W * DPR, H * DPR, 0, sy, W, H);
    for (const n of neurons) if (n.role) drawN(n);

    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round";

    // Axons of active neurons glow faintly.
    for (const e of edges) {
      const k = Math.max(e.a.hover, e.b.hover) * 0.3;
      if (k > 0.01) {
        ctx.globalAlpha = k * e.alpha;
        ctx.strokeStyle = "rgba(127,216,255,0.9)";
        ctx.lineWidth = 1.4 * S;
        strokeRange(ctx, e, 0, e.total);
      }
      if (e.termGlow > 0.01) {
        ctx.globalAlpha = e.termGlow * e.alpha;
        drawTerminals(ctx, e, "rgba(243,217,164,0.95)", 2.2);
        e.termGlow *= Math.exp(-dt * 2.2);
      }
    }

    // Action potentials: a bright head, a fading trail, and nodes of Ranvier
    // sparking in sequence as the signal leaps between them.
    for (const p of pulses) {
      p.d += p.speed * dt;
      const e = p.e;
      const pos = (d) => (p.forward ? d : e.total - d);
      ctx.globalAlpha = e.alpha;
      const steps = 12;
      for (let k = 0; k < steps; k++) {
        const d0 = p.d - (p.trail * k) / steps, d1 = p.d - (p.trail * (k + 1)) / steps;
        if (d0 <= 0) break;
        const f = 1 - k / steps;
        ctx.strokeStyle = `rgba(${Math.round(160 + 60 * f)},${Math.round(200 + 30 * f)},255,${f * f * 0.9})`;
        ctx.lineWidth = (0.6 + 2.2 * f) * S;
        const a0 = pos(Math.max(0, d1)), a1 = pos(d0);
        strokeRange(ctx, e, Math.min(a0, a1), Math.max(a0, a1));
      }
      for (const nd of e.nodes) {
        const behind = p.d - (p.forward ? nd : e.total - nd);
        if (behind < 0 || behind > 160 * S) continue;
        const [x, y] = pointAt(e, nd);
        glowDot(x, y, 7 * S, `rgba(243,217,164,${0.8 * Math.exp(-behind / (45 * S))})`);
      }
      const [hx, hy] = pointAt(e, pos(Math.min(p.d, e.total)));
      glowDot(hx, hy, 11 * S, window.Mind && window.Mind.nous ? "rgba(255,230,170,0.95)" : "rgba(235,248,255,0.95)");
    }
    ctx.globalAlpha = 1;
    const arrived = pulses.filter((p) => p.d >= p.e.total);
    pulses = pulses.filter((p) => p.d < p.e.total);
    arrived.forEach((p) => p.arrive());

    // Soma halos and action-potential shockwaves.
    for (const n of neurons) {
      if (n.act < 0.03) continue;
      const r = n.R * 4;
      const g = ctx.createRadialGradient(n.x, n.y, n.R * 0.4, n.x, n.y, r);
      g.addColorStop(0, `rgba(155,123,255,${0.32 * n.act * n.depth})`);
      g.addColorStop(0.4, `rgba(127,216,255,${0.1 * n.act * n.depth})`);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, TAU); ctx.fill();
    }
    for (const r of rings) {
      r.age += dt;
      const k = r.age / 1.2;
      if (k >= 1 || r.s <= 0) continue;
      ctx.strokeStyle = `rgba(190,170,255,${(1 - k) * 0.45 * r.s})`;
      ctx.lineWidth = 1.4 * S * (1 - k) + 0.3;
      ctx.beginPath(); ctx.arc(r.x, r.y, r.R * (1.1 + k * 3), 0, TAU); ctx.stroke();
    }
    rings = rings.filter((r) => r.age < 1.2);
    ctx.globalCompositeOperation = "source-over";

    for (const [, el] of nodeEls) el.classList.toggle("lit", el._neuron.act > 0.45);

    requestAnimationFrame(frame);
  }

  function glowDot(x, y, r, color) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, color);
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
  }

  // ---- Boot ---------------------------------------------------------------

  window.addEventListener("pointermove", (e) => { pointer.x = e.clientX; pointer.y = e.clientY; });
  window.addEventListener("pointerleave", () => { pointer.x = pointer.y = -1e4; });
  document.addEventListener("mouseleave", () => { pointer.x = pointer.y = -1e4; });

  // Background neurons react to clicks too — they just don't go anywhere.
  canvas.addEventListener("click", (e) => {
    const n = hitTest(e.clientX, e.clientY + window.scrollY);
    if (!n) return;
    if (window.Mind) window.Mind.click();
    if (n === center) fire(center, 3, null, 1, true);
    else if (n.role === "module") activate(n);
    else fire(n, 2, null, 1, false);
  });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (window.innerWidth === W && Math.abs(window.innerHeight - H) < 120) return;
      pulses = []; rings = []; layout();
    }, 150);
  });

  window.addEventListener("pageshow", (e) => { if (e.persisted) departing = false; });

  const start = () => {
    layout();
    nextFire = performance.now() + 2500; // first spontaneous volley shortly after arrival
    requestAnimationFrame(frame);
  };
  // Wait for the fonts so overlay labels measure correctly.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(start);
  else start();
})();
