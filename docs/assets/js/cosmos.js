/* ==========================================================================
   Cosmos — the ambient void behind every page.
   A pre-rendered Milky Way band (nebula glow, dust lanes, star field) with
   live twinkling, drifting motes, and quantum particles that flicker through
   superposition before collapsing out of existence.
   Also wires the shared crossbar menu and page-transition veil.
   ========================================================================== */
(function () {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const canvas = document.getElementById("cosmos");
  const ctx = canvas.getContext("2d");
  const DPR = Math.min(window.devicePixelRatio || 1, 2);
  const MARGIN = 40; // extra field around the viewport so parallax never shows an edge

  let W = 0, H = 0;
  let sky = null;          // offscreen static layer
  let twinklers = [];
  let motes = [];
  let quanta = [];
  let mouse = { x: 0.5, y: 0.5, sx: 0.5, sy: 0.5 };
  let last = performance.now();

  const rand = (a, b) => a + Math.random() * (b - a);
  const gauss = () => {
    let u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  // The galactic band runs diagonally across the field.
  function bandFrame(w, h) {
    const angle = -0.42;
    return {
      cx: w * 0.5, cy: h * 0.52,
      ux: Math.cos(angle), uy: Math.sin(angle),   // along the band
      nx: -Math.sin(angle), ny: Math.cos(angle),  // across the band
      len: Math.hypot(w, h),
      width: Math.min(w, h) * 0.16,
    };
  }

  function bandPoint(b, t, spread) {
    // Slight S-curve so the band feels organic rather than ruled.
    const along = t * b.len * 0.6;
    const bend = Math.sin(t * 2.4) * b.width * 0.35;
    const across = spread * b.width + bend;
    return [b.cx + b.ux * along + b.nx * across, b.cy + b.uy * along + b.ny * across];
  }

  function buildSky() {
    const w = W + MARGIN * 2, h = H + MARGIN * 2;
    const off = document.createElement("canvas");
    off.width = Math.round(w * DPR);
    off.height = Math.round(h * DPR);
    const g = off.getContext("2d");
    g.scale(DPR, DPR);

    // Deep base with a faint violet vignette.
    const base = g.createRadialGradient(w * 0.5, h * 0.5, 0, w * 0.5, h * 0.5, Math.hypot(w, h) * 0.6);
    base.addColorStop(0, "#07051a");
    base.addColorStop(0.5, "#03020b");
    base.addColorStop(1, "#010005");
    g.fillStyle = base;
    g.fillRect(0, 0, w, h);

    const b = bandFrame(w, h);
    const area = w * h;

    // Nebula glow along the band — many soft additive clouds.
    g.globalCompositeOperation = "lighter";
    const palette = [
      [107, 63, 160], [42, 47, 122], [31, 111, 139], [160, 64, 110], [70, 40, 140], [30, 70, 130],
    ];
    const clouds = Math.round(90 + area / 16000);
    for (let i = 0; i < clouds; i++) {
      const t = gauss() * 0.55;
      const [x, y] = bandPoint(b, t, gauss() * 0.45);
      const core = Math.exp(-t * t * 6); // brighter, warmer near the galactic core
      const r = rand(40, 180) * (0.6 + core * 0.8);
      let [cr, cg, cb] = palette[(Math.random() * palette.length) | 0];
      if (Math.random() < core * 0.5) [cr, cg, cb] = [201, 163, 106];
      const a = rand(0.007, 0.024) * (0.45 + core);
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, `rgba(${cr},${cg},${cb},${a})`);
      grad.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
      g.fillStyle = grad;
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      g.fill();
    }

    // A few isolated nebulae away from the band.
    for (let i = 0; i < 3; i++) {
      const x = rand(0, w), y = rand(0, h), r = rand(120, 320);
      const [cr, cg, cb] = palette[(Math.random() * palette.length) | 0];
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, `rgba(${cr},${cg},${cb},0.03)`);
      grad.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
      g.fillStyle = grad;
      g.fillRect(x - r, y - r, r * 2, r * 2);
    }

    // Dust lanes: dark filaments threading the band's core.
    g.globalCompositeOperation = "source-over";
    for (let i = 0; i < 260; i++) {
      const t = rand(-0.9, 0.9);
      const [x, y] = bandPoint(b, t, Math.sin(t * 7) * 0.12 + gauss() * 0.06);
      const r = rand(10, 46);
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, "rgba(2,1,6,0.16)");
      grad.addColorStop(1, "rgba(2,1,6,0)");
      g.fillStyle = grad;
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      g.fill();
    }

    // Star field — dense in the band, sparse elsewhere.
    g.globalCompositeOperation = "lighter";
    const stars = Math.round(area / 430);
    for (let i = 0; i < stars; i++) {
      let x, y;
      if (Math.random() < 0.55) {
        [x, y] = bandPoint(b, rand(-1, 1), gauss() * 0.5);
      } else {
        x = rand(0, w); y = rand(0, h);
      }
      const r = Math.random() < 0.97 ? rand(0.2, 0.8) : rand(0.8, 1.5);
      const hue = Math.random();
      const col = hue < 0.6 ? "255,255,255" : hue < 0.82 ? "190,210,255" : "255,226,190";
      g.fillStyle = `rgba(${col},${rand(0.25, 0.9)})`;
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      g.fill();
    }

    // Bright stars with diffraction glints.
    for (let i = 0; i < Math.round(area / 120000) + 3; i++) {
      const x = rand(0, w), y = rand(0, h), r = rand(1, 2.2);
      const grad = g.createRadialGradient(x, y, 0, x, y, r * 8);
      grad.addColorStop(0, "rgba(255,255,255,0.9)");
      grad.addColorStop(0.15, "rgba(200,210,255,0.35)");
      grad.addColorStop(1, "rgba(160,140,255,0)");
      g.fillStyle = grad;
      g.beginPath();
      g.arc(x, y, r * 8, 0, Math.PI * 2);
      g.fill();
      g.strokeStyle = "rgba(210,220,255,0.25)";
      g.lineWidth = 0.6;
      g.beginPath();
      g.moveTo(x - r * 10, y); g.lineTo(x + r * 10, y);
      g.moveTo(x, y - r * 10); g.lineTo(x, y + r * 10);
      g.stroke();
    }
    g.globalCompositeOperation = "source-over";
    sky = off;

    // Live twinkling stars, biased toward the band like the static ones.
    twinklers = [];
    const tCount = Math.round(area / 15000);
    for (let i = 0; i < tCount; i++) {
      let x, y;
      if (Math.random() < 0.55) [x, y] = bandPoint(b, rand(-1, 1), gauss() * 0.5);
      else { x = rand(0, w); y = rand(0, h); }
      twinklers.push({ x, y, r: rand(0.5, 1.4), phase: rand(0, 6.28), speed: rand(0.6, 2.2) });
    }

    motes = [];
    for (let i = 0; i < Math.round(area / 42000); i++) {
      motes.push({
        x: rand(0, w), y: rand(0, h),
        vx: rand(-4, 4), vy: rand(-3, 3),
        r: rand(0.6, 1.8), phase: rand(0, 6.28),
        hue: Math.random() < 0.5 ? "155,123,255" : "127,216,255",
      });
    }
  }

  // ---- Quantum particles --------------------------------------------------
  // Each particle is born in superposition: several ghost states jitter around
  // a probability cloud, then the wavefunction collapses to one bright point
  // with a brief ring, and it winks out. Some are born as virtual pairs that
  // drift apart and re-annihilate.

  function spawnQuantum() {
    const w = W + MARGIN * 2, h = H + MARGIN * 2;
    const pair = Math.random() < 0.3;
    const q = {
      x: rand(0, w), y: rand(0, h),
      age: 0,
      life: rand(2.2, 4.2),
      spread: rand(6, 16),
      ghosts: [],
      hue: Math.random() < 0.5 ? [127, 216, 255] : Math.random() < 0.6 ? [190, 150, 255] : [243, 217, 164],
      pair,
      angle: rand(0, Math.PI * 2),
    };
    const n = pair ? 2 : 3 + ((Math.random() * 3) | 0);
    for (let i = 0; i < n; i++) q.ghosts.push({ ox: gauss(), oy: gauss(), ph: rand(0, 6.28) });
    quanta.push(q);
  }

  function drawQuantum(q, t) {
    const p = q.age / q.life;
    const [r, gC, bC] = q.hue;
    const fadeIn = Math.min(1, p / 0.15);
    const collapseAt = 0.72;

    if (q.pair) {
      // Virtual particle pair: separate, then annihilate with a flash.
      const sep = Math.sin(Math.min(1, p / 0.85) * Math.PI) * q.spread * 2.4;
      const ax = Math.cos(q.angle) * sep, ay = Math.sin(q.angle) * sep;
      const a = fadeIn * (p < 0.85 ? 0.8 : 0);
      for (const s of [1, -1]) {
        dot(q.x + ax * s, q.y + ay * s, 1.3, `rgba(${r},${gC},${bC},${a})`, 6);
      }
      if (a > 0) {
        ctx.strokeStyle = `rgba(${r},${gC},${bC},${a * 0.18})`;
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        ctx.moveTo(q.x - ax, q.y - ay);
        ctx.quadraticCurveTo(q.x + ay * 0.6, q.y - ax * 0.6, q.x + ax, q.y + ay);
        ctx.stroke();
      }
      if (p >= 0.85) flash(q, (p - 0.85) / 0.15);
      return;
    }

    if (p < collapseAt) {
      // Superposition: ghosts shimmer inside a faint probability cloud.
      const cloud = ctx.createRadialGradient(q.x, q.y, 0, q.x, q.y, q.spread * 2.2);
      cloud.addColorStop(0, `rgba(${r},${gC},${bC},${0.07 * fadeIn})`);
      cloud.addColorStop(1, `rgba(${r},${gC},${bC},0)`);
      ctx.fillStyle = cloud;
      ctx.beginPath();
      ctx.arc(q.x, q.y, q.spread * 2.2, 0, Math.PI * 2);
      ctx.fill();
      for (const gh of q.ghosts) {
        const jitter = Math.sin(t * 9 + gh.ph) * 0.35;
        const gx = q.x + (gh.ox + jitter) * q.spread;
        const gy = q.y + (gh.oy - jitter) * q.spread;
        const flick = 0.35 + 0.35 * Math.sin(t * 13 + gh.ph * 3);
        dot(gx, gy, 1.1, `rgba(${r},${gC},${bC},${flick * fadeIn})`, 5);
      }
    } else {
      flash(q, (p - collapseAt) / (1 - collapseAt));
    }
  }

  function flash(q, k) {
    // Collapse: a single point surges, a ring expands, and it is gone.
    const [r, gC, bC] = q.hue;
    const a = 1 - k;
    dot(q.x, q.y, 1.6 + (1 - k) * 1.2, `rgba(255,255,255,${a})`, 10);
    ctx.strokeStyle = `rgba(${r},${gC},${bC},${a * 0.5})`;
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.arc(q.x, q.y, 2 + k * q.spread * 1.8, 0, Math.PI * 2);
    ctx.stroke();
  }

  function dot(x, y, r, color, glow) {
    const grad = ctx.createRadialGradient(x, y, 0, x, y, r * glow);
    grad.addColorStop(0, color);
    grad.addColorStop(0.25, color.replace(/[\d.]+\)$/, (m) => (parseFloat(m) * 0.35).toFixed(3) + ")"));
    grad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, r * glow, 0, Math.PI * 2);
    ctx.fill();
  }

  // ---- Loop ---------------------------------------------------------------

  function resize() {
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = Math.round(W * DPR);
    canvas.height = Math.round(H * DPR);
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    buildSky();
    if (reduceMotion) frame(performance.now());
  }

  let spawnClock = 0;
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const t = now / 1000;

    // Ease the parallax toward the pointer.
    mouse.sx += (mouse.x - mouse.sx) * 0.03;
    mouse.sy += (mouse.y - mouse.sy) * 0.03;
    const px = -MARGIN + (0.5 - mouse.sx) * MARGIN * 0.9;
    const py = -MARGIN + (0.5 - mouse.sy) * MARGIN * 0.9;

    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    ctx.drawImage(sky, px, py, W + MARGIN * 2, H + MARGIN * 2);

    ctx.save();
    ctx.translate(px, py);
    ctx.globalCompositeOperation = "lighter";

    for (const s of twinklers) {
      const a = 0.15 + 0.6 * Math.pow(0.5 + 0.5 * Math.sin(t * s.speed + s.phase), 3);
      ctx.fillStyle = `rgba(230,235,255,${a})`;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    const w = W + MARGIN * 2, h = H + MARGIN * 2;
    for (const m of motes) {
      m.x = (m.x + m.vx * dt + w) % w;
      m.y = (m.y + m.vy * dt + h) % h;
      const a = 0.18 + 0.14 * Math.sin(t * 0.7 + m.phase);
      dot(m.x, m.y, m.r, `rgba(${m.hue},${a})`, 4);
    }

    if (!reduceMotion) {
      spawnClock -= dt;
      if (spawnClock <= 0 && quanta.length < 20) {
        spawnQuantum();
        spawnClock = rand(0.12, 0.5);
      }
      for (const q of quanta) { q.age += dt; drawQuantum(q, t); }
      quanta = quanta.filter((q) => q.age < q.life);
    }

    ctx.restore();
    if (!reduceMotion) requestAnimationFrame(frame);
  }

  window.addEventListener("pointermove", (e) => {
    mouse.x = e.clientX / Math.max(1, W);
    mouse.y = e.clientY / Math.max(1, H);
  });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 150);
  });

  resize();
  if (!reduceMotion) requestAnimationFrame(frame);

  // ---- Shared chrome: crossbar menu + page veil ----------------------------

  const bar = document.querySelector(".crossbar");
  const toggle = document.querySelector(".menu-toggle");
  if (bar && toggle) {
    toggle.addEventListener("click", () => {
      const open = bar.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  // ---- Thought counter ------------------------------------------------------
  // Every 10 exchanges between neurons form one thought. Clicking neurons 20
  // times within 10 seconds enters nous, which doubles each new thought for
  // as long as the clicking keeps up. The count is kept in localStorage so it
  // follows the visitor across pages and visits.

  const Mind = (function () {
    const KEY = "jw.thoughts";
    const PER_THOUGHT = 10;
    const NOUS_CLICKS = 20, NOUS_WINDOW = 10000, NOUS_GAP = 1500;
    const el = document.querySelector(".thoughts");
    let thoughts = 0, exchanges = 0, nous = false;
    let clicks = [], nousTimer = null, bumpTimer = null;

    try {
      const saved = JSON.parse(localStorage.getItem(KEY));
      if (saved && Number.isFinite(saved.thoughts)) {
        thoughts = Math.max(0, Math.floor(saved.thoughts));
        exchanges = clamp0(Math.floor(saved.exchanges) || 0, PER_THOUGHT - 1);
      }
    } catch (e) { /* storage unavailable: start fresh */ }

    function clamp0(v, max) { return Math.max(0, Math.min(max, v)); }

    function save() {
      try { localStorage.setItem(KEY, JSON.stringify({ thoughts, exchanges })); } catch (e) { /* ignore */ }
    }

    function render(bump) {
      if (!el) return;
      el.querySelector(".t-count").textContent = thoughts.toLocaleString();
      el.querySelector(".t-label").textContent = thoughts === 1 ? "thought" : "thoughts";
      el.style.setProperty("--p", (exchanges / PER_THOUGHT) * 100 + "%");
      el.classList.toggle("nous", nous);
      document.body.classList.toggle("nous", nous);
      if (bump) {
        el.classList.remove("bump");
        void el.offsetWidth; // restart the animation
        el.classList.add("bump");
        clearTimeout(bumpTimer);
        bumpTimer = setTimeout(() => el.classList.remove("bump"), 900);
      }
    }

    render(false);

    return {
      exchange() {
        exchanges++;
        let formed = false;
        if (exchanges >= PER_THOUGHT) {
          exchanges = 0;
          thoughts += nous ? 2 : 1;
          formed = true;
        }
        render(formed);
        save();
      },
      click() {
        const now = performance.now();
        clicks.push(now);
        clicks = clicks.filter((t) => now - t < NOUS_WINDOW);
        if (!nous && clicks.length >= NOUS_CLICKS) { nous = true; render(false); }
        clearTimeout(nousTimer);
        nousTimer = setTimeout(() => { nous = false; clicks = []; render(false); }, NOUS_GAP);
      },
      get nous() { return nous; },
    };
  })();
  window.Mind = Mind;

  // Fade out through the veil before following internal links.
  window.Cosmos = {
    depart(href, x, y) {
      const veil = document.querySelector(".veil");
      if (veil && x != null) {
        veil.style.setProperty("--vx", x + "px");
        veil.style.setProperty("--vy", y + "px");
      }
      document.body.classList.add("leaving");
      setTimeout(() => { window.location.href = href; }, reduceMotion ? 0 : 520);
    },
  };

  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[data-veil]");
    if (!a || e.defaultPrevented || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    window.Cosmos.depart(a.getAttribute("href"), e.clientX, e.clientY);
  });

  requestAnimationFrame(() => document.body.classList.add("ready"));
  // Returning via the back button restores a page frozen mid-fade.
  window.addEventListener("pageshow", (e) => {
    if (e.persisted) document.body.classList.remove("leaving");
  });
})();
