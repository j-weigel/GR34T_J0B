/* ==========================================================================
   Zoom out — the nebula brain.

   Pinching out on a trackpad (or touch screen) pulls the camera back: the
   neural network shrinks into the void while a brain made of ~60k glowing
   particles resolves around it. The brain can be dragged to turn a full 360°
   in any direction; pinching in, double-clicking, or pressing Esc returns to
   the neurons.

   The anatomy is procedural but follows real landmarks and proportions:
   two hemispheres split by the longitudinal fissure; frontal, parietal,
   occipital, and temporal lobes; the central and lateral (Sylvian) sulci;
   meandering gyri and sulci over the cortex; the cerebellum with its
   transverse folia tucked beneath the occipital lobes; and the brainstem —
   midbrain, the bulging pons, the medulla — tapering into the spinal cord.

   Coordinates while building are roughly centimetres: x = right, y = up,
   z = anterior (toward the forehead).
   ========================================================================== */
(function () {
  "use strict";

  const canvas = document.getElementById("brain");
  if (!canvas) return;
  const gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false });
  const toggle = document.querySelector(".zoom-toggle");
  if (!gl) {
    // Without WebGL there is no brain to show; leave pinch-zoom to the browser.
    if (toggle) toggle.hidden = true;
    return;
  }

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const DPR = Math.min(window.devicePixelRatio || 1, 2);
  const TAU = Math.PI * 2;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const smooth = (a, b, v) => { const t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const mix = (a, b, t) => a + (b - a) * t;
  const mix3 = (a, b, t) => [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)];

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // 3D simplex noise (after Stefan Gustavson).
  function makeNoise(rng) {
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) { const j = (rng() * (i + 1)) | 0; const t = p[i]; p[i] = p[j]; p[j] = t; }
    const perm = new Uint8Array(512);
    for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
    const G = [1,1,0, -1,1,0, 1,-1,0, -1,-1,0, 1,0,1, -1,0,1, 1,0,-1, -1,0,-1, 0,1,1, 0,-1,1, 0,1,-1, 0,-1,-1];
    const F3 = 1 / 3, G3 = 1 / 6;
    const corner = (gi, x, y, z) => {
      let t = 0.6 - x * x - y * y - z * z;
      if (t < 0) return 0;
      gi = (gi % 12) * 3;
      t *= t;
      return t * t * (G[gi] * x + G[gi + 1] * y + G[gi + 2] * z);
    };
    return function (x, y, z) {
      const s = (x + y + z) * F3;
      const i = Math.floor(x + s), j = Math.floor(y + s), k = Math.floor(z + s);
      const t = (i + j + k) * G3;
      const x0 = x - (i - t), y0 = y - (j - t), z0 = z - (k - t);
      let i1, j1, k1, i2, j2, k2;
      if (x0 >= y0) {
        if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
        else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
        else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
      } else {
        if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
        else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
        else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
      }
      const ii = i & 255, jj = j & 255, kk = k & 255;
      return 32 * (
        corner(perm[ii + perm[jj + perm[kk]]], x0, y0, z0) +
        corner(perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]], x0 - i1 + G3, y0 - j1 + G3, z0 - k1 + G3) +
        corner(perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]], x0 - i2 + 2 * G3, y0 - j2 + 2 * G3, z0 - k2 + 2 * G3) +
        corner(perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]], x0 - 1 + 3 * G3, y0 - 1 + 3 * G3, z0 - 1 + 3 * G3)
      );
    };
  }

  // ---- Anatomy --------------------------------------------------------------

  const COLORS = {
    frontal: [0.62, 0.49, 1.0],
    parietal: [0.5, 0.84, 1.0],
    occipital: [0.42, 0.52, 1.0],
    temporal: [0.9, 0.56, 0.83],
    cerebellum: [0.96, 0.85, 0.63],
    stem: [0.46, 0.9, 0.82],
  };

  function buildBrain() {
    const rng = mulberry32(20260928);
    const noise = makeNoise(rng);
    const gauss = () => {
      let u = 0, v = 0;
      while (!u) u = rng();
      while (!v) v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
    };

    const pos = [], col = [], size = [], phase = [], nrm = [];
    // n: outward surface normal, or null for volumetric haze (lit evenly).
    const emit = (x, y, z, c, bright, sz, n) => {
      const j = 0.92 + rng() * 0.16;
      pos.push(x, y, z);
      if (n) nrm.push(n[0], n[1], n[2]); else nrm.push(0, 0, 0);
      col.push(c[0] * bright * j, c[1] * bright * j, c[2] * bright * j);
      size.push(sz);
      phase.push(rng() * TAU);
    };

    // Area-uniform sample on an ellipsoid surface (radii r), returned as an offset.
    // With sq > 2 the ellipsoid swells toward a rounded box (a superellipsoid),
    // which gives the hemispheres their full, squared-off crown.
    const onEllipsoid = (r, sq) => {
      const minR = Math.min(r[0], r[1], r[2]);
      for (;;) {
        let ux = gauss(), uy = gauss(), uz = gauss();
        const l = Math.hypot(ux, uy, uz);
        ux /= l; uy /= l; uz /= l;
        const w = Math.sqrt((ux / r[0]) ** 2 + (uy / r[1]) ** 2 + (uz / r[2]) ** 2) * minR;
        if (rng() < w) {
          if (sq) {
            const m = Math.pow(Math.abs(ux) ** sq + Math.abs(uy) ** sq + Math.abs(uz) ** sq, 1 / sq);
            ux /= m; uy /= m; uz /= m;
          }
          return [ux * r[0], uy * r[1], uz * r[2]];
        }
      }
    };
    const normalOf = (d, r) => {
      const nx = d[0] / (r[0] * r[0]), ny = d[1] / (r[1] * r[1]), nz = d[2] / (r[2] * r[2]);
      const l = Math.hypot(nx, ny, nz) || 1;
      return [nx / l, ny / l, nz / l];
    };
    const tilt = (o, a) => [o[0], o[1] * Math.cos(a) - o[2] * Math.sin(a), o[1] * Math.sin(a) + o[2] * Math.cos(a)];
    const inside = (p, e) => {
      const d = tilt([p[0] - e.c[0], p[1] - e.c[1], p[2] - e.c[2]], -e.tilt);
      const q = e.sq || 2;
      return Math.pow(Math.abs(d[0] / e.r[0]) ** q + Math.abs(d[1] / e.r[1]) ** q + Math.abs(d[2] / e.r[2]) ** q, 2 / q);
    };

    const cerebellum = { c: [0, -3.35, -5.3], r: [4.7, 2.05, 2.6], tilt: 0 };

    // Cortical folding: the zero set of warped 3D noise traces meandering
    // grooves (sulci); everything between them reads as gyri.
    const fold = (x, y, z) => {
      const f = 0.5;
      const wx = x * f + 0.55 * noise(y * 0.31 + 11, z * 0.31, x * 0.31);
      const wy = y * f + 0.55 * noise(z * 0.31, x * 0.31 + 23, y * 0.31);
      const wz = z * f * 0.8 + 0.55 * noise(x * 0.31, y * 0.31, z * 0.31 + 37);
      return noise(wx, wy, wz);
    };

    for (const s of [1, -1]) {
      // Centred close to the midline so the medial edges stand nearly full
      // height and the longitudinal fissure stays a narrow cleft.
      const main = { c: [s * 2.15, 1.2, -0.3], r: [4.05, 4.3, 8.1], tilt: 0, sq: 2.4 };
      const temporal = { c: [s * 4.05, -1.9, 1.1], r: [2.2, 1.85, 4.1], tilt: 0.2 };
      const parts = [
        { e: main, n: 60000, kind: "main", sq: main.sq },
        { e: temporal, n: 17000, kind: "temporal", sq: 0 },
      ];

      for (const part of parts) {
        const others = parts.filter((q) => q !== part).map((q) => q.e);
        let made = 0, guard = 0;
        while (made < part.n && guard++ < part.n * 6) {
          const raw = onEllipsoid(part.e.r, part.sq);
          const o = tilt(raw, part.e.tilt);
          let x = part.e.c[0] + o[0], y = part.e.c[1] + o[1], z = part.e.c[2] + o[2];
          const p0 = [x, y, z];
          // Keep only the outer surface of the union of lobes; the slight
          // tolerance leaves a crease where they meet (the lateral sulcus).
          if (others.some((e) => inside(p0, e) < 0.94)) continue;

          let X = s * x; // distance from the midline in this hemisphere
          // Frontal lobe: narrower toward the pole, flat orbital underside.
          const fk = smooth(2.5, 8, z) ** 2;
          X = 0.2 + (X - 0.2) * (1 - 0.2 * fk);
          if (y < -0.2) y = -0.2 + (y + 0.2) * (1 - 0.42 * smooth(1.5, 4.5, z));
          y -= 0.5 * fk;
          // Occipital lobe tapers to the occipital pole.
          const ok = smooth(-3.5, -8, z) ** 2;
          X = 0.2 + (X - 0.2) * (1 - 0.28 * ok);
          y = 1.0 + (y - 1.0) * (1 - 0.22 * ok);
          // Parietal crown sits highest.
          if (y > 2) y += 0.35 * Math.exp(-(((z + 1.5) / 3.5) ** 2)) * smooth(2, 5, y);
          // Medial wall of the hemisphere, facing the longitudinal fissure.
          const medial = X < 0.26;
          if (medial) X = 0.2 + rng() * 0.06;

          // Surface normal, for sinking sulci and for lighting.
          let [nx, ny, nz] = tilt(normalOf(raw, part.e.r), part.e.tilt);
          if (medial) { nx = -s; ny = 0; nz = 0; }

          // Gyri: broad rounded ridges (h → 1 on the crown) between narrow
          // sulci along the fold pattern's zero set.
          const n = fold(X, y, z);
          const W = 0.3;
          let h = smooth(0, W, Math.abs(n));

          // Central sulcus: runs from the crown, just behind the midpoint,
          // obliquely down and forward toward the lateral sulcus.
          const zc = -1.1 + 0.42 * X + 0.25 * Math.sin(y * 1.4 + X);
          if (part.kind === "main" && y > -0.3 && !medial) h = Math.min(h, smooth(0.04, 0.45, Math.abs(z - zc)));
          // Lateral (Sylvian) fissure: the frontal/parietal rim dips where it overhangs the temporal lobe.
          if (part.kind === "main") {
            const v = inside([x, y, z], temporal);
            if (v < 1.4) h = Math.min(h, smooth(0.94, 1.4, v));
          }
          // Sulci are mostly empty, which is what makes the folds legible.
          const sulcus = 1 - smooth(0.15, 0.45, h);
          if (rng() < sulcus) continue;
          // The medial wall is almost always hidden behind the other hemisphere;
          // keep it sparse and dim so it doesn't ghost through.
          if (medial && rng() < 0.65) continue;

          // Relief: tilt the normal down the flanks of each gyrus so the key
          // light rakes across the folds.
          if (!medial && Math.abs(n) < W) {
            const eps = 0.06;
            const gx = (fold(X + eps, y, z) - n) / eps * s;
            const gy = (fold(X, y + eps, z) - n) / eps;
            const gz = (fold(X, y, z + eps) - n) / eps;
            const t = Math.abs(n) / W;
            const k = 0.34 * 6 * t * (1 - t) / W * Math.sign(n);
            let hx = gx * k, hy = gy * k, hz = gz * k;
            const along = hx * nx + hy * ny + hz * nz;
            hx -= along * nx; hy -= along * ny; hz -= along * nz;
            nx -= hx; ny -= hy; nz -= hz;
            const l = Math.hypot(nx, ny, nz) || 1;
            nx /= l; ny /= l; nz /= l;
          }
          const lift = (h - 0.65) * 0.34;
          x = s * X + nx * lift; y += ny * lift; z += nz * lift;
          let bright = 0.3 + 0.9 * Math.pow(h, 1.5);
          if (medial) bright *= 0.3;

          let c;
          if (part.kind === "temporal") {
            c = COLORS.temporal;
          } else {
            const toParietal = smooth(zc - 0.6, zc + 0.6, z);
            const occEdge = -4.9 + 0.25 * y;
            c = mix3(COLORS.parietal, COLORS.frontal, toParietal);
            c = mix3(COLORS.occipital, c, smooth(occEdge - 0.8, occEdge + 0.8, z));
          }
          // Gyral crests catch a little more light.
          c = mix3(c, [1, 1, 1], 0.22 * smooth(0.6, 1, h));
          emit(x, y, z, c, bright * 0.8, 0.0048 + 0.0022 * h + rng() * 0.001, [nx, ny, nz]);
          made++;
        }
      }

      // Nebular interior: a faint cloud filling the hemisphere.
      for (let i = 0; i < 2600; i++) {
        const o = [gauss() * 0.45, gauss() * 0.45, gauss() * 0.45];
        const x = main.c[0] + o[0] * main.r[0], y = main.c[1] + o[1] * main.r[1], z = main.c[2] + o[2] * main.r[2];
        if (inside([x, y, z], main) > 0.8 || s * x < 0.4) continue;
        const c = z > 0 ? COLORS.frontal : z > -4.5 ? COLORS.parietal : COLORS.occipital;
        emit(x, y, z, c, 0.12, 0.012 + rng() * 0.01);
      }
    }

    // Cerebellum, with the fine transverse folia that give it its layered look.
    {
      const e = cerebellum;
      let made = 0, guard = 0;
      while (made < 16000 && guard++ < 120000) {
        const o = onEllipsoid(e.r, 0);
        let x = e.c[0] + o[0], y = e.c[1] + o[1], z = e.c[2] + o[2];
        if (y > e.c[1]) y = e.c[1] + (y - e.c[1]) * 0.78; // flattened beneath the tentorium
        const ax = Math.abs(x);
        if (ax < 0.8 && z < e.c[2]) z += 0.4 * (1 - ax / 0.8); // posterior notch at the vermis
        if (ax < 1.2 && z > e.c[2] + 1.2) continue; // clear the front for the brainstem
        const ang = Math.atan2(y - e.c[1], z - e.c[2]);
        const folia = 0.5 + 0.5 * Math.sin(ang * 30 + noise(x * 0.4, y * 0.4, z * 0.4) * 2.2);
        if (rng() < (1 - folia) * 0.55) continue;
        const bright = 0.3 + 0.7 * folia * folia;
        emit(x, y, z, mix3(COLORS.cerebellum, [1, 1, 1], 0.2 * folia), bright * 0.8, 0.004 + rng() * 0.002, normalOf(o, e.r));
        made++;
      }
    }

    // Brainstem: midbrain → pons (anterior bulge with transverse fibres) →
    // medulla → spinal cord, angling down and slightly back.
    for (let i = 0; i < 9000; i++) {
      const t = rng() * 1.45;
      const cy = -0.6 - 6.2 * t;
      const cz = -1.4 - 1.1 * t - 0.4 * t * t;
      const pons = Math.exp(-(((t - 0.36) / 0.13) ** 2));
      const r = t > 1 ? 0.62 - 0.1 * (t - 1) : 0.72 + 0.45 * (1 - t) + 0.55 * pons;
      const th = rng() * TAU;
      const front = Math.max(0, Math.sin(th));
      const x = Math.cos(th) * r;
      const z = cz + Math.sin(th) * r * 0.85 + 0.45 * pons * front;
      const y = cy;
      if (inside([x, y, z], cerebellum) < 1) continue;
      const fibres = 0.6 + 0.4 * Math.sin(y * 11) * pons;
      const fade = t > 1 ? 1 - (t - 1) / 0.45 : 1;
      emit(x, y, z, COLORS.stem, 0.65 * fibres * fade, 0.0042 + rng() * 0.002, [Math.cos(th), 0, Math.sin(th)]);
    }

    // Gas: a few large, faint sprites that give the whole brain its nebular haze.
    const count = pos.length / 3;
    for (let i = 0; i < 1100; i++) {
      const k = (rng() * count) | 0;
      const c = [col[k * 3], col[k * 3 + 1], col[k * 3 + 2]];
      const m = Math.max(c[0], c[1], c[2]) || 1;
      emit(pos[k * 3] + gauss() * 0.4, pos[k * 3 + 1] + gauss() * 0.4, pos[k * 3 + 2] + gauss() * 0.4,
        [c[0] / m, c[1] / m, c[2] / m], 0.05, 0.05 + rng() * 0.05);
    }
    // Stardust drifting around it.
    for (let i = 0; i < 2600; i++) {
      const d = 9 + Math.abs(gauss()) * 5;
      let ux = gauss(), uy = gauss(), uz = gauss();
      const l = Math.hypot(ux, uy, uz);
      emit((ux / l) * d, (uy / l) * d * 0.8, (uz / l) * d, rng() < 0.5 ? COLORS.parietal : COLORS.frontal, 0.25, 0.0025);
    }

    // Centre and normalise so the brain spans roughly [-1, 1].
    const P = new Float32Array(pos.length);
    const CY = -1.2, CZ = -0.4, K = 1 / 8.4;
    for (let i = 0; i < pos.length; i += 3) {
      P[i] = pos[i] * K;
      P[i + 1] = (pos[i + 1] - CY) * K;
      P[i + 2] = (pos[i + 2] - CZ) * K;
    }
    return {
      pos: P, col: new Float32Array(col), size: new Float32Array(size), phase: new Float32Array(phase),
      nrm: new Float32Array(nrm), count: pos.length / 3,
    };
  }

  // ---- WebGL ----------------------------------------------------------------

  const VERT = `
    attribute vec3 aPos;
    attribute vec3 aCol;
    attribute float aSize;
    attribute float aPhase;
    attribute vec3 aNrm;
    uniform mat3 uRot;
    uniform float uZoom;
    uniform float uFocal;
    uniform vec2 uNdc;
    uniform float uPx;
    uniform float uTime;
    uniform float uAlpha;
    varying vec3 vCol;
    varying float vA;
    void main() {
      vec3 p = uRot * aPos * uZoom;
      float depth = 3.2 - p.z;
      if (depth < 0.08) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
      float f = uFocal / depth;
      gl_Position = vec4(p.xy * f * uNdc, 0.0, 1.0);
      float px = aSize * uZoom * f * uPx;
      gl_PointSize = clamp(px, 1.0, 64.0);
      // Sub-pixel points dim instead of shrinking; large ones spread their light.
      float energy = px < 1.0 ? px : 1.0;
      // Surfaces facing away fade out, as if the brain were opaque; a key light
      // from the upper left brings out the relief of the folds.
      float lit = 0.75;
      if (dot(aNrm, aNrm) > 0.1) {
        vec3 n = uRot * aNrm;
        float facing = smoothstep(-0.3, 0.3, n.z);
        float lambert = max(0.0, dot(n, normalize(vec3(-0.45, 0.55, 0.7))));
        lit = mix(0.06, 0.35 + 0.85 * lambert, facing);
      }
      float back = lit * mix(0.6, 1.0, smoothstep(-1.0, 1.0, p.z / uZoom));
      float twinkle = 0.8 + 0.2 * sin(uTime * 1.3 + aPhase);
      vCol = aCol;
      vA = uAlpha * energy * back * twinkle;
    }
  `;
  const FRAG = `
    precision mediump float;
    varying vec3 vCol;
    varying float vA;
    void main() {
      vec2 d = gl_PointCoord - 0.5;
      float r2 = dot(d, d) * 4.0;
      if (r2 > 1.0) discard;
      float a = exp(-r2 * 3.2) * vA;
      gl_FragColor = vec4(vCol * a, a * 0.6);
    }
  `;

  function shader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, shader(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const U = {};
  for (const n of ["uRot", "uZoom", "uFocal", "uNdc", "uPx", "uTime", "uAlpha"]) U[n] = gl.getUniformLocation(prog, n);

  let brain = null;
  function upload() {
    brain = buildBrain();
    const bind = (name, data, n) => {
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0);
    };
    bind("aPos", brain.pos, 3);
    bind("aCol", brain.col, 3);
    bind("aSize", brain.size, 1);
    bind("aPhase", brain.phase, 1);
    bind("aNrm", brain.nrm, 3);
  }

  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);

  let W = 0, H = 0;
  function resize() {
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = Math.round(W * DPR);
    canvas.height = Math.round(H * DPR);
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  window.addEventListener("resize", resize);
  resize();

  // ---- View state -----------------------------------------------------------

  // progress: 0 = among the neurons, 1 = the whole brain in view.
  let progress = 0, target = 0, resting = 0, gesturing = false, gestureTimer = null;
  let yaw = -Math.PI / 2 + 0.55, pitch = 0.18, vYaw = 0, vPitch = 0;
  let dragging = false, lastX = 0, lastY = 0, lastMoveT = 0;

  const neural = document.getElementById("neural");
  const layer = document.querySelector(".neural-layer");

  function setProgress(v) {
    progress = clamp(v, 0, 1);
    target = progress;
    gesturing = true;
    clearTimeout(gestureTimer);
    gestureTimer = setTimeout(release, 180);
  }
  // Let go of a gesture: settle into whichever view it is closer to committing to.
  function release() {
    gesturing = false;
    target = resting === 0 ? (progress > 0.22 ? 1 : 0) : (progress < 0.78 ? 0 : 1);
  }
  function goTo(v) {
    if (!brain) upload();
    gesturing = false;
    target = v;
  }

  const ease = (t) => 1 - Math.pow(1 - t, 3);

  function applyView() {
    const e = ease(progress);
    // The network recedes toward the centre of the current view.
    const scale = 1 - 0.92 * e;
    const fade = 1 - smooth(0.05, 0.6, progress);
    if (neural) {
      neural.style.transform = progress ? `scale(${scale})` : "";
      neural.style.opacity = progress ? fade : "";
    }
    if (layer) {
      layer.style.transformOrigin = `${W / 2}px ${window.scrollY + H / 2}px`;
      layer.style.transform = progress ? `scale(${scale})` : "";
      layer.style.opacity = progress ? fade : "";
    }
    document.documentElement.style.overflow = progress > 0 ? "hidden" : "";
    const open = progress > 0.5;
    document.body.classList.toggle("brain-open", open);
    canvas.style.display = progress > 0.001 ? "block" : "none";
    if (toggle) {
      toggle.textContent = resting === 1 || open ? "Return" : "Zoom out";
      toggle.setAttribute("aria-pressed", String(open));
    }
  }

  function rotation() {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    // R = Rx(pitch) · Ry(yaw), column-major for WebGL.
    return new Float32Array([
      cy, sp * sy, -cp * sy,
      0, cp, sp,
      sy, -sp * cy, cp * cy,
    ]);
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;

    if (!gesturing && progress !== target) {
      const k = reduceMotion ? 1 : Math.min(1, dt * 3.2);
      progress += (target - progress) * k;
      if (Math.abs(target - progress) < 0.002) progress = target;
      if (progress === target) resting = target;
    }
    applyView();

    if (progress > 0.001 && brain) {
      if (!dragging) {
        yaw += vYaw * dt;
        pitch += vPitch * dt;
        vYaw *= Math.exp(-dt * 2.2);
        vPitch *= Math.exp(-dt * 2.2);
        if (!reduceMotion && Math.abs(vYaw) < 0.12) yaw += 0.1 * dt; // slow idle drift
      }
      const minDim = Math.min(W, H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniformMatrix3fv(U.uRot, false, rotation());
      gl.uniform1f(U.uZoom, mix(6, 1, ease(progress)));
      gl.uniform1f(U.uFocal, 2.55);
      gl.uniform2f(U.uNdc, minDim / W, minDim / H);
      gl.uniform1f(U.uPx, (minDim * DPR) / 2);
      gl.uniform1f(U.uTime, now / 1000);
      gl.uniform1f(U.uAlpha, Math.pow(progress, 1.3));
      gl.drawArrays(gl.POINTS, 0, brain.count);
    }
    requestAnimationFrame(frame);
  }

  // ---- Input ----------------------------------------------------------------

  // Chrome, Edge, and Firefox report a trackpad pinch as ctrl + wheel.
  window.addEventListener("wheel", (e) => {
    if (e.ctrlKey) {
      e.preventDefault();
      if (!brain) upload();
      const unit = e.deltaMode === 1 ? 16 : 1;
      setProgress(progress + e.deltaY * unit * 0.012);
    } else if (progress > 0.5) {
      // Two-finger scroll turns the brain.
      e.preventDefault();
      yaw += e.deltaX * 0.004;
      pitch += e.deltaY * 0.004;
    }
  }, { passive: false });

  // Safari reports pinches as gesture events.
  let gestureStart = 0;
  window.addEventListener("gesturestart", (e) => {
    e.preventDefault();
    if (!brain) upload();
    gestureStart = progress;
  });
  window.addEventListener("gesturechange", (e) => {
    e.preventDefault();
    setProgress(gestureStart + (1 - e.scale) * 1.4);
  });
  window.addEventListener("gestureend", (e) => {
    e.preventDefault();
    clearTimeout(gestureTimer);
    release();
  });

  // Touch screens: two-finger pinch.
  let pinch0 = 0, pinchStart = 0;
  const spread = (t) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
  window.addEventListener("touchstart", (e) => {
    if (e.touches.length === 2) {
      if (!brain) upload();
      pinch0 = spread(e.touches);
      pinchStart = progress;
    }
  }, { passive: true });
  window.addEventListener("touchmove", (e) => {
    if (e.touches.length === 2 && pinch0) {
      e.preventDefault();
      setProgress(pinchStart + (1 - spread(e.touches) / pinch0) * 1.3);
    }
  }, { passive: false });
  window.addEventListener("touchend", (e) => {
    if (pinch0 && e.touches.length < 2) {
      pinch0 = 0;
      clearTimeout(gestureTimer);
      release();
    }
  });

  // Drag to rotate, with momentum on release.
  canvas.addEventListener("pointerdown", (e) => {
    if (progress < 0.5) return;
    dragging = true;
    canvas.classList.add("dragging");
    canvas.setPointerCapture(e.pointerId);
    lastX = e.clientX; lastY = e.clientY; lastMoveT = performance.now();
    vYaw = vPitch = 0;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const now = performance.now();
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    const k = 3.6 / Math.min(W, H);
    yaw += dx * k;
    pitch += dy * k;
    const dts = Math.max(0.008, (now - lastMoveT) / 1000);
    vYaw = (dx * k) / dts;
    vPitch = (dy * k) / dts;
    lastX = e.clientX; lastY = e.clientY; lastMoveT = now;
  });
  const endDrag = () => {
    if (!dragging) return;
    dragging = false;
    canvas.classList.remove("dragging");
    if (performance.now() - lastMoveT > 90) vYaw = vPitch = 0; // held still before letting go
    vYaw = clamp(vYaw, -6, 6);
    vPitch = clamp(vPitch, -6, 6);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  // Double-click (or double-tap) returns to the neurons.
  canvas.addEventListener("dblclick", () => goTo(0));
  let lastTap = 0;
  canvas.addEventListener("pointerup", (e) => {
    if (e.pointerType !== "touch") return;
    const now = performance.now();
    if (now - lastTap < 320) goTo(0);
    lastTap = now;
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && progress > 0) goTo(0);
  });

  if (toggle) toggle.addEventListener("click", () => goTo(resting === 1 || progress > 0.5 ? 0 : 1));

  window.Brain = {
    get progress() { return progress; },
    zoomOut: () => goTo(1),
    zoomIn: () => goTo(0),
  };

  // Build the brain once the page has settled, so first paint stays quick.
  (window.requestIdleCallback || ((f) => setTimeout(f, 600)))(() => { if (!brain) upload(); });
  requestAnimationFrame(frame);
})();
