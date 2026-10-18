/* Lumière · SAM 2.1 Hiera Tiny in TensorFlow.js (WebGL)
 *
 * Tap-to-select for the Object mask part. A port of the image path of Meta's SAM 2
 * (https://github.com/facebookresearch/sam2, Apache-2.0): the Hiera image encoder with its
 * FPN neck, the prompt encoder, and the two-way-transformer mask decoder. The video memory
 * modules are not needed for a single photo and are left out.
 *
 * The weights come from tools/convert_sam2.py (models/sam2-tiny/). They stay in memory as
 * float16 and go to the GPU one block at a time, so the GPU never holds all of them at once.
 *
 *   const sam = await LumiereSAM2.load("models/sam2-tiny/", fraction => …);
 *   const feats = await sam.encode(imageOrCanvas);              // once per photo, the slow part
 *   const out = await sam.decode(feats, [[x, y, label], …]);    // per tap, fast
 *     // x, y in 0–1 of the photo; label 1 = include, 0 = exclude, 2 / 3 = box corners
 *     // out.logits: 256×256 mask logits for the chosen mask; out.score: predicted quality
 *   sam.release(feats);
 */
(function () {
  "use strict";
  const SIZE = 1024, EMB = 64;

  // float16 → float32 through a lookup table
  let F16 = null;
  function f16table() {
    if (F16) return F16; F16 = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      F16[h] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024); }
    return F16;
  }

  async function load(base, onProgress) {
    const man = await (await fetch(base + "manifest.json")).json();
    if (man.format !== "lumiere-sam2") throw new Error("Unknown selection model format.");
    const all = new Uint8Array(man.bytes); let got = 0;
    for (const f of man.files) {   // in order, so progress is meaningful on slow connections
      const r = await fetch(base + f); if (!r.ok) throw new Error("The object selection model could not be downloaded.");
      const b = new Uint8Array(await r.arrayBuffer()); all.set(b, got); got += b.length; onProgress?.(got / man.bytes);
    }
    const table = f16table(), raw = {};
    for (const e of man.tensors) raw[e.name] = e;
    const count = shape => shape.reduce((a, b) => a * b, 1);
    // Float32 copy of one stored tensor (dequantized if it was stored as int8).
    function f32(name) {
      const e = raw[name]; if (!e) throw new Error("missing weight " + name); const n = count(e.shape), out = new Float32Array(n);
      if (e.dtype === "float16") { const h = new Uint16Array(all.buffer, e.offset, n); for (let i = 0; i < n; i++) out[i] = table[h[i]]; }
      else { const q = new Int8Array(all.buffer, e.offset, n), [rows, cols] = e.shape, G = e.group, sc = new Uint16Array(all.buffer, e.scale, (rows / G) * cols);
        for (let r = 0; r < rows; r++) { const g = ((r / G) | 0) * cols; for (let c = 0; c < cols; c++) out[r * cols + c] = q[r * cols + c] * table[sc[g + c]]; } }
      return out;
    }
    const cache = new Map();
    // Encoder weights are uploaded when used and freed with the block; decoder weights stay.
    const W = (name, keep) => { if (cache.has(name)) return cache.get(name); const t = tf.tensor(f32(name), raw[name].shape); if (keep) cache.set(name, tf.keep(t)); return t; };
    const D = name => W(name, true);
    return model(man, W, D, f32, () => { for (const t of cache.values()) t.dispose(); cache.clear(); });
  }

  function model(man, W, D, f32, freeCache) {
    const linear = (x, k, w = W) => { const s = x.shape, y = tf.add(tf.matMul(x.reshape([-1, s[s.length - 1]]), w(k + ".w")), w(k + ".b")); return y.reshape([...s.slice(0, -1), y.shape[1]]); };
    // Memory: phones give a page little GPU memory, so every helper frees its temporary results
    // as soon as it returns (each runs in its own small tf.tidy), and big steps run in slices.
    const layerNorm = (x, k, eps, w = W) => tf.tidy(() => { const { mean, variance } = tf.moments(x, -1, true); return x.sub(mean).mul(tf.rsqrt(variance.add(eps))).mul(w(k + ".g")).add(w(k + ".b")); });
    const gelu = x => tf.tidy(() => x.mul(tf.erf(x.mul(Math.SQRT1_2)).add(1)).mul(0.5));   // exact GELU, as torch.nn.GELU
    // softmax(q kᵀ / √d) v on [batch, n, d] tensors, in slices of queries so the score matrix stays small (≤ 2 M values)
    function sdpa(q, k, v) {
      const [B, Nq, d] = q.shape, Nk = k.shape[1], LIMIT = 1 << 21, step = Math.max(1, Math.floor(LIMIT / (B * Nk)));
      const one = qq => tf.tidy(() => tf.matMul(tf.softmax(tf.matMul(qq.mul(1 / Math.sqrt(d)), k, false, true)), v));
      if (step >= Nq) return one(q);
      const parts = []; for (let i = 0; i < Nq; i += step) { const n = Math.min(step, Nq - i); parts.push(tf.tidy(() => one(q.slice([0, i, 0], [B, n, d])))); }
      const out = tf.concat(parts, 1); parts.forEach(t => t.dispose()); return out;
    }
    // Wait until the GPU has caught up: it releases freed memory only after running the queued work.
    const gpuWait = async t => { const p = t.slice(t.shape.map(() => 0), t.shape.map(() => 1)); await p.data(); p.dispose(); };

    /* ── Hiera trunk: blocks laid out as in hieradet.py ── */
    const cfg = man.trunk, stageEnds = cfg.stages.map((_, i) => cfg.stages.slice(0, i + 1).reduce((a, b) => a + b) - 1);
    const qPool = stageEnds.slice(0, -1).map(x => x + 1), blocks = [];
    { let dim = cfg.embed_dim, heads = cfg.heads, stage = 1;
      for (let i = 0; i < stageEnds.at(-1) + 1; i++) {
        let dimOut = dim; const window = cfg.global.includes(i) ? 0 : cfg.windows[stage - 1];
        if (stageEnds.includes(i - 1)) { dimOut = dim * 2; heads *= 2; stage++; }
        blocks.push({ dim, dimOut, heads, window, qStride: qPool.includes(i) }); dim = dimOut; } }

    function windowPartition(x, w) {
      const [, H, Wd, C] = x.shape, Hp = Math.ceil(H / w) * w, Wp = Math.ceil(Wd / w) * w;
      if (Hp > H || Wp > Wd) x = tf.pad(x, [[0, 0], [0, Hp - H], [0, Wp - Wd], [0, 0]]);
      return [x.reshape([Hp / w, w, Wp / w, w, C]).transpose([0, 2, 1, 3, 4]).reshape([-1, w, w, C]), [Hp, Wp]];
    }
    function windowUnpartition(x, w, [Hp, Wp], [H, Wd]) {
      const C = x.shape[3]; x = x.reshape([Hp / w, Wp / w, w, w, C]).transpose([0, 2, 1, 3, 4]).reshape([1, Hp, Wp, C]);
      return Hp > H || Wp > Wd ? x.slice([0, 0, 0, 0], [1, H, Wd, C]) : x;
    }
    function attention(x, k, b) {   // MultiScaleAttention on [B, h, w, C]; each intermediate is freed once its next form exists
      const [B, h, w] = x.shape, hd = b.dimOut / b.heads, flat = x.reshape([-1, b.dim]);
      // q, k and v as three products, instead of one combined result that is then split (three more copies)
      const wq = W(k + ".qkv.w"), wb = W(k + ".qkv.b"), ws = tf.split(wq, 3, 1), bs = tf.split(wb, 3);
      const proj = j => tf.tidy(() => tf.matMul(flat, ws[j]).add(bs[j]));
      const heads = (t, n) => tf.tidy(() => t.reshape([B, n, b.heads, hd]).transpose([0, 2, 1, 3]).reshape([B * b.heads, n, hd]));
      let q = proj(0), hq = h, wq2 = w;
      if (b.qStride) { const p = tf.tidy(() => tf.maxPool(q.reshape([B, h, w, b.dimOut]), 2, 2, "valid")); q.dispose(); q = p; [, hq, wq2] = q.shape; }
      const Q = heads(q, hq * wq2); q.dispose();
      const [K, V] = [1, 2].map(j => { const t = proj(j), r = heads(t, h * w); t.dispose(); return r; });
      [wq, wb, ...ws, ...bs].forEach(t => t.dispose()); flat.dispose();
      const o = sdpa(Q, K, V); [Q, K, V].forEach(t => t.dispose());
      const out = tf.tidy(() => linear(o.reshape([B, b.heads, hq * wq2, hd]).transpose([0, 2, 1, 3]).reshape([B, hq, wq2, b.dimOut]), k + ".proj")); o.dispose();
      return out;
    }
    // Blocks run in slices sized so that each temporary result holds at most about BUDGET values
    // (8 MB), waiting for the GPU after each slice so freed memory is really released.
    const BUDGET = 1 << 21;
    async function sliced(x, n, fn) {   // fn on slices of the first axis of x, results joined again
      const N = x.shape[0], rest = x.shape.slice(1), parts = [];
      for (let i = 0; i < N; i += n) { const m = Math.min(n, N - i);
        const part = tf.tidy(() => fn(N <= n ? x : x.slice([i, ...rest.map(() => 0)], [m, ...rest]))); parts.push(part); await gpuWait(part); }
      if (parts.length === 1) return parts[0];
      const out = tf.concat(parts); parts.forEach(t => t.dispose()); return out;
    }
    // Where a block widens the channels, its shortcut is projected (and pooled) in bands of an even
    // number of rows, rather than projecting all positions at once.
    async function skipPath(xn, k, b) {
      const [, H, Wd, C] = xn.shape, rows = Math.max(2, Math.floor(BUDGET / (Wd * b.dimOut)) & ~1), parts = [];
      for (let r = 0; r < H; r += rows) {
        const part = tf.tidy(() => { const s = linear(xn.slice([0, r, 0, 0], [1, Math.min(rows, H - r), Wd, C]), k + ".skip"); return b.qStride ? tf.maxPool(s, 2, 2, "valid") : s; });
        parts.push(part); await gpuWait(part);
      }
      if (parts.length === 1) return parts[0];
      const out = tf.concat(parts, 1); parts.forEach(t => t.dispose()); return out;
    }
    async function block(x, i) {   // x stays the caller's to dispose
      const b = blocks[i], k = "blk" + i, [, H, Wd] = x.shape;
      // attention half
      const xn = layerNorm(x, k + ".n1", 1e-6);
      const shortcut = b.dim === b.dimOut ? x : await skipPath(xn, k, b);
      let a;
      if (b.window > 0) {
        const h = tf.tidy(() => windowPartition(xn, b.window)[0]); xn.dispose();
        const wa = await sliced(h, Math.max(1, Math.floor(BUDGET / (3 * b.dimOut * b.window * b.window))), part => attention(part, k, b)); h.dispose();
        const win = b.qStride ? b.window / 2 : b.window, [, H2, W2] = shortcut.shape;
        a = tf.tidy(() => windowUnpartition(wa, win, [Math.ceil(H2 / win) * win, Math.ceil(W2 / win) * win], [H2, W2])); wa.dispose();
      } else { a = tf.tidy(() => attention(xn, k, b)); xn.dispose(); }
      const y = shortcut.add(a); a.dispose(); if (shortcut !== x) shortcut.dispose();
      await gpuWait(y);
      // feed-forward half, in slices of positions
      const [, H3, W3, C3] = y.shape, w = ["fc1.w", "fc1.b", "fc2.w", "fc2.b", "n2.g", "n2.b"].map(n => W(k + "." + n)), [fc1w, fc1b, fc2w, fc2b, g, bb] = w;
      const flat = y.reshape([-1, C3]);
      const ff = await sliced(flat, Math.max(256, Math.floor(BUDGET / (4 * C3))), r => { const { mean, variance } = tf.moments(r, -1, true), n = r.sub(mean).mul(tf.rsqrt(variance.add(1e-6))).mul(g).add(bb);
        return r.add(tf.matMul(gelu(tf.matMul(n, fc1w).add(fc1b)), fc2w).add(fc2b)); });
      w.forEach(t => t.dispose()); flat.dispose(); y.dispose();
      const out = ff.reshape([1, H3, W3, C3]); ff.dispose(); return out;
    }
    // Hiera's position embedding: the 7×7 background embedding resized bicubically (as torch's
    // F.interpolate, align_corners=False, A = −0.75) plus the 8×8 window embedding tiled.
    function posEmbed(h, w) {
      const C = cfg.embed_dim, bg = f32("pos"), win = f32("pos_win"), S = 7, ws = 8;
      const cubic = t => { const A = -0.75, w0 = ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A, w1 = ((A + 2) * t - (A + 3)) * t * t + 1,
        w2 = ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1; return [w0, w1, w2, 1 - w0 - w1 - w2]; };
      const taps = (n, N) => Array.from({ length: n }, (_, o) => { const s = (o + 0.5) * N / n - 0.5, i0 = Math.floor(s);
        return { idx: [-1, 0, 1, 2].map(d => Math.min(N - 1, Math.max(0, i0 + d))), wt: cubic(s - i0) }; });
      const tx = taps(w, S), ty = taps(h, S), out = new Float32Array(h * w * C), row = new Float32Array(S * w);
      for (let c = 0; c < C; c++) {
        for (let y = 0; y < S; y++) for (let x = 0; x < w; x++) { const { idx, wt } = tx[x]; let v = 0; for (let t = 0; t < 4; t++) v += wt[t] * bg[c * S * S + y * S + idx[t]]; row[y * w + x] = v; }
        for (let y = 0; y < h; y++) { const { idx, wt } = ty[y];
          for (let x = 0; x < w; x++) { let v = 0; for (let t = 0; t < 4; t++) v += wt[t] * row[idx[t] * w + x];
            out[(y * w + x) * C + c] = v + win[c * ws * ws + (y % ws) * ws + (x % ws)]; } }
      }
      return out;
    }
    let pos = null;

    // image (anything drawImage accepts) → normalized 1024×1024 NHWC; the photo is stretched, as SAM 2 does
    function prepare(img) {
      const c = document.createElement("canvas"); c.width = c.height = SIZE; const g = c.getContext("2d"); g.imageSmoothingQuality = "high";
      g.drawImage(img, 0, 0, SIZE, SIZE); const px = g.getImageData(0, 0, SIZE, SIZE).data, x = new Float32Array(SIZE * SIZE * 3);
      for (let i = 0, j = 0; i < px.length; i += 4) for (let k = 0; k < 3; k++) x[j++] = (px[i + k] / 255 - man.mean[k]) / man.std[k];
      c.width = c.height = 0; return x;
    }

    async function encode(img, opts = {}) {
      const input = img instanceof Float32Array ? img : prepare(img);
      if (!pos) pos = posEmbed(SIZE / 4, SIZE / 4);
      let x = tf.tidy(() => tf.conv2d(tf.tensor4d(input, [1, SIZE, SIZE, 3]), W("patch.w"), 4, [[0, 0], [3, 3], [3, 3], [0, 0]]).add(W("patch.b"))
        .add(tf.tensor4d(pos, [1, SIZE / 4, SIZE / 4, cfg.embed_dim])));
      // The two high-resolution outputs (s0, s1) are made as soon as their stage ends, so the
      // large stage results need not be kept; the two small ones are kept for the neck.
      const feats = [], hi = [];
      for (let i = 0; i < blocks.length; i++) {
        const y = await block(x, i);
        if (!feats.includes(x)) x.dispose();
        x = y; const s = stageEnds.indexOf(i);
        if (s === 0 || s === 1) hi[s] = tf.tidy(() => linear(x, s ? "neck.s1" : "neck.s0"));
        if (s >= 2 || (s >= 0 && opts.debug)) feats[s] = x;
        opts.progress?.((i + 1) / (blocks.length + 1));
        await gpuWait(x); await tf.nextFrame();
      }
      if (!feats.includes(x)) x.dispose();
      const embed = tf.tidy(() => linear(feats[2], "neck.l2").add(tf.image.resizeNearestNeighbor(linear(feats[3], "neck.l3"), [EMB, EMB])).add(W("no_mem")));
      const out = { s0: hi[0], s1: hi[1], embed };
      if (opts.debug) out.trunk = feats; else feats.forEach(t => t?.dispose());
      opts.progress?.(1);
      return out;
    }

    /* ── prompt encoder (on the CPU: a few hundred numbers) ── */
    const gauss = f32("pe.gauss"), ptEmb = f32("pe.points"), notPt = f32("pe.not_a_point");
    function pe(x, y, out, o) {   // x, y in 0–1
      const cx = 2 * x - 1, cy = 2 * y - 1;
      for (let j = 0; j < 128; j++) { const v = 2 * Math.PI * (cx * gauss[j] + cy * gauss[128 + j]); out[o + j] = Math.sin(v); out[o + 128 + j] = Math.cos(v); }
    }
    let densePE = null;
    function getDensePE() {
      if (densePE) return densePE; const a = new Float32Array(EMB * EMB * 256);
      for (let i = 0; i < EMB; i++) for (let j = 0; j < EMB; j++) pe((j + 0.5) / EMB, (i + 0.5) / EMB, a, (i * EMB + j) * 256);
      return (densePE = tf.keep(tf.tensor2d(a, [EMB * EMB, 256])));
    }
    // points: [[x, y, label]] with x, y in model pixels (0–1024); a padding point is appended, as SAM 2's predictor does
    function sparse(points) {
      const n = points.length + 1, a = new Float32Array(n * 256);
      points.forEach(([x, y, label], i) => { pe((x + 0.5) / SIZE, (y + 0.5) / SIZE, a, i * 256); for (let c = 0; c < 256; c++) a[i * 256 + c] += ptEmb[label * 256 + c]; });
      a.set(notPt, (n - 1) * 256); return tf.tensor2d(a, [n, 256]);
    }

    /* ── mask decoder ── */
    function attn(k, q, kk, v) {   // SAM's Attention: 8 heads, internal width from the weights
      const qp = linear(q, k + ".q", D), kp = linear(kk, k + ".k", D), vp = linear(v, k + ".v", D), inner = qp.shape[1], hd = inner / 8;
      const heads = t => t.reshape([-1, 8, hd]).transpose([1, 0, 2]);
      const o = sdpa(heads(qp), heads(kp), heads(vp)).transpose([1, 0, 2]).reshape([-1, inner]);
      return linear(o, k + ".o", D);
    }
    const mlp = (x, k, n) => { for (let j = 0; j < n; j++) { x = linear(x, `${k}.${j}`, D); if (j < n - 1) x = tf.relu(x); } return x; };

    async function decode(f, prompts) {
      const pts = prompts.map(([x, y, l]) => [x * SIZE, y * SIZE, l]);
      // Each layer runs in its own scope, so only the current tokens and image keys are kept.
      const qpe = tf.tidy(() => tf.concat([D("dec.tokens"), sparse(pts)])), kpe = getDensePE();
      let queries = qpe, keys = tf.tidy(() => f.embed.reshape([EMB * EMB, 256]).add(D("pe.no_mask")));
      const next = (q, k) => { if (queries !== qpe && queries !== q) queries.dispose(); if (keys !== k) keys.dispose(); queries = q; keys = k; };
      for (let i = 0; i < 2; i++) {
        const k = "tw" + i;
        next(...tf.tidy(() => {
          let q = i === 0 ? attn(k + ".self", queries, queries, queries) : (() => { const a = queries.add(qpe); return queries.add(attn(k + ".self", a, a, queries)); })();
          q = layerNorm(q, k + ".n1", 1e-5, D);
          q = layerNorm(q.add(attn(k + ".t2i", q.add(qpe), keys.add(kpe), keys)), k + ".n2", 1e-5, D);
          q = layerNorm(q.add(linear(tf.relu(linear(q, k + ".fc1", D)), k + ".fc2", D)), k + ".n3", 1e-5, D);
          const kk = layerNorm(keys.add(attn(k + ".i2t", keys.add(kpe), q.add(qpe), q)), k + ".n4", 1e-5, D);
          return [q, kk];
        }));
      }
      next(tf.tidy(() => layerNorm(queries.add(attn("tw.final", queries.add(qpe), keys.add(kpe), keys)), "tw.nf", 1e-5, D)), keys);
      // upscale ×4 with the high-resolution features, then one mask per mask token
      const up1 = tf.tidy(() => gelu(layerNorm(tf.depthToSpace(linear(keys.reshape([1, EMB, EMB, 256]), "up1", D), 2).add(f.s1), "up.ln", 1e-6, D)));
      const up2 = tf.tidy(() => gelu(tf.depthToSpace(linear(up1, "up2", D), 2).add(f.s0))); up1.dispose();
      const res = tf.tidy(() => {
        const tok = i => queries.slice([i, 0], [1, 256]);
        const hyper = tf.concat([0, 1, 2, 3].map(i => mlp(tok(2 + i), "hyper" + i, 3)));      // 4×32
        return [tf.matMul(hyper, up2.reshape([-1, 32]), false, true), tf.sigmoid(mlp(tok(1), "iou", 3))];   // 4×65536 masks, 4 scores
      });
      up2.dispose(); queries.dispose(); keys.dispose(); qpe.dispose();
      const [m, s] = await Promise.all(res.map(t => t.data())); res.forEach(t => t.dispose());
      return { all: m, scores: Array.from(s), size: 4 * EMB, ...choose(m, s, prompts) };
    }
    // SAM 2's advice: one tap is ambiguous, so take the best of the three alternative masks;
    // with more prompts use the single-mask output unless it is unstable.
    function choose(m, s, prompts) {
      const N = 4 * EMB * 4 * EMB, pick = i => ({ index: i, logits: m.subarray(i * N, (i + 1) * N), score: s[i] });
      const best = [1, 2, 3].reduce((a, b) => (s[b] > s[a] ? b : a));
      if (prompts.length === 1 && prompts[0][2] <= 1) return pick(best);
      let inner = 0, outer = 0; for (let i = 0; i < N; i++) { const v = m[i]; if (v > 0.05) inner++; if (v > -0.05) outer++; }
      return pick(outer === 0 || inner / outer >= 0.98 ? 0 : best);
    }
    const release = f => { for (const k in f) (Array.isArray(f[k]) ? f[k] : [f[k]]).forEach(t => t.dispose?.()); };
    const dispose = () => { freeCache(); densePE?.dispose(); densePE = null; pos = null; };
    return { encode, decode, release, dispose, prepare, blocks };
  }

  window.LumiereSAM2 = { load };
})();
