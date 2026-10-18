// Checks the TensorFlow.js port of SAM 2 (sam2.js) against PyTorch reference outputs.
//
//   python tools/convert_sam2.py CHECKPOINT --sam2-src SRC --reference photo.jpg --ref-out REF
//   node tools/check_sam2.mjs REF [path/to/tf.min.js]
//
// Runs in headless Chromium on the WebGL backend (the one phones use). Without a GPU this
// takes a few minutes. Reports the largest differences of the image features and how well
// the masks agree.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = fileURLToPath(new URL("..", import.meta.url)), ref = process.argv[2], tfPath = process.argv[3];
if (!ref) { console.error("usage: node tools/check_sam2.mjs REFERENCE_DIR [tf.min.js]"); process.exit(2); }
const types = { ".html": "text/html", ".json": "application/json", ".js": "text/javascript" };
const server = createServer(async (req, res) => {
  const u = new URL(req.url, "http://x").pathname;
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }).end("<!doctype html><title>check</title>"); return; }
  const path = u.startsWith("/ref/") ? join(ref, u.slice(5)) : normalize(join(root, decodeURIComponent(u)));
  try { res.writeHead(200, { "content-type": types[extname(path)] || "application/octet-stream" }).end(await readFile(path)); } catch { res.writeHead(404).end(); }
}).listen(0, "127.0.0.1");
await new Promise(r => server.once("listening", r));
const url = `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
try {
  const page = await browser.newPage();
  page.on("console", m => console.log("  [page]", m.text())); page.on("pageerror", e => console.log("  [error]", e.message));
  await page.goto(url);
  if (tfPath) await page.addScriptTag({ content: await readFile(tfPath, "utf8") }); else await page.addScriptTag({ url: "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js" });
  await page.addScriptTag({ url: url + "sam2.js" });
  const report = await page.evaluate(async () => {
    tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 0); await tf.setBackend("webgl"); await tf.ready();
    const get = async n => new Float32Array(await (await fetch("/ref/" + n)).arrayBuffer());
    const diff = (a, b) => { let m = 0, s = 0, r = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; s += d; r = Math.max(r, Math.abs(b[i])); } return { max: +m.toFixed(5), mean: +(s / a.length).toFixed(6), range: +r.toFixed(3) }; };
    const out = { backend: tf.getBackend() }; let t0 = performance.now();
    const sam = await LumiereSAM2.load("/models/sam2-tiny/"); out.loadSeconds = +((performance.now() - t0) / 1000).toFixed(1);
    t0 = performance.now(); const f = await sam.encode(await get("input.f32"), { debug: true }); await f.embed.data(); out.encodeSeconds = +((performance.now() - t0) / 1000).toFixed(1);
    for (let i = 0; i < 4; i++) out["trunk" + i] = diff(await f.trunk[i].data(), await get(`trunk${i}.f32`));
    for (const k of ["s0", "s1", "embed"]) out[k] = diff(await f[k].data(), await get(k + ".f32"));
    const { cases } = await (await fetch("/ref/cases.json")).json(); out.cases = [];
    for (let i = 0; i < cases.length; i++) {
      t0 = performance.now(); const r = await sam.decode(f, cases[i].points.map(([x, y, l]) => [x / 1024, y / 1024, l])); const ms = Math.round(performance.now() - t0);
      const want = await get(`case${i}.f32`), N = 256 * 256, iou = [];
      for (let m = 0; m < 4; m++) { let a = 0, u = 0; for (let j = 0; j < N; j++) { const p = r.all[m * N + j] > 0, q = want[m * N + j] > 0; if (p && q) a++; if (p || q) u++; } iou.push(+(a / Math.max(1, u)).toFixed(4)); }
      out.cases.push({ ms, maskAgreement: iou, scores: r.scores.map(x => +x.toFixed(3)), want: cases[i].iou.map(x => +x.toFixed(3)), chosen: r.index });
    }
    sam.release(f); return out;
  });
  console.log(JSON.stringify(report, null, 1));
} finally { await browser.close(); server.close(); }
