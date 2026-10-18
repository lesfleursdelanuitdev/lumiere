// Smoke test: serves the repository, opens the app in headless Chromium at phone size, and checks
// that the sample loads, the light list and editor render, a new gradient light can be drawn,
// and nothing throws. Pass --selection to also run AI subject selection (slow without a GPU).
//
//   npm install && npx playwright install chromium   (once)
//   npm test
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = fileURLToPath(new URL("..", import.meta.url));
const types = { ".html": "text/html", ".json": "application/json", ".js": "text/javascript" };
const server = createServer(async (req, res) => {
  let path = normalize(join(root, decodeURIComponent(new URL(req.url, "http://x").pathname)));
  if (!path.startsWith(root)) { res.writeHead(403).end(); return; }
  if (path.endsWith("/")) path = join(path, "index.html");
  try { const body = await readFile(path); res.writeHead(200, { "content-type": types[extname(path)] || "application/octet-stream" }).end(body); }
  catch { res.writeHead(404).end(); }
}).listen(0, "127.0.0.1");
await new Promise(r => server.once("listening", r));
const url = `http://127.0.0.1:${server.address().port}/`;

const failures = [];
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failures.push(what); };
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
try {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const errors = []; page.on("pageerror", e => errors.push(e.message));
  await page.goto(url); await page.waitForFunction(() => (typeof state !== "undefined" && state.layers.length === 2) || document.querySelector("#fatal"), null, { timeout: 60000 }).catch(() => {});
  check(await page.evaluate(() => typeof state !== "undefined" && state.layers.length === 2), "sample photo loads with two lights");
  check(await page.locator(".light").count() === 2, "light list shows both lights");
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "no horizontal scrolling at phone width");

  await page.click('[data-act="edit"] >> nth=0');
  check(await page.locator(".ed-head").count() === 1, "tapping a light opens its editor");
  for (const s of ["blurs", "mask", "light"]) { await page.click(`[data-seg="section"][data-val="${s}"]`); check(await page.locator(`[data-seg="section"][data-val="${s}"][aria-pressed="true"]`).count() === 1, `${s} section opens`); }

  await page.click('[data-act="to-list"]'); await page.click('[data-act="new-light"]'); await page.click('[data-new="radial"]');
  await page.mouse.move(120, 330); await page.mouse.down(); await page.mouse.move(190, 360, { steps: 6 }); await page.mouse.up();
  const drawn = await page.evaluate(() => { const l = state.layers.find(x => x.id === state.activeId); return l.light.type === "radial" && l.light.rx > 0.05 && Math.abs(l.light.rx - l.light.ry) < 1e-9; });
  check(drawn, "a new radial gradient can be drawn by dragging");
  await page.click("#undo"); check(await page.evaluate(() => state.layers.length === 3), "undo works");

  if (process.argv.includes("--selection")) {
    await page.click('[data-seg="section"][data-val="mask"]'); await page.click('[data-act="add-piece"][data-val="subject"]');
    await page.waitForFunction(() => /Done in|failed|could not/i.test(document.querySelector("#toast").textContent), null, { timeout: 600000 });
    check(await page.evaluate(() => { const l = state.layers.find(x => x.id === state.activeId); return !!l.mask.pieces.at(-1).data; }), "Select Subject fills a mask");
  }
  check(errors.length === 0, "no page errors" + (errors.length ? ": " + errors.join("; ") : ""));
} finally { await browser.close(); server.close(); }
if (failures.length) { console.error(`\n${failures.length} check(s) failed`); process.exit(1); }
console.log("\nall checks passed");
