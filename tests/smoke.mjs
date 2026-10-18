// Smoke test: serves the repository, opens the app in headless Chromium at phone size, and checks
// the start screen, opening a photo, the light list (including reordering by mouse and by touch),
// the editor and its transitions, drawing a light, and undo. Pass --selection to also run AI
// subject selection (slow without a GPU).
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
const order = page => page.evaluate(() => [...state.layers].reverse().map(l => l.name).join(" / "));
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
try {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true });
  const errors = []; page.on("pageerror", e => errors.push(e.message));
  await page.goto(url); await page.waitForTimeout(500);

  // Start screen: empty canvas with a drop zone, nothing else to do yet.
  check(await page.isVisible("#drop .dz"), "app opens on an empty canvas with a drop zone");
  check(await page.isHidden("#sheet"), "no editing panel before a photo is open");
  check(await page.isDisabled("#exportBtn"), "Export is disabled before a photo is open");

  // Open a photo through the drop zone's file input (a generated PNG).
  const png = Buffer.from(await page.evaluate(() => { const c = document.createElement("canvas"); c.width = 640; c.height = 480; const g = c.getContext("2d");
    g.fillStyle = "#101018"; g.fillRect(0, 0, 640, 480); g.fillStyle = "#c04070"; g.beginPath(); g.arc(320, 230, 120, 0, 7); g.fill(); return c.toDataURL("image/png").split(",")[1]; }), "base64");
  await page.setInputFiles("#dropFile", { name: "test-photo.png", mimeType: "image/png", buffer: png });
  await page.waitForFunction(() => assets.has("base") && !document.querySelector("#sheet").hidden, null, { timeout: 30000 });
  check(await page.isHidden("#drop"), "opening a photo hides the drop zone");
  check(await page.evaluate(() => state.doc.w === 640 && state.doc.h === 480), "the opened photo is the document");
  check(await page.isEnabled("#exportBtn"), "Export is enabled once a photo is open");

  // The sample, for a document with lights.
  await page.goto(url); await page.click("#trySample");
  await page.waitForFunction(() => typeof state !== "undefined" && state.layers.length === 2, null, { timeout: 60000 }).catch(() => {});
  check(await page.locator(".light").count() === 2, "sample photo loads with two lights in the list");
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "no horizontal scrolling at phone width");

  // Reorder with a mouse: drag the top light's handle below the second.
  const before = await order(page);
  const grip = await page.locator(".light .grip").first().boundingBox(), rowH = (await page.locator(".light").first().boundingBox()).height;
  await page.mouse.move(grip.x + grip.width / 2, grip.y + grip.height / 2); await page.mouse.down();
  await page.mouse.move(grip.x + grip.width / 2, grip.y + grip.height / 2 + rowH * 1.2, { steps: 8 }); await page.mouse.up(); await page.waitForTimeout(400);
  const after = await order(page);
  check(after === before.split(" / ").reverse().join(" / "), `dragging the handle reorders the lights (${before} → ${after})`);
  await page.click("#undo"); await page.waitForTimeout(100);
  check(await order(page) === before, "undo restores the previous order");

  // Reorder by touch: press the handle and drag with a finger.
  const cdp = await page.context().newCDPSession(page);
  const touch = (type, x, y) => cdp.send("Input.dispatchTouchEvent", { type, touchPoints: type === "touchEnd" ? [] : [{ x, y }] });
  const gx = grip.x + grip.width / 2, gy = grip.y + grip.height / 2;
  await touch("touchStart", gx, gy); for (let i = 1; i <= 8; i++) await touch("touchMove", gx, gy + rowH * 1.2 * i / 8); await touch("touchEnd"); await page.waitForTimeout(400);
  check(await order(page) === after, "dragging the handle with a finger reorders the lights");
  await page.click("#undo"); await page.waitForTimeout(100);
  const nm = await page.locator(".light .name").first().boundingBox(), nx = nm.x + nm.width / 2, ny = nm.y + nm.height / 2;
  await touch("touchStart", nx, ny); await page.waitForTimeout(600); for (let i = 1; i <= 8; i++) await touch("touchMove", nx, ny + rowH * 1.2 * i / 8); await touch("touchEnd"); await page.waitForTimeout(500);
  check(await order(page) === after && await page.locator(".ed-head").count() === 0, "long-pressing a row and dragging reorders without opening the editor");
  await page.click("#undo"); await page.waitForTimeout(100);

  // Editor, with animated transitions between list, editor and tabs.
  await page.click('[data-act="edit"] >> nth=0');
  check(await page.locator(".ed-head").count() === 1, "tapping a light opens its editor");
  check(await page.evaluate(() => document.querySelectorAll("#sheet .ghost").length + document.getAnimations().length > 0), "opening the editor animates");
  for (const s of ["blurs", "mask", "light"]) { await page.click(`[data-seg="section"][data-val="${s}"]`); check(await page.locator(`[data-seg="section"][data-val="${s}"][aria-pressed="true"]`).count() === 1, `${s} tab opens`); }
  await page.waitForTimeout(400);
  check(await page.locator("#sheet .ghost").count() === 0, "transition copies are cleaned up");

  // Draw a new radial light (a new gradient starts at size 0.3; drawing must change it).
  await page.click('[data-act="to-list"]'); await page.waitForTimeout(350); await page.click('[data-act="new-light"]'); await page.click('[data-new="radial"]'); await page.waitForTimeout(400);
  await page.mouse.move(120, 330); await page.mouse.down(); await page.mouse.move(190, 360, { steps: 6 }); await page.mouse.up();
  const drawn = await page.evaluate(() => { const l = state.layers.find(x => x.id === state.activeId); return l.light.type === "radial" && Math.abs(l.light.rx - 0.3) > 0.01 && Math.abs(l.light.rx - l.light.ry) < 1e-9; });
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
