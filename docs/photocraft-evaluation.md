# PhotoCraft evaluation

Before building Lumière, the original plan was to adapt PhotoCraft, an open-source Rust image editor, as the
engine. This is what checking its source showed, and why Lumière uses its own small GPU renderer instead.

_Note: written when the project was called "Light Studio"._

Source: `storytold/photocraft` @ 8c30a41 (2026-10-17), workspace version 0.10.0, MIT OR Apache-2.0.

## Plan claims vs. the code

| Plan needs | In PhotoCraft? | Where |
|---|---|---|
| Crates algo, color, compose, doc, engine, gpu + photocraft-web | Yes (plus 20 more crates) | `crates/*`, `apps/photocraft-web` |
| Engine separate from desktop UI | Yes — `photocraft-engine` does not depend on egui | `crates/engine/Cargo.toml` |
| Small command API | Already exists: `Session::execute(id, json) -> json`, with `precheck` and a `"coalesce"` key that merges slider drags into one undo step | `crates/engine/src/lib.rs:649` |
| Screen, Overlay, Lighten, Color Dodge | Yes, CPU and GPU (WGSL) | `crates/color/src/blend.rs`, `crates/gpu/src/compose.wgsl` |
| Radial / linear gradient light layers | Yes — `Fill::Gradient` with styles, opacity stops, offset | `crates/doc/src/lib.rs:324` |
| Gaussian + radial blur | Yes; radial has draft/good/best quality (maps to the plan's preview/final) | `crates/algo/src/blur/radial.rs` |
| Non-destructive blur | Smart filters on smart objects | `crates/engine/src/smart_cmds.rs` |
| **Radial center outside the image** | **Algorithm: yes. Command layer: no** — `centerX`/`centerY` are validated to 0..1 | `crates/engine/src/filters/validation.rs:44` |
| Select Subject / Select Object | Yes: `select.subject`, `select.object` (rect or point) | `crates/engine/src/smartselect_cmds.rs` |
| Mask refine, feather, invert | Yes: `select.selectAndMask` (guided filter, decontaminate), `select.modify.feather`, `select.inverse`; layer masks keep non-destructive `feather` + `density` | `crates/algo/src/matting.rs`, `crates/doc/src/lib.rs:245` |
| Undo/redo | Yes, in the engine | engine history |
| Web renderer, GPU + fallback | WebGPU with WebGL2 fallback via eframe/wgpu | `apps/photocraft-web/Cargo.toml` |

## Risks the plan should name explicitly

1. **Select Subject is classical, not AI.** Saliency + GrabCut; its own doc comment says it "works well for a
   distinct object on a comparatively plain background; it is not a semantic segmenter." Petals on busy
   backgrounds are exactly the hard case. Budget the AI-segmentation fallback in Phase 4 rather than
   treating it as optional.
2. **Filters run on the CPU, single-threaded in wasm.** The GPU crate composites layers and layer effects;
   radial/Gaussian blur are CPU filters, and `rayon` is disabled on wasm32. Dragging the radial center at
   30–60 FPS likely needs a GPU radial-blur shader in `engine/lighting/` (or a small, downscaled preview).
   See timings below.
3. **The web app is the whole desktop UI.** `photocraft-web` runs the egui `PhotocraftApp` in a canvas.
   Light Studio's React UI can't reuse it; it needs its own wasm-bindgen bridge over `Session::execute`
   plus a way to get rendered pixels/textures into the React-owned canvas. That bridge is the real
   Phase 1 work.
4. **Moving the radial center outside the image** needs a one-line validation change (or a lighting-side
   wrapper). The algorithm already guards against out-of-domain centers.
5. **Upstream moves fast** (PR numbers past 7,000; 0.5 → 0.10 in a few months). Pin a commit and plan
   periodic rebases; keep Light Studio code out of upstream crates as the plan says.

## Measurements

**WebAssembly build:** `cargo build -p photocraft-engine --target wasm32-unknown-unknown --release`
succeeds with Rust 1.97 (3 m 47 s, 3 warnings). Phase 1 step 1 is not a blocker.

**Single-threaded CPU timings** (native x86 server core, `rayon` limited to 1 thread to mimic wasm;
wasm in a phone browser will be slower, not faster). Synthetic noise image, zoom method, amount 40.

| Size | Radial draft | Radial good | Radial best | Radial, center outside image | Gaussian r20 |
|---|---|---|---|---|---|
| 1080×720 | 1,214 ms | 1,651 ms | 1,635 ms | 1,570 ms | 29 ms |
| 1620×1080 | 3,179 ms | 5,725 ms | 5,845 ms | 3,563 ms | 68 ms |

Conclusions:
- Radial blur on the CPU is ~40–100× too slow for dragging (target is 16–33 ms per frame). Even
  "draft" quality at a tiny preview misses the target. **A GPU radial-blur shader is required for
  interactive preview**; the CPU filter is fine for final export and as the reference for visual tests.
- Gaussian blur is fast enough for slider feedback at preview size on the CPU.
- Passing a center outside the image straight to the algorithm works (no crash, similar cost).

## Suggested changes to the plan

- Phase 1: replace "Build PhotoCraft's existing WASM application" with "Build a wasm-bindgen bridge over
  `photocraft-engine::Session::execute`" — the existing web app is the egui desktop UI and won't be reused.
- Phase 1: add "GPU radial-blur preview shader (WGSL, WebGPU + WebGL2), checked against the CPU filter".
  This is the item that decides whether the drag interaction is viable; prototype it first.
- Phase 3: allow radial centers outside 0..1 (validation change or lighting-side wrapper).
- Phase 4: treat AI segmentation as planned work, not a fallback; test the classical Select Subject on
  your own flower photographs early (it can be done natively with the CLI before any UI exists).
- Section 2: the proposed `editor.addLightLayer` / `setRadialBlur` API maps cleanly onto existing
  command ids plus the `"coalesce"` key for one undo step per drag.
