# Lumière

*by Les Fleurs de la Nuit*

A mobile-first photo editor for adding and shaping light in photographs. It is built around one
method: stack light layers over a photograph, where each layer has its own light, blurs, mask,
blend mode and opacity. The photograph itself is never blurred or changed.

Everything runs in the browser on the device: rendering on the GPU (WebGL2), subject and object
selection with a small on-device AI model. Nothing is uploaded.

## What it does

**Light layers.** Each light layer has:

| Setting | Options |
|---|---|
| Light | Radial gradient (an ellipse you draw, stretch and rotate; color temperature swatches) or a light image (built-in Window, Bokeh, Light leak, Sun flare, or your own) |
| Blurs | Any number of radial (zoom or spin) and Gaussian blurs, applied in order to this light only. The radial blur center can sit outside the photo. |
| Mask | Lightroom-style parts: Subject, Object (tap the object, or drag a box), Brush, Radial gradient, Linear gradient. The first part is the base; each later part adds, subtracts or intersects. Parts can be inverted. The whole mask has Density, Feather and Invert. |
| Blend mode | Screen, Overlay, Lighten, Color Dodge |
| Opacity | 0–100% |

**Presets.** One tap adds a set of ready-made lights placed around the subject: Halo, Window
beams, Spotlight, Golden side light, Moon rim, Night bokeh. Every light a preset adds is an
ordinary, fully editable layer.

**Working with it.** The app opens on an empty canvas with a drop zone: drop a photo or choose
one from your library (or try the sample). Lights are listed top to bottom; reorder them by
dragging the handle on the left, or by long-pressing a row. Tap a light to open its editor. The
sheet slides between the list and the editor, and between lights, so you keep your place.

**The editor** has a two-row header: back, the light's name, "show this light alone", and a ⋯
menu (Rename, Duplicate, Hide, Light above or below, Delete); then Opacity next to a blend-mode
menu. Below are the Light, Blurs and Mask tabs, each with its main controls first. Blurs and mask
parts open as accordions and are added from one "Add" menu each. Double-tap any slider to reset
it; tap or swipe the sheet's grabber for a taller panel.

**Look and feel.** The interface is built from the "Darkroom" design system in
`design-system/` (tokens, components and icons); the app page adds only layout.

**Also:** undo/redo for every change, pinch to zoom, export to JPEG or PNG at full resolution
(up to about 16.7 megapixels, a browser limit for now), and editable project files (`.lumiere.json`).

## Run it

It is a static site with no build step. Serve the folder over HTTP (the selection model is
loaded with `fetch`, which does not work from `file://`):

```sh
python3 -m http.server 8080      # or: npm start
```

Then open <http://localhost:8080>. To try it on a phone, open the same address from the phone
using your computer's local IP address, or host the folder on any static host.

The app loads TensorFlow.js from jsDelivr, its fonts from Google Fonts, and its styles from
`design-system/styles.css`. The selection models are downloaded only when first used:
`models/u2netp.json` (about 3 MB) for Subject and presets, and `models/sam2-tiny/` (62 MB) for
tap-to-select on Object parts. Both run on the device; photos are never uploaded.

## Test

```sh
npm install
npx playwright install chromium
npm test                    # start screen, opening a photo, reordering (mouse and touch), editor and its menus, 44 px targets, drawing, undo
npm run test:selection      # also runs Select Subject and tap-to-select (very slow without a GPU)
```

Without internet access, set `TFJS_PATH` to a local `tf.min.js` (TensorFlow.js 4.22.0) for the
selection test. To check the SAM 2 port against PyTorch after changing `sam2.js`, write reference
outputs with `tools/convert_sam2.py --reference photo.jpg --ref-out REF` and run
`node tools/check_sam2.mjs REF`.

## Repository layout

| Path | What it is |
|---|---|
| `index.html` | The whole app: UI, WebGL2 renderer, mask compositing, selection, presets |
| `models/u2netp.json` | U²-Net-p salient-object model, converted for TensorFlow.js (float16 weights) |
| `tools/convert_u2netp.py` | Rebuilds `models/u2netp.json` from the ONNX model |
| `sam2.js` | SAM 2.1 Tiny (tap-to-select) in TensorFlow.js: image encoder, prompt encoder, mask decoder |
| `models/sam2-tiny/` | Its weights (float16, six 10 MB files and a manifest) |
| `tools/convert_sam2.py` | Rebuilds `models/sam2-tiny/` from Meta's checkpoint; can write PyTorch reference outputs |
| `tools/check_sam2.mjs` | Compares `sam2.js` with those reference outputs in a headless browser |
| `tests/smoke.mjs` | Headless browser smoke test |
| `design-system/` | The "Darkroom" design system: tokens (`styles.css`), living style guide with Dev Mode and Copy-as-prompt (`index.html`), concept screens, icon sprite |
| `docs/PLAN.md` | Product and design plan: the layer model, interface and roadmap |
| `docs/ai-models.md` | Plan for on-device AI: candidate models (SAM 2, Depth Anything, BiRefNet), licenses, evaluation |
| `docs/photocraft-evaluation.md` | Why the app does not build on PhotoCraft (measurements) |

## How rendering works

For each light layer, bottom to top:

1. Draw the light alone into its own GPU texture.
2. Run that layer's blurs in order, on that texture only.
3. Build the layer's mask (on the CPU at up to 1024 px, then feathered on the GPU).
4. Blend the result onto everything below with the layer's blend mode, opacity and mask.

While you drag, radial blurs use fewer samples; when you let go, the preview is redrawn at full
quality. Exports re-run the same steps at the photo's full resolution with more samples.

## Licenses

Third-party components and their licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This repository does not yet have a license of its own.
