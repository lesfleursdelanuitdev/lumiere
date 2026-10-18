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
| Mask | Lightroom-style parts: Subject, Object, Brush, Radial gradient, Linear gradient. The first part is the base; each later part adds, subtracts or intersects. Parts can be inverted. The whole mask has Density, Feather and Invert. |
| Blend mode | Screen, Overlay, Lighten, Color Dodge |
| Opacity | 0–100% |

**Presets.** One tap adds a set of ready-made lights placed around the subject: Halo, Window
beams, Spotlight, Golden side light, Moon rim, Night bokeh. Every light a preset adds is an
ordinary, fully editable layer.

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

The app loads TensorFlow.js from jsDelivr and its fonts from Google Fonts. The selection model
(`models/u2netp.json`, about 3 MB) is downloaded the first time someone uses Select Subject,
Select Object or a preset.

## Test

```sh
npm install
npx playwright install chromium
npm test                    # loads the app, opens the editor, draws a light, checks undo
npm run test:selection      # also runs AI subject selection (slow without a GPU)
```

## Repository layout

| Path | What it is |
|---|---|
| `index.html` | The whole app: UI, WebGL2 renderer, mask compositing, selection, presets |
| `models/u2netp.json` | U²-Net-p salient-object model, converted for TensorFlow.js (float16 weights) |
| `tools/convert_u2netp.py` | Rebuilds `models/u2netp.json` from the ONNX model |
| `tests/smoke.mjs` | Headless browser smoke test |
| `docs/PLAN.md` | Product and design plan: the layer model, interface and roadmap |
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
