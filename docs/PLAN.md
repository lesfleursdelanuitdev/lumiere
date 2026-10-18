# Lumière: product and design plan

Lumière by Les Fleurs de la Nuit is a mobile-first web app for adding light to photographs the
way a lighting photographer does it in Photoshop, without desktop menus or tiny controls.
Web first; native apps later.

## 1. The method the app is built around

A finished image is a photograph with *n* light layers stacked on top. Each light layer is a set
of settings, not a sequence of steps: they can be changed in any order, and any earlier layer can
be reopened and changed at any time.

| Setting | Definition |
|---|---|
| **Light** | A radial gradient (an ellipse: position, two radii, rotation, softness, color) or a light overlay image (position, scale, rotation, brightness). |
| **Blurs** | A list of blurs, applied one after another, each to the result of the previous one. Each is radial (zoom or spin, strength, center that may be outside the photo) or Gaussian (radius). Each can be reordered or switched off. Blurs affect only this layer. |
| **Mask** | A list of mask parts combined Lightroom-style (below), plus Density, Feather and Invert for the whole mask, as in Photoshop. |
| **Blend mode** | Screen (the usual choice), Overlay, Lighten, Color Dodge. |
| **Opacity** | Layer opacity, separate from mask density, as in Photoshop. |

### Mask parts

Five kinds: **Subject**, **Object**, **Brush**, **Radial gradient**, **Linear gradient**.

- The first part sets the area. Each later part is **Add** (union), **Subtract** or **Intersect**.
- Each part can be inverted or switched off.
- Typical use: *Subject, inverted* keeps a light off the flower so it sits behind it. A Brush
  part set to Subtract then carves out a detail by hand.
- The mask decides where the light shows: clear means the light shows; masked means it is held back.

## 2. Interface (phone first)

- **Start:** an empty canvas with a drop zone. Drop a photo (desktop) or choose one from the
  library (phone); "Or try a sample photo" loads a demo. The sheet slides up once a photo is open.
  Dropping a file onto the app later opens it as the new photo, keeping the lights.
- The photograph fills the screen. The top bar holds the name, Undo, Redo, Open and Export.
- One bottom sheet, three states:
  1. **Bar:** "Lights · n" and "Presets".
  2. **Lights list:** each row has a reorder handle, a thumbnail of that light alone, its name
     and a summary ("Screen · 70% · 2 blurs · mask"), and a show/hide eye. "+ New light" and
     "Presets" sit at the top. The photograph is shown at the bottom as never blurred or changed.
  3. **Layer editor:** reached by tapping a light. The header has back, the light's name,
     previous/next light (or swipe the header), and show/hide. Opacity and blend mode are always
     visible. Three tabs: **Light**, **Blurs**, **Mask**. At the bottom: Show this light alone,
     Duplicate, Delete.
- **Reordering:** drag the handle on the left of a row, or long-press anywhere on the row and
  drag. Other rows move aside as you drag; the list scrolls when you reach its edge.
- **Transitions:** the sheet's height eases between states; list ⇄ editor and light ⇄ light
  slide sideways, the Light/Blurs/Mask tabs slide their content and move the underline, and
  opening a blur or mask part expands it. All motion is turned off when the device asks for
  reduced motion.
- **Touch rule:** one finger edits what is selected (the gradient, a blur center, the brush, a
  mask shape, an object box). Two fingers always pan and zoom.
- **The sheet gets out of the way:** while a finger is on the photo, the sheet shrinks to one
  line showing the current value, so the light being shaped is never hidden.
- **Drawing a gradient:** press where the light should be centered and drag outward. Afterwards,
  drag the center to move it and the side handles to stretch and rotate it. Dragging anywhere
  else moves it without the finger covering it; "Fine drag" moves it at a quarter of the speed.

## 3. Presets

One tap adds an ordinary stack of light layers, positioned around the subject found by Select
Subject, so the preset fits the photograph. Every layer stays fully editable.

| Preset | What it adds |
|---|---|
| Halo | Warm 3200K glow behind the subject (Gaussian blur, masked off the subject) |
| Window beams | Window light image above the subject, zoom-blurred into beams from outside the frame |
| Spotlight | Neutral light on the subject itself (Overlay) |
| Golden side light | Low 2700K light from the left with a light leak |
| Moon rim | Cool 7500K edge light around the outline (Color Dodge) |
| Night bokeh | Bokeh image behind the subject |

Later: save your own stack of lights as a preset ("recipes").

## 4. Architecture today

A single static page (`index.html`), no build step.

- **Renderer:** WebGL2. Each light is drawn alone, blurred, masked, then blended. Radial blur
  uses fewer samples while dragging and more when you let go; export re-renders at full size.
- **Masks:** each part produces a coverage map (up to 1024 px on the long side); parts are
  combined on the CPU, uploaded once per change, feathered on the GPU.
- **Selection:** U²-Net-p (salient object detection, about 1.1 M parameters) runs with
  TensorFlow.js on the GPU. Subject runs on the whole photo; Object runs on a box the user drags
  (plus a 10% margin). The 320 px model output is refined to the photo's edges with a guided filter.
- **Projects:** JSON with the photo, light images and masks embedded.
- **Undo:** snapshots of the layer stack; mask pixel data is shared between snapshots and copied
  only when edited.

Why not PhotoCraft: its filters run on one CPU core in the browser; radial blur took 1.2 s for a
1080×720 preview, about 40–100 times too slow for dragging. See `photocraft-evaluation.md`.

## 5. Roadmap

1. **Test on real phones.** A recent iPhone, an older iPhone and two Android phones: frame rate
   while dragging (the app shows it), selection time, memory with large photos.
2. **Selection quality on your own photographs.** Fine petals, translucent petals, busy
   backgrounds. If U²-Net-p is not good enough, try a larger model (for example BiRefNet, MIT
   license) with a WebGPU path, and tap-to-select (point prompts) for Object.
3. **Installable app and offline use:** web app manifest, service worker, IndexedDB autosave so a
   project survives a reload or a closed tab.
4. **Large photos:** tiled export above about 16.7 megapixels; 16-bit render targets where the
   GPU supports them, to avoid banding in smooth gradients.
5. **Code structure:** split `index.html` into modules (renderer, document model, masks,
   selection, UI) with unit tests for the mask maths and the project format, before it grows further.
6. **Recipes:** save and share your own presets.
7. **Native apps:** keep the project format as the shared contract; reimplement the renderer's
   few shaders in Metal (iOS) and Vulkan or GLES (Android), and compare against the web renderer
   with visual regression tests.

## 6. Open questions

- License for this repository.
- Where the app will be hosted, and how it links into the other Les Fleurs de la Nuit tools.
- Whether light images need more controls (stretch, flip, color tint).
