# Lumière: AI models plan

How on-device AI (mostly from the Hugging Face ecosystem) could extend Lumière without
changing what makes it work: **the AI decides what and where; Lumière's renderer decides how.**
Models produce masks, depth maps or suggested layer settings. The light itself is always an
ordinary, editable layer drawn by the WebGL2 renderer, never a regenerated photograph.

Status: exploration. Nothing here is built yet. Facts marked *(verify)* need checking against
the current model cards and library docs before any work starts.

## Where Lumière is today

- **Select Subject and Select Object** run U²-Net-p (Apache-2.0, about 3 MB) on the phone with
  TensorFlow.js on WebGL. Object selection runs the same model inside a box the user drags.
- A guided filter snaps the model's soft output to edges in the photo.
- The result becomes a **mask part**: the first part is the base, later parts Add, Subtract or
  Intersect. Any new model only has to produce one more kind of part.
- Projects store the mask results, not the models, so files stay portable.
- Presets already place lights from the subject's position and size (`subjectStats`).

## Principles

1. **Local first.** Selection and depth run on the device: private, no per-edit cost, works
   offline once downloaded. Cloud is only for optional, heavier features.
2. **Commercially usable licenses only** (Apache-2.0, MIT, BSD). Lumière may become a
   subscription product, so non-commercial weights (CC BY-NC) are out unless licensed.
3. **Self-host the weights.** Ship model files from Lumière's own origin, as with
   `models/u2netp.json`. Hosted previews may block downloads from other sites, and it pins the
   exact version.
4. **Download on first use** and cache it; show the size before a large download.
5. **Measure before replacing.** Every candidate is compared with U²-Net-p on the same photos.

## Candidates

| Model | Use in Lumière | License | Notes |
|---|---|---|---|
| **SAM 2 Hiera Tiny** | Tap-to-select for Object: tap a flower (or several points, or a box) | Apache-2.0 | Two parts: a heavy image encoder that runs once per photo, and a light decoder that runs per tap. Check whether Transformers.js supports SAM 2 directly or only SAM / SlimSAM *(verify)*. |
| SlimSAM, MobileSAM, EfficientSAM | Same, smaller | Apache-2.0 / MIT *(verify each)* | Fallbacks if SAM 2 is too heavy for phones. |
| **Depth Anything V2 Small** | A new **Depth** mask part; depth-aware presets | Apache-2.0 | Only *Small* is Apache-2.0. **Base and Large are CC BY-NC 4.0**: do not use. Transformers.js has a depth-estimation pipeline for it *(verify the ONNX build)*. |
| **BiRefNet** (or a lite variant) | Finer edges: petals, translucent edges, foliage | MIT | Larger than U²-Net-p; likely needs WebGPU to be fast. |
| BRIA RMBG 2.0 | Fine matting | CC BY-NC 4.0 | Excluded unless commercially licensed. RMBG 1.4 is also non-commercial. |
| U²-Net-p (current) | Subject, Object (box) | Apache-2.0 | The baseline. |

## How each fits the existing app

### 1. Tap-to-select (first)

- A new way to make an **Object** part: tap the thing instead of dragging a box. Extra taps
  add or remove areas; a box still works.
- Run the encoder when the user first chooses Object on a photo and keep its output for that
  photo, so each further tap is near instant.
- Output: a coverage map at mask resolution, refined with the existing guided filter, stored as
  the part's data like today.

### 2. Depth

- A new mask part, **Depth range**, as in Lightroom: near and far limits with a soft edge. It
  combines with Subject or Object through the usual Add / Subtract / Intersect.
- Lets a light sit *behind* the flower while the foreground stays dark, without hand-painting.
- Later: depth-aware presets (a back light that only reaches what is behind the subject), and
  depth-weighted blur falloff.

### 3. Finer edges

- Offer a "Refine edge" option on Subject and Object parts that runs the larger model only
  where the mask is uncertain (its soft border), to keep cost down.

### 4. Light placement suggestions

- Start with rules, no new model: the subject mask (where and how big), depth (what is behind),
  and the photo's brightness give ranked suggestions such as "back light, warm, upper left,
  blur center on the flower". Each suggestion is applied as normal, editable layers, like a
  preset.
- Later: saved projects are pairs of *photo + layer settings*. With permission, your own
  projects could tune the rules or train a small model that predicts settings.

### 5. Generated light images (last, optional)

- Text-to-image could create new light images (window shapes, foliage shadows, bokeh). They
  must come out on black (or with transparency) to blend with Screen, and would be inserted as
  a light image layer.
- Needs cloud inference: per-image cost and an upload. A natural premium feature.
- Much of the value is available sooner by making the procedural light images (window, bokeh,
  leak, flare, already drawn in code) take more parameters.

## Where models run

| Option | When |
|---|---|
| WebGPU in the browser | Default where available. Recent Safari ships WebGPU *(verify the iOS version)*; Chrome on Android has it. |
| WebGL (TensorFlow.js) or WASM | Fallback for older devices; slower. |
| Cloud | Optional heavy features (generation, very large models), probably for subscribers. |

**One runtime, eventually.** Today the app uses TensorFlow.js. Transformers.js runs models
through ONNX Runtime Web. Running both doubles download size and GPU memory. If the new models
are adopted, plan to move U²-Net-p to ONNX as well (or keep it as the light fallback) rather
than keep two engines for good.

**Memory budget.** Phones close tabs that use too much memory. Load one model at a time, free
it when the user leaves the mask tab, and keep the photo's encoder output only while it is
being used.

## Evaluation plan (step one)

1. **Test set:** 20–30 of your own photographs that represent the work: single flowers, groups,
   translucent and fine petals, busy backgrounds, low light. For each, a hand-made reference
   mask of the subject, and for some, of one chosen object.
2. **Scores per model:**
   - mask accuracy (intersection over union) and edge accuracy (boundary F-score)
   - time on two real phones (a recent iPhone and a mid-range Android): first run including
     download, and later runs
   - peak memory, and whether the tab survives
   - download size
3. **Interaction (for tap-to-select):** number of taps to reach a good mask compared with
   dragging a box.
4. **Decision:** adopt a model only if it beats U²-Net-p on accuracy where it matters (petal
   edges) and stays usable on the mid-range phone.

A small test page (or a `--models` flag on the smoke test) can run every candidate over the set
and print the table.

## Staging

| Stage | Work | Depends on |
|---|---|---|
| 1 | Evaluation set and harness; SAM-family tap-to-select prototype | Your photographs and reference masks |
| 2 | Depth range mask part; depth-aware presets | Stage 1 harness |
| 3 | Edge refinement (BiRefNet) where the evaluation shows a need | Stage 1 results |
| 4 | Rule-based light suggestions | Stages 1–2 |
| 5 | Optional cloud features: generated light images, learned suggestions | Product and pricing decisions |

## A public demo on Hugging Face Spaces

Lumière is a static site with no build step, so it can be published as a static Space (free)
almost as-is: the page, `design-system/`, and `models/`. Useful as a portfolio piece next to
the real site, not a replacement for it. If Lumière's code is kept private, publish a reduced
demo instead.

## Open questions

- Which phones matter most to your audience (sets the memory and speed budget)?
- Which photographs go in the evaluation set, and who draws the reference masks?
- Is cloud inference acceptable for any feature, and at what price?
- Should project files (with permission) ever be used to improve suggestions?
