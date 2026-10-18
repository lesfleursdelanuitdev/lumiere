# Bug: the whole page goes black on iPhone during Select Subject

Status: **fixes 1 and 2 done, waiting for a retest on the iPhone**. Diagnosed from measurements; not
yet reproduced on an iPhone.

## Report

- **Device:** iPhone 17 Pro, Safari, on <https://lumiere.lesfleursdelanuit.com>.
- **Photo:** her own, from the iPhone 17 Pro (24 or 48 MP) or a Nikon Z7 (45.7 MP, 8256 × 5504).
- **Steps:** she opened the photo and added a **Subject** mask part. The "loading the selection
  model" message appeared, then the entire page went black: no top bar, no panel, no message.
  Nothing came back.
- **Reproducible:** yes, twice in a row.

## What the symptom means

When the *whole* page disappears on iOS without a message, the browser engine (WebKit)
underneath the page has stopped. Either iOS killed the page's process, or it killed the
**graphics process**, which runs all WebGL work on iOS. iOS kills a process without warning when
it uses more memory than iOS allows it. That limit is far below the phone's total memory.

A single lost WebGL drawing surface would only black out the photo; the top bar and panel would
stay. That is not what she saw.

## Measurements

Measured in headless Chromium with software WebGL at iPhone 17 Pro screen size. Chromium is not
iOS, so the absolute numbers differ, but where the memory goes is the same.

**Process memory, opening a 45.7 MP photo and running Select Subject:**

| Moment | Page process | Graphics process |
|---|---|---|
| App open, no photo | 104 MB | 122 MB |
| 45.7 MP photo open | 205–265 MB | 204 MB |
| **During Select Subject** | 309 MB | **1,722 MB** |
| After Select Subject | 260 MB | 454 MB |

**Peak GPU memory held by the AI models** (TensorFlow.js `tf.profile`, independent of the photo):

| Model | Peak |
|---|---|
| Select Subject (U²-Net-p, 320 × 320 input) | **746 MB** |
| Tap-to-select, reading the photo (SAM 2.1 Tiny encoder) | **429 MB** |
| Tap-to-select, one tap (decoder) | not measured reliably yet (the profile reported the encoder's peak again) |

## Diagnosis

**Most likely cause: Select Subject briefly needs about 750 MB of GPU memory, and iOS kills the
graphics process.** It is reproducible because it happens every time the model runs.

Why so much: `runNet` in `index.html` (the U²-Net-p runner in `ensureNet`) runs about 490
steps inside one `tf.tidy`. It also stores every step's output in a `Map` until the last step,
so nothing is freed until the whole model has run. Most outputs are needed by only one or two
later steps.

**Contributing causes:**

1. **Full-resolution photos stay in memory.** `setDocument` keeps the decoded original
   (`assets.base.img`; a 45.7 MP photo is about 180 MB decoded) and a base64 text copy of the
   file (`dataURL`, about 30 MB for a Nikon JPEG) for saving projects. Selection redraws the
   full-size image several times (`predictRect`, `guideGray`), each needing it decoded.
2. **Tap-to-select is also heavy (about 430 MB).** It is not involved in this report, but it
   would likely hit the same limit. Its peak is in the global attention blocks (score slices
   of up to 16.7 M values) and the first stage (slices of 8,192 positions with GELU temporaries).
3. **No recovery when the WebGL context is lost.** Lumière never listens for
   `webglcontextlost`. It cannot help when iOS kills the whole graphics process, but it turns
   milder GPU resets into a permanently black photo.
4. **Two GPU users at once.** Lumière's renderer and TensorFlow.js each hold their own WebGL
   context and textures while a model runs.

**Confidence:** high that the subject model's memory is the main problem; medium that it alone
explains this report. To confirm, ask her to try:

1. "Try a sample photo", open a light, then Mask → Add mask part → **Subject**.
2. The **Halo** preset (it also runs Select Subject).

If either goes black, the photo is ruled out and fix 1 is the fix. If neither does, the large
photo matters more than measured, and fix 3 moves up.

## Fixes

In order of priority. Each has a check that proves it worked.

### 1. Free the subject model's results as soon as they are used (critical) · done

- **Where:** `ensureNet` → `runNet` in `index.html`.
- **Change:** before running, count how many later steps read each output. After each step,
  dispose of any input whose count has reached zero. Keep only the output tensor. Run the steps
  outside one big `tf.tidy`, or in small tidies.
- **Expected:** peak GPU memory from 746 MB to well under 100 MB, with identical results.
- **Check:** `tf.profile` peak below 100 MB; Select Subject masks identical to before on the
  sample photo; the friend's photo no longer crashes.
- **Result:** fixing the model alone was not enough. Two more things were holding memory:
  1. **The GPU runs behind JavaScript.** A freed result is only released once the GPU has
     finished the queued steps that use it. JavaScript queued all 490 steps almost at once, so
     nearly everything was still held. `runNet` now waits for the GPU every 4 steps. Waiting
     after every step saved a little more memory but would cost about a second per selection
     on a phone.
  2. **Hidden scratch memory in convolutions.** TensorFlow.js computes a convolution through an
     intermediate matrix ("im2col") that does not appear in its own memory count. For this
     model's first stage that matrix is about 240 MB. It is now turned off
     (`WEBGL_CONV_IM2COL = false`) in favour of direct convolution.

  | Measurement | Before | After |
  |---|---|---|
  | Subject model's tensors, peak (`tf.profile`) | 746 MB | 114 MB |
  | Graphics process, sample photo, peak above idle | ~1,050 MB | ~270 MB |
  | Graphics process, 45.7 MP photo, peak | 1,722 MB | 549 MB |
  | Subject mask vs before | | max difference 0.00006 (0–1 scale) |
  | Tap-to-select vs PyTorch (same setting) | 99.96–100% | 99.96–100% |

  114 MB is above the 100 MB target. What is left is mostly results the model genuinely
  needs to keep for its later steps. The selection test now fails if the peak goes above
  150 MB.

### 2. Lower tap-to-select's peak (high) · done

- **Where:** `sam2.js`.
- **Change:**
  - Lower the attention score slice limit (`LIMIT`, now 2²⁴ values) to about 2²².
  - Use smaller position slices (`TOKENS`, now 8,192) in the first stage.
  - Free each weight right after the step that uses it.
  - Measure the decoder on its own.
- **Expected:** encoder peak under about 150 MB, a little slower.
- **Check:** `tf.profile` peaks; `tools/check_sam2.mjs` still matches PyTorch.
- **Result:** the plan above was not enough on its own. A step-by-step memory trace showed
  where the peaks were, and each one was fixed:
  1. **Everything kept until the end of a step:** each encoder block and the whole decoder ran
     in one `tf.tidy`, so all temporary results lived until the end. Blocks now free each
     result as soon as the next one exists, and the decoder runs layer by layer.
  2. **Slices sized by position count:** the last stages have far wider positions, so one
     slice could hold 45 MB. Slices are now sized by memory: about 2 million values (8 MB) per
     temporary result, in every stage.
  3. **Wide intermediates:** q, k and v were computed as one combined result and then split
     (three extra copies); now three products, each freed once rearranged into heads.
     Attention scores are computed in slices of at most 2 million values (was 16.7 million).
     The shortcut that widens channels is computed in bands of rows.
  4. **The GPU running behind:** each slice now waits for the GPU, as in fix 1.

  | Measurement | Before | After |
  |---|---|---|
  | Reading the photo, tensor peak (`tf.profile`) | 562 MB | 160 MB |
  | One tap, tensor peak | 429 MB | 127 MB |
  | Graphics process while reading the photo, above idle | ~930 MB | ~600 MB |
  | Masks vs PyTorch (`tools/check_sam2.mjs`) | 99.96–100% | 99.96–100% |
  | Reading the photo, software GPU | 100–110 s | 115–130 s |

  The graphics-process number improved much less than the tensor peak. That process here is
  a software GPU (SwiftShader) that compiles each of the encoder's ~740 GPU programs into CPU
  code, and that memory seems to grow with the work done rather than with the data. A real
  GPU does not do this, so the tensor peak is the better guide to what an iPhone needs.
  Also: measuring with `tf.profile` itself inflates the graphics process (about 2 GB instead
  of 0.9 GB), so process numbers above were measured without it. The selection test now fails
  if reading the photo or a tap goes above 200 MB.

### 3. Work from a reduced copy of large photos (high)

- **Where:** `setDocument`, `openPhoto`, `projectJSON`, `renderFull`, and the selection code
  that draws `assets.base.img`.
- **Change:**
  - On opening, make a working copy at most about 4096 px on the long side for preview,
    selection and masks.
  - Keep the original **file** (compressed, as the browser gave it) instead of the decoded image
    and its base64 copy.
  - Decode the full original only when exporting, and free it afterwards.
  - Build the project file's text only when saving.
- **Expected:** about 200 MB less in the page process with a 45.7 MP photo; export quality
  unchanged.
- **Check:** process memory with the Z7-size test photo; export still full size (within the
  existing 16.7 MP browser limit).

### 4. One AI job at a time, and give the GPU back afterwards (medium)

- **Change:**
  - While a model runs, pause Lumière's own redrawing; it resumes when the job ends.
  - Never run the subject model and tap-to-select at the same time; queue them.
  - Free tap-to-select's stored photo features when the photo changes (already done) and when
    the user leaves the Mask tab for a while.
- **Expected:** a lower peak from the two GPU users overlapping.

### 5. Recover when the WebGL context is lost (medium)

- **Where:** the WebGL setup at the top of the renderer in `index.html`.
- **Change:** on `webglcontextlost`, prevent the default and show "Restoring the picture…". On
  `webglcontextrestored`, recompile the shaders and re-upload the photo, light images and mask
  textures from memory, then redraw. If TensorFlow.js loses its context, reload its models on
  the next use and say so.
- **Expected:** a GPU reset becomes a short flicker instead of a black photo. The document is
  never lost.
- **Check:** force a loss in a test with the `WEBGL_lose_context` extension; the photo comes back
  and undo still works.

### 6. Guards so this does not come back (medium)

- **Memory budget in the tests:** `npm run test:selection` fails if Select Subject peaks above
  150 MB or tap-to-select above 200 MB (`tf.profile`). Run it before every release.
- **Big-photo test:** the smoke test also opens a 45.7 MP photo (generated, not committed).
- **Before downloading the 62 MB model:** check the device's WebGL limits, and on devices that
  look too small, explain that tap-to-select is not available and offer the box.

## Done when

- The friend's iPhone 17 Pro runs Select Subject, a preset and tap-to-select with her Nikon Z7
  photo and with an iPhone photo, twice in a row, without the page going black.
- Peaks measured by the tests: Select Subject < 150 MB (now 114 MB), tap-to-select < 150 MB.
- Masks unchanged: Select Subject identical on the sample; tap-to-select still matches PyTorch.
- Also checked on an older iPhone and a mid-range Android phone.

## Open questions

- Was her photo from the iPhone or the Nikon? (It should not matter after fixes 1 and 3.)
- Does the sample photo crash too? (Confirms the diagnosis.)
- Can someone with a Mac connect the iPhone to Safari's Web Inspector while reproducing it? It
  would show which process died.
