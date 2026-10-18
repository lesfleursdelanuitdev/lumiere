# Lumière design system · "Darkroom"

The design tokens and component catalogue for Lumière. Open `index.html` in a browser (no build,
no server; `file://` works). From there, **Concept preview** shows two screens built only from these
components, and `icons.html` shows the icon set.

| File | What it is |
|---|---|
| `styles.css` | The single source of styles: tokens as CSS custom properties (`:root`), then component classes |
| `index.html` | Living style guide: foundations + every component with its variants and states. **Dev Mode** inspects a component (props, tokens, metrics); **Copy as prompt** produces an implementation prompt for any language; **Install design system** produces one prompt for the whole system |
| `concept.html` | Two static concept screens: the start screen and the light editor |
| `icons.html` | The inline SVG icon sprite (canonical copy) |

## Structure

Cross-platform: **Material's token model** with **MUI's component catalogue** and prop conventions
(`variant`, `size`, `color`).

- **Palette:** `primary / secondary / error / warning / success / info`, each with `main` (the bare
  name, e.g. `--primary`), `-light`, `-dark` and `--on-*` (contrastText), plus a `-container` /
  `--on-*-container` pair for quiet emphasis. Text: `--text`, `--text-secondary`, `--text-disabled`.
  Background: `--bg` (default), `--paper`. Surface ladder: `--sunken → --bg → --paper → --raised → --overlay`.
  Action states: `--action-hover / -pressed / -selected / -focus / -disabled / -disabled-bg`.
- **Typography:** `h1–h6`, `subtitle1/2`, `body1/2`, `button`, `caption`, `overline` + `mono` for values.
  Manrope (display) × IBM Plex Sans (text) × IBM Plex Mono (numbers).
- **Spacing:** compact 4 px unit (`--sp-2 … --sp-64`). **Sizes:** `--target` 44 px minimum hit area.
- **Shape:** `--r-none/xs/sm/md/lg/full` (8 for controls, 12 for sheets and cards).
- **Shadows:** `--elev-0…3` (MUI 0 → 0, 1–2 → 1, 3–8 → 2, 9–24 → 3). Flat, with hairlines.
- **Transitions, z-index, breakpoints:** MUI values.

## Theme decisions (and why)

- **Graphite neutrals** at OKLCH chroma 0.004, hue 55: neutral enough to judge a photo against, a hair
  warm so it doesn't feel clinical. The UI must never colour-cast the image.
- **Safelight primary** `#FA653D` (OKLCH .69 / .19 / 36): the one active colour, like a darkroom's
  safelight. **Cool-steel secondary** `#92B7CC` for links and blur handles.
- **Error** is a pink-red `#F45A82`, deliberately apart from the orange primary, and status is always
  shown with an icon and words, never colour alone.

## Contrast (WCAG, computed from the hex values)

| Pair | Ratio |
|---|---|
| `--text` on `--bg` / `--paper` / `--raised` / `--overlay` | 15.9 / 14.6 / 13.4 / 12.0 |
| `--text-secondary` on `--bg` / `--paper` / `--raised` / `--overlay` | 8.2 / 7.6 / 6.9 / 6.2 |
| `--primary` with `--on-primary` (ink) · as text on `--bg` | 6.3 · 6.3 |
| `--secondary` with ink · `--error` with ink · `--warning` · `--success` · `--info` | 8.9 · 6.0 · 10.7 · 9.7 · 8.6 |
| every `--*-container` with its `--on-*-container` | ≥ 8.6 |
| `--outline-strong` (input borders) on `--paper` | 3.2 (UI ≥ 3:1) |

`--*-dark` variants (3.7–3.9:1 with ink) are for borders and pressed edges only, never under text.

## How to extend

- **Add a token:** add the custom property to the right group in `styles.css` `:root`, add its name to
  `TOKEN_GROUPS` in `index.html` (so prompts include it), and reference it by name. Never inline a value.
- **Add a component:** a CSS class block in `styles.css` that uses only tokens, a
  `<section id="<key>">` in `index.html` showing every variant and state with `data-insp="<key>"`,
  and a `SPECS["<key>"]` entry (name, file, desc, props, tokens, metrics). The Copy-as-prompt button
  appears automatically. App-specific components go under **Extensions**.
- **Theming:** add a `:root[data-theme="…"]` block that overrides tokens only; components don't change.
  Keep every fill paired with its `--on-*` and re-check contrast with the WCAG formula.
- **Icons:** add a `<symbol id="i-…" viewBox="0 0 24 24">` to the sprite in `icons.html`, then copy the
  sprite block into `index.html` and `concept.html` (they embed it to stay self-contained).
