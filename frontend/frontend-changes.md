# Frontend Changes — Dark/Light Theme Toggle

## Files Modified

### `index.html`
- Bumped CSS cache-bust version to `v=11` and JS version to `v=10`.
- Added a `<button id="themeToggle">` element positioned outside `.container` (fixed overlay) containing inline SVG moon icon (shown in dark mode) and sun icon (shown in light mode). Both carry `aria-hidden="true"`; the button itself has `aria-label` and `title` for accessibility.

### `style.css`
- **Light theme variables** — Added `[data-theme="light"]` block with a full set of CSS custom property overrides:
  - `--background: #f8fafc`, `--surface: #ffffff`, `--surface-hover: #e2e8f0`
  - `--text-primary: #1e293b`, `--text-secondary: #64748b`
  - `--border-color: #e2e8f0`, `--assistant-message: #f1f5f9`
  - `--shadow` uses lower opacity for lighter elevation feel
  - New toggle-specific vars: `--toggle-bg`, `--toggle-border`, `--toggle-color`, `--toggle-hover-bg`
- **Smooth transitions** — Added a selector list targeting `body`, sidebar, chat containers, inputs, messages, and the toggle button; each transitions `background-color`, `color`, and `border-color` over 0.25 s.
- **`.theme-toggle` styles** — Fixed-position circle button (40×40 px, `border-radius: 50%`) in the top-right corner. Hover changes border and icon to `--primary-color`. Focus shows the existing blue focus ring.
- **Icon visibility rules** — `.icon-sun { display: none }` by default; `.icon-moon { display: block }`. Reversed under `[data-theme="light"]` so the correct icon always shows.
- **Light theme code-block overrides** — `[data-theme="light"]` rules lower the `rgba(0,0,0,…)` background on `code` and `pre` from 0.2 to 0.06 opacity for legibility, and adjust the sources list divider opacity.

### `script.js`
- Added `applyTheme(theme)` — sets `data-theme` on `<html>` and persists to `localStorage`.
- Added `initTheme()` — reads `localStorage`; falls back to `prefers-color-scheme` media query so the initial theme matches the OS preference.
- Added `toggleTheme()` — flips between `"dark"` and `"light"`.
- `themeToggle` DOM ref added alongside existing refs; `initTheme()` called before other setup in `DOMContentLoaded`; click listener wired in `setupEventListeners()`.
