# Frontend Changes: Dark/Light Theme Toggle

## Summary

Added a toggle button that lets users switch between the existing dark theme and a new light theme. The choice persists across sessions via `localStorage`.

## Files Changed

### `frontend/index.html`

- Added an inline script at the very top of `<head>` (before the stylesheet) that reads the saved theme from `localStorage` and sets `data-theme` on `<html>` synchronously, so the correct theme is applied before first paint (no flash of the wrong theme).
- Added a `#themeToggle` `<button>` as the first element in `<body>`, containing two inline SVG icons (sun and moon). It has `aria-label`, `aria-pressed`, and `title` attributes for accessibility; being a native `<button>`, it's keyboard-focusable and activates with Enter/Space by default.
- Bumped cache-busting query params (`style.css?v=11`, `script.js?v=11`).

### `frontend/style.css`

- Added a `:root[data-theme="light"]` block that overrides the existing CSS custom properties (`--background`, `--surface`, `--surface-hover`, `--text-primary`, `--text-secondary`, `--border-color`, `--assistant-message`, `--shadow`, `--focus-ring`, `--welcome-bg`, `--welcome-border`) with light, high-contrast equivalents. `--primary-color`/`--primary-hover`/`--user-message` are kept the same across themes for brand consistency.
- Added `transition: background-color/color/border-color/box-shadow` to `body` and the main surface elements (sidebar, chat area, input, buttons, message bubbles, etc.) so switching themes animates smoothly instead of snapping.
- Added `.theme-toggle` styles: a circular fixed-position button pinned to the top-right corner (`position: fixed; top/right: 1rem`), using theme variables for its own background/border/color so it themes itself automatically. Includes hover/active/focus-visible states (focus ring reuses `--focus-ring`, matching other interactive elements).
- Added a crossfade + rotate animation between the sun and moon icons, driven purely by CSS via the `:root[data-theme="light"]` selector — no JS needed to swap icons.
- Added a small-screen media query to shrink the button/icons slightly on mobile.

### `frontend/script.js`

- Added `themeToggle` to the cached DOM elements and wired a `click` listener to it in `setupEventListeners()`.
- Added `initTheme()` (called on `DOMContentLoaded`) which re-applies the saved/default theme and syncs the button's ARIA state (the inline `<head>` script already set the attribute pre-paint; this keeps the button's accessible state in sync).
- Added `toggleTheme()` which flips `data-theme` between `"dark"` and `"light"` on `<html>` and persists the choice to `localStorage`.
- Added `applyTheme(theme)` helper that sets `data-theme`, and updates `aria-pressed`/`aria-label` on the toggle button to reflect the current state for screen readers.

## Verification

Served the frontend statically and drove it with Playwright: confirmed the button toggles dark → light → (on reload) light theme correctly, icon crossfade works, and the light theme has good text/background contrast throughout the welcome message, sidebar, and input areas.
