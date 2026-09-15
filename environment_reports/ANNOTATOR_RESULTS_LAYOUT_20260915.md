# Annotator / Results split — 2026-09-15

Implemented Annotate/Results navigation with shared player and queue, retained legacy review links, collapsed batch controls, raw-value signal rows, vertical tick labels, playback cursors and click/keyboard seeking. Annotation hotkeys are disabled in Results. Existing per-run and signal selections remain available.

Validation: GJS parsed app.js and workspace.js successfully. Focused executable checks passed for legacy/new route IDs, Analysis subroutes, raw chart bounds, cursor attributes and hidden signals. git diff --check passed. No GPU inference was run. No browser automation was available; real browser width/font visual acceptance remains unverified. Refresh the browser to load the updated versioned static assets; backend changes are not needed.

Follow-up: moved both execution panels to #/runs, expanded advanced parameters, added a browser-persisted Results sticky-player toggle, and scoped video/annotation keyboard shortcuts to Annotate/Results. GJS syntax checks and strict HTML nesting/unique-ID checks passed. Browser visual checks remain pending.
