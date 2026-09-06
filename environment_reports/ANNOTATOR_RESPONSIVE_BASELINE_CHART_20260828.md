# Annotator Responsive Baseline Chart Report — 2026-08-28

## Result

The baseline signal curve now resizes with the video timeline instead of remaining constrained by a fixed 520px SVG viewport. The plotted curve, baseline-chart onset overlay, and playback-timeline markers use the same normalized frame domain and responsive horizontal track geometry.

## Root cause

`static/app.js` emitted an SVG with a `520 x 105` viewBox while CSS fixed its height at 105px. With the browser's default `preserveAspectRatio="xMidYMid meet"`, a wider chart viewport kept the plotted SVG content at its intrinsic 520px width. The CSS-positioned progress markers continued to expand with their container, so the curve and timeline diverged as the page width changed.

## Implementation

- `static/app.js` now uses a normalized `0 ... 100` horizontal SVG coordinate system and explicitly sets `preserveAspectRatio="none"`.
- The curve SVG and evaluation onset markers are siblings inside `evaluation-chart-plot`, so both use exactly the same responsive plot box.
- The plot box and playback `timeline-markers` share the `--timeline-track-inset` CSS variable. Signal strokes use non-scaling SVG strokes so resizing changes the horizontal extent without distorting line weight.
- The video frame domain remains `0 ... total_frames-1`; only display geometry changed. Signal values and onset frame semantics are unchanged.

## Verification

- `python3 -m unittest discover -s tools/lf3r_annotator/tests -p 'test_*.py' -v`: 12/12 tests passed, including frontend geometry contract and server baseline-output tests.
- Python compilation of annotator modules passed.
- Static contract checks confirm normalized SVG coordinates, `preserveAspectRatio="none"`, the shared plot container, and the shared responsive inset.
- Firefox headless was attempted at a local loopback server but could not produce a screenshot in this environment because its software compositor failed to map the framebuffer; the temporary server was stopped cleanly. No browser screenshot is claimed as evidence.
- `verify_pipeline.py` was also run; its failure is pre-existing for the current 136-record manifest because that verifier still expects 3–12 primary natural records and at least one controlled record. Its annotator-file requirement passed.
