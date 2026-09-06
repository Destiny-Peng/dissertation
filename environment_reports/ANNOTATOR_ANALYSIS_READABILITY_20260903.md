# LF3R Analysis Readability and Information Hierarchy - 2026-09-03

## Result

The live annotator Analysis page was reorganized around a conclusion-first dashboard. The default view is the 125-rollout primary_natural selection with Q95 threshold, local level feature, 16-frame scale, and all events. reference_natural, natural_observation, and controlled_analysis remain explicit choices.

Analysis now has six hash-routed sub-tabs:

- Overview: live annotation statistics, outcome distribution, method coverage, and compact conclusions.
- Method comparison: readable horizontal metric bars for one feature/scale/threshold slice at a time.
- Failure types: an HTML heatmap with short method/signal labels, selectable metrics, and sample counts.
- Event explorer: server-paginated event rows with filters, sorting, and Review links.
- Signal shape: one method/signal/event-group onset-aligned curve at a time, with native samples, median, IQR, and onset marker.
- Archive & downloads: collapsed legacy material and declared snapshot artifact links.

Categorical high-cardinality data no longer use one overloaded rotated SVG axis. Continuous onset curves retain responsive SVG viewBox geometry, clipping, title/aria-label metadata, and horizontal-scroll protection where a genuine detail table needs it. The static WeeklySummary/9.3/9.3.html report uses the same hierarchy and presents compact evidence plus download links instead of copying full CSV contents into the page.

## API changes

GET /api/analysis now returns dashboard/compact mode by default. It includes live statistics, snapshot provenance and freshness, method coverage, aggregate rows, detail counts, available tabs, and server-declared artifact links. Large event-level and frame arrays are not sent on the initial page load. Legacy consumers can request GET /api/analysis?view=full.

GET /api/analysis/details provides on-demand pagination for fixed analysis kinds. It validates the kind, page, page size (1-100), supported filters, and supported sort fields before reading the current legal snapshot. GET /api/analysis/artifacts/<name> serves only an allow-listed artifact from the current snapshot and rejects arbitrary paths or path traversal.

The browser loads non-Overview data lazily when a sub-tab is entered and keeps the existing rollout selection, Review filters, annotation state, and Review links. The API still tolerates older snapshots that lack localization or change-point files and reports unavailable sections instead of blocking the live overview.

## Interpretation and data integrity

This is a presentation and read-path change only. It does not rerun temporal analysis, baseline inference, or rollout generation. No rollout media, manifest, annotation, or baseline raw output was modified. Existing native sampling alignment and semantic event colors are retained. Localization, recall, false-alarm, F1, AUROC, hit-rate, and error values remain descriptive diagnostics on the annotated collection, not independently validated detector-performance results.

## Verification

- Existing server and frontend contract tests were extended for dashboard metadata, details pagination, artifact allow-listing, six sub-tabs, readable chart containers, and SVG clipping.
- Python compilation covers server.py, verify_pipeline.py, analyze_baseline_temporal_signals.py, and run_openvla_libero10_natural.py.
- The static report was checked for the new default scope, compact sections, and archive links.
- Browser screenshot/manual visual validation was not run in this environment; the implementation uses responsive CSS rules for 75%-160% root scale and narrow windows. No GPU inference was started.
