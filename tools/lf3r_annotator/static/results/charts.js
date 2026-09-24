"use strict";

window.LF3RResultsCharts = (function createResultsCharts() {
  function signalDomain(method, name, values) {
      // ProcVLM progress is emitted as a percentage on the 0–100 scale.
      if (method === "procvlm" && name === "progress") {
        return { low: 0, high: 100, fixed: true };
      }
  
      // Robo-Dopamine progress and its perspective components share the same
      // normalized [0, 1] semantics. Hop is a temporal difference and may be
      // signed, so keep hop/component-hop on the original data-driven scale.
      if (method === "robo_dopamine" && (name === "progress" || /_progress$/.test(name))) {
        return { low: 0, high: 1, fixed: true };
      }
  
      var low = Math.min.apply(null, values);
      var high = Math.max.apply(null, values);
      if (low === high) {
        var pad = Math.max(Math.abs(low) * 0.05, 0.01);
        low -= pad;
        high += pad;
      }
      return { low: low, high: high, fixed: false };
    }

  function renderLocalizationPredictionCurve(method, result, record) {
    if (method !== "robo_dopamine" || !result || !result.localization_prediction) return "";
    var prediction = result.localization_prediction;
    var frames = Array.isArray(prediction.frames) ? prediction.frames : [];
    var scores = Array.isArray(prediction.sigmoid_scores) ? prediction.sigmoid_scores : [];
    var count = Math.min(frames.length, scores.length);
    if (!count) return "";
  
    var rows = [];
    for (var index = 0; index < count; index += 1) {
      var frame = Number(frames[index]);
      var score = Number(scores[index]);
      if (!Number.isFinite(frame) || !Number.isFinite(score)) continue;
      rows.push({ frame: frame, score: Math.max(0, Math.min(1, score)) });
    }
    if (!rows.length) return "";
  
    var maxFrame = Math.max.apply(null, rows.map(function (row) { return row.frame; }));
    var domain = timelineDomainMax(record, maxFrame);
    var points = rows.map(function (row) {
      return (Math.max(0, Math.min(domain, row.frame)) / Math.max(1, domain) * 100).toFixed(4)
        + ',' + (87 - row.score * 79).toFixed(4);
    }).join(' ');
  
    return '<section class="signal-row evaluation-localization-curve" data-localization-curve>'
      + '<div class="signal-row-title">Localization score</div><div class="signal-axis-layout">'
      + '<div class="signal-y-axis" aria-label="Localization score axis"><span>1</span><span>0.5</span><span>0</span></div>'
      + '<div class="evaluation-chart"><div class="timeline-track evaluation-chart-plot" data-signal-seek data-frame-max="'
      + domain + '" tabindex="0" role="slider" aria-label="Localization score: click to seek video; arrow keys step frames"'
      + ' aria-valuemin="0" aria-valuemax="' + domain + '" aria-valuenow="' + currentFrame() + '">'
      + '<svg viewBox="0 0 100 105" preserveAspectRatio="none" role="img" aria-label="Localization checkpoint score versus video frame">'
      + '<title>Localization checkpoint sigmoid score; raw logits are preserved in the saved localization artifact</title>'
      + '<path d="M0 8H100 M0 47.5H100 M0 87H100" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>'
      + '<polyline fill="none" stroke="#d6a5f5" stroke-width="1.8" vector-effect="non-scaling-stroke" points="'
      + points + '"/></svg>'
      + '<div class="evaluation-chart-markers" data-evaluation-onset-markers data-frame-max="' + domain + '">'
      + renderEvaluationOnsetMarkers(record, domain)
      + renderLocalizationPredictionMarker(method, result, domain)
      + '</div><div class="signal-playhead" data-signal-playhead style="left:'
      + (Math.max(0, Math.min(domain, currentFrame())) / Math.max(1, domain) * 100)
      + '%"></div></div></div></div>'
      + '<div class="signal-frame-axis"><span>0</span><span>video frame</span><span>' + domain + '</span></div>'
      + '<div class="evaluation-meta">Full checkpoint output curve · sigmoid score shown; raw logits are saved alongside it.</div>'
      + '</section>';
  }

  function renderSignalChart(method, result, record) {
      var samples = (result.samples || []).filter(function (sample) {
        return sample && Number.isFinite(Number(sample.frame)) && sample.signals;
      });
      var names = [];
      samples.forEach(function (sample) {
        Object.keys(sample.signals).forEach(function (name) {
          if (names.indexOf(name) === -1) names.push(name);
        });
      });
      if (!samples.length || !names.length) {
        return '<div class="evaluation-empty">No numeric signal samples to plot.</div>';
      }
  
      var domain = timelineDomainMax(
        record,
        Math.max.apply(null, samples.map(function (sample) { return Number(sample.frame); }))
      );
      var colors = ["#67d9b5", "#77bdfb", "#e7c15c", "#ef8d53", "#d6a5f5", "#f07869"];
      var visibility = evaluationSignalVisibility(method, record, names);
      var legend = names.map(function (name, index) {
        var visible = visibility[name] !== false;
        return '<button type="button" class="evaluation-signal-toggle' + (visible ? '' : ' is-hidden')
          + '" data-evaluation-signal-toggle data-signal-name="' + escapeHtml(name)
          + '" aria-pressed="' + visible + '" title="Show/hide '
          + escapeHtml(evaluationSignalLabel(name)) + '"><i style="background:'
          + colors[index % colors.length] + '"></i>'
          + escapeHtml(evaluationSignalLabel(name)) + '</button>';
      }).join('');
  
      var plots = names.map(function (name, index) {
        var rows = samples.filter(function (sample) {
          return sample.signals[name] != null && Number.isFinite(Number(sample.signals[name]));
        });
        if (!rows.length) return '';
  
        var values = rows.map(function (sample) { return Number(sample.signals[name]); });
        var axis = signalDomain(method, name, values);
        var low = axis.low;
        var high = axis.high;
        var span = high - low;
        var points = rows.map(function (sample) {
          var value = Number(sample.signals[name]);
          // Fixed semantic ranges should not be distorted by a malformed sample;
          // clamp only for drawing while the original numeric value remains in
          // the output/history data.
          var plottedValue = axis.fixed ? Math.max(low, Math.min(high, value)) : value;
          return (Math.max(0, Math.min(domain, Number(sample.frame))) / domain * 100).toFixed(4)
            + ',' + (87 - (plottedValue - low) / span * 79).toFixed(4);
        }).join(' ');
  
        var scaleTitle = axis.fixed
          ? 'Fixed semantic value scale ' + low + '–' + high + '; video frame 0–' + domain
          : 'Original signal values; video frame 0–' + domain;
  
        return '<section class="signal-row" data-signal-row data-signal-name="' + escapeHtml(name) + '"'
          + (visibility[name] === false ? ' hidden' : '')
          + '><div class="signal-row-title">'
          + (names.length > 1 ? escapeHtml(evaluationSignalLabel(name)) : '')
          + '</div><div class="signal-axis-layout">'
          + '<div class="signal-y-axis" aria-label="Vertical axis values"><span>'
          + formatEvaluationNumber(high) + '</span><span>'
          + formatEvaluationNumber((high + low) / 2) + '</span><span>'
          + formatEvaluationNumber(low) + '</span></div>'
          + '<div class="evaluation-chart"><div class="timeline-track evaluation-chart-plot" data-signal-seek data-frame-max="'
          + domain + '" tabindex="0" role="slider" aria-label="'
          + escapeHtml(evaluationSignalLabel(name))
          + ': click to seek video; arrow keys step frames" aria-valuemin="0" aria-valuemax="'
          + domain + '" aria-valuenow="' + currentFrame() + '">'
          + '<svg viewBox="0 0 100 105" preserveAspectRatio="none" role="img" aria-label="'
          + escapeHtml(evaluationSignalLabel(name)) + ' versus video frame">'
          + '<title>' + escapeHtml(scaleTitle) + '</title>'
          + '<path d="M0 8H100 M0 47.5H100 M0 87H100" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>'
          + '<polyline fill="none" stroke="' + colors[index % colors.length]
          + '" stroke-width="1.5" vector-effect="non-scaling-stroke" points="'
          + points + '"/></svg>'
          + '<div class="evaluation-chart-markers" data-evaluation-onset-markers data-frame-max="'
          + domain + '">' + renderEvaluationOnsetMarkers(record, domain)
          + (typeof renderLocalizationPredictionMarker === "function"
            ? renderLocalizationPredictionMarker(method, result, domain)
            : "")
          + '</div><div class="signal-playhead" data-signal-playhead style="left:'
          + (Math.max(0, Math.min(domain, currentFrame())) / domain * 100)
          + '%"></div></div></div></div>'
          + '<div class="signal-frame-axis"><span>0</span><span>video frame</span><span>'
          + domain + '</span></div></section>';
      }).join('');
  
      return '<div class="evaluation-chart-legend">' + legend + '</div>' + plots;
    }

  return {
    renderLocalizationPredictionCurve: renderLocalizationPredictionCurve,
    renderSignalChart: renderSignalChart
  };
})();
