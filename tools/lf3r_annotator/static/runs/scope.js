"use strict";

window.LF3RDatasetScopes = (function createDatasetScopeController() {
  function isControlled(record) {
    return Boolean(record) && (
      record.analysis_partition === "controlled_analysis"
      || record.source_kind === "controlled_injected"
    );
  }

  function suiteLabel(value, records) {
    var suite = String(value || "");
    var explicit = (records || []).find(function (record) {
      return String(record.task_suite || "") === suite && record.task_suite_label;
    });
    if (explicit) return String(explicit.task_suite_label);
    if (suite === "libero_10") return "LIBERO-10";
    if (suite === "libero_spatial") return "LIBERO-Spatial";
    return suite || "Unknown suite";
  }

  function partitionLabel(value) {
    var labels = {
      natural_observation: "Natural observation",
      controlled_analysis: "Controlled",
      real_robot_analysis: "Real robot"
    };
    return labels[value] || String(value || "unknown");
  }

  function datasetGroups() {
    var records = state.rollouts || [];
    var suites = [];
    var counts = {};
    var controlledCount = 0;

    records.forEach(function (record) {
      if (isControlled(record)) {
        controlledCount += 1;
        return;
      }
      var suite = String(record.task_suite || "").trim();
      if (!suite) return;
      if (!Object.prototype.hasOwnProperty.call(counts, suite)) {
        counts[suite] = 0;
        suites.push(suite);
      }
      counts[suite] += 1;
    });

    return {
      suites: suites.map(function (suite) {
        return {
          value: suite,
          label: suiteLabel(suite, records),
          count: counts[suite]
        };
      }),
      controlledCount: controlledCount,
      totalCount: records.length
    };
  }

  function partitionGroups() {
    var records = state.rollouts || [];
    var values = [];
    var counts = {};
    records.forEach(function (record) {
      var value = String(record.analysis_partition || "unknown");
      if (!Object.prototype.hasOwnProperty.call(counts, value)) {
        counts[value] = 0;
        values.push(value);
      }
      counts[value] += 1;
    });
    return values.map(function (value) {
      return { value: value, label: partitionLabel(value), count: counts[value] };
    });
  }

  function scopeRows(groups) {
    var rows = groups.suites.slice();
    if (groups.controlledCount) {
      rows.push({
        value: "controlled_analysis",
        label: "Controlled",
        count: groups.controlledCount
      });
    }
    rows.push({
      value: "all",
      label: "All loaded rollouts",
      count: groups.totalCount
    });
    return rows;
  }

  function scopeLabels() {
    var groups = datasetGroups();
    var labels = {
      all: "All loaded rollouts",
      controlled_analysis: "Controlled"
    };
    groups.suites.forEach(function (row) {
      labels[row.value] = row.label;
    });
    return labels;
  }

  function scopeLabel(scope) {
    var labels = scopeLabels();
    return labels[scope] || String(scope || "unspecified scope");
  }

  function matchesBaseline(record, scope) {
    if (scope === "all") return true;
    if (scope === "controlled_analysis") return isControlled(record);
    return !isControlled(record)
      && String(record && record.task_suite || "") === String(scope || "");
  }

  function matchesPartition(record, partition) {
    return partition === "all"
      || String(record && record.analysis_partition || "unknown") === String(partition || "");
  }

  function optionHtml(row) {
    return '<option value="' + escapeHtml(row.value) + '">' + escapeHtml(row.label)
      + " (" + Number(row.count || 0) + ")</option>";
  }

  function replaceScopeSelect(select, rows, preferred) {
    if (!select) return { changed: false, value: preferred || "all" };
    var previous = preferred || select.value || "";
    select.innerHTML = rows.map(optionHtml).join("");
    var valid = rows.some(function (row) { return row.value === previous; });
    var fallback = rows.length ? rows[0].value : "all";
    select.value = valid ? previous : fallback;
    return { changed: select.value !== previous, value: select.value };
  }

  function replacePartitionSelect(select, rows, preferred) {
    if (!select) return { changed: false, value: preferred || "all" };
    var previous = preferred || select.value || "all";
    select.innerHTML = '<option value="all">All partitions</option>'
      + rows.map(optionHtml).join("");
    var valid = previous === "all" || rows.some(function (row) {
      return row.value === previous;
    });
    select.value = valid ? previous : "all";
    return { changed: select.value !== previous, value: select.value };
  }

  function decorateHelp(key, entry) {
    if (!entry || key !== "baseline.scope") return entry;
    return Object.assign({}, entry, {
      default: "first loaded task suite",
      description: "Task-suite scopes are discovered from the task_suite values in the manifests loaded for the current server run. Controlled rollouts remain a separate scope, and All loaded rollouts selects the full catalog."
    });
  }

  function applyStaticLabels() {
    var title = document.getElementById("baselineBatchTitle");
    if (title) title.textContent = "Run one method over a dataset group";

    var memory = document.getElementById("baselineBatchMemoryUtilization");
    var memoryLabel = memory && memory.closest("label");
    var memoryCaption = memoryLabel && memoryLabel.querySelector("span");
    if (memoryCaption) memoryCaption.textContent = "vLLM free-memory target";

    var resourceNote = document.querySelector("#baselineBatchForm .batch-resource-warning");
    if (resourceNote) {
      resourceNote.textContent = "GPU choice is user-managed. The status panel is informational; the WebUI does not block launch based on utilization or free-memory thresholds.";
    }

    var suite = document.getElementById("rolloutGenerationSuite");
    if (suite) {
      var libero10 = suite.querySelector('option[value="libero_10"]');
      var spatial = suite.querySelector('option[value="libero_spatial"]');
      if (libero10) libero10.textContent = "LIBERO-10";
      if (spatial) spatial.textContent = "LIBERO-Spatial";
    }

    var baselineJobs = document.getElementById("baselineBatchJobs");
    if (baselineJobs) baselineJobs.setAttribute("aria-label", "Persistent baseline jobs");
    var baselineLog = document.getElementById("baselineBatchLog");
    if (baselineLog) baselineLog.setAttribute("aria-label", "Selected baseline job log");
    var activity = baselineJobs && baselineJobs.closest(".runs-activity");
    var activityTitle = activity && activity.querySelector(".runs-activity-heading h3");
    if (activityTitle) activityTitle.textContent = "Baseline jobs";
  }

  function refresh() {
    applyStaticLabels();
    if (!(state.rollouts || []).length) return;

    var groups = datasetGroups();
    var rows = scopeRows(groups);

    var baselineScope = byId("baselineBatchScope");
    var baselineResult = replaceScopeSelect(
      baselineScope,
      rows,
      baselineScope && baselineScope.value
    );
    var resultFilter = byId("baselineBatchResultFilter");
    if (resultFilter && resultFilter.value === "missing_valid"
        && typeof baselineBatchCoverageChanged === "function") {
      baselineBatchCoverageChanged();
    } else if (baselineResult.changed && typeof baselineBatchScopeChanged === "function") {
      baselineBatchScopeChanged();
    } else if (typeof updateBaselineBatchSelection === "function") {
      updateBaselineBatchSelection();
    }

    ["analysisOutcomeRunScope", "analysisHopScope"].forEach(function (id) {
      var analysisScope = byId(id);
      if (!analysisScope) return;
      var result = replaceScopeSelect(
        analysisScope,
        rows,
        analysisScope.value
      );
      if (result.changed) {
        analysisScope.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });

    var partitionRows = partitionGroups();
    var partitionSelect = byId("analysisPartitionFilter");
    var preferredPartition = null;
    try { preferredPartition = workspaceState.liveFilters.partition; } catch (_) {}
    var partitionResult = replacePartitionSelect(
      partitionSelect,
      partitionRows,
      preferredPartition || (partitionSelect && partitionSelect.value)
    );
    try { workspaceState.liveFilters.partition = partitionResult.value; } catch (_) {}
    if (partitionSelect && partitionResult.changed) {
      partitionSelect.dispatchEvent(new Event("change", { bubbles: true }));
    }

    var partitionCaption = partitionSelect && partitionSelect.closest("label")
      ? partitionSelect.closest("label").querySelector("span")
      : null;
    if (partitionCaption) partitionCaption.textContent = "Partition";

    var runHelp = document.querySelector(".analysis-run-help");
    if (runHelp) {
      runHelp.textContent = "Uses completed baseline outputs only; it does not start GPU inference. Dataset groups are discovered from the task_suite values in the manifests loaded for this server run.";
    }
    var analysisSubtitle = document.querySelector("#analysisView .page-subtitle");
    if (analysisSubtitle) {
      analysisSubtitle.textContent = "Conclusion-first view of the currently loaded annotated datasets. Task suites and partitions are discovered from the active manifests.";
    }
  }

  applyStaticLabels();
  window.setTimeout(refresh, 0);

  return {
    refresh: refresh,
    matchesBaseline: matchesBaseline,
    matchesPartition: matchesPartition,
    scopeLabel: scopeLabel,
    decorateHelp: decorateHelp
  };
})();
