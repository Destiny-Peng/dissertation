"use strict";

(function installDynamicDatasetScopes() {
  if (!window.state || typeof window.byId !== "function") return;

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
    return suite;
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
    var valid = previous === "all" || rows.some(function (row) { return row.value === previous; });
    select.value = valid ? previous : "all";
    return { changed: select.value !== previous, value: select.value };
  }

  function scopeDisplayMap(groups) {
    var labels = {
      all: "All loaded rollouts",
      controlled_analysis: "Controlled"
    };
    groups.suites.forEach(function (row) { labels[row.value] = row.label; });
    return labels;
  }

  function installScopeMatching() {
    var dynamicMatcher = function (record, scope) {
      if (scope === "all") return true;
      if (scope === "controlled_analysis") return isControlled(record);
      return !isControlled(record) && String(record.task_suite || "") === String(scope || "");
    };
    window.baselineBatchMatchesScope = dynamicMatcher;
    try { baselineBatchMatchesScope = dynamicMatcher; } catch (_) {}

    var partitionMatcher = function (record, partition) {
      return partition === "all"
        || String(record.analysis_partition || "unknown") === String(partition || "");
    };
    window.workspacePartitionMatches = partitionMatcher;
    try { workspacePartitionMatches = partitionMatcher; } catch (_) {}
  }

  function refreshDatasetScopes() {
    if (!(state.rollouts || []).length) return;

    installScopeMatching();
    var groups = datasetGroups();
    var rows = scopeRows(groups);
    var labels = scopeDisplayMap(groups);

    if (window.BASELINE_BATCH_SCOPE_LABELS) {
      Object.keys(window.BASELINE_BATCH_SCOPE_LABELS).forEach(function (key) {
        delete window.BASELINE_BATCH_SCOPE_LABELS[key];
      });
      Object.keys(labels).forEach(function (key) {
        window.BASELINE_BATCH_SCOPE_LABELS[key] = labels[key];
      });
    }
    try {
      Object.keys(BASELINE_BATCH_SCOPE_LABELS).forEach(function (key) {
        delete BASELINE_BATCH_SCOPE_LABELS[key];
      });
      Object.keys(labels).forEach(function (key) {
        BASELINE_BATCH_SCOPE_LABELS[key] = labels[key];
      });
    } catch (_) {}

    var baselineScope = byId("baselineBatchScope");
    var baselineResult = replaceScopeSelect(baselineScope, rows, baselineScope && baselineScope.value);
    if (baselineResult.changed && typeof window.baselineBatchScopeChanged === "function") {
      window.baselineBatchScopeChanged();
    } else if (typeof window.updateBaselineBatchSelection === "function") {
      window.updateBaselineBatchSelection();
    } else {
      try {
        if (typeof updateBaselineBatchSelection === "function") updateBaselineBatchSelection();
      } catch (_) {}
    }

    var analysisRunScope = byId("analysisRunScope");
    var preferredRunScope = null;
    try { preferredRunScope = workspaceState.analysisRunScope; } catch (_) {}
    var runResult = replaceScopeSelect(
      analysisRunScope,
      rows,
      preferredRunScope || (analysisRunScope && analysisRunScope.value)
    );
    try { workspaceState.analysisRunScope = runResult.value; } catch (_) {}
    if (analysisRunScope && runResult.changed) {
      analysisRunScope.dispatchEvent(new Event("change", { bubbles: true }));
    }

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

    var partitionLabelNode = partitionSelect && partitionSelect.closest("label")
      ? partitionSelect.closest("label").querySelector("span") : null;
    if (partitionLabelNode) partitionLabelNode.textContent = "Partition";

    var runHelp = document.querySelector(".analysis-run-help");
    if (runHelp) {
      runHelp.textContent = "Uses completed baseline outputs only; it does not start GPU inference. Dataset groups are discovered from the task_suite values in the manifests loaded for this server run.";
    }
    var analysisSubtitle = document.querySelector("#analysisView .page-subtitle");
    if (analysisSubtitle) {
      analysisSubtitle.textContent = "Conclusion-first view of the currently loaded annotated datasets. Task suites and partitions are discovered from the active manifests.";
    }
  }

  var previousPersistentJobScope = window.persistentJobScope;
  if (typeof previousPersistentJobScope === "function") {
    var dynamicPersistentJobScope = function (job) {
      if (job && job.job_type === "baseline" && job.scope) {
        var groups = datasetGroups();
        var labels = scopeDisplayMap(groups);
        if (labels[job.scope]) return labels[job.scope];
      }
      return previousPersistentJobScope(job);
    };
    window.persistentJobScope = dynamicPersistentJobScope;
    try { persistentJobScope = dynamicPersistentJobScope; } catch (_) {}
  }

  var previousDataChanged = window.lf3rWorkspaceDataChanged;
  window.lf3rWorkspaceDataChanged = function () {
    if (typeof previousDataChanged === "function") previousDataChanged();
    refreshDatasetScopes();
  };

  refreshDatasetScopes();
})();
