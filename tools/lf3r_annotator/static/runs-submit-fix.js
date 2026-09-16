"use strict";

(function installBatchSubmitFix() {
  var form = document.getElementById("baselineBatchForm");
  if (!form || form.dataset.submitFixInstalled === "true") return;
  form.dataset.submitFixInstalled = "true";

  var originalStart = window.startBaselineBatch;
  if (typeof originalStart === "function") {
    form.removeEventListener("submit", originalStart);
  }

  function setRejectedRequestLog(message, body) {
    var log = document.getElementById("baselineBatchLog");
    if (!log) return;
    log.textContent = [
      "Request rejected before a baseline job was created.",
      "",
      String(message || "Unknown request validation error"),
      "",
      "Submitted request:",
      JSON.stringify(body, null, 2)
    ].join("\n");
  }

  async function startBaselineBatchFixed(event) {
    if (event) event.preventDefault();
    if (!window.state || state.baselineBatchSubmitting) return;

    var method = byId("baselineBatchMethod").value;
    var scope = byId("baselineBatchScope").value;
    var condition = byId("baselineBatchCondition").value || "full_instruction";
    var memory = Number(byId("baselineBatchMemoryUtilization").value);
    readBaselineBatchWorkers();

    var range = baselineBatchTotalRange();
    var summary = baselineBatchWorkerSummary();
    var workers = (state.baselineBatchWorkers || []).map(function (worker) {
      return {
        gpu: String(worker.gpu || "").trim(),
        start_index: baselineBatchNumber(worker.start_index),
        end_index: baselineBatchNumber(worker.end_index)
      };
    });

    if (!range.valid || summary.invalid) {
      setBaselineBatchStatus("Total range and every worker must use valid ranges inside the scope.", "error");
      return;
    }
    if (workers.some(function (worker) { return !/^\d+$/.test(worker.gpu); })) {
      setBaselineBatchStatus("Every worker GPU must be one numeric CUDA device index.", "error");
      return;
    }
    if (!Number.isFinite(memory) || memory <= 0 || memory > 1) {
      setBaselineBatchStatus("Free-memory fraction must be between 0 and 1.", "error");
      byId("baselineBatchMemoryUtilization").focus();
      return;
    }

    var selected = baselineBatchSelectionRecords();
    if (!selected.length) {
      setBaselineBatchStatus("This scope and worker assignment selects no rollouts.", "warning");
      return;
    }

    var uniqueGpus = [];
    workers.forEach(function (worker) {
      if (uniqueGpus.indexOf(worker.gpu) === -1) uniqueGpus.push(worker.gpu);
    });
    var gpu = uniqueGpus.join(",") || "0";

    var confirmation = "Run " + method + " for " + instructionConditionLabel(condition)
      + " over " + selected.length + " unique rollout(s)? This launches GPU inference."
      + " It will start " + workers.length + " rollout worker(s).";
    if (!window.confirm(confirmation)) return;

    state.baselineBatchSubmitting = true;
    updateBaselineBatchSelection();
    setBaselineBatchStatus("Starting " + method + " batch for " + selected.length + " rollout(s)…", "");
    byId("baselineBatchLog").textContent = "";

    var body = {
      baseline: method,
      scope: scope,
      instruction_condition: condition,
      gpu: gpu,
      memory_utilization: memory,
      start_index: range.start,
      limit: null,
      options: baselineBatchOptions()
    };

    /* One-worker requests do not need an explicit worker row. Let the server
       create that row from its authoritative condition/scope record count.
       This avoids rejecting a request when the browser's cached rollout list
       and the current instruction-variant manifest differ by a small amount. */
    if (workers.length === 1) {
      body.parallel_workers = 1;
      if (!(range.start === 0 && range.end === range.matched)) {
        body.end_index = range.end;
      }
    } else {
      body.end_index = range.end;
      body.parallel_workers = workers.length;
      body.workers = workers;
    }

    try {
      var response = await fetch("/api/baselines/run-batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });

      var responseText = await response.text();
      var payload = {};
      try {
        payload = responseText ? JSON.parse(responseText) : {};
      } catch (_) {
        payload = { error: responseText || "Server returned a non-JSON response" };
      }

      if (!response.ok) {
        var message = payload.error || ("HTTP " + response.status + " while starting baseline batch");
        setRejectedRequestLog(message, body);
        console.error("LF3R baseline batch request rejected", { status: response.status, message: message, request: body, response: payload });
        throw new Error(message);
      }

      state.baselineBatchJob = payload.job;
      rememberPersistentJob(payload.job);
      await loadBaselineBatchLog(payload.job.job_id);
      pollBaselineBatchJob(payload.job.job_id);
    } catch (error) {
      setBaselineBatchStatus("Batch baseline error: " + error.message, "error");
      if (!byId("baselineBatchLog").textContent.trim()) {
        setRejectedRequestLog(error.message, body);
      }
    } finally {
      state.baselineBatchSubmitting = false;
      updateBaselineBatchSelection();
    }
  }

  window.startBaselineBatch = startBaselineBatchFixed;
  form.addEventListener("submit", startBaselineBatchFixed);
})();
