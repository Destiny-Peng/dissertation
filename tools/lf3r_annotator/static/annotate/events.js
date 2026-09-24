"use strict";

/* Annotate, Results, Runs, and shared UI event wiring. */

function installEvents() {
  try {
    var collapsed = JSON.parse(
      sessionStorage.getItem("lf3r.results.baselineCollapsed") || "{}"
    );
    if (collapsed && typeof collapsed === "object") {
      state.baselineCollapsed = collapsed;
    }
    state.roboPosthocCheckpoint =
      sessionStorage.getItem("lf3r.results.roboPosthocCheckpoint") || "";
  } catch (_error) {}
  ["searchInput", "originFilter", "manifestFilter", "outcomeFilter", "reviewFilter"].forEach(function (id) {
    byId(id).addEventListener(id === "searchInput" ? "input" : "change", applyFilters);
  });
  byId("frameSlider").addEventListener("input", function (event) {
    seekFrame(Number(event.target.value));
  });
  document.querySelectorAll("[data-step]").forEach(function (button) {
    button.addEventListener("click", function () { stepFrames(Number(button.dataset.step)); });
  });
  byId("addFailureEvent").addEventListener("click", function () {
    addFailureEvent();
  });
  byId("failureEvents").addEventListener("click", function (event) {
    var removeButton = event.target.closest("[data-remove-event]");
    if (removeButton) {
      removeFailureEvent(Number(removeButton.dataset.removeEvent));
      return;
    }
    var setButton = event.target.closest("[data-event-set]");
    if (setButton) {
      var setCard = setButton.closest(".failure-event-card");
      setActiveFailureEvent(Number(setCard.dataset.eventIndex));
      setActiveEventFrame(setButton.dataset.eventSet, currentFrame());
      return;
    }
    var clearButton = event.target.closest("[data-event-clear]");
    if (clearButton) {
      var clearCard = clearButton.closest(".failure-event-card");
      setActiveFailureEvent(Number(clearCard.dataset.eventIndex));
      setActiveEventFrame(clearButton.dataset.eventClear, null);
      return;
    }
    var card = event.target.closest(".failure-event-card");
    if (card) setActiveFailureEvent(Number(card.dataset.eventIndex));
  });
  byId("failureEvents").addEventListener("input", function (event) {
    if (!event.target.dataset.eventField) return;
    syncFailureEventInput(event.target);
    markDirty();
  });
  byId("failureEvents").addEventListener("focusin", function (event) {
    var card = event.target.closest(".failure-event-card");
    if (card) setActiveFailureEvent(Number(card.dataset.eventIndex));
  });
  byId("outcomeLabel").addEventListener("change", function () {
    if ((this.value === "failure" || this.value === "recovered_success") && !state.failureEvents.length) {
      addFailureEvent();
    }
  });
  byId("annotationForm").addEventListener("submit", saveAnnotation);
  byId("annotationForm").addEventListener("input", markDirty);
  byId("playButton").addEventListener("click", togglePlayback);
  byId("playOverlay").addEventListener("click", togglePlayback);
  byId("rolloutVideo").addEventListener("click", togglePlayback);
  byId("rolloutVideo").addEventListener("play", function () {
    byId("playButton").textContent = "Pause";
    byId("playOverlay").classList.add("hidden");
  });
  byId("rolloutVideo").addEventListener("pause", function () {
    byId("playButton").textContent = "Play";
    byId("playOverlay").classList.remove("hidden");
  });
  byId("rolloutVideo").addEventListener("timeupdate", function () {
    var record = selectedRollout();
    if (!record || byId("rolloutVideo").seeking) return;
    state.currentFrame = Math.min(
      Number(record.total_frames) - 1,
      Math.floor(byId("rolloutVideo").currentTime * Number(record.fps) + 0.0001)
    );
    byId("frameSlider").value = state.currentFrame;
    updateReadout();
  });
  byId("speedSelect").addEventListener("change", function () {
    byId("rolloutVideo").playbackRate = Number(byId("speedSelect").value);
  });
  byId("previousButton").addEventListener("click", function () { navigate(-1); });
  byId("nextButton").addEventListener("click", function () { navigate(1); });
  byId("instructionCondition").addEventListener("change", function () {
    state.instructionCondition = this.value || "full_instruction";
    state.baselineRuns = null;
    state.baselineRunsCondition = null;
    state.baselineRunsLoading = null;
    var batchCondition = byId("baselineBatchCondition");
    if (batchCondition) {
      batchCondition.value = state.instructionCondition;
      baselineBatchScopeChanged();
    }
    var record = selectedRollout();
    if (!record) return;
    renderInstructionVariantControl(record);
    loadEvaluation(record.id);
  });
  bindResultsEvents();
  byId("baselineBatchMethod").addEventListener("change", function () {
    updateBaselineBatchAdvancedFields();
    if (baselineBatchResultFilterValue() === "missing_valid") {
      baselineBatchCoverageChanged();
    } else {
      updateBaselineBatchSelection();
    }
  });
  byId("baselineBatchScope").addEventListener("change", function () {
    if (baselineBatchResultFilterValue() === "missing_valid") {
      baselineBatchCoverageChanged();
    } else {
      baselineBatchScopeChanged();
    }
  });
  byId("baselineBatchCondition").addEventListener("change", function () {
    if (baselineBatchResultFilterValue() === "missing_valid") {
      baselineBatchCoverageChanged();
    } else {
      baselineBatchScopeChanged();
    }
  });
  byId("baselineBatchResultFilter").addEventListener("change", baselineBatchCoverageChanged);
  byId("baselineBatchStartIndex").addEventListener("input", baselineBatchRangeChanged);
  byId("baselineBatchLimit").addEventListener("input", updateBaselineBatchSelection);
  byId("baselineBatchEndIndex").addEventListener("input", baselineBatchRangeChanged);
  byId("baselineBatchAddWorker").addEventListener("click", addBaselineBatchWorker);
  byId("baselineBatchWorkers").addEventListener("input", function () {
    readBaselineBatchWorkers();
    updateBaselineBatchSelection();
  });
  byId("baselineBatchWorkers").addEventListener("click", function (event) {
    var button = event.target.closest("[data-remove-worker]");
    if (!button) return;
    removeBaselineBatchWorker(Number(button.dataset.removeWorker));
  });
  byId("baselineBatchForm").addEventListener("submit", startBaselineBatch);
  ["rolloutGenerationTaskStart", "rolloutGenerationTaskEnd", "rolloutGenerationTrials", "rolloutGenerationRenderResolution", "rolloutGenerationRecordResolution"].forEach(function (id) {
    byId(id).addEventListener("input", updateRolloutGenerationSelection);
  });
  byId("rolloutGenerationVideoViewMode").addEventListener("change", updateRolloutGenerationSelection);
  byId("rolloutGenerationLogSafeFeatures").addEventListener("change", updateRolloutGenerationSelection);
  byId("rolloutGenerationSuite").addEventListener("change", function () {
    var isSpatial = byId("rolloutGenerationSuite").value === "libero_spatial";
    byId("rolloutGenerationRenderResolution").value = "256";
    byId("rolloutGenerationRecordResolution").value = isSpatial ? "256" : "224";
    updateRolloutGenerationSelection();
  });
  byId("rolloutGenerationForm").addEventListener("submit", startRolloutGeneration);
  document.addEventListener("click", function (event) {
    if (window.LF3RRunsJobs && typeof window.LF3RRunsJobs.handleClick === "function"
        && window.LF3RRunsJobs.handleClick(event)) {
      return;
    }
    var button = event.target.closest("[data-persistent-job-log]");
    if (!button) return;
    var output = null;
    document.querySelectorAll("[data-persistent-job-log-output]").forEach(function (node) {
      if (node.dataset.persistentJobLogOutput === button.dataset.persistentJobLog) output = node;
    });
    if (output && !output.hidden) {
      state.persistentJobLogOpen[button.dataset.persistentJobLog] = false;
      output.hidden = true;
      button.textContent = "View log";
      button.setAttribute("aria-expanded", "false");
      return;
    }
    loadPersistentJobLog(
      button.dataset.persistentJobLog,
      button.dataset.persistentJobType,
      button
    );
  });
  document.addEventListener("pointerover", function (event) {
    var target = cliHelpTarget(event);
    if (target) {
      if (cliHelpState.target !== target) showCliHelp(target);
    }
  });
  document.addEventListener("pointerout", function (event) {
    var target = cliHelpTarget(event);
    if (target && (!event.relatedTarget || !target.contains(event.relatedTarget))) scheduleHideCliHelp();
  });
  document.addEventListener("focusin", function (event) {
    var target = cliHelpTarget(event);
    if (target) showCliHelp(target);
  });
  document.addEventListener("focusout", function (event) {
    var target = cliHelpTarget(event);
    if (target && (!event.relatedTarget || !target.contains(event.relatedTarget))) scheduleHideCliHelp();
  });
  window.addEventListener("resize", function () {
    if (cliHelpState.target) positionCliHelp(cliHelpState.target);
  });
  loadCliHelpMetadata();
  updateBaselineBatchAdvancedFields();
  updateRolloutGenerationSelection();
  window.lf3rRenderJobCards = renderPersistentJobCards;
  window.lf3rGetPersistentJobs = function (jobType) {
    return (state.persistentJobs || []).filter(function (job) {
      return !jobType || job.job_type === jobType;
    });
  };
  window.lf3rRefreshPersistentJobs = loadPersistentJobs;
  window.setTimeout(loadPersistentJobs, 0);
  window.addEventListener("beforeunload", function (event) {
    if (state.dirty) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  document.addEventListener("keydown", function (event) {
    if (["annotate", "results"].indexOf(document.body.dataset.view) === -1) return;
    if (event.key === "/" && !isTypingTarget(event.target)) {
      event.preventDefault();
      byId("searchInput").focus();
      return;
    }
    if (isTypingTarget(event.target)) return;
    if (document.body.dataset.view === "results" && ["1", "2", "3", "s", "S"].indexOf(event.key) !== -1) return;
    if (event.code === "Space") {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      stepFrames(event.shiftKey ? -10 : -1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      stepFrames(event.shiftKey ? 10 : 1);
    } else if (event.key === "1") {
      setActiveEventFrame("causal_onset_frame", currentFrame());
    } else if (event.key === "2") {
      setActiveEventFrame("observable_onset_frame", currentFrame());
    } else if (event.key === "3") {
      setActiveEventFrame("terminal_failure_frame", currentFrame());
    } else if (event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveAnnotation();
    } else if (event.key.toLowerCase() === "n") {
      navigate(1);
    } else if (event.key.toLowerCase() === "p") {
      navigate(-1);
    }
  });
}
