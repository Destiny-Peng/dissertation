"use strict";

(function installResultsRunClickBridge() {
  if (window.lf3rResultsRunClickBridgeInstalled) return;
  window.lf3rResultsRunClickBridgeInstalled = true;

  function status(message) {
    var node = document.getElementById("evaluationStatus");
    if (node) node.textContent = message;
  }

  function handle(event) {
    var target = event.target;
    var button = target && target.closest ? target.closest("[data-run-baseline]") : null;
    if (!button) return;

    var methodsHost = document.getElementById("evaluationMethods");
    if (!methodsHost || !methodsHost.contains(button)) return;

    // Capture at window level, before both the legacy evaluationMethods listener
    // and configurator document listeners. This gives the Results action one
    // deterministic entry point even when cards are re-rendered.
    event.preventDefault();
    event.stopImmediatePropagation();

    var method = String(button.dataset.runBaseline || "");
    var opener = window.lf3rOpenSingleBaselineConfig;
    if (typeof opener !== "function") {
      status("Baseline configurator did not load. Refresh this page after updating the WebUI assets.");
      console.error("LF3R Results baseline configurator is unavailable", { method: method });
      return;
    }

    try {
      var opened = opener(method);
      if (opened === false) {
        // V3 writes the detailed reason into evaluationStatus itself.
        console.error("LF3R Results baseline configurator refused to open", { method: method });
      }
    } catch (error) {
      status("Baseline configuration UI error: " + (error && error.message ? error.message : String(error)));
      console.error("LF3R Results baseline configurator failed", error);
    }
  }

  window.addEventListener("click", handle, true);
})();
