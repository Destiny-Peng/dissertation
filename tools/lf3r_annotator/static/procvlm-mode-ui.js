"use strict";

(function installProcvlmInferenceMode() {
  var STORAGE_KEY = "lf3r.procvlm.inferenceMode";
  var VALID_MODES = { base: true, lora: true };

  function readMode() {
    try {
      var value = sessionStorage.getItem(STORAGE_KEY) || "base";
      return VALID_MODES[value] ? value : "base";
    } catch (_) {
      return "base";
    }
  }

  function writeMode(value) {
    if (!VALID_MODES[value]) return;
    try { sessionStorage.setItem(STORAGE_KEY, value); } catch (_) {}
  }

  function modeOptionsHtml(value) {
    return [
      '<option value="base"' + (value === "base" ? " selected" : "") + '>Base model</option>',
      '<option value="lora"' + (value === "lora" ? " selected" : "") + '>One-shot LoRA</option>'
    ].join("");
  }

  function updateBatchUi() {
    var select = document.getElementById("procvlmInferenceMode");
    if (!select) return;
    var mode = select.value;
    writeMode(mode);

    var form = document.getElementById("baselineBatchForm");
    if (!form) return;
    var methodSelect = document.getElementById("baselineBatchMethod");
    var isProcvlm = !methodSelect || methodSelect.value === "procvlm";
    var wrapper = select.closest("label");
    if (wrapper) wrapper.hidden = !isProcvlm;

    var modelInput = form.querySelector('[data-batch-option="model_path"]');
    if (modelInput) {
      var label = modelInput.closest("label");
      var caption = label && label.querySelector("span");
      if (caption) {
        caption.textContent = isProcvlm && mode === "lora"
          ? "LoRA checkpoint directory"
          : "Model path (optional)";
      }
      modelInput.placeholder = isProcvlm && mode === "lora"
        ? "Path to saved LoRA adapter directory"
        : "Use configured checkpoint";
    }

    var valueHead = document.getElementById("procvlmEnableValueHead");
    if (valueHead) {
      var valueHeadLabel = valueHead.closest("label");
      if (valueHeadLabel) valueHeadLabel.hidden = !isProcvlm || mode === "lora";
    }
  }

  function installBatchMode() {
    var form = document.getElementById("baselineBatchForm");
    if (!form || document.getElementById("procvlmInferenceMode")) return;
    var modelInput = form.querySelector('[data-batch-option="model_path"]');
    if (!modelInput) return;
    var modelLabel = modelInput.closest("label");
    if (!modelLabel || !modelLabel.parentNode) return;

    var wrapper = document.createElement("label");
    wrapper.dataset.batchMethods = "procvlm";
    wrapper.innerHTML = '<span>Inference mode</span><select id="procvlmInferenceMode">'
      + modeOptionsHtml(readMode()) + '</select>';
    modelLabel.parentNode.insertBefore(wrapper, modelLabel);
    wrapper.querySelector("select").addEventListener("change", updateBatchUi);
    var methodSelect = document.getElementById("baselineBatchMethod");
    if (methodSelect) methodSelect.addEventListener("change", updateBatchUi);
    updateBatchUi();
  }

  function updateSingleUi(select) {
    if (!select) return;
    var mode = select.value;
    writeMode(mode);
    var drawer = document.getElementById("singleBaselineDrawer");
    if (!drawer) return;

    var modelInput = drawer.querySelector('[data-option="model_path"]');
    if (modelInput) {
      var label = modelInput.closest("label");
      var caption = label && label.querySelector("span");
      if (caption) {
        caption.textContent = mode === "lora"
          ? "LoRA checkpoint directory"
          : "Model path";
      }
      modelInput.placeholder = mode === "lora"
        ? "Path to saved LoRA adapter directory"
        : "Use configured checkpoint";
    }

    var valueHead = drawer.querySelector('[data-option="procvlm_enable_value_head"]');
    if (valueHead) {
      var valueHeadLabel = valueHead.closest("label");
      if (valueHeadLabel) valueHeadLabel.hidden = mode === "lora";
    }
  }

  function installSingleMode() {
    var drawer = document.getElementById("singleBaselineDrawer");
    if (!drawer) return;
    var procvlmField = drawer.querySelector('[data-option="procvlm_window_size"]');
    if (!procvlmField) return;

    var core = drawer.querySelector("[data-core]");
    if (!core) return;
    var select = drawer.querySelector("[data-procvlm-inference-mode]");
    if (!select) {
      var label = document.createElement("label");
      label.innerHTML = '<span>Inference mode</span><select data-procvlm-inference-mode>'
        + modeOptionsHtml(readMode()) + '</select>';
      core.insertBefore(label, core.firstChild);
      select = label.querySelector("select");
      select.addEventListener("change", function () { updateSingleUi(select); });
    }
    updateSingleUi(select);
  }

  installBatchMode();

  var drawerObserver = new MutationObserver(function () {
    installSingleMode();
  });
  drawerObserver.observe(document.body, { childList: true, subtree: true });
  installSingleMode();

  var originalFetch = window.fetch.bind(window);
  window.fetch = function patchedProcvlmFetch(input, init) {
    try {
      var url = typeof input === "string" ? input : (input && input.url) || "";
      var isBaselineRequest = url.indexOf("/api/baselines/run/") === 0
        || url.indexOf("/api/baselines/run-batch") === 0;
      if (isBaselineRequest && init && typeof init.body === "string") {
        var payload = JSON.parse(init.body);
        if (payload && payload.baseline === "procvlm") {
          var isSingle = url.indexOf("/api/baselines/run/") === 0;
          var modeSelect = isSingle
            ? document.querySelector("#singleBaselineDrawer [data-procvlm-inference-mode]")
            : document.getElementById("procvlmInferenceMode");
          var mode = modeSelect && VALID_MODES[modeSelect.value]
            ? modeSelect.value
            : readMode();
          payload.options = payload.options || {};
          payload.options.procvlm_use_lora = mode === "lora";
          if (mode === "lora") {
            // The server maps explicit LoRA mode to the current persistent
            // worker's official PEFT/value-head loading path.
            delete payload.options.procvlm_enable_value_head;
          }
          init = Object.assign({}, init, { body: JSON.stringify(payload) });
        }
      }
    } catch (error) {
      console.error("Could not attach ProcVLM inference mode to request", error);
    }
    return originalFetch(input, init);
  };
})();
