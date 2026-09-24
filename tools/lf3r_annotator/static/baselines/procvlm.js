"use strict";

window.LF3RProcvlmMode = (function createProcvlmModeController() {
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

  function setText(node, value) {
    if (node && node.textContent !== value) node.textContent = value;
  }

  function setPlaceholder(node, value) {
    if (node && node.placeholder !== value) node.placeholder = value;
  }

  function setHidden(node, value) {
    if (node && node.hidden !== value) node.hidden = value;
  }

  function selectedMode(context) {
    var select = context === "single"
      ? document.querySelector("#singleBaselineDrawer [data-procvlm-inference-mode]")
      : document.getElementById("procvlmInferenceMode");
    return select && VALID_MODES[select.value] ? select.value : readMode();
  }

  function applyOptions(options, context) {
    var output = Object.assign({}, options || {});
    var mode = selectedMode(context);
    output.procvlm_use_lora = mode === "lora";
    if (mode === "lora") delete output.procvlm_enable_value_head;
    return output;
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
    setHidden(wrapper, !isProcvlm);

    var modelInput = form.querySelector('[data-batch-option="model_path"]');
    if (modelInput) {
      var label = modelInput.closest("label");
      var caption = label && label.querySelector("span");
      setText(
        caption,
        isProcvlm && mode === "lora" ? "LoRA checkpoint directory" : "Model path (optional)"
      );
      setPlaceholder(
        modelInput,
        isProcvlm && mode === "lora"
          ? "Path to saved LoRA adapter directory"
          : "Use configured checkpoint"
      );
    }

    var valueHead = document.getElementById("procvlmEnableValueHead");
    if (valueHead) {
      var valueHeadLabel = valueHead.closest("label");
      setHidden(valueHeadLabel, !isProcvlm || mode === "lora");
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
      setText(caption, mode === "lora" ? "LoRA checkpoint directory" : "Model path");
      setPlaceholder(
        modelInput,
        mode === "lora" ? "Path to saved LoRA adapter directory" : "Use configured checkpoint"
      );
    }

    var valueHead = drawer.querySelector('[data-option="procvlm_enable_value_head"]');
    if (valueHead) {
      var valueHeadLabel = valueHead.closest("label");
      setHidden(valueHeadLabel, mode === "lora");
    }
  }

  function refreshSingle(method) {
    if (method !== "procvlm") return;
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

    var modelInput = drawer.querySelector('[data-option="model_path"]');
    var modelLabel = modelInput && modelInput.closest("label");
    var modeLabel = select.closest("label");
    if (modelLabel && modelLabel.parentNode !== core) {
      if (modeLabel && modeLabel.nextSibling) {
        core.insertBefore(modelLabel, modeLabel.nextSibling);
      } else {
        core.appendChild(modelLabel);
      }
    }

    updateSingleUi(select);
  }

  installBatchMode();

  return {
    readMode: readMode,
    applyOptions: applyOptions,
    refreshBatch: updateBatchUi,
    refreshSingle: refreshSingle
  };
})();
