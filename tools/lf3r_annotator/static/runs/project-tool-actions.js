"use strict";

window.LF3RProjectToolActions = (function createProjectToolActions() {
  function install(client) {
    function text(id) {
      var node = document.getElementById(id);
      return node ? String(node.value || "").trim() : "";
    }

    function numberValue(id, fallback) {
      var value = Number(text(id));
      return Number.isFinite(value) ? value : fallback;
    }

    var prepareStamp = new Date().toISOString()
      .replace(/[-:]/g, "")
      .replace(/\..*/, "")
      .replace("T", "_");
    document.getElementById("safePrepareOutput").value =
      "outputs/safe_training/datasets/web_" + prepareStamp;
    document.getElementById("safeTrainDataset").value =
      document.getElementById("safePrepareOutput").value;
    document.getElementById("safeValidateDataset").value =
      document.getElementById("safePrepareOutput").value;

    document.getElementById("safePrepareRun").addEventListener("click", function () {
      var output = text("safePrepareOutput");
      document.getElementById("safeTrainDataset").value = output;
      document.getElementById("safeValidateDataset").value = output;
      client.submit("safe_prepare", {
        dataset_role: text("safePrepareRole"),
        partition: text("safePreparePartition"),
        run_name: text("safePrepareRunName"),
        output: output
      }).catch(function () {});
    });

    document.getElementById("safeTrainRun").addEventListener("click", function () {
      client.submit("safe_train", {
        dataset_dir: text("safeTrainDataset"),
        model: text("safeTrainModel"),
        gpu: text("safeTrainGpu"),
        epochs: numberValue("safeTrainEpochs", 1000),
        batch_size: numberValue("safeTrainBatch", 512),
        hidden_dim: numberValue("safeTrainHidden", 256),
        seed: text("safeTrainSeed"),
        logs_root: text("safeTrainLogs"),
        normalize: document.getElementById("safeTrainNormalize").checked
      }).catch(function () {});
    });

    document.getElementById("safeValidateRun").addEventListener("click", function () {
      client.submit("safe_validate", {
        dataset_dir: text("safeValidateDataset"),
        checkpoint: text("safeValidateCheckpoint"),
        model: text("safeValidateModel"),
        gpu: text("safeValidateGpu"),
        output: text("safeValidateOutput"),
        hidden_dim: numberValue("safeTrainHidden", 256)
      }).catch(function () {});
    });

    function exportOptions(dryRun) {
      var outcomes = Array.prototype.slice
        .call(document.querySelectorAll("#exportOutcomes input:checked"))
        .map(function (node) { return node.value; });
      return {
        outcomes: outcomes,
        dataset_role: text("exportDatasetRole"),
        review_status: text("exportReviewStatus"),
        output_dir: text("exportOutputDir"),
        dry_run: !!dryRun
      };
    }

    document.getElementById("exportDryRun").addEventListener("click", function () {
      client.submit("export_cases", exportOptions(true)).catch(function () {});
    });
    document.getElementById("exportRun").addEventListener("click", function () {
      client.submit("export_cases", exportOptions(false)).catch(function () {});
    });

    document.getElementById("roboSweepRun").addEventListener("click", function () {
      var intervals = text("roboSweepIntervals")
        .split(",")
        .map(function (item) { return Number(item.trim()); })
        .filter(function (item) { return Number.isInteger(item) && item > 0; });
      client.submit("robo_interval_sweep", {
        rollout_ids: [text("roboSweepRollout")],
        intervals: intervals,
        gpu: text("roboSweepGpu"),
        memory_utilization: numberValue("roboSweepMemory", 0.6),
        output_dir: text("roboSweepOutput")
      }).catch(function () {});
    });

    document.getElementById("validateBaselinesRun").addEventListener("click", function () {
      client.submit("validate_baselines", { check_environments: true }).catch(function () {});
    });
    document.getElementById("validateVariantsRun").addEventListener("click", function () {
      client.submit("validate_variants", {}).catch(function () {});
    });

    function renderRolloutOptions(rows) {
      var select = document.getElementById("roboSweepRollout");
      if (!select) return;
      select.innerHTML = (rows || []).map(function (row) {
        var label = (row.task_suite || "")
          + " · task " + row.task_id
          + " · ep " + row.episode_index
          + " · " + row.id;
        return '<option value="' + row.id + '">' + label + '</option>';
      }).join("");
    }

    async function loadRolloutOptions() {
      var select = document.getElementById("roboSweepRollout");
      var cached = (typeof state !== "undefined" && state && Array.isArray(state.rollouts))
        ? state.rollouts
        : [];
      if (cached.length) {
        renderRolloutOptions(cached);
        return;
      }
      try {
        var response = await fetch("/api/rollouts", { cache: "no-store" });
        var payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Could not load rollouts");
        renderRolloutOptions(payload.rollouts || []);
      } catch (error) {
        if (select) {
          select.innerHTML = '<option value="">'
            + String(error.message || error)
            + '</option>';
        }
      }
    }

    loadRolloutOptions();
    return { loadRolloutOptions: loadRolloutOptions };
  }

  return { install: install };
})();
