"use strict";

/* Results post-hoc actions. */

async function runPosthocLocalization(scope, button) {
  var record = selectedRollout();
  if (!record || !state.evaluation || !state.evaluation.methods) return;
  var result = state.evaluation.methods.robo_dopamine;
  if (!result || !result.available || !result.run || !result.run.run_root) return;
  var card = button.closest("[data-evaluation-method]");
  var input = card && card.querySelector("[data-posthoc-localization-ckpt]");
  var status = card && card.querySelector("[data-posthoc-localization-status]");
  var checkpoint = input ? input.value.trim() : "";
  if (!checkpoint) {
    if (status) status.textContent = "Enter a localization checkpoint first.";
    return;
  }
  state.roboPosthocCheckpoint = checkpoint;
  try {
    sessionStorage.setItem("lf3r.results.roboPosthocCheckpoint", checkpoint);
  } catch (_error) {}
  var original = button.textContent;
  button.disabled = true;
  button.textContent = scope === "all" ? "Applying…" : "Running…";
  if (status) {
    status.textContent = scope === "all"
      ? "Applying checkpoint to every saved fused rollout in this Robo-Dopamine run…"
      : "Running localization head on this saved fused result…";
  }
  try {
    var response = await fetch(
      "/api/baselines/posthoc-localization/" + encodeURIComponent(record.id),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_root: result.run.run_root,
          checkpoint: checkpoint,
          scope: scope,
          source_worker_result: (result.raw_files || []).find(function (path) {
            return /\/worker_result[.]json$/.test(String(path));
          }) || ""
        })
      }
    );
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Post-hoc localization failed");
    var data = payload.posthoc_localization || {};
    var completed = Array.isArray(data.results) ? data.results.length : 0;
    var failed = Array.isArray(data.errors) ? data.errors.length : 0;
    state.baselineRunNotice = "Post-hoc localization: " + completed
      + " result(s)" + (failed ? ", " + failed + " skipped/failed" : "");
    await loadEvaluation(record.id);
  } catch (error) {
    if (status) status.textContent = "Post-hoc localization error: " + error.message;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function selectPosthocLocalization(select) {
  if (!state.evaluation || !state.evaluation.methods) return;
  var result = state.evaluation.methods.robo_dopamine;
  if (!result || !Array.isArray(result.posthoc_localizations)) return;
  var index = Number(select.value);
  var chosen = result.posthoc_localizations[index];
  if (!chosen) return;
  result.localization_prediction = chosen;
  renderEvaluationPanel(state.evaluation);
}
