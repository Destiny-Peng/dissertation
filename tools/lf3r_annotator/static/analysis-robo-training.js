"use strict";

(function installUnifiedLocalizationTrainingUi() {
  function node(id) { return document.getElementById(id); }

  var experiment = node("analysisLocalizationExperiment");
  var successPanel = node("analysisSuccessNegativePanel");
  var labelLossPanel = node("analysisLabelLossPanel");
  if (!experiment || !successPanel || !labelLossPanel) return;

  var sharedPairs = [
    ["analysisBiLstmDevice", "analysisLabelLossDevice"],
    ["analysisBiLstmBatchSize", "analysisLabelLossBatchSize"],
    ["analysisBiLstmRepeats", "analysisLabelLossRepeats"],
    ["analysisBiLstmEpochs", "analysisLabelLossEpochs"],
    ["analysisBiLstmPatience", "analysisLabelLossPatience"],
    ["analysisBiLstmLearningRate", "analysisLabelLossLearningRate"],
    ["analysisBiLstmWeightDecay", "analysisLabelLossWeightDecay"],
    ["analysisBiLstmGradClip", "analysisLabelLossGradClip"]
  ];

  function mirror(sourceId, targetId) {
    var source = node(sourceId);
    var target = node(targetId);
    if (!source || !target) return;
    var copy = function () {
      target.value = source.value;
    };
    source.addEventListener("change", copy);
    source.addEventListener("input", copy);
  }

  sharedPairs.forEach(function (pair) {
    mirror(pair[0], pair[1]);
    mirror(pair[1], pair[0]);
  });

  function render() {
    var value = experiment.value;
    successPanel.classList.toggle("hidden", value !== "success_negative");
    labelLossPanel.classList.toggle("hidden", value !== "label_loss");
  }

  experiment.addEventListener("change", render);
  render();
})();
