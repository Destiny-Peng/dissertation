"use strict";

window.LF3RProjectTools = (function createProjectToolsController() {
  function install() {
    var view = document.getElementById("runsView");
    if (!view || view.dataset.projectToolsInstalled === "true") return;

    var client = window.LF3RProjectToolClient;
    var actionsModule = window.LF3RProjectToolActions;
    var manifestModule = window.LF3RManifestTools;
    if (!client || typeof client.install !== "function" || typeof client.submit !== "function") return;
    if (!actionsModule || typeof actionsModule.install !== "function") return;
    if (!manifestModule || typeof manifestModule.install !== "function") return;

    view.dataset.projectToolsInstalled = "true";

    var actions = actionsModule.install(client) || {};
    var manifest = manifestModule.install(client, {
      refreshRolloutOptions: actions.loadRolloutOptions
    }) || {};

    client.install({
      onJobUpdate: manifest.onJobUpdate,
      onJobsUpdated: manifest.onJobsUpdated
    });
  }

  return { install: install };
})();
