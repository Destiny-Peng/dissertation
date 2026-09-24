"use strict";

/* Parameter-help popover state and rendering. */

var cliHelpState = {
  entries: null,
  popover: null,
  target: null,
  hideTimer: null
};

function cliHelpEntry(key) {
  if (!cliHelpState.entries) return null;
  var parts = String(key || "").split(".");
  var value = cliHelpState.entries;
  parts.forEach(function (part) {
    if (value) value = value[part];
  });
  var entry = value && typeof value === "object" ? value : null;
  if (window.LF3RDatasetScopes
      && typeof window.LF3RDatasetScopes.decorateHelp === "function") {
    return window.LF3RDatasetScopes.decorateHelp(key, entry);
  }
  return entry;
}

function loadCliHelpMetadata() {
  fetch("/static/parameter_help.json", { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) throw new Error("parameter help unavailable");
      return response.json();
    })
    .then(function (payload) {
      cliHelpState.entries = payload || {};
      if (cliHelpState.target) showCliHelp(cliHelpState.target);
    })
    .catch(function () {
      cliHelpState.entries = {};
    });
}

function ensureCliHelpPopover() {
  if (cliHelpState.popover) return cliHelpState.popover;
  var popover = document.createElement("div");
  popover.id = "cliHelpPopover";
  popover.className = "cli-help-popover";
  popover.setAttribute("role", "tooltip");
  popover.hidden = true;
  document.body.appendChild(popover);
  cliHelpState.popover = popover;
  return popover;
}

function positionCliHelp(target) {
  var popover = cliHelpState.popover;
  if (!popover || !target || popover.hidden) return;
  var rect = target.getBoundingClientRect();
  var margin = 8;
  var left = Math.max(margin, Math.min(rect.left, window.innerWidth - popover.offsetWidth - margin));
  var top = rect.bottom + margin;
  if (top + popover.offsetHeight > window.innerHeight - margin) {
    top = rect.top - popover.offsetHeight - margin;
  }
  popover.style.left = Math.round(Math.max(margin, left)) + "px";
  popover.style.top = Math.round(Math.max(margin, top)) + "px";
}

function showCliHelp(target) {
  if (!target || !target.dataset.cliHelp) return;
  if (cliHelpState.hideTimer) {
    window.clearTimeout(cliHelpState.hideTimer);
    cliHelpState.hideTimer = null;
  }
  var popover = ensureCliHelpPopover();
  var key = target.dataset.cliHelp;
  var entry = cliHelpEntry(key);
  var lines = entry ? [
    "CLI: " + (entry.cli || "shared setting"),
    "Default: " + (entry.default == null ? "unset" : entry.default),
    "Applies to: " + (entry.applies_to || "web UI"),
    entry.forwarded_as ? "Forwarded as: " + entry.forwarded_as : "",
    entry.aliases && entry.aliases.length ? "Aliases: " + entry.aliases.join(", ") : "",
    entry.description || ""
  ] : ["Parameter help: " + key, "Loading CLI metadata..."];
  popover.textContent = lines.filter(Boolean).join("\n");
  popover.hidden = false;
  cliHelpState.target = target;
  target.classList.add("cli-help-target");
  target.setAttribute("aria-describedby", "cliHelpPopover");
  positionCliHelp(target);
}

function hideCliHelp() {
  if (cliHelpState.hideTimer) {
    window.clearTimeout(cliHelpState.hideTimer);
    cliHelpState.hideTimer = null;
  }
  if (cliHelpState.target) {
    cliHelpState.target.classList.remove("cli-help-target");
    if (cliHelpState.target.getAttribute("aria-describedby") === "cliHelpPopover") {
      cliHelpState.target.removeAttribute("aria-describedby");
    }
  }
  if (cliHelpState.popover) cliHelpState.popover.hidden = true;
  cliHelpState.target = null;
}

function scheduleHideCliHelp() {
  if (cliHelpState.hideTimer) window.clearTimeout(cliHelpState.hideTimer);
  cliHelpState.hideTimer = window.setTimeout(hideCliHelp, 80);
}

function cliHelpTarget(event) {
  var node = event && event.target;
  return node && node.closest ? node.closest("[data-cli-help]") : null;
}
