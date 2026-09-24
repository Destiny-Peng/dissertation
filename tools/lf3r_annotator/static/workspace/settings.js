"use strict";

/* Workspace settings lifecycle. */

function workspaceNormalizeLoadedSettings(settings) {
  if (!settings) return Object.assign({}, SETTINGS_DEFAULTS);
  var usesLegacyDefaults = Object.keys(SETTINGS_LEGACY_DEFAULT_PALETTE).every(function (key) {
    return String(settings[key] || "").toLowerCase() === SETTINGS_LEGACY_DEFAULT_PALETTE[key];
  });
  return usesLegacyDefaults
    ? Object.assign({}, settings, {
        background_color: SETTINGS_DEFAULTS.background_color,
        surface_color: SETTINGS_DEFAULTS.surface_color,
        surface_raised_color: SETTINGS_DEFAULTS.surface_raised_color,
        control_color: SETTINGS_DEFAULTS.control_color,
        text_color: SETTINGS_DEFAULTS.text_color,
        muted_color: SETTINGS_DEFAULTS.muted_color,
        accent_color: SETTINGS_DEFAULTS.accent_color
      })
    : settings;
}

function workspaceApplySettings(settings) {
  var root = document.documentElement;
  var mapping = {
    background_color: "--bg",
    surface_color: "--surface",
    surface_raised_color: "--surface-raised",
    control_color: "--control",
    text_color: "--text",
    muted_color: "--muted",
    accent_color: "--accent"
  };
  Object.keys(mapping).forEach(function (field) {
    if (settings && settings[field]) root.style.setProperty(mapping[field], settings[field]);
  });
  root.style.setProperty("--font-scale", String(settings && settings.font_scale || 1));
  root.style.setProperty("--review-font-scale", String(settings && settings.review_font_scale || 1));
  root.style.setProperty("--analysis-font-scale", String(settings && settings.analysis_font_scale || 1));
  root.style.setProperty("--control-font-scale", String(settings && settings.control_font_scale || 1));
  document.body.dataset.density = settings && settings.density || "comfortable";
}

function workspaceSettingsFromForm() {
  var fields = ["background_color", "surface_color", "surface_raised_color", "control_color", "text_color", "muted_color", "accent_color"];
  var settings = {};
  fields.forEach(function (field) {
    var input = document.querySelector('[data-settings-field="' + field + '"]');
    settings[field] = input ? input.value : SETTINGS_DEFAULTS[field];
  });
  var scale = byId("settingsFontScale");
  var reviewScale = byId("settingsReviewFontScale");
  var analysisScale = byId("settingsAnalysisFontScale");
  var controlScale = byId("settingsControlFontScale");
  var density = byId("settingsDensity");
  settings.font_scale = scale ? Number(scale.value) : SETTINGS_DEFAULTS.font_scale;
  settings.review_font_scale = reviewScale ? Number(reviewScale.value) : SETTINGS_DEFAULTS.review_font_scale;
  settings.analysis_font_scale = analysisScale ? Number(analysisScale.value) : SETTINGS_DEFAULTS.analysis_font_scale;
  settings.control_font_scale = controlScale ? Number(controlScale.value) : SETTINGS_DEFAULTS.control_font_scale;
  settings.density = density ? density.value : SETTINGS_DEFAULTS.density;
  return settings;
}

function workspaceUpdateColorOutputs(settings) {
  document.querySelectorAll("[data-settings-value]").forEach(function (output) {
    var field = output.dataset.settingsValue;
    output.value = settings[field] || "";
    output.textContent = settings[field] || "";
  });
  [
    ["settingsFontScale", "settingsFontScaleValue"],
    ["settingsReviewFontScale", "settingsReviewFontScaleValue"],
    ["settingsAnalysisFontScale", "settingsAnalysisFontScaleValue"],
    ["settingsControlFontScale", "settingsControlFontScaleValue"]
  ].forEach(function (pair) {
    var scale = byId(pair[0]);
    var output = byId(pair[1]);
    if (scale && output) output.textContent = Math.round(Number(scale.value) * 100) + "%";
  });
}

function workspacePopulateSettings(settings) {
  var fields = ["background_color", "surface_color", "surface_raised_color", "control_color", "text_color", "muted_color", "accent_color"];
  fields.forEach(function (field) {
    var input = document.querySelector('[data-settings-field="' + field + '"]');
    if (input) input.value = settings[field];
  });
  [
    ["settingsFontScale", "font_scale"],
    ["settingsReviewFontScale", "review_font_scale"],
    ["settingsAnalysisFontScale", "analysis_font_scale"],
    ["settingsControlFontScale", "control_font_scale"]
  ].forEach(function (pair) {
    var input = byId(pair[0]);
    if (input) input.value = settings[pair[1]] == null ? SETTINGS_DEFAULTS[pair[1]] : settings[pair[1]];
  });
  if (byId("settingsDensity")) byId("settingsDensity").value = settings.density;
  workspaceUpdateColorOutputs(settings);
}

function workspaceSettingsStatus(message, kind) {
  var status = byId("settingsStatus");
  status.textContent = message;
  status.className = "analysis-status" + (kind ? " " + kind : "");
}

async function workspaceLoadSettings() {
  if (workspaceState.settingsLoading || workspaceState.settingsLoaded) return;
  workspaceState.settingsLoading = true;
  try {
    var response = await fetch("/api/settings", { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load shared settings");
    workspaceState.settings = workspaceNormalizeLoadedSettings(
      payload.settings || Object.assign({}, SETTINGS_DEFAULTS)
    );
    workspaceState.settingsDraft = Object.assign({}, workspaceState.settings);
    workspaceState.settingsLoaded = true;
    workspaceState.settingsDirty = false;
    workspaceApplySettings(workspaceState.settings);
    workspacePopulateSettings(workspaceState.settings);
    workspaceSettingsStatus(payload.updated_at ? "Shared settings loaded · updated " + payload.updated_at : "Using default shared settings; save to create the project config.", "");
  } catch (error) {
    workspaceState.settings = Object.assign({}, SETTINGS_DEFAULTS);
    workspaceState.settingsDraft = Object.assign({}, SETTINGS_DEFAULTS);
    workspaceState.settingsLoaded = true;
    workspaceApplySettings(workspaceState.settings);
    workspacePopulateSettings(workspaceState.settings);
    workspaceSettingsStatus("Settings API unavailable; previewing defaults. " + error.message, "warning");
  } finally {
    workspaceState.settingsLoading = false;
  }
}

async function workspaceSaveSettings(event) {
  if (event) event.preventDefault();
  var settings = workspaceSettingsFromForm();
  workspaceApplySettings(settings);
  var button = byId("settingsSave");
  button.disabled = true;
  workspaceSettingsStatus("Saving shared settings…", "");
  try {
    var response = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings)
    });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not save shared settings");
    workspaceState.settings = payload.settings;
    workspaceState.settingsDraft = Object.assign({}, payload.settings);
    workspaceState.settingsDirty = false;
    workspaceApplySettings(payload.settings);
    workspacePopulateSettings(payload.settings);
    workspaceSettingsStatus("Shared settings saved · updated " + (payload.updated_at || "now"), "");
  } catch (error) {
    workspaceState.settingsDirty = true;
    workspaceSettingsStatus("Settings save failed: " + error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function workspaceHandleSettingsInput() {
  var settings = workspaceSettingsFromForm();
  workspaceState.settingsDraft = settings;
  workspaceState.settingsDirty = true;
  workspaceApplySettings(settings);
  workspaceUpdateColorOutputs(settings);
  workspaceSettingsStatus("Previewing unsaved shared settings; save to publish them.", "warning");
}

function workspaceHandlePreset(name) {
  if (!SETTINGS_PRESETS[name]) return;
  workspaceState.settingsDraft = Object.assign({}, SETTINGS_PRESETS[name]);
  workspacePopulateSettings(workspaceState.settingsDraft);
  workspaceApplySettings(workspaceState.settingsDraft);
  workspaceState.settingsDirty = true;
  workspaceSettingsStatus("Previewing the " + name + " preset; save to publish it.", "warning");
}

function workspaceResetSettings() {
  workspaceState.settingsDraft = Object.assign({}, SETTINGS_DEFAULTS);
  workspacePopulateSettings(workspaceState.settingsDraft);
  workspaceApplySettings(workspaceState.settingsDraft);
  workspaceState.settingsDirty = true;
  workspaceSettingsStatus("Previewing defaults; save to publish them.", "warning");
}
