"use strict";

window.LF3RRunsGpu = (function createRunsGpuController() {
  function formatGiB(mib) {
    if (!Number.isFinite(Number(mib))) return "—";
    return (Number(mib) / 1024).toFixed(1) + " GiB";
  }

  function normalizePrefix(prefix) {
    var value = String(prefix || "runsGpu").trim();
    return value || "runsGpu";
  }

  function ids(prefix) {
    var base = normalizePrefix(prefix);
    return {
      refresh: base + "Refresh",
      cards: base + "Cards",
      timestamp: base + "Timestamp"
    };
  }

  async function refreshGpuStatus(prefix) {
    var key = normalizePrefix(prefix);
    var names = ids(key);
    var button = document.getElementById(names.refresh);
    var cards = document.getElementById(names.cards);
    var timestamp = document.getElementById(names.timestamp);
    if (!button || !cards || !timestamp) return;

    button.disabled = true;
    button.textContent = "Refreshing…";
    try {
      var response = await fetch("/api/gpu-status", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read GPU status");
      var status = payload.gpu_status || {};
      if (!status.available) throw new Error(status.error || "GPU status unavailable");
      var gpus = status.gpus || [];
      cards.innerHTML = gpus.length ? gpus.map(function (gpu) {
        var freeFraction = Number(gpu.memory_free_fraction);
        var freePercent = Number.isFinite(freeFraction) ? Math.round(freeFraction * 100) : 0;
        return [
          '<article class="runs-gpu-card">',
            '<div class="runs-gpu-card-top"><strong>GPU ' + gpu.index + '</strong><span>' + (gpu.gpu_utilization_percent == null ? '—' : gpu.gpu_utilization_percent + '% util') + '</span></div>',
            '<div class="runs-gpu-name">' + String(gpu.name || "NVIDIA GPU") + '</div>',
            '<div class="runs-gpu-memory"><span><b>' + formatGiB(gpu.memory_free_mib) + '</b> free / ' + formatGiB(gpu.memory_total_mib) + '</span><span>' + freePercent + '% free</span></div>',
            '<div class="runs-gpu-meter"><i style="width:' + Math.max(0, Math.min(100, freePercent)) + '%"></i></div>',
            '<div class="runs-gpu-meta"><span>' + (gpu.temperature_c == null ? '—' : gpu.temperature_c + '°C') + '</span><span>' + (gpu.power_draw_w == null ? '—' : gpu.power_draw_w.toFixed(0) + ' W') + '</span></div>',
          '</article>'
        ].join("");
      }).join("") : '<div class="runs-gpu-empty">No NVIDIA GPUs reported.</div>';
      timestamp.textContent = "Refreshed " + new Date(status.queried_at || Date.now()).toLocaleTimeString();
    } catch (error) {
      cards.innerHTML = '<div class="runs-gpu-empty error">' + String(error.message || error) + '</div>';
      timestamp.textContent = "Refresh failed";
    } finally {
      button.disabled = false;
      button.textContent = "Refresh GPU";
    }
  }

  function createStrip(options) {
    var config = options || {};
    var prefix = normalizePrefix(config.prefix);
    var names = ids(prefix);
    var gpuStrip = document.createElement("section");
    gpuStrip.className = "runs-gpu-strip";
    gpuStrip.dataset.gpuStripPrefix = prefix;
    gpuStrip.innerHTML = [
      '<div class="runs-gpu-heading">',
      '<div><p class="eyebrow">' + String(config.eyebrow || "GPU STATUS") + '</p><strong>' + String(config.title || "Current device state") + '</strong><span id="' + names.timestamp + '">Not refreshed yet</span></div>',
      '<button id="' + names.refresh + '" class="ghost-button" type="button">Refresh GPU</button>',
      '</div>',
      '<div id="' + names.cards + '" class="runs-gpu-cards"><div class="runs-gpu-empty">Press Refresh GPU to query nvidia-smi.</div></div>'
    ].join("");
    return gpuStrip;
  }

  function install(prefix) {
    var key = normalizePrefix(prefix);
    var names = ids(key);
    var button = document.getElementById(names.refresh);
    if (!button || button.dataset.runsGpuInstalled === "true") return false;
    button.dataset.runsGpuInstalled = "true";
    button.addEventListener("click", function () { refreshGpuStatus(key); });
    return true;
  }

  return {
    createStrip: createStrip,
    install: install,
    refresh: refreshGpuStatus
  };
})();