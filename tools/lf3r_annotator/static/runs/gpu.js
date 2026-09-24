"use strict";

window.LF3RRunsGpu = (function createRunsGpuController() {
  function formatGiB(mib) {
      if (!Number.isFinite(Number(mib))) return "—";
      return (Number(mib) / 1024).toFixed(1) + " GiB";
    }
  
    async function refreshGpuStatus() {
      var button = document.getElementById("runsGpuRefresh");
      var cards = document.getElementById("runsGpuCards");
      var timestamp = document.getElementById("runsGpuTimestamp");
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

  function createStrip() {
    var gpuStrip = document.createElement("section");
    gpuStrip.className = "runs-gpu-strip";
    gpuStrip.innerHTML = [
    '<div class="runs-gpu-heading">',
    '<div><p class="eyebrow">GPU STATUS</p><strong>Current device state</strong><span id="runsGpuTimestamp">Not refreshed yet</span></div>',
    '<button id="runsGpuRefresh" class="ghost-button" type="button">Refresh GPU</button>',
    '</div>',
    '<div id="runsGpuCards" class="runs-gpu-cards"><div class="runs-gpu-empty">Press Refresh GPU to query nvidia-smi.</div></div>'
    ].join("");
    return gpuStrip;
  }

  function install() {
    var button = document.getElementById("runsGpuRefresh");
    if (!button || button.dataset.runsGpuInstalled === "true") return;
    button.dataset.runsGpuInstalled = "true";
    button.addEventListener("click", refreshGpuStatus);
  }

  return {
    createStrip: createStrip,
    install: install,
    refresh: refreshGpuStatus
  };
})();
