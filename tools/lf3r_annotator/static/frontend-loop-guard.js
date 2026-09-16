"use strict";

(function installFrontendLoopGuard() {
  if (window.__lf3rFrontendLoopGuardInstalled) return;
  window.__lf3rFrontendLoopGuardInstalled = true;

  function wrapObserver(name, NativeObserver, limit, windowMs) {
    if (typeof NativeObserver !== "function") return NativeObserver;

    function GuardedObserver(callback) {
      var windowStart = performance.now();
      var callbackCount = 0;
      var tripped = false;

      var observer = new NativeObserver(function (entries, nativeObserver) {
        var now = performance.now();
        if (now - windowStart >= windowMs) {
          windowStart = now;
          callbackCount = 0;
        }
        callbackCount += 1;

        if (callbackCount > limit) {
          if (!tripped) {
            tripped = true;
            nativeObserver.disconnect();
            var report = {
              observer: name,
              callbacks: callbackCount,
              window_ms: windowMs,
              at: new Date().toISOString()
            };
            window.__lf3rFrontendLoopGuardTrip = report;
            console.error(
              "LF3R frontend safety disconnected a runaway " + name + ".",
              report
            );
          }
          return;
        }

        callback(entries, nativeObserver);
      });
      return observer;
    }

    GuardedObserver.prototype = NativeObserver.prototype;
    return GuardedObserver;
  }

  window.MutationObserver = wrapObserver(
    "MutationObserver",
    window.MutationObserver,
    200,
    1000
  );

  if (typeof window.ResizeObserver === "function") {
    window.ResizeObserver = wrapObserver(
      "ResizeObserver",
      window.ResizeObserver,
      120,
      1000
    );
  }
})();
