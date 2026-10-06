"use strict";

(function installStandaloneTactileLink() {
  var navigation = document.querySelector(".page-navigation");
  if (!navigation || navigation.querySelector("[data-tactile-standalone]")) return;

  var link = document.createElement("a");
  link.href = "/static/tactile/index.html";
  link.textContent = "Tactile";
  link.setAttribute("data-tactile-standalone", "true");
  link.title = "Open the standalone tactile review UI";
  navigation.appendChild(link);
})();
