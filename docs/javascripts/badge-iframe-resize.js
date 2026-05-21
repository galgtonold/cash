/* Auto-size Cash badge iframes to their content height.
 *
 * Two paths cooperate so neither side has to win the race:
 *
 * 1. The iframe's wrapper document (built by scripts/build_badge_examples.py)
 *    posts a `{ type: "cash-badge-resize", height: N }` message to the parent
 *    on load, on every <details> toggle, and on any layout change.
 *
 * 2. As a safety net for the initial load (when the iframe's load event may
 *    fire before this script registers its listener), the parent also polls
 *    every iframe.cash-badge on its own load and on each iframe's load event,
 *    reading the height directly from the iframe's contentDocument.
 */
(function () {
  "use strict";

  function setHeight(iframe, h) {
    if (typeof h !== "number" || h <= 0) return;
    iframe.style.height = (h + 2) + "px";
  }

  function measureFromContent(iframe) {
    try {
      var doc = iframe.contentDocument;
      if (!doc || doc.readyState !== "complete") return 0;
      return Math.ceil(doc.documentElement.getBoundingClientRect().height);
    } catch (e) {
      return 0;  // cross-origin or not loaded yet
    }
  }

  function pollAll() {
    var frames = document.querySelectorAll("iframe.cash-badge");
    for (var i = 0; i < frames.length; i += 1) {
      (function (iframe) {
        // Measure immediately if already loaded.
        setHeight(iframe, measureFromContent(iframe));
        // And again whenever the iframe (re)loads.
        iframe.addEventListener("load", function () {
          setHeight(iframe, measureFromContent(iframe));
        });
      })(frames[i]);
    }
  }

  // Listener for messages from the iframe-side script.
  window.addEventListener("message", function (event) {
    var data = event.data;
    if (!data || data.type !== "cash-badge-resize") return;
    var frames = document.querySelectorAll("iframe.cash-badge");
    for (var i = 0; i < frames.length; i += 1) {
      if (frames[i].contentWindow === event.source) {
        setHeight(frames[i], data.height);
        return;
      }
    }
  });

  if (document.readyState === "complete" || document.readyState === "interactive") {
    pollAll();
  } else {
    document.addEventListener("DOMContentLoaded", pollAll);
  }
})();
