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

/* ----- mermaid diagram sizing -----
 *
 * mkdocs-mermaid2-plugin renders each SVG with `width="100%"` AND an inline
 * `style="max-width: <natural-px>"` matching the diagram's intrinsic size.
 * The combination means small diagrams (whose natural width is below the
 * column width) get stretched to fill the column anyway, blowing node font
 * sizes up to the point where labels are larger than section headings.
 *
 * CSS can't selectively override the `width="100%"` attribute, and SVG
 * elements don't honor `width: max-content` for sizing the way HTML does.
 * So we post-process each SVG after mermaid renders:
 *   - read the inline `max-width` (mermaid's measured natural width)
 *   - apply it as the SVG's `width` style, capped at 100% of column
 *
 * Result: small diagrams render at their natural size; large diagrams
 * still shrink to fit the column. */
(function () {
  "use strict";

  function fitMermaidSvgs() {
    var svgs = document.querySelectorAll(".mermaid svg");
    for (var i = 0; i < svgs.length; i += 1) {
      var svg = svgs[i];
      if (svg.dataset.cashFitted === "1") continue;
      var maxWidth = svg.style.maxWidth;  // mermaid's inline natural width
      if (!maxWidth) continue;
      svg.style.width = maxWidth;
      svg.removeAttribute("width");  // defeat width="100%" attribute
      svg.dataset.cashFitted = "1";
    }
  }

  // mermaid renders asynchronously; poll for a short window after load.
  function startFitting() {
    fitMermaidSvgs();
    var attempts = 0;
    var iv = setInterval(function () {
      fitMermaidSvgs();
      attempts += 1;
      if (attempts >= 20) clearInterval(iv);  // 20 * 200 ms = 4 s window
    }, 200);
  }

  if (document.readyState === "complete" || document.readyState === "interactive") {
    startFitting();
  } else {
    document.addEventListener("DOMContentLoaded", startFitting);
  }
})();
