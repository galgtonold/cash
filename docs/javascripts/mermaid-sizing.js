/* Keep wide mermaid diagrams readable on narrow screens.
 *
 * Material renders each ```mermaid fence with `mermaid.render()` and puts the
 * SVG into a closed shadow root, where no stylesheet on the page can reach
 * it. The SVG comes out as `width="100%"` with an inline `max-width` of its
 * natural width, so a diagram wider than the column is scaled down to fit:
 * the 1,050px overview flowchart had 5px labels on a phone.
 *
 * This wraps `mermaid.render` as Material loads mermaid, and adds a
 * `min-width` of min(natural width, 560px) to each SVG it returns. A diagram
 * then shrinks at most to 560px, and on a narrower column its container
 * (`.md-typeset .mermaid`, `overflow-x: auto` in cash-design.css) scrolls
 * sideways. Small diagrams keep their natural size.
 *
 * Material loads mermaid only when `typeof mermaid == "undefined"`; the
 * accessor below still reads as undefined until the library assigns itself
 * (`globalThis.mermaid = ...`). If anything here fails, the diagram renders
 * exactly as it would without this file.
 */
(function () {
  "use strict";

  var MIN_WIDTH_PX = 560;

  function withMinWidth(svg) {
    return svg.replace(/^(\s*<svg\b[^>]*\bstyle=")([^"]*max-width:\s*([\d.]+)px;?)/, function (all, head, style, natural) {
      var min = Math.min(parseFloat(natural), MIN_WIDTH_PX);
      return head + style + (/;\s*$/.test(style) ? "" : ";") + " min-width: " + min + "px;";
    });
  }

  function wrap(lib) {
    if (!lib || typeof lib.render !== "function" || lib.__cashSized) return lib;
    var render = lib.render;
    lib.render = function () {
      return Promise.resolve(render.apply(this, arguments)).then(function (out) {
        try {
          if (out && typeof out.svg === "string") out.svg = withMinWidth(out.svg);
        } catch (e) { /* leave the diagram as mermaid drew it */ }
        return out;
      });
    };
    lib.__cashSized = true;
    return lib;
  }

  if (typeof window.mermaid !== "undefined") {
    wrap(window.mermaid);
    return;
  }
  var current;
  try {
    Object.defineProperty(window, "mermaid", {
      configurable: true,
      enumerable: true,
      get: function () { return current; },
      set: function (value) { current = wrap(value); }
    });
  } catch (e) { /* not definable: diagrams render unchanged */ }
})();
