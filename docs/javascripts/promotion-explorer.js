/* Promotion explorer — page 7 (Storage & tiers).
 * Progressive enhancement: replaces the static fallback table inside
 * .cash-promotion-explorer with two sliders + a live promotion verdict.
 * Implements src/cash/backends/tiered_backend.py _default_promotion_policy:
 *   time < 1.0s            -> RAM only (too cheap)
 *   time > size / 100MB/s  -> persist to disk
 *   else                   -> RAM only (read-back costlier than recompute) */
(function () {
  "use strict";

  var DISK_MB_PER_S = 100; // 100 MB/s estimate, matches _disk_bandwidth_est

  function decide(timeS, sizeMB) {
    if (timeS < 1.0) {
      return {
        persist: false,
        label: "L1 only (RAM)",
        why: "Recomputing takes under 1 second — not worth a disk write."
      };
    }
    var readTime = sizeMB / DISK_MB_PER_S;
    if (timeS > readTime) {
      return {
        persist: true,
        label: "L1 + L2 (persisted to disk)",
        why: "Recompute (" + timeS.toFixed(1) + " s) costs more than reading " +
             sizeMB + " MB back from disk (~" + readTime.toFixed(1) + " s)."
      };
    }
    return {
      persist: false,
      label: "L1 only (RAM)",
      why: "Reading " + sizeMB + " MB back (~" + readTime.toFixed(1) +
           " s) would cost more than recomputing (" + timeS.toFixed(1) + " s)."
    };
  }

  function slider(min, max, step, value, label, unit) {
    var wrap = document.createElement("label");
    wrap.className = "cash-pe-field";
    var cap = document.createElement("span");
    cap.className = "cash-pe-cap";
    var out = document.createElement("output");
    out.textContent = value + " " + unit;
    cap.textContent = label + ": ";
    cap.appendChild(out);
    var input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    wrap.appendChild(cap);
    wrap.appendChild(input);
    return { wrap: wrap, input: input, out: out, unit: unit };
  }

  function build(root) {
    root.textContent = "";

    var time = slider(0, 10, 0.1, 3.0, "Compute time", "s");
    var size = slider(0, 1500, 10, 100, "Result size", "MB");

    var panel = document.createElement("div");
    panel.className = "cash-pe-panel";
    panel.setAttribute("aria-live", "polite");

    function update() {
      var t = parseFloat(time.input.value);
      var s = parseFloat(size.input.value);
      time.out.textContent = t.toFixed(1) + " " + time.unit;
      size.out.textContent = s + " " + size.unit;
      var d = decide(t, s);
      panel.className = "cash-pe-panel " + (d.persist ? "persist" : "ram");
      var pill = document.createElement("span");
      pill.className = "cash-pe-pill";
      pill.textContent = d.label;
      var why = document.createElement("p");
      why.className = "cash-pe-why";
      why.textContent = d.why;
      panel.textContent = "";
      panel.appendChild(pill);
      panel.appendChild(why);
    }

    time.input.addEventListener("input", update);
    size.input.addEventListener("input", update);

    var controls = document.createElement("div");
    controls.className = "cash-pe-controls";
    controls.appendChild(time.wrap);
    controls.appendChild(size.wrap);

    root.appendChild(controls);
    root.appendChild(panel);
    update();
  }

  function init() {
    var roots = document.querySelectorAll(".cash-promotion-explorer");
    for (var i = 0; i < roots.length; i++) {
      if (roots[i].dataset.cashPeReady === "1") continue;
      roots[i].dataset.cashPeReady = "1";
      build(roots[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
