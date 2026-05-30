/* Invalidation playground for docs/how-it-works/invalidation.md.
 *
 * Progressively enhances <div class="cash-invalidation-playground">. The div
 * ships with a static fallback table inside it; this script replaces that
 * content with an interactive 4-cell pipeline. Marking a "change" on any node
 * lights up that node and everything downstream as STALE (recompute); upstream
 * and unrelated nodes stay FRESH (cache hit). Vanilla JS, no dependencies.
 */
(function () {
  "use strict";

  // Linear pipeline: each cell depends on the previous one (dep = index or null).
  var CELLS = [
    { code: "raw = pd.read_csv('data.csv')", dep: null },
    { code: "clean = raw.dropna()",          dep: 0 },
    { code: "features = engineer(clean)",     dep: 1 },
    { code: "model = train(features)",        dep: 2 }
  ];

  var TRIGGERS = [
    { label: "Change data.csv", node: 0 },
    { label: "Edit raw",        node: 0 },
    { label: "Edit clean",      node: 1 },
    { label: "Edit features",   node: 2 }
  ];

  function staleSet(changed) {
    // A cell is stale if it is the changed node or depends (transitively) on a stale cell.
    var stale = {};
    if (changed === null) return stale;
    for (var i = 0; i < CELLS.length; i++) {
      if (i === changed) stale[i] = true;
      else if (CELLS[i].dep !== null && stale[CELLS[i].dep]) stale[i] = true;
    }
    return stale;
  }

  function build(root) {
    root.innerHTML = "";
    var current = null;

    var controls = document.createElement("div");
    controls.className = "cash-ip-controls";

    var chain = document.createElement("div");
    chain.className = "cash-ip-chain";

    var status = document.createElement("div");
    status.className = "cash-ip-status";
    status.setAttribute("aria-live", "polite");

    TRIGGERS.forEach(function (t, idx) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "cash-ip-trigger";
      btn.textContent = t.label;
      btn.addEventListener("click", function () {
        var key = t.node + ":" + idx;
        current = (current === key) ? null : key;
        Array.prototype.forEach.call(controls.children, function (b) {
          b.classList.remove("active");
        });
        if (current !== null) btn.classList.add("active");
        render(t.node, current === null ? null : t.node);
      });
      controls.appendChild(btn);
    });

    root.appendChild(controls);
    root.appendChild(chain);
    root.appendChild(status);

    function render(_clicked, changed) {
      var stale = staleSet(changed === undefined ? null : changed);
      chain.innerHTML = "";
      CELLS.forEach(function (cell, i) {
        var pill = document.createElement("div");
        pill.className = "cash-ip-cell " + (stale[i] ? "stale" : "fresh");
        pill.innerHTML =
          "<code>" + cell.code + "</code>" +
          '<span class="cash-ip-tag">' + (stale[i] ? "recompute" : "cache hit") + "</span>";
        chain.appendChild(pill);
        if (i < CELLS.length - 1) {
          var arrow = document.createElement("div");
          arrow.className = "cash-ip-arrow";
          arrow.textContent = "↓";
          chain.appendChild(arrow);
        }
      });
      var n = Object.keys(stale).length;
      status.textContent = (changed === null || changed === undefined)
        ? "Nothing changed — all 4 cells restore from cache."
        : n + " of 4 cells recompute; " + (4 - n) + " still hit the cache.";
    }

    render(null, null);
  }

  function init() {
    var roots = document.querySelectorAll(".cash-invalidation-playground");
    for (var i = 0; i < roots.length; i++) {
      if (roots[i].dataset.cashIpReady === "1") continue;
      roots[i].dataset.cashIpReady = "1";
      build(roots[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
