/* Sort + filter for the cash comparison matrix.
 *
 * Looks for table.cash-matrix-table on the page, and an input with id
 * "cash-matrix-filter". Progressively enhances:
 *  - Click a column header → cycle ascending / descending / unsorted.
 *  - Type in the filter input → rows whose first cell does not contain
 *    the input text are hidden.
 *
 * No dependencies. Falls back gracefully: if either element is missing,
 * the page still works as a static table.
 */
(function () {
  "use strict";

  function rankCell(text) {
    // Sort order: ✅ best, ⚠️ middle, ❌ worst, anything else (e.g. capability name) → lex.
    var t = text.trim();
    if (t === "✅") return 0;
    if (t === "⚠️") return 1;
    if (t === "❌") return 2;
    return t.toLowerCase();
  }

  function compareCells(a, b) {
    var ra = rankCell(a), rb = rankCell(b);
    if (typeof ra === "number" && typeof rb === "number") return ra - rb;
    if (typeof ra === "number") return -1;
    if (typeof rb === "number") return 1;
    return ra < rb ? -1 : ra > rb ? 1 : 0;
  }

  function init() {
    var table = document.querySelector("table.cash-matrix-table");
    if (!table) return;
    if (table.dataset.cashMatrixReady === "1") return;
    table.dataset.cashMatrixReady = "1";
    var tbody = table.tBodies[0];
    if (!tbody) return;
    var headers = table.tHead ? table.tHead.rows[0].cells : [];
    var originalOrder = Array.prototype.slice.call(tbody.rows);

    function sortBy(colIdx, dir) {
      var sorted = originalOrder.slice().sort(function (r1, r2) {
        var c1 = r1.cells[colIdx] ? r1.cells[colIdx].textContent : "";
        var c2 = r2.cells[colIdx] ? r2.cells[colIdx].textContent : "";
        var cmp = compareCells(c1, c2);
        return dir === "descending" ? -cmp : cmp;
      });
      sorted.forEach(function (r) { tbody.appendChild(r); });
    }

    function resetSort() {
      originalOrder.forEach(function (r) { tbody.appendChild(r); });
    }

    function setSortAttr(active, dir) {
      for (var i = 0; i < headers.length; i++) {
        headers[i].removeAttribute("aria-sort");
      }
      if (active != null && dir) active.setAttribute("aria-sort", dir);
    }

    for (var i = 0; i < headers.length; i++) {
      (function (idx) {
        var h = headers[idx];
        h.setAttribute("tabindex", "0");
        h.setAttribute("role", "button");
        h.addEventListener("click", function () {
          var cur = h.getAttribute("aria-sort");
          var next = cur === "ascending" ? "descending"
                   : cur === "descending" ? null
                   : "ascending";
          if (next == null) { resetSort(); setSortAttr(null, null); }
          else { sortBy(idx, next); setSortAttr(h, next); }
        });
        h.addEventListener("keydown", function (e) {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); h.click(); }
        });
      })(i);
    }

    var filterInput = document.getElementById("cash-matrix-filter");
    if (filterInput) {
      filterInput.addEventListener("input", function () {
        var q = filterInput.value.trim().toLowerCase();
        for (var i = 0; i < tbody.rows.length; i++) {
          var row = tbody.rows[i];
          var first = row.cells[0] ? row.cells[0].textContent.toLowerCase() : "";
          if (!q || first.indexOf(q) >= 0) row.classList.remove("cash-matrix-row-hidden");
          else row.classList.add("cash-matrix-row-hidden");
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
