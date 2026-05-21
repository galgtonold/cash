/* "Reclaim your time" calculator for docs/why-cash.md.
 *
 * Progressively enhances any <div class="cash-calculator"> on the page,
 * replacing its empty contents with three labelled inputs and a live
 * output line. Pure vanilla JS, no dependencies.
 *
 * Formula (documented inline on the page):
 *   hours_saved_per_week =
 *     (iterations_per_day × workdays × cold_run_minutes × HIT_RATE) / 60
 *   working_days_reclaimed_per_month =
 *     hours_saved_per_week × 4.33 / WORKDAY_HOURS
 *
 * HIT_RATE assumes 90% of iterations hit the cache after the first.
 */
(function () {
  "use strict";

  var HIT_RATE = 0.9;
  var WORKDAY_HOURS = 8;
  var WEEKS_PER_MONTH = 4.33;

  function buildCalculator(root) {
    root.innerHTML = [
      '<div class="cash-calculator-row">',
      '  <label for="cash-calc-mins">Pipeline cold-run minutes</label>',
      '  <input id="cash-calc-mins" type="number" min="0" step="1" value="12">',
      '</div>',
      '<div class="cash-calculator-row">',
      '  <label for="cash-calc-iters">Iterations per workday</label>',
      '  <input id="cash-calc-iters" type="number" min="0" step="1" value="8">',
      '</div>',
      '<div class="cash-calculator-row">',
      '  <label for="cash-calc-days">Workdays per week</label>',
      '  <input id="cash-calc-days" type="number" min="0" max="7" step="1" value="5">',
      '</div>',
      '<div class="cash-calculator-output" aria-live="polite">',
      '  <span data-cash-calc-output></span>',
      '</div>',
      '<div class="cash-calculator-caveat">',
      '  Estimate assumes a 90% cache hit rate on iterations and an 8-hour workday. Your mileage varies.',
      '</div>'
    ].join("\n");

    var mins  = root.querySelector("#cash-calc-mins");
    var iters = root.querySelector("#cash-calc-iters");
    var days  = root.querySelector("#cash-calc-days");
    var out   = root.querySelector("[data-cash-calc-output]");

    function recompute() {
      var m = Math.max(0, Number(mins.value)  || 0);
      var i = Math.max(0, Number(iters.value) || 0);
      var d = Math.max(0, Number(days.value)  || 0);
      var hoursPerWeek = (i * d * m * HIT_RATE) / 60;
      var daysPerMonth = (hoursPerWeek * WEEKS_PER_MONTH) / WORKDAY_HOURS;
      out.textContent =
        "≈ " + hoursPerWeek.toFixed(1) + " hours/week saved · " +
        "~" + daysPerMonth.toFixed(1) + " workdays/month reclaimed";
    }
    [mins, iters, days].forEach(function (el) {
      el.addEventListener("input", recompute);
    });
    recompute();
  }

  function init() {
    var roots = document.querySelectorAll(".cash-calculator");
    for (var i = 0; i < roots.length; i++) {
      if (roots[i].dataset.cashCalcReady === "1") continue;
      roots[i].dataset.cashCalcReady = "1";
      buildCalculator(roots[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
