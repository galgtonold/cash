/* "Reclaim your time" calculator for docs/why-cash.md.
 *
 * Progressively enhances any <div class="cash-calculator"> on the page,
 * replacing its empty contents with three sliders and a live output line.
 * Pure vanilla JS, no dependencies.
 *
 * Sliders rather than number inputs: this is a "what if" toy, and the point is
 * to drag a value and watch the answer move. Spinner arrows are a poor fit for
 * that — they are tiny hit targets, awful on touch, and they invite typing an
 * exact figure into what is explicitly an estimate. A range also cannot be put
 * into an invalid or empty state, so the output is never blank or NaN.
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

  var FIELDS = [
    {
      id: "mins",
      label: "Pipeline cold-run time",
      min: 1, max: 60, step: 1, value: 12,
      unit: function (v) { return v === 1 ? "1 min" : v + " min"; }
    },
    {
      id: "iters",
      label: "Iterations per workday",
      min: 1, max: 30, step: 1, value: 8,
      unit: function (v) { return v === 1 ? "1 run" : v + " runs"; }
    },
    {
      id: "days",
      label: "Workdays per week",
      min: 1, max: 7, step: 1, value: 5,
      unit: function (v) { return v === 1 ? "1 day" : v + " days"; }
    }
  ];

  function rowHtml(f) {
    return [
      '<div class="cash-calculator-row">',
      '  <label for="cash-calc-' + f.id + '">' + f.label + '</label>',
      '  <output class="cash-calculator-value" id="cash-calc-' + f.id + '-out"',
      '          for="cash-calc-' + f.id + '"></output>',
      '  <input class="cash-calculator-slider" id="cash-calc-' + f.id + '"',
      '         type="range" min="' + f.min + '" max="' + f.max + '"',
      '         step="' + f.step + '" value="' + f.value + '">',
      '</div>'
    ].join("\n");
  }

  function buildCalculator(root) {
    root.innerHTML =
      FIELDS.map(rowHtml).join("\n") +
      [
        '<div class="cash-calculator-output" aria-live="polite">',
        '  <span class="cash-calculator-figure" data-cash-calc-hours></span>',
        '  <span class="cash-calculator-sub" data-cash-calc-days></span>',
        '</div>',
        '<div class="cash-calculator-caveat">',
        '  Assumes a 90% cache hit rate on iterations and an 8-hour workday.',
        '  A rough estimate, not a measurement — see',
        '  <a href="benchmarks.md">Benchmarks</a> for numbers you can reproduce.',
        '</div>'
      ].join("\n");

    var inputs = FIELDS.map(function (f) {
      return {
        field: f,
        el: root.querySelector("#cash-calc-" + f.id),
        out: root.querySelector("#cash-calc-" + f.id + "-out")
      };
    });
    var hoursOut = root.querySelector("[data-cash-calc-hours]");
    var daysOut = root.querySelector("[data-cash-calc-days]");

    function recompute() {
      var v = {};
      inputs.forEach(function (i) {
        var n = Number(i.el.value) || 0;
        v[i.field.id] = n;
        i.out.textContent = i.field.unit(n);
        // Paint the filled portion of the track, so the slider reads as a
        // value and not just a knob on a line.
        var pct = ((n - i.field.min) / (i.field.max - i.field.min)) * 100;
        i.el.style.setProperty("--cash-slider-pct", pct + "%");
      });

      var hoursPerWeek = (v.iters * v.days * v.mins * HIT_RATE) / 60;
      var daysPerMonth = (hoursPerWeek * WEEKS_PER_MONTH) / WORKDAY_HOURS;
      hoursOut.textContent = "≈ " + hoursPerWeek.toFixed(1) + " hours a week";
      daysOut.textContent =
        "that's about " + daysPerMonth.toFixed(1) + " workdays a month";
    }

    inputs.forEach(function (i) {
      i.el.addEventListener("input", recompute);
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
