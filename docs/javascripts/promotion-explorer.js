/* Promotion explorer — page 7 (Storage & tiers).
 * Progressive enhancement: replaces the static fallback table inside
 * .cash-promotion-explorer with sliders + a live promotion verdict.
 *
 * Mirrors the real decision in src/cash/backends/tiered_backend.py
 * (_cost_model_promote) with the smart-persistence settings the factory
 * installs by default (src/cash/backends/factory.py):
 *   execution_time < 0.1s                       -> RAM only (compute floor)
 *   time - est_restore > 0.20 * time            -> persist to disk
 *   else                                        -> RAM only
 * est_restore comes from the fitted cost model in
 * src/cash/notebook/cost_model.py: a + b * size_bytes, per
 * (family, backend, operation). The coefficients below are the "disk" /
 * "deserialize" row for each family and MUST be kept in sync with that
 * module if it is refitted. */
(function () {
  "use strict";

  var COMPUTE_FLOOR_S = 0.1;   // _SMART_PERSIST_COMPUTE_FLOOR_S
  var MIN_SAVINGS = 0.20;      // CashConfig.min_cache_savings_pct
  var BYTES_PER_MB = 1024 * 1024;

  /* (a, b) for ("<family>", "disk", "deserialize") in cost_model._COEFFS. */
  var FAMILIES = [
    { label: "Other / unknown", family: "_GENERIC", a: 1.038392e-2, b: 1.976465e-9 },
    { label: "pandas DataFrame", family: "dataframe_numeric", a: 8.107757e-3, b: 1.576640e-9 },
    { label: "pandas Series", family: "series_numeric", a: 1.278298e-2, b: 1.537032e-9 },
    { label: "numpy ndarray", family: "ndarray_dense", a: 9.604519e-3, b: 1.547854e-9 },
    { label: "dict", family: "dict_shallow", a: 1.038392e-2, b: 1.976465e-9 },
    { label: "list / tuple", family: "list_flat", a: 8.612852e-3, b: 1.395863e-9 },
    { label: "bytes", family: "bytes", a: 9.503194e-3, b: 4.242689e-10 }
  ];

  function estimatedRestoreS(fam, sizeMB) {
    return fam.a + fam.b * (sizeMB * BYTES_PER_MB);
  }

  function fmt(s) {
    return s < 1 ? (s * 1000).toFixed(0) + " ms" : s.toFixed(2) + " s";
  }

  function decide(timeS, sizeMB, fam) {
    if (timeS < COMPUTE_FLOOR_S) {
      return {
        persist: false,
        label: "L1 only (RAM)",
        why: "Under the " + COMPUTE_FLOOR_S + " s compute floor — disk I/O alone " +
             "would cost more than rerunning it. Note the floor is per entry: " +
             "many cheap statements never add up to a persisted one."
      };
    }
    var restore = estimatedRestoreS(fam, sizeMB);
    var saved = timeS - restore;
    var required = MIN_SAVINGS * timeS;
    if (saved > required) {
      return {
        persist: true,
        label: "L1 + L2 (persisted to disk)",
        why: "Predicted restore of " + sizeMB + " MB is " + fmt(restore) +
             ", so a hit saves " + fmt(saved) + " — more than the " +
             fmt(required) + " (20% of compute) the write has to earn."
      };
    }
    return {
      persist: false,
      label: "L1 only (RAM)",
      why: "Predicted restore of " + sizeMB + " MB is " + fmt(restore) +
           ", leaving only " + fmt(saved) + " of savings — short of the " +
           fmt(required) + " (20% of compute) required to justify the write."
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

  function typePicker() {
    var wrap = document.createElement("label");
    wrap.className = "cash-pe-field";
    var cap = document.createElement("span");
    cap.className = "cash-pe-cap";
    cap.textContent = "Value type: ";
    var select = document.createElement("select");
    for (var i = 0; i < FAMILIES.length; i++) {
      var opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = FAMILIES[i].label;
      select.appendChild(opt);
    }
    wrap.appendChild(cap);
    wrap.appendChild(select);
    return { wrap: wrap, input: select };
  }

  function build(root) {
    root.textContent = "";

    var time = slider(0, 10, 0.1, 3.0, "Compute time", "s");
    var size = slider(0, 1500, 10, 100, "Result size", "MB");
    var type = typePicker();

    var panel = document.createElement("div");
    panel.className = "cash-pe-panel";
    panel.setAttribute("aria-live", "polite");

    function update() {
      var t = parseFloat(time.input.value);
      var s = parseFloat(size.input.value);
      var fam = FAMILIES[parseInt(type.input.value, 10)] || FAMILIES[0];
      time.out.textContent = t.toFixed(1) + " " + time.unit;
      size.out.textContent = s + " " + size.unit;
      var d = decide(t, s, fam);
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
    type.input.addEventListener("change", update);

    var controls = document.createElement("div");
    controls.className = "cash-pe-controls";
    controls.appendChild(time.wrap);
    controls.appendChild(size.wrap);
    controls.appendChild(type.wrap);

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
