/* Cacheability checker — page 4 (Safety).
 * Progressive enhancement: replaces the static fallback table inside
 * .cash-cacheability-checker with a clickable snippet picker + verdict panel.
 * Verdict tiers: "ok" (cached), "warn" (cached + warning), "no" (not cached). */
(function () {
  "use strict";

  var SNIPPETS = [
    {
      code: "df = pd.read_csv('data.csv')",
      verdict: "ok",
      title: "Cached",
      why: "A plain read. The result is cached, and the file is registered as a dependency — if data.csv changes, the cache invalidates automatically."
    },
    {
      code: "result = df.groupby('k').sum()",
      verdict: "ok",
      title: "Cached",
      why: "A pure transformation: no mutation, no side effect, no randomness. Cash caches the result keyed on the inputs' lineage."
    },
    {
      code: "total += 1",
      verdict: "ok",
      title: "Cached",
      why: "An augmented assignment mutates total, but total is also the statement's own output — so the mutation is just a new version of the variable. Its lineage advances and the new value is cached."
    },
    {
      code: "data.append(4)",
      verdict: "no",
      title: "Not cached",
      why: "append() mutates data in place, and this statement doesn't produce data — there is no output to hang the new version on. Cash bumps data's lineage (so downstream cells see the change) and re-runs the statement every time, rather than restoring a deserialised copy and breaking every other reference to the list."
    },
    {
      code: "del lookup['stale']",
      verdict: "no",
      title: "Not cached",
      why: "A subscript delete mutates lookup, and unlike lookup['k'] = v it leaves no store target the analyzer can read as an output. Nothing can attribute the new version to this statement, so it always re-executes."
    },
    {
      code: "x = np.random.randn(100)",
      verdict: "warn",
      title: "Cached + warning",
      why: "Unseeded randomness. Cash still caches — and the badge marks the row 'unseeded', because the value is a frozen replay rather than a fresh draw. Seed the module first (np.random.seed(0)) for real reproducibility, use @cash:no-cache to redraw every run, or @cash:allow-random to silence the warning."
    },
    {
      code: "model.fit(X, y)",
      verdict: "no",
      title: "Not cached by default",
      why: "A bare fit mutates its receiver in place, so it takes the ordinary mutation path: re-executed every run, never serialised. Opt in with the # @cash:cache-fit annotation, which caches the fitted state and restores it in place. For reliable ML caching, wrap training in a function that returns the model and decorate it with @cash.cache."
    },
    {
      code: "df.to_parquet('out.pq')",
      verdict: "no",
      title: "Not cached",
      why: "A file write is a side effect. Replaying it from cache would skip writing the file, so Cash flags the statement uncacheable and always runs it. It does NOT count as a mutation of df: to_parquet reads the frame, so df's lineage is left alone."
    },
    {
      code: "r = requests.post(url, json=payload)",
      verdict: "no",
      title: "Not cached",
      why: "A mutating network call (POST/PUT/DELETE/PATCH) is a side effect: a cache hit would never send the request, so Cash always executes it. Read-style requests.get() calls are cacheable, like reading a file."
    }
  ];

  var VERDICT_LABEL = { ok: "CACHED", warn: "CACHED + WARNING", no: "NOT CACHED" };

  function build(root) {
    root.textContent = "";

    var list = document.createElement("div");
    list.className = "cash-cc-list";

    var panel = document.createElement("div");
    panel.className = "cash-cc-panel";
    panel.setAttribute("aria-live", "polite");

    var buttons = [];

    function select(i) {
      for (var j = 0; j < buttons.length; j++) {
        var on = j === i;
        buttons[j].classList.toggle("active", on);
        buttons[j].setAttribute("aria-pressed", on ? "true" : "false");
      }
      var s = SNIPPETS[i];
      panel.className = "cash-cc-panel " + s.verdict;
      var pill = document.createElement("span");
      pill.className = "cash-cc-pill";
      pill.textContent = VERDICT_LABEL[s.verdict];
      var head = document.createElement("div");
      head.className = "cash-cc-verdict";
      head.appendChild(pill);
      var headText = document.createElement("strong");
      headText.textContent = " " + s.title;
      head.appendChild(headText);
      var why = document.createElement("p");
      why.className = "cash-cc-why";
      why.textContent = s.why;
      panel.textContent = "";
      panel.appendChild(head);
      panel.appendChild(why);
    }

    for (var i = 0; i < SNIPPETS.length; i++) {
      (function (idx) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "cash-cc-snippet";
        b.setAttribute("aria-pressed", "false");
        var code = document.createElement("code");
        code.textContent = SNIPPETS[idx].code;
        b.appendChild(code);
        var dot = document.createElement("span");
        dot.className = "cash-cc-dot " + SNIPPETS[idx].verdict;
        b.appendChild(dot);
        b.addEventListener("click", function () { select(idx); });
        list.appendChild(b);
        buttons.push(b);
      })(i);
    }

    root.appendChild(list);
    root.appendChild(panel);
    select(0);
  }

  function init() {
    var roots = document.querySelectorAll(".cash-cacheability-checker");
    for (var i = 0; i < roots.length; i++) {
      if (roots[i].dataset.cashCcReady === "1") continue;
      roots[i].dataset.cashCcReady = "1";
      build(roots[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
