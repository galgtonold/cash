# Report a bug

Bugs, wrong results and unclear docs go to the
[issue tracker on GitHub](https://github.com/galgtonold/cash/issues/new?template=bug_report.md).
A report that includes the steps below can usually be reproduced on the
first try.

!!! warning "Security problems"
    Do not open a public issue for something an attacker could use. Report
    it privately, as the [security policy](security.md) describes.

## What to include

- The versions: `cash version`, `python --version` and your OS. For a
  notebook, also the frontend (JupyterLab, VS Code, Colab).
- The smallest code that shows the problem, and what you expected instead.
- What cash said about it, from the steps below for your path.

=== "Decorator"

    <!-- claim: cash/decorator/explain.py:describe_state_change @7b3bcda1, cash/decorator/explain.py:Explainer.explain @7736721e -->
    Run the program with `CASH_VERBOSE=1`. cash prints one line per call,
    and a miss says what changed. For a single call,
    `print(f.explain(*args))` says whether it would hit, and why, without
    running it.

    ```bash
    CASH_VERBOSE=1 python your_script.py
    ```

    [Seeing what cash did](decorator.md#seeing-what-cash-did) shows what
    these lines look like.

=== "Notebook"

    <!-- claim: cash/notebook/badge_renderer/view_builder.py:_bug_report_url @79a186f6 -->
    Open the badge's panel: it ends with a **Report incorrect caching**
    link. The link opens a new issue filled in with the badge's rows, the
    source of the notebook's cells, your cash and Python versions and the
    backend. Read it before you submit and remove anything private.

    For a problem the badge does not show, turn on the debug log, run the
    cells again and paste what it prints:

    ```python { .nb-cell }
    %cash_debug on
    ```

    See [Debugging a notebook](tutorials/feature-guides/debugging-and-monitoring.md)
    for what the log says.

## Questions

A question that is not a bug fits better in
[GitHub Discussions](https://github.com/galgtonold/cash/discussions).

## Related

- [Security policy](security.md): how to report a vulnerability privately.
- [Writing cache-safe cells](known-limitations.md): notebook behaviour that
  is known and documented, with the fix for each case.
- [Changelog](changelog.md): whether a newer release already fixed it.
