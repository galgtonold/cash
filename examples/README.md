# cash examples

Start with the live tour. It runs in your browser, with nothing to install:
[open it in Colab](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb)
or [in Binder](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb).

To run the others locally, install cash with pandas support:

```bash
pip install "cash-lib[pandas]"
```

In a notebook, judge each cell by its badge, not by the clock: CACHED means
cash restored the result, EXECUTED means the code ran.

## Notebook examples

Each one starts with `import cash` then `%cash_on`, and has **Try this**
steps that change one thing and show what runs again.

| Notebook | Shows | Needs |
|---|---|---|
| [`try_cash_binder.ipynb`](try_cash_binder.ipynb) | The live tour: a Monte Carlo model, edits upstream and downstream, the decorator | numpy |
| [`demo_cell_caching.ipynb`](demo_cell_caching.ipynb) | Statement caching, a changed input, `# @cash:ttl` | nothing |
| [`cache_calls_demo.ipynb`](cache_calls_demo.ipynb) | Caching of the functions a cell calls | nothing |
| [`file_caching_demo.ipynb`](file_caching_demo.ipynb) | A CSV read that runs again when the file changes | pandas |
| [`file_tracking_demo.ipynb`](file_tracking_demo.ipynb) | File reads through `open`, pandas, `pathlib` and numpy | pandas, numpy |
| [`demo_notebook_caching.ipynb`](demo_notebook_caching.ipynb) | A sales pipeline: joins, aggregates, a plot, `%cash_off` | pandas, numpy, matplotlib |
| [`financial_analysis_demo.ipynb`](financial_analysis_demo.ipynb) | Rolling features, a loop cached per iteration, branches, provenance | pandas, numpy |
| [`cfd_simulation_demo.ipynb`](cfd_simulation_demo.ipynb) | A Navier-Stokes solver, a grid study, saving and reloading results | numpy, scipy |
| [`large_scale_projects/`](large_scale_projects/) | Ten analyses of real datasets: NYC taxi trips, Wikipedia pageviews, census data and others | pandas, numpy, matplotlib, scipy, scikit-learn, and hundreds of MB to several GB of downloads |

The large-scale notebooks download or generate their data on the first run.
Notebooks 02 to 07 expect the repository root as the working directory and
write to `examples/large_scale_projects/data/`; the others write to `data/`
under the working directory.

## Script examples

| Script | Shows | Needs |
|---|---|---|
| [`script_caching_demo.py`](script_caching_demo.py) | `@cash.cache` in a plain script: repeated calls, arguments, a file the function reads, `ttl` | nothing |

Run it twice from the repository root. The second run answers sections 1 and
2 from the cache.

```bash
python examples/script_caching_demo.py
```

## Magics used in the notebooks

| Magic | What it does |
|---|---|
| `%cash_on` / `%cash_off` | Start or stop caching |
| `%cash_stats` | Hits, misses and time saved this session |
| `%cash_provenance df` | How `df` was computed and what it depends on |
| `%cash_provenance df --graph` | The same, as a dependency tree |
| `%cash_provenance df --time` | The same, with a timeline of the computations |

The [magics reference](https://cash-lib.readthedocs.io/en/latest/magics/)
covers the rest.
