"""``X = vec.fit_transform(texts)`` leaves ``vec`` fitted on every run.

Round 23 (r23s4). The statement returns ``X`` AND fits ``vec`` in place; it
was cached with ``X`` as its only output, so a hit restored ``X`` and left the
freshly constructed ``vec`` unfitted -- silent in the same kernel,
``NotFittedError`` after a restart. The same in a loop:
``labels_k = km.fit_predict(Z); models[k] = km`` kept unfitted models.
"""
import pytest

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration]

TEXTS = ("from sklearn.feature_extraction.text import TfidfVectorizer\n"
         "texts = [f'doc {i} topic {i % 7} alpha{i % 13} beta{i % 5} delta{i}' for i in range(20000)]")
FIT = "vectorizer = TfidfVectorizer(ngram_range=(1, 2))\nX = vectorizer.fit_transform(texts)"
CHECK = "print('FITTED', hasattr(vectorizer, 'vocabulary_'), X.shape[0])"

DATA = ("import numpy as np\nfrom sklearn.cluster import KMeans\n"
        "Z = np.random.RandomState(0).rand(3000, 20)")
SWEEP = ("models = {}\n"
         "for k in [3, 5]:\n"
         "    km = KMeans(n_clusters=k, n_init=3, random_state=0)\n"
         "    labels_k = km.fit_predict(Z)\n"
         "    models[k] = km\n"
         "print('FITTED', all(hasattr(m, 'labels_') for m in models.values()))")


def test_fit_transform_on_a_rerun_and_after_a_restart(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", TEXTS, FIT, CHECK])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "FITTED True 20000" in nb_runner.get_output(4)
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    assert "FITTED True" in nb_runner.get_output(4), "a re-run left the vectorizer unfitted"
    nb_runner.restart()
    nb_runner.run_all()
    assert "FITTED True" in nb_runner.get_output(4), "Restart & Run All left the vectorizer unfitted"


def test_fit_predict_in_a_loop_keeps_fitted_models(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", DATA, SWEEP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "FITTED True" in nb_runner.get_output(3)
    nb_runner.run_cell(3)
    assert "FITTED True" in nb_runner.get_output(3), "a re-run kept unfitted models"
    nb_runner.restart()
    nb_runner.run_all()
    assert "FITTED True" in nb_runner.get_output(3), "Restart & Run All kept unfitted models"
