"""``history = net.fit(X, epochs=3)`` trains ``net`` on every Run All.

Keras-style training returns a history. Cached like a pure capture, the next
Run All in the same kernel restored ``history`` and never trained ``net``, so
predictions came from an untrained model, with no warning.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

NET = (
    "import time\n"
    "class Net:\n"
    "    def __init__(self):\n"
    "        self.w = 0.0\n"
    "    def fit(self, X, epochs=1):\n"
    "        hist = []\n"
    "        for _ in range(epochs):\n"
    "            time.sleep(0.1)\n"
    "            self.w += sum(X)\n"
    "            hist.append(self.w)\n"
    "        return {'loss': hist}\n"
    "    def predict(self, x):\n"
    "        return self.w * x"
)


def test_a_second_run_all_trains_the_model_again(nb_runner):
    nb_runner.create_notebook(
        [NET, "X = [1.0, 2.0]\nnet = Net()", "history = net.fit(X, epochs=3)", "pred = net.predict(10)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("pred") == "90.0", "first Run All"
    nb_runner.run_all()
    assert nb_runner.peek("pred") == "90.0", "second Run All"
