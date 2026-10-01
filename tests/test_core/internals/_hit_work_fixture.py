"""Functions for test_a_hit_does_not_redo_key_work.py.

A real module (not collected by pytest), so its globals are module globals
the key folds, and the tests can rebind or edit them from outside.
"""

import dataclasses

SCALE = 2.5
NAME = "exp1"
SHAPE = (3, (4, "x"))
COLS = ["a", "b"]
CONFIG = {"lr": 0.1, "layers": [64, 32]}
RATE = 0.1


def reads_constants(x):
    return x * SCALE if NAME and SHAPE else 0.0


def reads_containers(x):
    return x * len(COLS) * CONFIG["lr"]


@dataclasses.dataclass
class Config:
    alpha: float = 0.5


class Scaler:
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return x * self.k * RATE


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = Scaler(cfg.alpha)

    def predict(self, x):
        return self.scaler.apply(x)


def builds_classes(x):
    return Model(Config(alpha=0.3)).predict(x)
