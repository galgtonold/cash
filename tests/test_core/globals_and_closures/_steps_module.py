"""Tables of code kept in another module, for
`test_code_held_by_another_module_or_a_class_keys_what_it_reads`."""

RATE = 2


def scale(x):
    return x * RATE


STEPS = [scale]
HANDLERS = {"scale": scale}


class Pipeline:
    DEFAULT_STEPS = [scale]

    def __init__(self, steps):
        self.steps = steps

    def run(self, x):
        for step in self.steps:
            x = step(x)
        return x


PIPE = Pipeline([scale])


class Lin:
    def predict(self, x):
        return x * RATE


MODELS = {"lin": Lin}
