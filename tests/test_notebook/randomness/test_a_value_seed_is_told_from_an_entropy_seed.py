"""A seed with a value is told from a seed from entropy in the same source."""

from cash.tracking.randomness.detect import get_value_seeding_rng_modules


def test_value_seeds_are_read():
    assert get_value_seeding_rng_modules("np.random.seed(0)") == {"numpy.random"}
    assert get_value_seeding_rng_modules("torch.manual_seed(1)") == {"torch"}


def test_an_entropy_seed_is_not_a_value_seed():
    assert get_value_seeding_rng_modules("np.random.seed()") == set()
    assert get_value_seeding_rng_modules("import random\nrandom.seed(None)\nnp.random.seed(3)") == {"numpy.random"}
