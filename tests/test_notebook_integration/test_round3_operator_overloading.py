"""Operator overloading & dunder methods — cash caching with custom operators."""

import textwrap

import pytest


@pytest.mark.stress
class TestContainerDunders:
    """Test container protocol dunders."""

    def test_change_propagation_dunders(self, nb_runner):
        """Operator result propagation after change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Money:
                    def __init__(self, amount, currency='USD'):
                        self.amount = amount
                        self.currency = currency
                    def __add__(self, other):
                        if self.currency != other.currency:
                            raise ValueError("Currency mismatch")
                        return Money(self.amount + other.amount, self.currency)
                    def __repr__(self):
                        return f"{self.amount} {self.currency}"

                a = Money(100)
                b = Money(50)
            """),
                textwrap.dedent("""\
                total = a + b
                print(f"total={total}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=150 USD" in nb_runner.get_output(2)

        # Change amount
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Money:
                def __init__(self, amount, currency='USD'):
                    self.amount = amount
                    self.currency = currency
                def __add__(self, other):
                    if self.currency != other.currency:
                        raise ValueError("Currency mismatch")
                    return Money(self.amount + other.amount, self.currency)
                def __repr__(self):
                    return f"{self.amount} {self.currency}"

            a = Money(200)
            b = Money(75)
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "total=275 USD" in nb_runner.get_output(2)
