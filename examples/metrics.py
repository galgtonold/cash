import cash
import time
from advanced_metrics import number


def increment(counter: int) -> int:
    """Increment the counter by 1."""
    return number(counter) + 1


def bla(asdf):
    print("Hi")

@cash.cache
def dep(a):
    time.sleep(1)
    return a + 2

@cash.cache
def fun(a, b):
    time.sleep(1)
    return a + b + dep(a) + 1

@cash.cache
def super_fun(a):
    time.sleep(2)
    return a + 12