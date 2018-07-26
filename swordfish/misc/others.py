# -*- encoding: utf-8 -*-


import time
from contextlib import contextmanager
from operator import add
from functools import reduce
import pandas as pd

__all__ = [
    'timer',
    'list_minus',
]


@contextmanager
def timer(name, output=None):
    t0 = time.time()
    yield
    msg = '%s took seconds: %.3f' % (name, time.time() - t0)
    if output is None:
        print(msg)
    else:
        output(msg)


def list_minus(l0, *ls):
    l = l0
    for ln in ls:
        l = _list_minus(l, ln)
    return l


def list_sum(*ls):
    return reduce(add, [list(l) for l in ls if l is not None])


def _list_minus(l0, l1):
    if isinstance(l1, str):
        l1 = [l1]
    return [v for v in l0 if v not in l1]


def value_counts(values, p=None):
    if p is None:
        return pd.value_counts(values).sort_index()
    if isinstance(p, list) | isinstance(p, pd.Index):
        assert len(p) == len(set(p))
        return pd.value_counts(list_sum(values, p))[p] - 1
    if p in values:
        return pd.value_counts(values)[p]
    raise TypeError('Wrong type of p.')


if __name__ == '__main__':
    pass