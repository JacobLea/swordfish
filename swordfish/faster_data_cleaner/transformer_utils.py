# -*- encoding: utf-8 -*-


from operator import itemgetter
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

__all__ = [
    'on_field',
]


def on_field(s, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(s), validate=False), *vec)
