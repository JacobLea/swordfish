# -*- encoding: utf-8 -*-


from swordfish.faster_data_cleaner.transformer_utils import on_field
import pandas as pd
from sklearn.pipeline import make_union
from sklearn.preprocessing import FunctionTransformer

__all__ = [
    'fast_group_apply',
]


def fast_group_apply(df: pd.DataFrame, by, apply_func):
    dfs = [df for _, df in df.groupby(by)]
    return stack_apply(dfs, apply_func)


def stack_apply(dfs, apply_func):
    apply_function = _ApplyFunction(apply_func)
    pipelines = [on_field(i, FunctionTransformer(apply_function.get_real_func(), validate=False))
                 for i in range(len(dfs))]
    feature_union = make_union(*pipelines, n_jobs=1)
    return apply_function.df_transformer(feature_union.fit_transform(dfs))


class _ApplyFunction(object):
    def __init__(self, base_func):
        self._base_func = base_func
        self._columns = []
        self._dtypes = []

    def get_real_func(self):
        def func(df):
            rtn = self._base_func(df)
            if isinstance(rtn, pd.Series):
                if len(self._columns) == 0:
                    self._columns = rtn.index
                    for idx in self._columns:
                        dt = type(rtn[idx])
                        if any([(v in str(dt)) for v in ('int', 'float')]):
                            self._dtypes.append(dt)
                        else:
                            self._dtypes.append(None)
                return rtn.values.reshape(-1, 1)
            else:
                raise ValueError(f'The apply function {self._base_func.__name__} must return a Series instance.')

        return func

    def df_transformer(self, data):
        df = pd.DataFrame(data.T, columns=self._columns)
        for c, dt in zip(self._columns, self._dtypes):
            if dt is not None:
                df[c] = df[c].astype(dt)
        return df

