# -*- encoding: utf-8 -*-

from operator import add, sub, mul, truediv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from swordfish.hyper_parameter_selection import SimulatedAnnealingSearch


def model_train_func(params, data, labels, **kwargs):
    model = Pipeline([('sc', StandardScaler()),
                      ('pca', PCA(whiten=True)),
                      ('ploy', PolynomialFeatures()),
                      ('linear_reg', Ridge(random_state=23, fit_intercept=True))
                      ])
    model.set_params(**params)
    model.fit(data, labels)
    return model


def eval_func(model, data, labels, validate_data, validate_labels):
    # range transform [0, inf) -> [1, 0)
    mse_train = mean_squared_error(labels, model.predict(data))
    mse_test = mean_squared_error(validate_labels, model.predict(validate_data))
    diff = (mse_test - mse_train) if mse_test > mse_train else 0
    invariance = 1 / (1 + np.log1p(diff))
    unbiasedness = 1 / (1 + np.log1p(mse_test))
    # like f score
    beta2 = 0.6
    return (1 + beta2) * invariance * unbiasedness / (beta2 * invariance + unbiasedness)


def test_sa_search():
    m, n = 2000, 60
    np.random.seed(4)
    x = np.dot((np.random.rand(m, n) - 0.5), (np.diag(np.logspace(-6, 0, n))))
    y_prefect = (- x ** 2 * 3 + x * 5 + 7).sum(axis=1)
    y = y_prefect + (np.random.rand(m) - 0.5) * 10
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    params = {'pca__n_components': 35, 'linear_reg__alpha': 0.1}
    param_steps = {'pca__n_components': [1, (add, sub)],
                   'linear_reg__alpha': [3, (mul, truediv)]}
    params_limitation = {'pca__n_components': (3, 50), 'linear_reg__alpha': (0.0001, 1000)}

    sa_searcher = SimulatedAnnealingSearch(model_train_func, params, param_steps, params_limitation, eval_func,
                                           random_seed=1, verbose=False)
    sa_searcher.fit(x_train, y_train, x_test, y_test)
    print('iteration times:', sa_searcher.iter_times_)
    print('trained times:', sa_searcher.trained_times_)
    print('best params:', sa_searcher.best_params_)
    print('best score:', sa_searcher.best_score_)
    print('mse of train data:', mean_squared_error(y_train, sa_searcher.predict(x_train)))
    print('mse of test data:', mean_squared_error(y_test, sa_searcher.predict(x_test)))
    print('prefect mse:', mean_squared_error(y, y_prefect))


if __name__ == '__main__':
    test_sa_search()
