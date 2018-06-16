# -*- encoding: utf-8 -*-

from operator import add, sub, mul, truediv
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from swordfish.hyper_parameter_selection import SimulatedAnnealingSearch


def model_train_func(params, data, labels, **kwargs):
    model = Pipeline([('sc', StandardScaler()),
                      ('pca', PCA(whiten=True)),
                      ('ploy', PolynomialFeatures()),
                      ('linear_reg', RidgeClassifier(random_state=23, fit_intercept=True))
                      ])
    model.set_params(**params)
    model.fit(data, labels)
    return model


def eval_func(model, data, labels, validate_data, validate_labels):
    eps = 1e-10
    acc_train = accuracy_score(labels, model.predict(data))
    acc_test = accuracy_score(validate_labels, model.predict(validate_data))
    diff = (acc_train - acc_test) if acc_train > acc_test else 0
    invariance = 1 / (1 + diff)
    beta2 = 0.8
    return (1 + beta2) * invariance * acc_test / (beta2 * invariance + acc_test + eps)


def mul_with_round(a, b):
    return round(a * b, 8)


def truediv_with_round(a, b):
    return round(a / b, 8)


def test_sa_search():
    x, y = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    params = {'pca__n_components': 3, 'linear_reg__alpha': 0.1}
    param_steps = {'pca__n_components': [1, (add, sub)],
                   'linear_reg__alpha': [3, (mul_with_round, truediv_with_round)]}
    params_limitation = {'pca__n_components': (2, 4), 'linear_reg__alpha': (0.0001, 1000)}

    sa_searcher = SimulatedAnnealingSearch(model_train_func, params, param_steps, params_limitation, eval_func,
                                           random_seed=1, verbose=False)
    sa_searcher.fit(x_train, y_train, x_test, y_test)
    print('iteration times:', sa_searcher.iter_times_)
    print('trained times:', sa_searcher.trained_times_)
    print('best params:', sa_searcher.best_params_)
    print('best score:', sa_searcher.best_score_)
    print('accuracy of train data:', accuracy_score(y_train, sa_searcher.predict(x_train)))
    print('accuracy of test data:', accuracy_score(y_test, sa_searcher.predict(x_test)))


if __name__ == '__main__':
    test_sa_search()
