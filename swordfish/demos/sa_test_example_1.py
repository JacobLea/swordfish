# -*- encoding: utf-8 -*-


from operator import add, sub, mul, truediv
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from swordfish.hyper_parameter_selection import SimulatedAnnealingSearch


def model_train_func(params, data, validate_data, **kwargs):
    model = xgb.train(params, dtrain=data, evals=[(data, 'train'), (validate_data, 'validate')])
    return model


def eval_func(model, data, validate_data, **kwargs):
    eps = 1e-10
    labels = data.get_label()
    validate_labels = validate_data.get_label()
    acc_train = accuracy_score(labels, model.predict(data))
    acc_test = accuracy_score(validate_labels, model.predict(validate_data))
    diff = (acc_train - acc_test) if acc_train > acc_test else 0
    invariance = 1 / (1 + diff)
    beta2 = 0.8
    return (1 + beta2) * invariance * acc_test / (beta2 * invariance + acc_test + eps)


def test_sa_search():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    params = dict(
        objective='multi:softmax',
        max_depth=4,
        colsample_bytree=0.8,
        subsample=0.5,
        gamma=1.,
        reg_lambda=1.,
        eta=0.1,
        min_child_weight=3,
        silent=1,
        nthread=4,
        num_class=3,
    )
    param_steps = dict(
        max_depth=[1, (add, sub)],
        colsample_bytree=[0.2, (add, sub)],
        subsample=[0.1, (add, sub)],
        gamma=[3, (mul, truediv)],
        reg_lambda=[3, (mul, truediv)],
        eta=[3, (mul, truediv)],
        min_child_weight=[2, (add, sub)],
    )

    params_limitation = dict(
        max_depth=(2, 10),
        colsample_bytree=(0.2, 1.),
        subsample=(0.1, 0.9),
        gamma=(0, 1000),
        reg_lambda=(0, 1000),
        eta=(0.001, 1.),
        min_child_weight=(2, 128),
    )

    sa_searcher = SimulatedAnnealingSearch(model_train_func, params, param_steps, params_limitation, eval_func,
                                           random_seed=None, verbose=True, bigger_limitation=0)
    data_train = xgb.DMatrix(x_train, y_train)
    data_test = xgb.DMatrix(x_test, y_test)
    sa_searcher.fit(data=data_train, validate_data=data_test)
    print('iteration times:', sa_searcher.iter_times_)
    print('trained times:', sa_searcher.trained_times_)
    print('best params:', sa_searcher.best_params_)
    print('best score:', sa_searcher.best_score_)
    print('accuracy score of train data:', accuracy_score(y_train, sa_searcher.best_model_.predict(data_train)))
    print('accuracy score of test data:', accuracy_score(y_test, sa_searcher.best_model_.predict(data_test)))


if __name__ == '__main__':
    test_sa_search()
