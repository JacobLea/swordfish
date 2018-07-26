# -*- encoding: utf-8 -*-

import numpy as np

__all__ = [
    'SimulatedAnnealingSearch',
]


class SimulatedAnnealingSearch(object):
    """ Search almost best hyper-parameters by simulated annealing algorithm for an estimator.

    Important members are fit, predict.

    To get more detail of simulated annealing algorithm, please see https://en.wikipedia.org/wiki/Simulated_annealing
    A easy-understanding explanation of it in Chinese is http://www.cnblogs.com/heaad/archive/2010/12/20/1911614.html

    Parameters
    ----------
    model_train_func : function object
        This is a estimator training function to train a estimator with training data and parameter as input and an
        estimator instance as return. It must be like:
        def func(params, x, labels, validate_x, validate_y):
            ...
            return estimator

    init_params : dict object
        This is the initial hyper-parameters for estimator training function. It should be like:
        {a hyper-parameter name: a hyper-parameter value, ...}

    params_chg_steps : dict object
        This contains functions and necessary constant to update hyper-parameters. It should be like:
        {a hyper-parameter name: [the second input of the following function objects, (func0, func1,...)], ...}
        The function objects should be like:
        func(old value, step) --> new value

    param_limitations : dict object.
        This contains the minimum and maximal values of hyer-parameters. It should be like:
        {a hyper-parameter name: (minimum value, maximal value), ...}

    eval_func : function object
        This is a function returning a score to evaluate if the fitted model is better or worse. Please make sure the
        score is as bigger as better. It should be like:
        def func(model, x, labels, validate_x, validate_y):
            ...
            return score

    bigger_limitation : float
        This number is to the number how much the current model need be bigger than the last one is called 'better'.
        The scope of it is depended on how you define eval_func.

    init_t : float.
        This number is the initial temperature in simulated annealing algorithm.

    end_t : float.
        This number is the end temperature in simulated annealing algorithm. It should be littler than init_t.

    rate : float
        This number is the lower rate of the temperature in simulated annealing algorithm. It should be in scope (0, 1).

    random_limitation_val : float
        In simulated annealing algorithm, if the current score is not better than last one, a random number in the
        scope (0, 1) must be generated to compare with current state value. The value is to adjust the scope of the
        the random number generator. It should be in scope (0, 1). It's more litter, the searcher iterates more times.

    random_seed : int or None
        This is the random seed to randomly change the values of the hyper-parameters. The actually seed is calculated
        by the number add the current number of iteration.

    max_iter : int
        This is to limited the max times of iteration. The iteration will stop once the times gets it.

    verbose : True or False
        It depends if the searcher prints log in real-time.


    Attributes
    ----------
    best_score_:
        The score of almost best model your get, which is also calculated by the function eval_func..
    
    best_model_
        The model with the highest score.
    
    best_params_
        The hyper-parameters which was used to get the best model..
    
    iter_times_
        The iteration times the seatcher ran.
    
    trained_times_
        The actually trained times the searcher ran.


    """

    def __init__(self, model_train_func, init_params: dict, params_chg_steps: dict, param_limitations: dict, eval_func,
                 bigger_limitation=0., init_t=2., end_t=0.01, rate=0.95, random_limitation_val=0.6, random_seed=None,
                 max_iter=None, verbose=True):
        assert callable(model_train_func)
        assert callable(eval_func)
        assert bigger_limitation >= 0.
        assert init_t > end_t > 0.
        assert 0 < rate < 1.
        assert 0 < random_limitation_val <= 1.
        self._all_params_keys = [k for k in init_params.keys()]
        assert all([(k in self._all_params_keys) for k in params_chg_steps.keys()])
        self._update_params_keys = [k for k in params_chg_steps.keys()]
        self._update_params_keys_cnt = len(self._update_params_keys)
        self._model_train_func = model_train_func
        self._init_params = init_params.copy()
        self._current_params = init_params.copy()
        self._params_chg_steps = params_chg_steps.copy()
        self._param_limitations = param_limitations.copy()
        self._eval_func = eval_func
        self._bigger_limitation = bigger_limitation
        self._init_t = init_t
        self._current_t = init_t
        self._end_t = end_t
        self._radio = rate
        self._random_seed = random_seed
        self._random_limitation_func = lambda: np.random.rand() * random_limitation_val
        self._params_last_op_idx = {k: None for k in params_chg_steps.keys()}
        self._is_fitted = False
        self._max_iter = max_iter
        self._verbose = verbose
        self._last_score = None
        self._params_history = []
        self._score_history = []
        self._uphill_history = []
        self._force_reset = False
        self.best_score_ = -np.inf
        self.best_model_ = None
        self.best_params_ = None
        self.iter_times_ = 0
        self.trained_times_ = 0

    def _random_update_current_params(self):
        changed = False
        for k in self._update_params_keys:
            step = self._params_chg_steps[k][0]
            ops = self._params_chg_steps[k][1]
            if (self._params_last_op_idx[k] is not None) & (not self._force_reset):
                op_idx = self._params_last_op_idx[k]
            else:
                op_idx = np.random.choice(np.array(len(ops) + 1))
                self._params_last_op_idx[k] = op_idx
            if op_idx > 0:
                op = ops[op_idx - 1]
                new_val = op(self._current_params[k], step)
                if self._param_limitations is not None:
                    if k in self._param_limitations.keys():
                        if (new_val < self._param_limitations[k][0]) | (new_val > self._param_limitations[k][1]):
                            continue
                self._current_params[k] = new_val
                changed = True
        return not changed

    def _reset_params_last_op_idx(self):
        for k in self._params_last_op_idx.keys():
            self._params_last_op_idx[k] = None

    def fit(self, data, labels=None, validate_data=None, validate_labels=None, rtn_model=False):
        if not self._is_fitted:
            while self._current_t > self._end_t:
                not_trained = True
                self._force_reset = False
                self.iter_times_ += 1
                for i, s in enumerate(self._params_history):
                    if s == self._current_params:
                        not_trained = False
                        score = self._score_history[i]
                        break
                model = None
                if not_trained:
                    model = self._model_train_func(params=self._current_params.copy(), data=data, labels=labels,
                                                   validate_data=validate_data, validate_labels=validate_labels)
                    score = self._eval_func(model, data=data, labels=labels, validate_data=validate_data,
                                            validate_labels=validate_labels)
                    self.trained_times_ += 1
                self._score_history.append(score)
                self._params_history.append(self._current_params.copy())
                if self._last_score is not None:
                    de = score - self._last_score
                    if de <= self._bigger_limitation:
                        if np.exp(de / self._current_t) < self._random_limitation_func():
                            break
                        self._reset_params_last_op_idx()
                if len(self._params_history) > 1:
                    if score > self._last_score:
                        uphill = (self._params_history[-2], self._params_history[-1])
                        self._force_reset = (uphill in self._uphill_history)
                        self._uphill_history.append(uphill)
                self._last_score = score
                if (score > self.best_score_ or self.iter_times_ == 0) and not_trained:
                    self.best_score_ = score
                    self.best_params_ = self._current_params.copy()
                    self.best_model_ = model
                self._current_t *= self._radio
                if self._verbose:
                    print(f'[{self.iter_times_}]: params: {self._current_params}')
                    print(f'[{self.iter_times_}]: score: {score}')
                if self._max_iter:
                    if self.iter_times_ >= self._max_iter:
                        break
                if self._random_seed is not None:
                    np.random.seed(self.iter_times_ + self._random_seed)
                is_frozen = self._random_update_current_params()
                while is_frozen:
                    self._reset_params_last_op_idx()
                    is_frozen = self._random_update_current_params()
            self._is_fitted = True
        if rtn_model:
            return self.best_model_

    def predict(self, x, **kwargs):
        if self._is_fitted:
            if hasattr(self.best_model_, 'predict'):
                return self.best_model_.predict(x, **kwargs)
            else:
                raise AttributeError(f'The final model has no method called "predict". \
                    Please use the attribute "{self.__class__.__name__}.best_model_" to get the final model.')
        else:
            raise Exception('The model has not been fitted yet.')
