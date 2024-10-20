""" This module implements a hybrid minimizer that starts by exploring the parameter space using the Differential
    Evolution method, after which a local minimizer is used with the best value as initial estimate. """

import numpy as np
from typing import List, Tuple, Callable, final
from scipy.optimize import differential_evolution, minimize, OptimizeResult


_SEED: final = 2 ** 20


def minimize_hybrid(fun,
                    global_bounds: List[Tuple[float, float]],
                    args: tuple = (),
                    local_method='BFGS',
                    jac: Callable = None,
                    hess: Callable = None,
                    hessp: Callable = None,
                    callback: Callable = None,
                    local_options: dict = None,
                    seed: int = _SEED,
                    popsize: int = 15,
                    global_iter: int = 100,
                    x0: np.ndarray = None,
                    workers: int = 1) -> OptimizeResult:

    result = differential_evolution(fun, bounds=global_bounds, args=args, callback=callback, seed=seed, popsize=popsize,
                                    maxiter=global_iter, x0=x0, workers=workers)

    x0 = result.x
    return minimize(fun, x0=x0, args=args, method=local_method, jac=jac, hess=hess, hessp=hessp, callback=callback,
                    options=local_options)
