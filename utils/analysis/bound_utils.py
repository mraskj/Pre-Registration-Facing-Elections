import math
import numpy as np

from typing import Union

def rho_h(l:float, m:int, loss_h:float, loss_min:float, n:int, r:int):
    pi = 1 / m
    rho_num = pi * np.exp(-l * (n - r) * (loss_h - loss_min))
    return rho_num

def KL(p, q):
    return sum(p * np.log(p / q) for p,q in zip(p, q))

def expected_loss(rho: Union[list, np.ndarray], loss_h:list):
    return sum(x * y for x, y in zip(rho, loss_h))

def optimal_lambda(rho, m:int, n:int, r:int, delta:float, e_loss:float):
    pi = 1 / m
    return 2 / (math.sqrt((2 * (n - r) * e_loss) / (KL(rho, [pi] * len(rho)) + np.log((2 * np.sqrt(n - r)) / delta)) + 1) + 1)

def PAC_Bayesian_Aggregation(val_loss:list, l:float, rho:list, m:int, delta:float, n:int, r:int):
    pi = 1 / m
    return expected_loss(rho, val_loss) / (1 - (l / 2)) + (KL(rho, [pi] * len(rho)) + np.log((2 * np.sqrt(n - r)) / delta)) / (l * (1 - (l / 2)) * (n - r))

