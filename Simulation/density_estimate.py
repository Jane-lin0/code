import array
import random

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut
import numpy as np

cv = LeaveOneOut()


def density_estimate(arr: array, a_grid: array) -> array:
    """
    预测 continuous treatment A 的密度函数 p_A(a)
    @param arr: 用于拟合 kernel，例如 df['a']
    @param a_grid: A 的值 a
    @return: density_estimated: ndarray:(a_grid,1), a 对应的密度函数估计值 p_A(a)
    """
    param_grid = {'bandwidth': np.logspace(-1, 1, num=20, base=10)}  # 生成 1/10 ~ 10 之间对数均匀的100个数
    grid_search = GridSearchCV(KernelDensity(kernel='gaussian'), param_grid, cv=cv)
    grid_search.fit(arr)  # 拟合模型
    # best_bandwith = grid_search.best_estimator_.bandwidth
    log_dens = grid_search.score_samples(a_grid)
    density_estimated = np.exp(log_dens)
    return density_estimated


