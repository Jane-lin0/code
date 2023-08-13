import array
import random

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut
import numpy as np


def density_estimate(arr: array, a_approx: array) -> array:
    """
    预测 continuous treatment A 的密度函数 p_A(a)
    @param arr: 用于拟合 kernel，例如 df_train['a']
    @param a_approx: A 的值 a，用于预测
    @return: density_estimated: ndarray:(a_approx,1), a 对应的密度函数估计值 p_A(a)
    """
    param_grid = {'bandwidth': np.logspace(-1, 1, num=20, base=10)}  # 生成 1/10 ~ 10 之间对数均匀的20个数
    grid_search = GridSearchCV(KernelDensity(kernel='gaussian'), param_grid, cv=LeaveOneOut())
    grid_search.fit(arr)  # 拟合模型
    # best_bandwith = grid_search.best_estimator_.bandwidth

    log_dens = grid_search.score_samples(a_approx) # 预测
    density_estimated = np.exp(log_dens)
    return density_estimated


# df_train = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='train')
# df_validation = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='validation')
# df_test = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='test')
# a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet2')
#
# df = pd.concat([df_train, df_validation, df_test], axis=0).reset_index()
#
# density_estimated = density_estimate(df[['a']], a_grid)  # 长度和 a_grid 一致



