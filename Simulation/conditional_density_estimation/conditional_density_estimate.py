import math

import numpy as np
import pandas as pd
import scipy.stats
import flexcode
from flexcode.regression_models import NN
import matplotlib.pyplot as plt


def cde_adjust(cde_list):
    """
    @param cde_row: 含 0 值的条件密度估计值
    @return: 不含 0 值的 cde
    """
    cde = []
    for val in cde_list:
        if val == 0:
            val += 1e-8
        cde.append(val)
    cde = np.array(cde)
    return cde


def cde_sample_estimate(cde_estimates, a_approx_index):
    cde_list = []
    for x_index, grid_index in enumerate(a_approx_index):
        cde_val = cde_estimates.iloc[x_index, grid_index]
        cde_list.append(cde_val)
    cde = cde_adjust(cde_list)  # 给 cde_list 的零值加上一个很小的值，避免求 pi 时除以 0 得到 inf
    return cde


'''
估计的条件密度几乎是零值
'''


def conditional_density_estimate(df_train, df_validation, df_test, n_grid):
    """
    A|X 的条件密度估计
    @param df_train: 训练数据，要包含两列：单变量df_train['x'], 连续治疗df_train['a']
    @param df_validation: 调参数据
    @param df_test: 验证数据集
    @param n_grid: 估计 n_grid 个条件密度函数点
    @return: cde: ndarray:(len(df_test['x']),n_grid)
             a_grid: ndarray:(n_grid,1)
    """
    model_flexcode = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine", regression_params={"k": 10})
    # R 中 basis_system 默认是 Fourier
    model_flexcode.fit(df_train['x'].values, df_train['a'].values)

    model_flexcode.tune(df_validation['x'].values, df_validation['a'].values)
    # flexcode_estimate_error = model_flexcode.estimate_error(df_test['x'].values, df_test['a'].values)
    cde, a_grid = model_flexcode.predict(df_test['x'].values, n_grid=n_grid) # 返回 n_grid 个函数点

    test_error = model_flexcode.estimate_error(df_test['x'].values, df_test['a'].values)
    # return cde, a_grid
    return cde, a_grid, test_error


def conditional_density_true(x_matrix, treatment_weights, a_grid):
    """
    @param x_matrix: shape: test_sample_num * covariance_num
    @param treatment_weights: W: A = w * X + epsilon, epsilon ~ N(0, 1)
    @param a_grid: 网格值
    @return: true conditional density at a_grid
    """
    std = 1
    mean = np.dot(x_matrix, treatment_weights)
    cde_true = np.empty(shape=(0, len(a_grid)))
    for u in mean:
        cde_true_u = []
        for a in a_grid:
            # 均值为u的正态分布在取值a处的值
            cde_true_u_a = 1 / (math.sqrt(2 * math.pi) * std) * math.exp(-((a - u) ** 2) / (2 * std ** 2))
            cde_true_u.append(cde_true_u_a)
        cde_true = np.vstack([cde_true, cde_true_u])
    return cde_true

# n = 1000
# df = data_generate(N=n)
# n1 = int(n*0.7)
# n2 = int(n*0.85)
# df_train = df.iloc[:n1, :]
# df_validation = df.iloc[n1:n2, :]
# df_test = df.iloc[n2:, :]
# cde, a_grid, test_error = conditional_density_estimate(df_train,df_validation,df_test,n_grid=1000)
#
#
# for i in range(5):
#     true_density = scipy.stats.norm.pdf(x=a_grid, loc=2*(df_test['x'].values[i]), scale=1)
#     plt.plot(a_grid, cde[i, :],color = "blue")
#     plt.plot(a_grid, true_density, color = "green")
#     plt.axvline(x=2*df_test['x'].values[i], color="yellow")
#     plt.show()
#
