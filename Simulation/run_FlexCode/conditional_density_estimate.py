import pandas as pd
import scipy.stats
import flexcode
from flexcode.regression_models import NN
import matplotlib.pyplot as plt
from Simulation.data_generating.data_generate_process import data_generate
'''
估计的条件密度几乎是零值
'''

def conditional_density_estimate(df_train,df_validation,df_test, n_grid):
    """
    A|X 的条件密度估计
    @param df_train: 训练数据，要包含两列：单变量df_train['x'], 连续治疗df_train['a']
    @param df_validation: 调参数据
    @param df_test: 验证数据集
    @param n_grid: 估计 n_grid 个条件密度函数点
    @return: cde: ndarray:(len(df_test['x']),n_grid)
             a_grid: ndarray:(n_grid,1)
    """
    model_flexcode = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",regression_params={"k":10})
    # R 中 basis_system 默认是 Fourier
    model_flexcode.fit(df_train['x'].values, df_train['a'].values)

    model_flexcode.tune(df_validation['x'].values,df_validation['a'].values)
    # flexcode_estimate_error = model_flexcode.estimate_error(df_test['x'].values, df_test['a'].values)
    cde, a_grid = model_flexcode.predict(df_test['x'].values, n_grid=n_grid) # 返回 n_grid 个函数点

    test_error = model_flexcode.estimate_error(df_test['x'].values, df_test['a'].values)
    # return cde, a_grid
    return cde, a_grid, test_error


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
