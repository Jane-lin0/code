import numpy as np
import pandas as pd

from Simulation.kernel_density_smoothing.density_estimate import density_estimate
from Simulation.kernel_setting import gaussian_kernel
from Simulation.conditional_survival_function.conditional_survival_estimate import conditional_survival_estimate

'''
data
'''
df_train = pd.read_excel("C:/Users/janline/Desktop/simulation_data/data.xlsx",sheet_name='train')
df_test = pd.read_excel("C:/Users/janline/Desktop/simulation_data/data.xlsx",sheet_name='test')
df = pd.concat([df_train, df_test], axis=0)

'''
conditional_density_estimate，A|X 的条件密度估计 p(a|x)
'''
# conditional_density_estimated, a_grid = conditional_density_estimate(df_train, df_validation, df_test, n_grid=1000)
# n_grid 等于 len(df_test)？可以大于
cde_estimates = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet1')
a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet2')

# 输出 a_true (用 a_grid 近似) 对应的 cde list
n_obs = len(df_test)
a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]  # 长度和 df_test 一致
cde_list =[]
for x_index,grid_index in enumerate(a_approx_index):
    cde_list.append(cde_estimates.iloc[x_index, grid_index])

'''
estimate density function of A: p(a), by kernel density smoothing
'''
a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index]).reshape(-1,1)
density_estimated = density_estimate(df[['a']], a_approx)
# df[['a']] 拟合模型，返回 a_approx 上的密度估计

# density_estimated = density_estimate(df[['a']], a_grid)
# # df[['a']] 拟合模型，返回 a_grid 上的密度估计

'''
pai(ai,xi) = p(a) / p(a|x)
'''
pi = density_estimated / cde_list
# pi_diag = np.diag(pi)  # len(df_test)
# a_grid 取的不一定是第 i 个样本对应的 ai，在计算 pi 和 kernel 时取的是 a_grid，如何解决？
# 取与 a_true 最近的 a_grid 进行估计

'''
conditional survival function estimate S(t|A,X)
'''
conditional_survival_estimated, time_grid = conditional_survival_estimate(df_train, df_test)  # 根据 df['o'] 生成的 time_grid


# A 类似于协变量 X

'''
kernel setting
'''
a = 1  # a 如何设定？
h = 0.7
kernel_val = [gaussian_kernel(ai, a, h) for ai in a_grid]

'''
calculate Sa(t)
'''
