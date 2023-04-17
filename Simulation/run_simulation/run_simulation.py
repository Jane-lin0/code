import numpy as np
import pandas as pd

from Simulation.data_generating.data_generate_process import data_generate
from Simulation.run_FlexCode.conditional_density_estimate import conditional_density_estimate
from Simulation.density_estimate import density_estimate
from Simulation.kernel_setting import gaussian_kernel
from Simulation.conditional_survival_function.conditional_survival_estimate import conditional_survival_estimate
import xlsxwriter

'''
data_generate
'''
df_train = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='train')
df_validation = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='validation')
df_test = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='test')

df = pd.concat([df_train, df_validation, df_test], axis=0)

'''
conditional_density_estimate，A|X 的条件密度估计
'''
# conditional_density_estimated, a_grid = conditional_density_estimate(df_train, df_validation, df_test, n_grid=1000)
# n_grid 等于 len(df_test)？可以大于
cde_estimates = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='sheet1')
a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='sheet2')

# 输出 a_true 对应的 cde list
n_obs = len(df_test)
nns = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]
cde_list =[]
for x_index,grid_index in enumerate(nns):
    cde_list.append(cde_estimates.iloc[x_index, grid_index])

'''
estimate density function of A by kernel density smoothing
'''
density_estimated = density_estimate(df[['a']], a_grid)
# df[['a']] 拟合模型，返回 a_grid 上的密度估计

'''
pai(ai,xi) = p(a) / p(a|x)
'''
pi = density_estimated / conditional_density_estimated

# pi_diag = np.diag(pi)  # len(df_test)
# a_grid 取的不一定是第 i 个样本对应的 ai，在计算 pi 和 kernel 时取的是 a_grid，如何解决？取和 a_true 最近的 a_grid 进行估计
# a_grid 取的点较为密集时，是否可以忽略这个问题？

'''
estimate S(t|A,X)
'''
survival_estimated, time_grid = conditional_survival_estimate(df_test['o'], df_test['e'])  # 根据 df['o'] 生成的 time_grid
# A 的信息包含在 o 中，模型拟合并未用到 A 的数据
# survival_estimated 根据时间排序，df_test 也是按 时间排序的（因为 df_test 截取自 df)，因此 survival_estimated 的顺序也对应 A 和 X 的顺序

'''
kernel setting
'''
a = 1  # a 如何设定？
h = 0.7
kernel_val = [gaussian_kernel(ai, a, h) for ai in a_grid]

'''
calculate Sa(t)
'''
