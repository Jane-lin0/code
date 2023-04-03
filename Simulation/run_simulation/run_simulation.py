import numpy as np
from data_generate_process import data_generate
from conditional_density_estimate import conditional_density_estimate
from density_estimate import density_estimate
from kernel_setting import gaussian_kernel
from conditional_survival_estimate import conditional_survival_estimate

'''
data_generate
'''
df = data_generate(n=3000)

'''
conditional_density_estimate，A|X 的条件密度估计
'''
df_train = df.iloc[:1000, :]
df_validation = df.iloc[1000:2000, :]
df_test = df.iloc[2000:, :]
conditional_density_estimated, a_grid = conditional_density_estimate(df_train,df_validation,df_test,n_grid=1000)
# n_grid 等于 len(df_test)？需进一步验证大于是否会报错

'''
estimate density function of A by kernel density smoothing
'''
density_estimated = density_estimate(df[['a']], a_grid)
# df[['a']] 拟合模型，返回 a_grid 上的密度估计

'''
pai(ai,xi) = p(a) / p(a|x)
'''
pi = density_estimated / conditional_density_estimated

pi_diag = np.diag(pi)  # len(df_test)
# a_grid 取的不一定是第 i 个样本对应的 ai，在计算 pi 和 kernel 时取的是 a_grid，如何解决？
# a_grid 取的点较为密集时，是否可以忽略这个问题？

'''
estimate S(t|A,X)
'''
survival_estimated, time_grid = conditional_survival_estimate(df_test['t'], df_test['e']) # 根据 df['t'] 生成的 time_grid
# A 的信息包含在 t 中，模型拟合并未用到 A 的数据
# survival_estimated 根据时间排序，df_test 也是按 时间 t 排序的（因为 df_test 截取自 df)，因此 survival_estimated 的顺序也对应 A 和 X 的顺序

'''
kernel setting
'''
a = 1           # a 如何设定？
h = 0.7
kernel_val = [gaussian_kernel(ai, a, h) for ai in a_grid]

'''
calculate Sa(t)
'''


