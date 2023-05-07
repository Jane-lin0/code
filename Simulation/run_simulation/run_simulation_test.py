import numpy as np
import pandas as pd
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from matplotlib import pyplot as plt

from Simulation.kernel_density_smoothing.density_estimate import density_estimate
from Simulation.kernel_setting import gaussian_kernel
from Simulation.conditional_survival_function.conditional_survival_estimate import conditional_survival_estimate, get_x_y
from Simulation.conditional_density_estimation.conditional_density_estimate import cde_sample_estimate
from Simulation.metrics import mean_squared_error_normalization, integrated_mean_squared_error_normalization, survival_true

'''
data
'''
N = 1000
cv = 5
path = f"C:/Users/janline/Desktop/simulation_data/{N}"

h = 0.7  # bandwidth 交叉验证选择
# ibs_for_bandwidth = dict()
# cindex_for_bandwidth = dict()
# for h in np.logspace(0.01, 1, 10):  # 100 个 h 运行很久
# for i in range(cv):  # 循环拟合数据集
i = 0
df_train = pd.read_excel(path + f"data{i}.xlsx", sheet_name='train')
df_test = pd.read_excel(path + f"data{i}.xlsx", sheet_name='test')
df = pd.concat([df_train, df_test], axis=0)

'''
conditional_density_estimate，A|X 的条件密度估计 p(a|x)  
'''
cde_estimates = pd.read_excel(path + f"CDE{i}.xlsx", sheet_name='Sheet1')  # a_grid 上的 cde
a_grid = pd.read_excel(path + f"CDE{i}.xlsx", sheet_name='Sheet2')

n_test = len(df_test)
a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_test)]  # 长度和 df_test 一致
cde = cde_sample_estimate(cde_estimates, a_approx_index)  # 输出 a_true (用 a_grid 近似) 对应的 cde

'''
estimate density function of A: p(a), by kernel density smoothing
'''
a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index]).reshape(-1, 1)
density_estimated = density_estimate(df[['a']], a_approx)   # df[['a']] 拟合模型，返回 a_approx 上的密度估计

'''
pai(ai,xi) = p(a) / p(a|x)
'''
pi = density_estimated / cde  # ndarray:(len(df_test),)
# 取与 a_true 最近的 a_grid (即 a_approx) 进行估计

'''
conditional survival function estimate S(t|A,X)
'''
# time_grid = np.linspace(start=min(df_test['o']), stop=max(df_test['o']), num=500)  # time_grid 设置可调整
# time_grid = df_test['o']
time_grid = np.linspace(start=min(df_train['o']), stop=max(df_train['o']), num=500)
conditional_survival_estimated = conditional_survival_estimate(df_train, df_test, time_grid)  # 未调参
# ndarray:(len(df_test), len(time_grid))

# A 类似于协变量 X ？

'''
kernel setting, calculate Sa(t)：每个 a_grid 下的生存函数估计
'''
# a = 1
# treatment_grid = np.linspace(min(a_approx), max(a_approx), num=n_test)  # 连续 treatment 估计取值的网格点，可调整
treatment_grid = a_approx

# 根据权重计算反事实生存函数
weight = np.empty(shape=(0, n_test))
for a in treatment_grid:
    kernel_values = gaussian_kernel(a_approx, a, h)
    w_a = pi * kernel_values  # ndarray:(len(df_test),)
    w_normalization = w_a / np.sum(w_a)  # 结果相对正常，含非 0 值
    weight = np.vstack([weight, w_normalization.reshape(1, -1)])
    # final result: weight = { ndarray:(len(treatment_grid), len(df_test))}

counterfactual_survival = weight @ conditional_survival_estimated
# ndarray:(len(treatment_grid), len(time_grid))

# validation set: MSE 评估 treatment 中位数的生存函数估计
a_median = np.median(df_test['a'])
idx = np.argmin(np.abs(treatment_grid - a_median))   # 长度和 df_test 一致
survival_estimate_a = counterfactual_survival[idx, :].reshape(1, -1)
survival_true_a = survival_true(treatment_grid[idx], time_grid, df_test)
mse = mean_squared_error_normalization(survival_estimate_a, survival_true_a, time_grid)

# validation set: IMSE 评估生存函数估计
survival_true_values = survival_true(treatment_grid, time_grid, df_test)
imse = integrated_mean_squared_error_normalization(counterfactual_survival, survival_true_values, time_grid)

# h_best, IBS_min = min(ibs_for_bandwidth.items(), key=lambda x: x[1])

# 基于 h_best 和 tuned_parameters 对 test set 评估

# 计算最佳 bandwidth 下的反事实生存函数估计
df_test = pd.read_excel(path + f"data{i}.xlsx", sheet_name='test')
n_test = len(df_test)
weight = np.empty(shape=(0, n_test))
for a in treatment_grid:
    kernel_values = gaussian_kernel(a_approx, a, h_best)
    w_a = pi * kernel_values  # ndarray:(len(df_test),)
    w_normalization = w_a / np.sum(w_a)  # 结果相对正常，含非 0 值
    weight = np.vstack([weight, w_normalization.reshape(1, -1)])
    # final result: weight = { ndarray:(len(treatment_grid), len(df_test))}
counterfactual_survival = weight @ conditional_survival_estimated

# true counterfactual survival
true_survival = survival_true(treatment_grid, time_grid, df_test)


# median potential survival time
# treatment = a 时，Sa(t) = P( T(a) >= t ) = 0.5 时对应的 time_grid
n_treat = len(treatment_grid)
median_survival = []
for i in range(n_treat):
    index = np.argmin(np.abs(counterfactual_survival[i, :] - 0.5))
    median_survival.append(time_grid[index])
median_survival = np.array(median_survival)

# 反事实生存函数估计和真实函数图像对比
treatment_idx = np.random.randint(low=0, high=len(treatment_grid) - 1, size=3)
colors = ['r', 'g', 'b']
for idx, color in zip(treatment_idx, colors):
    survival_est = counterfactual_survival[idx, :]
    plt.step(time_grid, survival_est, where="post", label=f"survival_est_{idx}", color=color)
    survival_true = true_survival[idx, :]
    plt.step(time_grid, survival_true, where="post", label=f"survival_true_{idx}", color=color, linestyle='--')
plt.legend()
plt.show()












