import numpy as np
import pandas as pd
from scipy.stats import expon
from sksurv.metrics import integrated_brier_score
from matplotlib import pyplot as plt

from Simulation.kernel_density_smoothing.density_estimate import density_estimate
from Simulation.kernel_setting import gaussian_kernel
from Simulation.conditional_survival_function.conditional_survival_estimate import conditional_survival_estimate,get_x_y
from Simulation.run_FlexCode.conditional_density_estimate import cde_adjust

'''
data
'''
N = 1000
path = f"C:/Users/janline/Desktop/simulation_data/{N}"
df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')
df_test = pd.read_excel(path+"data.xlsx",sheet_name='test')
df = pd.concat([df_train, df_test], axis=0)

'''
conditional_density_estimate，A|X 的条件密度估计 p(a|x)
'''
# conditional_density_estimated, a_grid = conditional_density_estimate(df_train, df_validation, df_test, n_grid=1000)
# n_grid 等于 len(df_test)？可以大于
cde_estimates = pd.read_excel(path+"CDE.xlsx",sheet_name='Sheet1')
a_grid = pd.read_excel(path+"CDE.xlsx",sheet_name='Sheet2')

# 输出 a_true (用 a_grid 近似) 对应的 cde list
n_test = len(df_test)
a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_test)]  # 长度和 df_test 一致
cde_list = []  # 0 值如何处理？调参
for x_index, grid_index in enumerate(a_approx_index):
    cde_val = cde_estimates.iloc[x_index, grid_index]
    cde_list.append(cde_val)

cde = cde_adjust(cde_list)  # 给 cde_list 的零值加上一个很小的值，避免求 pi 时除以 0 得到 inf

'''
estimate density function of A: p(a), by kernel density smoothing
'''
a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index]).reshape(-1, 1)
density_estimated = density_estimate(df[['a']], a_approx)
# df[['a']] 拟合模型，返回 a_approx 上的密度估计

# density_estimated = density_estimate(df[['a']], a_grid)
# # df[['a']] 拟合模型，返回 a_grid 上的密度估计

'''
pai(ai,xi) = p(a) / p(a|x)
'''
pi = density_estimated / cde  # ndarray:(len(df_test),)
# pi_diag = np.diag(pi)  # len(df_test)
# a_grid 取的不一定是第 i 个样本对应的 ai，在计算 pi 和 kernel 时取的是 a_grid，如何解决？取与 a_true 最近的 a_grid (即 a_approx) 进行估计

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
# 转化为权重计算

# h = 0.7  # 交叉验证选择
IBS_for_bandwidth = dict()
for h in np.logspace(0.01, 1, 100):
    weight = np.empty(shape=(0, n_test))
    for a in treatment_grid:
        kernel_values = gaussian_kernel(a_approx, a, h)
        w_a = pi * kernel_values  # ndarray:(len(df_test),)
        w_normalization = w_a / np.sum(w_a)  # 结果相对正常，含非 0 值
        weight = np.vstack([weight, w_normalization.reshape(1, -1)])
        # final result: weight = { ndarray:(len(treatment_grid), len(df_test))}

    counterfactual_survival = weight @ conditional_survival_estimated
    # ndarray:(len(treatment_grid), len(time_grid))

    # integrated_brier_score 评估生存函数估计
    x_train, y_train = get_x_y(df_train, col_event='e', col_time='o')
    x_test, y_test = get_x_y(df_test, col_event='e', col_time='o')
    IBS = integrated_brier_score(y_train, y_test, counterfactual_survival, time_grid)  # 并未用到真实生存函数
    IBS_for_bandwidth[h] = IBS


# plot bandwidth-Integrated brier score
h_values = list(IBS_for_bandwidth.keys())
ibs_values = list(IBS_for_bandwidth.values())
plt.figure()
plt.plot(h_values, ibs_values)
plt.xlabel('bandwidth')
plt.ylabel('Integrated brier score')
plt.show()


# 计算最佳 bandwidth 下的反事实生存函数估计
h_best, IBS_min = min(IBS_for_bandwidth.items(), key=lambda x: x[1])
weight = np.empty(shape=(0, n_test))
for a in treatment_grid:
    kernel_values = gaussian_kernel(a_approx, a, h_best)
    w_a = pi * kernel_values  # ndarray:(len(df_test),)
    w_normalization = w_a / np.sum(w_a)  # 结果相对正常，含非 0 值
    weight = np.vstack([weight, w_normalization.reshape(1, -1)])
    # final result: weight = { ndarray:(len(treatment_grid), len(df_test))}
counterfactual_survival = weight @ conditional_survival_estimated


# true counterfactual survival
true_survival = np.empty(shape=(0, len(time_grid)))
for a in treatment_grid:
    lambda_idx = np.argmin(np.abs(df_test['a'] - a))
    lambda_i = df_test['lambda'][lambda_idx]
    survival_a = []
    for t in time_grid:
        survival_t = 1 - expon.cdf(t, scale=1 / lambda_i)
        survival_a.append(survival_t)
    true_survival = np.vstack([true_survival, survival_a])  # ndarray:(len(treatment_grid), len(time_grid))


# median potential survival time
# treatment = a 时，Sa(t) = P( T(a) >= t ) = 0.5 时对应的 time_grid
n_treat = len(treatment_grid)
median_survival = []
for i in range(n_treat):
    index = np.argmin(np.abs(counterfactual_survival[i, :] - 0.5))
    median_survival.append(time_grid[index])
median_survival = np.array(median_survival)


# 反事实生存函数估计和真实函数图像对比
treatment_idx = np.random.randint(low=0, high=len(treatment_grid)-1, size=3)
colors = ['r', 'g', 'b']
for idx, color in zip(treatment_idx, colors):
    survival_est = counterfactual_survival[idx, :]
    plt.step(time_grid, survival_est, where="post", label=f"survival_est_{idx}", color=color)
    survival_true = true_survival[idx, :]
    plt.step(time_grid, survival_true, where="post", label=f"survival_true_{idx}", color=color, linestyle='--')
plt.legend()
plt.show()


# a = 3.766
# kernel_values = gaussian_kernel(a_approx, a, h)
# w_a = pi * kernel_values
# w_normalization = w_a / np.sum(w_a)  # 结果相对正常，含非 0 值
# pi 存在 inf 时，w 全部结果都是 0

# 计算 survival_est 的分子分母都是 inf， survival_est = nan
# survival_estimates = []
# col = 0
# for time in time_grid:
#     conditional_survival_est = conditional_survival_estimated[:, col]
#     for treat in treatment_grid:
#         kernel_val = gaussian_kernel(a_approx, treat, h)
#         survival_est = np.sum(pi * conditional_survival_est * kernel_val)/np.sum(pi * kernel_val)
#         survival_estimates.append(survival_est)
#     col += 1
# survival_estimates = np.array(survival_estimates).reshape(len(time_grid), len(treatment_grid))










