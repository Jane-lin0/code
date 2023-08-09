import numpy as np
import pandas as pd
import os
from scipy.stats import expon
from sklearn.model_selection import train_test_split
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from matplotlib import pyplot as plt

from Simulation.data_generating.DGP_pysurvival import SimulationModel
from Simulation.kernel_density_smoothing.density_estimate import density_estimate
from Simulation.kernel_setting import gaussian_kernel
from Simulation.conditional_survival_function.conditional_survival_estimate import conditional_survival_estimate, get_x_y
from Simulation.conditional_density_estimation.conditional_density_estimate import cde_adjust
from Simulation.metrics import mean_squared_error_normalization, integrated_mean_squared_error_normalization
from Simulation.metrics import survival_true, get_best_bandwidth

class CounterfactualSurvFtn(object):
    def __init__(self, surv_distribution='exponential', sample_num=500):
        self.surv_distribution = surv_distribution
        self.sample_num = sample_num
        # self.sample_data = None
        self.train_data = None
        self.test_data = None

    def data_generate(self, test_size=0.2):
        sim = SimulationModel(survival_distribution=self.surv_distribution,
                              risk_type='linear',
                              alpha=1,
                              beta=1
                              )
        dataset = sim.generate_data(num_samples=self.sample_num, num_features=4,
                                    feature_weights=[-2, 1, 2] + [1],  # beta  gamma
                                    treatment_weights=[4, 2, 1])  # W
        dataset.columns = ['x1', 'x2', 'x3', 'a', 'o', 'e', 'lambda']
        self.train_data, self.test_data = train_test_split(dataset, test_size=test_size)
        return self.train_data, self.test_data


    def fit(self):




    def tune(self):


    def predict(self):


    def estimate_error(self):



'''
data
'''
N = 200
survival_distribution = 'exponential'
cv = 5
path = f"C:/Users/janline/Desktop/simulation_data/{N}"

# h = 0.7  # bandwidth 交叉验证选择
# ibs_for_bandwidth = dict()
# cindex_for_bandwidth = dict()
mse_for_h = []
imse_for_h = []

'''
bandwidth choose
'''
h_list = np.logspace(-2, 0, num=20)   # 0.01 至 1 之间的 20 个数
# for h in np.logspace(0.01, 1, 10):  # 100 个 h 运行很久
for h in h_list:
    mse_list = []
    imse_list = []
    for i in range(cv):  # 循环拟合数据集
        df_train = pd.read_excel(path+f"data{i}.xlsx", sheet_name='train')
        df_test = pd.read_excel(path+f"data{i}.xlsx", sheet_name='test')
        df = pd.concat([df_train, df_test], axis=0)

        '''
        conditional_density_estimate，A|X 的条件密度估计 p(a|x) 
        '''
        cde_estimates = pd.read_excel(path+f"CDE{i}.xlsx", sheet_name='Sheet1')
        a_grid = pd.read_excel(path+f"CDE{i}.xlsx", sheet_name='Sheet2')

        # 输出 a_true (用 a_grid 近似) 对应的 cde list
        n_test = len(df_test)
        a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_test)]  # 长度和 df_test 一致
        cde_list = []
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

        '''
        kernel setting, calculate Sa(t)：每个 a_grid 下的生存函数估计
        '''
        # a = 1
        treatment_grid = np.linspace(min(a_approx), max(a_approx), num=n_test)  # 连续 treatment 估计取值的网格点，可调整
        # treatment_grid = a_approx

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

        # MSE 评估生存函数估计
        a_median = np.median(df_test['a'])
        idx = np.argmin(np.abs(treatment_grid - a_median))  # 长度和 df_test 一致
        survival_estimate_a = counterfactual_survival[idx, :].reshape(1, -1)
        survival_true_a = survival_true(treatment_grid[idx], time_grid, df_test)
        mse = mean_squared_error_normalization(survival_estimate_a, survival_true_a, time_grid)
        mse_list.append(mse)

        # IMSE 评估生存函数估计
        survival_true_values = survival_true(treatment_grid, time_grid, df_test)
        imse = integrated_mean_squared_error_normalization(counterfactual_survival, survival_true_values, time_grid)
        imse_list.append(imse)

    mse_for_h.append(np.mean(mse_list))
    imse_for_h.append(np.mean(imse_list))


# plot bandwidth and MSE
plt.figure()
plt.plot(h_list, mse_for_h, marker='o')
plt.xlabel('bandwidth')
plt.ylabel('Mean squared error')
plt.show()

# plot bandwidth and IMSE
plt.figure()
plt.plot(h_list, imse_for_h, marker='o')
plt.xlabel('bandwidth')
plt.ylabel('Integrated mean squared error')
plt.show()

# 计算最佳 bandwidth 下的反事实生存函数估计  # 计算出来用于 test data
h_best_for_mse, min_mse = get_best_bandwidth(error_list=mse_for_h, h_list=h_list)
h_best_for_imse, min_imse = get_best_bandwidth(error_list=imse_for_h, h_list=h_list)
weight = np.empty(shape=(0, n_test))
for a in treatment_grid:
    kernel_values = gaussian_kernel(a_approx, a, h_best_for_mse)
    w_a = pi * kernel_values  # ndarray:(len(df_test),)
    w_normalization = w_a / np.sum(w_a)  # 结果相对正常，含非 0 值
    weight = np.vstack([weight, w_normalization.reshape(1, -1)])
    # final result: weight = { ndarray:(len(treatment_grid), len(df_test))}
counterfactual_survival = weight @ conditional_survival_estimated

# true counterfactual survival
survival_true_values = np.empty(shape=(0, len(time_grid)))
for a in treatment_grid:
    lambda_idx = np.argmin(np.abs(df_test['a'] - a))
    lambda_i = df_test['lambda'][lambda_idx]
    survival_a = []
    for t in time_grid:
        survival_t = 1 - expon.cdf(t, scale=1 / lambda_i)
        survival_a.append(survival_t)
    survival_true_values = np.vstack([survival_true_values, survival_a])  # ndarray:(len(treatment_grid), len(time_grid))


# median potential survival time
# treatment = a 时，Sa(t) = P( T(a) >= t ) = 0.5 时对应的 time_grid
n_treat = len(treatment_grid)
median_survival = []
for i in range(n_treat):
    index = np.argmin(np.abs(counterfactual_survival[i, :] - 0.5))
    median_survival.append(time_grid[index])
median_survival = np.array(median_survival)

# 反事实生存函数估计和真实函数图像对比
# treatment_idx = np.random.randint(low=0, high=len(treatment_grid)-1, size=3)
treatment_idx = equal_space(length=len(treatment_grid), indices_num=5)
colors = ['r', 'g', 'b', 'y', 'pink']
j = 1
for idx, color in zip(treatment_idx, colors):
    survival_est = counterfactual_survival[idx, :]
    plt.step(time_grid, survival_est, where="post", label=f"survival_est_{j}", color=color)
    survival_t = survival_true_values[idx, :]
    plt.step(time_grid, survival_t, where="post", label=f"survival_true_{j}", color=color, linestyle='--')
    j += 1
plt.legend()
plt.xlabel('time')
plt.ylabel('survival probability')

# 将图像保存到本地
desktop = os.path.expanduser("~/Desktop")
filename = f"h{int(100 * h)}.png"
filepath = os.path.join(desktop, filename)
plt.savefig(filepath)
plt.show()





# integrated_brier_score 评估生存函数估计
# x_train, y_train = get_x_y(df_train, col_event='e', col_time='o')
# x_test, y_test = get_x_y(df_test, col_event='e', col_time='o')
# ibs = integrated_brier_score(y_train, y_test, counterfactual_survival, time_grid)  # 并未用到真实生存函数
# ibs_list.append(ibs)

# calculate c_index
# t_approx_index = [np.argmin(np.abs(time_grid - df_test['o'][i])) for i in range(n_test)]  # 长度和 df_test 一致
# survival_est = []
# for x_index, grid_index in enumerate(t_approx_index):
#     val = counterfactual_survival[x_index, grid_index]
#     survival_est.append(val)
# survival_est = np.array(survival_est)
# event = (df_test['e'] == 1).values
# time = df_test['o'].values
# c_index = concordance_index_censored(event, time, estimate=survival_est)
# cindex_list.append(c_index)

# ibs_for_bandwidth[h] = np.mean(ibs_list)
# cindex_for_bandwidth[h] = np.mean(cindex_list)

# h_best, IBS_min = min(ibs_for_bandwidth.items(), key=lambda x: x[1])

# plot bandwidth and c_index
# h_values = list(cindex_for_bandwidth.keys())
# cindex_values = list(cindex_for_bandwidth.values())
# plt.figure()
# plt.plot(h_values, cindex_values, marker='o')
# plt.xlabel('bandwidth')
# plt.ylabel('C index')
# plt.show()

# plot bandwidth and Integrated brier score
# h_values = list(ibs_for_bandwidth.keys())
# ibs_values = list(ibs_for_bandwidth.values())
# plt.figure()
# plt.plot(h_values, ibs_values, marker='o')
# plt.xlabel('bandwidth')
# plt.ylabel('Integrated brier score')
# plt.show()




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