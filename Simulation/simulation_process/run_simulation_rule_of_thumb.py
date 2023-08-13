import numpy as np
import pandas as pd
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from matplotlib import pyplot as plt
import os
from tabulate import tabulate
from Simulation.kernel_density_smoothing.density_estimate import density_estimate
from Simulation.kernel_setting import gaussian_kernel
from Simulation.conditional_survival_function.conditional_survival_estimate import conditional_survival_estimate, get_x_y
from Simulation.conditional_density_estimation.conditional_density_estimate import cde_sample_estimate
from Simulation.metrics import mean_squared_error_normalization, integrated_mean_squared_error_normalization
from Simulation.metrics import survival_true, get_best_bandwidth, median_survival_time
from Simulation.ouput import print_latex, subset_index, subset, equal_space

'''
data
'''
N = 1000
# cv = 5
path = f"C:/Users/janline/Desktop/simulation_data/{N}"

# h = 1  # bandwidth 交叉验证选择
# h = 0.75  # bandwidth 交叉验证选择
# h = 0.5  # bandwidth 交叉验证选择
# h = 0.25  # bandwidth 交叉验证选择
# ibs_for_bandwidth = dict()
# cindex_for_bandwidth = dict()
# for h in np.logspace(0.01, 1, 10):  # 100 个 h 运行很久
# for i in range(cv):  # 循环拟合数据集
mse_for_h = []
imse_for_h = []
summary_median_survival_pred = np.empty(shape=(0, 200))
summary_median_survival_true = np.empty(shape=(0, 200))
h_list = [1, 0.75, 0.5, 0.25]
for h in h_list:
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
    time_grid = np.linspace(start=min(df_train['o']), stop=max(df_train['o']), num=500)  # 便于输出表格
    conditional_survival_estimated = conditional_survival_estimate(df_train, df_test, time_grid)  # 未调参
    # ndarray:(len(df_test), len(time_grid))

    # A 类似于协变量 X ？

    '''
    kernel setting, calculate Sa(t)：每个 a_grid 下的生存函数估计
    '''
    # a = 1
    treatment_grid = np.linspace(min(a_approx), max(a_approx), num=n_test)  # 连续 treatment 估计取值的网格点，可调整
    # treatment_grid = a_approx  # 不是从小到大排序的，不便于对比输出的反事实结果

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
    mse_for_h.append(mse)

    # validation set: IMSE 评估生存函数估计
    survival_true_values = survival_true(treatment_grid, time_grid, df_test['a'], df_test['lambda'])
    imse = integrated_mean_squared_error_normalization(counterfactual_survival, survival_true_values, time_grid)
    imse_for_h.append(imse)

    # 不同治疗下的 median potential survival time 估计
    median_survival_pred = median_survival_time(counterfactual_survival, time_grid)
    # median_survival_pred = median_survival_time(counterfactual_survival, treatment_grid, time_grid)
    summary_median_survival_pred = np.vstack((summary_median_survival_pred, median_survival_pred))

    # 不同治疗下的 median potential survival time 真实值：Sa(t) = P( T(a) >= t ) = 0.5 时对应的 time_grid
    median_survival_true = median_survival_time(survival_true_values, time_grid)
    # median_survival_true = median_survival_time(survival_true_values, treatment_grid, time_grid)
    summary_median_survival_true = np.vstack((summary_median_survival_true, median_survival_true))

    '''
    # 等间隔抽取20行5列，输出为 latex 结果
    '''
    row_index, col_index = subset_index(counterfactual_survival.shape, row_num=20, col_num=5)

    counterfactual_survival_output = subset(counterfactual_survival, row_index, col_index)
    survival_true_values_output = subset(survival_true_values, row_index, col_index)

    summary_survival_output = np.hstack((treatment_grid[row_index], counterfactual_survival_output, survival_true_values_output))
    print_latex(summary_survival_output)

    # print_latex(counterfactual_survival_output)
    # table_counterfactual_survival = tabulate(counterfactual_survival_output, tablefmt="latex", floatfmt=".4f") # 输出保留4位小数
    # table_counterfactual_survival = f"\\label{int(h*100)}\n{table_counterfactual_survival}\n"  # 加 label
    # print(table_counterfactual_survival, "=" * 100)  # 打印结果，复制粘贴到latex

    # print_latex(survival_true_values_output)
    # table_survival_true = tabulate(survival_true_values_output, tablefmt="latex", floatfmt=".4f")
    # table_survival_true = f"\\label{int(h*100)}\n{table_survival_true}\n"
    # print(table_survival_true, "=" * 100)

    # 反事实生存函数估计和真实函数图像对比
    # treatment_idx = np.random.randint(low=0, high=len(treatment_grid) - 1, size=5)
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


# （不同治疗下的）中位生存时间在不同bandwidth下的对比
summary_median_survival_output = np.vstack((treatment_grid.T, summary_median_survival_pred, summary_median_survival_true)).T[row_index]
# table_median_survival = tabulate(summary_median_survival_output, tablefmt="latex", floatfmt=".4f")

# summary_median_survival_true_output = np.vstack((treatment_grid.T, summary_median_survival_true)).T[row_index]
# table_median_survival_true = tabulate(summary_median_survival_true_output, tablefmt="latex", floatfmt=".4f")

# mse 和 imse 值在不同bandwidth下的对比
summary_arr = np.array([h_list, mse_for_h, imse_for_h])
summary_mse_and_imse = pd.DataFrame(summary_arr, index=['h_opt', 'MSE', 'IMSE'])
# table_error_summary = tabulate(summary_output, tablefmt="latex", floatfmt=".4f")

for ndarray in [summary_median_survival_output, summary_mse_and_imse]:
    print_latex(ndarray)

# h_best, IBS_min = min(ibs_for_bandwidth.items(), key=lambda x: x[1])

# 基于 h_best 和 tuned_parameters 对 test set 评估

# # 计算最佳 bandwidth 下的反事实生存函数估计
# df_test = pd.read_excel(path + f"data{i}.xlsx", sheet_name='test')
# n_test = len(df_test)
# weight = np.empty(shape=(0, n_test))
# for a in treatment_grid:
#     kernel_values = gaussian_kernel(a_approx, a, h_best)
#     w_a = pi * kernel_values  # ndarray:(len(df_test),)
#     w_normalization = w_a / np.sum(w_a)  # 结果相对正常，含非 0 值
#     weight = np.vstack([weight, w_normalization.reshape(1, -1)])
#     # final result: weight = { ndarray:(len(treatment_grid), len(df_test))}
# counterfactual_survival = weight @ conditional_survival_estimated
#
# # true counterfactual survival
# true_survival = survival_true(treatment_grid, time_grid, df_test)












