import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# from Simulation.data_generating.data_generate_process import data_generate
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.metrics import concordance_index_censored
import rpy2.robjects as robjects
import rpy2
import os
from sklearn.metrics import mean_squared_error
from scipy.integrate import dblquad
from scipy.integrate import nquad


# survival_est = np.arange(10).reshape(2, -1)
# survival_true = np.arange(10).reshape(2, -1)
# grid = np.arange(5)
# # grid = np.tile(grid.flatten(), len(survival_est))
# # survival_est = survival_est.flatten()
# # survival_true = survival_true.flatten()
# #
# # term1 = np.trapz(survival_est**2, grid)
# # term2 = np.trapz(survival_est * survival_true, grid)
# # mise = term1 - 2 * term2
# m = integrated_mean_squared_error(survival_est, survival_true, grid)

# grid = np.arange(5)
# grid = np.tile(grid.flatten(), 3)

# a = np.arange(5)
# b = np.tile(a, 3)
# b = a.item()
# .reshape(2,-1)
# b = a.flatten()

# # 定义被积函数
# def func(x, y):
#     return np.exp(-x*y)
#
# # 定义积分区间
# x_range = [0, 1]
# y_range = [lambda x: 0, lambda x: 1 - x]
#
# # 计算积分
# result, error = nquad(func, [x_range, y_range])
#
# print("结果:", result)
# print("误差:", error)

# # 定义被积函数
# def func(x, y):
#     return x+y
#
# # 定义积分区间和积分区域
# a, b = 0, 1
# low_fun = lambda x: 0
# upper_fun = lambda x: 1 - x
#
# # 计算积分
# result, error = dblquad(func, a, b, low_fun, upper_fun)
#
# print("结果:", result)
# print("误差:", error)


# a=np.trapz([1,2,3], x=[4,6,8])

# y_true = [[0.5, 1],[-1, 1],[7, -6]]
# y_pred = [[0, 2],[-1, 2],[8, -5]]
# mse = mean_squared_error(y_true, y_pred) # 0.708
# MSE =((0.5 - 0)**2 + (1 - 2)**2 + (-1 + 1)**2 + (-1 + 2)**2 + (7 - 8)**2 + (-6 + 5)**2)/6  # 0.70833


# os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.0'
#
#
# # rpy2.rinterface.set_R_HOME('/path/to/R')
#
# print(rpy2.__version__) # 3.5.11
# import rpy2.situation as rpy2situation
# # print(rpy2situation.get_r_version())
# # import rpy2.situation as sit
# #
# # print(sit.get_r_home())
# # print(sit.get_rversion())
#
#
# # os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.0'  # 根据实际安装路径进行修改
# # # 执行一条R语言命令
# result = robjects.r('paste("Hello", "world!")')
# print(result[0])

# N = 1000
# path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')
# df_test = pd.read_excel(path+"data.xlsx",sheet_name='test')
# df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
#
# a_median = np.median(df_test['a'])
#
# T = df['o']
# treatment_col = df['a']
# num_samples = len(df)
# C = np.random.exponential(scale=np.mean(T) + np.std(T), size=num_samples)     # 删失时间服从指数分布
# C1 = np.random.uniform(low=0, high=treatment_col)

# c = np.array([[4], [5], [1]])
# b = c.flatten()

# treatment_col = np.random.binomial(10, 0.5, size=100)
# C = np.random.uniform(low=0, high=treatment_col)

# a = np.logspace(0.01, 1, 10)
# b = np.logspace(-2, 0, num=10)  # 带宽取值范围在0.01到1之间
# estimator = CoxPHSurvivalAnalysis()
# estimator.fit(data_x_numeric, data_y)
# prediction = estimator.predict(data_x_numeric)
# result = concordance_index_censored(data_y["Status"], data_y["Survival_in_days"], prediction)
# c_index = result[0]

# lis = [1,2,3,4,5]
# a = np.mean(lis)

# cv = 5
# for i in range(cv):
#     print(i+1)

# train_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# train_event = np.array([1, 1, 0, 1, 0])
# test_time = np.array([6.0, 7.0, 8.0])
# # 检查测试集时间是否超出最大观测时间点
# extends = test_time > np.max(train_time)
# if np.any(extends):
#     # 将超出的时间值替换为最大观测时间点
#     test_time = np.where(extends, np.max(train_time), test_time)  # 直接替换


# N = 1000
# path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')
# df_test = pd.read_excel(path+"data.xlsx",sheet_name='test')
# df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
#
# T = df['o']
# treatment_col = df['a']
# num_samples = len(df)
# C = np.random.exponential(scale=np.mean(T) + np.std(T), size=num_samples)     # 删失时间服从指数分布
# C1 = np.random.uniform(low=0, high=treatment_col)

#
# # 检查测试集时间是否超出最大观测时间点
# extends = df_test['o'] > np.max(df_train['o'])
# if np.any(extends):
#     # 将超出的时间值替换为最大观测时间点
#     test_time = np.where(extends, np.max(df_train['o']), df_test['o'])


#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 随机抽取索引
# # 进行交叉验证
# for train_index, test_index in kf.split(df):
#     df_val = df.loc[test_index]
#     # print(test_index, "="*50)

# my_dict = {'a': 3, 'b': 2, 'c': 1}
# k, min_value = min(my_dict.items(), key=lambda x: x[1])
#
# x = my_dict.keys()
# y = my_dict.values()
#
# plt.figure()
# plt.plot(x, y, marker='o')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# a = np.random.choice(250)

# df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, index=[1,3,5])
# idx = df['col1'].idxmax()
# row_to_drop = df.loc[idx]
# df_temp = row_to_drop.to_frame().T


# df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, index=[1,3,5]).reset_index(drop=True)
# idx = np.argmin(df['col1'])
# row_to_drop = df.loc[idx, :]
# df = df.drop(idx)

# test = np.random.randint(1,10,size=5)
# idx = np.argmin(test)

# time_points = np.arange(1, 1000)

# N = 1000
# path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# df_test = pd.read_excel(path+"data.xlsx",sheet_name='test')
# df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')
#
# # idx_test = np.random.choice(df_test.index, size=1).item()
# # idx_test = np.random.choice(df_test.index, size=1)
# # row_to_train = df_test.loc[idx_test]
#
# print(df_test.describe())
# print(df_train.describe())
# print(type(df_test['o']))
# print(df_test['o'].min())
# print(type(df_test['o'].min()))
# print(df_test['o'].min() < df_test['o'].max())

# treatment_idx = np.random.randint(low=0, high=10, size=3)
# colors = ['r', 'g', 'b']
# for idx, color in zip(treatment_idx, colors):
#     print(idx, color)

# estimator = CoxPHSurvivalAnalysis()
# estimator.fit(data_x_numeric, data_y)
# estimator.score()
#
# # Estimate the survival function
# survival = estimator.predict_survival_function(data[['time']])
#
# # Calculate the integrated Brier score
# ibs = integrated_brier_score(survival, data['status'], data['time'], t_max=100)



# N = 100
# path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# data_generate(N, path)

# path = "C:/Users/janline/Desktop/"
# df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')

# list_t = [1,2,3,4,5,6]
#
# weight = np.empty(shape=(0,6))
# # list_normalization = np.array(list_t) / sum(list_t)
# # weight = np.vstack([weight0,list_normalization])
# weight = np.vstack([weight,list_t])
#
# list_t1 = [1,2,3,4,5,5]
# # list_normalization1 = np.array(list_t1) / sum(list_t1)
# # weight = np.vstack([weight, list_normalization1])
# weight = np.vstack([weight,list_t1])
#
# n = weight.shape[0]
# res = []
# for i in range(n):
#     index = np.argmin(np.abs(weight[i, :] - 3))
#     res.append(weight[i, index])


# arr = np.array(list_t).reshape(2,3)
# arr_T = arr.T
# multiply = arr_T @ arr

# time = 1.02
# treat = 3.766
# conditional_survival_est = conditional_survival_estimated[:, col]  # ndarray:(150,)
# kernel_val = gaussian_kernel(a_approx, treat, h)  # ndarray:(150,1), list 150

# fenzi = 0
# for i in [0]:
#     print(pi[i] , conditional_survival_est[i] , kernel_val[i],
#           pi[i] * conditional_survival_est[i] * kernel_val[i])

# total = pi * conditional_survival_est * kernel_val
# fenzi = np.sum(total)
#
# total1 = pi * kernel_val
# fenmu = np.sum(total1)

# survival_est = np.sum(pi * conditional_survival_est * kernel_val)/np.sum(pi * kernel_val)
# survival_estimates.append(survival_est)

# a = np.array([1,2,3]).reshape(-1,1)
# b = a
# c = a*b
# d = a*a*a
# e = sum(d)

# df_test = pd.read_excel("C:/Users/janline/Desktop/simulation_data/data.xlsx",sheet_name='test')
# a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet2')
#
#
# n_obs = len(df_test)
# a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]  # 长度和 df_test 一致
# a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index]).reshape(-1,1)
# # a_approx = [a_grid.loc[i].item() for i in a_approx_index]
#
# treatment_grid = np.linspace(min(a_approx), max(a_approx), num=100)  # 连续 treatment 取值网格点
# for treat in treatment_grid:
#     print(treat)

# a = [-1,1,2]+[1]

# df_train = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='train')
# df_validation = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='validation')
# df_test = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='test')
#
# df = pd.concat([df_train, df_validation, df_test], axis=0)
#
# cde_estimates = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet1')
# a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet2')
#
# n_obs = len(df_test)
# a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]  # 长度和 df_test 一致
# a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index])




#
# n_obs = len(df_test)
# nns = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]
# cde_list =[]
# for x_index,grid_index in enumerate(nns):
#     cde_list.append(cde_estimates.iloc[x_index, grid_index])
#
# # cde_list = [cde_estimates[range(n_obs), nns]]
#
# # num_samples = 1000
# # da = np.random.randn(num_samples).reshape(-1,1)