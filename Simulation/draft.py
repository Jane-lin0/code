import numpy as np
import pandas as pd


# list_t = [1,2,3,4,5,6]
#
# weight0 = np.empty(shape=(0,6))
# list_normalization = np.array(list_t) / sum(list_t)
# weight = np.vstack([weight0,list_normalization])
#
# list_t1 = [1,2,3,4,5,5]
# list_normalization1 = np.array(list_t1) / sum(list_t1)
# weight = np.vstack([weight, list_normalization1])
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