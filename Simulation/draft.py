import numpy as np
import pandas as pd


df_train = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='train')
df_validation = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='validation')
df_test = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='test')

df = pd.concat([df_train, df_validation, df_test], axis=0)

cde_estimates = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet1')
a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet2')

n_obs = len(df_test)
a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]  # 长度和 df_test 一致
a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index])




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