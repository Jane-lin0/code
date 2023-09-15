import time

import numpy as np
import pandas as pd
from Simulation.conditional_density_estimation.conditional_density_estimate import conditional_density_true
from Simulation.metrics import integrated_mean_squared_error, integrated_mean_squared_error_normalization
from Simulation.simulation_process.simulation_main import CounterfactualSurvFtn
from Simulation.conditional_density_estimation.Flexcode_rpy2 import run_flexcode_empirical

start_time = time.time()

'''========== ========== 参数修改 ========== ========== '''
# n_list = [200, 400, 800]
n_list = [200, 400, 600, 800, 1000]
# sample_list = [200, 400, 800]   # 200-700，间隔100
# sample_list = [200]
'''！！！！！记得修改 run_flexcode 函数的样本数 N ！！！！！'''
bandwidth = 0.25
# bandwidth_list = np.array([0.25, 0.5, 0.75, 1])
# bandwidth_list = np.logspace(-2, 0, num=15)  # 0.001 至 1 之间的 10 个数
cv = 5
survival_distribution = 'exponential'
test_size = 0.15
treatment_weights = [4, 2, 1]
# validation_evaluation_method = 'rmse'
simulation_times = 200  # 30
'''========== ========== ========== ========== ========== '''

imse_list = []
for n in n_list:
    path = fr"C:\Users\janline\OneDrive - stu.xmu.edu.cn\学校\论文\论文代码\simulation_data\test\{n}"
    imse_for_n = []
    for i in range(simulation_times):
        model = CounterfactualSurvFtn(path=path, cv=cv)

        # 数据生成
        model.data_generate_empirical(sample_num=n, survival_distribution=survival_distribution,
                                      treatment_weights=treatment_weights, test_size=test_size)
        df_train = pd.read_excel(path + "data.xlsx", sheet_name='train')
        df_test = pd.read_excel(path + "data.xlsx", sheet_name='test')

        # 模型估计
        run_flexcode_empirical(sample_num=n)  # 调用 flexcode 计算 conditional density
        cde_estimates = pd.read_excel(path + "CDE.xlsx", sheet_name='Sheet1').values
        a_grid = pd.read_excel(path + "CDE.xlsx", sheet_name='Sheet2').values

        # 真实条件密度如何求？
        # 确定参数，在 a_grid 上的真实取值，A = W * X + epsilon，A|X ~ N(W * X, 1)
        x_matrix = df_test.iloc[:, :3].values
        cde_true = conditional_density_true(x_matrix=x_matrix, treatment_weights=treatment_weights, a_grid=a_grid)

        # 条件密度如何评估？IMSE
        imse_cde = integrated_mean_squared_error_normalization(cde_estimates, cde_true, a_grid)
        imse_for_n.append(imse_cde)
    imse_list.append(np.mean(imse_for_n))

'''
n_list = [200, 400, 600, 800, 1000]
200 次平均
imse_list = [2.452739149093216, 2.1786972007094088, 2.0720803119448554, 2.0512903535086426, 1.9951831231777066]
'''

# n_list = [200, 400, 800]
# 30 次平均
# imse_list = [2.407272576112908, 2.2152714441857326, 2.070453781780857]


# imse_cde = integrated_mean_squared_error(cde_estimates, cde_true, a_grid)
# 200: 6.0596163958827445
# 300: 9.433575069072441

# df_cde = pd.DataFrame(cde_estimates)
# df_cde_true = pd.DataFrame(cde_true)
#
# writer = pd.ExcelWriter(r"C:\Users\janline\Desktop\cde_test.xlsx", engine='xlsxwriter')
# df_cde.to_excel(writer, sheet_name='estimate')
# df_cde_true.to_excel(writer, sheet_name='true')
# writer.close()