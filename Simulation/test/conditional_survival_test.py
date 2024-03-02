import numpy as np
import pandas as pd
from Simulation.conditional_density_estimation.conditional_density_estimate import conditional_density_true
from Simulation.conditional_survival_function.conditional_survival_estimate import conditional_survival_estimate
from Simulation.metrics import integrated_mean_squared_error, integrated_mean_squared_error_normalization, survival_true
from Simulation.simulation_process.main_class import CounterfactualSurvFtn


'''========== ========== 参数修改 ========== ========== '''
# n_list = [200, 400, 800]
n_list = [200, 400, 600, 800, 1000]
# sample_list = [200, 400, 800]   # 200-700，间隔100
# n = 200
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
# path = fr"C:\Users\janline\OneDrive - stu.xmu.edu.cn\学校\论文\论文代码\simulation_data\test\{n}"
'''========== ========== ========== ========== ========== '''

imse_adj_list = []
imse_list = []
for n in n_list:
    path = fr"C:\Users\janline\OneDrive - stu.xmu.edu.cn\学校\论文\论文代码\simulation_data\test\{n}"
    imse_for_n = []
    imse_adj_for_n = []
    for i in range(simulation_times):
        model = CounterfactualSurvFtn(path=path, cv=cv)

        # 数据生成
        model.data_generate_empirical(sample_num=n, survival_distribution=survival_distribution,
                                      treatment_weights=treatment_weights, test_size=test_size)
        df_train = pd.read_excel(path + "data.xlsx", sheet_name='train')
        df_test = pd.read_excel(path + "data.xlsx", sheet_name='test')

        # 条件生存函数在 time_grid 上的估计
        time_grid = np.linspace(start=min(df_train['o']), stop=max(df_train['o']), num=500)   # 基于观测时间设定 time_grid
        conditional_survival_estimated = conditional_survival_estimate(df_train, df_test, time_grid)  # 未调参

        # 条件生存函数在 time_grid 上的真实值
        treatment_grid = df_test['a'].values
        treatment_testSet = df_test['a']
        lambda_testSet = df_test['lambda']
        conditional_survival_true = survival_true(survival_distribution, treatment_grid, time_grid,
                                                     treatment_testSet=treatment_testSet, lambda_testSet=lambda_testSet)
        # imse 评估
        imse_adj = integrated_mean_squared_error_normalization(conditional_survival_estimated, conditional_survival_true, time_grid)
        imse = integrated_mean_squared_error(conditional_survival_estimated, conditional_survival_true, time_grid)
        imse_adj_for_n.append(imse_adj)
        imse_for_n.append(imse)
    imse_adj_list.append(np.mean(imse_adj_for_n))
    imse_list.append(np.mean(imse_for_n))

# n_list = [200, 400, 600, 800, 1000]
# imse_adj_list = [0.02757304919520204, 0.014716542780154307, 0.010845856946218478, 0.008861310103042128, 0.008125679625383972]
# imse_list = [0.3327624063309571, 0.3785080189449529, 0.4254881410411292, 0.4726557686073943, 0.5539120678736589]
