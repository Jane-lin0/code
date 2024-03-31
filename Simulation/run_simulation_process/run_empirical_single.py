import time
from datetime import datetime

import numpy as np
import pandas as pd
from Simulation.simulation_process.main_class import CounterfactualSurvFtn
from Simulation.conditional_density_estimation.Flexcode_rpy2 import run_flexcode_empirical, run_flexcode_test


def run_convergence_empirical(n, bandwidth, survival_distribution, path, test_size):
    # 模型导入
    model = CounterfactualSurvFtn(path=path)

    # 数据生成
    model.data_generate_empirical(sample_num=n, survival_distribution=survival_distribution,
                                  test_size=test_size)
    df_train = pd.read_excel(path+"data.xlsx", sheet_name='train')
    df_test = pd.read_excel(path+"data.xlsx", sheet_name='test')

    # 模型拟合调参
    # run_flexcode_validation()   # 调用 flexcode 计算 conditional density
    # best_bandwidth = model.fit(bandwidth_list=bandwidth_list, evaluation_method=validation_evaluation_method, visualization=True)

    # 模型估计
    run_flexcode_empirical(sample_num=n)  # 调用 flexcode 计算 conditional density
    cde_estimates = pd.read_excel(path+"CDE.xlsx", sheet_name='Sheet1')
    a_grid = pd.read_excel(path+"CDE.xlsx", sheet_name='Sheet2')

    counterfactual_survival_pred = model.predict(df_train, df_test, cde_estimates, a_grid, bandwidth)
    print("counterfactual survival calculation completed")

    # 误差评估
    IMSE = model.estimate_error(counterfactual_survival_pred, method='imse')
    RISE = model.estimate_error(counterfactual_survival_pred, method='rise')
    RMSE = model.estimate_error(counterfactual_survival_pred, method='rmse')
    median_survival_time_bias = model.estimate_error(counterfactual_survival_pred, method='bias')

    # model.visualization()   # 生存函数的估计和真实值对比  # treatment A 取单个值时无法画图

    return IMSE, RISE, RMSE, median_survival_time_bias


# if __name__ == '__main__':
#     N = 200
#     bandwidth = 0.25
#     survival_distribution = 'exponential'
#     run_date = datetime.today().strftime('%Y%m%d')
#     path_base = r"C:\Users\janline\Desktop\毕业论文\论文代码\simulation_data\simulation_empirical"
#     path = f"{path_base}/{run_date}/{N}"
#     test_size = 0.2
#     imse, mse, rmse, median_survival_time_bias = run_convergence_empirical(n=N, bandwidth=bandwidth,
#                                                                            survival_distribution=survival_distribution,
#                                                                            path=path, test_size=test_size)



