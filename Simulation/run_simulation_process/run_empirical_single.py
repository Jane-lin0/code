import time
import numpy as np
import pandas as pd
from Simulation.simulation_process.simulation_main import CounterfactualSurvFtn
from Simulation.conditional_density_estimation.Flexcode_rpy2 import run_flexcode_empirical, run_flexcode_test


def run_convergence_empirical(n, bandwidth, cv, survival_distribution, path, test_size):

    # 模型导入
    model = CounterfactualSurvFtn(path=path, cv=cv)

    # 数据生成
    model.data_generate_empirical(sample_num=n, survival_distribution=survival_distribution,
                                  treatment_weights=[4, 2, 1], test_size=test_size)
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
    IMSE = model.estimate_error(counterfactual_survival_pred, df_test['a'], df_test['lambda'], method='imse')
    MSE = model.estimate_error(counterfactual_survival_pred, df_test['a'], df_test['lambda'], method='mse')
    RMSE = model.estimate_error(counterfactual_survival_pred, df_test['a'], df_test['lambda'], method='rmse')
    median_survival_time_bias = model.estimate_error(counterfactual_survival_pred, df_test['a'], df_test['lambda'], method='bias')

    return IMSE, MSE, RMSE, median_survival_time_bias


# if __name__ == 'main':




