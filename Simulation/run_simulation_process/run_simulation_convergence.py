import time
import numpy as np
import pandas as pd
from Simulation.simulation_process.simulation_main import CounterfactualSurvFtn
from Simulation.conditional_density_estimation.Flexcode_rpy2 import run_flexcode_validation, run_flexcode_test

start_time = time.time()

'''========== ========== 参数修改 ========== ========== '''
N = 200
'''！！！！！记得修改 run_flexcode_validation 函数的样本数 N ！！！！！'''
survival_distribution = 'exponential'
cv = 5
path = f"C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/{N}"
test_size = 0.15
# bandwidth_list = np.logspace(-2, 0, num=15)  # 0.001 至 1 之间的 10 个数
bandwidth_list = np.logspace(-4, 1, num=20)
validation_evaluation_method = 'rmse'
'''========== ========== ========== ========== ========== '''

# 模型导入
model = CounterfactualSurvFtn(path=path, cv=cv)

# 数据生成
model.data_generate(sample_num=N, survival_distribution=survival_distribution, test_size=test_size)
df_train = pd.read_excel(path+"data.xlsx", sheet_name='train')
df_test = pd.read_excel(path+"data.xlsx", sheet_name='test')

# 模型拟合调参
run_flexcode_validation()   # 调用 flexcode 计算 conditional density
best_bandwidth = model.fit(bandwidth_list=bandwidth_list, evaluation_method=validation_evaluation_method, visualization=True)

# 模型估计
run_flexcode_test(sample_num=N)  # 调用 flexcode 计算 conditional density
cde_estimates = pd.read_excel(path+"CDE.xlsx", sheet_name='Sheet1')
a_grid = pd.read_excel(path+"CDE.xlsx", sheet_name='Sheet2')

counterfactual_survival_pred = model.predict(df_train, df_test, cde_estimates, a_grid, best_bandwidth)
print("counterfactual survival calculation completed")

# 误差评估
IMSE = model.estimate_error(counterfactual_survival_pred, df_test['a'], df_test['lambda'], method='imse')
MSE = model.estimate_error(counterfactual_survival_pred, df_test['a'], df_test['lambda'], method='mse')
RMSE = model.estimate_error(counterfactual_survival_pred, df_test['a'], df_test['lambda'], method='rmse')
median_survival_time_bias = model.estimate_error(counterfactual_survival_pred, df_test['a'], df_test['lambda'], method='bias')

print(f"running time {(time.time() - start_time)/60:.2f} minutes")
