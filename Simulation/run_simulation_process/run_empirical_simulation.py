import time
from datetime import datetime
import numpy as np
import pandas as pd
from Simulation.output import mean_std_calculation
from Simulation.run_simulation_process.run_empirical_single import run_convergence_empirical

start_time = time.time()

'''========== ========== 参数修改 ========== ========== '''
# N = 300
sample_list = [200, 500, 1000, 2000, 3000]   # 200-700，间隔100
# sample_list = [200]
bandwidth = 0.25
# bandwidth_list = np.array([0.25, 0.5, 0.75, 1])
# bandwidth_list = np.logspace(-2, 0, num=15)  # 0.001 至 1 之间的 10 个数
# cv = 5
survival_distribution = 'exponential'
path_base = r"C:\Users\janline\Desktop\毕业论文\论文代码\simulation_data\simulation_empirical"
test_size = 0.2
# validation_evaluation_method = 'rmse'
treatment_num = 1      #11
simulation_times = 200  # 30
'''========== ========== ========== ========== ========== '''

simulation_mean_imse = np.empty(shape=(0, 1))
simulation_mean_rise = np.empty(shape=(0, treatment_num))
simulation_mean_rmse = np.empty(shape=(0, treatment_num))
simulation_mean_bias = np.empty(shape=(0, treatment_num))
simulation_std_imse = np.empty(shape=(0, 1))
simulation_std_rise = np.empty(shape=(0, treatment_num))
simulation_std_rmse = np.empty(shape=(0, treatment_num))
simulation_std_bias = np.empty(shape=(0, treatment_num))

run_date = datetime.today().strftime('%Y%m%d')
for N in sample_list:
    path = f"{path_base}/{run_date}/{N}"
    imse_list = []
    rise_list = []       # 单个 treatment 的误差
    rmse_list = []
    bias_list = []
    # mse_array = np.empty(shape=(0, treatment_num))  # treatment_num 个 treatment 的误差
    # rmse_array = np.empty(shape=(0, treatment_num))
    # bias_array = np.empty(shape=(0, treatment_num))

    for i in range(simulation_times):
        imse, rise, rmse, median_survival_time_bias = run_convergence_empirical(n=N, bandwidth=bandwidth,
                                                                               survival_distribution=survival_distribution,

                                                                               path=path, test_size=test_size)
        imse_list.append(imse)
        rise_list.append(rise)
        rmse_list.append(rmse)
        bias_list.append(median_survival_time_bias)
        # mse_array = np.vstack([mse_array, mse])
        # rmse_array = np.vstack([rmse_array, rmse])
        # bias_array = np.vstack([bias_array, median_survival_time_bias])

    df_imse = pd.DataFrame(imse_list, columns=['IMSE'])
    mean_imse, std_imse, df_imse = mean_std_calculation(df_imse)
    simulation_mean_imse = np.vstack([simulation_mean_imse, mean_imse])
    simulation_std_imse = np.vstack([simulation_std_imse, std_imse])

    df_rise = pd.DataFrame(rise_list)
    mean_rise, std_rise, df_rise = mean_std_calculation(df_rise)
    simulation_mean_rise = np.vstack([simulation_mean_rise, mean_rise])
    simulation_std_rise = np.vstack([simulation_std_rise, std_rise])

    df_rmse = pd.DataFrame(rmse_list)
    mean_rmse, std_rmse, df_rmse = mean_std_calculation(df_rmse)
    simulation_mean_rmse = np.vstack([simulation_mean_rmse, mean_rmse])
    simulation_std_rmse = np.vstack([simulation_std_rmse, std_rmse])

    df_bias = pd.DataFrame(bias_list)
    mean_bias, std_bias, df_bias = mean_std_calculation(df_bias)
    simulation_mean_bias = np.vstack([simulation_mean_bias, mean_bias])
    simulation_std_bias = np.vstack([simulation_std_bias, std_bias])

    writer = pd.ExcelWriter(path + "Error_Summary.xlsx", engine='xlsxwriter')
    df_imse.to_excel(writer, sheet_name='imse')
    df_rise.to_excel(writer, sheet_name='rise')
    df_rmse.to_excel(writer, sheet_name='rmse')
    df_bias.to_excel(writer, sheet_name='bias')
    writer.close()


df_simulation_mean_imse = pd.DataFrame(simulation_mean_imse, index=sample_list)
df_simulation_mean_rise = pd.DataFrame(simulation_mean_rise, index=sample_list)
df_simulation_mean_rmse = pd.DataFrame(simulation_mean_rmse, index=sample_list)
df_simulation_mean_bias = pd.DataFrame(simulation_mean_bias, index=sample_list)
df_simulation_std_imse = pd.DataFrame(simulation_std_imse, index=sample_list)
df_simulation_std_rise = pd.DataFrame(simulation_std_rise, index=sample_list)
df_simulation_std_rmse = pd.DataFrame(simulation_std_rmse, index=sample_list)
df_simulation_std_bias = pd.DataFrame(simulation_std_bias, index=sample_list)

writer = pd.ExcelWriter(f"{path_base}/{run_date}/Simulation_Summary.xlsx", engine='xlsxwriter')
df_simulation_mean_imse.to_excel(writer, sheet_name='imse_mean')
df_simulation_mean_rise.to_excel(writer, sheet_name='rise_mean')
df_simulation_mean_rmse.to_excel(writer, sheet_name='rmse_mean')
df_simulation_mean_bias.to_excel(writer, sheet_name='bias_mean')
df_simulation_std_imse.to_excel(writer, sheet_name='imse_std')
df_simulation_std_rise.to_excel(writer, sheet_name='rise_std')
df_simulation_std_rmse.to_excel(writer, sheet_name='rmse_std')
df_simulation_std_bias.to_excel(writer, sheet_name='bias_std')
writer.close()
 
print(f"running time {(time.time() - start_time)/60:.2f} minutes")   # running time 0.38 minutes
