# #

# def generate_data(survival_distribution, sample_num, a):
#     mean = np.array([4, 5, 0.5])
#     covariance_matrix = np.array([[1, 0, 0.5],
#                                  [0, 1, 0.5],
#                                  [0.5, 0.5, 0.1]])
#     samples = np.random.multivariate_normal(mean=mean, cov=covariance_matrix, size=sample_num)
#     X = samples[:, :-1]
#     beta = np.array([1, -1])
#     treatment = samples[:, -1]
#     if survival_distribution == 'exponential':
#         # x = self.u_0 + self.u_1 * np.random.uniform(low=0, high=1, size=sample_num)
#         # treatment = self.w * x + np.random.normal(loc=0, scale=1, size=sample_num)
#         true_time = - np.log(np.random.uniform(low=0, high=1, size=sample_num)) * np.exp(- (np.dot(X, beta) - a))
#         censor_time = np.random.uniform(low=0, high=np.max(true_time), size=sample_num)
#         observed_time = np.minimum(true_time, censor_time)
#         event = 1 * (observed_time == true_time)
#         dataset = pd.DataFrame(
#             data=np.c_[X, treatment, observed_time, event],
#             columns=['x1', 'x2', 'treatment', 'time', 'event']
#         )
#         print(f"event rate of dataset: {sum(event) / sample_num}")
#         return dataset
#
# def generate_data(survival_distribution, sample_num, a):
# mean = np.array([4, 5, 0.5])
# covariance_matrix = np.array([[1, 0, 0.5],
# [0, 1, 0.5],
# [0.5, 0.5, 0.1]])
# samples = np.random.multivariate_normal(mean=mean, cov=covariance_matrix, size=sample_num)
# X = samples[:, :-1]
# beta = np.array([1, -1])
# treatment = samples[:, -1]
# if survival_distribution == 'exponential':
# # x = self.u_0 + self.u_1 * np.random.uniform(low=0, high=1, size=sample_num)
# # treatment = self.w * x + np.random.normal(loc=0, scale=1, size=sample_num)
# true_time = - np.log(np.random.uniform(low=0, high=1, size=sample_num)) * np.exp(- (np.dot(X, beta) - a))
# censor_time = np.random.uniform(low=0, high=np.max(true_time), size=sample_num)
# observed_time = np.minimum(true_time, censor_time)
# event = 1 * (observed_time == true_time)
# dataset = pd.DataFrame(
# data=np.c_[X, treatment, observed_time, event],
# columns=['x1', 'x2', 'treatment', 'time', 'event']
# )
# print(f"event rate of dataset: {sum(event) / sample_num}")
# return dataset
#
#
# # # 绘制散点图
# # plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
# # plt.title('Multivariate Normal Distribution')
# # plt.xlabel('X1')
# # plt.ylabel('X2')
# # plt.show()
#
#
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # # 创建一个3x3的子图布局
# # fig, axes = plt.subplots(3, 3, figsize=(8, 8))
# #
# # # 循环遍历每个子图
# # for i in range(3):
# #     for j in range(3):
# #         # 生成一些示例数据（可以根据需要替换成你的数据）
# #         x = np.linspace(0, 10, 100)
# #         y1 = np.sin(x)
# #         y2 = np.cos(x)
# #
# #         # 在子图中绘制两条折线
# #         axes[i, j].plot(x, y1, label='Line 1')
# #         axes[i, j].plot(x, y2, label='Line 2')
# #
# #         # 添加标题（可选）
# #         axes[i, j].set_title(f'Subplot {i + 1},{j + 1}')
# #
# #         # 添加图例
# #         axes[i, j].legend()
# #
# # # 调整子图之间的间距
# # plt.tight_layout()
# #
# # # 显示图形
# # plt.show()
#
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # # 创建一个3x3的子图布局
# # fig, axes = plt.subplots(3, 3, figsize=(8, 8))
# #
# # # 循环遍历每个子图
# # for i in range(3):
# #     for j in range(3):
# #         # 生成一些示例数据（可以根据需要替换成你的数据）
# #         data = np.random.rand(10, 10)
# #
# #         # 在子图中绘制数据
# #         axes[i, j].imshow(data, cmap='viridis')
# #
# #         # 添加标题（可选）
# #         axes[i, j].set_title(f'Subplot {i + 1},{j + 1}')
# #
# # # 调整子图之间的间距
# # plt.tight_layout()
# #
# # # 显示图形
# # plt.show()
#
# # import numpy as np
# # from scipy import integrate
# #
# # treatment_grid = np.arange(0, 3, 0.1)
# # time_grid = np.linspace(start=0, stop=10, num=50)
# # u_0 = 2
# # u_1 =  1
# # arg_lambda = lambda a, x : a + x
# # true_survival = np.empty(shape=(0, len(time_grid)))
# # for a in treatment_grid:
# #     # idx = np.argmin(np.abs(treatment_testSet - a))
# #     # x = feature_testSet[idx]
# #     f = lambda x, t: np.exp(- arg_lambda(a, x) * t) / u_1
# #
# #     # def survival_func(t):
# #     #     result, error = integrate.quad(lambda x: f(x, t), u_0, u_0 + u_1)  # integrate.quad返回元组（result，error）
# #     #     return result
# #
# #     # 定义一个矢量化的生存函数
# #     survival_func = np.vectorize(lambda t: integrate.quad(lambda x: f(x, t), u_0, u_0 + u_1)[0])
# #
# #     survival_of_a = survival_func(time_grid).reshape(1, -1)
# #     true_survival = np.vstack([true_survival, survival_of_a])
#
#
#
# # import numpy as np
# # from scipy import integrate
# #
# #
# # def survival_true(self, treatment_grid, time_grid):
# #     true_survival = np.empty(shape=(0, len(time_grid)))
# #     for a in treatment_grid:
# #         # idx = np.argmin(np.abs(treatment_testSet - a))
# #         # x = feature_testSet[idx]
# #         f = lambda x, t: np.exp(- self.arg_lambda(a, x) * t) / self.u_1
# #
# #         def survival_func(t):
# #             return integrate.quad(lambda x: f(x, t), self.u_0, self.u_0 + self.u_1)[
# #                 0]  # integrate.quad返回元组（result，error）
# #
# #         survival_of_a = survival_func(time_grid)
# #         true_survival = np.vstack([true_survival, survival_of_a])
# #
# #     return true_survival
# #
# # true_survival = survival_true(self, np.arange(10), np.arange(15))
#
#
# # import numpy as np
# # from scipy.stats import expon
# # a = 1
# # func = lambda x : x + a
# # survival_t = func(np.arange(10))
#
# # def test(N):
# #     arg_lambda = lambda a, x: a + x
# #     a = np.linspace(1, stop=10, num=N)
# #     x = np.random.uniform(low=1, high=2, size=N)
# #     return np.arange(N) / arg_lambda(a, x)
# #
# # res = test(10)
#
# # def update_scores(N, K, updates):
# #     # 初始化学生成绩数组，全部为60
# #     scores = [60] * N
# #
# #     # 执行K次更新操作
# #     for i in range(K):
# #         start, end, step = updates[i]
# #
# #         # 确保start和end在有效范围内
# #         start = max(0, start)
# #         end = min(N - 1, end)
# #
# #         # 对指定范围内的学生成绩进行更新
# #         for j in range(start, end + 1):
# #             scores[j] += step
# #
# #     return scores
# #
# # # 输入参数
# # N = 10  # 学生人数
# # K = 3   # 更新操作次数
# # updates = [(1, 5, 10), (3, 7, -5), (0, 9, 20)]  # 更新操作数组，每个操作为一个三元组
# #
# # # 调用函数进行更新操作
# # result = update_scores(N, K, updates)
# #
# # # 打印最终的学生成绩数组
# # print(result)
#
#
# # import time
# # from datetime import datetime
# # import numpy as np
# # import pandas as pd
# # import threading
# # from Simulation.output import mean_std_calculation
# # from Simulation.run_simulation_process.run_empirical_single import run_convergence_empirical
# #
# # def run_simulation(N, bandwidth, survival_distribution, path, test_size, simulation_times, treatment_num, result_lock, result_dict):
# #     imse_list = []
# #     mse_array = np.empty(shape=(0, treatment_num))
# #     rmse_array = np.empty(shape=(0, treatment_num))
# #     bias_array = np.empty(shape=(0, treatment_num))
# #
# #     for i in range(simulation_times):
# #         imse, mse, rmse, median_survival_time_bias = run_convergence_empirical(n=N, bandwidth=bandwidth,
# #                                                                                survival_distribution=survival_distribution,
# #                                                                                path=path, test_size=test_size)
# #         imse_list.append(imse)
# #         mse_array = np.vstack([mse_array, mse])
# #         rmse_array = np.vstack([rmse_array, rmse])
# #         bias_array = np.vstack([bias_array, median_survival_time_bias])
# #
# #     df_imse = pd.DataFrame(imse_list, columns=['IMSE'])
# #     mean_imse, std_imse, df_imse = mean_std_calculation(df_imse)
# #
# #     df_mse = pd.DataFrame(mse_array)
# #     mean_mse, std_mse, df_mse = mean_std_calculation(df_mse)
# #
# #     df_rmse = pd.DataFrame(rmse_array)
# #     mean_rmse, std_rmse, df_rmse = mean_std_calculation(df_rmse)
# #
# #     df_bias = pd.DataFrame(bias_array)
# #     mean_bias, std_bias, df_bias = mean_std_calculation(df_bias)
# #
# #     # 使用锁来保护对结果字典的访问
# #     with result_lock:
# #         result_dict[N] = {
# #             'mean_imse': mean_imse,
# #             'std_imse': std_imse,
# #             'mean_mse': mean_mse,
# #             'std_mse': std_mse,
# #             'mean_rmse': mean_rmse,
# #             'std_rmse': std_rmse,
# #             'mean_bias': mean_bias,
# #             'std_bias': std_bias
# #         }
# #
# # def main():
# #     sample_list = [600, 800, 1000]
# #     bandwidth = 0.25
# #     survival_distribution = 'exponential'
# #     run_date = datetime.today().strftime('%Y%m%d')
# #     path_base = fr"C:\Users\janline\OneDrive - stu.xmu.edu.cn\学校\论文\论文代码\simulation_data\simulation_empirical\{run_date}"
# #     test_size = 0.15
# #     treatment_num = 11
# #     simulation_times = 2
# #
# #     result_lock = threading.Lock()
# #     result_dict = {}
# #
# #     threads = []
# #
# #     for N in sample_list:
# #         path = f"{path_base}/{N}"
# #         thread = threading.Thread(target=run_simulation,
# #                                   args=(N, bandwidth, survival_distribution, path, test_size,
# #                                         simulation_times, treatment_num, result_lock, result_dict))
# #         threads.append(thread)
# #         thread.start()
# #
# #     for thread in threads:
# #         thread.join()
# #
# #     # 将结果写入Excel文件，这部分代码需要稍作修改以适应多线程结果
# #     write_results_to_excel(result_dict, treatment_num=treatment_num, output_file=path_base)
# #
# # if __name__ == "__main__":
# #     start_time = time.time()
# #     main()
# #     print(f"running time {(time.time() - start_time)/60:.2f} minutes")
# #
# #
# #
# # # import numpy as np
# # # import pandas as pd
# # #
# # #
# # # def write_results_to_excel(results, treatment_num, output_file):
# # #     # 创建一个用于存储所有指标的DataFrame
# # #     sample_list = []
# # #     simulation_mean_imse = np.empty(shape=(0, 1))
# # #     simulation_mean_mse = np.empty(shape=(0, treatment_num))
# # #     simulation_mean_rmse = np.empty(shape=(0, treatment_num))
# # #     simulation_mean_bias = np.empty(shape=(0, treatment_num))
# # #     simulation_std_imse = np.empty(shape=(0, 1))
# # #     simulation_std_mse = np.empty(shape=(0, treatment_num))
# # #     simulation_std_rmse = np.empty(shape=(0, treatment_num))
# # #     simulation_std_bias = np.empty(shape=(0, treatment_num))
# # #
# # #     for N, result in results.items():
# # #         sample_list.append(N)
# # #         simulation_mean_imse = np.vstack([simulation_mean_imse, result['mean_imse']])
# # #         simulation_std_imse = np.vstack([simulation_std_imse, result['std_imse']])
# # #         simulation_mean_mse = np.vstack([simulation_mean_mse, result['mean_mse']])
# # #         simulation_std_mse = np.vstack([simulation_std_mse, result['std_mse']])
# # #         simulation_mean_rmse = np.vstack([simulation_mean_rmse, result['mean_rmse']])
# # #         simulation_std_rmse = np.vstack([simulation_std_rmse, result['std_rmse']])
# # #         simulation_mean_bias = np.vstack([simulation_mean_bias, result['mean_bias']])
# # #         simulation_std_bias = np.vstack([simulation_std_bias, result['std_bias']])
# # #
# # #     df_simulation_mean_imse = pd.DataFrame(simulation_mean_imse, index=sample_list)
# # #     df_simulation_mean_mse = pd.DataFrame(simulation_mean_mse, index=sample_list)
# # #     df_simulation_mean_rmse = pd.DataFrame(simulation_mean_rmse, index=sample_list)
# # #     df_simulation_mean_bias = pd.DataFrame(simulation_mean_bias, index=sample_list)
# # #     df_simulation_std_imse = pd.DataFrame(simulation_std_imse, index=sample_list)
# # #     df_simulation_std_mse = pd.DataFrame(simulation_std_mse, index=sample_list)
# # #     df_simulation_std_rmse = pd.DataFrame(simulation_std_rmse, index=sample_list)
# # #     df_simulation_std_bias = pd.DataFrame(simulation_std_bias, index=sample_list)
# # #
# # #     writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
# # #     df_simulation_mean_imse.to_excel(writer, sheet_name='imse_mean')
# # #     df_simulation_mean_mse.to_excel(writer, sheet_name='mse_mean')
# # #     df_simulation_mean_rmse.to_excel(writer, sheet_name='rmse_mean')
# # #     df_simulation_mean_bias.to_excel(writer, sheet_name='bias_mean')
# # #     df_simulation_std_imse.to_excel(writer, sheet_name='imse_std')
# # #     df_simulation_std_mse.to_excel(writer, sheet_name='mse_std')
# # #     df_simulation_std_rmse.to_excel(writer, sheet_name='rmse_std')
# # #     df_simulation_std_bias.to_excel(writer, sheet_name='bias_std')
# # #     writer.close()
# # #
# # # # 示例用法
# # # results = {
# # #     200: {
# # #         'mean_imse': [0.123],  # 转换为列表形式
# # #         'std_imse': [0.045],   # 转换为列表形式
# # #         'mean_mse': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
# # #         'std_mse': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # #         'mean_rmse': [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.1, 1.21],
# # #         'std_rmse': [0.011, 0.022, 0.033, 0.044, 0.055, 0.066, 0.077, 0.088, 0.099, 0.11, 0.121],
# # #         'mean_bias': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # #         'std_bias': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011]
# # #     },
# # #     300: {
# # #         'mean_imse': [0.123],  # 转换为列表形式
# # #         'std_imse': [0.045],  # 转换为列表形式
# # #         'mean_mse': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
# # #         'std_mse': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # #         'mean_rmse': [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.1, 1.21],
# # #         'std_rmse': [0.011, 0.022, 0.033, 0.044, 0.055, 0.066, 0.077, 0.088, 0.099, 0.11, 0.121],
# # #         'mean_bias': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # #         'std_bias': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011]
# # #     },
# # #     400: {
# # #         'mean_imse': [0.123],  # 转换为列表形式
# # #         'std_imse': [0.045],  # 转换为列表形式
# # #         'mean_mse': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
# # #         'std_mse': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # #         'mean_rmse': [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.1, 1.21],
# # #         'std_rmse': [0.011, 0.022, 0.033, 0.044, 0.055, 0.066, 0.077, 0.088, 0.099, 0.11, 0.121],
# # #         'mean_bias': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # #         'std_bias': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011]
# # #     }
# # # }
# # #
# # #
# # # write_results_to_excel(results, treatment_num=11, output_file='./results.xlsx')
# # # #
# # # #
# # # # # import pandas as pd
# # # # #
# # # # #
# # # # # def write_results_to_excel(results, output_file):
# # # # #     writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
# # # # #
# # # # #     for N, result in results.items():
# # # # #         df = pd.DataFrame({
# # # # #             'mean_imse': [result['mean_imse']],
# # # # #             'std_imse': [result['std_imse']],
# # # # #             'mean_mse': result['mean_mse'],
# # # # #             'std_mse': result['std_mse'],
# # # # #             'mean_rmse': result['mean_rmse'],
# # # # #             'std_rmse': result['std_rmse'],
# # # # #             'mean_bias': result['mean_bias'],
# # # # #             'std_bias': result['std_bias']
# # # # #         })
# # # # #
# # # # #         # 写入不同N值的结果到不同sheet
# # # # #         sheet_name = f'N_{N}'
# # # # #         df.to_excel(writer, sheet_name=sheet_name, index=False)
# # # # #
# # # # #     writer.save()
# # # # #
# # # # # # 示例用法
# # # # # results = {
# # # # #     200: {
# # # # #         'mean_imse': 0.123,
# # # # #         'std_imse': 0.045,
# # # # #         'mean_mse': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
# # # # #         'std_mse': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # # # #         'mean_rmse': [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.1, 1.21],
# # # # #         'std_rmse': [0.011, 0.022, 0.033, 0.044, 0.055, 0.066, 0.077, 0.088, 0.099, 0.11, 0.121],
# # # # #         'mean_bias': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # # # #         'std_bias': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011]
# # # # #     },
# # # # #     300: {
# # # # #         'mean_imse': 0.11,
# # # # #         'std_imse': 0.22,
# # # # #         'mean_mse': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
# # # # #         'std_mse': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # # # #         'mean_rmse': [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.1, 1.21],
# # # # #         'std_rmse': [0.011, 0.022, 0.033, 0.044, 0.055, 0.066, 0.077, 0.088, 0.099, 0.11, 0.121],
# # # # #         'mean_bias': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
# # # # #         'std_bias': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011]
# # # # #     }
# # # # # }
# # # # #
# # # # # output_file = 'results.xlsx'
# # # # # write_results_to_excel(results, output_file)
# # # #
# # # #
# # # # # from datetime import datetime
# # # # #
# # # # # # # 获取今天的日期
# # # # # # today = datetime.today()
# # # # # #
# # # # # # # 格式化日期为 "YYYYMMDD" 的字符串
# # # # # # run_date = today.strftime("%Y%m%d")
# # # # #
# # # # # run_date = datetime.today().strftime('%Y%m%d')
# # # # #
# # # # # print(run_date)
# # # #
# # # # # import random
# # # # # import numpy as np
# # # # # treatment_grid = np.linspace(0, 10, num=150)
# # # # # treatment_grid_val = np.percentile(treatment_grid, [5, 25, 50, 75, 95])
# # # #
# # # # # import numpy as np
# # # # # from matplotlib import pyplot as plt
# # # # #
# # # # # evaluation_method = 'rmse'
# # # # #
# # # # # bandwidth_list = np.logspace(-4, 1, num=20)
# # # # # error_for_bandwidth_list = np.array([random.randint(1, 10) for i in range(20)])
# # # # # plt.plot(bandwidth_list, error_for_bandwidth_list, marker='o')
# # # # # plt.xlabel('bandwidth')
# # # # # plt.ylabel(f"{evaluation_method.upper()}")
# # # # # plt.show()
# # # #
# # # #
# # # # # import numpy as np
# # # # #
# # # # # bandwidth_list = np.logspace(-4, 1, num=20)
# # # #
# # # # # import pandas as pd
# # # # # from sklearn.model_selection import KFold
# # # # #
# # # # # from Simulation.data_generating.data_generate_process import train_validation_split
# # # # #
# # # # # N = 200
# # # # # cv = 5
# # # # # path = f"C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/{N}"
# # # # # df = pd.read_excel(path+'data.xlsx', sheet_name='train')
# # # # # # train_validation_split(df=df_train, cv=cv, save_path=path)  # split to validation and test set
# # # # # kf = KFold(n_splits=cv, shuffle=True)  # 随机分割数据，不设置 random_state，避免重复
# # # # # i = 0
# # # # # for train_index, validation_index in kf.split(df):
# # # # #     df_train = df.loc[train_index]
# # # # #     df_validation = df.loc[validation_index]
# # # # #
# # # # #     # df_train, df_test = time_moderate(df_train, df_test)  # 调整时间，避免计算综合 brier score 时报错
# # # # #
# # # # #     df_train.sort_values(by='o', ascending=True, inplace=True)
# # # # #     df_validation.sort_values(by='o', ascending=True, inplace=True)
# # # # #     # # 是否要排序？要排序，一是便于后续条件生存函数的估计,二是排序后样本的顺序和treatment的顺序一致，否则 IBS 的计算有误
# # # # #
# # # # #     # df_train.sort_values(by='a', ascending=True, inplace=True)
# # # # #     # df_test.sort_values(by='a', ascending=True, inplace=True)   # 便于对比输出的反事实结果？不需要对比
# # # # #
# # # # #     # 将数据存储到本地
# # # # #     writer = pd.ExcelWriter(path + f"data{i}.xlsx", engine='xlsxwriter')
# # # # #     df_train.to_excel(writer, sheet_name='train', index=False)
# # # # #     df_validation.to_excel(writer, sheet_name='validation', index=False)
# # # # #     # writer.save()
# # # # #     writer.close()
# # # # #     i += 1
# # # #
# # # # # import numpy as np
# # # # # from Simulation.ouput import subset_index
# # # # #
# # # # # mat_test = np.arange(60).reshape(10, 6)
# # # # # time_test = np.arange(20)
# # # # # row_index, col_index = subset_index(mat_test.shape, row_num=5, col_num=mat_test.shape[1])
# # # # # out_test = time_test[row_index]
# # # #
# # # # # # main.py
# # # # # import draft1
# # # #
# # # # # print("This is the main program")
# # # # # draft1.some_function()
# # # #
# # # # # import matplotlib.pyplot as plt
# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # from sklearn.model_selection import KFold
# # # # # # from Simulation.data_generating.data_generate_process import data_generate
# # # # # from sksurv.linear_model import CoxPHSurvivalAnalysis
# # # # # from sksurv.metrics import concordance_index_censored, integrated_brier_score
# # # # # from sksurv.metrics import concordance_index_censored
# # # # # import rpy2.robjects as robjects
# # # # # import rpy2
# # # # # import os
# # # # # from sklearn.metrics import mean_squared_error
# # # # # from scipy.integrate import dblquad
# # # # # from scipy.integrate import nquad
# # # # # from tabulate import tabulate
# # # # # import os
# # # # # import time
# # # #
# # # # # import rpy2.robjects as robjects
# # # # # robjects.r('''
# # # # #         # create a function `f`
# # # # #         f <- function(r, verbose=FALSE) {
# # # # #             if (verbose) {
# # # # #                 cat("I am calling f().\n")
# # # # #             }
# # # # #             2 * pi * r
# # # # #         }
# # # # #         # call the function `f` with argument value 3
# # # # #         f(3)
# # # # #         ''')
# # # # # r_f = robjects.r['f']
# # # # # res = r_f(2)  # <rpy2.robjects.vectors.FloatVector object at 0x00000171AFEB8300> [RTYPES.REALSXP] R classes: ('numeric',)[12.566371]
# # # # # a = res + 1  # 不是单纯的一个数
# # # #
# # # #
# # # # # # 定义R代码字符串
# # # # # r_code = """
# # # # # library(readxl)
# # # # # library(FlexCoDE)
# # # # # library(writexl)
# # # # #
# # # # # N <- 1000
# # # # # path <- paste0("C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/",N)
# # # # #
# # # # # for (i in 0:4) {
# # # # #   df <- read_excel(paste0(path, "data", i, ".xlsx"), sheet = "train")
# # # # #   df_test <- read_excel(paste0(path, "data", i, ".xlsx"), sheet = "test")
# # # # #
# # # # #   set.seed(1)
# # # # #   sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
# # # # #
# # # # #   data_train  <- df[sample, ]
# # # # #   ntrain = nrow(data_train)
# # # # #   xtrain = data_train[1:ntrain,1:3]
# # # # #   ztrain = data_train[1:ntrain,4]
# # # # #
# # # # #   data_validation   <- df[!sample, ]
# # # # #   nvalidation = nrow(data_validation)
# # # # #   xvalidation = data_validation[1:nvalidation,1:3]
# # # # #   zvalidation = data_validation[1:nvalidation,4]
# # # # #
# # # # #   data_test <- df_test
# # # # #   ntest = nrow(data_test)
# # # # #   xtest = data_test[1:ntest,1:3]
# # # # #   ztest = data_test[1:ntest,4]
# # # # #
# # # # #   # conditional density estimation caculation
# # # # #   fit = fitFlexCoDE(xtrain,ztrain,xvalidation,zvalidation,xtest,ztest,
# # # # #                     nIMax = 10,
# # # # #                     regressionFunction = regressionFunction.NW,
# # # # #                     n_grid = 1000)
# # # # #   predictedValues = predict(fit,xtest,B=1000)  # B的大小决定cde的稀疏
# # # # #   cde = as.data.frame(predictedValues$CDE)
# # # # #   grid = as.data.frame(predictedValues$z)
# # # # #   names(grid) = c('a')
# # # # #
# # # # #   # par(mfrow=c(2,2))
# # # # #   # # par(mar=c(1,1,1,1))
# # # # #   # for (col in 1:4){
# # # # #   #   plot(predictedValues$z,predictedValues$CDE[col,],col='lightblue')  # z_grid, cde
# # # # #   #   loc = as.numeric(4*xtest[col,1]+2*xtest[col,2]+xtest[col,3])   # A = W * X
# # # # #   #   lines(predictedValues$z,dnorm(predictedValues$z,loc,1),col='red')  #真实cd
# # # # #   # }
# # # # #
# # # # #   output_list = list(cde,grid)
# # # # #   write_xlsx(output_list, path = paste0(path, "CDE", i, ".xlsx"))
# # # # # }
# # # # # """
# # # # # # 执行R代码
# # # # # robjects.r(r_code)
# # # # # robjects.r(r_code, encoding='latin1')
# # # # '''
# # # # 虽报错，但可行
# # # # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd4 in position 0: invalid continuation byte
# # # # R[write to console]: randomForest 4.7-1.1
# # # # R[write to console]: Type rfNews() to see new features/changes/bug fixes.
# # # # '''
# # # #
# # # #
# # # # '''' 不可行 '''
# # # # # import subprocess
# # # # # with open('temp_script.R', 'w') as f:
# # # # #     f.write(r_code)
# # # # #
# # # # # # 运行R脚本
# # # # # subprocess.run(['Rscript', 'temp_script.R'])
# # # #
# # # # # # 读取R脚本输出文件
# # # # # with open('output_file.txt', 'r', encoding='latin1') as f:  # 这里使用'latin1'编码
# # # # #     r_output = f.read()
# # # # #
# # # # # # 打印R脚本的输出
# # # # # print(r_output)
# # # #
# # # #
# # # # ''' 以下代码可行 '''
# # # # import rpy2.robjects as robjects
# # # # from rpy2.robjects.packages import importr
# # # #
# # # # # # 导入R的stats包
# # # # # # stats = importr('stats')
# # # # #
# # # # # # 创建Python中的数据
# # # # # x = [1, 2, 3, 4, 5]
# # # # # y = [2, 4, 6, 8, 10]
# # # # #
# # # # # # 将Python中的数据转换为R中的对象
# # # # # robjects.globalenv['x'] = robjects.FloatVector(x)
# # # # # robjects.globalenv['y'] = robjects.FloatVector(y)
# # # # #
# # # # # # 在R中执行线性回归
# # # # # lm_model = stats.lm('y ~ x')
# # # # #
# # # # # # 打印回归结果
# # # # # print(lm_model)
# # # #
# # # # # 不可行
# # # # # import rpy2.robjects as robjects
# # # # # from rpy2.robjects.packages import importr
# # # # #
# # # # # # 加载R包
# # # # # base = importr("base")
# # # # #
# # # # # # 调用R包中的函数
# # # # # result = robjects.r['mean'](robjects.IntVector([1, 2, 3, 4, 5]))
# # # # #
# # # # # # 将R对象转换为Python对象
# # # # # python_result = robjects.conversion.rpy2py(result)
# # # # #
# # # # # print("Mean:", python_result)
# # # #
# # # #
# # # #
# # # # # test_change = np.arange(12).reshape(3, -1)
# # # # # table_test = print_latex(test_change)
# # # # # # if table_test == table_test_change:
# # # # # #     print("True")
# # # # #
# # # # # # def print_latex(matrix_output):
# # # # # #     """
# # # # # #     @param matrix_output: ndarray
# # # # # #     @return: 将 matrix_output 打印成 latex 格式
# # # # # #     """
# # # # # #     table_output = tabulate(matrix_output, tablefmt="latex", floatfmt=".4f")    # 输出保留4位小数
# # # # # #     matrix_name = f"{matrix_output}"  # 有误，需要的是 matrix_output 的变量名，但赋值的是变量 matrix_output，
# # # # # #     new_name = f"table_{matrix_name}"
# # # # # #     exec(f"{new_name} = '{table_output}' ")
# # # # # #     print("=" * 100, f"{new_name}:\n {table_output}", "\n")
# # # # # #     # print("=" * 100,table_output, "\n")       # 打印结果，复制粘贴到latex
# # # # # #     return new_name
# # # #
# # # # # a_test_for_change = np.arange(12).reshape(3, -1)
# # # # # b = a_test_for_change * 2
# # # # #
# # # # # def find_var_name(obj):
# # # # #     """
# # # # #     Find the name of a variable that refers to the given object.
# # # # #     """
# # # # #     for name, val in locals().items():
# # # # #         if val is obj:
# # # # #             return f"{name}"
# # # # #     for name, val in globals().items():
# # # # #         if val is obj:
# # # # #             return f"{name}"
# # # # #
# # # # #
# # # # # val_name = find_var_name(a_test_for_change)
# # # # #
# # # # # print(val_name)  # Output: obj
# # # #
# # # #
# # # # # def find_var_name(obj):
# # # # #     """
# # # # #     Find the name of a variable that refers to the given object.
# # # # #     """
# # # # #     for name, val in locals().items():
# # # # #         if val is obj:
# # # # #             return name
# # # # #     for name, val in globals().items():
# # # # #         if val is obj:
# # # # #             return name
# # # # #     return None
# # # # #
# # # # # a_test_for_change = np.arange(12).reshape(3,-1)
# # # # # val_name = find_var_name(a_test_for_change)
# # # # #
# # # # # print(val_name)
# # # #
# # # #
# # # # # a = np.arange(10).reshape(2,-1)
# # # # # a_name = f" '{a}' "
# # # #
# # # # #
# # # # # # 遍历全局变量字典
# # # # # for name in globals():
# # # # #     # 如果变量与指定变量a相同，则将变量名存储到val_name中
# # # # #     if globals()[name] is a:
# # # # #         val_name = name
# # # # #         break
# # # #
# # # # # 输出变量名和值
# # # # # print(val_name)    # 输出：a
# # # #
# # # #
# # # # # # 将 字符串 "apple" 的变量名从 a 改成 fruit
# # # # # a = "apple"
# # # # # new_name = "fruit"
# # # # # exec(f"{new_name} = '{a}' ", globals())
# # # #
# # # # # a = "apple"
# # # # # for i, c in enumerate(a):
# # # # #     new_name = f"b{i}"
# # # # #     exec(f"{new_name} = '{c}'", globals()) # 将变量添加到全局作用域中
# # # # #     print(f"{new_name}: {c}")
# # # # # print(b0) # 此处访问变量b0，将会输出a
# # # #
# # # #
# # # # # print("=" * 100, "\n")
# # # # # a = 1
# # # # # print(a)
# # # # # print("=" * 100, "\n")
# # # #
# # # # # # h_list = np.array([0.01       0.01098541 0.01206793 0.01325711 0.01456348 0.01599859, 0.01757511 0.01930698 0.02120951 0.02329952 0.02559548 0.02811769, 0.03088844 0.03393222 0.03727594 0.04094915 0.04498433 0.04941713, 0.05428675 0.05963623 0.06551286 0.07196857 0.07906043 0.08685114, 0.09540955 0.10481131 0.11513954 0.12648552 0.13894955 0.1526418, 0.16768329 0.184207   0.20235896 0.22229965 0.24420531 0.26826958, 0.29470517 0.32374575 0.35564803 0.39069399 0.42919343 0.47148664, 0.51794747 0.5689866  0.62505519 0.68664885 0.75431201 0.82864277, 0.91029818 1.        ])
# # # # # h_list = [0.01       0.01098541 0.01206793 0.01325711 0.01456348 0.01599859, 0.01757511 0.01930698 0.02120951 0.02329952 0.02559548 0.02811769, 0.03088844 0.03393222 0.03727594 0.04094915 0.04498433 0.04941713, 0.05428675 0.05963623 0.06551286 0.07196857 0.07906043 0.08685114, 0.09540955 0.10481131 0.11513954 0.12648552 0.13894955 0.1526418, 0.16768329 0.184207   0.20235896 0.22229965 0.24420531 0.26826958, 0.29470517 0.32374575 0.35564803 0.39069399 0.42919343 0.47148664, 0.51794747 0.5689866  0.62505519 0.68664885 0.75431201 0.82864277, 0.91029818 1.        ]
# # # # #
# # # # # num_h = len(h_list)
# # # #
# # # # # start_time = time.time()
# # # # # for i in range(10):
# # # # #     h_list = [1, 0.75, 0.5, 0.25]
# # # # #     h_list1 = [1, 0.75, 0.5, 0.25]
# # # # #     h_list2 = [1, 0.75, 0.5, 0.25]
# # # # #     arr = np.array([h_list, h_list1, h_list2])
# # # # #     end_time = time.time()
# # # # #     run_time = end_time - start_time
# # # # #     print("程序运行时间为：", run_time, "秒")
# # # #
# # # #
# # # # # h = 0.25
# # # # # # 在这里绘制您的图像
# # # # # plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# # # # # plt.xlabel('time')
# # # # # plt.ylabel('survival probability')
# # # # #
# # # # # # 将图像保存到本地
# # # # # desktop = os.path.expanduser("~/Desktop")
# # # # # filename = f"h{int(100*h)}.png"
# # # # # filepath = os.path.join(desktop, filename)
# # # # # plt.savefig(filepath)
# # # #
# # # # # summary_median_survival = np.empty(shape=(0, 6))
# # # # # arr = np.arange(6)
# # # # # summary_median_survival = np.vstack((summary_median_survival, arr))
# # # # # arr1 = np.random.randint(1,10,6)
# # # # # summary_median_survival = np.vstack((summary_median_survival, arr1))
# # # #
# # # # # h = 0.25
# # # # # # table_pred = f"table_counterfactual_survival_{int(h * 100)}"
# # # # # arr = np.arange(12).reshape(3, 4)
# # # # # row_index = np.array([0, 2])
# # # # # aa = arr.T[row_index]
# # # # # df = pd.DataFrame(arr, index=['h_opt','mse','imse'])
# # # # # table = tabulate(df, tablefmt="latex", floatfmt=".4f")
# # # # # table_pred = f"\\label{int(h*100)}\n{table}\n"
# # # # # table_pred1 = f"\\label{int(h*100)}\n{table}\n"
# # # # # print(table_pred, "-"*100, table_pred1, "="*100)
# # # #
# # # # # table_out = f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{My table caption}}\n\\label{{table_counterfactual_survival_{int(h * 100)}}}\n{table}\n\\end{{table}}"
# # # # # f"table_out_{h}" == table_out
# # # # # a = np.arange(0, 10, 3)
# # # #
# # # # # # 创建一个200x500的随机数组
# # # # # data = np.random.rand(20, 50)
# # # # #
# # # # # # 设置表头和表尾
# # # # # header = ['Column {}'.format(i+1) for i in range(data.shape[1])]
# # # # # footer = [''] * data.shape[1]
# # # # #
# # # # # # 将数组转换为包含LaTeX表格的字符串
# # # # # table = tabulate(data, headers=header, showindex=False, tablefmt="latex")
# # # # #
# # # # # # 在字符串中添加sidewaystable环境的开始和结束标记
# # # # # table = '\\begin{sidewaystable}\\centering\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{%s}\n%s\\end{tabular}}\n\\end{sidewaystable}' % ('|' + '|'.join(['c']*data.shape[1]) + '|', table)
# # # # #
# # # # # # 输出LaTeX表格
# # # # # print(table)
# # # #
# # # #
# # # # # # 创建一个200x500的随机数组
# # # # # data = np.random.rand(200, 50)
# # # # #
# # # # # # 设置表头和表尾
# # # # # header = ['Column {}'.format(i+1) for i in range(data.shape[1])]
# # # # # footer = [''] * data.shape[1]
# # # # #
# # # # # # 将数组转换为包含LaTeX表格的字符串
# # # # # table = tabulate(data, headers=header, showindex=False, tablefmt='latex')
# # # # #
# # # # # # 在字符串中添加tabularx的开始和结束标记  # 自动调整但没调整成功
# # # # # table = '\\begin{table}\\centering\\begin{tabularx}{\\textwidth}{%s}\n%s\\end{tabularx}\\end{table}' % ('|' + '|'.join(['X']*data.shape[1]) + '|', table)
# # # # #
# # # # # # 输出LaTeX表格
# # # # # print(table)
# # # #
# # # #
# # # # # # 创建一个200x500的随机数组  带标题
# # # # # data = np.random.rand(20, 50)
# # # # #
# # # # # # 设置表头和表尾
# # # # # header = ['Column {}'.format(i+1) for i in range(data.shape[1])]
# # # # # footer = [''] * data.shape[1]
# # # # #
# # # # # # 将数组转换为包含LaTeX表格的字符串
# # # # # table = tabulate(data, headers=header, showindex=False, tablefmt='latex')
# # # # #
# # # # # # 在字符串中添加longtable的开始和结束标记
# # # # # table = '\\begin{longtable}{%s}\n%s\\end{longtable}' % ('|' + '|'.join(['c']*data.shape[1]) + '|', table)
# # # #
# # # # # 输出LaTeX表格
# # # # # print(table)
# # # #
# # # #
# # # # # # 创建一个200x500的随机数组  # A4 页面显示不出来
# # # # # data = np.random.rand(10, 20)
# # # #
# # # # # 将数组转换为包含LaTeX表格的字符串
# # # # # table = tabulate(np.concatenate((data[-5:, ])), tablefmt="latex")
# # # # #
# # # # # # 输出LaTeX表格
# # # # # print(table)
# # # #
# # # #
# # # # # h = 1000**(-1/5)
# # # # #
# # # # # h3 = 3*h
# # # # #
# # # # # h4 = 4*h
# # # #
# # # # # survival_est = np.arange(12).reshape(3, 4)
# # # # # empty_est = np.zeros(shape=survival_est.shape)
# # # #
# # # # # survival_est = np.arange(10).reshape(2, -1)
# # # # # survival_true = np.arange(10).reshape(2, -1)
# # # # # grid = np.arange(5)
# # # # # # grid = np.tile(grid.flatten(), len(survival_est))
# # # # # # survival_est = survival_est.flatten()
# # # # # # survival_true = survival_true.flatten()
# # # # # #
# # # # # # term1 = np.trapz(survival_est**2, grid)
# # # # # # term2 = np.trapz(survival_est * survival_true, grid)
# # # # # # mise = term1 - 2 * term2
# # # # # m = integrated_mean_squared_error(survival_est, survival_true, grid)
# # # #
# # # # # grid = np.arange(5)
# # # # # grid = np.tile(grid.flatten(), 3)
# # # #
# # # # # a = np.arange(5)
# # # # # b = np.tile(a, 3)
# # # # # b = a.item()
# # # # # .reshape(2,-1)
# # # # # b = a.flatten()
# # # #
# # # # # # 定义被积函数
# # # # # def func(x, y):
# # # # #     return np.exp(-x*y)
# # # # #
# # # # # # 定义积分区间
# # # # # x_range = [0, 1]
# # # # # y_range = [lambda x: 0, lambda x: 1 - x]
# # # # #
# # # # # # 计算积分
# # # # # result, error = nquad(func, [x_range, y_range])
# # # # #
# # # # # print("结果:", result)
# # # # # print("误差:", error)
# # # #
# # # # # # 定义被积函数
# # # # # def func(x, y):
# # # # #     return x+y
# # # # #
# # # # # # 定义积分区间和积分区域
# # # # # a, b = 0, 1
# # # # # low_fun = lambda x: 0
# # # # # upper_fun = lambda x: 1 - x
# # # # #
# # # # # # 计算积分
# # # # # result, error = dblquad(func, a, b, low_fun, upper_fun)
# # # # #
# # # # # print("结果:", result)
# # # # # print("误差:", error)
# # # #
# # # #
# # # # # a=np.trapz([1,2,3], x=[4,6,8])
# # # #
# # # # # y_true = [[0.5, 1],[-1, 1],[7, -6]]
# # # # # y_pred = [[0, 2],[-1, 2],[8, -5]]
# # # # # mse = mean_squared_error(y_true, y_pred) # 0.708
# # # # # MSE =((0.5 - 0)**2 + (1 - 2)**2 + (-1 + 1)**2 + (-1 + 2)**2 + (7 - 8)**2 + (-6 + 5)**2)/6  # 0.70833
# # # #
# # # #
# # # # # os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.0'
# # # # #
# # # # #
# # # # # # rpy2.rinterface.set_R_HOME('/path/to/R')
# # # # #
# # # # # print(rpy2.__version__) # 3.5.11
# # # # # import rpy2.situation as rpy2situation
# # # # # # print(rpy2situation.get_r_version())
# # # # # # import rpy2.situation as sit
# # # # # #
# # # # # # print(sit.get_r_home())
# # # # # # print(sit.get_rversion())
# # # # #
# # # # #
# # # # # # os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.0'  # 根据实际安装路径进行修改
# # # # # # # 执行一条R语言命令
# # # # # result = robjects.r('paste("Hello", "world!")')
# # # # # print(result[0])
# # # #
# # # # # N = 1000
# # # # # path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# # # # # df_train = pd.read_excel(path+"data.xlsx", sheet_name='train')
# # # # # df_test = pd.read_excel(path+"data.xlsx", sheet_name='test')
# # # # # # df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
# # # # # #
# # # # # # a_median = np.median(df_test['a'])
# # # # # lambda_index = df_test['lambda'][0]
# # # # #
# # # # # T = df['o']
# # # # # treatment_col = df['a']
# # # # # num_samples = len(df)
# # # # # C = np.random.exponential(scale=np.mean(T) + np.std(T), size=num_samples)     # 删失时间服从指数分布
# # # # # C1 = np.random.uniform(low=0, high=treatment_col)
# # # #
# # # # # c = np.array([[4], [5], [1]])
# # # # # b = c.flatten()
# # # #
# # # # # treatment_col = np.random.binomial(10, 0.5, size=100)
# # # # # C = np.random.uniform(low=0, high=treatment_col)
# # # #
# # # # # a = np.logspace(0.01, 1, 10)
# # # # # b = np.logspace(-2, 0, num=10)  # 带宽取值范围在0.01到1之间
# # # # # estimator = CoxPHSurvivalAnalysis()
# # # # # estimator.fit(data_x_numeric, data_y)
# # # # # prediction = estimator.predict(data_x_numeric)
# # # # # result = concordance_index_censored(data_y["Status"], data_y["Survival_in_days"], prediction)
# # # # # c_index = result[0]
# # # #
# # # # # lis = [1,2,3,4,5]
# # # # # a = np.mean(lis)
# # # #
# # # # # cv = 5
# # # # # for i in range(cv):
# # # # #     print(i+1)
# # # #
# # # # # train_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# # # # # train_event = np.array([1, 1, 0, 1, 0])
# # # # # test_time = np.array([6.0, 7.0, 8.0])
# # # # # # 检查测试集时间是否超出最大观测时间点
# # # # # extends = test_time > np.max(train_time)
# # # # # if np.any(extends):
# # # # #     # 将超出的时间值替换为最大观测时间点
# # # # #     test_time = np.where(extends, np.max(train_time), test_time)  # 直接替换
# # # #
# # # #
# # # # # N = 1000
# # # # # path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# # # # # df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')
# # # # # df_test = pd.read_excel(path+"data.xlsx",sheet_name='test')
# # # # # df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
# # # # #
# # # # # T = df['o']
# # # # # treatment_col = df['a']
# # # # # num_samples = len(df)
# # # # # C = np.random.exponential(scale=np.mean(T) + np.std(T), size=num_samples)     # 删失时间服从指数分布
# # # # # C1 = np.random.uniform(low=0, high=treatment_col)
# # # #
# # # # #
# # # # # # 检查测试集时间是否超出最大观测时间点
# # # # # extends = df_test['o'] > np.max(df_train['o'])
# # # # # if np.any(extends):
# # # # #     # 将超出的时间值替换为最大观测时间点
# # # # #     test_time = np.where(extends, np.max(df_train['o']), df_test['o'])
# # # #
# # # #
# # # # #
# # # # # kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 随机抽取索引
# # # # # # 进行交叉验证
# # # # # for train_index, test_index in kf.split(df):
# # # # #     df_val = df.loc[test_index]
# # # # #     # print(test_index, "="*50)
# # # #
# # # # # my_dict = {'a': 3, 'b': 2, 'c': 1}
# # # # # k, min_value = min(my_dict.items(), key=lambda x: x[1])
# # # # #
# # # # # x = my_dict.keys()
# # # # # y = my_dict.values()
# # # # #
# # # # # plt.figure()
# # # # # plt.plot(x, y, marker='o')
# # # # # plt.xlabel('x')
# # # # # plt.ylabel('y')
# # # # # plt.show()
# # # #
# # # # # a = np.random.choice(250)
# # # #
# # # # # df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, index=[1,3,5])
# # # # # idx = df['col1'].idxmax()
# # # # # row_to_drop = df.loc[idx]
# # # # # df_temp = row_to_drop.to_frame().T
# # # #
# # # #
# # # # # df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, index=[1,3,5]).reset_index(drop=True)
# # # # # idx = np.argmin(df['col1'])
# # # # # row_to_drop = df.loc[idx, :]
# # # # # df = df.drop(idx)
# # # #
# # # # # test = np.random.randint(1,10,size=5)
# # # # # idx = np.argmin(test)
# # # #
# # # # # time_points = np.arange(1, 1000)
# # # #
# # # # # N = 1000
# # # # # path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# # # # # df_test = pd.read_excel(path+"data.xlsx",sheet_name='test')
# # # # # df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')
# # # # #
# # # # # # idx_test = np.random.choice(df_test.index, size=1).item()
# # # # # # idx_test = np.random.choice(df_test.index, size=1)
# # # # # # row_to_train = df_test.loc[idx_test]
# # # # #
# # # # # print(df_test.describe())
# # # # # print(df_train.describe())
# # # # # print(type(df_test['o']))
# # # # # print(df_test['o'].min())
# # # # # print(type(df_test['o'].min()))
# # # # # print(df_test['o'].min() < df_test['o'].max())
# # # #
# # # # # treatment_idx = np.random.randint(low=0, high=10, size=3)
# # # # # colors = ['r', 'g', 'b']
# # # # # for idx, color in zip(treatment_idx, colors):
# # # # #     print(idx, color)
# # # #
# # # # # estimator = CoxPHSurvivalAnalysis()
# # # # # estimator.fit(data_x_numeric, data_y)
# # # # # estimator.score()
# # # # #
# # # # # # Estimate the survival function
# # # # # survival = estimator.predict_survival_function(data[['time']])
# # # # #
# # # # # # Calculate the integrated Brier score
# # # # # ibs = integrated_brier_score(survival, data['status'], data['time'], t_max=100)
# # # #
# # # #
# # # #
# # # # # N = 100
# # # # # path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# # # # # data_generate(N, path)
# # # #
# # # # # path = "C:/Users/janline/Desktop/"
# # # # # df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')
# # # #
# # # # # list_t = [1,2,3,4,5,6]
# # # # #
# # # # # weight = np.empty(shape=(0,6))
# # # # # # list_normalization = np.array(list_t) / sum(list_t)
# # # # # # weight = np.vstack([weight0,list_normalization])
# # # # # weight = np.vstack([weight,list_t])
# # # # #
# # # # # list_t1 = [1,2,3,4,5,5]
# # # # # # list_normalization1 = np.array(list_t1) / sum(list_t1)
# # # # # # weight = np.vstack([weight, list_normalization1])
# # # # # weight = np.vstack([weight,list_t1])
# # # # #
# # # # # n = weight.shape[0]
# # # # # res = []
# # # # # for i in range(n):
# # # # #     index = np.argmin(np.abs(weight[i, :] - 3))
# # # # #     res.append(weight[i, index])
# # # #
# # # #
# # # # # arr = np.array(list_t).reshape(2,3)
# # # # # arr_T = arr.T
# # # # # multiply = arr_T @ arr
# # # #
# # # # # time = 1.02
# # # # # treat = 3.766
# # # # # conditional_survival_est = conditional_survival_estimated[:, col]  # ndarray:(150,)
# # # # # kernel_val = gaussian_kernel(a_approx, treat, h)  # ndarray:(150,1), list 150
# # # #
# # # # # fenzi = 0
# # # # # for i in [0]:
# # # # #     print(pi[i] , conditional_survival_est[i] , kernel_val[i],
# # # # #           pi[i] * conditional_survival_est[i] * kernel_val[i])
# # # #
# # # # # total = pi * conditional_survival_est * kernel_val
# # # # # fenzi = np.sum(total)
# # # # #
# # # # # total1 = pi * kernel_val
# # # # # fenmu = np.sum(total1)
# # # #
# # # # # survival_est = np.sum(pi * conditional_survival_est * kernel_val)/np.sum(pi * kernel_val)
# # # # # survival_estimates.append(survival_est)
# # # #
# # # # # a = np.array([1,2,3]).reshape(-1,1)
# # # # # b = a
# # # # # c = a*b
# # # # # d = a*a*a
# # # # # e = sum(d)
# # # #
# # # # # df_test = pd.read_excel("C:/Users/janline/Desktop/simulation_data/data.xlsx",sheet_name='test')
# # # # # a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet2')
# # # # #
# # # # #
# # # # # n_obs = len(df_test)
# # # # # a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]  # 长度和 df_test 一致
# # # # # a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index]).reshape(-1,1)
# # # # # # a_approx = [a_grid.loc[i].item() for i in a_approx_index]
# # # # #
# # # # # treatment_grid = np.linspace(min(a_approx), max(a_approx), num=100)  # 连续 treatment 取值网格点
# # # # # for treat in treatment_grid:
# # # # #     print(treat)
# # # #
# # # # # a = [-1,1,2]+[1]
# # # #
# # # # # df_train = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='train')
# # # # # df_validation = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='validation')
# # # # # df_test = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='test')
# # # # #
# # # # # df = pd.concat([df_train, df_validation, df_test], axis=0)
# # # # #
# # # # # cde_estimates = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet1')
# # # # # a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet2')
# # # # #
# # # # # n_obs = len(df_test)
# # # # # a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]  # 长度和 df_test 一致
# # # # # a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index])
# # # #
# # # #
# # # #
# # # #
# # # # #
# # # # # n_obs = len(df_test)
# # # # # nns = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]
# # # # # cde_list =[]
# # # # # for x_index,grid_index in enumerate(nns):
# # # # #     cde_list.append(cde_estimates.iloc[x_index, grid_index])
# # # # #
# # # # # # cde_list = [cde_estimates[range(n_obs), nns]]
# # # # #
# # # # # # num_samples = 1000
# # # # # # da = np.random.randn(num_samples).reshape(-1,1)