import time
from datetime import datetime
import numpy as np
import pandas as pd
from Simulation.output import mean_std_calculation
from Simulation.run_simulation_process.run_empirical_single import run_convergence_empirical
import multiprocessing
from rpy2.rinterface_lib import openrlib
import threading

# 由于Python的全局解释器锁（GIL），threading在CPU密集型任务中的性能可能受到限制。
# 如果代码中的run_convergence_empirical函数是CPU密集型的，并且希望充分利用多核处理器，
# 可能需要考虑使用multiprocessing模块来实现多进程并行化，以避免GIL的限制。


def run_simulation(N, bandwidth, survival_distribution, path, test_size, simulation_times, treatment_num, result_dict):
    imse_list = []
    mse_array = np.empty(shape=(0, treatment_num))
    rmse_array = np.empty(shape=(0, treatment_num))
    bias_array = np.empty(shape=(0, treatment_num))

    with openrlib.rlock:
        for i in range(simulation_times):
            imse, mse, rmse, median_survival_time_bias = run_convergence_empirical(n=N, bandwidth=bandwidth,
                                                                                   survival_distribution=survival_distribution,
                                                                                   path=path, test_size=test_size)
            imse_list.append(imse)
            mse_array = np.vstack([mse_array, mse])
            rmse_array = np.vstack([rmse_array, rmse])
            bias_array = np.vstack([bias_array, median_survival_time_bias])

    df_imse = pd.DataFrame(imse_list, columns=['IMSE'])
    mean_imse, std_imse, df_imse = mean_std_calculation(df_imse)

    df_mse = pd.DataFrame(mse_array)
    mean_mse, std_mse, df_mse = mean_std_calculation(df_mse)

    df_rmse = pd.DataFrame(rmse_array)
    mean_rmse, std_rmse, df_rmse = mean_std_calculation(df_rmse)

    df_bias = pd.DataFrame(bias_array)
    mean_bias, std_bias, df_bias = mean_std_calculation(df_bias)

    # 使用锁来保护对结果字典的访问
    # with result_lock:
    result_dict[N] = {
        'mean_imse': mean_imse,
        'std_imse': std_imse,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'mean_bias': mean_bias,
        'std_bias': std_bias
    }
    # result_queue.put((N, mean_imse, std_imse, mean_mse, std_mse, mean_rmse, std_rmse, mean_bias, std_bias))


def write_results_to_excel(results, treatment_num, output_file):
    # 创建一个用于存储所有指标的DataFrame
    sample_list = []
    simulation_mean_imse = np.empty(shape=(0, 1))
    simulation_mean_mse = np.empty(shape=(0, treatment_num))
    simulation_mean_rmse = np.empty(shape=(0, treatment_num))
    simulation_mean_bias = np.empty(shape=(0, treatment_num))
    simulation_std_imse = np.empty(shape=(0, 1))
    simulation_std_mse = np.empty(shape=(0, treatment_num))
    simulation_std_rmse = np.empty(shape=(0, treatment_num))
    simulation_std_bias = np.empty(shape=(0, treatment_num))

    for N, result in results.items():
        sample_list.append(N)
        simulation_mean_imse = np.vstack([simulation_mean_imse, result['mean_imse']])
        simulation_std_imse = np.vstack([simulation_std_imse, result['std_imse']])
        simulation_mean_mse = np.vstack([simulation_mean_mse, result['mean_mse']])
        simulation_std_mse = np.vstack([simulation_std_mse, result['std_mse']])
        simulation_mean_rmse = np.vstack([simulation_mean_rmse, result['mean_rmse']])
        simulation_std_rmse = np.vstack([simulation_std_rmse, result['std_rmse']])
        simulation_mean_bias = np.vstack([simulation_mean_bias, result['mean_bias']])
        simulation_std_bias = np.vstack([simulation_std_bias, result['std_bias']])

    df_simulation_mean_imse = pd.DataFrame(simulation_mean_imse, index=sample_list)
    df_simulation_mean_mse = pd.DataFrame(simulation_mean_mse, index=sample_list)
    df_simulation_mean_rmse = pd.DataFrame(simulation_mean_rmse, index=sample_list)
    df_simulation_mean_bias = pd.DataFrame(simulation_mean_bias, index=sample_list)
    df_simulation_std_imse = pd.DataFrame(simulation_std_imse, index=sample_list)
    df_simulation_std_mse = pd.DataFrame(simulation_std_mse, index=sample_list)
    df_simulation_std_rmse = pd.DataFrame(simulation_std_rmse, index=sample_list)
    df_simulation_std_bias = pd.DataFrame(simulation_std_bias, index=sample_list)

    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    df_simulation_mean_imse.to_excel(writer, sheet_name='imse_mean')
    df_simulation_mean_mse.to_excel(writer, sheet_name='mse_mean')
    df_simulation_mean_rmse.to_excel(writer, sheet_name='rmse_mean')
    df_simulation_mean_bias.to_excel(writer, sheet_name='bias_mean')
    df_simulation_std_imse.to_excel(writer, sheet_name='imse_std')
    df_simulation_std_mse.to_excel(writer, sheet_name='mse_std')
    df_simulation_std_rmse.to_excel(writer, sheet_name='rmse_std')
    df_simulation_std_bias.to_excel(writer, sheet_name='bias_std')
    writer.close()


def main():
    sample_list = [600, 800, 1000]
    bandwidth = 0.25
    survival_distribution = 'exponential'
    run_date = datetime.today().strftime('%Y%m%d')
    path_base = fr"C:\Users\janline\OneDrive - stu.xmu.edu.cn\学校\论文\论文代码\simulation_data\simulation_empirical\{run_date}"
    test_size = 0.15
    treatment_num = 11
    simulation_times = 2

    # result_lock = threading.Lock()
    result_dict = {}
    threads = []

    # result_queue = multiprocessing.Queue()
    # processes = []

    for N in sample_list:
        path = f"{path_base}/{N}"
        thread = threading.Thread(target=run_simulation,
                                  args=(N, bandwidth, survival_distribution, path, test_size,
                                        simulation_times, treatment_num, result_dict))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # 将结果写入Excel文件，这部分代码需要稍作修改以适应多进程结果
    write_results_to_excel(result_dict, treatment_num=treatment_num, output_file=path_base)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"running time {(time.time() - start_time)/60:.2f} minutes")


    # process = multiprocessing.Process(target=run_simulation,
    #                                   args=(N, bandwidth, survival_distribution, path, test_size,
    #                                         simulation_times, treatment_num, result_queue))
    # processes.append(process)
    # process.start()

    # for process in processes:
    #     process.join()
    #
    # results = {}
    #
    # while not result_queue.empty():
    #     N, mean_imse, std_imse, mean_mse, std_mse, mean_rmse, std_rmse, mean_bias, std_bias = result_queue.get()
    #     results[N] = {
    #         'mean_imse': mean_imse,
    #         'std_imse': std_imse,
    #         'mean_mse': mean_mse,
    #         'std_mse': std_mse,
    #         'mean_rmse': mean_rmse,
    #         'std_rmse': std_rmse,
    #         'mean_bias': mean_bias,
    #         'std_bias': std_bias
    #     }

# import time
# from datetime import datetime
# import numpy as np
# import pandas as pd
# import threading
#
# from Simulation.output import mean_std_calculation
# from Simulation.run_simulation_process.run_empirical_single import run_convergence_empirical
#
# def run_simulation(N, bandwidth, survival_distribution, path, test_size, simulation_times):
#     imse_list = []
#     mse_array = np.empty(shape=(0, treatment_num))
#     rmse_array = np.empty(shape=(0, treatment_num))
#     bias_array = np.empty(shape=(0, treatment_num))
#
#     for i in range(simulation_times):
#         imse, mse, rmse, median_survival_time_bias = run_convergence_empirical(n=N, bandwidth=bandwidth,
#                                                                                survival_distribution=survival_distribution,
#                                                                                path=path, test_size=test_size)
#         imse_list.append(imse)
#         mse_array = np.vstack([mse_array, mse])
#         rmse_array = np.vstack([rmse_array, rmse])
#         bias_array = np.vstack([bias_array, median_survival_time_bias])
#
#     df_imse = pd.DataFrame(imse_list, columns=['IMSE'])
#     mean_imse, std_imse, df_imse = mean_std_calculation(df_imse)
#     simulation_mean_imse[N] = mean_imse
#     simulation_std_imse[N] = std_imse
#
#     df_mse = pd.DataFrame(mse_array)
#     mean_mse, std_mse, df_mse = mean_std_calculation(df_mse)
#     simulation_mean_mse[N] = mean_mse
#     simulation_std_mse[N] = std_mse
#
#     df_rmse = pd.DataFrame(rmse_array)
#     mean_rmse, std_rmse, df_rmse = mean_std_calculation(df_rmse)
#     simulation_mean_rmse[N] = mean_rmse
#     simulation_std_rmse[N] = std_rmse
#
#     df_bias = pd.DataFrame(bias_array)
#     mean_bias, std_bias, df_bias = mean_std_calculation(df_bias)
#     simulation_mean_bias[N] = mean_bias
#     simulation_std_bias[N] = std_bias
#
# start_time = time.time()
# sample_list = [600, 800, 1000]
# bandwidth = 0.25
# survival_distribution = 'exponential'
# path_base = r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\学校\论文\论文代码\simulation_data\simulation_empirical"
# test_size = 0.15
# treatment_num = 11
# simulation_times = 200
# run_date = datetime.today().strftime('%Y%m%d')
#
# simulation_mean_imse = {}
# simulation_std_imse = {}
# simulation_mean_mse = {}
# simulation_std_mse = {}
# simulation_mean_rmse = {}
# simulation_std_rmse = {}
# simulation_mean_bias = {}
# simulation_std_bias = {}
#
# threads = []
#
# for N in sample_list:
#     path = f"{path_base}/{run_date}/{N}"
#     thread = threading.Thread(target=run_simulation, args=(N, bandwidth, survival_distribution, path, test_size, simulation_times))
#     threads.append(thread)
#     thread.start()
#
# for thread in threads:
#     thread.join()
#
# # 将结果写入Excel文件，这部分代码需要稍作修改以适应多线程结果
# # ...
#
# print(f"running time {(time.time() - start_time)/60:.2f} minutes")
