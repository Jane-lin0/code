import numpy as np
import pandas as pd
import os
from scipy.stats import expon
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from matplotlib import pyplot as plt
from Simulation.data_generating.DGP_pysurvival import SimulationModel
from Simulation.data_generating.data_generate import generate_data
from Simulation.data_generating.data_processing import train_test_data_split, train_validation_split
from Simulation.kernel_density_smoothing.density_estimate import density_estimate
from Simulation.kernel_setting import gaussian_kernel
from Simulation.conditional_survival_function.conditional_survival_estimate import conditional_survival_estimate, get_x_y
from Simulation.conditional_density_estimation.conditional_density_estimate import cde_adjust
from Simulation.metrics import mean_squared_error_normalization, integrated_mean_squared_error_normalization, \
    median_survival_time, restricted_mean_squared_error
from Simulation.metrics import survival_true, get_best_bandwidth
from Simulation.output import equal_space, subset_index, subset, treatment_subset_index


class CounterfactualSurvFtn():
    def __init__(self, path):
        self.survival_true_subset = None
        self.survival_pred_subset = None
        self.data_path = path   # the path to save data
        # self.cv = cv    # cross validation for train validation data
        self.survival_distribution = None
        self.time_grid = None    # 估计结果中的 time 取值网格点
        self.treatment_arg = 0.5     # 先看单个 treatment 取值的结果是否收敛
        self.treatment_grid = None  # 估计结果中的 treatment 取值网格点
        # self.treatment_eval_grid = np.arange(start=0, stop=0.9, step=0.1)   # 可能取值偏小
        # self.bandwidth = None     # 单独传入，便于经验法则计算
        self.error_for_bandwidth_list = None  # 画 bandwidth 选择的图
        self.x_beta = None
        # np.linspace(0, 1, 11)  # 生成 11 个数：0，0.1，···，1
        # self.u_0 = None
        # self.u_1 = None  # the weights to generate X, X = u_0 + u_1 * uniform[0,1]
        # self.w = None  # the weights to generate A, A = w * X + N(0,1)
        # self.arg_lambda = None  # the function to generate argument lambda

    # def data_generate(self, sample_num, survival_distribution, test_size, cv):
    #     """
    #     @param sample_num:
    #     @param survival_distribution: exponential, weibull, gompertz, log_logistic, log_normal
    #     @param test_size:
    #     @return:
    #     """
    #     self.survival_distribution = survival_distribution
    #     sim = SimulationModel(survival_distribution=self.survival_distribution,
    #                           risk_type='linear',
    #                           alpha=1,
    #                           beta=1
    #                           )
    #     dataset = sim.generate_data(num_samples=sample_num, num_features=4,
    #                                 feature_weights=[-2, 1, 2] + [1],  # beta  gamma
    #                                 treatment_weights=[4, 2, 1])  # W
    #     # lambda = exp(-1 * x + 1 * a) * alpha , a = 2 * x
    #     dataset.columns = ['x1', 'x2', 'x3', 'a', 'o', 'e', 'lambda']
    #     # self.treatment_quantile = np.percentile(dataset['a'], [0, 25, 50, 75, 100])  # 后续对估计进行评估
    #     train_test_data_split(dataset, test_size=test_size, save_path=self.data_path)
    #     # train test split，不设置 random_state，避免重复
    #     df_train = pd.read_excel(self.data_path+'data.xlsx', sheet_name='train')
    #     train_validation_split(df=df_train, cv=cv, save_path=self.data_path)  # split to validation and test set
    #     print(f"dataset generated and saved to {self.data_path}")

    def data_generate_empirical(self, survival_distribution, sample_num, test_size):
        self.survival_distribution = survival_distribution
        # self.u_0, self.u_1 = 2, 1  # the weights to generate X, X = u_0 + u_1 * uniform[0,1]
        # self.w = 2                 # the weights to generate A, A = w * X + N(0,1)
        # self.arg_lambda = lambda a, x: a + x    # the function to generate argument lambda
        dataset, self.x_beta = generate_data(survival_distribution=survival_distribution, sample_num=sample_num, a=self.treatment_arg)  # 设定 X、A 为正态分布
        dataset.columns = ['x1', 'x2', 'a', 'o', 'e']
        train_test_data_split(dataset, test_size=test_size, save_path=self.data_path)  # 不设置 random_state以避免重复
        print(f"dataset generated and saved to {self.data_path}")

    def fit(self, bandwidth_list, evaluation_method):
        """
        @param bandwidth_list: the bandwidth list to choose best bandwidth
        @return: return best bandwidth
        """
        error_for_bandwidth_list = []
        for h in bandwidth_list:
            # self.bandwidth = h
            error_for_h = []
            for i in range(self.cv):
                df_train = pd.read_excel(self.data_path + f"data{i}.xlsx", sheet_name='train')
                df_validation = pd.read_excel(self.data_path + f"data{i}.xlsx", sheet_name='validation')
                cde_estimates = pd.read_excel(self.data_path + f"CDE{i}.xlsx", sheet_name='Sheet1')
                a_grid = pd.read_excel(self.data_path + f"CDE{i}.xlsx", sheet_name='Sheet2')
                survival_pred = self.predict(df_train, df_validation, cde_estimates, a_grid, h)
                # survival_pred = self.predict(df_train, df_validation, cde_estimates, a_grid)
                error = self.estimate_error(survival_pred, method=evaluation_method)
                if evaluation_method != 'imse':
                    error = np.mean(error)
                error_for_h.append(error)
            error_for_bandwidth_list.append(np.mean(error_for_h))
        self.error_for_bandwidth_list = error_for_bandwidth_list
        # self.bandwidth = get_best_bandwidth(error_list=error_for_bandwidth_list, h_list=bandwidth_list)
        best_bandwidth = get_best_bandwidth(error_list=error_for_bandwidth_list, h_list=bandwidth_list)
        print(f"best bandwidth is {best_bandwidth}")
        # if visualization:
        #     self.visualization(bandwidth_list, error_for_bandwidth_list, evaluation_method)
        # return best_bandwidth

    # def predict(self, df_train, df_validation, cde_estimates, a_grid, best_bandwidth):
    def predict(self, df_train, df_validation, cde_estimates, a_grid, bandwidth):
        """
        @param df_train:
        @param df_validation:
        @param cde_estimates:
        @param a_grid:
        @param bandwidth: bandwidth
        @return:
        """
        # df = pd.concat([df_train, df_validation], axis=0)
        '''
        conditional_density_estimate，A|X 的条件密度估计 p(a|x) 
        '''
        # 输出 a_true (用 a_grid 近似) 对应的 cde list
        n_validation = len(df_validation)
        a_approx_index = [np.argmin(np.abs(a_grid - df_validation['a'][j])) for j in range(n_validation)]  # 长度和 df_validation 一致
        cde_list = []
        for x_index, grid_index in enumerate(a_approx_index):
            cde_val = cde_estimates.iloc[x_index, grid_index]
            cde_list.append(cde_val)
        cde = cde_adjust(cde_list)  # 给 cde_list 的零值加上一个很小的值，避免求 pi 时除以 0 得到 inf
        '''
        estimate density function of A: p(a), by kernel density smoothing
        '''
        a_approx = np.array([a_grid.loc[k].item() for k in a_approx_index]).reshape(-1, 1)
        density_estimated = density_estimate(df_train[['a']], a_approx)  # df_train 拟合
        # df[['a']] 拟合模型，返回 a_approx 上的密度估计
        '''
        pai(ai,xi) = p(a) / p(a|x)
        '''
        pi = density_estimated / cde  # ndarray:(len(df_test),)
        # 取与 a_true 最近的 a_grid (即 a_approx) 进行估计
        '''
        conditional survival function estimate S(t|A,X)
        '''
        # time_grid = np.linspace(start=min(df_test['o']), stop=max(df_test['o']), num=500)  # time_grid 设置可调整
        # time_grid = df_test['o']
        # self.time_grid = np.linspace(start=0, stop=max(df_train['o']), num=500)
        self.time_grid = np.linspace(start=min(df_train['o']), stop=max(df_train['o']), num=500)
        conditional_survival_estimated = conditional_survival_estimate(df_train, df_validation, self.time_grid)  # 未调参
        # ndarray:(len(df_test), len(time_grid))
        '''
        kernel setting, calculate hat{Sa(t)}：每个 a_grid 下的生存函数估计
        '''
        # a = 1
        self.treatment_grid = np.arange(0, max(a_approx)+0.1, 0.1)  # 连续 treatment 估计取值的网格点，间隔为 0.1，减少估计的treatment网格点
        # self.treatment_grid = self.treatment_arg
        # self.treatment_grid = np.arange(0, max(a_approx)+0.01, 0.01)  # 连续 treatment 估计取值的网格点，间隔为 0.01
        # self.treatment_grid = np.linspace(0, max(a_approx), num=n_validation)  # 连续 treatment 估计取值的网格点，可调整
        # if not self.treatment_grid_eval:
        #     self.treatment_grid_eval = np.percentile(self.treatment_grid, [5, 25, 50, 75, 95])    # 取5个分位数
        # self.treatment_grid = np.linspace(min(a_approx), max(a_approx), num=n_validation)  # 连续 treatment 估计取值的网格点，可调整
        # treatment_grid = a_approx

        # 根据权重计算反事实生存函数
        weight = np.empty(shape=(0, n_validation))
        for a in self.treatment_grid:
        # for a in self.treatment_eval_grid:
        # for a in [self.treatment_arg]:         # treatment_arg 是浮点数，加上[]变成可迭代的列表
            # kernel_values = gaussian_kernel(a_approx, a, self.bandwidth)
            kernel_values = gaussian_kernel(a_approx, a, bandwidth)
            w_a = pi * kernel_values  # ndarray:(len(df_test),)
            w_normalization = w_a / np.sum(w_a)  # 结果相对正常，含非 0 值
            weight = np.vstack([weight, w_normalization.reshape(1, -1)])
            # final result: weight = { ndarray:(len(treatment_grid), len(df_test))}

        counterfactual_survival = weight @ conditional_survival_estimated
        # ndarray:(len(treatment_grid), len(time_grid))
        return counterfactual_survival

    def estimate_error(self, survival_pred, method):
        """  
        @param survival_pred: ndarray:(len(treatment_grid), len(time_grid))
        # @param treatment_num: the treatment num to estimate error, i.e. the output row
        @param method: the error calculation method, imse, mse, rmse, bias
        @return: estimate error
        """
        # survival_true_values = survival_true(self.survival_distribution, self.treatment_grid, self.time_grid,
        #                                      self.u_0, self.u_1, self.arg_lambda)
        # row_index, col_index = subset_index(survival_pred.shape, row_num=self.treatment_num, col_num=survival_pred.shape[1])
        survival_true_subset = survival_true(self.survival_distribution, self.treatment_arg, self.time_grid, self.x_beta)  # 用 Oracle 计算
        row_index = np.searchsorted(self.treatment_grid, self.treatment_arg)
        col_index = np.arange(len(self.time_grid))
        # col_index = equal_space(length=survival_pred.shape[1], indices_num=survival_pred.shape[1])
        # self.treatment_eval_grid 的 indices: self.treatment_grid 的前 11 行
        # 保险起见用 searchsorted 查找 treatment_eval_grid 在 treatment_grid 中的索引
        # row_index, col_index = treatment_subset_index(survival_pred.shape, row_list=treatment_eval_grid,
        #                                               col_num=survival_pred.shape[1])
        survival_pred_subset = survival_pred[row_index, col_index]
        # survival_pred_subset = subset(survival_pred, row_index, col_index)
        # survival_true_subset = survival_true(self.survival_distribution, self.treatment_grid[row_index], self.time_grid,
        #                                 self.u_0, self.u_1, self.arg_lambda)
        self.survival_pred_subset = survival_pred_subset
        self.survival_true_subset = survival_true_subset
        # self.survival_true_subset = survival_true_subset
        if isinstance(self.treatment_arg, float):
            treatment_num = 1
        # treatment_num = len(self.treatment_arg)
        # treatment_num = len(self.treatment_eval_grid)
        if survival_pred_subset.shape[0] != treatment_num or survival_true_subset.shape[0] != treatment_num:
            survival_pred_subset = survival_pred_subset.reshape(treatment_num, -1)
            survival_true_subset = survival_true_subset.reshape(treatment_num, -1)
        # else:
        #     pass

        if method == 'imse':
            imse = integrated_mean_squared_error_normalization(survival_pred_subset, survival_true_subset, self.time_grid)
            # imse = integrated_mean_squared_error_normalization(survival_pred, survival_true_values, self.time_grid)
            return imse

        elif method == 'rise':
            rise_list = []
            for idx in range(treatment_num):
                survival_pred_a = survival_pred_subset[idx, :].reshape(1, -1)
                survival_true_a = survival_true_subset[idx, :].reshape(1, -1)
                rise = mean_squared_error_normalization(survival_pred_a, survival_true_a, self.time_grid)
                rise_list.append(rise)
            return rise_list

        elif method == 'rmse':
            rmse_list = []
            for idx in range(treatment_num):
                survival_pred_a = survival_pred_subset[idx, :].reshape(1, -1)
                survival_true_a = survival_true_subset[idx, :].reshape(1, -1)
                rmse = restricted_mean_squared_error(survival_pred_a, survival_true_a, self.time_grid)
                rmse_list.append(rmse)
            return rmse_list

        elif method == 'bias':
            median_survival_time_pred = median_survival_time(survival_pred_subset, self.time_grid)
            median_survival_time_true = median_survival_time(survival_true_subset, self.time_grid)
            median_survival_time_bias = median_survival_time_pred - median_survival_time_true
            return median_survival_time_bias

        else:
            print('No such method')

    def visualization(self):
        # 创建一个3x3的子图布局
        fig, axes = plt.subplots(3, 3)
        row_index = 0
        # 循环遍历每个子图
        for i in range(3):
            for j in range(3):
                x = self.time_grid
                survival_true_val = self.survival_true_subset[row_index, :]
                survival_pred_val = self.survival_pred_subset[row_index, :]

                # 在子图中绘制数据
                axes[i, j].plot(x, survival_true_val, label=r'$S_a(t)$')
                axes[i, j].plot(x, survival_pred_val, label=r'$\hat{S}_a(t)$', linestyle='--')

                axes[i, j].set_xlabel('t')
                # axes[i, j].set_ylabel()

                # 添加标题
                axes[i, j].set_title('Treatment A = {:.1f}'.format(self.treatment_eval_grid[row_index]))

                # 缩小坐标轴标签的字体大小
                axes[i, j].tick_params(axis='both', which='both', labelsize=8)
                # 设置 x 和 y 轴的范围
                # axes[i, j].set_xlim(0, 2)  # 根据实际情况设置范围
                axes[i, j].set_ylim(-0.1, 1.1)  # 根据实际情况设置范围

                # 去掉坐标轴方框
                # axes[i, j].spines['top'].set_color('none')
                # # axes[i, j].spines['bottom'].set_color('none')
                # # axes[i, j].spines['left'].set_color('none')
                # axes[i, j].spines['right'].set_color('none')

                row_index += 1   # 更换下一个 treatment

        axes[i, j].legend()

        # 调整子图之间的间距
        plt.tight_layout()

        # 显示图形
        plt.show()

    # def visualization(self, bandwidth_list, error_for_bandwidth_list, evaluation_method):
    #     colors = ['r', 'g', 'b', 'y', 'pink']
    #     # bandwidth-error plot
    #     plt.figure()
    #     plt.plot(bandwidth_list, error_for_bandwidth_list, marker='o')
    #     plt.xlabel('bandwidth')
    #     plt.ylabel(f"{evaluation_method.upper()}")
    #     plt.show()



# if __name__ == "__main__":
#     # 反事实生存函数估计和真实函数图像对比
#     # treatment_idx = np.random.randint(low=0, high=len(treatment_grid)-1, size=3)
#     treatment_idx = equal_space(length=len(treatment_grid), indices_num=5)
#     j = 1
#     for idx, color in zip(treatment_idx, colors):
#         survival_est = counterfactual_survival[idx, :]
#         plt.step(time_grid, survival_est, where="post", label=f"survival_est_{j}", color=color)
#         survival_t = survival_true_values[idx, :]
#         plt.step(time_grid, survival_t, where="post", label=f"survival_true_{j}", color=color, linestyle='--')
#         j += 1
#     plt.legend()
#     plt.xlabel('time')
#     plt.ylabel('survival probability')
#
#     # 将图像保存到本地
#     desktop = os.path.expanduser("~/Desktop")
#     filename = f"h{int(100 * h)}.png"
#     filepath = os.path.join(desktop, filename)
#     plt.savefig(filepath)
#     plt.show()


    # def estimate_error_mse(self, survival_pred, treatment_testSet, lambda_testSet):
    #     # MSE 评估 treatment 中位数的生存函数估计
    #     a_median = np.median(treatment_testSet)
    #     idx = np.argmin(np.abs(self.treatment_grid - a_median))  # 长度和 df_test 一致
    #     survival_estimate_a = survival_pred[idx, :].reshape(1, -1)
    #     survival_true_a = survival_true(self.survival_distribution, self.treatment_grid[idx], self.time_grid,
    #                                     treatment_testSet=treatment_testSet, lambda_testSet=lambda_testSet)
    #     mse = mean_squared_error_normalization(survival_estimate_a, survival_true_a, self.time_grid)
    #     return mse
    #
    # def estimate_error_imse(self, survival_pred, treatment_testSet, lambda_testSet):
    #     # IMSE 评估生存函数估计
    #     survival_true_values = survival_true(self.survival_distribution, self.treatment_grid, self.time_grid,
    #                                          treatment_testSet=treatment_testSet, lambda_testSet=lambda_testSet)
    #     imse = integrated_mean_squared_error_normalization(survival_pred, survival_true_values, self.time_grid)
    #     return imse

    # def estimate_error_rmse(self, survival_pred, treatment_testSet, lambda_testSet):
    #     survival_true_values = survival_true(self.survival_distribution, self.treatment_grid, self.time_grid,
    #                                          treatment_testSet=treatment_testSet, lambda_testSet=lambda_testSet)
    #     rmse = restricted_mean_squared_error(survival_pred, survival_true_values, self.time_grid)
    #     return rmse

    # def median_survival_time_bias(self, survival_pred, treatment_testSet, lambda_testSet):
    #     median_survival_time_pred = median_survival_time(survival_pred, self.time_grid)
    #     survival_true_values = survival_true(self.survival_distribution, self.treatment_grid, self.time_grid,
    #                                          treatment_testSet=treatment_testSet, lambda_testSet=lambda_testSet)
    #     median_survival_time_true = median_survival_time(survival_true_values, self.time_grid)
    #     median_survival_time_bias = median_survival_time_pred - median_survival_time_true
    #
    #     row_index, col_index = subset_index(median_survival_time_bias.shape, row_num=5, col_num=1)  # 取 5 个 等距 treatment
    #     median_survival_time_bias_output = subset(median_survival_time_bias, row_index, col_index)
    #     return median_survival_time_bias  # 5*1 矩阵



    # evaluation
    # df_test = pd.read_excel(self.data_path + "data.xlsx", sheet_name='test')



# def data_generate(self, test_size=0.2):
#     sim = SimulationModel(survival_distribution=self.surv_distribution,
#                           risk_type='linear',
#                           alpha=1,
#                           beta=1
#                           )
#     dataset = sim.generate_data(num_samples=self.sample_num, num_features=4,
#                                 feature_weights=[-2, 1, 2] + [1],  # beta  gamma
#                                 treatment_weights=[4, 2, 1])  # W
#     dataset.columns = ['x1', 'x2', 'x3', 'a', 'o', 'e', 'lambda']
#     self.train_data, self.test_data = train_test_split(dataset, test_size=test_size)
#     return self.train_data, self.test_data











