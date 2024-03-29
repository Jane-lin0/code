import numpy as np
from scipy import integrate
from scipy.stats import expon


def restricted_mean_squared_error(survival_est, survival_true, grid):
    grid = grid.flatten()
    survival_est = survival_est.flatten()
    survival_true = survival_true.flatten()
    rmse = (np.trapz(survival_est - survival_true, grid)) ** 2
    # rmse = np.trapz(survival_est - survival_true, grid)
    return rmse


def mean_squared_error(survival_est, survival_true, grid):
    """
    mean_squared_error at treatment a
    @param survival_est:
    @param survival_true:
    @param grid: time_grid
    @return: mean_squared_error
    """
    grid = grid.flatten()
    survival_est = survival_est.flatten()
    survival_true = survival_true.flatten()
    mse = np.trapz((survival_est - survival_true) ** 2, grid)
    return mse


def mean_squared_error_normalization(survival_est, survival_true, grid):
    normalization_term = np.trapz(survival_true ** 2, grid).item()
    mse = mean_squared_error(survival_est, survival_true, grid)
    return mse / normalization_term


'''
优化：真实生存函数用以下方式积分
integrate.quad(f, 0, 1)
'''


def integrated_mean_squared_error(survival_est, survival_true, grid):
    """
    integrated_mean_squared_error at all treatment
    @param survival_est:
    @param survival_true:
    @param grid: time grid
    @return:
    """
    mse_list = []
    for treatment_col in range(len(survival_est)):
        survival_est_col = survival_est[treatment_col, :]
        survival_true_col = survival_true[treatment_col, :]
        mse_col = mean_squared_error(survival_est_col, survival_true_col, grid)
        mse_list.append(mse_col)
    mise = np.sum(mse_list)
    return mise


def integrated_mean_squared_error_normalization(survival_est, survival_true, grid):
    empty_est = np.zeros(shape=survival_est.shape)
    normalization_term = integrated_mean_squared_error(empty_est, survival_true, grid)
    imse = integrated_mean_squared_error(survival_est, survival_true, grid)
    return imse / normalization_term


def survival_point_estimate(counterfactual_survival, treatment_point, time_point, treatment_grid, time_grid):
    treatment_idx = np.argmin(np.abs(treatment_grid - treatment_point))
    time_idx = np.argmin(np.abs(time_grid - time_point))

    return counterfactual_survival[treatment_idx, time_idx]


def survival_true(survival_distribution, treatment_grid, time_grid, x_beta):
    if survival_distribution == 'exponential':
        true_survival = np.empty(shape=(0, len(time_grid)))
        # for a in treatment_grid:
        for a in [treatment_grid]:   # 此时 treatment_grid 是 float
            survival_of_a = []
            for t in time_grid:
                survival_a_t = np.mean(-t * np.exp(x_beta - a))
                survival_of_a.append(survival_a_t)
            true_survival = np.vstack([true_survival, survival_of_a])
    else:
        true_survival = None

    return true_survival


# def survival_true(survival_distribution, treatment_grid, time_grid, u_0, u_1, arg_lambda):
#     if survival_distribution == 'exponential':
#         true_survival = np.empty(shape=(0, len(time_grid)))
#         for a in treatment_grid:
#             # idx = np.argmin(np.abs(treatment_testSet - a))
#             # x = feature_testSet[idx]
#             f = lambda x, t: np.exp(- arg_lambda(a, x) * t) / u_1
#             survival_func = np.vectorize(lambda t: integrate.quad(lambda x: f(x, t), u_0, u_0 + u_1)[0])  # 矢量化函数
#             # def survival_func(t):
#             #     return integrate.quad(lambda x: f(x, t), u_0, u_0 + u_1)[0]  # integrate.quad返回元组（result，error）
#             survival_of_a = survival_func(time_grid).reshape(1, -1)
#             true_survival = np.vstack([true_survival, survival_of_a])
#     else:
#         true_survival = None
#
#     return true_survival


# def survival_true(survival_distribution, treatment_grid, time_grid, treatment_testSet, lambda_testSet):
#     """
#     @param treatment_grid:
#     @param time_grid:
#     @param treatment_testSet: the treatment in test set
#     @param lambda_testSet: the parameter of exponential distribution in test set
#     @return: true counterfactual survival function
#     """
#     if survival_distribution == 'exponential':
#         true_survival = np.empty(shape=(0, len(time_grid)))
#         for a in treatment_grid:
#             lambda_idx = np.argmin(np.abs(treatment_testSet - a))
#             lambda_i = lambda_testSet[lambda_idx]
#             survival_a = []
#             for t in time_grid:
#                 survival_t = 1 - expon.cdf(t, scale=1 / lambda_i)   # exponential
#                 survival_a.append(survival_t)
#             true_survival = np.vstack([true_survival, survival_a])  # ndarray:(len(treatment_grid), len(time_grid))
#     else:
#         true_survival = None
#     return true_survival


def median_survival_time(survival_matrix, time_grid):
    """
    @param survival_matrix: ndarray: (len(treatment_grid), len(time_grid))，基于 treatment_grid 和 time_grid 上的生存概率
    # @param treatment_grid:
    @param time_grid:
    @return: treatment_grid 对应的中位生存时间
    treatment = a 时，Sa(t) = P( T(a) >= time_grid ) = 0.5 时对应的 time_grid
    """
    # n_treat = len(treatment_grid)
    n_treat = survival_matrix.shape[0]
    median_survival = []
    for i in range(n_treat):
        index = np.argmin(np.abs(survival_matrix[i, :] - 0.5))
        median_survival.append(time_grid[index])
    median_survival = np.array(median_survival)
    return median_survival


def integrated_brier_score():
    return ibs


def c_index():
    return cindex


def get_best_bandwidth(error_list, h_list):
    """
    @param error_list:
    @param h_list:
    @return: bandwidth with minimum error
    """
    min_error = min(error_list)
    min_error_idx = error_list.index(min_error)
    h_best = h_list[min_error_idx]
    return h_best

# def mean_squared_error(survival_est, survival_true, grid):
#     """
#     mean_squared_error at treatment a
#     @param survival_est:
#     @param survival_true:
#     @param grid: time_grid
#     @return: mean_squared_error
#     """
#     grid = grid.flatten()
#     survival_est = survival_est.flatten()
#     survival_true = survival_true.flatten()
#     term1 = np.trapz(survival_est**2, grid)
#     term2 = np.trapz(survival_est * survival_true, grid)
#     term3 = np.trapz(survival_true**2, grid)
#     mse1 = (term1 - 2 * term2)/term3 + 1   # 0.16502006915174572   # 误差较大，废弃，用 mse2 算
#     mse2 = np.trapz((survival_est - survival_true)**2, grid)       # 验证结果是否一致  # 0.03348476226538029
#     return mse1, mse2
