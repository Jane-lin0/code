import numpy as np
from scipy.stats import expon


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
    mse = np.trapz((survival_est - survival_true)**2, grid)
    return mse


def mean_squared_error_normalization(survival_est, survival_true, grid):
    normalization_term = np.trapz(survival_true**2, grid).item()
    mse = mean_squared_error(survival_est, survival_true, grid)
    return mse / normalization_term


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


def integrated_brier_score():

    return ibs


def concordance_index():

    return cindex


def survival_true(treatment_grid, time_grid, df_test):
    true_survival = np.empty(shape=(0, len(time_grid)))
    for a in treatment_grid:
        lambda_idx = np.argmin(np.abs(df_test['a'] - a))
        lambda_i = df_test['lambda'][lambda_idx]
        survival_a = []
        for t in time_grid:
            survival_t = 1 - expon.cdf(t, scale=1 / lambda_i)
            survival_a.append(survival_t)
        true_survival = np.vstack([true_survival, survival_a])  # ndarray:(len(treatment_grid), len(time_grid))
    return true_survival

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
#     mse1 = (term1 - 2 * term2)/term3 + 1   # 0.16502006915174572  # 误差较大，废弃，用 mse2 算
#     mse2 = np.trapz((survival_est - survival_true)**2, grid)  # 验证结果是否一致  # 0.03348476226538029
#     return mse1, mse2