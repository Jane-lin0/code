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
    term1 = np.trapz(survival_est**2, grid)
    term2 = np.trapz(survival_est * survival_true, grid)
    mse = term1 - 2 * term2
    return mse

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
    mise = np.mean(mse_list)
    return mise


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

