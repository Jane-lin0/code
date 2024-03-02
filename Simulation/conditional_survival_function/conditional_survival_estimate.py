import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored


def get_x_y(dataset, col_event, col_time):
    y = np.empty(dtype=[(col_event, bool), (col_time, np.float64)], shape=dataset.shape[0])
    y[col_event] = (dataset[col_event] == 1).values
    y[col_time] = dataset[col_time].values
    x = dataset.drop([col_event, col_time], axis=1)  # 除了event和time列，剩余是X
    # x = dataset.drop([col_event, col_time], axis=1)
    # x = dataset.drop([col_event, col_time, 'lambda'], axis=1)  # 除了event和time列，还应该去除lambda参数列，剩余是X
    return x, y


def conditional_survival_estimate(df_train, df_test, time_grid):
    """
    估计条件生存函数
    @param df_train: 训练集
    @param df_test: 测试集
    @param time_grid: 估计条件生存函数的时间点
    @return: conditional_survival_estimates: (len(test_samples), len(time_grid))
    """
    x_train, y_train = get_x_y(df_train, col_event='e', col_time='o')
    x_test, y_test = get_x_y(df_test, col_event='e', col_time='o')

    estimator = CoxPHSurvivalAnalysis()
    estimator.fit(x_train, y_train)

    pred_survival = estimator.predict_survival_function(x_test)  # test set 中每个样本点的生存函数估计
    conditional_survival_estimates = []
    # time_grid = estimator.event_times_  # time_grid 如何设置？基于 test set 的生存时间设置
    # time_points = np.arange(1, 1000)
    for i, survival_func in enumerate(pred_survival):
        # for time in time_grid:
        #     conditional_survival_estimates.append(survival_func(time))
        conditional_survival_estimates.extend(survival_func(time_grid))

    conditional_survival_estimates = np.array(conditional_survival_estimates).reshape(len(df_test), len(time_grid))

    return conditional_survival_estimates


# df_train = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='train')
# df_test = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='test')
#
# cse, t_grid = conditional_survival_estimate(df_train, df_test)
#
# cse_col1 = cse[:, 0]

# x_train, y_train = get_x_y(df_train, col_event='e', col_time='o')
# x_test, y_test = get_x_y(df_test, col_event='e', col_time='o')

# estimator.fit(x_train, y_train)

# coef = pd.Series(estimator.coef_, index=data_x_numeric.columns)
# pred_surv = estimator.predict_survival_function(x_test.iloc[:5])
# # time_points = np.arange(1, 1000)
# time_points = estimator.event_times_
# for i, surv_func in enumerate(pred_surv):
#     plt.step(time_points, surv_func(time_points), where="post", label="Sample %d" % (i + 1))
# plt.legend()
# plt.show()

# prediction = estimator.predict(x_test)
# score = estimator.score(x_test, y_test)  # 0.67


# rsf = RandomSurvivalForest(random_state=123)
#
# param_grid = {'n_estimators': np.array([int(i) for i in np.linspace(1,100,num=100)])}
# grid_search = GridSearchCV(rsf, param_grid)
# grid_search.fit(x_train, y_train)
# score = grid_search.score(x_test, y_test)


# rsf.fit(x_train, y_train)
# survival_funcs = rsf.predict_survival_function(x_test.iloc[:6])
#
# # for i, s in enumerate(survival_funcs):
# #     plt.step(rsf.event_times_, s, where="post", label=str(i))
# for fn in survival_funcs:
#     plt.step(fn.x, fn(fn.x), where="post")
# # fn.x 表示时间 t，fn(fn.x) 表示对应时间 t 的生存函数值
# plt.ylabel("Survival probability")
# plt.xlabel("Time in days")
# plt.legend()
# plt.show()
#
# score = rsf.score(x_test, y_test)  # 0.56






# x_train = df_train['x'][:, np.newaxis]
# # y_train = pd.concat([df_train['e'],df_train['o']],axis=1)
# y_train = []
# for i in range(len(df_train)):
#     event_indicator = (df_train['e'][i] == 1)
#     time = df_train['o'][i]
#     yi = f'({event_indicator}, {time})'
#     # (False, 2178.),(True, 0.000951634919868563)
#     y_train.append(yi)
# y_train = np.array(y_train)


# kmf = KaplanMeierFitter()
# def conditional_survival_estimate(survival_time, event_indicator):
#     """
#     和 x 无关
#     @param survival_time: df['t']
#     @param event_indicator: df['event']
#     @return: survival_estimated: ndarray:(len(df['t'])+1,)
#              time_grid: ndarray:(len(df['t'])+1,)
#     """
#     kmf.fit(survival_time, event_indicator)
#     survival_estimated = kmf.survival_function_['KM_estimate'].values
#     time_grid = kmf.survival_function_.index.values
#     return survival_estimated, time_grid

# df = data_generate(N=100)
# survival_estimated,time_grid = conditional_survival_estimate(df['o'],df['e'])




