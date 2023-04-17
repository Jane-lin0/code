import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from Simulation.data_generating.data_generate_process import data_generate
from sksurv.ensemble import RandomSurvivalForest

df_train = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='train')
df_validation = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='validation')
df_test = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='test')

df_train = pd.concat([df_train, df_validation], axis=0).reset_index()
x_train = df_train['x'][:, np.newaxis]
# y_train = pd.concat([df_train['e'],df_train['o']],axis=1)
y_train = []
for i in range(len(df_train)):
    event_indicator = (df_train['e'][i] == 1)
    time = df_train['o'][i]
    yi = f'({event_indicator}, {time})'
    # (False, 2178.),(True, 0.000951634919868563)
    y_train.append(yi)
y_train = np.array(y_train)
# x_test =


# def conditional_surv_estimation():

rsf = RandomSurvivalForest(random_state=123)
rsf.fit(x_train, y_train)


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




