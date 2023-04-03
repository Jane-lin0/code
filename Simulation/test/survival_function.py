from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd

# set parameters for event and censoring distributions
lam = 0.1  # rate parameter for exponential distribution
tmax = 100  # maximum event time
cmin = 50  # minimum censoring time
cmax = 150  # maximum censoring time

# generate uncensored event times
event_times = np.random.exponential(1/lam, size=100)

# generate censoring times
censoring_times = np.random.uniform(cmin, cmax, size=100)

# determine which event times are censored
censored = event_times >= censoring_times

# create DataFrame with event times, censoring times, and censored indicator
df = pd.DataFrame({'EventTime': event_times, 'CensoringTime': censoring_times, 'Censored': censored})


# 创建Kaplan-Meier估计器对象
kmf = KaplanMeierFitter()

# 计算生存函数
kmf.fit(df['CensoringTime'], event_observed=df['Censored'])
# kmf.fit(df['time'], event_observed=df['event'])

# 输出生存曲线
kmf.plot_survival_function()
# kmf.plot()
