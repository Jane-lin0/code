from lifelines import KaplanMeierFitter
from data_generate_process import data_generate

kmf = KaplanMeierFitter()


def conditional_survival_estimate(survival_time, event_indicator):
    """
    @param survival_time: df['t']
    @param event_indicator: df['event']
    @return: survival_estimated: ndarray:(len(df['t'])+1,)
             time_grid: ndarray:(len(df['t'])+1,)
    """
    kmf.fit(survival_time, event_indicator)
    survival_estimated = kmf.survival_function_['KM_estimate'].values
    time_grid = kmf.survival_function_.index.values
    return survival_estimated, time_grid

# df = data_generate(n=100)
# survival_estimated,time_grid = conditional_survival_estimate(df['t'],df['e'])


# 在时间 tj 下，不同 A 和 X 对应的 S(t|A,X) 取值如何计算？

