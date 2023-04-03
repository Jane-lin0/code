import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from lifelines import CoxPHFitter

# Set seed for reproducibility
np.random.seed(123)

# Number of observations
n = 1000

# Generate covariates
x1 = np.random.normal(loc=0,scale=1,size=n)
x2 = np.random.binomial(n=1,p=0.5,size=n)
x3 = np.random.uniform(low=-1,high=1,size=n)

# Generate true coefficients
beta = np.array([1,-2,3,4])

# Generate treatment variable
# epsilon = np.random.normal(size=n)
epsilon = np.random.gumbel(loc=0,scale=1,size=n)  # epsilon 的期望约等于-0.5772

# Calculate true linear predictor
x_beta = beta[0] + beta[1] * x1 + beta[2] * x2 + beta[3] * x3 - epsilon
# epsilon服从标准极值分布，T服从指数分布

# Calculate true survival time
# lambd = np.exp(-1*x_beta)  # regression model
beta_para = np.exp(x_beta)  # regression model

t = np.random.exponential(scale=beta_para, size=n)  # scale是均值

# Set censoring time
censoring_time = beta_para + 1 # 右删失，定时删失
# censoring_time = np.random.normal(scale=expect_lifetime,size=n) #随机删失


# Generate censoring indicator
delta = np.where(t <= censoring_time, 1, 0)
delta_sum=sum(delta) # 事件发生率为0.7~0.8

observed_time = np.minimum(t, censoring_time)

# Create survival data
surv_data = pd.DataFrame({'time': observed_time, 'event': delta, 'x1': x1, 'x2': x2, 'x3': x3})

# 创建Kaplan-Meier估计器对象
kmf = KaplanMeierFitter()

# 计算生存函数
kmf.fit(surv_data['time'], event_observed=surv_data['event'])
# kmf.fit(df['time'], event_observed=df['event'])

# 输出生存曲线
# kmf.plot_survival_function()
survival_pred=kmf.survival_function_at_times(times=beta_para) # Return a Pandas series of the predicted survival value at specific times
print(f'survival_pred:{survival_pred}')