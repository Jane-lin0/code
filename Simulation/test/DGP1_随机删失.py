import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from lifelines import KaplanMeierFitter

# Generate covariates
n_samples = 1000
n_covariates = 3
X = np.random.normal(size=(n_samples, n_covariates))
# 生成具有 n_samples 行、n_covariates 列的数组，其中每个元素都是从标准正态分布中随机采样得到的数值

# Define treatment effect and hazard function
beta = np.array([0.5, -0.5, 0.2])

expect_lifetime = np.exp(np.dot(X, beta))

# Generate survival times from exponential distribution
# survival_times = np.random.exponential(1/hazard)
survival_times = np.random.exponential(expect_lifetime)

# Generate censoring times from uniform distribution
censoring_times = np.random.normal(scale=expect_lifetime,size=n_samples) #随机删失，censor 太少,event 0.948

# Create survival object and fit proportional hazards model
survival = np.minimum(survival_times, censoring_times)
censor=(survival_times<=censoring_times)+0
event_num=sum(censor)

kmf = KaplanMeierFitter()
kmf.fit(survival)

# cph = CoxPHFitter()
# cph.fit(X, survival, event_observed=survival_times)

# Visualize survival function and hazard function
kmf.plot_survival_function()

# Fit Cox proportional Ts model
# cph = CoxPHFitter()
# cph.fit(surv_data, duration_col='time', event_col='status', formula='x1 + x2')
# cph.print_summary()
