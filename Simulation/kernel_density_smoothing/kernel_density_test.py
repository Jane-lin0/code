import random
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
'''
核密度平滑是一种用于估计随机变量的概率密度函数的非参数方法。
它涉及通过在每个数据点放置一个核函数来估计数据点的密度，然后平滑生成的分布。
'''
# Generate some random data
# random.seed(123)
# X = np.random.randn(100)  # 生成100个来自标准正态分布的样本点

rng=np.random.RandomState(123)
X = rng.randn(100)[:,np.newaxis] # 生成100个来自标准正态分布的样本点

# Initialize the kernel density estimator
kde = KernelDensity(kernel='gaussian')

# Define the bandwidth parameter grid
param_grid = {'bandwidth': np.logspace(-1, 1, num=20,base=10)} # 生成 1/10 ~ 10 之间对数均匀的20个数
# 对于高斯核，bandwidth 是标准差

# Perform leave-one-out cross-validation to estimate the optimal bandwidth
cv = LeaveOneOut()
grid_search = GridSearchCV(kde, param_grid, cv=cv)
# grid_search = GridSearchCV(kde, param_grid, cv=10) # cv = 10 和 leave one out 结果一样，cv = 5 更不平滑
# grid_search = GridSearchCV(kde, param_grid, scoring='neg_mean_squared_error', cv=cv)
# 均方误差评估模型，非常不平滑，效果很差

grid_search.fit(X) # 拟合模型

best_bandwidth = grid_search.best_params_['bandwidth']

# kde_result = KernelDensity(kernel="gaussian", bandwidth=best_bandwidth).fit(X)

X_plot = np.linspace(-5,5,1000)[:,np.newaxis]
log_dens = grid_search.score_samples(X_plot)
# log_dens = kde_result.score_samples(X_plot)
pred_density_val = np.exp(log_dens)

true_density_val = norm.pdf(x=X_plot,loc=0,scale=1)

plt.plot(X_plot, pred_density_val)
plt.plot(X_plot, true_density_val)
plt.show()
