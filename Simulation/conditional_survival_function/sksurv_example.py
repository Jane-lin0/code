import matplotlib.pyplot as plt
from sksurv.datasets import load_whas500
from sksurv.ensemble import RandomSurvivalForest

# 没有删失
X, y = load_whas500() # y 包含 event、time
X = X.astype(float)
estimator = RandomSurvivalForest().fit(X, y)
surv_funcs = estimator.predict_survival_function(X.iloc[:5]) # 估计前 5 个样本的生存函数

for fn in surv_funcs:
   plt.step(fn.x, fn(fn.x), where="post")
# fn.x 表示时间 t，fn(fn.x) 表示对应时间 t 的生存函数值

t = y[1]
# void 类型
plt.ylim(0, 1)
plt.show()