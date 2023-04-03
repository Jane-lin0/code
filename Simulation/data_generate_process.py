import numpy as np
import pandas as pd

# Set seed for reproducibility
# np.random.seed(123)

def data_generate(n):
    # 设置误差项的分布
    eta1 = np.random.uniform(low=0, high=1, size=n)
    epsilon1 = np.random.randn(n)
    epsilon2 = np.random.randn(n)

    # x 对 t 和 y 的影响都是线性
    x = 3 + 4 * eta1  # 单个协变量，x 的期望为5
    a = 2 * x + epsilon1  # continuous treatment A 是期望为 2*x 的正态分布
    t = (a - 10) ** 2 + x + epsilon2  # potential outcome
    c = np.mean(t) + 1    # censor time   初步设定，需要再研究一下如何设置 censor time
    o = np.minimum(t, c)  # observed_time
    event = np.where(t <= c, 1, 0)  # delta = 1, event happens

    df = pd.DataFrame()
    df['x'] = x
    df['a'] = a
    df['t'] = t
    df['c'] = c
    df['o'] = o
    df['e'] = event

    df.sort_values(by='t', ascending=True, inplace=True)  # 输出按时间（potential outcome）排序的结果

    return df

df_test = data_generate(n=100)

