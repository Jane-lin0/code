import pandas as pd
from random import sample
from sklearn.model_selection import train_test_split
from Simulation.data_generating.DGP_pysurvival import SimulationModel

sim = SimulationModel( survival_distribution='exponential',
                       risk_type='linear',
                       alpha=1,
                       beta=1
                       )


def data_generate(N):
    dataset = sim.generate_data(num_samples = N, num_features=4,
                            feature_weights=[-2, 1, 2]+[1],
                            treatment_weights=[1, 0, 0])
    # lambda = exp(-1 * x + 1 * a) * alpha , a = 2 * x
    dataset.columns = ['x1', 'x2', 'x3', 'a', 'o', 'e']
    # dataset.sort_values(by='o',ascending=True,inplace=True)  # 便于后续条件生存函数的估计
    return dataset


n = 10000
df = data_generate(N=n)
df_train, df_test = train_test_split(df, test_size=0.25, random_state=123)
df_train.sort_values(by='o', ascending=True, inplace=True)
df_test.sort_values(by='o', ascending=True, inplace=True)  # 是否要排序？
# print(df_train.describe())
# print(df_test.describe())

# 将数据存储到桌面
writer = pd.ExcelWriter('C:/Users/janline/Desktop/data.xlsx',engine='xlsxwriter' )
df_train.to_excel(writer,sheet_name='train',index=False)
df_test.to_excel(writer,sheet_name='test',index=False)
writer.save()
