import pandas as pd

from Simulation.data_generating.DGP_pysurvival import SimulationModel

sim = SimulationModel( survival_distribution='exponential',
                       risk_type='linear',
                       alpha=1,
                       beta=1
                       )


def data_generate(N):
    dataset = sim.generate_data(num_samples = N, num_features=2,
                            feature_weights=[-1,1],
                            treatment_weights=[2])
    # lambda = exp(-1 * x + 1 * a) * alpha , a = 2 * x
    dataset.columns = ['x', 'a', 'o', 'e']
    dataset.sort_values(by='o',ascending=True,inplace=True)  # 便于后续条件生存函数的估计
    return dataset


n = 1000
df = data_generate(N=n)
n1 = int(n*0.7)
n2 = int(n*0.85)
df_train = df.iloc[:n1, :]
df_validation = df.iloc[n1:n2, :]
df_test = df.iloc[n2:, :]

# 将数据存储到桌面
writer = pd.ExcelWriter('C:/Users/janline/Desktop/data.xlsx',engine='xlsxwriter' )
df_train.to_excel(writer,sheet_name='train',index=False)
df_validation.to_excel(writer,sheet_name='validation',index=False)
df_test.to_excel(writer,sheet_name='test',index=False)
writer.save()
