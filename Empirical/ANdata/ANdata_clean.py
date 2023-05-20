import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from Empirical.ANdata.feature_selection import feature_select

data = pd.read_csv(r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\桌面\empirical_data\ANdata\ANdata.csv")
# print(data.isnull().sum())
# nan_index = np.isnan(data).any(axis=1)
# data = data[~nan_index]

# print(data['Locale.'].value_counts())
location = pd.get_dummies(data['Locale.'], prefix='location')
df_value = pd.concat([data.drop('Locale.', axis=1), location], axis=1)

df = pd.DataFrame()
# treatment
df['a'] = df_value['loginsPerExaminee']
df_value.drop('loginsPerExaminee', axis=1, inplace=True)

# observed outcome
df['o'] = df_value['passing2014'] - df_value['passing2013']
# passing_change = df_value['passing2014'] - df_value['passing2013']
# df['o'] = passing_change - np.min(passing_change)
df_value.drop(['passing2013', 'passing2014'], axis=1, inplace=True)
# df['o'] = df_value['meanScale2014'] - df_value['meanScale2013']
# df_value.drop(['meanScale2013', 'meanScale2014'], axis=1, inplace=True)

# censor
df['e'] = (df['o'] > 0) + 0
# df['e'] = (passing_change > 0) + 0
# print(df['o'].describe())
# print(np.mean(df['o'] > 0))

# covariance select by tree
df_selected = feature_select(df_value, df['o'])

# full data
df = pd.concat([df_selected, df], axis=1)
# df.to_excel(r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\桌面\empirical_data\ANdata\ANdata_clean.xlsx", index=False)


# train test split
kf = KFold(n_splits=5, shuffle=True, random_state=123)
save_path = r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\桌面\empirical_data\ANdata\ANdata"

i = 0
for train_index, test_index in kf.split(df):
    df_train = df.loc[train_index]
    df_test = df.loc[test_index]

    # 将数据存储到本地
    writer = pd.ExcelWriter(save_path + f"_{i}.xlsx", engine='xlsxwriter')
    df_train.to_excel(writer, sheet_name='train', index=False)
    df_test.to_excel(writer, sheet_name='test', index=False)
    writer.save()
    i += 1

print("dataset saved")
