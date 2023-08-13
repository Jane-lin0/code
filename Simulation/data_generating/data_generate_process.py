import pandas as pd
import numpy as np
from random import sample
from sklearn.model_selection import train_test_split, KFold


def time_moderate(df_train, df_test):
    """
    将 df_train 的时间范围调整到在 df_test 之内，即 df_train['o'] 被包含于 df_test['o']，IBS 的计算要求
    @param df_train:
    @param df_test:
    @return:
    """
    while df_train['o'].min() < df_test['o'].min():
        idx_train = df_train['o'].idxmin()  # int
        row_to_test = df_train.loc[idx_train]  # series
        df_train = df_train.drop(idx_train)

        # idx_test = df_test['o'].idxmin()
        idx_test = np.random.choice(df_test.index, size=1).item()
        # 不加 item 是ndarray，row_to_train = df_test.loc[idx_test]是dataframe，为保持一致，转化成int
        row_to_train = df_test.loc[idx_test]
        df_test = df_test.drop(idx_test)

        df_train = pd.concat([df_train, row_to_train.to_frame().T], axis=0)
        df_test = pd.concat([df_test, row_to_test.to_frame().T], axis=0)

    while df_train['o'].max() > df_test['o'].max():
        idx_train = df_train['o'].idxmax()
        row_to_test = df_train.loc[idx_train]
        df_train = df_train.drop(idx_train)

        # idx_test = df_test['o'].idxmax()
        idx_test = np.random.choice(df_test.index, size=1).item()
        row_to_train = df_test.loc[idx_test]
        df_test = df_test.drop(idx_test)

        df_train = pd.concat([df_train, row_to_train.to_frame().T])
        df_test = pd.concat([df_test, row_to_test.to_frame().T])

    return df_train, df_test


def train_validation_split(df, cv, save_path):
    kf = KFold(n_splits=cv, shuffle=True)  # 不设置 random_state，避免重复
    i = 0
    for train_index, validation_index in kf.split(df):
        df_train = df.loc[train_index]
        df_validation = df.loc[validation_index]

        # df_train, df_test = time_moderate(df_train, df_test)  # 调整时间，避免计算综合 brier score 时报错

        df_train.sort_values(by='o', ascending=True, inplace=True)
        df_validation.sort_values(by='o', ascending=True, inplace=True)
        # # 是否要排序？要排序，一是便于后续条件生存函数的估计,二是排序后样本的顺序和treatment的顺序一致，否则 IBS 的计算有误

        # df_train.sort_values(by='a', ascending=True, inplace=True)
        # df_test.sort_values(by='a', ascending=True, inplace=True)   # 便于对比输出的反事实结果？不需要对比

        # 将数据存储到本地
        writer = pd.ExcelWriter(save_path + f"data{i}.xlsx", engine='xlsxwriter')
        df_train.to_excel(writer, sheet_name='train', index=False)
        df_validation.to_excel(writer, sheet_name='validation', index=False)
        # writer.save()
        writer.close()
        i += 1


def train_test_data_split(dataset, test_size, save_path):
    """
    @param dataset: the full dataset, including train, validation and test part
    @param test_size:
    @param save_path: the save path of test data
    @return: the train data to be split for validation and test set
    """
    df_train, df_test = train_test_split(dataset, test_size=test_size)  # train test split，不设置 random_state，避免重复
    df_train.to_excel(save_path + "data.xlsx", sheet_name='train', index=False)
    df_test.to_excel(save_path + "data.xlsx", sheet_name='test', index=False)
    return df_train


# N = 1000
# path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# data_generate(N, path)


# n = 10000
# df = data_generate(N=n)
# df_train, df_test = train_test_split(df, test_size=0.25, random_state=123)
# df_train.sort_values(by='o', ascending=True, inplace=True)
# df_test.sort_values(by='o', ascending=True, inplace=True)  # 排序便于后面估计的对应
# # print(df_train.describe())
# # print(df_test.describe())
#
# # 将数据存储到桌面
# writer = pd.ExcelWriter(path+'data.xlsx',engine='xlsxwriter' )
# df_train.to_excel(writer,sheet_name='train',index=False)
# df_test.to_excel(writer,sheet_name='test',index=False)
# writer.save()

# def time_moderate(df_train, df_test):
#     """
#     将 df_test 的时间范围调整到在 df_train 之内
#     @param df_train:
#     @param df_test:
#     @return:
#     """
#     while df_test['o'].min() < df_train['o'].min():
#         idx_test = df_test['o'].idxmin()  # int
#         row_to_train = df_test.loc[idx_test]  # series
#         df_test = df_test.drop(idx_test)
#
#         # idx_test = df_test['o'].idxmin()
#         idx_train = np.random.choice(df_train.index, size=1).item()  # 为保持一致，转化成int
#         row_to_test = df_train.loc[idx_train]
#         df_train = df_train.drop(idx_train)
#
#         df_train = pd.concat([df_train, row_to_train.to_frame().T], axis=0)
#         df_test = pd.concat([df_test, row_to_test.to_frame().T], axis=0)
#
#     while df_test['o'].max() > df_train['o'].max():
#         idx_train = df_test['o'].idxmax()
#         row_to_test = df_test.loc[idx_train]
#         df_test = df_test.drop(idx_train)
#
#         # idx_test = df_train['o'].idxmax()
#         idx_test = np.random.choice(df_train.index, size=1).item()
#         row_to_train = df_train.loc[idx_test]
#         df_train = df_train.drop(idx_test)
#
#         df_train = pd.concat([df_train, row_to_train.to_frame().T])
#         df_test = pd.concat([df_test, row_to_test.to_frame().T])
#
#     return df_train, df_test
