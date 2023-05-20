import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# 创建随机森林回归模型
regressor = RandomForestRegressor()


def feature_select(df_value, y):
    """
    @param df_value: X
    @param y:
    @return: selected X
    """
    # 训练回归模型
    regressor.fit(df_value, y)

    # 根据特征的重要性进行排序
    feature_importances = regressor.feature_importances_

    # 根据特征重要性选择阈值，选择最重要的特征
    # threshold = np.mean(feature_importances)  # 可以根据需求进行调整
    threshold = np.mean(feature_importances) + np.std(feature_importances)
    # threshold = 0.1

    # 创建特征选择器
    sfm = SelectFromModel(regressor, threshold=threshold)

    # 应用特征选择器到特征矩阵
    X_selected = sfm.transform(df_value)

    # 获取所选特征的索引
    selected_features = sfm.get_support(indices=True)

    # 获取所选特征的列名
    selected_feature_names = df_value.columns[selected_features]

    # 将新特征存储到 DataFrame
    df_selected = pd.DataFrame(X_selected, columns=selected_feature_names)

    return df_selected