import scipy.stats
import flexcode
from flexcode.regression_models import NN
import matplotlib.pyplot as plt

def conditional_density_estimate(df_train,df_validation,df_test,n_grid):
    """
    A|X 的条件密度估计
    @param df_train: 训练数据，要包含两列：单变量df_train['x'], 连续治疗df_train['a']
    @param df_validation: 调参数据
    @param df_test: 验证数据集
    @return: conditional_density_estimated: ndarray:(len(df_test['x']),n_grid)
             a_grid: ndarray:(n_grid,1)
    """
    model_flexcode = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",regression_params={"k":20})

    model_flexcode.fit(df_train['x'].values, df_train['a'].values)

    model_flexcode.tune(df_validation['x'].values,df_validation['a'].values)
    # flexcode_estimate_error = model_flexcode.estimate_error(df_test['x'].values, df_test['a'].values)
    conditional_density_estimated, a_grid = model_flexcode.predict(df_test['x'].values, n_grid=n_grid) # 返回 n_grid 个函数点

    return conditional_density_estimated, a_grid

# for i in range(10):
#     true_density = scipy.stats.norm.pdf(x=a_grid, loc=df_test['x'].values[i], scale=1)
#     plt.plot(a_grid, conditional_density_estimate[i, :],color = "blue")
#     plt.plot(a_grid, true_density, color = "green")
#     plt.axvline(x=df_test['x'].values[i], color="yellow")
#     plt.show()