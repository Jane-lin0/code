import numpy as np
import scipy.stats
import flexcode
from flexcode.regression_models import NN
import matplotlib.pyplot as plt

# Generate data p(z | x) = N(x, 1)
def generate_data(n_draws):
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x.reshape((len(x), 1)), z.reshape((len(z), 1))

x_train, z_train = generate_data(10000)
x_validation, z_validation = generate_data(10000)
x_test, z_test = generate_data(10000)
# x_test = {ndarray:(10000,1)}
# z_test = {ndarray:(10000,1)}

# Parameterize model
model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",
                               regression_params={"k":20})
# KNeighborsRegressor

# Fit and tune model
model.fit(x_train, z_train)
model.tune(x_validation, z_validation)

# Estimate CDE loss
print(model.estimate_error(x_test, z_test))

# Calculate conditional density estimates
cdes, z_grid = model.predict(x_test, n_grid=200)
'''
x_test – ndarray:(10000,1)：A numpy matrix of covariates at which to predict
n_grid – int, the number of grid points at which to predict the conditional density
z_grid = np.linspace(0.0, 1.0, n_grid)

Returns:
A numpy matrix where each row is a conditional density estimate at the grid points
cdes={ndarray:(10000,200)}
z_grid={ndarray:(200,1)}
为 x_test 的每个值，都返回 200 个 z 的概率密度估计值
'''

for ii in range(10):
    true_density = scipy.stats.norm.pdf(x=z_grid, loc=x_test[ii], scale=1)
    plt.plot(z_grid, cdes[ii, :],color = "blue")
    plt.plot(z_grid, true_density, color = "green")
    # plt.axvline(x=z_test[ii], color="red")
    plt.axvline(x=x_test[ii], color="yellow")
    plt.show()

    '''
    # 为什么 true_density 是由 x_test 而不是 z_test 为均值生成的pdf ? 因为 z 的概率密度真实值是基于 x 生成的正态分布
    # z_grid 是生成 pdf 的横坐标范围，一般是一组等间距的数字。
    # 返回值 true_density 就是在这个正态分布上，z_grid 上每个位置上的概率密度值
    '''
