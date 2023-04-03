from data_generate_process import data_generate
import numpy as np
import pandas as pd
import flexcode
from flexcode.regression_models import NN
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from scipy.stats import gaussian_kde




# a = np.array([[1,2,3],[4,5,6]])
# b = np.diag(a)


# a = np.array([i for i in range(5)])
# b = a[i]
# density_a = np.array([1,2,3])
# density_a_conditional_x = np.array([[1,2,3],[4,5,6]])
# pai = density_a / density_a_conditional_x

# def gaussian_kernel_smoothing(x, bandwidth=0.2):
#     kde = gaussian_kde(x, bw_method=bandwidth)
#     return kde(x)
#
# # 示例使用
# n = 100
# x = np.random.randn(n)
# y = gaussian_kernel_smoothing(x)




# df=pd.DataFrame(columns = ['1', '2', '3', '4', '5', '6'],
#                 index = ['treatment1','treatment2','treatment3'])
# df['treatment1'] = np.array([1.5,2.1,1.9,2.8,1.4,1.8])
# df['treatment2'] = np.array([1.8,2,2,2.7,1.6,2.3])
# df['treatment3'] = np.array([1.9,2.5,2.5,2.6,2.1,2.4])

# # load the data
# digits = load_digits()
#
# # project the 64-dimensional data to a lower dimension
# pca = PCA(n_components=15, whiten=False)
# data = pca.fit_transform(digits.data)
#
# # use grid search cross-validation to optimize the bandwidth
# params = {"bandwidth": np.logspace(-1, 1, 20)}
# grid = GridSearchCV(KernelDensity(), params)
# grid.fit(data)
#
# print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
#
# # use the best estimator to compute the kernel density estimate
# kde = grid.best_estimator_
#
# # sample 44 new points from the data
# new_data = kde.sample(44, random_state=0)
# new_data = pca.inverse_transform(new_data)
#
# # turn data into a 4x11 grid
# new_data = new_data.reshape((4, 11, -1))
# real_data = digits.data[:44].reshape((4, 11, -1))
#
# # plot real digits and resampled digits
# fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
# for j in range(11):
#     ax[4, j].set_visible(False)
#     for i in range(4):
#         im = ax[i, j].imshow(
#             real_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
#         )
#         im.set_clim(0, 16)
#         im = ax[i + 5, j].imshow(
#             new_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
#         )
#         im.set_clim(0, 16)
#
# ax[0, 5].set_title("Selection from the input data")
# ax[5, 5].set_title('"New" digits drawn from the kernel density model')
#
# plt.show()

# n=1000
# df_train = data_generate( n )
# x = np.random.normal( 0, 1, n )
# x.reshape(len(x),1)
# print(type(df_train['x']))
# print(type(x))