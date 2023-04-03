import numpy as np
from scipy.stats import gaussian_kde


def gaussian_kernel(x, a, h):
    '''
    @param x: point
    @param a: center parameter
    @param h: bandwidth, sigma
    @return: pdf of gaussian
    '''
    return np.exp(-(x-a) ** 2 / (2 * h ** 2)) / (np.sqrt(2 * np.pi) * h)


# 优化：尝试其他 kernel

# x = 3
# a = 2
# h= 0.7
# kde = gaussian_kernel(x,a, h=h)