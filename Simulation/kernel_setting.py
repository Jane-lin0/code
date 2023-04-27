import numpy as np
from scipy.stats import gaussian_kde


def gaussian_kernel(a_approx, a, h):
    '''
    @param a_approx: float,point
    @param a: center parameter
    @param h: bandwidth, sigma
    @return: pdf of gaussian
    '''
    kernel_values = []
    for ai in a_approx:
        val = np.exp(-(ai-a) ** 2 / (2 * h ** 2)) / (np.sqrt(2 * np.pi) * h)
        kernel_values.append(val)
    kernel_values = np.array(kernel_values)
    # .reshape(-1,len(a_approx))
    return kernel_values


# 优化：尝试其他 kernel

# x = 3
# a = 2
# h= 0.7
# kde = gaussian_kernel(x,a, h=h)