import numpy as np


def gaussian_kernel(x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def gaussian_kernel_smoothing(data, window_size=5, sigma=1):
    k = (window_size - 1) // 2
    smoothed_data = np.zeros_like(data)

    # 计算高斯核函数
    kernel = np.zeros(window_size)
    for i in range(-k, k + 1):
        kernel[i + k] = gaussian_kernel(i, sigma)

    # 执行高斯平滑
    for i in range(len(data)):
        kernel_slice = kernel[max(0, k - i):min(window_size, 2 * k + 1 - i)]
        data_slice = data[max(0, i - k):min(len(data), i + k + 1)]

        # 添加一个元素，使 kernel_slice 和 data_slice 的长度相同
        if len(kernel_slice) < len(data_slice):
            kernel_slice = np.append(kernel_slice, 0)
        elif len(kernel_slice) > len(data_slice):
            data_slice = np.append(data_slice, 0)

        smoothed_data[i] = np.sum(kernel_slice * data_slice) / np.sum(kernel_slice)

    return smoothed_data


data = np.array([1, 2, 3, 4, 5])
smoothed_data = gaussian_kernel_smoothing(data, window_size=3, sigma=1)