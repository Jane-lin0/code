import numpy as np
import pandas as pd


def generate_data(survival_distribution, sample_num, a):
    mean = np.array([4, 5, 0.5])
    covariance_matrix = np.array([[1, 0, 0.5],
                                 [0, 1, 0.5],
                                 [0.5, 0.5, 0.1]])
    samples = np.random.multivariate_normal(mean=mean, cov=covariance_matrix, size=sample_num)
    X = samples[:, :-1]
    beta = np.array([1, -1])
    treatment = samples[:, -1]
    if survival_distribution == 'exponential':
        # x = self.u_0 + self.u_1 * np.random.uniform(low=0, high=1, size=sample_num)
        # treatment = self.w * x + np.random.normal(loc=0, scale=1, size=sample_num)
        true_time = - np.log(np.random.uniform(low=0, high=1, size=sample_num)) * np.exp(- (np.dot(X, beta) - a))
        censor_time = np.random.uniform(low=0, high=np.max(true_time), size=sample_num)
        observed_time = np.minimum(true_time, censor_time)
        event = 1 * (observed_time == true_time)
        x_beta = np.dot(X, beta)
        dataset = pd.DataFrame(
            data=np.c_[X, treatment, observed_time, event],
            columns=['x1', 'x2', 'treatment', 'time', 'event']
        )
        print(f"event rate of dataset: {sum(event) / sample_num}")
        return dataset, x_beta    # 返回 x_beta 用于计算 Oracle真实生存函数

    elif survival_distribution == 'weibull':
        return None

    else:
        return None


if __name__ == "__main__":
    data = generate_data(survival_distribution='exponential', sample_num=100, a=0.2)