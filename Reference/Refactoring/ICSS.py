import numpy as np
from tqdm import tqdm


def compute_CUSUM(X):
    CUSUM = np.cumsum(X ** 2)
    return CUSUM


def compute_gamma(X, T, m):
    mean_X = np.mean(X)
    r = X - mean_X  # 잔차 계산
    r_squared = r ** 2
    sigma_squared = np.mean(r_squared)

    gamma = np.zeros(m + 1)
    for i in range(0, m + 1):
        gamma_i = np.sum((r_squared[i:T] - sigma_squared) * (r_squared[0:T - i] - sigma_squared))
        gamma[i] = gamma_i / T
    return gamma


def compute_lambda(gamma, m):
    lambda_hat = gamma[0] + 2 * np.sum((1 - np.arange(1, m + 1) / (m + 1)) * gamma[1:m + 1])
    return lambda_hat


def compute_D_prime(CUSUM, T, lambda_hat):
    k = np.arange(T)
    D_prime = (CUSUM - (k + 1) / T * CUSUM[-1]) / np.sqrt(lambda_hat)
    return D_prime


def compute_percentile(D_prime, T, percent=95):
    D_prime_abs = np.abs(D_prime) * np.sqrt(T / 2)
    critical_value = np.percentile(D_prime_abs, percent)
    return critical_value


def ol_detect(hour_df, diff_value, window_size, significant_level):
    lambda_lst = []
    cv_lst = []
    current_lst = []

    ol_lst = []

    for i in tqdm(range(window_size, len(diff_value) + 1, 1)):

        count = 0
        filter = []
        for val in ol_lst:
            if (i - window_size <= val) and (val < i):
                count += 1
                filter.append(val - (i))

        X = diff_value[i - window_size - count: i].copy()

        if len(filter) != 0:
            X = np.delete(X, filter)

        N = len(X)
        T = N
        m = int(T ** (1 / 4))

        # CUSUM 계산
        CUSUM = compute_CUSUM(X)
        # gamma 계산
        gamma = compute_gamma(X, T, m)
        # lambda_hat 계산
        lambda_hat = compute_lambda(gamma, m)
        lambda_lst.append(lambda_hat)
        # D_prime 계산
        D_prime = compute_D_prime(CUSUM, T, lambda_hat)
        # critical value 계산
        critical_value = compute_percentile(D_prime, T, significant_level)
        cv_lst.append(critical_value)
        # 현재 통계량 계산
        current_lst.append(np.abs(D_prime[-2]) * np.sqrt(T / 2))

        if current_lst[-1] > critical_value:
            ol_lst.append(i - 1)

    ol_lst = list(np.where((np.array(current_lst) > np.array(cv_lst)))[0] + window_size)

    ol_lst_time = []
    for point in ol_lst:
        ol_lst_time.append(hour_df.index[point])

    return ol_lst_time
