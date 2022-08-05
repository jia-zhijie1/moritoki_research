import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#確率pを普通のOU_processとスケーリングを合わせるために変更する。
def up_probability(n=100, T=1, alpha=1.5, sigma=1.2, potential=0.5):
    p =  np.exp(-(alpha * potential / sigma) * np.sqrt(T/n))/(np.exp((alpha * potential / sigma) * np.sqrt(T/n)) + np.exp(-(alpha * potential / sigma) * np.sqrt(T/n)))
    return p


def Delta(n=100, T=1, sigma=1.2):
    delta = sigma * np.sqrt(T / n)
    return delta


def discrete_OU_model_generator (n=100, T=1, alpha=1.5, sigma=1.2, z_0 = 0, fig_mode=False):
    delta_t = 1 * T / n
    partition = [i * delta_t for i in range(n*T + 1)]

    Z = np.zeros(n*T + 1)
    #Z[0] = z_0

    for i in range(1, n*T+1):
        increment = np.random.choice([Delta(n, T, sigma), -Delta(n, T, sigma)], p=[up_probability(alpha,Z[i-1]), 1-up_probability(alpha,Z[i-1])])
        Z[i] = Z[i-1] + increment

    data_array = np.array([partition, Z]).T

    #DataFrameでまとめる
    df = pd.DataFrame(data_array, columns=['timestamp', 'process'])

    if fig_mode:
        fig, ax = plt.subplots()

        # plot the process X
        ax.plot(partition, Z, color='blue', label='process')
        ax.set_xlabel('time(s)')
        ax.set_ylabel('process X')

        # 以下はそんなに関係ないから気にしなくていい．
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')

        plt.legend()
        plt.show()

    return df, Z, partition


if __name__ == '__main__':
    import json
    with open('parameter.json') as p:
        parameter = json.load(p)

    num_path = parameter["OU_Process"]["num_path"]
    n = parameter["Deep_Learning"]["num_partition"]
    T = parameter["OU_Process"]["T"]
    alpha = parameter["OU_Process"]["alpha"]
    sigma = parameter["OU_Process"]["sigma"]
    z_0 = parameter["OU_Process"]["x_0"]

    fig, ax = plt.subplots()
    for i in range(num_path):
        df, Z, partition = discrete_OU_model_generator(n, T, alpha, sigma, z_0, False)
        ax.plot(partition, Z)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('process Z')
    plt.show()