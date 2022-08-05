# OUモデルのデータを発生させる

#パッケージのインストール
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
'''
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
'''

e = math.e

import parameter

#ブラウン運動の発生
def brownian_motion(n, T):
    delta_t = 1 * T / n
    partition = [i * delta_t for i in range(n + 1)]

    # ブラウン運動の差分（平均：０，分散：時間の差分）
    random_increments = np.random.normal(loc=0.0,
                                         scale=np.sqrt(delta_t),
                                         size=n)
    '''
    where loc = "mean(average)", scale = "variance", n = the number of increments.
    '''
    # making data like a Brownian motion
    brownian_motion = np.cumsum(
        random_increments)  # calculate the brownian motion
    # insert the initial condition
    brownian_motion = np.insert(brownian_motion, 0, 0.0)

    return brownian_motion, random_increments, partition


#OUモデルの作成
def simulate_OU(n=100, T=1, theta=1, mu=0, sigma=1.2, x_0=0, fig_mode=False):
    # BMの作成
    random_increment = brownian_motion(n, T)[1]
    partition = brownian_motion(n, T)[2]

    #processと時間を格納するリストを作成
    X = np.zeros(n + 1)
    X[0] = x_0

    for i, t in zip(range(n + 1), partition):
        X[i] = np.exp(-theta *
                   t) * x_0 + mu * (1 - np.exp(-theta * t)) + sigma * sum([
                       np.exp((-theta * (t - u))) * dB
                       for u, dB in zip(partition[0:i+1], random_increment[0:i+1])
                   ])

    #print('X size : {}, partition size : {}'.format(X.size, len(partition)))

    data_array = np.array([partition, X]).T

    #DataFrameでまとめる
    df = pd.DataFrame(data_array, columns=['timestamp', 'process'])

    #プロットして表示
    if fig_mode:
        fig, ax = plt.subplots()

        # plot the process X
        ax.plot(partition, X, color='blue', label='process')
        ax.set_xlabel('time(s)')
        ax.set_ylabel('process X')

        # 以下はそんなに関係ないから気にしなくていい．
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')

        plt.legend()
        plt.show()

    return df, X, partition


#OUモデルの確認
if __name__ == '__main__':
    num_path = parameter.num_path
    n = parameter.num_partition
    T = parameter.T
    alpha = parameter.alpha
    sigma = parameter.sigma
    x_0 = parameter.x_0
    mu = parameter.mu

    fig, ax = plt.subplots()

    for i in range(num_path):
        df2, X, partition = simulate_OU(n, T, alpha, mu, sigma, x_0, False)
        print(X)
        ax.plot(partition, X)
    
    ax.set_xlabel('time(s)')
    ax.set_ylabel('OU process X')
    plt.show()
