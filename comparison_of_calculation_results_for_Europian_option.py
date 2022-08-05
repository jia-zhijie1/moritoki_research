from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


import discrete_OU_model_generator

import json
with open('parameter.json') as p:
    parameter = json.load(p)

x_0 = parameter["OU_Process"]["x_0"]
alpha = parameter["OU_Process"]["alpha"]
sigma = parameter["OU_Process"]["sigma"]
K = parameter["Option"]["strike_price"]
n = parameter["Deep_Learning"]["num_partition"]
T = parameter["OU_Process"]["T"]

Num_partition = []
Difference = []

mu_hat = x_0 * np.exp(- alpha * T)
sigma_hat_p2 = sigma**2  * (1 - np.exp(- 2 * alpha * T)) / ( 2 * alpha)

#OUとdiscrete OUの関数が同じならこっち
theoretical_price_OU_model_case = np.exp(mu_hat + sigma_hat_p2 /2) * norm.sf((np.log(K) - mu_hat - sigma_hat_p2)/np.sqrt(sigma_hat_p2), loc=0, scale=1) - K * norm.sf((np.log(K) - mu_hat)/np.sqrt(sigma_hat_p2), loc=0, scale=1)


p = discrete_OU_model_generator.up_probability
delta = None
sum_probability_dic = {1:{0: 1 - p(n, T, alpha, sigma, x_0), 1: p(n, T, alpha, sigma, x_0)}}
sum_probability = None

def weight_deltas(k):
    return [i * delta for i in range (-k, k+1, 2)]

def sum_probability_for_weight(k=3, position = 3):
    if k in sum_probability_dic.keys():
        if position in sum_probability_dic[k].keys():
            pass
        else:
            if position == k:
                sum_probability = p(n, T, alpha, sigma, x_0 + weight_deltas(k-1)[k-1]) * sum_probability_for_weight(k-1, k-1)
            elif position == 0:
                sum_probability = (1 - p(n, T, alpha, sigma, x_0 + weight_deltas(k-1)[0])) * sum_probability_for_weight(k-1, 0)
                #sum_probability = np.prod([1 - p(alpha, x_0 - i * delta) for i in range(k)])
            else:
                if k-1 in sum_probability_dic.keys():
                    if position in sum_probability_dic[k-1].keys():
                        pass
                    else:
                        sum_probability1 = sum_probability_for_weight(k-1, position)
                        sum_probability_dic[k-1][position] = sum_probability1
                    if position - 1 in sum_probability_dic[k-1].keys():
                        pass
                    else:
                        sum_probability2 = sum_probability_for_weight(k-1, position - 1)
                        sum_probability_dic[k-1][position - 1] = sum_probability2
                else:
                    sum_probability1 = sum_probability_for_weight(k-1, position)
                    sum_probability_dic[k-1] = {position : sum_probability1}
                    sum_probability2 = sum_probability_for_weight(k-1, position - 1)
                    sum_probability_dic[k-1][position - 1] = sum_probability2
                sum_probability = p(n, T, alpha, sigma, x_0 + weight_deltas(k-1)[position - 1]) * sum_probability_dic[k-1][position - 1] + (1 - p(n, T, alpha, sigma, x_0 + weight_deltas(k-1)[position])) * sum_probability_dic[k-1][position]
            sum_probability_dic[k][position] = sum_probability
    
    else:
        if position == k:
            sum_probability = p(n, T, alpha, sigma, x_0 + weight_deltas(k-1)[position - 1]) * sum_probability_for_weight(k-1, position - 1)
        elif position == 0:
            sum_probability = (1 - p(n, T, alpha, sigma, x_0 + weight_deltas(k-1)[0])) * sum_probability_for_weight(k-1, 0)
        else:
            sum_probability1 = sum_probability_for_weight(k-1, position)
            sum_probability_dic[k-1] = {position : sum_probability1}
            sum_probability2 = sum_probability_for_weight(k-1, position - 1)
            sum_probability_dic[k-1][position - 1] = sum_probability2
            sum_probability = p(n, T, alpha, sigma, x_0 + weight_deltas(k-1)[position - 1]) * sum_probability_dic[k-1][position - 1] + (1 - p(n, T, alpha, sigma, x_0 + weight_deltas(k-1)[position])) * sum_probability_dic[k-1][position]

        sum_probability_dic[k] = {position : sum_probability}
    
    return sum_probability_dic[k][position]


def theoritical_price_discrete_OU_model_case(n=100, time=0.01, K = 0.2):
    k = (int) (n * time)
    weights = [x_0 + deltas for deltas in weight_deltas(k)]
    #print([np.exp(y) for y in weights])
    probabilities = [sum_probability_for_weight(k, i) for i in range(k+1)]
    #print(probabilities)
    return sum([max(0, np.exp(weight) - K) * p for weight, p in zip(weights, probabilities)])




import sys
sys.setrecursionlimit(50000000)

import time

start = time.time()
for n in range(100, 301, 10):
    print("-----------------------------------------------------")
    print("n={}".format(n))
    Num_partition.append(n)

    # OU model case theoretical value
    print("OU_model_case:")
    print(theoretical_price_OU_model_case)

    #discrete OU model case thoretical value
    delta = discrete_OU_model_generator.Delta(n, T, sigma)
    sum_probability_dic = {1:{0: 1 - p(n, T, alpha, sigma, x_0), 1: p(n, T, alpha, sigma, x_0)}}
    sum_probability = None

    discrete_part = theoritical_price_discrete_OU_model_case(n, T, K)
    print("discrete_OU_model_case:")
    print(discrete_part)
    



    print("difference:")
    print(abs(discrete_part - theoretical_price_OU_model_case))
    Difference.append(abs(discrete_part - theoretical_price_OU_model_case))
    
    """
    if n % 10 == 0:
        print(sum_probability_dic)
    """
print("-----------------------------------------------------")
print("経過時間:{}".format(time.time() - start))


fig, ax = plt.subplots()
ax.plot(Num_partition, Difference)
ax.set_xlabel('n: num_partition')
ax.set_ylabel('difference')
plt.show()