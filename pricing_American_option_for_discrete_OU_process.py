import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import discrete_OU_model_generator

p = discrete_OU_model_generator.up_probability
Delta = discrete_OU_model_generator.Delta


d = {}
val1 = None
val2 = None

def american_option_pricing(n=100, time=0.01, T=1, alpha=1.1, sigma=1.2, x=0.8, K=0.2, r=0.03):   
    delta = Delta(n, T, sigma)
    dt = T/n
    x_plus = x + delta
    x_minus = x - delta

    if abs(time - T) < 1e-5:
        return max(0, x-K)
  
    if time+dt in d.keys():
        if x_plus in d[time+dt].keys():
            pass
        else:
            val1 = american_option_pricing(n,time+dt, T, alpha, sigma, x_plus, K, r)
            d[time+dt][x_plus] = val1 
        if x_minus in d[time+dt].keys():
            pass
        else:
            val2 = american_option_pricing(n,time+dt, T, alpha, sigma, x_minus, K, r)
            d[time+dt][x_minus] = val2
    else:
        val1 = american_option_pricing(n, time+dt, T, alpha, sigma, x_plus, K, r)
        d[time+dt] = {x_plus : val1}
        val2 = american_option_pricing(n,time+dt, T, alpha, sigma, x_minus, K, r)
        d[time+dt][x_minus] = val2
    
    if time in d.keys():
        if x in d[time].keys():
            pass
        else:
            val = max(max(0, x-K), (p(alpha, x) * d[time+dt][x_plus] + (1-p(alpha, x)) * d[time+dt][x_minus])/(1+r))
            d[time][x] = val
    else:
        val = max(max(0, x-K), (p(alpha, x) * d[time+dt][x_plus] + (1-p(alpha, x)) * d[time+dt][x_minus])/(1+r))
        d[time] = {x:val}

    return d[time][x]


if __name__ == '__main__':
    import time
    n = 100
    t = 0.01
    T = 1
    alpha = 0.008
    sigma = 1.2
    x = 0.09
    K = 0.1
    r = 0.03

    start = time.time()

    print(american_option_pricing(n, t, T, alpha, sigma, x, K, r))
    print(f'total : {(time.time() - start)} second.')