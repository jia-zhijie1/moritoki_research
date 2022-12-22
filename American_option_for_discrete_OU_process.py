import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import json
with open('parameter.json') as p:
    parameter = json.load(p)


S_0 = parameter["OU_Process"]["x_0"]
alpha = parameter["OU_Process"]["alpha"]
sigma = parameter["OU_Process"]["sigma"]
n = parameter["Deep_Learning"]["num_partition"]
T = parameter["OU_Process"]["T"]
K = parameter["Option"]["strike_price"]
r = 0.1  #interest rate


def Delta(n=100, T=1, sigma=1.2): # from sobajima's Delta
    delta = sigma * np.sqrt(T / np.floor(n*T)) #floor:ガウス記号
    return delta

def up_probability(n=100, T=1, alpha=1.5, sigma=3.1,i=1,j=1):
    delta=Delta(n, T, sigma)
    mas= alpha/(sigma**2) #式を短く書けるため、特に意味ありません
    k=2*j-i
    p =  np.exp(-mas*k*delta)/(np.exp(-mas*k*delta)+np.exp(mas*k*delta))
    return p

def discrete_price(n=4, T=1,  sigma=3.1,K=0.2,S_0=1):#普通の二項モデルによるアメリカンプットオプション　ｋで売る権利
    S= np.zeros((n+3, n+3))
    V= np.zeros((n+3, n+3))
    delta =sigma * np.sqrt(T / n)
    u=np.exp(delta)
    d=np.exp(-delta)
    p= (np.exp(r*(T/n))-d)/(u-d)
    
    for i in range(0,n+1):
        for j in range(i+1):
            S[i+1,j+1] = S_0*(u**j)*((d)**(i-j))
        
    for j in range(n+1):
        V[n+1,j+1]=max(K-S[n+1,j+1],0)

    for i in range(n+1):
        i=n+1-i
        for j in range(1,i+1):
            V[i,j] = max(K-S[i,j],np.exp(-r*(T/n))*(p*V[i+1,j+1]+(1-p)*V[i+1,j]))
    return(V[1,1])

def price_discrete_OU_process(n=4, T=1, alpha=1.5, sigma=3.1,K=0.2,S_0=1):
    S= np.zeros((n+1, n+1)) 
    V= np.zeros((n+1, n+1))
    delta = Delta(n, T, sigma)
    u=np.exp(delta)
    d=np.exp(-delta)
    
    for i in range(0,n+1):
        for j in range(i+1):
            S[i,j] = S_0*( u**j)*( d**(i-j))
        
    for j in range(n+1):
        V[n,j]=max(K-S[n,j],0) #プットオプション、ゴールオプションならmax(S[n,j]-K,0)

    for i in range(n):
        i=n-i
        for j in range(1,i+1):
            p= up_probability(n, T, alpha, sigma, i-1,j-1)
            V[i-1,j-1] = max(K-S[i-1,j-1],np.exp(-r*(T/n))*(p*V[i,j]+(1-p)*V[i,j-1]))
    return(V[0,0])#,V,S



def pltplt(a,b,c):#分割の区間[ａ，ｂ]とｃステップ
    Num_partition = np.arange(a,b,c) #等差数列を生成
    y_np = np.zeros([Num_partition.shape[0],2])
    for n in Num_partition:
        for i in np.where(Num_partition==n):
            i=int(i)
        print("-----------------------------------------------------")
        print("n={}".format(n))
        # OU model case theoretical value
        price_discrete_OU_model_case = price_discrete_OU_process(n, T=5/12, alpha=1.5, sigma=0.2,K=60,S_0=62)
        y_np[i,0]= price_discrete_OU_model_case
        print("OU_model_case:")
        print(price_discrete_OU_model_case)

        #discrete  case thoretical value
        discrete_part = discrete_price(n, T=5/12, sigma=0.2,K=60,S_0=62)
        y_np[i,1]= discrete_part
        print("discrete_case:")
        print(discrete_part)
    print("-----------------------------------------------------")

    for i in range(y_np.shape[1]):
        plt.plot(Num_partition,y_np[:,i])
    plt.show()
