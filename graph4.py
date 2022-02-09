import csv
from numpy.core.fromnumeric import argmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
import parameter

T = parameter.T
num_partition = parameter.num_partition
len_partition = T / num_partition

K = parameter.K
b = parameter.b
s_max =K * (math.e ** b)

Xlist = []
firstS = []
Time = []
for i in range(num_partition):
    #windows版
    #df = pd.read_csv(filepath_or_buffer="/Users/mk042.DESKTOP-K9G0PKU/Desktop/sss21moritoki/csv_experiment/to_csv_out_{}.csv".format(i), encoding="ms932", sep=",")

    #mac版
    df = pd.read_csv(filepath_or_buffer="experiment1/to_csv_out_{}.csv".format(i), encoding="ms932", sep=",")
    #df = df.sort_values(by="X",ascending=False)
    #df["boundary_check"] = [x ** 2 for x in df["boundary_check"]]
    """min_number = df["boundary_check"].idxmin()
    firstS.append(df["S"].iloc[min_number])
    Time.append(T - (i+1)*len_partition)"""
    for j in range(len(df["value"])):
        if (df["boundary_check"].iloc[j]) < 0:
            firstS.append(df["S"].iloc[j])
            Time.append(T - (i+1)*len_partition)
            break
        elif (df["boundary_check"].iloc[j]) < 1e-3:
            firstS.append(df["S"].iloc[j])
            Time.append(T - (i+1)*len_partition)
            break
    
#print(firstS,Time)
plt.plot(Time, firstS,linewidth=5)
plt.xlim([0,T])
plt.ylim([0,s_max])
plt.show()