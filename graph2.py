import csv
from numpy.core.fromnumeric import argmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

T = 1
num_partition = 10
len_partition = T / num_partition

Xlist = []
firstX = []
Time = []
for i in range(10):
    Time.append(T-(i+1)*len_partition)
    #windows版
    #df = pd.read_csv(filepath_or_buffer="/Users/mk042.DESKTOP-K9G0PKU/Desktop/sss21moritoki/csv_experiment/to_csv_out_{}.csv".format(i), encoding="ms932", sep=",")

    #mac版
    df = pd.read_csv(filepath_or_buffer="/Users/garammasala/sss21/csv/to_csv_out_{}.csv".format(i), encoding="ms932", sep=",")
    df = df.sort_values(by="X",ascending=False)
    for j in range(len(df["value"])):
        if (df["left"].iloc[i]) > 0 and (df["right"].iloc[i]) > 0:
            firstX.append(df["X"].iloc[j])
        if j == (len(df["value"])-1):
            firstX.append(df["X"].iloc[argmax(df["right"])])
    Xlist.append(statistics.mean(firstX))
    

plt.plot(Time, firstX)
plt.xlim([0,1])
plt.ylim([-1,1])
plt.show()