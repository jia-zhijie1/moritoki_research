import csv
from numpy.core.fromnumeric import argmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

T = 1
linelen = 3
num_partition = 10
len_partition = T / num_partition

firstXn10 = []
Timen10 = []
for i in range(num_partition):
    df = pd.read_csv(filepath_or_buffer="/Users/garammasala/sss21/csv_N{}/to_csv_out_{}.csv".format(num_partition,i), encoding="ms932", sep=",")
    #df = df.sample(frac=1).reset_index(drop=True)
    for j in range(len(df["value"])):
        if (df["right"].iloc[j]) < 0:
            firstXn10.append(df["X"].iloc[j])
            Timen10.append(T-(i+1)*len_partition)
            break

plt.plot(Timen10, firstXn10,linewidth=linelen,label ="N=10")

num_partition = 30
len_partition = T / num_partition

firstXn30 = []
Timen30 = []
for i in range(num_partition):
    df = pd.read_csv(filepath_or_buffer="/Users/garammasala/sss21/csv_N{}/to_csv_out_{}.csv".format(num_partition,i), encoding="ms932", sep=",")
    #df = df.sample(frac=1).reset_index(drop=True)
    for j in range(len(df["value"])):
        if (df["right"].iloc[j]) < 0:
            firstXn30.append(df["X"].iloc[j])
            Timen30.append(T-(i+1)*len_partition)
            break

plt.plot(Timen30, firstXn30,linewidth=linelen,label ="N=30")

num_partition = 50
len_partition = T / num_partition

firstXn50 = []
Timen50 = []
for i in range(num_partition):
    df = pd.read_csv(filepath_or_buffer="/Users/garammasala/sss21/csv_N{}/to_csv_out_{}.csv".format(num_partition,i), encoding="ms932", sep=",")
    #df = df.sample(frac=1).reset_index(drop=True)
    for j in range(len(df["value"])):
        if (df["right"].iloc[j]) < 0:
            firstXn50.append(df["X"].iloc[j])
            Timen50.append(T-(i+1)*len_partition)
            break

plt.plot(Timen50, firstXn50,linewidth=linelen,label ="N=50")

num_partition = 100
len_partition = T / num_partition

firstXn100 = []
Timen100 = []
for i in range(100):
    df = pd.read_csv(filepath_or_buffer="/Users/garammasala/sss21/csv_N100/to_csv_out_{}.csv".format(i), encoding="ms932", sep=",")
    #df = df.sample(frac=1).reset_index(drop=True)
    for j in range(len(df["value"])):
        if (df["right"].iloc[j]) < 0:
            firstXn100.append(df["X"].iloc[j])
            Timen100.append(T-(i+1)*len_partition)
            break

plt.plot(Timen100, firstXn100,linewidth=linelen,label ="N=100")


plt.xlabel("Time", {"fontsize": 10})
plt.ylabel("X", {"fontsize": 10})
plt.tick_params(labelsize=10)
plt.legend(prop={"size": 10}, loc="best")
plt.grid()
plt.xlim([0,1])
plt.ylim([7.5,10])
plt.show()