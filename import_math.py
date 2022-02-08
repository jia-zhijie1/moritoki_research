import math
import numpy as np
import matplotlib.pyplot as plt


firstX = []
Time = []

T = 1
num_partition = 100
len_partition = T / num_partition

firstX = []
Time = []
for i in range(num_partition):
    #windows版
    # #df = pd.read_csv(filepath_or_buffer="/Users/mk042.DESKTOP-K9G0PKU/Desktop/sss21moritoki/csv_experiment/to_csv_out_{}.csv".format(i), encoding="ms932", sep=",")
    
    # #mac版
    df = pd.read_csv(filepath_or_buffer="/Users/garammasala/sss21/csv_experiment/to_csv_out_{}.csv".format(i), encoding="ms932", sep=",")
    df = df.sort_values(by="X")
    for j in range(len(df["value"])):
        Time.append(T-(i+1)*len_partition)
        if (df["value"].iloc[i])**2 < 1e-6:
            X.append(df["value"].iloc[j])
            break
        elif j == len(df["value"]) -1:
            value = [x ** 2 for x in df["value"]]
            firstX.append(df["X"][np.argmin(value)])

plt.plot(Time,firstX)
