#パッケージのインストール
import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import pandas as pd

import csv

import matplotlib.pyplot as plt

import math

import mk_folder

#別ファイルでパラメータを設定
import parameter

#学習時間を計測
import time
#学習状況の進捗を表示
from tqdm import tqdm

#各種パラメータ
num_epoch = parameter.num_epoch
num_data = parameter.num_data
num_partition = parameter.num_partition #時間分割の数
epsilon = parameter.epsilon
a = parameter.a #一様分布
b = (-1)*a #一様分布
K = parameter.K #行使価格
T = parameter.T #expiration

learning_rate = parameter.learning_rate
num_testdata = parameter.num_testdata
h = parameter.h #微分するときの微小変化
r = parameter.r #interest rate
q = parameter.q #dividend
alpha = (r-q-1)/2
beta = alpha**2 + r
len_partition = T / num_partition



#CPUとGPUどっちも使えるようにするやつ(Macはそんなに意味ないかも)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:0')
#print(device)
print(torch.cuda.is_available())
#print(torch.cuda.current_device())

#モデルを構築
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


#phi
def phi(X):
    y = [[float((math.e)**(alpha * x) * max((math.e)**x - 1, 0))] for x in X]
    y = torch.tensor(y, requires_grad = True, dtype=torch.float32)
    return y

#データを生成(dataloaderとdatageneratorを使う)
def data_gen(n=1000):
    X = torch.tensor(np.random.uniform(
        low=a, high=b, size=n).reshape(n, 1), requires_grad = False, dtype=torch.float32)
    y = phi(X)
    dataset = TensorDataset(X, y)
    dataloder = DataLoader(dataset, batch_size=100, shuffle=True)
    return X, dataloder


#数値微分
def differential(f, x):
    return (f(x+h) - f(x-h)) / (2 * h)

#2階微分
def sec_diff(f,x):
    return (differential(f, x+h) - differential(f, x-h)) / (2 * h)

#x2階微分のところの計算
def times(X,Y): 
    xy = torch.mul(X,Y)
    return torch.tensor(xy, requires_grad = True, dtype=torch.float32)



#オリジナルの誤差関数を定義
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, diff, Phi):
        loss = ((target - output)**2 / len_partition + diff**2 + beta * output**2 + (1/epsilon) * F.relu(Phi-output)).mean()
        #loss = ((target - output)**2 / len_partition + diff**2 + beta * output**2).mean()
        return loss


def main():
    traindataloder = data_gen(n = num_data)[1]
    pre_model = phi
    file_path = mk_folder.mk_folder(T=T, num_partition=num_partition)
    
    #テストデータをとる
    testdata_X = data_gen(n = num_testdata)[0]
    testdata_X = testdata_X.to(device)
    Xs = testdata_X.cpu().clone().detach().numpy().flatten().tolist()

    for i in tqdm(range(num_partition)):
        model = Net()
        model = model.to(device)
        criterion = CustomLoss()
        optimizer = optim.SGD(model.parameters(), lr = learning_rate)
        
        for n in range(num_epoch):
            for inputs, target in traindataloder:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)
                diff = (model(inputs + h) - model(inputs - h)) / (2 * h)
                Phi = phi(inputs).to(device)
                loss = criterion(output, target, diff,Phi)
                #print(loss)
                loss.backward()
                optimizer.step()

            if n % 50 == 0:
                print(loss)

        Us = model(testdata_X).cpu().clone().detach().numpy().flatten().tolist()
        
        #t偏微分 (model(X) - pre_model(X)) / len_partition
        t_diff = (pre_model(testdata_X).to(device) - model(testdata_X).to(device)) / len_partition
        t_diffs = t_diff.cpu().clone().detach().numpy().flatten().tolist()
        #t_diffs.append(t_diff)

        #x1階偏微分
        x_diff = differential(model,testdata_X).to(device)
        x_diffs = x_diff.cpu().clone().detach().numpy().flatten().tolist()

        # x2階偏微分 sec_diff(model,X)
        xx_diff = sec_diff(model,testdata_X).to(device)
        xx_diffs = xx_diff.cpu().clone().detach().numpy().flatten().tolist()


        """#r * u 
        ru = r * model(X)
        ru = ru .cpu().clone().detach().numpy().flatten().tolist()"""

        Bus = [beta * y for y in Us]

        #評価式　(t偏微分) - 1/2 x^2 (x2階偏微分)
        left = [x - y + z for x, y, z in zip(t_diffs, xx_diffs, Bus)]
        #left = [x + y + z for x, y, z in zip(t_diffs, xx_diffs, ru)]
        

        #右側の計算
        phis = phi(testdata_X).cpu().clone().detach().numpy().flatten().tolist()
        right = [x - y for x, y in zip(Us, phis)]

        #評価
        value = [min(x, y) for x, y in zip(left, right)]

        #元々のSの計算
        Ss = [K * (math.e ** x) for x in Xs]

        #元々のCの計算
        Cs = [K * y / (math.e ** (alpha * x)) for x, y in zip(Xs, Us)]

        #元々のPhiの計算
        C_Phi = [K * max((math.e)** x - 1, 0) for x in Xs]
        C_Phis = C_Phi

        boundary = [x - y for x, y in zip(Cs,C_Phis)]
        
        dataarray =np.array([Xs, t_diffs, x_diffs ,xx_diffs, Bus, left, right, value, Ss, Cs, C_Phis, boundary]).T
        
        value_df = pd.DataFrame(dataarray, columns = ["X", "t_diff", "x_diff", "(1/2)xx_diff", "Beta_u", "left", "right", "value", "S", "C", "Phi", "boundary_check"])
        value_df = value_df.sort_values(by="S")
        
        #csvファイル書き出し
        #Windowsならこっち
        #value_df.to_csv("/Users/mk042.DESKTOP-K9G0PKU/Desktop/sss21moritoki/csv/to_csv_out_{}.csv".format(i))

        #macならこっち
        value_df.to_csv(file_path+"/to_csv_out_{}.csv".format(i))

        X = data_gen(n = num_data)[0].to(device)
        y = model(X).clone().detach().requires_grad_(True).to(device)
        pre_model = model
        pre_model = pre_model.to(device)
        dataset = TensorDataset(X, y)
        traindataloder = DataLoader(dataset, batch_size=100, shuffle=True)

    firstS = []
    Time = []
    for i in range(num_partition):
        # #mac版
        df = pd.read_csv(filepath_or_buffer=file_path+"/to_csv_out_{}.csv".format(i), encoding="ms932", sep=",")
        df = df.sort_values(by="S")
        for j in range(len(df["value"])):
            if (df["boundary_check"].iloc[j]) < 1e-2 and (df["boundary_check"].iloc[j]) > 0:
                firstS.append(df["S"].iloc[j])
                Time.append(T - (i+1)*len_partition)
                break
    plt.plot(Time,firstS,linewidth=3)
    s_max = K * (math.e ** b)
    plt.ylim([0,s_max])
    plt.xlim([0,T])
    plt.show()



if __name__ == '__main__':
    main()