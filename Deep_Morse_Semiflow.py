#パッケージのインストール
import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

#
#from functorch import grad, vmap, vjp, jacrev
from torch.autograd import grad

#OU_Processを生成するファイル
import OU_model_generator
simulate_OU = OU_model_generator.simulate_OU

#学習時間を計測
import time
#学習状況の進捗を表示
from tqdm import tqdm

#各種パラメータ
#parameter.jsonを参照する
import json
with open('parameter.json') as p:
    parameter = json.load(p)

num_epoch = parameter["Deep_Learning"]["num_epoch"]
num_data = parameter["Deep_Learning"]["num_data"]
num_partition = parameter["Deep_Learning"]["num_partition"]
learning_rate = parameter["Deep_Learning"]["learning_rate"]
num_testdata = parameter["Deep_Learning"]["num_testdata"]

epsilon = parameter["Hyperparameter"]["epsilon"]

K = parameter["Option"]["strike_price"]

T = parameter["OU_Process"]["T"]
alpha = parameter["OU_Process"]["alpha"]
sigma = parameter["OU_Process"]["sigma"]
x_0 = parameter["OU_Process"]["x_0"]

h = 1e-4
r = 0.01  #interest rate
q = 0.008  #dividend
T = parameter["OU_Process"]["T"]  #expiration
len_partition = T / num_partition

#CPUとGPUどっちも使えるようにするやつ(Macはそんなに意味ないかも)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


#Phi オプションを行使したときのペイオフ、(アメリカン)オプションなので(x-K)^+
def phi(X):
    y = [[float(max(x - K, 0))] for x in X]
    y = torch.tensor(y, requires_grad=True, dtype=torch.float32)
    return y


#データを生成(dataloaderとdatageneratorを使う)
def OU_process_gen(n=1000):
    OU_processes = np.array([simulate_OU(n, T, alpha, 0, sigma, x_0, False)[1]])
    for i in range(n-1):
        OU_processes = np.append(OU_processes, np.array([OU_model_generator.simulate_OU(n, T, alpha, 0, sigma, x_0, False)[1]]), axis=0)
    """X = torch.tensor(np.random.uniform(low=a, high=b, size=n).reshape(n, 1),
                     requires_grad=False,
                     dtype=torch.float32)
    y = phi(X)
    dataset = TensorDataset(X, y)
    dataloder = DataLoader(dataset, batch_size=100, shuffle=True)"""
    return OU_processes.T

def data_loder(X, y):
    dataset = TensorDataset(X, y)
    dataloder = DataLoader(dataset, batch_size=100, shuffle=True)
    return dataloder


#数値微分
def differential(f, x):
    return (f(x + h) - f(x - h)) / (2 * h)


#2階微分
def sec_diff(f, x):
    return (differential(f, x + h) - differential(f, x - h)) / (2 * h)


#x2階微分のところの計算
def times(X, Y):
    xy = torch.mul(X, Y)
    return torch.tensor(xy, requires_grad=True, dtype=torch.float32)


#オリジナルの誤差関数を定義
class CustomLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, diff, Phi):
        #新しい版(スライド見る感じこっち？)
        #loss = (diff ** 2 + (1/epsilon) ** 2 * F.relu(target - output) ** 2).mean()
        #loss = ((target - output) ** 2 / len_partition + diff ** 2 + (1/epsilon) ** 2 * F.relu(target - output) ** 2).mean()
        loss = ((target - output)**2 / len_partition + diff**2 +
                (1 / epsilon)**2 * F.relu(Phi - output)**2).mean()
        return loss



"""def test():
    batch_size, feature_size = 3, 5
    weights = torch.randn(feature_size, requires_grad=True)
    def model(feature_vec):
        # Very simple linear model with activation
        return feature_vec.dot(weights).relu()
    examples = torch.randn(batch_size, feature_size)
    result = vmap(grad(model))(examples)
    print(result)


def test2():
    
    batch_size, feature_size = 3, 5
    def model(weights, feature_vec):
        # Very simple linear model with activation
        assert feature_vec.dim() == 1
        return feature_vec.dot(weights).relu()
    def compute_loss(weights, example, target):
        y = model(weights, example)
        return ((y - target) ** 2).mean()  # MSELoss
    weights = torch.randn(feature_size, requires_grad=True)
    examples = torch.randn(batch_size, feature_size)
    targets = torch.randn(batch_size)
    inputs = (weights, examples, targets)
    print(*inputs)
    grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)"""

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad

def main():
    """test2()
    def test_model(x):
        return model(x.reshape(-1, 1))"""
    OU_processes = OU_process_gen(num_data)
    y = phi(np.array([[input_data] for input_data in OU_processes[num_partition]]))

    for i in tqdm(range(1,num_partition+1)):
        X = torch.tensor(np.array([[input_data] for input_data in OU_processes[num_partition - i]]), requires_grad=False,
                     dtype=torch.float32)
        traindataloder = data_loder(X, y)

        model = Net()
        model = model.to(device)
        criterion = CustomLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        for n in range(num_epoch):
            for inputs, target in traindataloder:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)
                #diff = (model(inputs + h) - model(inputs - h)) / (2 * h)
             
                #print(inputs)
                #diff = jacrev(jacrev(model))
                #diff = gradient(inputs, output)
                diff = sec_diff(model, inputs)
                
                Phi = phi(inputs).to(device)
                loss = criterion(output, target, diff, Phi)
                loss.backward()
                optimizer.step()

            #if n % 50 == 0:
            #print(loss)

        """#テストデータをとる
        X = data_gen(n=num_testdata)[0]
        X = X.to(device)
        Xs = X.cpu().clone().detach().numpy().flatten().tolist()

        # t偏微分 (model(X) - pre_model(X)) / len_partition
        t_diff = (pre_model(X).to(device) - model(X)) / len_partition
        t_diff = t_diff.cpu().clone().detach().numpy()
        t_diffs = t_diff.flatten().tolist()
        #t_diffs.append(t_diff)

        #x1階微分
        x_diff = differential(model, X)
        x_diffs = x_diff.cpu().clone().detach().numpy().flatten().tolist()

        # x2階微分 sec_diff(model,X)
        #XX = torch.mul(X,X)
        #xx_diff = (-1) * (1/2) * torch.mul(XX,sec_diff(model,X)).to(device)
        xx_diff = (1 / 2) * sec_diff(model, X).to(device)
        xx_diffs = xx_diff.cpu().clone().detach().numpy().flatten().tolist()

        #-(r-q) * X x_diff
        #rqx_xdiff = (-1) * (r-q)
        #r * u 
        ru = r * model(X)
        ru = ru .cpu().clone().detach().numpy().flatten().tolist()

        #評価式　-(t偏微分) - 1/2 x^2 (x2階偏微分)
        left = [x + y for x, y in zip(t_diffs, xx_diffs)]
        #left = [x + y + z for x, y, z in zip(t_diffs, xx_diffs, ru)]

        #右側の計算
        right = (model(X).to(device) - phi(X).to(device)).flatten().tolist()

        #評価
        value = [max(x, y) for x, y in zip(left, right)]

        dataarray = np.array(
            [Xs, t_diffs, x_diffs, xx_diffs, left, right, value]).T

        value_df = pd.DataFrame(dataarray,
                                columns=[
                                    "X", "t_diff", "x_diff", "(1/2)xx_diff",
                                    "left", "right", "value"
                                ])

        #csvファイル書き出し
        #Windowsならこっち
        #value_df.to_csv("/Users/mk042.DESKTOP-K9G0PKU/Desktop/sss21moritoki/csv/to_csv_out_{}.csv".format(i))

        #macならこっち
        value_df.to_csv(
            "/Users/garammasala/sss21/csv/to_csv_out_{}.csv".format(i))"""

        y = model(X).clone().detach().requires_grad_(True)

    """firstX = []
    Time = []
    for i in range(num_partition):
        #windows版
        # #df = pd.read_csv(filepath_or_buffer="/Users/mk042.DESKTOP-K9G0PKU/Desktop/sss21moritoki/csv_experiment/to_csv_out_{}.csv".format(i), encoding="ms932", sep=",")

        # #mac版
        df = pd.read_csv(
            filepath_or_buffer="/Users/garammasala/sss21/csv/to_csv_out_{}.csv"
            .format(i),
            encoding="ms932",
            sep=",")
        df = df.sort_values(by="X")
        Time.append(T - (i + 1) * len_partition)
        for j in range(len(df["value"])):
            if (df["left"].iloc[i]) > 0 and (df["right"].iloc[i]) > 0:
                firstX.append(df["X"].iloc[j])
                break
            if j == len(df["value"]) - 1:
                value = [x**2 for x in df["right"]]
                firstX.append(df["X"][np.argmax(value)])
    plt.plot(Time, firstX)
    plt.show()"""


if __name__ == '__main__':
    main()