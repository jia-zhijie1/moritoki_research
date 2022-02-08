#パッケージのインストール
import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import pandas as pd
#import plotly.graph_objects as go

#import matplotlib.pyplot as plt

#学習時間を計測
import time
#学習状況の進捗を表示
from tqdm import tqdm

#各種パラメータ
num_epoch = 500
num_data = 1000
num_partition = 100 #number of partition
penalty = 1000

learning_rate = 0.001
a = -5 #一様分布
b = 5 #一様分布
K = 2 #行使価格
num_testdata = 100
h = 1e-4
r = 0.01 #risk free rate
T = 1 #expiration
len_partition = T / num_partition


#CPUとGPUどっちも使えるようにするやつ(Macはそんなに意味ないかも)
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
#device = torch.device('cuda:0')
#print(device)
print(torch.cuda.is_available())
print(torch.cuda.current_device())

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


#Phi
def phi(X):
    y = [[float(max(x - K, 0))] for x in X]
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
def diff(f, x):
    return (f(x+h) - f(x-h)) / (2 * h)

#2階微分
def sec_diff(f,x):
    return (diff(f, x+h) - diff(f, x-h)) / (2 * h)


#オリジナルの誤差関数を定義
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, diff):
        #loss = ((output - target) ** 2).mean()
        loss = ((output - target) ** 2 + diff ** 2).mean() + penalty * max((target.mean() - output.mean()), 0)
        return loss


def main():
    model = Net()
    model = model.to(device)
    criterion = CustomLoss()
    X, traindataloder = data_gen(n = num_data)

    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    #lossを保存
    #losses = []
    models = []
    testdata = torch.tensor(np.random.uniform(low=a, high=b, size = num_testdata).reshape(num_testdata, 1), dtype=torch.float32)
    t_diffs = []
    xx_diffs = []
    values = []
    times = []
    pre_model = phi

    for i in tqdm(range(num_partition)):
        #モデルに学習させる
        model = Net()
        model = model.to(device)
        for n in range(num_epoch):
            for inputs, target in traindataloder:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)
                diff = (model(inputs + h) - model(inputs - h)) / (2 * h)
                loss = criterion(output, target, diff)
                #losses.append(loss)
                loss.backward()
                optimizer.step()

            """if n % 50 == 0:
                print(loss)"""

        #テストデータをとる
        X = data_gen(n = num_testdata)[0]
        X = X.to(device)

        # 時間　(num_partition - i) / num_partition
        t = (num_partition - i) / num_partition
        times.append(t)
        
        # t偏微分 (model(X) - pre_model(X)) / len_partition
        t_diff = (pre_model(X).to(device) - model(X)) / len_partition
        t_diff = t_diff.cpu().clone().detach().numpy()
        t_diffs = t_diff.flatten().tolist()
        #t_diffs.append(t_diff)

        # x2階微分 sec_diff(model,X)
        xx_diff = (-1) * (1/2) * sec_diff(model,X).cpu().clone().detach().numpy()
        xx_diffs = xx_diff.flatten().tolist()

        #評価式　(t偏微分) - 1/2 (x2階偏微分)
        left = (t_diff + xx_diff).flatten().tolist()
        #value = value.clone().detach().numpy()
        #values.append(value)

        #右側の計算
        right = (model(X).to(device) - phi(X).to(device)).flatten().tolist()

        #評価
        value = [min(x, y) for x, y in zip(left, right)]

        
        dataarray =np.array([t_diffs, xx_diffs, left, right, value]).T
        #print(dataarray)
        value_df = pd.DataFrame(dataarray, columns = ["t_diff", "(-1/2)xx_diff", "left", "right", "value"])
        #print(value_df)
        #value_df.to_csv("/Users/mk042.DESKTOP-K9G0PKU/Desktop/sss21moritoki/csv/to_csv_out_{}.csv".format(i))
        value_df.to_csv("/Users/garammasala/sss21/csv/to_csv_out_{}.csv".format(i))#mac用
        """fig = go.Figure(data=[go.Table(
            columnwidth =  [5, 5, 5, 5], #カラム幅の変更
            header=dict( values=value_df.columns, align='center', font_size=10),
            cells=dict( values=value_df.values.T, align='center', font_size=10)
            )])

        fig.update_layout(title={'text': "simulating",'y':0.85,'x':0.5,'xanchor': 'center'})#タイトル位置の調整
        fig.layout.title.font.size= 12 #タイトルフォントサイズの変更
        fig.write_image("simulating.jpg")#,height=600, width=800)"""

        #y = torch.tensor(model(X), requires_grad = True, dtype=torch.float32)
        y = model(X).clone().detach().requires_grad_(True)
        pre_model = model
        pre_model = pre_model.to(device)
        dataset = TensorDataset(X, y)
        traindataloder = DataLoader(dataset, batch_size=100, shuffle=True)

    #print(values)
    

    """print("学習完了")
    print("学習時間")
    print("エポック数:", num_epoch, "学習データ数:", num_data)"""
    """plt.plot(torch.tensor(losses))
    plt.title("sample")
    plt.show()"""



    #print("ベンチマーク")
    

    #print("検証データ:")
    #print(testdata)

    #print("モデルの予測値:")
    #print(model(testdata))

    #print("テスト誤差:")
    """for _ in range(10):
        testdata = torch.tensor(np.random.uniform(low=a, high=b, size = num_testdata).reshape(num_testdata, 1), dtype=torch.float32)
        testoutput = model(testdata)
        testdiff = (model(testdata + h) - model(testdata - h)) / (2 * h)
        testtarget = torch.tensor([[float(max(x - 0.2, 0))] for x in testdata], requires_grad = True, dtype=torch.float32)
        testloss = CustomLoss()
        print(testloss(testoutput, testtarget, testdiff))

    #y_test = torch.tensor([[float(max(x - K, 0))] for x in testdata])
    #print("正解:")
    #print(y_test)"""


if __name__ == '__main__':
    main()