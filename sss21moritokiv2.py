#パッケージのインストール
import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

#各種パラメータ
num_epoch = 1000
learning_rate = 0.001
num_data = 2000
a = 0 #一様分布
b = 1 #一様分布
K = 0.2 #行使価格
num_testdata = 2000
h = 1e-5

#CPUとGPUどっちも使えるようにするやつ(Macはそんなに意味ないかも)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

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


#データを生成(dataloaderとdatageneratorを使う)
def data_gen(n=1000):
    X = torch.tensor(np.random.uniform(
        low=a, high=b, size=n).reshape(n, 1), requires_grad = True, dtype=torch.float32)
    y = [[float(max(x - 0.2, 0))] for x in X]
    y = torch.tensor(y, requires_grad = True, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    dataloder = DataLoader(dataset, batch_size=100, shuffle=True)
    return dataloder

#数値微分
def diff(f, x):
    h = 1e-4
    nd = (f(x+h) - f(x-h))/(2 * h)
    return nd

#オリジナルの誤差関数を定義
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, diff):
        #loss = ((output - target) ** 2).mean()
        loss = ((output - target) ** 2 + diff ** 2).mean()
        return loss


#モデルに学習させる
def main():
    model = Net()
    model = model.to(device)
    criterion = CustomLoss()
    traindataloder = data_gen(n = num_data)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    losses = []

    for n in range(num_epoch):
        for inputs, target in traindataloder:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            target = target.to(device)
            inputs.retain_grad()
            output = model(inputs)
            diff = (model(inputs + h) - model(inputs - h)) / (2 * h)
            #outputsum = torch.sum(model(inputs))
            #outputsum.backward(retain_graph=True)
            #print(inputs.grad)
            loss = criterion(output, target, diff)
            losses.append(loss)
            #loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if n % 50 == 0:
            print(loss)
    print("学習完了")
    """plt.plot(losses)
    plt.title("sample")
    plt.show()"""
    #検証
    testdata = torch.tensor(np.random.uniform(low=a, high=b, size = num_testdata).reshape(num_testdata, 1), dtype=torch.float32)

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