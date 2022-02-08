#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
# In[2]:


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


# #CPUとGPU両方使えるように

# In[3]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ## データセットを作る
# まずコードが動くかどうか確認したいので、100個ぐらいでOK

# # 一様分布のモデル

# In[4]:


def simulate_Unifrom(n = 100, a = 0, b = 100, fig_mode = False):
    # 一様分布に従う確率変数を生成
    X = np.random.uniform(a, b, n)
    
    data_array = np.array([X]).T
    df = data_array
    #df = pd.DataFrame(data_array, columns = ['process'])
    
    if fig_mode:
        fig, ax = plt.subplots()

        # plot the process X and volatility sigma
        ax.plot(X, label = 'process')
        ax.set_ylabel('process X')

        # 以下はそんなに関係ないから気にしなくていい．
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')

        plt.legend()
    
    return df


# # ハイパーパラメータの設定

# In[5]:


#partitionの数
n = 20000
#満期
T = 1 
#リスクフリーレート
r = 0.001
#行使価格
K = 0.2


# In[6]:


#初期値の設定
#要らないかも
X_0 = 1
mu = 0
sigma = 1


# #データの作成

# In[7]:


#一様分布のパラメータ
a = 0
b = 1


# #DATAs = []
# DATAs_t = []
# for _ in range(N):
#     df_path = simulate_Unifrom(n = n, a = a, b = b, fig_mode = False)
#     X = df_path["process"].values
#     DATAs.append(df_path.values)
#     Phi_data = []
#     for x in df_path["process"]:
#         Phi_x = max( (x - K) , 0)
#         Phi_data.append(Phi_x)
#     DATAs_t.append(Phi_data)
# DATAs = np.array(DATAs)
# DATAs_t = np.array(DATAs_t)
# #Phiの取り方がまだいまいちよく分からん　一旦放置

# In[8]:


DATAs = simulate_Unifrom(n = n, a = a, b = b, fig_mode = False)


# In[9]:


def datagenerator(n = 100000, a = 0, b = 1, batch_size = 100):
    X = simulate_Unifrom(n = n, a = a, b = b, fig_mode = False)
    y = [[float(max(x - K, 0))] for x in X]
    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    return dataloader


# ## ニューラルネットワーク層を作る
# まずは1層か2層でOK

# #モデルの作成

# In[10]:


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear( 1, 128 )
        self.fc2 = nn.Linear( 128, 256 )
        self.fc3 = nn.Linear( 256, 128 )
        self.fc4 = nn.Linear( 128, 64 )
        self.fc5 = nn.Linear( 64, 1 )

    def forward(self, x):
        # フォワードパスを定義
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# In[11]:


# モデル（NeuralNetworkクラス）のインスタンス化
model = NeuralNetwork()


# ## 誤差関数を定義

# 最小化したいのは、
# $ \int (|\Phi(x)-u(x)|^2) + u'(x))dx$

# 今は一様分布で、上の期待値をモンテカルロで数値計算して考える

# In[12]:


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        loss = nn.MSELoss()(output, target)
        return loss


# In[13]:


criterion = CustomLoss()


# #最適化

# In[14]:


optimizer = optim.SGD(model.parameters(), lr = 0.001)


# #データ

# In[15]:


traindata  = datagenerator(n, a, b)


# #学習

# In[16]:


epochs = 100


# In[17]:


model = model.to(device)


# In[18]:


for epoch in range(epochs):
    loss = 0
    for inputs, target in traindata:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    if epoch%50 == 0:
        print(loss)


# #テスト(誤差を見る)

# In[19]:


DATA_test = torch.Tensor(simulate_Unifrom(n = 10, a = a, b = b, fig_mode = False))


# In[20]:


DATA_test


# In[21]:


model(DATA_test)


# In[25]:


torch.tensor([[float(max(x - K, 0))] for x in DATA_test])


# $\Phi(x) = (x -K)^+$

# In[ ]:


loss_fn = CustomLoss( )


# In[ ]:


loss_fn(model(DATA_test), torch.Tensor(teacher1_data( DATA_test )), DATA_test)


# # メモ
