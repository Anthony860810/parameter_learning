import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset

PERTUBTIME = 2
A_data_len = [67 ,100 ,100 ,100]
batch_size = 10
lr = 0.001
epochs = 250
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CreatDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )
    def forward(self, x):
        out = self.layers(x)
        return out


def DataGenerate():
    feat_data = pd.read_csv("TSfeatA1Benchmark.csv",index_col = 0)
    lab_data = pd.read_csv("A1_Label.csv",index_col = 0)
    feat_data = feat_data.values
    lab_data = lab_data.values
    mean_thre = np.reshape(((lab_data[:,-1] + lab_data[:,-2])/2),(-1,1))
    lab_data = np.append(lab_data,mean_thre,axis = 1)
    #Prepare for final training data
    training_data = np.array([])
    feat_data_reglab = np.append(feat_data,mean_thre,axis = 1)
    scaler_Train = MinMaxScaler(feature_range=(-1, 1))
    feat_data_reglab = scaler_Train.fit_transform(feat_data_reglab,(-1,1))

    for i in range(A_data_len[0]):

        #for unhandling data
        if lab_data[i,0] == 0:
            
            training_data = np.append(training_data,feat_data_reglab[i,:])
            training_data = np.append(training_data,0)
            continue

        #for data who didn't utilize X'(t)
        elif lab_data[i,1] == 1:

            buffer = np.array([])

            for j in range(int(3*PERTUBTIME)):

                per_ele = np.random.randint(-1,1, size=feat_data_reglab.shape[1])
                final_input = feat_data_reglab[i,:] + (feat_data_reglab[i,:]*per_ele/100)
                buffer = np.append(buffer,final_input)
                buffer = np.append(buffer,1)
            
            buffer = np.reshape(buffer,(-1,feat_data_reglab.shape[1]+1))
            training_data = np.append(training_data,buffer)

        #for optsaliency map
        else:

            buffer = np.array([])

            for j in range(PERTUBTIME):

                per_ele = np.random.randint(-1,1, size=feat_data_reglab.shape[1])
                final_input = feat_data_reglab[i,:] + (feat_data_reglab[i,:]*per_ele/100)
                buffer = np.append(buffer,final_input)
                buffer = np.append(buffer,0)
            
            buffer = np.reshape(buffer,(-1,feat_data_reglab.shape[1]+1))
            training_data = np.append(training_data,buffer)
    #[:,-1] ==> label of Xt (for classify),[:,-2] ==> label of threshold (for regression)
    training_data = np.reshape(training_data,(-1,feat_data_reglab.shape[1]+1))
    #print(training_data,training_data.shape)
    return training_data
def SelectFeatures(dataset):
    pca = PCA(n_components=10)
    newdata = pca.fit_transform(dataset)
    return newdata

def Split(x,y):
    return  train_test_split(x, y, test_size=0.1, random_state=0, shuffle=True)

if __name__ == '__main__':
    data = DataGenerate()
    y = data[:,-2]
    ## 後三個都是evaluation後三個都是evaluation
    x= data[:-3]
    #print(data[:,-4])
    x = SelectFeatures(data)
    X_train, X_test, y_train, y_test =Split(x,y)
    
    torch_train_dataset = CreatDataset(X_train, y_train)
    torch_test_dataset = CreatDataset(X_test, y_test)
    loader_train = DataLoader(
            dataset=torch_train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True)
    loader_test = DataLoader(
            dataset=torch_test_dataset,
            batch_size=batch_size,
            num_workers=4,
    )

    model = Net()
    model.to(device, dtype=torch.double)
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ## Train
    loss_list=[]
    best_model = model
    best_score = sys.maxsize
    print(best_score)
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss=0.0
        for batch_idx, (data, target) in enumerate(loader_train):
            data, target = data.to(device), target.to(device)
            #data = data.view(-1 , 10)
            y_hat = model(data)
            target = target.to(dtype=float)
            loss = loss_func(y_hat, target)
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if running_loss < best_score:
            best_score = running_loss
            best_model = model

        print("Epoch_" + str(epoch) + "_loss: " + str(running_loss) )
        
        ##loss_list.append(running_loss)


    y_pred = []
    real = []
    mse=0
    mae=0
    for batch_idx, (data, target) in tqdm(enumerate(loader_test)):
            data, target = data.to(device), target.to(device)
            print(target)
            #data = data.view(-1 , 10)
            y_hat=best_model(data)
            mse += nn.MSELoss()(target, y_hat)
            mae += nn.L1Loss()(target, y_hat)
            y_pred.append(y_hat.cpu().detach().numpy().flatten())
            real.append(target.cpu().detach().numpy().flatten())
    print("MSE: "+str(mse.item()))
    print("MAE: "+str(mae.item()))
    
    y_pred = np.array(y_pred)
    y_pred.flatten()
    real = np.array(real)
    real.flatten()
    y=[]
    y_hat=[]
    for i in range(len(y_pred)):
        print(y_pred[i])
        for j in range(len(y_pred[i])):
            y.append(real[i][j])
            y_hat.append(y_pred[i][j])
    
    plt.plot(y, color="blue", label ="target")
    plt.plot(y_hat, color="red", label="predict")
    plt.legend()
    plt.show()
    #print(SelectedData.shape)'''